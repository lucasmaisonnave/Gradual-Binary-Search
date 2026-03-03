import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import json
from main import load_rotate_quantize_model, update_quantizer
import data_utils
import eval_utils
import utils
import gc
import os
from copy import deepcopy
import model_utils
from accelerate import infer_auto_device_map, dispatch_model
# import PrefixQuant.utils.model_utils as p_model_utils
from tqdm import tqdm
import torch.distributed as dist
import time
import torch.multiprocessing as mp

n_layers = {
    'meta-llama/Llama-2-7b-hf': 32,
    'meta-llama/Llama-2-13b-hf': 40,
    'meta-llama/Llama-3.2-1B': 16,
    'meta-llama/Llama-3.1-8B': 32,
    'meta-llama/Meta-Llama-3-8B': 32,
    'mistralai/Mistral-7B-v0.1': 32,
    'mistralai/Mistral-7B-Instruct-v0.3': 32,
    'facebook/opt-13b': 40,
    'bigscience/bloom-7b1': 32,
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': 32,
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B': 28,
    'Qwen/Qwen2.5-0.5B-Instruct': 24,
    'Qwen/Qwen2.5-1.5B-Instruct': 28,
    'Qwen/Qwen2.5-7B-Instruct': 28,
    'microsoft/Phi-4-mini-instruct': 28,
}

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def f_test_ppl(args, model, testenc):
    # Prepare data
    device = model.device
    testenc = testenc['input_ids'].to(device)
    seqlen = args.seqlen
    nsamples = testenc.numel() // seqlen
    model = model.to(device)
    model.eval()
    nlls = []

    with torch.no_grad():
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)]
            labels = testenc[:, (i * seqlen):((i + 1) * seqlen)]
            outputs = model(batch, labels=labels)
            nlls.append(outputs.loss * seqlen)

    nlls_tensor = torch.stack(nlls).sum() if nlls else torch.tensor(0.0, device=device)
    total_samples = torch.tensor(len(nlls), device=device)

    ppl = torch.exp(nlls_tensor / (total_samples * seqlen))
    print(f"Perplexity: {ppl.item()}")
    return ppl

def mgpu_test_ppl(rank, world_size, args, model, testenc, return_dict=None):
    # Distributed setup
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # Prepare data
    testenc = testenc['input_ids'].to(device)
    seqlen = args.seqlen
    nsamples = testenc.numel() // seqlen
    model = model.to(device)
    model.eval()
    nlls = []

    # Each GPU processes its own subset of samples
    with torch.no_grad():
        for i in tqdm(range(rank, nsamples, world_size), desc=f"GPU {rank}"):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)]
            labels = testenc[:, (i * seqlen):((i + 1) * seqlen)]
            outputs = model(batch, labels=labels)
            nlls.append(outputs.loss * seqlen)

    # Aggregate results across GPUs
    nlls_tensor = torch.stack(nlls).sum() if nlls else torch.tensor(0.0, device=device)
    total_samples = torch.tensor(len(nlls), device=device)

    dist.all_reduce(nlls_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    # Final computation on rank 0
    if rank == 0:
        ppl = torch.exp(nlls_tensor / (total_samples * seqlen))
        print(f"Perplexity: {ppl.item()}")
        if return_dict is not None:
            return_dict['ppl'] = ppl

    cleanup()


class GridSearch():
    def __init__(self, args, b=[8, 6, 4], alpha=0.1, optim_down8b=False):
        self.model_name = args.model
        self.b = b
        self.alpha = alpha
        self.n_layers = n_layers[args.model]
        self.folder = './grid_search/' + self.model_name + '/'
        self.filename = 'rotate' if args.rotate else '' 
        self.filename += '_W_RTN' if args.w_rtn else '_W_GPTQ' 
        self.filename += '_inv' if args.inv else '' 
        self.filename += '_W' + str(args.w_bits) + 'A' + str(args.a_bits) + 'KV' + str(args.v_bits) 
        self.filename += '_start_bit_'  + str(args.start_bit)
        self.filename_noexp = self.filename + ".json"
        self.filename += '_expand_' + str(int(args.expand.split('/')[-1].split('.')[0].split('_')[0])) if '/0.json' not in args.expand else ''
        self.filename += '_alpha_'+ str(alpha) if alpha != 0.1 else ''
        self.filename += '_max_iter_'+ str(args.max_iterations) if args.max_iterations != 10 else ''
        self.filename += ".json"
        self.args = args
        self.world_size = torch.cuda.device_count()

        print('Loading Test')
        self.testloader = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=args.seqlen,
            hf_token=args.hf_token,
            cache_dir=args.cache_dir,
            eval_mode=True
        )
        print('Loading Train')
        self.trainloader = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=args.seqlen,
            hf_token=args.hf_token,
            eval_mode=False,
            cache_dir=args.cache_dir,
            grid = True
        )
        size = int(alpha * self.trainloader.input_ids.shape[-1])
        self.trainloader.input_ids = self.trainloader.input_ids[:,:size]

        model = model_utils.get_model(args.model, args.hf_token, args.cache_dir)
        self.model, self.prefixed_key_values = load_rotate_quantize_model(self.args, model)


        self.epsilon = 1e-3
        self.max_iterations = args.max_iterations
        self.optim_down8b = optim_down8b  # If True, always search down_proj at 8 bits

    
    def append_to_json(self, file_name, data):
        try:
            # Read existing data from the file
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            with open(file_name, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            # If the file doesn't exist, start with an empty list
            existing_data = []
            
        except json.JSONDecodeError:
            # If the file is empty or not valid JSON, start with an empty list
            existing_data = []

        existing_data.append(data)

        with open(file_name, 'w') as file:
            json.dump(existing_data, file, indent=4)
    
    def f(self, config, eval=False):
        update_quantizer(self.args, self.model, config)
        self.model.eval()
        torch.cuda.empty_cache()
        test_ppl = None
        train_ppl = None
        if eval:
            test_ppl = eval_utils.evaluator(self.model, self.testloader, utils.DEV, self.args)
        else:
            torch.cuda.empty_cache()
            train_ppl = eval_utils.evaluator(self.model, self.trainloader, utils.DEV, self.args)
        return train_ppl, test_ppl

    def resume(self):
        with open(self.folder + self.filename, 'r') as file:
            data = json.load(file)
            return data[-1]['config']

    def search(self):

        print('Starting Search')
        if self.args.resume_gs:
            with open(self.folder + self.filename, 'r') as file:
                existing_data = json.load(file)
                # remove last proj optimisation from file
                resume_p = len(existing_data[-1]['config']['proj']) - 1
                resume_l = existing_data[-1]['config']['layer']
                while len(existing_data[-1]['config']['proj']) - 1 == resume_p:
                    existing_data.pop()

                last_data = existing_data[-1]

                resume_config = last_data['config']['max_bit']
                resume_ppl = last_data['ppl']['train']
                with open(self.folder + self.filename, 'w') as file:
                    json.dump(existing_data, file, indent=4)
                print("Resuming grid search from: layer {} proj {}".format(resume_l, resume_p))
        else:
            resume_l = 0
            resume_p = 0
            
        for bit in self.b:
            proj = ['q_proj', 'k_proj', 'v_proj', 'qk_rotation', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] 
            tmp = {}
            base_tmp = {'max': 1, 'bit': self.args.start_bit}
            for p in proj:
                tmp[p] = deepcopy(base_tmp)
            if self.args.resume_gs:
                list_max_bit = resume_config
                best_max_bit = deepcopy(resume_config)
                f_best_train = resume_ppl
            else:
                list_max_bit = [deepcopy(tmp) for _ in range(self.n_layers)]
                best_max_bit = [deepcopy(tmp) for _ in range(self.n_layers)]
                f_best_train = 1e5
            f_best_test = 1e5
            for i in range(resume_l,self.n_layers):
                if self.args.inv:
                    i = self.n_layers - 1 - i
                print("Layer " + str(i))
                for l in range(resume_p, len(proj)):
                    if self.args.inv:
                        l = len(proj) - 1 - l
                    print("Proj " + proj[l])
                    a = 0
                    b = 1
                    iterations = 0
                    m = (a + b) / 2

                    # Optimize down_proj in 8 bits if requested
                    if self.optim_down8b and proj[l] == 'down_proj':
                        _bit = 8
                    else:
                        _bit = bit

                    list_max_bit[i][proj[l]]['max'] = m
                    list_max_bit[i][proj[l]]['bit'] = _bit

                    config = {'max_bit': list_max_bit, 'layer': i,  'bit': bit, "optim_down8b": self.args.optim_down8b, 'inv': self.args.inv, 'proj': proj[:l + 1]}
                    print("m = {:.2f}".format(m))
                    fm, ftest = self.f(config)
                    results = {'config': config, 'ppl': {'train': fm, 'test': ftest}}
                    self.append_to_json(self.folder + self.filename, results)

                    # Ternary search: at each iteration test both the left midpoint
                    # (a+m)/2 and the right midpoint (m+b)/2, then keep the half
                    # whose candidate has lower PPL. This halves the interval each
                    # iteration and always converges toward the optimum.
                    while b - a > self.epsilon and iterations < self.max_iterations:
                        x_left = (a + m) / 2
                        x_right = (m + b) / 2

                        # Evaluate left candidate
                        list_max_bit[i][proj[l]]['max'] = x_left
                        list_max_bit[i][proj[l]]['bit'] = _bit
                        config = {'max_bit': list_max_bit, 'layer': i, 'bit': bit, "optim_down8b": self.args.optim_down8b, 'inv': self.args.inv, 'proj': proj[:l + 1]}
                        print("x_left = {:.2f}".format(x_left))
                        fx_left, fxtest_left = self.f(config)
                        self.append_to_json(self.folder + self.filename, {'config': config, 'ppl': {'train': fx_left, 'test': fxtest_left}})

                        # Evaluate right candidate
                        list_max_bit[i][proj[l]]['max'] = x_right
                        list_max_bit[i][proj[l]]['bit'] = _bit
                        config = {'max_bit': list_max_bit, 'layer': i, 'bit': bit, "optim_down8b": self.args.optim_down8b, 'inv': self.args.inv, 'proj': proj[:l + 1]}
                        print("x_right = {:.2f}".format(x_right))
                        fx_right, fxtest_right = self.f(config)
                        self.append_to_json(self.folder + self.filename, {'config': config, 'ppl': {'train': fx_right, 'test': fxtest_right}})

                        # Keep the half with the lower PPL
                        if fx_left <= fx_right:
                            b = m
                            m, fm, ftest = x_left, fx_left, fxtest_left
                        else:
                            a = m
                            m, fm, ftest = x_right, fx_right, fxtest_right

                        iterations += 1

                    # Keep the found clipping ratio if it improved PPL.
                    # When start_bit > _bit (e.g. warm-starting from 16-bit), always
                    # accept the search result: the baseline was at a higher bit-width
                    # so comparing PPL values directly would be unfair.
                    if fm < f_best_train or self.args.start_bit > _bit:
                        best_max_bit[i][proj[l]]['max'] = m
                        best_max_bit[i][proj[l]]['bit'] = _bit
                        f_best_train = fm
                        f_best_test = ftest
                    list_max_bit = deepcopy(best_max_bit)
                resume_p = 0
            config['max_bit'] = deepcopy(best_max_bit)
            if f_best_test is None:
                _, f_best_test = self.f(config, eval = True)
            best_config_bit = {'config': config, 'ppl': {'train': f_best_train, 'test': f_best_test}}  
            self.append_to_json(self.folder + 'best_config_' + self.filename, best_config_bit)



if __name__ == '__main__':
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
    args.exp_name = int(args.expand.split('/')[-1].split('.')[0].split('_')[0])
    GD = GridSearch(args, b=[args.w_bits], alpha=args.alpha)
    
    if not args.eval:
        start_time = time.time()
        GD.search()
        end_time = time.time()
        duration_seconds = end_time - start_time
        duration_hours = duration_seconds / 3600

        print(f"Elapsed time: {duration_hours:.6f} hours")
    else:
        print("Evaluating")
        if not args.lm_eval:
            folder = './grid_search/' + args.model + '/'
            config = None
            if args.grid_search:
                with open(folder + 'best_config_' + GD.filename_noexp, 'r') as file:
                    config = json.load(file)[0]['config']
            
            ppl_train, ppl_test = GD.f(config, eval=True)
            csv_file = "./results/results_ppl.csv"
            row = [args.model, args.w_bits, args.grid_search, args.exp_name, ppl_test]
            import csv
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        else:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
        
            GD.model.to(utils.DEV)
            
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
            hflm = HFLM(pretrained=GD.model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

            # task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
            results = lm_eval.simple_evaluate(hflm, tasks=args.tasks, batch_size=args.lm_eval_batch_size)['results']

            metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
            metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
            print(metric_vals)
            L = [args.model,args.w_bits,args.grid_search,int(args.expand.split('/')[-1].split('.')[0])]
            for v in metric_vals:
                L.append(metric_vals[v])
            import csv
            csv_file = "./results/results_tasks.csv"
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(L)