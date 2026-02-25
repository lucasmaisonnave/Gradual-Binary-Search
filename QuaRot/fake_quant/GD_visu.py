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
import torch.multiprocessing as mp
import json

cache_dir = '/data1/is156025/lm270675/.cache/huggingface/hub'
n_layers = {'meta-llama/Llama-2-7b-hf': 32,
'meta-llama/Llama-3.2-1B' : 16,
'meta-llama/Llama-3.1-8B' : 32,
'meta-llama/Llama-2-13b-hf': 40, 
'mistralai/Mistral-7B-v0.1' : 32,
'mistralai/Mistral-7B-Instruct-v0.3': 32,
'meta-llama/Llama-2-7b-hf': 32, 
'meta-llama/Meta-Llama-3-8B' : 32,
'facebook/opt-13b': 40,
'bigscience/bloom-7b1' : 32,
'deepseek-ai/DeepSeek-R1-Distill-Llama-8B':32,
'Qwen/Qwen2.5-7B-Instruct' : 28,
'Qwen/Qwen2.5-1.5B-Instruct' : 28,
'Qwen/Qwen2.5-0.5B-Instruct' : 24,
'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' : 28,
'microsoft/Phi-4-mini-instruct': 28
}

quant_error = 0

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def f_test_ppl(args, model, testenc):
    
    # Préparation des données
    device = model.device
    testenc = testenc['input_ids'].to(device)
    seqlen = args.seqlen
    nsamples = testenc.numel() // seqlen
    model = model.to(device)
    model.eval()
    nlls = []

    # Chaque GPU traite son subset de données
    with torch.no_grad():
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)]
            labels = testenc[:, (i * seqlen):((i + 1) * seqlen)]
            outputs = model(batch, labels=labels)
            nlls.append(outputs.loss * seqlen)

    # Agrégation des résultats
    nlls_tensor = torch.stack(nlls).sum() if nlls else torch.tensor(0.0, device=device)
    total_samples = torch.tensor(len(nlls), device=device)

    ppl = torch.exp(nlls_tensor / (total_samples * seqlen))
    print(f"Perplexité : {ppl.item()}")
    return ppl

def mgpu_test_ppl(rank, world_size, args, model, testenc, return_dict=None):
    # Initialisation distribuée
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # Préparation des données
    testenc = testenc['input_ids'].to(device)
    seqlen = args.seqlen
    nsamples = testenc.numel() // seqlen
    model = model.to(device)
    model.eval()
    nlls = []

    # Chaque GPU traite son subset de données
    with torch.no_grad():
        for i in tqdm(range(rank, nsamples, world_size), desc=f"GPU {rank}"):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)]
            labels = testenc[:, (i * seqlen):((i + 1) * seqlen)]
            outputs = model(batch, labels=labels)
            nlls.append(outputs.loss * seqlen)

    # Agrégation des résultats
    nlls_tensor = torch.stack(nlls).sum() if nlls else torch.tensor(0.0, device=device)
    total_samples = torch.tensor(len(nlls), device=device)

    # Synchronisation entre GPUs
    dist.all_reduce(nlls_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    # Calcul final sur le GPU 0
    if rank == 0:
        ppl = torch.exp(nlls_tensor / (total_samples * seqlen))
        print(f"Perplexité : {ppl.item()}")
        if return_dict is not None:
            return_dict['ppl'] = ppl

    cleanup()

def get_hook(name, N):
    
    
    def hook_quant_error(model, input, output):
        global quant_error
        name_ = name
        quant_error += ((input[0] - output).square().mean().sqrt() / N).item()
    
    return hook_quant_error


class GridSearch():
    def __init__(self, args, b = [8, 6, 4], alpha = 0.1, optim_down8b = False):

        # On va quantifier le modèle à chaque nouvelle configuration
        # La fonction quantize_model doit donc prendre en compte bit, max ainsi que l'indice de la couche à modifier
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
        self.filename += '_visu.json'
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

        seqlen = args.seqlen
        nsamples = self.trainloader.input_ids.numel() // seqlen

        for name, layer in self.model.named_modules():
            qu = name.split('.')[-1]
            if 'proj' in qu and '.1.' in name:
                # quant_error[qu] = 0
                layer.quantizer.register_forward_hook(get_hook(qu, nsamples))
            if 'qk_rotation' in qu and '.1.' in name:
                layer.k_quantizer.register_forward_hook(get_hook(qu, nsamples))

        self.epsilon = 1e-3
        self.max_iterations = 10
        self.optim_down8b = optim_down8b    # If True we apply grid search in 8 bits for all down_proj

    
    def append_to_json(self, file_name, data):

            # Read existing data from the file
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
           
        # Write the updated data back to the file
        with open(file_name, 'w') as file:
            json.dump(data, file, indent=4)
    
    def f(self, config, eval = False):
        update_quantizer(self.args, self.model, config)
        # if self.prefixed_key_values is not None:
        #     self.prefixed_key_values = p_model_utils.mv_kv_cache(self.prefixed_key_values, self.model)
        self.model.eval()
        torch.cuda.empty_cache()
        test_ppl = None
        train_ppl = None
        from torch.multiprocessing import Manager
        manager = Manager()
        return_dict = manager.dict()
        if eval:
            
            test_ppl = eval_utils.evaluator(self.model, self.testloader, utils.DEV, args) #, prefixed_key_values)
            # test_ppl = f_test_ppl(args, self.model, self.testloader)
            # mp.spawn(mgpu_test_ppl, args=(self.world_size, self.args, self.model, self.testloader, return_dict), nprocs=self.world_size, join=True)
            # test_ppl = return_dict.get('ppl', None)

        else:
            torch.cuda.empty_cache()
            train_ppl = eval_utils.evaluator(self.model, self.trainloader, utils.DEV, args) #, prefixed_key_values)
            # mp.spawn(mgpu_test_ppl, args=(self.world_size, self.args, self.model, self.trainloader, res), nprocs=self.world_size, join=True)

        return train_ppl, test_ppl

    def resume(self):
        with open(self.folder + self.filename, 'r') as file:
            data = json.load(file)
            return data[-1]['config']

    def search(self):

        print('Starting Search')
        
        global quant_error
        results = {}
        bit = args.w_bits
        d = 10
        proj = ['q_proj', 'k_proj', 'v_proj', 'qk_rotation', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] 
        tmp = {}
        base_tmp = {'max': 1, 'bit': args.start_bit}
        base_res = {'ppl':[0]*(d+1), 'quant_error':[0]*(d+1)}
        
        for p in proj:
            tmp[p] = deepcopy(base_tmp)
            results[p] = deepcopy(base_res)
        CR = [0.1 + j * 0.9 / d for j in range(d + 1)]
        i = 1
        list_max_bit = [deepcopy(tmp) for _ in range(self.n_layers)]
        if os.path.exists(self.folder + 'fig1.json'):
            with open(self.folder + 'fig1.json', 'r') as file:
                results = json.load(file)
        print("Layer " + str(i))
        resume_CR = 8
        resume_l = 7
        for l in range(resume_l, len(proj)):
            print("Proj " + proj[l])
            for j in range(resume_CR,len(CR)):
                quant_error = 0
                cr = CR[j]
                list_max_bit[i][proj[l]]['max'] = cr
                list_max_bit[i][proj[l]]['bit'] = bit

                config = {'max_bit': list_max_bit, 'layer': i,  'bit': bit, "optim_down8b": self.args.optim_down8b, 'inv': self.args.inv, 'proj': proj[:l + 1]}
                print("cr = {:.2f}".format(cr))
                fm, ftest = self.f(config)
                results[proj[l]]['ppl'][j] = fm
                results[proj[l]]['quant_error'][j] = quant_error
                self.append_to_json(self.folder + 'fig1.json', results)
            list_max_bit[i][proj[l]]['max'] = 1
            list_max_bit[i][proj[l]]['bit'] = 16
            resume_CR=0






if __name__ == '__main__':
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
    args.exp_name = int(args.expand.split('/')[-1].split('.')[0].split('_')[0])
    GD = GridSearch(args, b=[args.w_bits])
    
    GD.search()