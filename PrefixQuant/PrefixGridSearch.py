import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from huggingface_hub import login
import matplotlib.animation as animation
import utils.model_utils as model_utils
from smoothquant.smoothquant.fake_quant import quantize_model
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import json
import random
import argparse
from utils import train_utils
import time

cache_dir = '/data1/is156025/lm270675/.cache/huggingface/hub'
login(token=os.environ.get("HF_TOKEN"))
dataset = 'wiki'
n_layers = {'meta-llama/Llama-2-7b-hf': 32,
'meta-llama/Llama-3.2-1B' : 16,
'meta-llama/Llama-3.1-8B' : 32,
'meta-llama/Llama-2-13b-hf': 40, 
'mistralai/Mistral-7B-v0.1' : 32, 
'meta-llama/Llama-2-7b-hf': 32, 
'meta-llama/Meta-Llama-3-8B' : 32,
'facebook/opt-13b': 40,
'bigscience/bloom-7b1' : 32}
context_size = 1024


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='meta-llama/Llama-2-7b-hf')
parser.add_argument("--a_bits", type=int, default=8)                # Activation bitwidth for layers "not quantized"
parser.add_argument("--w_bits", type=int, default=8)                # Weight bitwidthfor layers "not quantized"
parser.add_argument("--eval", action="store_true")                  # Evaluation mode
parser.add_argument("--optim_down8b", action="store_true")          # Apply optimisation in 8 bits for down projections
parser.add_argument("--bit", type=int, default=4)                   # bit to apply grid search
parser.add_argument("--fp4", action="store_true")                   # Use fp4 format for down projections
parser.add_argument("--inv", action="store_true")                   # Inverse optimisation and start by the end
parser.add_argument("--sep_f", action="store_true")  
parser.add_argument("--wd", type=float, default=4e-6)   
parser.add_argument("--lr", type=float, default=1e-3)  
parser.add_argument("--lmbda", type=float, default=0.1)   
parser.add_argument("--epochs", type=int, default=10)   
args = parser.parse_args()

model_name = args.model_path
folder = './grid_search/' + model_name + '/'
filename = 'grid_search_prefix_' + str(args.bit) 
od8b = '_down8' if args.optim_down8b else ''
inv = '_inv' if args.inv else ''
bos = '_bos_f' if args.sep_f else '' 
filename = filename + od8b + inv + bos 

if not os.path.exists(folder):
    os.makedirs(folder)


def seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed(0)


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=None, alpha = 1):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)
        self.dataset = self.dataset[:,:int(alpha * self.dataset.shape[1])]

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model, one_shot = False, past_key_values=None):
        model.eval()
        nlls = []
        self.n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // context_size
        r = range(self.n_samples) if not one_shot else range(1)
        for i in tqdm.tqdm(r, desc="Evaluating..."):
            batch = self.dataset[:, (i * context_size) : ((i + 1) * context_size)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch, prefixed_key_values=past_key_values).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * context_size) : ((i + 1) * context_size)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float()
            if not loss.isnan():
                nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / len(nlls))



class GridSearch():
    def __init__(self, model_name, b = [8, 6, 4], N = 10, alpha = 0.15, optim_down8b = False):

        # On va quantifier le modèle à chaque nouvelle configuration
        # La fonction quantize_model doit donc prendre en compte bit, max ainsi que l'indice de la couche à modifier
        self.model_name = model_name
        self.b = b
        self.N = N
        self.alpha = alpha
        self.n_layers = n_layers[model_name]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        # Pour LLaMA2-7B

        self.prefix_tokens = self.tokenizer.encode(".")[1:] + self.tokenizer.encode("\n")[1:] + self.tokenizer.encode(self.tokenizer.bos_token)[1:] 

        dataset_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir)
        dataset_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir)
        self.PPL_train = Evaluator(dataset_train, self.tokenizer, "cuda", alpha = self.alpha)
        self.PPL_test = Evaluator(dataset_test, self.tokenizer, "cuda")
        self.cosine_sim = torch.nn.CosineSimilarity()

        self.epsilon = 1e-3
        self.max_iterations = 10
        self.optim_down8b = optim_down8b    # If True we apply grid search in 8 bits for all down_proj

    def get_hook(self, name, layer):
        def hook_act(model, input, output):
            global act32bits
            
            feature = output[0]

            if act32bits == None:
                act32bits = torch.zeros((self.n_layers,) + tuple(feature.shape)).cuda()
            
            act32bits[layer] = feature

        def hook_cos_sim(model, input, output):
            global mean_cos_sim

            feature = output[0]           
            mean_cos_sim += self.cosine_sim(feature, act32bits[layer]).mean()


        if name == 'act':
            return hook_act
        elif name == 'cos_sim':
            return hook_cos_sim
        
    def hooking(self, hook_name, p = None):
        if hook_name == 'act':
            for name, layer in self.model.named_modules():
                if isinstance(layer, LlamaDecoderLayer):
                    id = int(name.split('.')[-1])
                    layer.register_forward_hook(self.get_hook('act', id))
        elif hook_name == 'cos_sim':
            for name, layer in self.model.named_modules():
                if isinstance(layer, LlamaDecoderLayer):
                    id = int(name.split('.')[-1])
                    if id == p:
                        layer.register_forward_hook(self.get_hook('cos_sim', p))

    def append_to_json(self, file_name, data):
        try:
            # Read existing data from the file
            with open(file_name, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            # If the file doesn't exist, start with an empty list
            existing_data = []
        except json.JSONDecodeError:
            # If the file is empty or not valid JSON, start with an empty list
            existing_data = []

        # Append the new data
        existing_data.append(data)

        # Write the updated data back to the file
        with open(file_name, 'w') as file:
            json.dump(existing_data, file, indent=4)

    def f(self, config):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, cache_dir=cache_dir).cuda()

        output = self.model(torch.tensor([self.prefix_tokens],device=self.model.device),return_dict=True)
        prefixed_key_values = output.past_key_values
        self.model.config.use_cache = True


        skip = {'layers': [False]*n_layers[self.model_name],
                'down_proj': [False]*n_layers[self.model_name],
                'o_proj': [False]*n_layers[self.model_name]}
        # Quantize model
        self.model = quantize_model(
            self.model,
            weight_quant="per_channel",
            act_quant='per_token',
            quantize_bmm_input=True,
            a_bits=8, 
            w_bits=8,
            spike='grid_search',
            GS_param=config,
            sep_f=args.sep_f,
            skip_layers=skip
        )

        prefixed_key_values = model_utils.mv_kv_cache(prefixed_key_values, self.model)

        return self.PPL_train.evaluate(self.model, past_key_values=prefixed_key_values).item(), self.PPL_test.evaluate(self.model, past_key_values=prefixed_key_values).item()

    def resume(self):
        with open(folder + filename, 'r') as file:
            data = json.load(file)
            return data[-1]['config']

    def search(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, cache_dir=cache_dir).cuda()

        # Calcul des activation en 32 bits
        # print("Hooking 32 bits ...")

        # self.hooking('act')

        # print('Computing 32bits activations')

        # self.PPL_train.evaluate(self.model, one_shot=True)

        # print('Done 32 bits')

        print('Starting test')

        config_resume = self.resume()
        list_max_bit = config_resume['max_bit']
        list_max_bit.append({})
        layer_resume = config_resume['layer']

        for bit in self.b:
            #list_max_bit = [{} for _ in range(self.n_layers)]
            for i in range(layer_resume+1, self.n_layers):
                if args.inv:
                    i = self.n_layers - 1 - i
                print("Layer " + str(i))
                proj = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] if not args.inv else ['down_proj', 'up_proj', 'gate_proj', 'o_proj', 'v_proj', 'k_proj', 'q_proj']
                for l in range(len(proj)):
                    print("Proj " + proj[l])
                    a = 0
                    b = 1
                    iterations = 0
                    m = (a + b) / 2
                    
                    # Here we optimize down proj differently in 8 bits all the time
                    if self.optim_down8b and proj[l] == 'down_proj':
                        _bit = 8
                    else:
                        _bit = bit


                    list_max_bit[i][proj[l]] = {"max": m, "bit": _bit}

                    config = {'max_bit': list_max_bit, 'layer': i,  'bit': bit, "optim_down8b": args.optim_down8b, 'inv': args.inv, 'bos_f': args.sep_f, 'proj': proj[:l + 1]}
                    print("m = {:.2f}".format(m))
                    fm, ftest = self.f(config)
                    results = {'config': config, 'ppl': {'train': fm, 'test': ftest}}
                    self.append_to_json(folder + filename, results)
                    

                    while b - a > self.epsilon and iterations < self.max_iterations:
                        if iterations % 2 == 0:
                            x = (a + m) / 2
                        else:
                            x = (m + b) / 2
                
                        list_max_bit[i][proj[l]] = x
                        list_max_bit[i][proj[l]] = {"max": x, "bit": _bit}
                        config = {'max_bit': list_max_bit, 'layer': i, 'bit': bit, "optim_down8b": args.optim_down8b, 'inv': args.inv, 'bos_f': args.sep_f, 'proj': proj[:l + 1]}
                        print("m = {:.2f}".format(x))
                        fx, fxtest = self.f(config)

                        results = {'config': config, 'ppl': {'train': fx, 'test': fxtest}}

                        self.append_to_json(folder + filename, results)
                        
                        if fx < fm:
                            if x < m:
                                b = m
                            else:
                                a = m
                            m, fm, ftest = x, fx, fxtest
                        else:
                            if x < m:
                                a = x
                            else:
                                b = x
                        
                        iterations += 1

            best_config_bit = {'config': config, 'ppl': {'train': fm, 'test': ftest}}  
            self.append_to_json(folder + 'best_config.json', best_config_bit)

def eval_one_batch(model, param_prefixed_key_values, data, i, lmbda):

    batch = data[:, (i * context_size) : ((i + 1) * context_size)].to(model.device)
    lm_logits = model(batch, prefixed_key_values=param_prefixed_key_values).logits
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = data[:, (i * context_size) : ((i + 1) * context_size)][:, 1:]
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    loss_entropy = torch.tensor(0).cuda()
    for ent in entropy:
        loss_entropy += ent / len(entropy)
    loss += lmbda * loss_entropy
    return loss


def finetune_prefix(model, tokenizer, prefixed_key_values, log, lmbda):
    # prefixed_key_values : tuple de taille (32,2) avec un tenseur de taille (1,32,4,128)
    l, n = len(prefixed_key_values), len(prefixed_key_values[0])
    param_prefixed_key_values = ()
    list_param = []
    for i in range(l):
        tmp = ()
        for j in range(n):
            param_kv = nn.Parameter(prefixed_key_values[i][j].detach())
            tmp += (param_kv,)
            list_param.append(param_kv)
        param_prefixed_key_values += tmp

    optimizer = torch.optim.AdamW(list_param, weight_decay=args.wd, lr = args.lr)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    log.info('Loading dataset')
    dataset_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir)
    dataset_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir)

    dataset_train = tokenizer(
            "\n\n".join(dataset_train["text"]), return_tensors="pt"
        ).input_ids.to(model.device)
    
    dataset_test = tokenizer(
            "\n\n".join(dataset_test["text"]), return_tensors="pt"
        ).input_ids.to(model.device)

    log.info('Start fine-tuning')
    for epoch in range(args.epochs):
        start_time = time.time()
        n_samples_train = dataset_train.size(1) // context_size
        n_samples_test = dataset_test.size(1) // context_size
        for i in tqdm.tqdm(range(n_samples_train), desc="Training..."):
            optimizer.zero_grad()
            train_loss = eval_one_batch(model, param_prefixed_key_values, dataset_train, i, lmbda)
            with torch.autograd.set_detect_anomaly(True):
                train_loss.backward()
            optimizer.step()
            lr_schedule.step()

            
        with torch.no_grad():
            val_loss = torch.tensor(0)
            for i in tqdm.tqdm(range(n_samples_test), desc="Evaluating..."):
                val_loss += eval_one_batch(model, param_prefixed_key_values, train_loss, i, lmbda)
        optimizer.zero_grad()
        log.info(f"epoch {epoch} train_loss:{train_loss.item()} val_loss:{val_loss.mean().item()} time {time.time()-start_time}")
        
    return


def get_hook(layer):
        def hook_attn(model, input, output):
            global entropy
            attn = output[1]
            attn[attn == 0] += 1
            entropy.append((-(attn * torch.log2(attn)).sum(-1)).mean())
            return
        
        return hook_attn

if __name__ == "__main__":
    entropy = []
    # Grid Search
    grid_search = GridSearch(model_name, b=[args.bit], optim_down8b=args.optim_down8b)

    logger = train_utils.create_logger("./log")
    logger.info(args)
    
    if not args.eval:
        grid_search.search()
    else:
        model = AutoModelForCausalLM.from_pretrained(grid_search.model_name, torch_dtype=torch.float16, cache_dir=cache_dir, output_attentions=True).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        
        output = model(torch.tensor([grid_search.prefix_tokens],device=model.device),return_dict=True)
        prefixed_key_values = output.past_key_values
        model.config.use_cache = True

        # Add hooks to get attention maps

        for name, layer in model.named_modules():
            if isinstance(layer, LlamaDecoderLayer):
                l = int(name.split('.')[-1])
                layer.register_forward_hook(get_hook(l))

        finetune_prefix(model, tokenizer, prefixed_key_values, logger, args.lmbda)

        # Créer une copie des prefix KV en parameter

        skip = {'layers': [False]*n_layers[grid_search.model_name],
                'down_proj': [False]*n_layers[grid_search.model_name],
                'o_proj': [False]*n_layers[grid_search.model_name]}
        # Quantize model
        # model = quantize_model(
        #     model,
        #     weight_quant="per_channel",
        #     act_quant='per_token',
        #     quantize_bmm_input=True,
        #     a_bits=4, 
        #     w_bits=4,
        #     skip_layers=skip
        # )
        prefixed_key_values = model_utils.mv_kv_cache(prefixed_key_values, model)
        print(grid_search.PPL_test.evaluate(model, past_key_values=prefixed_key_values).item())