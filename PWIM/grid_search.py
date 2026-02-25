import torch
import torch.nn as nn
import tqdm
from huggingface_hub import login
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from smoothquant.smoothquant.fake_quant import quantize_model
import os
import json
from ShortGPT.short_gpt.short_hf import ShortHFModel
import argparse
import matplotlib.pyplot as plt
import numpy as np

login(token=os.environ.get("HF_TOKEN"))

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='meta-llama/Meta-Llama-3-8B')
parser.add_argument("--a_bits", type=int, default=8)                # Activation bitwidth for layers "not quantized"
parser.add_argument("--w_bits", type=int, default=8)                # Weight bitwidthfor layers "not quantized"
parser.add_argument("--eval", action="store_true")                  # Evaluation mode
parser.add_argument("--optim_down8b", action="store_true")          # Apply optimisation in 8 bits for down projections
parser.add_argument("--bit", type=int, default=4)                   # bit to apply grid search
parser.add_argument("--fp4", action="store_true")                   # Use fp4 format for down projections
parser.add_argument("--inv", action="store_true")                   # Inverse optimisation and start by the end
parser.add_argument("--sep_f", action="store_true")  
args = parser.parse_args()


n_layers = {'bigscience/bloom-7b1' : 32,
'meta-llama/Llama-2-13b-hf': 40, 
'mistralai/Mistral-7B-v0.1' : 32, 
'meta-llama/Llama-2-7b-hf': 32, 
'meta-llama/Meta-Llama-3-8B' : 32,
'facebook/opt-13b': 40,
'bigscience/bloom-7b1' : 32}

layersD = {'meta-llama/Llama-2-13b-hf': [[3, 38], [39]], 
'mistralai/Mistral-7B-v0.1' : [[1, 30, 31], [31]], # [[1, 30, 31], [31]], 
'meta-llama/Llama-2-7b-hf': [[1, 30], [31]],
'meta-llama/Meta-Llama-3-8B' : [[1, 31], []],
'bigscience/bloom-7b1' : [[1, 31], []]}




model_name = args.model_path
context_size = 2048
cache_dir = '/data1/is156025/lm270675/.cache/huggingface/hub'
mean_cos_sim = 0
act32bits = None
folder = './results/grid_search/' + model_name + '/'
filename = 'grid_search_WA_' + str(args.bit) + 'bits_dichotomie_per_proj'
od8b = '_down8' if args.optim_down8b else '' 
inv = '_inv' if args.inv else ''
bos = '_bos_f' if args.sep_f else '' 
filename = filename + od8b + inv + bos + 'train_test.json'
if not os.path.exists(folder):
    os.mkdir(folder)


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
    def evaluate(self, model, one_shot = False):
        model.eval()
        nlls = []
        self.n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // context_size
        r = range(self.n_samples) if not one_shot else range(1)
        for i in tqdm.tqdm(r, desc="Evaluating..."):
            batch = self.dataset[:, (i * context_size) : ((i + 1) * context_size)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
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

        dataset_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        dataset_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
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

        return self.PPL_train.evaluate(self.model).item(), self.PPL_test.evaluate(self.model).item()

    def search(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, cache_dir=cache_dir).cuda()

        # Calcul des activation en 32 bits
        # print("Hooking 32 bits ...")

        # self.hooking('act')

        # print('Computing 32bits activations')

        # self.PPL_train.evaluate(self.model, one_shot=True)

        # print('Done 32 bits')

        print('Starting test')


        for bit in self.b:
            list_max_bit = [{} for _ in range(self.n_layers - 1)]
            for i in range(self.n_layers):
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



if __name__ == "__main__":

    # Grid Search
    grid_search = GridSearch(model_name, b=[args.bit], optim_down8b=args.optim_down8b)
    
    if not args.eval:
        grid_search.search()
    else:
        with open(folder + 'best_config.json', 'r') as file:
            data = json.load(file)
            for d in data:
                if d['config']['bit'] == args.bit and d['config']['optim_down8b'] == args.optim_down8b and d['config']['inv'] == args.inv:
                    config = d['config']

        # PPL = []
        # for j in range(32):

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir).cuda()

        skip = {'layers': [False]*n_layers[model_name],
                'down_proj': [True]*n_layers[model_name],
                'o_proj': [False]*n_layers[model_name]}
        # skip['layers'][0] = True
        # skip['layers'][1] = True
        # skip['layers'][2] = True
        #layers_qdp_down.pop(j)

        # Quantize model
        model = quantize_model(
            model,
            weight_quant="per_channel",
            act_quant='per_token',
            quantize_bmm_input=True,
            a_bits=args.a_bits, 
            w_bits=args.w_bits,
            spike='grid_search',
            GS_param=config,
            skip_layers=skip,
            fp4=args.fp4,
            sep_f=args.sep_f
        )
        print(grid_search.PPL.evaluate(model).item())
        #     PPL.append(grid_search.PPL.evaluate(model).item())
        # plt.scatter( [i for i in range(32)], PPL)
        # plt.xlabel('layer')
        # plt.ylabel('PPL')
        # plt.grid()
        # plt.savefig(folder + 'PPLvsLayer_pop_down_proj.png')
        




