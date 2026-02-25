import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.smoothquant.smooth import smooth_lm
from smoothquant.smoothquant.fake_quant import quantize_model
import tqdm
from huggingface_hub import login
from datasets import load_dataset
import argparse
from load import load_short_model
from smoothquant.smoothquant.calibration import get_act_scales
import os
from lm_eval import evaluator, tasks
import json
from lm_eval.api.model import LM
import random

login(token=os.environ.get("HF_TOKEN"))
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--model_path", type=str, default='meta-llama/Meta-Llama-3-8B')
parser.add_argument("--n_samples", type=int, default=None)          # Number of sample to compute ppl
parser.add_argument("--N", type=int, default=6)                     # Number of layers to delete
parser.add_argument("--a_bits", type=int, default=8)                # Activation bitwidth
parser.add_argument("--w_bits", type=int, default=8)                # Weight bitwidth
parser.add_argument("--smooth", action="store_true")                # Use smoothing from SmoothQuant
parser.add_argument("--quantize", action="store_true")              # Quantize model
parser.add_argument("--sep_f", action="store_true")                 # Do not quantize SEP token
parser.add_argument("--log", action="store_true")                   # Apply log2 activation quantization
parser.add_argument("--act_quant", type=str, default='per_token')   # Type of activation quantization : per_token or per_tensor
parser.add_argument("--mix", action="store_true")                   # Apply mix precision
parser.add_argument("--spike", type=str, default=None)              # Use FP8 for layer not quantized
parser.add_argument("--rand", action="store_true")                  # Test random layers for mix precision
parser.add_argument("--seed", type=int, default=0)                  # Set random seed

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

def seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


args = parser.parse_args()
alpha = args.alpha
model_path = args.model_path
act_scales_path = act_scales_path = "/data1/is156025/lm270675/meta-labo/LLM/smoothquant/act_scales/" + model_path + "/act_scales.sm"
n_samples = args.n_samples
cache_dir = '/data1/is156025/lm270675/.cache/huggingface/hub'
context_size = 2048
# layers not to quantize
seed(args.seed)
layers_qdp_down = []
layers_qdp_out = []
if args.mix:
    if not args.rand:
        layers_qdp_down = layersD[model_path][0]
        layers_qdp_out = layersD[model_path][1]
    else:
        L = []
        N = n_layers[model_path]
        while len(L) < 2:
            r = torch.randint(0,N - 1, (1,)).item()
            while r in layersD[model_path]:
                r = torch.randint(0,N - 1, (1,)).item()
            L.append(r)
        layers_qdp_down = L

        L = []
        r = torch.randint(0,N - 1, (1,)).item()
        while r in layersD[model_path]:
            r = torch.randint(0,N - 1, (1,)).item()
        L.append(r)
        layers_qdp_out = L
        

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // context_size
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
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


# tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer, model = load_short_model(args.model_path, args.N, 'wiki')

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, cache_dir=cache_dir).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
PPL = Evaluator(dataset, tokenizer, "cuda", n_samples=n_samples)

if args.smooth:
    if not os.path.exists(act_scales_path):
        act_scales = get_act_scales(
        model, tokenizer, "/data1/is156025/lm270675/meta-labo/EAH-ViT/LLM/smoothquant/dataset/val.jsonl.zst", 512, 512
        )
        os.makedirs(os.path.dirname(act_scales_path), exist_ok=True)
        torch.save(act_scales, act_scales_path)
    else:
        act_scales = torch.load(act_scales_path)
    smooth_lm(model, act_scales, alpha)
if args.quantize:
    model = quantize_model(
        model,
        weight_quant="per_channel",
        act_quant=args.act_quant,
        quantize_bmm_input=True,
        sep_f=args.sep_f,
        log=args.log,
        layers_qdp_out=layers_qdp_out, 
        layers_qdp_down=layers_qdp_down, 
        a_bits=args.a_bits, 
        w_bits=args.w_bits,
        spike=args.spike
    )

ppl = PPL.evaluate(model)
print(f"Perplexity: {ppl}")
import csv

csv_file = './results/ppl_suite.csv'
new_row = [model_path, args.seed, args.act_quant, args.smooth, args.mix, args.spike, args.w_bits, args.a_bits, ppl.item()]

with open(csv_file, 'a', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    
    # Write the new row to the CSV file
    writer.writerow(new_row)


