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
from lm_eval.models.huggingface import HFLM


login(token=os.environ.get("HF_TOKEN"))
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--model_path", type=str, default='meta-llama/Meta-Llama-3-8B')
parser.add_argument("--result_path", type=str, default='./results/evaluation_results.json')
parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--N", type=int, default=6)
parser.add_argument("--a_bits", type=int, default=8)
parser.add_argument("--w_bits", type=int, default=8)
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--sep_f", action="store_true")
parser.add_argument("--log", action="store_true")
parser.add_argument("--act_quant", type=str, default='per_token')
parser.add_argument("--mix", action="store_true")

os.environ["CUDA_VISIBLE_DEVICES"]="1"
layersD = {'meta-llama/Llama-2-13b-hf': [3, 38, -1], 
'mistralai/Mistral-7B-v0.1' : [1, 31, -1], 
'meta-llama/Llama-2-7b-hf': [1, 31], 
'meta-llama/Meta-Llama-3-8B' : [1, 31],
'bigscience/bloom-7b1' : [1, 31]}

args = parser.parse_args()
alpha = args.alpha
model_path = args.model_path
act_scales_path = act_scales_path = "/data1/is156025/lm270675/meta-labo/LLM/smoothquant/act_scales/" + model_path + "/act_scales.sm"
n_samples = args.n_samples
cache_dir = '/data1/is156025/lm270675/.cache/huggingface/hub'
context_size = 2048
# layers not to quantize
layers_qdp = []
if args.mix:    
    layers_qdp = layersD[model_path]


# tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer, model = load_short_model(args.model_path, args.N, 'wiki')

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, cache_dir=cache_dir).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

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
        layers_qdp=layers_qdp, 
        a_bits=args.a_bits, 
        w_bits=args.w_bits
    )


model.eval()
eval_model = HFLM(pretrained=model, tokenizer=tokenizer, device="cuda")


# Pass these datasets to the evaluator
results = evaluator.simple_evaluate(
    model=eval_model,
    tasks=["hellaswag", "piqa", "winogrande", "openbookqa", "rte", "copa", "lambada_standard"],
    num_fewshot=0,
    batch_size=256,
    device="cuda"
)
    
print(results['results'])
    # Save results to a JSON file
file_path = args.result_path
config = {
    "model": model_path, 
    "act_quant": args.act_quant, 
    "smooth": args.smooth, 
    "mix":args.mix, 
    "w_bits":args.w_bits, 
    "a_bits":args.a_bits
}
json_file = {
    "config" : config,
    "results": results['results']
}


def append_to_json(file_path, new_data):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            # If the file is not a valid JSON, start with an empty list
            data = []
    else:
        # If the file doesn't exist or is empty, start with an empty list
        data = []
    
    # Append new data
    if isinstance(data, list):
        data.append(new_data)
    elif isinstance(data, dict):
        # If it's a dict, we'll add a new key-value pair
        key = max([int(k) for k in data.keys() if k.isdigit()], default=-1) + 1
        data[str(key)] = new_data
    else:
        # If data is neither list nor dict, start a new list with the existing data and new data
        data = [data, new_data]
    
    # Write updated data back to file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


append_to_json(file_path, json_file)
