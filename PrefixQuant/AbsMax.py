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
import utils.rotation_utils as rotation_utils
from utils.quant_utils import register_online_had, wrap_to_quant_model
import random
from utils.quant_utils import init_k_quantizer, init_input_quantizer, init_v_quantizer, init_weight_quantizer
import argparse

    
parser = argparse.ArgumentParser()
cache_dir = '/data1/is156025/lm270675/.cache/huggingface/hub'
login(token=os.environ.get("HF_TOKEN"))
N = 2
dataset = 'wiki'
io = 'input'
n_layers = {'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B' : 48}
# 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' : 32,
# 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B' : 40
# 'meta-llama/Llama-3.2-1B' : 16,
# 'meta-llama/Llama-3.1-8B' : 32,
# 'meta-llama/Llama-2-13b-hf': 40, 
# 'mistralai/Mistral-7B-v0.1' : 32, 
# 'meta-llama/Llama-2-7b-hf': 32, 
# 'meta-llama/Meta-Llama-3-8B' : 32,
# 'facebook/opt-13b': 40
# 'bigscience/bloom-7b1' : 32
context_size = 1024
ALPHA = 0.1
from_ft = False
BF16 = True

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser.add_argument("--wbits", type=int, default=16, help="quantization bits")
parser.add_argument("--w_group_size", type=int, default=-1, help="quantization group size")
parser.add_argument("--w_asym", dest="w_asym", action="store_true", help="Set w_asym to True")
parser.add_argument("--w_sym", dest="w_asym", action="store_false", help="Set w_asym to False")
parser.set_defaults(w_asym=False)
parser.add_argument("--input_bits", type=int, default=16, help="quantization bits")
parser.add_argument("--input_group_size", type=int, default=-1, help="quantization group size")
parser.add_argument("--input_mode", type=str, default='dynamic',help="quantization type")
parser.add_argument("--input_asym", dest="input_asym", action="store_true", help="Set input_asym to True")
parser.add_argument("--input_sym", dest="input_asym", action="store_false", help="Set input_asym to False")
parser.set_defaults(input_asym=False)
parser.add_argument("--k_bits", type=int, default=16,help="")
parser.add_argument("--v_bits", type=int, default=16,help="")
parser.add_argument("--kv_group_size", type=int, default=128,help="default as head-wise")
parser.add_argument("--k_pre_rope", action="store_true")
parser.add_argument("--kv_mode", type=str, default='dynamic',help="quantization type")
parser.add_argument("--kv_asym", dest="kv_asym", action="store_true", help="Set kv_asym to True")
parser.add_argument("--kv_sym", dest="kv_asym", action="store_false", help="Set kv_asym to False")
parser.set_defaults(kv_asym=False)
parser.add_argument("--mse_init", action="store_true", help="init step size through MSE instead of MIN-MAX")
parser.add_argument("--asym_mse_init", action="store_true", help="init step size through MSE instead of MIN-MAX")
parser.add_argument("--skip_qk_weight_init", action="store_true")
parser.add_argument("--block_qk_weight_init", action="store_true")
parser.add_argument("--mse_init_size", type=int, default=8, help="sample number used in mse_init; actually, even 4 or 2 is enough")
parser.add_argument("--fp_mse_init", action="store_true", help="use full-precision block input during the mse init process")

args = parser.parse_args()

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

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

for model_name in n_layers:
    print('-----' + model_name + '-----')
    abs_max = {}
    attn = []
    ind_sample = 0
    hooks = []
    # act = {}
    # act_scales_path = "/data1/is156025/lm270675/meta-labo/LLM/smoothquant/act_scales/" + model_name + "/act_scales.sm"
    # tokenizer, model = load_short_model(model_name, N, dataset)
    if not from_ft:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir, output_attentions=True).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained('./models/' + model_name + '/wiki/fine_tune_' + str(ALPHA) + '_bf16/' if BF16 else '_fp16/', torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    def get_dataset(name):
        
        prompt = (
                f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
            )

        def apply_prompt_template(sample):
            return {
                "prompt": prompt.format(dialog=sample["dialogue"]),
                "summary": sample["summary"],
            }
        def tokenize_add_label(sample):
            prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
            summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)
            sample = {
                "input_ids": prompt + summary,
                "attention_mask" : [1] * (len(prompt) + len(summary)),
                "labels": [-100] * len(prompt) + summary,
                }
            
            return sample
        
        def tokenize_wiki(sample):
            prompt = tokenizer.encode(tokenizer.bos_token + sample["text"] +  tokenizer.eos_token, add_special_tokens=False)
            sample = {
                "input_ids": prompt,
                "attention_mask" : [1] * len(prompt),
                "labels": prompt,
                }
            return sample
        
        if name == 'samsum':
            dataset = load_dataset("samsum", revision="refs/convert/parquet", split="train")
            dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
            dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
        if name == 'wiki':
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            dataset = dataset.filter(lambda example: example['text'] != '' and '=' not in example['text'])
            #dataset = dataset.map(tokenize_wiki, remove_columns=list(dataset.features), batched=True)

        return dataset
    
    def H(X):
        X[X == 0] = 1
        return - (X * np.log2(X)).sum()

    def get_hook(name, type):
        
        def hook_abs(model, input, output):
            global abs_max
            global ind_sample
             
            if abs_max[name] == None:
                abs_max[name] = torch.zeros((n_layers[model_name],) )

            if io == 'input':
                feature = input[0]
            else:
                feature = output

            # if act[name] == None:
            #     act[name] = torch.zeros((n_layers[model_name],) + tuple(feature.shape))
            
            abs_max[name][ind_sample // len(abs_max.keys())] = feature.detach().flatten().abs().max()
            # act[name][ind_sample // len(abs_max.keys())] = feature.abs()

            ind_sample += 1

        def hook_attn(model, input, output):
            global attn
            
            attn.append(output[1].detach())
            return
        if type == 'abs':
            return hook_abs
        if type == 'attn':
            return hook_attn
    
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # data_tok = tokenizer(
    #             "\n\n".join(data["text"]), return_tensors="pt"
    #         ).input_ids.to(model.device)
    PPL = Evaluator(data, tokenizer, "cuda")
    model.eval()

    # Prefix tokens
    # prefix_tokens = tokenizer.encode(".\n" + tokenizer.bos_token)[1:]
    # prefix_tokens = tokenizer.encode(".")[1:] + tokenizer.encode("\n")[1:] + tokenizer.encode(tokenizer.bos_token)[1:] 

    # prefix_tokens = tokenizer.encode(tokenizer.bos_token)[1:]
    # prefix_tokens = tokenizer.encode("the." + tokenizer.bos_token)[1:]

    # output = model(torch.tensor([prefix_tokens],device=model.device),return_dict=True)
    # prefixed_key_values = output.past_key_values
    # model.config.use_cache = True

    # prefixed_key_values = model_utils.mv_kv_cache(prefixed_key_values, model)

    # Hook for attention maps
    # print("Hooking...")
    # for name, layer in model.named_modules():
    #     if isinstance(layer, LlamaDecoderLayer):
    #         hooks.append(layer.register_forward_hook(get_hook('', 'attn')))

    
    # # n_samples = 1 #data.size(1) // context_size
    
    # attn = []
    # batch = PPL.dataset[:, :context_size].cuda()
    # with torch.no_grad():
    #     model(batch)

    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # folder = "attn_weights/" + model_name + '/'
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # for i in range(32):
    #     fig, ax = plt.subplots(figsize=(12, 12))
    #     tokens = tokenizer.decode(batch[0]).split()
    #     N = 50
    #     S = 0
    #     sns.heatmap(attn[0][0,i].cpu().numpy()[S:S + N, S:S + N], cmap="Blues", xticklabels=tokens[S:S + N], yticklabels=tokens[S:S + N])
    #     ax.set_title("Attention Map")
    #     ax.set_xlabel("Query Tokens")
    #     ax.set_ylabel("Key Tokens")
    #     plt.savefig(folder + str(i) + ".png")
    #     plt.close()

    
    path = "./outliers/rotations/" + io + '/' + model_name
    if from_ft:
        path = path + '_ft_' + str(ALPHA) + '_bf16' if BF16 else '_fp16'
    if not os.path.exists(path):
        os.makedirs(path)


    # Rotations
    # rotation_utils.fuse_layer_norms(model, rmsn = False)
    # rotation_utils.rotate_model(model, rotate_mode='hadamard', online=True)
    # model.half()

    # # Quantization
    # wrap_to_quant_model(model)

    # # Online hadamrd matrix
    # register_online_had(model)

    # # wrap rope for online_had and rope output capture
    # rope_function_name = model_utils.get_rope_function_name(model)
    # layers = model_utils.get_layers(model)
    # for layer in layers:
    #     rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
    #                 layer.self_attn, 
    #                 rope_function_name, 
    #                 config=model.config,
    #                 online_had=True) 
        
    
    
    # Hook abs value
    for name, layer in model.named_modules():
        if len(layer._modules) == 0 and 'emb' not in name and ('.layer' in name or '.h' in name):
            name_ = name.split('.')[-1]
            abs_max[name_] = None
            # act[name_] = None
            hooks.append(layer.register_forward_hook(get_hook(name_, 'abs')))


    attn = []
    ind_sample = 0
    batch = PPL.dataset[:, :context_size].cuda()[:, 1:]
    with torch.no_grad():
        model(batch)# , past_key_values=prefixed_key_values)


    print('Token analysis')
    fig, ax = plt.subplots(figsize=(18, 6))
    for name in abs_max:
        m = abs_max[name]
        X = np.linspace(1, m.shape[0], m.shape[0])
        if name == 'post_attention_layernorm':
            name = 'post_attn_layernorm'
        plt.plot(X, m, label = name, linewidth=5)

    plt.grid()
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.xlabel('N° Layer', fontsize=35)
    plt.ylabel('Abs max', fontsize=35)
    plt.title('Abs max over layer for a ' + model_name)
    plt.legend(bbox_to_anchor=(1.3, 0.5), loc='center', fontsize=20)
    plt.tight_layout()
    plt.savefig(path + "/absmax_over_layer.pdf", format="pdf")
    plt.close()
    print('Done')

    PPL
    attn = []
    ind_sample = 0
    for hook in hooks:
        hook.remove()
    # prefixed_key_values = model_utils.mv_kv_cache(prefixed_key_values, model)
    print(PPL.evaluate(model).item())#, past_key_values=prefixed_key_values).item())