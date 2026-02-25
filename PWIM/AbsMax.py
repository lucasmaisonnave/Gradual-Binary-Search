from load import load_short_model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import tqdm
import matplotlib.pyplot as plt
from smoothquant.smoothquant.smooth import smooth_lm
from smoothquant.smoothquant.fake_quant import quantize_model
import numpy as np
import os
from smoothquant.smoothquant.calibration import get_act_scales
import matplotlib.animation as animation
cache_dir = '/data1/is156025/lm270675/.cache/huggingface/hub'

N = 2
dataset = 'wiki'
io = 'input'
n_layers = {'meta-llama/Llama-3.1-8B' : 32}
# 'meta-llama/Llama-3.2-1B' : 16,
# 'meta-llama/Llama-3.1-8B' : 32,
# 'meta-llama/Llama-2-13b-hf': 40, 
# 'mistralai/Mistral-7B-v0.1' : 32, 
# 'meta-llama/Llama-2-7b-hf': 32, 
# 'meta-llama/Meta-Llama-3-8B' : 32,
# 'facebook/opt-13b': 40
# 'bigscience/bloom-7b1' : 32
context_size = 2048
ALPHA = 0.1
token = 0
from_ft = False
BF16 = True

for model_name in n_layers:
    print('-----' + model_name + '-----')
    entropy = {}
    delta = {}
    abs_max = {}
    histo = {}
    act = {}
    act_scales_path = "/data1/is156025/lm270675/meta-labo/LLM/smoothquant/act_scales/" + model_name + "/act_scales.sm"
    # tokenizer, model = load_short_model(model_name, N, dataset)
    if not from_ft:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained('./models/' + model_name + '/wiki/fine_tune_' + str(ALPHA) + '_bf16/' if BF16 else '_fp16/', torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # if not os.path.exists(act_scales_path):
    #     act_scales = get_act_scales(
    #     model, tokenizer, "/data1/is156025/lm270675/meta-labo/EAH-ViT/LLM/smoothquant/dataset/val.jsonl.zst", 512, context_size
    #     )
    #     os.makedirs(os.path.dirname(act_scales_path), exist_ok=True)
    #     torch.save(act_scales, act_scales_path)
    # else:
    #     act_scales = torch.load(act_scales_path)
    
    # print("SmoothQuanting...")
    # smooth_lm(model, act_scales, 0.5)
    # model = quantize_model(
    #     model,
    #     weight_quant="per_channel",
    #     act_quant="per_tensor",
    #     quantize_bmm_input=True,
    # )

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

    def get_hook(name):
        def hook(model, input, output):
            global entropy
            global delta
            global ind_sample
            global histo
            global act
            b = 12
             
            if entropy[name] == None:
                entropy[name] = torch.zeros((n_layers[model_name], ))
                delta[name] = torch.zeros((n_layers[model_name], ) )
                abs_max[name] = torch.zeros((n_layers[model_name],) )
                histo[name] = np.zeros((n_layers[model_name], 2, 2**b + 1) )

            if io == 'input':
                feature = input[0]
            else:
                feature = output

            if act[name] == None:
                act[name] = torch.zeros((n_layers[model_name],) + tuple(feature.shape))
            
            abs_max[name][ind_sample // len(entropy.keys())] = feature.flatten().abs().max()
            act[name][ind_sample // len(entropy.keys())] = feature.abs()
            # f = feature.flatten().detach().cpu()
            # hist = np.histogram(f, 2**b)
            # histo[name][ind_sample // len(entropy.keys()), 0][:-1] = hist[0]
            # histo[name][ind_sample // len(entropy.keys()), 1] = hist[1]
            # hist = hist / hist.sum()
            # h = H(hist)
            # entropy[name][ind_sample // len(entropy.keys())] = h

            
            # f = feature.detach().cpu()
            # f[:,token] = f[:,token] * 0
            # f = f.flatten()
            # hist = np.histogram(f, 2**b)[0]
            # hist = hist / hist.sum()
            # h2 = H(hist)
            # delta[name][ind_sample // len(entropy.keys())] = h2 - h

            ind_sample += 1
        return hook
    
    data = get_dataset(dataset)
    model.eval()

    print("Hooking...")
    for name, layer in model.named_modules():
        if len(layer._modules) == 0 and 'emb' not in name and ('.layer' in name or '.h' in name):
            name_ = name.split('.')[-1]
            entropy[name_] = None
            delta[name_] = None
            abs_max[name_] = None
            act[name_] = None
            layer.register_forward_hook(get_hook(name_))

    data = tokenizer(
                "\n\n".join(data["text"]), return_tensors="pt"
            ).input_ids.to(model.device)
    n_samples = 1 #data.size(1) // context_size
    ind_sample = 0

    for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
        batch = data[:, (i * context_size) : ((i + 1) * context_size)].cuda()
        with torch.no_grad():
            model(batch)
    path = "./iconip/activations/" + io + '/' + model_name
    if from_ft:
        path = path + '_ft_' + str(ALPHA) + '_bf16' if BF16 else '_fp16'
    if not os.path.exists(path):
        os.makedirs(path)
    # print("Plotting Delta...")
    # fig, ax = plt.subplots(figsize=(10, 5))
    # for k in delta:
    #     h = delta[k]
    #     X = np.linspace(1, h.shape[0], h.shape[0])
    #     plt.plot(X, h, label = k)

    # plt.grid()
    # plt.xlabel('N° Layer')
    # plt.ylabel('Entropy')
    # plt.title('Delta entropy over layer for a ' + model_name + ' W8A8 quantization, token ' + str(token))
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='center left')
    # plt.tight_layout()
    # plt.savefig(path + "/delta_entropy_over_layer_token" + str(token) + ".png")
    # plt.close()
    # print('Done')

    # print("Plotting Entropy...")
    # fig, ax = plt.subplots(figsize=(10, 5))
    # for k in entropy:
    #     h = entropy[k]
    #     X = np.linspace(1, h.shape[0], h.shape[0])
    #     plt.plot(X, h, label = k)

    # plt.grid()
    # plt.xlabel('N° Layer')
    # plt.ylabel('Entropy')
    # plt.title('Entropy over layer for a ' + model_name + ' W8A8 quantization')
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='center left')
    # plt.tight_layout()
    # plt.savefig(path + "/entropy_over_layer.png")
    # plt.close()
    # print('Done')

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

    # print('Animate histogram')

    # fps = 2
    # frn = n_layers[model_name]

    # def create_anim(name, bins):
    #     fig, ax = plt.subplots()
    #     bar_container = ax.bar(x = histo[name][0,1, :-1], height = histo[name][0,0, :-1], width = 0.01, color='royalblue', edgecolor = "midnightblue")
    #     title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
    #                     transform=ax.transAxes, ha="center")
    #     def prepare_animation(bar_container):

    #         def animate(frame_number):
    #             # simulate new data coming in
    #             ax.clear()
    #             bar_container = ax.bar(x = histo[name][frame_number,1, :-1], height = histo[name][frame_number,0, :-1], width = 0.01, color='royalblue', edgecolor = "midnightblue")
    #             title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
    #                     transform=ax.transAxes, ha="center")
    #             title.set_text("Histogram of " + name + ", " + model_name + u", layer = {}".format(frame_number))
    #             # for count, rect in zip(histo[name][frame_number,0, :-1], bar_container.patches):
    #             #     rect.set_height(count)
    #             # return bar_container.patches
    #             return bar_container
    #         return animate

    #     ani = animation.FuncAnimation(fig, prepare_animation(bar_container), frn,
    #                                 repeat=False, blit=True, interval=1000 / fps)
        
    #     if not os.path.exists(path + "/hist/"):
    #         os.makedirs(path + "/hist/")

    #     ani.save(path + "/hist/" + name + ".gif")

    # for name in histo:
    #     print(name)
    #     bins = histo[name]
    #     create_anim(name, bins)


    # Plot Activations
    # layer = 19
    # name = 'k_proj'
    # from matplotlib import cm

    # X = np.linspace(0, act[name].shape[-1] - 1, act[name].shape[-1])
    # Y = np.linspace(0, act[name].shape[-2] - 1, act[name].shape[-2])
    # X, Y = np.meshgrid(X, Y)
    # Z = act[name][layer, 0].detach().cpu().numpy()

    
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    # #cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(path + "/act_" + name + str(layer) + ".png", format="png", dpi=200)