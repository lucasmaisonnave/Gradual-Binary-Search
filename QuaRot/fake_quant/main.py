import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import hadamard_utils
import os
import torch.nn as nn
import torch.multiprocessing as mp
from model_utils import QWEN_MODEL
import json
import matplotlib.pyplot as plt
import numpy as np
io = 'input'
abs_max_before_quant = {}
abs_max_after_quant = {}
hooks = []



def update_quantizer(args, model, config):
    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and ("llama" in args.model or 'Qwen' in args.model or 'mistral' in args.model or 'deepseek' in args.model):
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
        
        for name in qlayers:            
            
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio
            layer_input_bits = args.a_bits
            layer_v_clip = args.v_clip_ratio
            if config is not None and 'lm_head' not in name:
                q_layer = int(name.split('.')[2])
                q_proj = name.split('.')[-1]
                layer_a_clip = config['max_bit'][q_layer][q_proj]['max']
                layer_input_bits = config['max_bit'][q_layer][q_proj]['bit']
                layer_v_clip = config['max_bit'][q_layer][q_proj]['max']
            
            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                            groupsize=args.v_groupsize,
                                            sym=not(args.v_asym),
                                            clip_ratio=layer_v_clip)
            
            if 'lm_head' in name: #Skip lm_head quantization   
                layer_input_bits = 16
            
            if 'down_proj' in name: #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

                
            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                            groupsize=layer_groupsize,
                                            sym=layer_a_sym,
                                            clip_ratio=layer_a_clip)

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            qlayers = quant_utils.find_qlayers(model, layers=[rotation_utils.QKRotationWrapper])
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                        "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for l, name in enumerate(qlayers):
                if config is not None:
                    k_quant_config["k_clip_ratio"] = config['max_bit'][l]['qk_rotation']['max']
                    k_quant_config["k_bits"] = config['max_bit'][l]['qk_rotation']['bit']
                qlayers[name].k_quantizer.configure(bits=k_quant_config["k_bits"], groupsize=-1, #we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                                   sym=k_quant_config["k_sym"], clip_ratio=k_quant_config["k_clip_ratio"])

def load_rotate_quantize_model(args, model, config = None):
        
    transformers.set_seed(args.seed)
    
    model.eval()
    prefixed_key_values=None
    try:
        with open(args.expand, 'r') as file:
            config_mix = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError('Expand file not found')
    expand_mlp = rotation_utils.AbsorbingDict(config_mix['mlp'])
    # Rotate the weights
    if args.rotate:
        if isinstance(model, QWEN_MODEL):
            rotation_utils.split_embed_head(model)
        rotation_utils.fuse_layer_norms(model)
        not_rot_kurt = utils.plot_mean_kurtosis_per_channel_group(model, './weights/' + args.model, 'not_rot')
        rotation_utils.rotate_model_mix_compute(model, args)
        rot_kurt = utils.plot_mean_kurtosis_per_channel_group(model, './weights/' + args.model,  'rot')
        if args.plot:
            utils.plot_delta_kurtosis_per_channel_group(not_rot_kurt, rot_kurt, './weights/' + args.model, 'delta')
            exit()
        # rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)
            
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if name != 'lm_head':
                i = name.split('.')[2]
            if 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size + expand_mlp[i])
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name and not isinstance(model, QWEN_MODEL):
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present
        
    if args.prefix:
        path_prefix_folder = './save/prefix/' + args.model 
        if os.path.exists(path_prefix_folder):
            prefixed_key_values = torch.load(path_prefix_folder + '/prefixed_key_values.pth')
        else:
            print("Prefix File not found or Prefix must be computed first : " + path_prefix_folder)
            exit()
    model.half()
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path, weights_only=False)
            model.load_state_dict(save_dict["model"])
            
        elif not args.w_rtn: # GPTQ Weight Quantization
            assert "llama" in args.model or 'deepseek' in args.model or 'mistral' in args.model or 'Qwen' in args.model or 'Phi' in args.model, "Only llama is supported for GPTQ!"
            
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False,
                hf_token=args.hf_token,
                cache_dir=args.cache_dir
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, model.device, args)
            save_dict["w_quantizers"] = quantizers
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
            
        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                        "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for l, layer in enumerate(layers):
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn, 
                            rope_function_name, 
                            config=model.config,
                            **k_quant_config)

    update_quantizer(args, model, config)

    return model.cuda(), prefixed_key_values


def main():    
    

    def get_hook(name, type):
        
        def hook_abs_before_quant(model, input, output):
            global abs_max_before_quant
                
            if abs_max_before_quant[name] == None:
                abs_max_before_quant[name] = []

            if io == 'input':
                feature = input[0]
            else:
                feature = output
            
            abs_max_before_quant[name].append(feature.detach().flatten().abs().max().item())
            # act[name][ind_sample // len(abs_max.keys())] = feature.abs()
        
        def hook_abs_after_quant(model, input, output):
            global abs_max_after_quant
                
            if abs_max_after_quant[name] == None:
                abs_max_after_quant[name] = []

            if io == 'input':
                feature = input[0]
            else:
                feature = output
            
            abs_max_after_quant[name].append(feature.detach().flatten().abs().max().item())
        if type == 'before':
            return hook_abs_before_quant
        if type == 'after':
            return hook_abs_after_quant
    
    import json
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
    folder = './grid_search/' + args.model + '/'
    filename = 'rotate' if args.rotate else '' 
    filename += '_W_RTN' if args.w_rtn else '_W_GPTQ' 
    filename += '_inv' if args.inv else '' 
    filename += '_W' + str(args.w_bits) + 'A' + str(args.a_bits) + 'KV' + str(args.v_bits) 
    filename += '_start_bit_'  + str(args.start_bit)
    # filename += '_expand_' + str(int(args.expand.split('/')[-1].split('.')[0].split('_')[0])) if '/0.json' not in args.expand else ''
    filename += ".json"
    config = None
    if args.grid_search:
        with open(folder + 'best_config_' + filename, 'r') as file:
            config = json.load(file)[0]['config']
    model = model_utils.get_model(args.model, args.hf_token, args.cache_dir)
    model, prefixed_key_values = load_rotate_quantize_model(args, model, config)
    bf_prefixed_key_values = None
    if prefixed_key_values is not None:
        bf_prefixed_key_values = ()
        for kv in prefixed_key_values:
            bf_prefixed_key_values += ((kv[0].to(torch.bfloat16), kv[1].to(torch.bfloat16)),)
    
    
    # Plot abs value
    if args.plot:
        # Evaluating on dataset
        testloader = data_utils.get_loaders(
                args.eval_dataset,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
                hf_token=args.hf_token,
                cache_dir=args.cache_dir,
                eval_mode=True
            )
        for name, layer in model.named_modules():
            qu = name.split('.')[-1]
            if 'proj' in qu:
                abs_max_before_quant[qu] = None
                abs_max_after_quant[qu] = None
                # act[name_] = None
                hooks.append(layer.register_forward_hook(get_hook(qu, 'before')))
                hooks.append(layer.module.register_forward_hook(get_hook(qu, 'after')))

        batch = testloader.input_ids[:, :model.seqlen].cuda()  # (1, text_len)
        for i, m in enumerate(model.model.layers):
            model.model.layers[i] = model.model.layers[i].cuda()
        model.model.embed_tokens = model.model.embed_tokens.cuda()
        with torch.no_grad():
            model(batch, past_key_values=bf_prefixed_key_values)# 

        path = "./outliers/rotations/" + io + '/' + args.model

        utils.plot_outliers(path,"/absmax_{}bits_before.pdf".format(args.w_bits), abs_max_before_quant, args.model)
        utils.plot_outliers(path,"/absmax_{}bits_after.pdf".format(args.w_bits), abs_max_after_quant, args.model)

        for hook in hooks:
            hook.remove()

        for i, m in enumerate(model.model.layers):
            model.model.layers[i] = model.model.layers[i].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        return
    
    # Eval ppl
    # dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    # if args.wandb:
    #     wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})

    if not args.lm_eval:
        return
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM

        
    
    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    if False:
        utils.cleanup_memory()
        input_ids = tokenizer("hello how are you?", return_tensors="pt").input_ids.cuda()
        output = model.generate(input_ids, max_length=30)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(generated_text)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    # task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    results = lm_eval.simple_evaluate(hflm, tasks=args.tasks, batch_size=args.lm_eval_batch_size)['results']

    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    print(metric_vals)
    L = [args.model,args.w_bits,args.grid_search,int(args.expand.split('/')[-1].split('.')[0].split('_')[0])]
    for v in metric_vals:
        L.append(metric_vals[v])
    import csv
    csv_file = "./results/results_tasks.csv"
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(L)
    if args.wandb:
        wandb.log(metric_vals)


if __name__ == '__main__':
    main()


