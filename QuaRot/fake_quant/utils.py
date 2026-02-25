import argparse
import pprint
import torch
import random
import numpy as np
import os
from datetime import datetime
import logging
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import kurtosis


from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

supported_models = [
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-13b-hf',
            'meta-llama/Llama-2-70b-hf',
            'meta-llama/Meta-Llama-3-8B',
            'meta-llama/Meta-Llama-3-70B',
            'facebook/opt-125m'
            ]
supported_datasets = ['wikitext2', 'ptb', 'c4']

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def llama_down_proj_groupsize(model, groupsize):
    
    assert groupsize > 1, 'groupsize should be greater than 1!'
    
    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsiz = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size/groupsize)
    assert groupsize*group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size//group_num
    assert down_proj_groupsize*group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize



def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

# Dump the log both to console and a log file.
def config_logging(log_file, level=logging.INFO):
    class LogFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                self._style._fmt = "%(message)s"
            else:
                self._style._fmt = "%(levelname)s: %(message)s"
            return super().format(record)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(LogFormatter())

    logging.basicConfig(level=level, handlers=[console_handler, file_handler])


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Model to load;')#, choices=supported_models)
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                        help='Dataset for Evaluation (default: wikitext2)', choices=supported_datasets,)
    parser.add_argument('--hf_token', type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument('--cache_dir', type=str, default="/data1/is156025/lm270675/.cache/huggingface/hub")
    parser.add_argument('--bsz', type=int, default=32,
                        help='Batch-size for PPL evaluation (default:32)')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False,
                        help='Plot abs max value of each layer')
    parser.add_argument('--seqlen', type=int, default=2048, help='Context Size')


    # Rotation Arguments
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=False, 
                        help='''Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys''')
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--rotation_seed', type=int, default=-1,
                        help='Random Seed for generating random matrix!!')
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply Hadamard rotation in FP32 (default: False)')

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize')
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric Activation quantization (default: False)')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio')


    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ')


    # General Quantization Arguments
    parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
                        help='Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8')

    # KV-Cache Quantization Arguments
    parser.add_argument('--v_bits', type=int, default=16,
                        help='''Number of bits for V-cache quantization. 
                        Note that quantizing the V-cache does not need any other rotation''')
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric V-cache quantization')
    parser.add_argument('--v_clip_ratio', type=float, default=1.0,
        help='Clip ratio for v-cache quantization. new_max = max * clip_ratio')
    
    parser.add_argument('--k_bits', type=int, default=16,
                        help='''Number of bits for K-cache quantization. 
                        Note that quantizing the K-cache needs another rotation for the keys/queries''')
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False, 
                        help='ASymmetric K-cache quantization')
    parser.add_argument('--k_pre_rope', action=argparse.BooleanOptionalAction, default=False, 
                        help='Pre-RoPE quantization for K-cache (not Supported yet!)')
    parser.add_argument('--k_clip_ratio', type=float, default=1.0,
        help='Clip ratio for k-cache quantization. new_max = max * clip_ratio')


    # Save/Load Quantized Model Arguments
    parser.add_argument('--load_qmodel_path', type=str, default=None,
                        help='Load the quantized model from the specified path!')
    parser.add_argument('--save_qmodel_path', type=str, default=None, 
                        help='Save the quantized model to the specified path!')

    # WandB Arguments
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)



    #Experiments Arguments
    parser.add_argument('--save_name', type=str, default=None, help='The path to save experiment data, '
                                                                    'including quantized models, dumped layer inputs, etc. The data will be saved in experiments/[model]/save_name. Default: [datetime].')
    parser.add_argument('--capture_layer_io', action=argparse.BooleanOptionalAction, default=False,
                        help='Capture the input and output of the specified decoder layer and dump into a file')
    parser.add_argument('--layer_idx', type=int, default=10, help='Which decoder layer to capture')

    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada"],
    )
    parser.add_argument('--lm_eval_batch_size', type=int, default=100, help='Batch size for evaluating with lm eval harness.')
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="Distribute the model on multiple GPUs for evaluation.",
    )

    # Grid Search
    parser.add_argument('--inv', action="store_true", help="Inverse grid search and start from last projection")
    parser.add_argument("--optim_down8b", action="store_true", help="Apply Grid search with down projection in 8 bits")
    parser.add_argument("--grid_search", action="store_true", help="Use grid search for eval")
    parser.add_argument("--eval", action="store_true", help="Compute eval test ppl")
    parser.add_argument("--resume_gs", action="store_true", help="Resume Grid Search from where it stopped")
    parser.add_argument('--start_bit', type=int, default=32,
                        help='''number of bits to start grid search for each layer''')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='''percent of WikiText to use for GBS''')
    parser.add_argument('--max_iterations', type=int, default=10,
                        help='''number of of iteration for GBS''')
    
    # Expand Hadamard
    parser.add_argument('--expand', type=str, default='./save/mix_compute/0.json',
                        help='''config file (json) describing the number of dimensions to expand hadamard matrice for each layer''')
    # Prefix
    parser.add_argument("--outlier_threshold", type=int, default=64, help="\eta in Eq.(3), indicating the oitlier threshold ratio detect outlier tokens, ")
    parser.add_argument("--prefix", action="store_true", help="Use prefix")
    parser.add_argument('--epoch', type=int, default=0,
                        help='''number of epoch to train prefixes''')
    parser.add_argument("--mse_init", action="store_true", help="init step size through MSE instead of MIN-MAX")
    parser.add_argument("--asym_mse_init", action="store_true", help="init step size through MSE instead of MIN-MAX")
    parser.add_argument("--skip_qk_weight_init", action="store_true")
    parser.add_argument("--block_qk_weight_init", action="store_true")
    parser.add_argument("--mse_init_size", type=int, default=8, help="sample number used in mse_init; actually, even 4 or 2 is enough")
    parser.add_argument("--fp_mse_init", action="store_true", help="use full-precision block input during the mse init process")
    parser.add_argument("--calib_dataset",type=str,default="pile",
            choices=["wikitext2", "ptb", "c4", "mix", "redpajama", "pile"],
            help="Where to extract calibration data from.")
    
    # EAM
    parser.add_argument('--eah', default='fp32',
                        type=str, help='fp32 or quant or div')
    parser.add_argument('--draw', action='store_true', help='draw entropy maps')
    parser.add_argument('--thresh_entropy', default=0.1,
                        type=float, help='threshold for mask entropy')
    parser.add_argument('--chunk_size', default=100,
                        type=int, help='number of lines to compute entropy')
    args = parser.parse_args()
    # if args.lm_eval:
    #     from lm_eval import tasks
    #     from lm_eval import utils as lm_eval_utils
    #     # from lm_eval.tasks import initialize_tasks
    #     from lm_eval.api.registry import ALL_TASKS
    #     # initialize_tasks()
    #     for task in args.tasks:
    #         if task not in lm_eval_utils.MultiChoice(ALL_TASKS):
    #             raise ValueError(f"Invalid task: {task}")

    # quant_type = f'w{args.w_bits}a{args.a_bits}_{args.rotate_mode}'
    if args.save_name is None:
        args.save_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    setattr(args, 'save_path',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', args.model, args.save_name))
    os.makedirs(args.save_path, exist_ok=True)

    config_logging(os.path.join(args.save_path, f'{args.save_name}.log'))
    
    assert args.a_groupsize == args.w_groupsize, 'a_groupsize should be the same as w_groupsize!'
    assert args.k_pre_rope == False, 'Pre-RoPE quantization is not supported yet!'

    if args.model == 'facebook/opt-125m' or args.model == 'facebook/opt-1.3b':
        logging.warning('Warning: OPT-125M/1.3B is only for debugging purposes!!')


    if args.wandb:
        assert args.wandb_id is not None and args.wandb_project is not None, 'WandB ID/project is not provided!'
        
    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)
    return args


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def distribute_model(model) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2."""
    no_split_module_classes = ['LlamaDecoderLayer']
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    cleanup_memory()



def plot_outliers(dir, filename, abs_max, model_name):
    import matplotlib.pyplot as plt
    if not os.path.exists(dir):
        os.makedirs(dir)

    print('Token analysis')
    fig, ax = plt.subplots(figsize=(18, 6))
    for name in abs_max:
        m = abs_max[name]
        X = np.linspace(1, len(m), len(m))
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
    plt.savefig(dir + filename, format="pdf")
    plt.close()
    print('Done')

def plot_projection_maxima(model, save_path="projection_max_plot.png"):
    projection_tags = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    max_values = {tag: [] for tag in projection_tags}

    # Step 1: Extract max |W| for each relevant Linear layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            max_val = weight.abs().max().item()
            for tag in projection_tags:
                if tag in name:
                    max_values[tag].append((name, max_val))
                    break

    # Step 2: Plot
    plt.figure(figsize=(16, 8))
    color_palette = plt.get_cmap("tab10")

    for idx, tag in enumerate(projection_tags):
        layers = max_values[tag]
        if not layers:
            continue  # skip missing projections
        # Sort by layer name for consistent order
        layers.sort(key=lambda x: x[0])
        layer_ids = list(range(len(layers)))
        values = [val for _, val in layers]
        plt.plot(layer_ids, values, label=tag, color=color_palette(idx % 10), marker='o')

    plt.xlabel("Layer Index")
    plt.ylabel("Max |Weight|")
    plt.title("Max Absolute Weight per Linear Projection Layer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")


def plot_mean_kurtosis_per_channel_group(model, dir, filename="mean_kurtosis_groups.png", group_size=128):

    projection_tags = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    grouped_kurtosis = {tag: [] for tag in projection_tags}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            weight = module.weight.float().data.cpu().numpy()
            out_features = weight.shape[0]
            num_groups = out_features // group_size

            for tag in projection_tags:
                if tag in name:
                    per_channel_kurts = []
                    for group_idx in range(num_groups):
                        group = weight[group_idx * group_size : (group_idx + 1) * group_size, :]
                        k = kurtosis(group.flatten(), fisher=True)
                        if np.isfinite(k):
                            per_channel_kurts.append(k)
                    avg_kurt = np.mean(per_channel_kurts)
                    grouped_kurtosis[tag].append(avg_kurt)
                    break
    # Plotting
    plt.figure(figsize=(16, 8))
    color_palette = plt.get_cmap("tab10")

    for idx, tag in enumerate(projection_tags):
        data = grouped_kurtosis[tag]
        plt.plot(np.linspace(0, len(data)-1, len(data)), data, label=tag, color=color_palette(idx % 10), marker='o')

    plt.xlabel("Projection Group Index")
    plt.ylabel(f"Mean Kurtosis over {group_size} Output Channels")
    plt.title("Mean Channel-wise Kurtosis per Grouped Projection Layer")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, filename))
    plt.close()
    print(f"Plot saved to: {os.path.join(dir, filename)}")
    return grouped_kurtosis

def plot_delta_kurtosis_per_channel_group(not_rot_kurt, rot_kurt, dir, filename="mean_kurtosis_groups.png"):
    color_palette = plt.get_cmap("tab10")
    for idx, proj in enumerate(not_rot_kurt):
        delta = (np.array(not_rot_kurt[proj]) - np.array(rot_kurt[proj])) / np.array(not_rot_kurt[proj]) * 100
        length = len(delta)
        plt.plot(np.linspace(0, length-1, length), delta, label=proj, color=color_palette(idx % 10), marker='o')
    plt.xlabel("Layer")
    plt.ylabel(f"% Kurtosis")
    plt.title("Decrease of Kurtosis after rotation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.yscale('symlog')
    plt.savefig(os.path.join(dir, filename))