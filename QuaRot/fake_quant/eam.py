from types import MethodType
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache
from transformers.utils import logging
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import math
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import eval_utils
import utils
import data_utils
from main import load_rotate_quantize_model
import model_utils

logger = logging.get_logger(__name__)

def attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # Scaled dot product attention : torch.nn.functional.scaled_dot_product_attention
        L, S = query_states.size(-2), key_states.size(-2)
        is_causal=False
        scale=None
        enable_gqa=False
        dropout_p=self.attention_dropout if self.training else 0.0
        scale_factor = 1 / math.sqrt(query_states.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query_states.dtype, device=query_states.device)
        if is_causal:
            assert causal_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query_states.dtype)

        if causal_mask is not None:
            if causal_mask.dtype == torch.bool:
                attn_bias.masked_fill_(causal_mask.logical_not(), float("-inf"))
            else:
                attn_bias = causal_mask + attn_bias

        if enable_gqa:
            key_states = key_states.repeat_interleave(query_states.size(-3)//key_states.size(-3), -3)
            value_states = value_states.repeat_interleave(query_states.size(-3)//value_states.size(-3), -3)

        attn_weight = query_states @ key_states.transpose(-2, -1) * scale_factor
        # Attention ici j'ai pas trouver de meilleure solution mais c'est pas ouf donc ça peut poser problème
        attn_weight += attn_bias[0]
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        if self.mask_attn is not None:
            attn_weight = attn_weight * (-self.mask_attn.int() + 1) + self.mean_attn
        # compute histograms to get entropy
        c = self.chunk_size
        k = self.current_chunk
        start = k * c
        end = start + c
        if self.entropy:
            with torch.no_grad():
                a = attn_weight[:,:, start:end].detach().clone()
                if self.count == None:
                    self.count = torch.zeros((a.shape[1], c, q_len, 2**self.b), dtype=torch.int16).cuda()
                    self.mean_attn = torch.zeros((a.shape[1], q_len, q_len), dtype=torch.float16).cuda()
                self.count += compute_histogram(a, b=self.b, min_val=0, max_val=1)
                self.mean_attn[:, start:end] += a.mean(dim=0)
        attn_output = attn_weight @ value_states

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


def transform_model(model, args):
    """
    Change the forward pass of the attention module by adding the computation of the entropy and the fixed mask in attention_forward
    """
        
    #for module in model.modules():
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention) or isinstance(module, LlamaSdpaAttention):
            setattr(module, "count", None)
            setattr(module, "entropy", False)
            setattr(module, "b", 4)
            module.forward = MethodType(attention_forward, module)
            setattr(module, "mean_attn", None)
            setattr(module, "mask_attn", None)
            setattr(module, "current_chunk", 0)
            setattr(module, "chunk_size", args.chunk_size)

    return model


def compute_histogram(tensor: torch.Tensor, b: int, min_val: float, max_val: float) -> torch.Tensor:
    """
    Calcule un histogramme de la dimension B d’un tenseur de forme (B, H, K, N).
    
    Args:
        tensor (torch.Tensor): Tenseur d’entrée de forme (B, H, K, N).
        b (int): Nombre de bits pour la quantification => nombre de bins = 2**b.
        min_val (float): Valeur minimale de la plage des valeurs.
        max_val (float): Valeur maximale de la plage des valeurs.
    
    Returns:
        torch.Tensor: Histogramme de forme (H, N, N, 2**b).
    """
    B, H, K, N = tensor.shape
    nbins = 2 ** b

    # Clamp puis scale les valeurs dans [0, nbins-1]
    # tensor = tensor.detach()
    tensor_clamped = tensor.clamp(min=min_val, max=max_val)
    scaled = ((tensor_clamped - min_val) / (max_val - min_val) * (nbins - 1)).round().long()  # Indices des bins
    # del tensor_clamped

    # Tensor des indices de histogramme à remplir
    # Créer un index pour chaque dimension H, N, N et la valeur de bin
    h_idx = torch.arange(H, device=tensor.device).view(1, H, 1, 1).expand(B, H, K, N)
    i_idx = torch.arange(K, device=tensor.device).view(1, 1, K, 1).expand(B, H, K, N)
    j_idx = torch.arange(N, device=tensor.device).view(1, 1, 1, N).expand(B, H, K, N)
    
    histogram = torch.zeros((H, K, N, nbins), device=tensor.device, dtype=torch.int16)
    ONES = torch.ones(B * H * K * N, device=tensor.device, dtype=torch.int16)
    # Accumuler les bins
    histogram.index_put_((h_idx.flatten(), i_idx.flatten(), j_idx.flatten(), scaled.flatten()), ONES, accumulate=True)
    return histogram


def compute_histogram_fast(tensor, b, min_val, max_val):
    """
    Calcule un histogramme de forme (H, N, N, 2**b) à partir d’un tenseur (B, H, N, N).
    """
    B, H, N, _ = tensor.shape
    nbins = 2 ** b

    # Clamp et scale
    tensor = tensor.clamp(min=min_val, max=max_val)
    scaled = ((tensor - min_val) / (max_val - min_val) * (nbins - 1)).round().long()

    # Indices 4D (H, N, N, BINS)
    h_idx = torch.arange(H, device=tensor.device).view(1, H, 1, 1).expand(B, H, N, N)
    i_idx = torch.arange(N, device=tensor.device).view(1, 1, N, 1).expand(B, H, N, N)
    j_idx = torch.arange(N, device=tensor.device).view(1, 1, 1, N).expand(B, H, N, N)

    # Calcule les indices plats pour scatter_add
    flat_idx = (
        h_idx * (N * N * nbins) +
        i_idx * (N * nbins) +
        j_idx * nbins +
        scaled
    ).flatten()

    # Histogramme 1D à remettre en forme ensuite
    histogram = torch.zeros((H * N * N * nbins), device=tensor.device, dtype=torch.int32)
    ones = torch.ones_like(flat_idx, dtype=torch.int32)

    histogram.scatter_add_(0, flat_idx, ones)
    histogram = histogram.view(H, N, N, nbins)

    return histogram

def save_results(args, prec1):
    w_bit = args.w_bits
    a_bit = args.a_bits
    model_name = args.model
    line = [model_name, w_bit, a_bit, args.thresh_entropy, args.eah, prec1]
    import csv
    with open('results.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(line)

def EAM(model, args, train_loader, quant=False):
    folder_savedir = './save/eam/' + args.model
    if not os.path.exists(folder_savedir):
        os.mkdir(folder_savedir)
    if quant:
        suff = '_W' + str(args.w_bits) + 'A' + str(args.a_bits) 
    else:
        suff = ''
    N = model.config.max_position_embeddings 

    if not (os.path.exists(folder_savedir + '/mask_entropy' + suff + '.pth') or os.path.exists(folder_savedir + '/mask_entropy' + suff + '.pkl')):
        print("Performing EAM ...")
        set_entropy(model, True)
        T_H = None
        for k in range(N // args.chunk_size):
            set_current_chunk(model, k)
            eval_utils.evaluator(model, train_loader, utils.DEV, args)
            # Get_entropy return a list of (32,100,2048) tensor
            H_eah = get_entropy(model)
            # T_H is of size (32,32,100,2048)
            t_h = torch.stack(H_eah)
            if T_H is None:
                T_H = t_h
            else:
                T_H = torch.cat((T_H, t_h), dim=2)
        
        M = len(train_loader)
        mean_attn = get_mean_attn(model)
        MEAN_ATTN = torch.stack(mean_attn) / M   
        set_entropy(model, False)
               
        torch.save(T_H, folder_savedir + '/mask_entropy' + suff + '.pth')
        torch.save(MEAN_ATTN, folder_savedir + '/mean_attn' + suff + '.pth')
        
    
    T_H = torch.load(folder_savedir + '/mask_entropy' + suff + '.pth')
    MEAN_ATTN = torch.load(folder_savedir + '/mean_attn' + suff + '.pth')
    
    MASK_ENTROPY = generate_sparsity_mask(T_H, args.thresh_entropy)
    MEAN_ATTN *= MASK_ENTROPY

    if args.draw:
        draw_entropy_maps_full(args, H_eah)
        # draw_entropy_maps(args, H_eah, quant=quant)
    return MASK_ENTROPY, MEAN_ATTN


def get_entropy(model):
    H = []
    for m in model.modules():
        if type(m) in [LlamaAttention, LlamaSdpaAttention]:
            c = m.count.permute(3,0,1,2)
            p = (c / c.sum(dim=0)).permute(1,2,3,0)
            p[p == 0] = 1
            h = - (p * torch.log2(p)).sum(dim=-1)
            H.append(h)
    return H

def get_distribution(model):
    P = []
    for m in model.modules():
        if type(m) in [LlamaAttention, LlamaSdpaAttention]:
            for c in m.count:
                c = c.permute(2,0,1)
                p = (c / c.sum(dim=0)).permute(1,2,0)
                # p[p == 0] = 1
                P.append(p.detach())
    return P


def get_mean_attn(model):
    M = []
    for m in model.modules():
        if type(m) in [LlamaAttention, LlamaSdpaAttention]:   
            M.append(m.mean_attn)
    return M



def draw_divergence_maps(args, P_FP32, P_Q):
    folder = './entropy_maps/' + args.model + '/W' + str(args.w_bits) + 'A' + str(args.a_bits) + '/dvrg'
    if not os.path.exists(folder):
        os.mkdir(folder)
    print('drawing divergence maps...')
    D_mean = []
    D_max = []
    for i, (p_fp32, p_q) in enumerate(zip(P_FP32, P_Q)):
        p_fp32 = p_fp32[0,1:]
        p_q = p_q[0,1:]
        n = int(math.sqrt(p_fp32.shape[0]))
        D = (p_fp32 * torch.log2(p_fp32 / p_q))
        D[p_fp32 == 0] = 0
        D[p_q == 0] = 0
        D = D.sum(dim=-1)
        D = D.reshape((n, n))
        plt.imshow(D.detach().cpu().numpy(), cmap='viridis')
        plt.title(r'$D_{mean} = $' + '{:.2f}'.format(D.mean()))
        plt.colorbar()
        plt.savefig(folder + '/dvrg_head' + str(i) + '.png')
        plt.close()
        D_mean.append(D.mean().item())
        D_max.append(D.max().item())
    x = np.arange(len(D_mean))
    plt.plot(x, D_mean, label='mean')
    plt.plot(x, D_max, label='max')
    plt.title(r'$D_{mean}$ and $D_{max}$ over layers')
    plt.grid(True)
    plt.legend()
    plt.savefig(folder + '/' + r'D_mean_over_layers' + '.png')
    plt.close()

def draw_divergence_maps_full(args, P_FP32, P_Q):
    folder = './entropy_maps/' + args.model + '/W' + str(args.w_bits) + 'A' + str(args.a_bits) + '/dvrg_full'
    if not os.path.exists(folder):
        os.mkdir(folder)
    print('drawing divergence maps...')
    D_mean = []
    D_max = []
    for i, (p_fp32, p_q) in enumerate(zip(P_FP32, P_Q)):
        D = (p_fp32 * torch.log2(p_fp32 / p_q))
        D[p_fp32 == 0] = 0
        D[p_q == 0] = 0
        D = D.sum(dim=-1)
        plt.imshow(D.detach().cpu().numpy(), cmap='viridis')
        plt.title(r'$D_{mean} = $' + '{:.2f}'.format(D.mean()))
        plt.colorbar()
        plt.savefig(folder + '/dvrg_full_head' + str(i) + '.png')
        plt.close()
        D_mean.append(D.mean().item())
        D_max.append(D.max().item())
    x = np.arange(len(D_mean))
    plt.plot(x, D_mean, label='mean')
    plt.plot(x, D_max, label='max')
    plt.title(r'$D_{mean}$ and $D_{max}$ over layers')
    plt.grid(True)
    plt.legend()
    plt.savefig(folder + '/' + r'D_mean_over_layers' + '.png')
    plt.close()

def draw_entropy_maps(args, H, quant=False):
    if not quant:
        folder = './entropy_maps/'+ args.model + '/fp32' 
    else:
        folder = './entropy_maps/' + args.model + '/W' + str(args.w_bits) + 'A' + str(args.a_bits)
    if not os.path.exists(folder):
        os.mkdir(folder)
    print('drawing entropy maps...')
    H_mean = []
    H_max = []
    for i, h in enumerate(H):
        h = h[0,1:]
        h = h.reshape((int(math.sqrt(h.shape[0])), int(math.sqrt(h.shape[0]))))
        plt.imshow(h.detach().cpu().numpy(), cmap='viridis')
        plt.title(r'$H_{mean} = $' + '{:.2f}'.format(h.mean()))
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16) 
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(folder + '/head' + str(i) + '.png')
        plt.close()
        H_mean.append(h.mean().item())
        H_max.append(h.max().item())
    x = np.arange(len(H_mean))
    plt.plot(x, H_mean, label='mean')
    plt.plot(x, H_max, label='max')
    plt.title(r'$H_{mean}$ and $H_{max}$ over layers')
    plt.grid(True)
    plt.legend()
    plt.savefig(folder + '/' + r'H_mean_over_layers' + '.png')
    plt.close()


def draw_entropy_maps_full(args, H, quant=False):
    # H is size L x C x N x N
    L, C, N, _ = H.shape
    if not quant:
        folder = './entropy_maps/'+ args.model + '/fp32' 
    else:
        folder = './entropy_maps/' + args.model + '/W' + str(args.w_bits) + 'A' + str(args.a_bits)
    if not os.path.exists(folder):
        os.mkdir(folder)
    print('drawing entropy maps...')
    H = H.reshape((L*C, N, N))
    H_mean = []
    H_max = []
    for i in range(L*C):
        plt.imshow(H[i].detach().cpu().numpy(), cmap='viridis')
        plt.title(r'$H_{mean} = $' + '{:.2f}'.format(H[i].mean()), fontsize=24)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16) 
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(folder + '/head_full' + str(i) + '.pdf')
        plt.close()
        H_mean.append(H[i].mean().item())
        H_max.append(H[i].max().item())
    x = np.arange(len(H_mean))
    plt.plot(x, H_mean, label='mean')
    plt.plot(x, H_max, label='max')
    plt.title(r'$H_{mean}$ and $H_{max}$ over layers')
    plt.grid(True)
    plt.legend()
    plt.savefig(folder + '/' + r'H_mean_full_over_layers' + '.pdf')
    plt.close()


def save_checkpoint(model, args):
    name = "./savedir/" + args.model + ".pth"
    torch.save(model.state_dict(), name)


def set_entropy(model, v):
    for name, module in model.named_modules():
        if isinstance(module, (LlamaAttention, LlamaSdpaAttention)):
            module.entropy = v

def set_current_chunk(model, k):
    for name, module in model.named_modules():
        if isinstance(module, (LlamaAttention, LlamaSdpaAttention)):
            module.current_chunk = k

def set_mask_entropy(model, mask):
    l = 0
    for name, module in model.named_modules():
        if isinstance(module, (LlamaAttention, LlamaSdpaAttention)):
            module.mask_attn = mask[l].cuda()
            l+=1

def set_mean_attn(model, mean_attn):
    l = 0
    for name, module in model.named_modules():
        if isinstance(module, (LlamaAttention, LlamaSdpaAttention)):
            module.mean_attn = mean_attn[l].cuda()
            l+=1


def generate_sparsity_mask(T: torch.Tensor, p: float) -> torch.Tensor:
    """
    Génère un masque binaire (0/1) où p% des plus petites valeurs de T sont mises à 0.

    Args:
        T (torch.Tensor): Le tenseur d'entrée.
        p (float): Le pourcentage de parcimonie (entre 0 et 1). Par exemple, 0.2 pour 20%.

    Returns:
        torch.Tensor: Un masque binaire du même shape que T avec 1 pour les p% plus petites valeurs et 0 ailleurs.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("Le pourcentage p doit être entre 0.0 et 1.0")

    # Aplatir le tenseur pour faciliter le tri
    flat_T = T.view(-1)

    # Calcul du seuil basé sur les p% plus petites valeurs
    k = int(p * flat_T.numel())
    if k == 0:
        return torch.ones_like(T, dtype=torch.uint8)  # Pas de valeur à mettre à 0

    # Trouver la valeur seuil
    threshold, _ = torch.kthvalue(flat_T.abs(), k)

    # Créer le masque
    mask = (T.abs() > threshold).to(dtype=torch.uint8)

    return -mask + 1

def random_mask(args):
    folder_savedir = './save/eam/' + args.model
    if not os.path.exists(folder_savedir + '/mask_entropy.pth'):
        print('Run --eah fp32 first')
        T_H = torch.load(folder_savedir + '/mask_entropy.pth')
        MEAN_ATTN = torch.load(folder_savedir + '/mean_attn.pth')
        MASK_ENTROPY = generate_sparsity_mask(T_H, args.thresh_entropy)
        idx = torch.randperm(MASK_ENTROPY.nelement())
        MASK_ENTROPY = MASK_ENTROPY.view(-1)[idx].view(MASK_ENTROPY.size())
    return MASK_ENTROPY, MEAN_ATTN


if __name__ == '__main__':
    args = utils.parser_gen()
    testloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=args.seqlen,
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
        eval_mode=True
    )
    trainloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=args.seqlen,
        hf_token=args.hf_token,
        eval_mode=False,
        cache_dir=args.cache_dir,
        grid = True
    )
    size = int(args.alpha * trainloader.input_ids.shape[-1])
    trainloader.input_ids = trainloader.input_ids[:,:size]
    model = model_utils.get_model(args.model, args.hf_token, args.cache_dir)
    transform_model(model, args)

    if args.eah == 'fp32':
        MASK_ENTROPY, MEAN_ATTN = EAM(model, args, trainloader)
    # Randomly generate a mask
    if args.eah == 'random':
        MASK_ENTROPY, MEAN_ATTN = random_mask(args)


    qmodel, _ = load_rotate_quantize_model(args, model)

    if args.eah == 'quant':
        MASK_ENTROPY, MEAN_ATTN = EAM(qmodel, args, trainloader, True)
        
    # Draw divergence
    if args.eah == 'div':
        set_entropy(model, True)
        eval_utils.evaluator(model, trainloader, utils.DEV, args)
        set_entropy(model, False)
        P_fp32 = get_distribution(model)
        set_entropy(qmodel, True)
        eval_utils.evaluator(qmodel, trainloader, utils.DEV, args)
        set_entropy(qmodel, False)
        P_q = get_distribution(qmodel)
        draw_divergence_maps_full(args, P_fp32, P_q)

    # if none: PTQ classic
    elif args.eah != 'none':
        set_mask_entropy(qmodel, MASK_ENTROPY)
        print(f"{args.thresh_entropy*100:.3f}% masked attention")
        set_mean_attn(qmodel, MEAN_ATTN)

    # Validation
    test_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    save_results(args, test_ppl)