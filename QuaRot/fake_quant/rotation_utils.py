import model_utils
import torch
import typing
import utils
import transformers
import tqdm, math
import quant_utils
from hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2 # , hadamard_transform
from fast_hadamard_transform import hadamard_transform
import json

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
            
def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)

         
            
def fuse_layer_norms(model):
    
    model_type = model_utils.get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion
    # if not isinstance(model, model_utils.QWEN_MODEL):
    for W in model_utils.get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
        
    layers = model_utils.get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.QWEN_MODEL or model_type == model_utils.MISTRAL_MODEL :
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == model_utils.PHI_MODEL:
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.qkv_proj])
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.gate_up_proj])
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
    
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    if model_type == model_utils.LLAMA_MODEL:
        type_to_replace = transformers.models.llama.modeling_llama.LlamaRMSNorm  
    elif model_type == model_utils.QWEN_MODEL: 
        type_to_replace = transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm 
    elif model_type == model_utils.MISTRAL_MODEL :
        type_to_replace = transformers.models.mistral.modeling_mistral.MistralRMSNorm 
    else:
        type_to_replace = torch.nn.LayerNorm
    
    model_utils.replace_modules(
        model,
        type_to_replace,
        lambda _: model_utils.RMSN(model.config.hidden_size),
        replace_layers=False,
    )
    

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')

def get_layer_name(model, target_layer):
    """
    Trouve le nom d'une couche spécifique dans un modèle.

    Args:
        model (torch.nn.Module): Le modèle contenant la couche.
        target_layer (torch.nn.Module): La couche dont on veut récupérer le nom.

    Returns:
        str: Le nom de la couche si trouvée, sinon None.
    """
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if child is target_layer:
                return child_name  # Retourne le nom exact de la couche dans son parent
    return None  # Si non trouvé

def expand_embedding(model, L, dim):
    name = get_layer_name(model, L)
    """
    Étend la matrice de poids W à la nouvelle forme new_shape en remplissant les nouvelles dimensions avec des zéros.
    """
    W = L.weight.data
    old_shape = W.shape
    new_dim =  old_shape[1] + dim
    expanded_W = torch.zeros((old_shape[0],new_dim), device=W.device, dtype=W.dtype)
    expanded_W[:old_shape[0], :old_shape[1]] = W  # Copie des anciennes valeurs

    new_embedding = torch.nn.Embedding(old_shape[0], new_dim, _weight=expanded_W)

    delattr(model, name)
    setattr(model, name, new_embedding)
    del L
    torch.cuda.empty_cache()
    
    return new_embedding

def expand_linear(model, L, in_dim, out_dim):
    name = get_layer_name(model, L)
    """
    Étend la matrice de poids W à la nouvelle forme new_shape en remplissant les nouvelles dimensions avec des zéros.
    """
    W = L.weight.data
    
    old_shape = W.shape
    new_in_dim =  old_shape[1] + in_dim
    new_out_dim =  old_shape[0] + out_dim
    expanded_W = torch.zeros((new_out_dim, new_in_dim), device=W.device, dtype=W.dtype)
    expanded_W[:old_shape[0], :old_shape[1]] = W  # Copie des anciennes valeurs

    new_linear = torch.nn.Linear(new_in_dim, new_out_dim, bias = L.bias is not None)
    new_linear.weight.data = expanded_W
    # if L.bias is not None:
    #     b = L.bias.data
    #     b_shape = b.shape[0]
    #     expanded_b = torch.zeros((b_shape+new_out_dim,), device=W.device, dtype=W.dtype)
    #     expanded_b[:b_shape] = b
    #     new_linear.bias.data = expanded_b

    delattr(model, name)
    setattr(model, name, new_linear)
    del L
    torch.cuda.empty_cache()

    return new_linear



def rotate_embeddings(model, Q: torch.Tensor, expand=0) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    # torch.nn.modules.sparse.Embedding
    for W in model_utils.get_embeddings(model, model_type):
        if expand:
            W = expand_embedding(model.model, W, expand)
            
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device=utils.DEV, dtype=dtype)
        

    
def rotate_attention_inputs(layer, Q, model_type, expand=0) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    if model_type == model_utils.PHI_MODEL:
        L = [layer.self.attn.qkv_proj]
    else:
        L = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
    for W in L:
        if expand:
            W = expand_linear(layer.self_attn, W, out_dim=0, in_dim=expand)
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device=utils.DEV, dtype=dtype)
        

def rotate_attention_output(layer, Q, model_type, expand=0) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.QWEN_MODEL or model_type == model_utils.PHI_MODEL or model_type == model_utils.MISTRAL_MODEL :
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    if expand:
        W = expand_linear(layer.self_attn, W, out_dim=expand, in_dim=0)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device=utils.DEV, dtype=dtype)
    
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device=utils.DEV, dtype=dtype)

def rotate_mlp_input(layer, Q, model_type, expand_in=0, expand_out=0):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.QWEN_MODEL or model_type == model_utils.MISTRAL_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.PHI_MODEL:
        mlp_inputs = [layer.mlp.gate_up_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        if expand_in or expand_out:
            W = expand_linear(layer.mlp, W, out_dim=expand_out, in_dim=expand_in)
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device=utils.DEV, dtype=dtype)
        
    
def rotate_mlp_output(layer, Q, model_type, expand_out=0, expand_in=0):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.QWEN_MODEL or model_type == model_utils.PHI_MODEL or model_type == model_utils.MISTRAL_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    if expand_in or expand_out:
        W = expand_linear(layer.mlp, W, out_dim=expand_out, in_dim=expand_in)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device=utils.DEV, dtype=dtype)
    
    apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device=utils.DEV, dtype=dtype)

def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation. 
    It reshapes X and applies Walsh-Hadamard transform to the last dimension. 
    Then, it will multiply the retult by another hadamard matrix.
    '''
    # from fast_hadamard_transform import hadamard_transform
    from hadamard_utils import get_had172
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1/math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input 
    return input.to(X.device).to(X.dtype).reshape(
        X.shape) 

def rotate_faster_down_proj(layer, model_type, hardK):
    # from fast_hadamard_transform import hadamard_transform
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.QWEN_MODEL or model_type == model_utils.MISTRAL_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Faster MLP is onlu supported for LLaMa models!')
    
    dtype = W.weight.data.dtype
    W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
    W.weight.data = W.weight.data.to(device=utils.DEV, dtype=dtype)


def rotate_head(model, Q: torch.Tensor, expand=0) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    if expand:
        W = expand_linear(model, W, out_dim=0, in_dim=expand)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device=utils.DEV, dtype=dtype)

def rotate_ov_proj(layer, model_type, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.QWEN_MODEL or model_type == model_utils.MISTRAL_MODEL:
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    if model_type == model_utils.QWEN_MODEL:
        apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)
    else:
        apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)


# @torch.inference_mode()
def rotate_model(model, args):
    Q = get_orthogonal_matrix(model.config.hidden_size,
                                                args.rotate_mode)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q)
    if not model_type == model_utils.QWEN_MODEL:
        rotate_head(model, Q)
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type)
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)

# Class to redirect a missing key to the 'other' key
class AbsorbingDict(dict):
    def __missing__(self, key):
        return self["other"]

def rotate_model_mix_compute(model, args):
    matrices = {}
    try:
        with open(args.expand, 'r') as file:
            config_mix = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError('Expand file not found')
    
    expand_attn = AbsorbingDict(config_mix['attention'])
    expand_mlp = AbsorbingDict(config_mix['mlp'])
    list_expand = list(set(list(expand_attn.values())))
    for k in list_expand:
        matrices[k] = get_orthogonal_matrix(model.config.hidden_size + k,
                                                args.rotate_mode)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    model_type = model_utils.model_type_extractor(model)
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    n_layers = len(layers)
    rotate_embeddings(model, matrices[expand_attn['0']], expand_attn['0'])
    rotate_head(model, matrices[expand_attn[str(n_layers - 1)]], expand_attn[str(n_layers - 1)])
    utils.cleanup_memory()
    
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        expand_curr = expand_attn[str(idx)]
        expand_curr_mlp = expand_mlp[str(idx)]
        idx_next = idx + 1 if idx + 1 <= n_layers - 1 else n_layers - 1
        expand_next = expand_attn[idx_next]
        rotate_attention_inputs(layers[idx], matrices[expand_curr], model_type, expand_curr)
        rotate_attention_output(layers[idx], matrices[expand_curr], model_type, expand_curr)
        rotate_mlp_input(layers[idx], matrices[expand_curr], model_type, expand_in=expand_curr, expand_out=expand_curr_mlp)
        rotate_mlp_output(layers[idx], matrices[expand_next], model_type, expand_out=expand_next, expand_in=expand_curr_mlp)
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)
        # For now we suppose we only increase dimension
        if expand_next > expand_curr:
            layer.hadamard_block.set(matrices[expand_curr], matrices[expand_next], expand_next - expand_curr, None)
            layer.hadamard_block.activate()

@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]

def register_online_rotation(module, Q:torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [-1, head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_quantizer.configure(bits=self.k_bits, groupsize=1, #we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                                   sym=self.k_sym, clip_ratio=self.k_clip_ratio)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape
        

        if self.k_groupsize == -1: #token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else: #head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)
        
        self.k_quantizer.free()
            
        return q, k



def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)

# Qwen model use the same module to compute embed tokens and lm_head, we need to split them to fuse layer_norm
def split_embed_head(model):
    lm_head = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    in_features = lm_head.in_features
    out_features = lm_head.out_features
    new_lm_head = torch.nn.Linear(in_features, out_features, bias=False)
    new_lm_head.weight.data = model.lm_head.weight.data
    setattr(model, 'lm_head', new_lm_head)
    return

