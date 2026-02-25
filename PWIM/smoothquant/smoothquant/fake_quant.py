import torch
from torch import nn
from functools import partial
from math import sqrt
from .fp_linear import FPMinMaxQuantLinear

# Treshold to detect spikes
TRSH = 100

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8, spike = None, GS_param=None):
    if spike == None:
        t_shape = t.shape
        t.view(-1, t_shape[-1])
        scales = t.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (n_bits - 1) - 1

        scales.clamp_(min=1e-5).div_(q_max)
        t.div_(scales).round_().mul_(scales)
    # Here we use fp8 format
    elif spike == 'FP8':
        e5m2_type = torch.float8_e5m2

        t = t.to(e5m2_type)
        t = t.to(torch.float16)
    # Here we use the maximum value set for greed serach
    elif spike == 'grid_search':
        t_shape = t.shape
        t.view(-1, t_shape[-1])
        scales = t.abs().max(dim=-1, keepdim=True)[0]

        # We just clamp so we dont change small values
        l = GS_param['layer']
        n = GS_param['max_bit'][l][GS_param['proj']]["max"]
        bit = GS_param['max_bit'][l][GS_param['proj']]['bit']
        M = n * scales.max()
        scales = torch.clamp(scales, max = M)
        q_max = 2 ** (bit - 1) - 1

        t = torch.clamp(t, min = -M, max = M)
        scales.clamp_(min=1e-5).div_(q_max)
        t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, spike=None, GS_param=None):
    if spike is None:
        t_shape = t.shape
        t.view(-1, t_shape[-1])
        scales = t.abs().max()
        q_max = 2 ** (n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        t.div_(scales).round_().mul_(scales)
    # Here we use fp8 format
    elif spike == 'FP8':
        e5m2_type = torch.float8_e5m2

        t = t.to(e5m2_type)
        t = t.to(torch.float16)
    # Here we use the maximum value set for greed serach
    elif spike == 'grid_search':
        t_shape = t.shape
        t.view(-1, t_shape[-1])
        scales = t.abs().max()

        # We just clamp so we dont change small values
        n = GS_param['n'][GS_param['layer']]
        bit = GS_param['bit']
        M = n * scales.max()
        scales = torch.clamp(scales, max = M)
        q_max = 2 ** (bit - 1) - 1

        t = torch.clamp(t, min = -M, max = M)
        scales.clamp_(min=1e-5).div_(q_max)
        t.div_(scales).round_().mul_(scales)

    return t

# RepQ-ViT implementation : https://github.com/zkkli/RepQ-ViT/blob/main/classification/quant/quantizer.py
@torch.no_grad()
def log2_quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    tmax = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1

    sign = (t > 0) * 2 - 1
    x_int = torch.round( (t.abs() / tmax * q_max).log2())
    mask = x_int >= 2 ** n_bits
    x_quant = torch.clamp(x_int, 0, n_bits - 1)
    # odd_mask = (x_quant%2) * (sqrt(2)-1) + 1
    # x_float_q = 2**(-1 * torch.ceil(x_quant/2)) * odd_mask * scales
    x_float_q = 2**x_quant / q_max * tmax
    x_float_q[mask] = 0
        
    return x_float_q * sign


class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=True,
        sep_f = False,
        log = False,
        a_bits = 8,
        spike = None,
        GS_param=None

    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.sep_free = sep_f

        self.register_buffer(
            "weight",
            torch.zeros(
                (self.out_features,
                self.in_features),
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=a_bits, spike = spike, GS_param=GS_param)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            if log:
                self.act_quant = partial(log2_quantize_activation_per_tensor_absmax, n_bits=a_bits)
            else:
                self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=a_bits, spike = spike, GS_param=GS_param)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        token = 0

         # set SEP to 0 id we don't want to quantize it (first token)
        if self.sep_free:
            sep_f = torch.clone(x[:,token])
            x[:,token] = x[:,token] * 0
        q_x = self.act_quant(x)
        # set sep to initial value
        if self.sep_free:
            q_x[:,token] = sep_f
        
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        # set sep to 0
        if self.sep_free:
            sep_f = torch.clone(y[:,token])
            y[:,token] = y[:,token] * 0   
        q_y = self.output_quant(y)
        # set sep to initial value
        if self.sep_free:
            q_y[:,token] = sep_f
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False, sep_f=False, log=False, a_bits=8, w_bits=8, spike=None, GS_param=None
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            sep_f=sep_f,
            log=log,
            a_bits=a_bits,
            spike=spike,
            GS_param=GS_param
        )
        if weight_quant == "per_channel":
            # import matplotlib.pyplot as plt
            # plt.hist(module.weight.flatten().detach().cpu().numpy(), bins = 512, range = (-0.1, 0.1))
            # plt.savefig('./data/histo.png')
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=w_bits if spike != 'grid_search' else GS_param['max_bit'][GS_param['layer']][GS_param['proj']]['bit']
            )  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=w_bits if spike != 'grid_search' else GS_param['max_bit'][GS_param['layer']][GS_param['proj']]['bit']
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_opt(
    model, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=True, sep_f = False, log = False, a_bits = 8, w_bits = 8
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, log=log, a_bits=a_bits, w_bits=w_bits
            )
            m.fc2 = W8A8Linear.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, log=log, a_bits=a_bits, w_bits=w_bits
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                sep_f=sep_f,
                log=log, 
                a_bits=a_bits, 
                w_bits=w_bits
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                sep_f=sep_f,
                log=log, 
                a_bits=a_bits, 
                w_bits=w_bits
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                sep_f=sep_f,
                log=log, 
                a_bits=a_bits, 
                w_bits=w_bits
            )
            m.out_proj = W8A8Linear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, log=log, a_bits=a_bits, w_bits=w_bits
            )
    return model


def quantize_llama_like(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, sep_f=False, fp4=False, skip_layers = None, a_bits=8, w_bits=8, spike=None, GS_param=None
):

    GS = spike == 'grid_search'
    id = GS_param['layer'] if GS_param is not None else -1
    proj = GS_param['proj'].copy() if GS_param is not None else None
    inv = GS_param['inv'] if GS_param is not None else None
    new_config = None
    for name, m in model.model.named_modules():
        if isinstance(m, torch.nn.Linear):
            l = int(name.split('.')[1])
            proj_name = name.split('.')[-1]
            bloc = name.split('.')[-2]
            quant_type = spike
            # Here we test if we use grid search and if the layer targeted is the right one (all layers before id or the opposite if inv is True)
            if GS:
                if (l > id and not inv) or (l < id and inv):
                    quant_type = None

                new_config = GS_param.copy()
                new_config['layer'] = l
                new_config['proj'] = proj_name

                if proj_name not in proj and l == id:
                    quant_type = None
            p = getattr(model.model.layers[l], bloc)
            # Quantize if layer is not in list layers_qdp_down for down proj and not in layers_qdp_out for o_proj
            # if skip_layers is not None: # Condition de sécurité dans le cas ou skip_layers n'est pas renseigné et que l'on n'a pas de dictionnaire
            if not ((proj_name == 'down_proj' and skip_layers['down_proj'][l]) or (proj_name == 'o_proj' and skip_layers['o_proj'][l]) or (skip_layers['layers'][l])):
                setattr(p, proj_name, W8A8Linear.from_float(
                    m, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, a_bits=a_bits, w_bits=w_bits, spike=quant_type, GS_param=new_config
                ))
            elif fp4:
                setattr(p, proj_name, FPMinMaxQuantLinear(m, "quant_forward", w_bit=4, a_bit=4, w_exponent_bit=2, a_exponent_bit=2))
            else:
                setattr(p, proj_name, W8A8Linear.from_float(
                    m, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, a_bits=a_bits, w_bits=w_bits, spike=None, GS_param=None
                ))
               
    return model


def quantize_mixtral(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, sep_f=False, log=False, a_bits=8, w_bits=8
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            m.w1 = W8A8Linear.from_float(
                m.w1, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, log=log, a_bits=a_bits, w_bits=w_bits
            )
            m.w2 = W8A8Linear.from_float(
                m.w2, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, log=log, a_bits=a_bits, w_bits=w_bits
            )
            m.w3 = W8A8Linear.from_float(
                m.w3, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, log=log, a_bits=a_bits, w_bits=w_bits
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                sep_f=sep_f, 
                log=log, 
                a_bits=a_bits, 
                w_bits=w_bits
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                sep_f=sep_f, 
                log=log, 
                a_bits=a_bits, 
                w_bits=w_bits
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                sep_f=sep_f, 
                log=log, 
                a_bits=a_bits, 
                w_bits=w_bits
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, log=log, a_bits=a_bits, w_bits=w_bits
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = W8A8Linear.from_float(
                m.gate, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, log=log, a_bits=a_bits, w_bits=w_bits
            )
    return model


def quantize_falcon(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True, sep_f=False, log=False, a_bits=8, w_bits=8
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = W8A8Linear.from_float(
                m.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, log=log, a_bits=a_bits, w_bits=w_bits
            )
            m.dense_4h_to_h = W8A8Linear.from_float(
                m.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, log=log, a_bits=a_bits, w_bits=w_bits
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = W8A8Linear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                sep_f=sep_f, 
                log=log, 
                a_bits=a_bits, 
                w_bits=w_bits
            )
            m.dense = W8A8Linear.from_float(
                m.dense, weight_quant=weight_quant, act_quant=act_quant, sep_f=sep_f, log=log, a_bits=a_bits, w_bits=w_bits
            )
    return model


def quantize_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, sep_f = False, fp4 = False, skip_layers = None, a_bits=8, w_bits=8, spike=None, GS_param=None
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    if isinstance(model, OPTPreTrainedModel):
        return quantize_opt(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            sep_f=sep_f,
            a_bits=a_bits, 
            w_bits=w_bits
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            sep_f=sep_f,
            fp4=fp4,
            skip_layers=skip_layers, 
            a_bits=a_bits, 
            w_bits=w_bits,
            spike=spike,
            GS_param=GS_param
            
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            sep_f=sep_f, 
            a_bits=a_bits, 
            w_bits=w_bits
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            sep_f=sep_f,
            a_bits=a_bits, 
            w_bits=w_bits
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
