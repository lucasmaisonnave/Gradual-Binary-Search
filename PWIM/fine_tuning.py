from tqdm.notebook import tqdm

from datasets import load_dataset
import torch
import os
from smoothquant.smoothquant.fake_quant import quantize_activation_per_tensor_absmax
from torch.autograd.function import Function
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from trl import SFTTrainer
from transformers import default_data_collator, Trainer, TrainingArguments

import argparse
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
login(token=os.environ.get("HF_TOKEN"))
cache_dir = '/data1/is156025/lm270675/.cache/huggingface/hub'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name", type=str, default="mistralai/Mistral-7B-v0.1", help="model name"
)
parser.add_argument(
    "--dataset", type=str, default="wiki", help="model name"
)

parser.add_argument("--alpha", type=float, default=0.1)
args = parser.parse_args()

# referencing https://github.com/meta-llama/llama-recipes/blob/main/recipes/finetuning/huggingface_trainer/peft_finetuning.ipynb
eval_prompt = """
Who is Leonardo Da Vinci ?
"""
n_layers = {'mistralai/Mistral-7B-v0.1': 32,
            'meta-llama/Llama-2-13b-hf': 40,
            'mistralai/Mistral-7B-v0.1' : 32, 
            'meta-llama/Llama-2-7b-hf': 32, 
            'meta-llama/Meta-Llama-3-8B' : 32,
            'facebook/opt-13b': 40,
            'bigscience/bloom-7b1' : 32} 
MAX_SEQ_LEN = 1024
model_name = args.model_name
embbed = []
# Quantization error importance
ALPHA = args.alpha
BF16 = True

# healing
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

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

def kurtosis(X):
    # Fist we compute the root mean square over tokens
    rms = X.square().mean(dim=1).sqrt()
    return rms.pow(4).mean() # / (rms.square().mean()).square()

class q_k(Function):
    """
        This is the quantization module.
        The input and output should be all on the interval [0, 1].
        bit is only defined on positive integer values.
    """
    @staticmethod
    def forward(ctx, input, bit):
        
        t_shape = input.shape
        t = input.view(-1, t_shape[-1])
        scales = t.abs().max()
        q_max = 2 ** (bit - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        t = t.div(scales).round().mul(scales)
        return t
        
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None



def QER(X, b = 8):
    q_x = q_k.apply(X, b)
    return torch.nn.MSELoss()(X, q_x)

def get_hook(name):
        def hook(model, input, output):
            global embbed
            if len(embbed) == n_layers[model_name]:
                del embbed
                embbed = []
            embbed.append(QER(input[0][0]))
        return hook

model.train()

# def create_peft_config(model):
#     peft_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         inference_mode=False,
#         r=32,
#         lora_alpha=16,
#         lora_dropout=0.05,
#         target_modules = ["input_layernorm", "post_attention_layernorm"]
#     )

#     model = get_peft_model(model, peft_config)
#     model.print_trainable_parameters()
#     return model, peft_config

# # create peft config
# model, lora_config = create_peft_config(model)

# freeze layers
layers_not_freezed = ["input_layernorm", "post_attention_layernorm"]
for name, param in model.named_parameters():
    name_ = name.split('.')[-2]
    if name_ not in layers_not_freezed:
        param.requires_grad = False
print("Hooking...")

for name, layer in model.named_modules():
    if len(layer._modules) == 0 and 'emb' not in name and '.layer' in name:
        name_ = name.split('.')[-1]
        if name_ == 'input_layernorm':
            layer.register_forward_hook(get_hook(name_))

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="adamw_torch_fused",
    logging_steps=25,
    save_strategy="no",
    learning_rate=1e-6,
    weight_decay=0.001,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    # warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="linear",
    warmup_ratio=0.1
)

# Create Trainer instance


# subclass trainer
class CustomTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        alpha = ALPHA
        outputs = model(**inputs)
        loss1 = outputs['loss']
        loss2 = embbed[0]
        for x in embbed[1:]:
            loss2 += x
        loss = loss1 + alpha * loss2
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    train_dataset=get_dataset(args.dataset),
    max_seq_length=None,
    tokenizer=tokenizer,
    dataset_text_field="text",
    args=training_args,
    packing=False,
)

# output_dir = "tmp/"

# config = {
#     'lora_config': lora_config,
#     'learning_rate': 1e-6,
#     'num_train_epochs': 1,
#     'per_device_train_batch_size': 1,
#     'gradient_checkpointing': False,
# }

# training_args = TrainingArguments(
#     output_dir=output_dir,
#     overwrite_output_dir=True,
#     # logging strategies
#     logging_strategy="steps",
#     logging_steps=10,
#     save_strategy="no",
#     optim="adamw_torch_fused",
#     **{k:v for k,v in config.items() if k != 'lora_config'}
# )

# # Create Trainer instance
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=get_dataset(args.dataset),
#     data_collator=default_data_collator,
#     callbacks=[],
# )

# Start training
trainer.train()

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

model_dir = os.path.join(os.path.join(os.path.join("./models/", model_name), args.dataset), "fine_tune_" + str(ALPHA) + "_bf16" if BF16 else "_fp16")
model.save_pretrained(model_dir, from_pt=True)