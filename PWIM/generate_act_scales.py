import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

from smoothquant.smoothquant.calibration import get_act_scales
from load import load_short_model


def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="mistralai/Mistral-7B-v0.1", help="model name"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/data1/is156025/lm270675/meta-labo/EAH-ViT/LLM/smoothquant/act_scales",
        help="where to save the act scales",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/data1/is156025/lm270675/meta-labo/EAH-ViT/LLM/smoothquant/dataset/val.jsonl.zst",
        help="location of the calibration dataset, we use the validation set of the Pile dataset",
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--N", type=int, default=10)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    N = args.N
    tokenizer, model = load_short_model(args.model_name, N)

    if not os.path.exists(args.dataset_path):
        print(f"Cannot find the dataset at {args.dataset_path}")
        print("Please download the Pile dataset and put the validation set at the path")
        print(
            "You can download the validation dataset of the Pile at https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst"
        )
        raise FileNotFoundError

    act_scales = get_act_scales(
        model, tokenizer, args.dataset_path, args.num_samples, args.seq_len
    )

    dir = os.path.join(os.path.join(args.output_path, args.model_name), 'N_{}'.format(N))
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    torch.save(act_scales, dir)


if __name__ == "__main__":
    main()
