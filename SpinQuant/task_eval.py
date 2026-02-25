# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger
import os
import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model, update_quantizer
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
import json
import lm_eval
from lm_eval.models.huggingface import HFLM


log: Logger = utils.get_logger("spinquant")

cache_dir =  "/data1/is156025/lm270675/.cache/huggingface/hub"
def train() -> None:
    # dist.init_process_group(backend="nccl")
    os.environ["WANDB_DISABLED"] = "true"
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = 0 #utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    # torch.distributed.barrier()
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        # config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
        cache_dir=cache_dir
    )
    model.cuda()

    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token
    )

    folder = './grid_search/' + model_args.input_model + '/'
    filename = 'rotate' if ptq_args.rotate else '' 
    filename += '_W_GPTQ' # '_W_RTN' if ptq_args.w_rtn else 
    filename += '_W' + str(ptq_args.w_bits) + 'A' + str(ptq_args.a_bits) + 'KV' + str(ptq_args.v_bits) + '.json'

    if ptq_args.gbs:
        with open(folder + 'best_config_' + filename, 'r') as file:
            config = json.load(file)[0]['config']
            update_quantizer(ptq_args, model, config)
    model = model.cuda()
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=ptq_args.lm_eval_batch_size)
    results = lm_eval.simple_evaluate(hflm, tasks=ptq_args.tasks, batch_size=ptq_args.lm_eval_batch_size)['results']

    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    L = [model_args.input_model,ptq_args.w_bits,ptq_args.gbs]
    for v in metric_vals:
        L.append(metric_vals[v])
    import csv
    fichier_csv = "./results/results_tasks.csv"
    with open(fichier_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(L)


if __name__ == "__main__":
    train()
