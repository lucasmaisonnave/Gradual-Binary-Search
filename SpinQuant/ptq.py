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
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from GradualBinarySearch import GBS

log: Logger = utils.get_logger("spinquant")

cache_dir =  "/data1/is156025/lm270675/.cache/huggingface/hub"
def train() -> None:
    # dist.init_process_group(backend="nccl")
    os.environ["WANDB_DISABLED"] = "true"
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = 0 #utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    # torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token, cache_dir=cache_dir
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        # config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
        cache_dir=cache_dir
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
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

    print("Perform GBS")
    if ptq_args.gbs:
        import json
        gbs = GBS(ptq_args, model_args, model, tokenizer, b=[ptq_args.a_bits])
        config_gbs = gbs.search()
    else:
        log.info("Complete tokenizer loading...")
        model.config.use_cache = False

        testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )

        dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
        log.info("wiki2 ppl is: {}".format(dataset_ppl))
        # dist.barrier()
        folder_csv = "./results/"
        if not os.path.exists(folder_csv):
            os.makedirs(folder_csv)
        nouvelle_ligne = [model_args.input_model, ptq_args.w_bits, ptq_args.gbs, dataset_ppl]
            # Ajout de la ligne au fichier CSV
        import csv
        with open(folder_csv + "results_ppl.csv", mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(nouvelle_ligne)


if __name__ == "__main__":
    train()
