# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
export CUDA_VISIBLE_DEVICES=3 #$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader|nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs) 
python ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--rotate \
--optimized_rotation_path "rot/"$1"/"$2"bits/R.bin" \
--gbs
# --load_qmodel_path /data1/is156025/lm270675/meta-labo/EAH-ViT/LLM/QuaRot/save/GPTQ/meta-llama/Meta-Llama-3-8B_GPTQ_4bit.pth 

