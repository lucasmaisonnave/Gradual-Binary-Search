# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
export CUDA_VISIBLE_DEVICES=5 #$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader|nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs) 

export MODEL='mistralai/Mistral-7B-v0.1'

export BIT=3

python ptq.py \
--input_model ${MODEL} \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits ${BIT} \
--a_bits ${BIT} \
--k_bits ${BIT} \
--v_bits ${BIT} \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--rotate \
--optimized_rotation_path "rot/"${MODEL}"/"${BIT}"bits/R.bin" 