#!/bin/bash
#SBATCH --time=7-0:00:00
#SBATCH --partition=gpu40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --job-name GBS_Llama-2-7b-hf_4bits
#SBATCH --output fai/GS_Llama-2-7b-hf_4bits.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
export MODEL='meta-llama/Llama-2-7b-hf'
export BIT=4
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
--optimized_rotation_path "rot/"${MODEL}"/"${BIT}"bits/R.bin" \
--gbs
# --load_qmodel_path /data1/is156025/lm270675/meta-labo/EAH-ViT/LLM/QuaRot/save/GPTQ/meta-llama/Meta-Llama-3-8B_GPTQ_4bit.pth 

