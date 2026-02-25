#!/bin/bash
#SBATCH --time=7-0:00:00
#SBATCH --partition=gpu40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --job-name GS_Mistral-7B-Instruct-v0.3_4bits
#SBATCH --output fai/GS_Mistral-7B-Instruct-v0.3_4bits.out
hostname 
echo $CUDA_VISIBLE_DEVICES
singularity run docker://nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

export BIT=4

export CACHE_DIR='/home/users/lmaisonnave/.cache/huggingface/hub'
export MODEL='mistralai/Mistral-7B-Instruct-v0.3'

python ./fake_quant/RotateGridSearch.py \
--model ${MODEL} \
--rotate \
--a_bits ${BIT} \
--v_bits ${BIT} \
--k_bits ${BIT} \
--w_bits ${BIT} \
--w_clip \
--bsz 6 \
--k_groupsize 128 \
--start_bit 16 \
--cache_dir ${CACHE_DIR} \
--load_qmodel_path ./save/GPTQ/${MODEL}_GPTQ_${BIT}bit.pth 