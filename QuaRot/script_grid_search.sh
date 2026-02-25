export CUDA_VISIBLE_DEVICES=3
export BIT=3

export CACHE_DIR='/data1/is156025/lm270675/.cache/huggingface/hub'
export EXPAND_DIR='./save/mix_compute/'

export MODEL='meta-llama/Llama-2-7b-hf'

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
--alpha 0.1 \
--expand ${EXPAND_DIR}0.json \
--max_iterations 8
# --load_qmodel_path "save/GPTQ/"${MODEL}"/"${BIT}"bits.pth"


# --load_qmodel_path ./save/GPTQ/${MODEL}_GPTQ_${BIT}bit.pth \
# --save_qmodel_path ./save/GPTQ/${MODEL}_GPTQ_${BIT}bit_expanded.pth \
# --resume_gs
# --prefix
# --inv
# --w_rtn \
# 472
# n_layers = {'meta-llama/Llama-2-7b-hf': 32,
# 'meta-llama/Llama-3.2-1B' : 16,
# 'meta-llama/Llama-3.1-8B' : 32,
# 'meta-llama/Llama-2-13b-hf': 40, 
# 'mistralai/Mistral-7B-v0.1' : 32,
# 'mistralai/Mistral-7B-Instruct-v0.3': 32,
# 'meta-llama/Llama-2-7b-hf': 32, 
# 'meta-llama/Meta-Llama-3-8B' : 32,
# 'facebook/opt-13b': 40,
# 'bigscience/bloom-7b1' : 32,
# 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B':32
# 'Qwen/Qwen2.5-7B-Instruct' : 28'
# 'Qwen/Qwen2.5-1.5B-Instruct' : 28'
# 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' : 28'
# }