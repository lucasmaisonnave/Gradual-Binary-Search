export CUDA_VISIBLE_DEVICES=3

export CACHE_DIR='/data1/is156025/lm270675/.cache/huggingface/hub'
export EXPAND_DIR='./save/mix_compute/'

export MODEL='meta-llama/Llama-2-7b-hf'

export BIT=3


python ./fake_quant/main.py \
--model ${MODEL} \
--a_bits ${BIT} \
--v_bits ${BIT} \
--k_bits ${BIT} \
--w_bits ${BIT} \
--w_clip \
--bsz 6 \
--rotate \
--k_groupsize 128 \
--cache_dir ${CACHE_DIR} \
--start_bit 16 \
--expand ${EXPAND_DIR}1348.json \
--lm_eval \
--lm_eval_batch_size 32 \
--grid_search 

