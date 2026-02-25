export CUDA_VISIBLE_DEVICES=2

export CACHE_DIR='/data1/is156025/lm270675/.cache/huggingface/hub'
export EXPAND_DIR='./save/mix_compute/'

export MODEL='meta-llama/Llama-2-7b-hf'
export EAH='fp32'
export BIT=4


python ./fake_quant/eam.py \
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
--expand ${EXPAND_DIR}0.json \
--w_rtn \
--eah ${EAH} \
--alpha 0.05 \
--thresh_entropy 0.1