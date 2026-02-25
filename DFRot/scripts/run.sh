export CUDA_VISIBLE_DEVICES=5


export MODEL='mistralai/Mistral-7B-Instruct-v0.3'
export BIT=3
export ALPHA=100


python3 main.py \
--model ${MODEL}  \
--rotate \
--w_bits ${BIT} \
--a_bits ${BIT} \
--k_bits ${BIT} \
--v_bits ${BIT} \
--w_clip \
--v_groupsize 128 \
--k_groupsize 128 \
--a_asym \
--k_asym \
--v_asym \
--rotate_mode orthogonal_procrustes \
--indices_path rms_norm_feature_hadamard_alter/${ALPHA}/${MODEL}-${BIT}.npy \
--fp32_had \
--seed 0 \
--lm_eval \
--gbs \
--lm_eval_batch_size 64 