export CUDA_VISIBLE_DEVICES=3


echo "------------ LLaMA3 8B ---------------"
export model_name="meta-llama/Llama-2-13b-hf"
python ppl_eval.py --model_path ${model_name} --quantize --act_quant 'per_tensor' --a_bits 8 --w_bits 8 --smooth 
