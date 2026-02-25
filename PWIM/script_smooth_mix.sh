export CUDA_VISIBLE_DEVICES=6

echo "------------ LLaMA2 7B ---------------"
export model_name="meta-llama/Llama-2-7b-hf"
python benchmark.py --model_path ${model_name} --quantize --smooth --mix --act_quant 'per_token' --a_bits 6 --w_bits 6 --result_path './results/evaluation_results_smooth_mix.json'

echo "------------ LLaMA2 13B ---------------"
export model_name="meta-llama/Llama-2-13b-hf"
python benchmark.py --model_path ${model_name} --quantize --smooth --mix --act_quant 'per_token' --a_bits 6 --w_bits 6 --result_path './results/evaluation_results_smooth_mix.json'

echo "------------ Mistral 7B ---------------"
export model_name="mistralai/Mistral-7B-v0.1"
python benchmark.py --model_path ${model_name} --quantize --smooth --mix --act_quant 'per_token' --a_bits 6 --w_bits 6 --result_path './results/evaluation_results_smooth_mix.json'

echo "------------ LLaMA3 8B ---------------"
export model_name="meta-llama/Meta-Llama-3-8B"
python benchmark.py --model_path ${model_name} --quantize --smooth --mix --act_quant 'per_token' --a_bits 6 --w_bits 6 --result_path './results/evaluation_results_smooth_mix.json'