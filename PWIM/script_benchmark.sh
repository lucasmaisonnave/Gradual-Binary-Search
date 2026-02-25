export CUDA_VISIBLE_DEVICES=3

echo "------------ LLaMA2 7B ---------------"
export model_name="meta-llama/Llama-2-7b-hf"

python benchmark.py --model_path ${model_name} --act_quant 'None' --a_bits 16 --w_bits 16

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_tensor' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_tensor' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_tensor' --a_bits 8 --w_bits 8

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_token' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_token' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_token' --a_bits 8 --w_bits 8

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_tensor' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_tensor' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_tensor' --a_bits 6 --w_bits 6

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_token' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_token' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_token' --a_bits 6 --w_bits 6

echo "------------ LLaMA2 13B---------------"
export model_name="meta-llama/Llama-2-13b-hf"
python benchmark.py --model_path ${model_name} --act_quant 'None' --a_bits 16 --w_bits 16

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_tensor' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_tensor' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_tensor' --a_bits 8 --w_bits 8

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_token' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_token' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_token' --a_bits 8 --w_bits 8

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_tensor' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_tensor' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_tensor' --a_bits 6 --w_bits 6

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_token' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_token' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_token' --a_bits 6 --w_bits 6

echo "------------ Mistral7B ---------------"
export model_name="mistralai/Mistral-7B-v0.1"
python benchmark.py --model_path ${model_name} --act_quant 'None' --a_bits 16 --w_bits 16

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_tensor' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_tensor' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_tensor' --a_bits 8 --w_bits 8

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_token' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_token' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_token' --a_bits 8 --w_bits 8

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_tensor' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_tensor' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_tensor' --a_bits 6 --w_bits 6

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_token' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_token' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_token' --a_bits 6 --w_bits 6

echo "------------ LLaMA3-8B ---------------"
export model_name="meta-llama/Meta-Llama-3-8B"
python benchmark.py --model_path ${model_name} --act_quant 'None' --a_bits 16 --w_bits 16

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_tensor' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_tensor' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_tensor' --a_bits 8 --w_bits 8

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_token' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_token' --a_bits 8 --w_bits 8
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_token' --a_bits 8 --w_bits 8

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_tensor' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_tensor' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_tensor' --a_bits 6 --w_bits 6

python benchmark.py --model_path ${model_name} --quantize --act_quant 'per_token' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --smooth --act_quant 'per_token' --a_bits 6 --w_bits 6
python benchmark.py --model_path ${model_name} --quantize --mix --act_quant 'per_token' --a_bits 6 --w_bits 6