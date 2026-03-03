# Project Memory: GBS-QuaRot

## What the project is
Implementation of Gradual Binary Search (GBS) on top of QuaRot (rotation-based quantization for LLMs).
GBS finds optimal per-layer, per-projection clipping ratios via a dichotomic search over a small calibration subset.
Also implements dimension expansion: zero-pad weight matrices so non-power-of-2 dimensions can use Hadamard (Paley) transforms.
Useful for architectures like Qwen that have incompatible dimensions for standard QuaRot.

## Key files
- fake_quant/RotateGridSearch.py — GBS algorithm (GridSearch class), main entry point
- fake_quant/main.py — load_rotate_quantize_model(), update_quantizer()
- fake_quant/rotation_utils.py — model rotation, expand_embedding/linear, rotate_model_mix_compute(), QKRotationWrapper
- fake_quant/hadamard_utils.py — Hadamard_block, get_hadK(), hadamard_paley(), Paley construction
- fake_quant/quant_utils.py — ActQuantizer, ActQuantWrapper, WeightQuantizer, add_actquant()
- fake_quant/utils.py — argument parser, DEV, cleanup_memory()
- fake_quant/model_utils.py — model type detection, get_transformer_layers()
- script_grid_search.sh — main launch script
- save/mix_compute/0.json — default expand config (0 = no expansion)

## Known issues / bugs to discuss with user
1. Hadamard_block.set() signature mismatch: hadamard_utils.py def set(self, H, expand, hidden_dim) but rotation_utils.py:405 calls it with 4 args: set(matrices[expand_curr], matrices[expand_next], expand_next-expand_curr, None)
2. Duplicate key 'meta-llama/Llama-2-7b-hf' in n_layers dict (lines 21 and 27 of RotateGridSearch.py)
3. GridSearch.f() line 205 uses global `args` instead of self.args
4. French/English mixed comments throughout
5. Hardcoded paths: cache_dir in RotateGridSearch.py line 20, './grid_search/rebutals/' in GridSearch.__init__
6. The --inv flag reversal inside loop body may be buggy with --resume_gs

## GBS binary search mechanics
- For each layer i, for each projection p: find best clipping ratio in [0,1]
- Start at m=0.5, evaluate PPL
- Odd iterations: test (a+m)/2 (left quarter); Even iterations: test (m+b)/2 (right quarter)
- Alternating pattern explores sub-regions gradually
- Best result is kept; if start_bit > _bit always accept (warm-start from high bits)
- After all projections for a layer: update list_max_bit = best_max_bit (greedy layer-by-layer)

## Dimension expansion config format
JSON file with 'attention' and 'mlp' keys, each mapping layer index (str) to expansion dim (int), plus 'other' as default.
Example: {"attention": {"0": 128, "other": 0}, "mlp": {"0": 64, "other": 0}}
0.json = no expansion (standard QuaRot)

## Supported models (in n_layers dict)
LLaMa-2 (7B, 13B), LLaMa-3 (1B, 8B), Mistral-7B, OPT-13B, BLOOM-7B, DeepSeek-R1, Qwen2.5 (0.5B,1.5B,7B), Phi-4-mini

## Projection order in GBS
['q_proj', 'k_proj', 'v_proj', 'qk_rotation', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
qk_rotation = K-cache quantizer clipping ratio
