# GBS: Gradual Binary Search for Optimal Quantization Clipping Ratios

GBS is an algorithm built on top of [QuaRot](https://arxiv.org/abs/2404.00456) that automatically finds the best per-layer, per-projection **clipping ratios** for quantized LLMs. It also introduces a **dimension expansion** feature that enables QuaRot to work on architectures with non-power-of-2 hidden dimensions (e.g. Qwen).

---

## Overview

### QuaRot background

[QuaRot](https://arxiv.org/abs/2404.00456) quantizes LLMs end-to-end (weights, activations, and KV cache) by applying rotation matrices to the hidden states. These rotations remove activation outliers without changing the model output, which makes aggressive quantization (4-bit or lower) much more accurate.

### What GBS adds

Quantization accuracy is sensitive to the **clipping ratio** — the fraction of the activation range actually used for quantization. A ratio of 1.0 uses the full range; smaller values clip outliers more aggressively. The optimal ratio varies per layer and per projection type (`q_proj`, `k_proj`, ..., `down_proj`).

GBS finds these ratios automatically using a **ternary search** (dichotomic algorithm):

1. For each transformer layer and each projection, search for the clipping ratio in `[0, 1]` that minimizes perplexity on a small calibration subset (controlled by `--alpha`).
2. At each iteration, evaluate both the left midpoint `(a + m) / 2` and the right midpoint `(m + b) / 2` of the current interval, and keep the half with lower perplexity. This halves the search interval at every step.
3. Results are accumulated greedily: the best clipping ratio found for each projection is kept fixed when optimizing the next one.
4. The full per-layer, per-projection configuration is saved to disk as a JSON file and can be reused directly at inference time.

### Dimension expansion

Some architectures (e.g. Qwen2.5) have hidden dimensions that are not a multiple of any dimension that admits a Hadamard matrix. GBS introduces a **dimension expansion** mechanism:

- Weight matrices are zero-padded to a larger dimension that supports a Hadamard transform.
- The Hadamard matrix for the expanded dimension is constructed via the **Paley algorithm** (when `n-1` is prime and `n ≡ 0 mod 4`) or via a set of pre-computed matrices (for factors 12, 20, 28, 36, 40, 52, 60, 108, 140, 156, 172).
- The expansion is configured per-layer via a JSON file, so different layers can use different expansion amounts or none at all.

---

## Supported models

| Model | Layers |
|---|---|
| meta-llama/Llama-2-7b-hf | 32 |
| meta-llama/Llama-2-13b-hf | 40 |
| meta-llama/Llama-3.2-1B | 16 |
| meta-llama/Llama-3.1-8B | 32 |
| meta-llama/Meta-Llama-3-8B | 32 |
| mistralai/Mistral-7B-v0.1 | 32 |
| mistralai/Mistral-7B-Instruct-v0.3 | 32 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 32 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 28 |
| Qwen/Qwen2.5-0.5B-Instruct | 24 |
| Qwen/Qwen2.5-1.5B-Instruct | 28 |
| Qwen/Qwen2.5-7B-Instruct | 28 |

---

## Installation

```bash
git clone <this-repo>
cd QuaRot
pip install -e .
```

Dependencies: `torch`, `transformers`, `accelerate`, `fast-hadamard-transform`, `sympy`, `scipy`, `tqdm`.

---

## Usage

### 1. Run GBS (search for clipping ratios)

```bash
python ./fake_quant/RotateGridSearch.py \
  --model meta-llama/Llama-2-7b-hf \
  --rotate \
  --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 \
  --w_clip \
  --bsz 6 \
  --k_groupsize 128 \
  --start_bit 16 \
  --alpha 0.1 \
  --max_iterations 8 \
  --expand ./save/mix_compute/0.json \
  --cache_dir /path/to/hf/cache
```

See `script_grid_search.sh` for a ready-to-run example.

### 2. Evaluate with the found configuration

```bash
python ./fake_quant/RotateGridSearch.py \
  --model meta-llama/Llama-2-7b-hf \
  --rotate \
  --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 \
  --w_clip \
  --expand ./save/mix_compute/0.json \
  --cache_dir /path/to/hf/cache \
  --eval \
  --grid_search
```

### 3. Resume an interrupted search

If a search is interrupted, resume it from the last completed projection:

```bash
python ./fake_quant/RotateGridSearch.py \
  ... (same arguments as the original run) \
  --resume_gs
```

---

## Key arguments

### GBS-specific

| Argument | Default | Description |
|---|---|---|
| `--start_bit` | 32 | Bit-width used as starting baseline for the search. Set to 16 to warm-start from a half-precision baseline. |
| `--alpha` | 0.1 | Fraction of the WikiText2 training set used as calibration data during search. |
| `--max_iterations` | 10 | Number of ternary search iterations per (layer, projection) pair. Each iteration uses 2 PPL evaluations. |
| `--expand` | `./save/mix_compute/0.json` | Path to the dimension expansion config. Use `0.json` for no expansion (standard QuaRot). |
| `--resume_gs` | False | Resume an interrupted search from the last checkpoint. |
| `--inv` | False | Run the search in reverse layer order (for experimental purposes). |
| `--optim_down8b` | False | Always search `down_proj` at 8 bits regardless of the target bit-width. |
| `--eval` | False | Skip the search and only evaluate PPL (requires a saved config when combined with `--grid_search`). |
| `--grid_search` | False | Load the best saved config from disk when evaluating. |

### Quantization

| Argument | Default | Description |
|---|---|---|
| `--w_bits` | 16 | Weight quantization bit-width. |
| `--a_bits` | 16 | Activation quantization bit-width. |
| `--k_bits` | 16 | Key-cache quantization bit-width. |
| `--v_bits` | 16 | Value-cache quantization bit-width. |
| `--w_groupsize` / `--a_groupsize` | -1 | Group size for quantization (-1 = per-token). Must be equal. |
| `--k_groupsize` | -1 | Group size for key-cache quantization. |
| `--w_clip` | False | Enable MSE-based weight clipping during GPTQ. |
| `--w_rtn` | False | Use round-to-nearest instead of GPTQ for weight quantization. |

### Rotation

| Argument | Default | Description |
|---|---|---|
| `--rotate` | False | Apply QuaRot rotation before quantization. |
| `--rotate_mode` | `hadamard` | Rotation type: `hadamard` (recommended) or `random`. |
| `--fp32_had` | False | Apply online Hadamard transforms in FP32 (more accurate, slower). |

---

## Dimension expansion config format

The `--expand` argument points to a JSON file that describes how many dimensions to add to each layer's hidden size before applying the Hadamard transform.

```json
{
  "attention": {
    "0": 128,
    "1": 128,
    "other": 0
  },
  "mlp": {
    "0": 64,
    "other": 0
  }
}
```

- Keys are layer indices (as strings). The special key `"other"` acts as a default for all unlisted layers.
- A value of `0` means no expansion (standard QuaRot behaviour).
- The `attention` and `mlp` sections control expansion independently for the attention block and the MLP block.
- Pre-computed Hadamard matrices for the expanded dimensions are cached in `./save/hadK/`.

The file `./save/mix_compute/0.json` contains all-zero expansions and is the default (equivalent to standard QuaRot).

---

## Output

Search results are saved incrementally to:

```
./grid_search/<model_name>/<run_name>.json
```

The best configuration found at the end of the search is saved to:

```
./grid_search/<model_name>/best_config_<run_name>.json
```

Each entry in the JSON records the config (per-layer clipping ratios and bit-widths) and the corresponding train/test perplexity:

```json
{
  "config": {
    "max_bit": [
      {
        "q_proj":       {"max": 0.875, "bit": 4},
        "k_proj":       {"max": 0.812, "bit": 4},
        "v_proj":       {"max": 0.906, "bit": 4},
        "qk_rotation":  {"max": 0.750, "bit": 4},
        "o_proj":       {"max": 0.843, "bit": 4},
        "gate_proj":    {"max": 0.968, "bit": 4},
        "up_proj":      {"max": 0.937, "bit": 4},
        "down_proj":    {"max": 0.781, "bit": 4}
      }
    ],
    "layer": 0,
    "bit": 4
  },
  "ppl": {
    "train": 5.234,
    "test": 5.456
  }
}
```

---

## QuaRot citation

GBS builds directly on QuaRot. If you use this code, please also cite the original QuaRot paper:

```bibtex
@article{ashkboos2024quarot,
  title={QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs},
  author={Ashkboos, Saleh and Mohtashami, Amirkeivan and Croci, Maximilian L and Li, Bo and Jaggi, Martin and Alistarh, Dan and Hoefler, Torsten and Hensman, James},
  journal={arXiv preprint arXiv:2404.00456},
  year={2024}
}
```
