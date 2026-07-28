# DeepSeek-V4 Quantization

Convert DeepSeek-V4 native FP8 weights to W8A8 or W4A8 quantized weights in ModelSlim format, loadable by vllm-ascend on Ascend NPU without `--quantization` flag.

## Supported Models

| Model | Config File | Quant Scheme | Auto-detected? |
|-------|------------|--------------|----------------|
| **DeepSeek-V4-Flash** | `quantize_config.yaml` | W8A8 (all Linear) | Yes (default) |
| **DeepSeek-V4-Pro** | `quantize_config_pro_w4a8.yaml` | W4A8 (routed experts) + W8A8 (attn, shared_experts, MTP) | No — must pass `--config` |
| **DeepSeek-V4-Flash-DSpark** | `quantize_config_dspark.yaml` | W8A8 (all Linear) | Yes (`dspark_block_size` in config.json) |
| **DeepSeek-V4-Pro-DSpark** | `quantize_config_dspark.yaml` | W8A8 (all Linear) | Yes (`dspark_block_size` in config.json) |

> Regardless of auto-detection, **always pass `--config` explicitly** for clarity. See Usage below.

**Key differences between variants**:
- **Flash**: 1 MTP layer, uses `e_proj`/`h_proj` for MTP projection
- **Pro**: 1 MTP layer, uses `e_proj`/`h_proj` for MTP projection; routed experts go W4A8 (4-bit) for smaller checkpoint size
- **DSpark** (Flash-DSpark + Pro-DSpark): 3 MTP draft layers, uses `main_proj` for MTP projection; draft `wo_a` kept as FP8 (vllm-ascend has dedicated dequant path)

## Prerequisites

- **Hardware**: Ascend NPU (Atlas 800I A2 / Atlas A3 Inference series) — optional, see `--device` below
- **Software**:
  - PyTorch (required)
  - torch_npu (optional, only needed for `--device npu`)
  - vllm-ascend (for loading the quantized weights)
- **Input**: DeepSeek-V4 FP8 HF weights:
  - [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
  - [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
  - [DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)
  - [DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark)

## Usage

> **Always pass `--config` explicitly.** Each model variant needs its own YAML config file. If omitted, the script auto-selects based on `config.json`'s `dspark_block_size` field (DSpark → `quantize_config_dspark.yaml`, otherwise → `quantize_config.yaml`), but passing it explicitly avoids ambiguity — especially for Pro, which is NOT auto-detected and must use `quantize_config_pro_w4a8.yaml`.

### DeepSeek-V4-Flash (W8A8)

```bash
python examples/quantization/deepseek_v4/quantize.py \
    --input_fp8_hf_path /path/to/DeepSeek-V4-Flash \
    --output_hf_path /path/to/DeepSeek-V4-Flash-w8a8 \
    --config examples/quantization/deepseek_v4/quantize_config.yaml
```

Serve:
```bash
vllm serve /path/to/DeepSeek-V4-Flash-w8a8
```

### DeepSeek-V4-Pro (W4A8 + W8A8 mixed)

```bash
python examples/quantization/deepseek_v4/quantize.py \
    --input_fp8_hf_path /path/to/DeepSeek-V4-Pro \
    --output_hf_path /path/to/DeepSeek-V4-Pro-w4a8 \
    --config examples/quantization/deepseek_v4/quantize_config_pro_w4a8.yaml
```

Serve:
```bash
vllm serve /path/to/DeepSeek-V4-Pro-w4a8
```

### DeepSeek-V4-Flash-DSpark (W8A8)

```bash
python examples/quantization/deepseek_v4/quantize.py \
    --input_fp8_hf_path /path/to/DeepSeek-V4-Flash-DSpark \
    --output_hf_path /path/to/DeepSeek-V4-Flash-DSpark-w8a8 \
    --config examples/quantization/deepseek_v4/quantize_config_dspark.yaml
```

Serve with speculative config:
```bash
vllm serve /path/to/DeepSeek-V4-Flash-DSpark-w8a8 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

### DeepSeek-V4-Pro-DSpark (W8A8)

```bash
python examples/quantization/deepseek_v4/quantize.py \
    --input_fp8_hf_path /path/to/DeepSeek-V4-Pro-DSpark \
    --output_hf_path /path/to/DeepSeek-V4-Pro-DSpark-w8a8 \
    --config examples/quantization/deepseek_v4/quantize_config_dspark.yaml
```

Serve with speculative config:
```bash
vllm serve /path/to/DeepSeek-V4-Pro-DSpark-w8a8 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

> **Note**: Pro-DSpark uses W8A8 (not W4A8) for all layers. The W4A8 config is only for non-DSpark Pro.

### Device Selection (`--device`)

The script can run quantization compute on either CPU or NPU. The bottleneck is disk I/O and FP8 dequantization (always CPU), not the quantization compute itself — so CPU mode is only ~1 min slower for Flash-scale models.

```bash
# Auto-detect: uses NPU if torch_npu is installed and NPU is available, otherwise CPU (default)
python examples/quantization/deepseek_v4/quantize.py \
    --input_fp8_hf_path /path/to/model \
    --output_hf_path /path/to/output \
    --config examples/quantization/deepseek_v4/quantize_config_pro_w4a8.yaml

# Force CPU (no NPU dependency, works on any machine)
python examples/quantization/deepseek_v4/quantize.py \
    --input_fp8_hf_path /path/to/model \
    --output_hf_path /path/to/output \
    --device cpu

# Force NPU
python examples/quantization/deepseek_v4/quantize.py \
    --input_fp8_hf_path /path/to/model \
    --output_hf_path /path/to/output \
    --device npu
```

| Device | Quant compute | Total time (Flash) | Requirements |
|--------|--------------|-------------------|--------------|
| `auto` (default) | NPU if available, else CPU | ~10 min (NPU) / ~13 min (CPU) | torch_npu for NPU |
| `cpu` | CPU | ~13 min | PyTorch only |
| `npu` | NPU | ~10 min | torch_npu + NPU hardware |

## Output Format

The output directory contains ModelSlim-format quantized weights:

| File | Description |
|------|-------------|
| `quant_model_weights-{N:05d}-of-{M:05d}.safetensors` | Sharded quantized weights (~4GB per shard) |
| `quant_model_weights.safetensors.index.json` | Shard index with `weight_map` and `total_size` |
| `quant_model_description.json` | Per-weight quant type (`W8A8_DYNAMIC`, `W4A8_DYNAMIC`, or `FLOAT`) |
| `config.json` | Model config with `quantization_config` removed |
| `*.py`, `*.json`, `*.jinja` | Copied from input (tokenizer, modeling code, etc.) |

### How vllm-ascend Loads the Output

1. `config.json` has no `quant_method` field
2. vllm-ascend's `AscendModelSlimConfig.override_quantization_method()` detects this on NPU and auto-selects the ascend path
3. `maybe_update_config()` loads `quant_model_description.json` to determine per-layer quantization scheme
4. No `--quantization` flag needed when serving

## Quantization Config (YAML)

### DeepSeek-V4-Flash (`quantize_config.yaml`)

W8A8 for all Linear layers:

- **`*attn*`** (exclude `wq_a`, `wkv`, `wo_a`, compressor, indexer) — quantize `wq_b`, `wo_b`
- **`*ffn*`** (exclude `*gate`) — quantize expert and shared_expert weights
- **`*e_proj`, `*h_proj`** — quantize MTP projection layers

Each quantized weight produces three tensors:
- `.weight` (int8) — quantized weight
- `.weight_scale` (float32) — per-channel scale
- `.weight_offset` (float32) — per-channel offset (zeros for symmetric quant)

### DeepSeek-V4-Pro (`quantize_config_pro_w4a8.yaml`)

Mixed W4A8 + W8A8 — routed experts use W4A8 (4-bit) for smaller checkpoint, everything else stays W8A8:

- **Rule 1: attn** → W8A8 (exclude `wq_a`, `wkv`, `wo_a`, compressor, indexer)
- **Rule 2: ffn routed experts** → W4A8 (routed experts only, exclude `*gate` and `shared_experts`)
- **Rule 3: shared_experts** → W8A8 (every token executes, precision-sensitive)
- **Rule 4: MTP projections** → W8A8 (`e_proj`, `h_proj` — Pro uses same MTP structure as Flash)

W4A8 quantized weights produce four tensors per layer:
- `.weight` (int8, packed int4) — two 4-bit values packed per int8 element, shape `[output//2, input]`
- `.weight_scale` (float32) — per-channel scale
- `.weight_offset` (float32) — per-channel offset (zeros for symmetric quant)
- `.scale_bias` (float32) — precomputed `(quant_weight * scale).sum(dim=1) * 8`

W4A8 algorithm aligns with msmodelslim:
- Per-channel symmetric quantization (minmax, Q_MAX=7, Q_MIN=-8)
- int4 packing: transpose → pack along output dim (`high<<4 | low&0x0F`) → transpose back
- `scale_bias` calculated from quantized integer values (not original weights)

### DeepSeek-V4-DSpark (`quantize_config_dspark.yaml`)

W8A8 for all Linear layers. DSpark has 3 MTP draft layers (vs Flash's 1), uses `main_proj` instead of `e_proj`/`h_proj`, and has `markov_head`/`confidence_head` (not quantized):

- **`*attn*`** (exclude `wq_a`, `wkv`, `wo_a`, compressor, indexer) — quantize `wq_b`, `wo_b` (both main + draft)
- **`*ffn*`** (exclude `*gate`) — quantize expert and shared_expert weights (both main + draft)
- **`*main_proj`** — quantize DSpark MTP main projection (replaces `e_proj`/`h_proj`)

**Draft wo_a special handling**: DSpark draft model `wo_a` (`mtp.N.attn.wo_a`) is kept as FP8 + `.scale` (not dequantized, not quantized). vllm-ascend's dspark `load_weights` has a dedicated FP8→BF16 dequant path for draft wo_a. Main model `wo_a` is dequantized to BF16 and marked `FLOAT` (same as Flash).

## How It Works

```
FP8 weight + .scale ──decode_fp8──▶ BF16 weight
                                          │
                              YAML match + dim==2?
                                    ├── W8A8 rule ── weight_quant_sym_perchannel ──▶ int8 + scale + offset (W8A8_DYNAMIC)
                                    ├── W4A8 rule ── weight_quant_w4a8_perchannel ──▶ packed int4 + scale + offset + scale_bias (W4A8_DYNAMIC)
                                    └── No match  ── keep original (FLOAT)

DSpark draft wo_a (mtp.N.attn.wo_a): skip dequant/quant, output FP8 + .scale as-is
```

Quantization logic references [msmodelslim](https://gitee.com/ascend/msit/tree/master/msmodelslim):
- FP8 dequant: `decode_fp8` / `decode_fp4` (128×128 block)
- W8A8 weight quant: per-channel symmetric (`clamp([-128, 127])`)
- W4A8 weight quant: per-channel symmetric minmax (Q_MAX=7), int4 packed in int8, scale_bias precomputed
- Output sharding: 4GB per shard (BufferedSafetensorsWriter)

## Crash Recovery (Checkpoint/Resume)

Quantizing large models (e.g. DeepSeek-V4-Pro-DSpark ~1.6T) can take hours. The script automatically saves a checkpoint after each input shard is processed, so if the process crashes (OOM, disk full, killed), re-running the same command resumes from the last completed shard — no progress is lost.

**Usage**: no extra flags needed. Just re-run the same command:

```bash
# If the previous run crashed, simply re-run:
python examples/quantization/deepseek_v4/quantize.py \
    --input_fp8_hf_path /path/to/DeepSeek-V4-Pro-DSpark \
    --output_hf_path /path/to/output
```

The script detects the checkpoint file (`.quantize_checkpoint.json` in the output directory), restores progress, skips already-completed input shards, and continues. The checkpoint is automatically deleted after `writer.close()` succeeds (all shards processed, index.json and quant_model_description.json written).

**What happens on crash**:
- Each input shard is force-flushed to disk + checkpoint written atomically (temp file + rename)
- On resume: stale temporary shard files (from a crash between flush and checkpoint) are cleaned up, completed shards are skipped
- On successful completion: checkpoint file is deleted, output is identical to a non-interrupted run
