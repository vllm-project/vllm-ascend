# MiniCPM3-4B Ascend NPU Adaptation — Skill Record

## Overview

This skill documents the process of adapting the MiniCPM3-4B model (OpenBMB) to run on Ascend NPU via `vllm-ascend`. The model uses Multi-head Latent Attention (MLA) similar to DeepSeek-V2 and LongRoPE position encoding.

## Key Findings

### 1. Model Natively Supported in vLLM

`MiniCPM3ForCausalLM` is already registered in vLLM's model registry (`vllm/model_executor/models/minicpm3.py`). No custom model registration in `vllm-ascend/models/` is needed — it works through vLLM's native model support.

**Model architecture:** Inherits from `MiniCPMForCausalLM` → `MiniCPMModel` with custom `MiniCPM3Attention` (MLA).

### 2. Critical Requirement: `trust_remote_code=True`

MiniCPM3-4B ships custom modeling code in its repository (`modeling_minicpm.py`, `configuration_minicpm.py`). You **must** pass `--trust-remote-code` when loading the model. Without it, the model fails with `ImportError: cannot import name 'MiniCPM3Config'`.

### 3. Model Config Issues with `rope_parameters`

The model's `config.json` has `rope_scaling` (not `rope_parameters`). vLLM internally normalizes this, but warnings appear:

```text
`rope_parameters`'s short_factor field must have length 48, got 16
```

These warnings are benign — the model works correctly despite them. The MiniCPM3-4B uses `qk_rope_head_dim=32`, so the factor arrays should have `32/2=16` elements, not 48.

### 4. Triton-Ascend PCH Compilation Bug

**Symptoms:** Runtime crash with:

```text
RuntimeError: Failed to compile ... precompiled.h.gch, error: ,cmd=['...', '-shared', '-fPIC', '-o', gch_path]
```

**Root cause:** `triton/backends/ascend/utils.py` passes `-shared -fPIC` flags for precompiled header (`.gch`) compilation. clang++ rejects `-shared` for header compilation.

**Fix:** Remove `-shared` from line 357 (keep `-fPIC` for PIE consistency):

```python
# BEFORE:
cc_cmd += ["-std=c++17", "-shared", "-fPIC", "-o", gch_path]
# AFTER:
cc_cmd += ["-std=c++17", "-fPIC", "-o", gch_path]
```

**Why `-fPIC` is needed:** Without `-fPIC`, the subsequent `.so` compilation using this PCH fails with `is pie differs in PCH file vs. current file`.

### 5. NPU Memory Requirements

| Component | Memory |
|-----------|--------|
| Model weights (BF16) | ~7.6 GiB |
| Peak activations | ~0.5 GiB |
| CUDA graphs (est) | ~1.9 GiB |
| Total (gpu_mem_util=0.85) | ~48.8 GiB budget |
| Available KV cache | ~38.6 GiB |

The 4B model fits comfortably on a single 910B3 NPU (64 GiB).

## Effective Prompts

### Initial Model Loading Test

```bash
ASCEND_RT_VISIBLE_DEVICES=5 python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(model='/data/zkx/weights/MiniCPM3-4B', trust_remote_code=True, ...)
outputs = llm.generate(['Hello'], SamplingParams(temperature=0.7, max_tokens=32))
print(outputs[0].outputs[0].text)
"
```

### Debugging Triton PCH Issue

```bash
# Reproduce manually:
/usr/bin/clang++ -x c++-header precompiled.h ... -shared -fPIC -o precompiled.h.gch
# → clang++: error: cannot specify -o when generating multiple output files
```

### Full Serving Test

```bash
ASCEND_RT_VISIBLE_DEVICES=5 vllm serve /path/to/MiniCPM3-4B \
  --served-model-name minicpm3-4b \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
```

## Best Practices

1. **Always use `--trust-remote-code`** for MiniCPM3 — the custom model code is required.
2. **Triton cache:** When debugging triton compilation issues, clear both caches:
   ```bash
   rm -rf /root/.triton/cache/* /tmp/triton/*
   ```
3. **Graph mode vs Eager:** The model works in both modes. Graph mode (default) is faster but requires the triton PCH fix. Use `--enforce-eager` as a fallback.
4. **NPU selection:** Use `ASCEND_RT_VISIBLE_DEVICES=5` to target NPU5 (or adjust for your setup).
5. **Test YAML config** follows the standard `vllm-ascend` pattern — `model_name` refers to the HuggingFace ID; override locally by editing the YAML.
6. **Accuracy targets:** GSM8K 5-shot: ~45% strict-match, ~50% flexible-extract.

## Files Modified / Created

| File | Action | Purpose |
|------|--------|---------|
| `triton/backends/ascend/utils.py:357` | Fixed | Remove `-shared` from GCH compilation flags |
| `tests/e2e/models/configs/MiniCPM3-4B.yaml` | Verified | Test configuration |
| `tests/e2e/models/configs/accuracy_groups_a2.json` | Updated | Added to nightly & pr-only groups |
| `docs/source/tutorials/models/MiniCPM3-4B.md` | Updated | Added local path examples & triton troubleshooting |
