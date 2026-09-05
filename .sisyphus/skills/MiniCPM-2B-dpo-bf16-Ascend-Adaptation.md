# MiniCPM-2B-dpo-bf16 Ascend NPU Adaptation — Skill Record

## Overview

This skill documents the adaptation of the MiniCPM-2B-dpo-bf16 model (OpenBMB) to run on Ascend NPU via `vllm-ascend`. This is a 2B-parameter dense decoder-only transformer, much simpler than MiniCPM3 (no MLA/LongRoPE), using standard GQA attention.

## Key Findings

### 1. Architecture: `MiniCPMForCausalLM`

The model uses architecture `MiniCPMForCausalLM`, which is already registered in vLLM's native model registry at `vllm/model_executor/models/minicpm.py`. No custom model registration in `vllm-ascend` is needed.

Architecture details:

- `hidden_size=2304`, `num_hidden_layers=40`, `intermediate_size=5760`
- `num_attention_heads=36`, `num_key_value_heads=36` (full GQA, no KV reduction)
- Standard SiLU + gate-up-down MLP
- RMSNorm
- `max_position_embeddings=4096`, no RoPE scaling

### 2. Trust Remote Code Required

Like all MiniCPM variants, this model ships custom modeling code (`modeling_minicpm.py`, `configuration_minicpm.py`) and requires `--trust-remote-code`.

### 3. NPU Memory Requirements

| Component | Memory |
|-----------|--------|
| Model weights (BF16) | ~4.0 GiB |
| Peak activations | ~0.3 GiB |
| CUDA graphs (est) | ~1.0 GiB |
| Available KV cache | ~42+ GiB |

The 2B model easily fits on a single 910B3 NPU (64 GiB) with ample KV cache headroom.

### 4. No Triton Issues

Unlike MiniCPM3 (which has MLA), this model uses standard attention. After applying the triton PCH fix from the MiniCPM3 adaptation, the model runs without issues in both eager and graph modes on NPU5.

## Effective Prompts

### Quick Model Loading Test

```bash
ASCEND_RT_VISIBLE_DEVICES=5 python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(model='/data/zkx/weights/MiniCPM-2B-dpo-bf16', trust_remote_code=True, ...)
outputs = llm.generate(['Hello'], SamplingParams(temperature=0.7, max_tokens=32))
print(outputs[0].outputs[0].text)
"
```

### Full Serving

```bash
ASCEND_RT_VISIBLE_DEVICES=5 vllm serve /data/zkx/weights/MiniCPM-2B-dpo-bf16 \
  --served-model-name minicpm-2b-dpo \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
```

## Best Practices

1. **`trust_remote_code` is mandatory** — both MiniCPM and MiniCPM3 ship custom code.
2. **Standard GQA** — no special attention backend needed; works with vLLM's default Attention layer.
3. **Very low memory footprint** — ~4GB weights, leaves ~42GB+ for KV cache.
4. **Graph mode works out of the box** — no `enforce_eager` needed after triton PCH fix.

## Files Created

| File | Purpose |
|------|---------|
| `tests/e2e/models/configs/MiniCPM-2B-dpo-bf16.yaml` | Test configuration (GSM8K 5-shot) |
| `docs/source/tutorials/models/MiniCPM-2B-dpo-bf16.md` | Tutorial documentation |
| `tests/e2e/models/configs/accuracy_groups_a2.json` | Added to nightly & pr-only groups |
