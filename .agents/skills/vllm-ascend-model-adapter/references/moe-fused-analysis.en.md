# MoE Fused Baseline

This document is an English companion to the Chinese `moe-fused-analysis.md`. It summarizes the current MoE execution capability in `vllm_ascend/ops/fused_moe`.

## Scope

The main files are:

- `vllm_ascend/ops/fused_moe/fused_moe.py`
- `vllm_ascend/ops/fused_moe/experts_selector.py`
- `vllm_ascend/ops/fused_moe/moe_comm_method.py`
- `vllm_ascend/ops/fused_moe/token_dispatcher.py`
- `vllm_ascend/ops/fused_moe/prepare_finalize.py`
- `vllm_ascend/ops/fused_moe/moe_mlp.py`
- `vllm_ascend/ops/fused_moe/moe_runtime_args.py`

## Core conclusion

The current Ascend MoE path is not a single fused kernel. It is a layered pipeline:

1. `prepare`
2. `router / top-k select`
3. `token dispatch`
4. `expert MLP`
5. `token combine`
6. `finalize`

The strongest current coverage is decoder-only routed MoE LLMs, especially the common pattern:

- top-k router
- two-stage expert MLP
- `SwiGLU`
- `down_proj`

## Main structure assumptions

- experts are usually represented as `w13/gate_up + swiglu + w2/down`
- router output must be reducible to `topk_ids + topk_weights`
- token dispatch is a formal contract, not just an implementation detail
- shared experts are supported, but are still analyzed separately from routed experts

## Communication paths

The current implementation contains multiple communication modes:

- `ALLGATHER`
- `ALLTOALL`
- `MC2`
- `FUSED_MC2`

The most reusable path is still the generic all-gather route. MC2 and fused MC2 are more specialized performance paths.

## Quantization

MoE quantization is already integrated into the formal execution stack. The repo contains dedicated quantized fused-MoE methods for several formats, including W8A8, W4A8, W4A16, and MX/FP8-related variants.

## Related documents

- [moe-router-analysis.en.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/moe-router-analysis.en.md)
- [moe-experts-analysis.en.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/moe-experts-analysis.en.md)
- [shared-expert-residual-mlp-analysis.en.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/shared-expert-residual-mlp-analysis.en.md)

For the exhaustive section-by-section Chinese deep dive, see the original:

- [moe-fused-analysis.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/moe-fused-analysis.md)
