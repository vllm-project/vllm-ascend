# MoE Fused Baseline

This document summarizes the current fused-MoE execution capability in `vllm_ascend/ops/fused_moe`.

## Scope

Main files:

- `vllm_ascend/ops/fused_moe/fused_moe.py`
- `vllm_ascend/ops/fused_moe/experts_selector.py`
- `vllm_ascend/ops/fused_moe/moe_comm_method.py`
- `vllm_ascend/ops/fused_moe/token_dispatcher.py`
- `vllm_ascend/ops/fused_moe/prepare_finalize.py`
- `vllm_ascend/ops/fused_moe/moe_mlp.py`
- `vllm_ascend/ops/fused_moe/moe_runtime_args.py`

## Core conclusion

The current Ascend MoE path is a layered pipeline:

1. `prepare`
2. `router / top-k select`
3. `token dispatch`
4. `expert MLP`
5. `token combine`
6. `finalize`

The strongest current coverage is decoder-only routed MoE LLMs with a standard two-stage expert MLP.

## Main assumptions

- experts usually look like `w13/gate_up + swiglu + w2/down`
- router output should reduce to `topk_ids + topk_weights`
- token dispatch is a formal runtime contract
- shared experts exist, but should be analyzed separately from routed experts
