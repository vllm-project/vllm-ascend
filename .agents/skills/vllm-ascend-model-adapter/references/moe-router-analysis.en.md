# MoE Router Analysis

This document analyzes vLLM Ascend’s current implementation capabilities at the MoE `router / gate` layer. The focus here is "router structure and selection logic", not the complete MoE pipeline; please read it in conjunction with `moe-fused-analysis.md` for the complete execution pipeline.

## 1. What problem does this layer solve?

The router layer currently mainly solves:

- How router logits becomes `topk_ids / topk_weights`
- Whether to use fused gating op
- Whether grouped top-k, hash routing, and bias correction are supported?
- How the router output is aligned with the token dispatch contract

## 2. Overview of current capabilities

The current main entrance of router is at:

- [experts_selector.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py)

The core features are:

- First determine whether you can go fused gating
- Otherwise fall back to native path
- The router output will directly enter the token dispatch / combine contract

Therefore, the current router is not a "single top-k operator", but a set of selection layers with shunt conditions.

## 3. Key capabilities currently implemented

### 3.1 fused gating is now a formal capability

The current implementation supports expert selection via a fused op, rather than relying solely on the Python/torch fallback.

Main paths include:

- `DeviceOperator.moe_gating_top_k(...)`
- `torch.ops._C_ascend.moe_gating_top_k_hash(...)`

This shows that the Ascend side has regarded the router as a performance-sensitive layer, not just the upstream model logic.

### 3.2 scoring function has explicit support for collections

It can be seen from the existing implementation and existing analysis that the current fused router has covered:

- `softmax`
- `sigmoid`
- `sqrtsoftplus`

And supports some additional behaviors:

- grouped top-k
- `renormalize`
- `e_score_correction_bias`
- `routed_scaling_factor`

Therefore, the current router capability is not only "standard Mixtral top-k", but already covers a variety of gating styles.

### 3.3 hash routing has special support

Currently in the code:

- Specialized hash routing path based on `tid2eid + input_ids`

This shows that the current warehouse has absorbed the routing semantics of DeepSeek V4 instead of only supporting simple router weight matmul.

### 3.4 shared expert will affect router output organization

In the current implementation, `mix_placement` will complete the expert id / weight of the shared expert.

This means that the current router output is not solely responsible for routed experts, it must also be compatible with the shared expert wiring method.

## 4. Current structural assumptions

The current implicit structural assumptions of the router layer include:

- router should still be reduced to `topk_ids + topk_weights` in the end
- The number of scoring functions is limited, it is best to hit the fused support set
- The router output must strictly match the subsequent dispatch contract
- grouped top-k and correction bias can be formal structures rather than unusual special cases

## 5. 已知边界与风险

The main boundaries of the current router layer are:

- `custom_routing_function` will destroy the universality of fused path
- Not all scoring + renormalize combinations are supported
- Hash routing is supported, but relies more on specific input contracts
- Router problems often appear to be "not supported by MoE", but the essence is that the gate rules are not in the current set.

## 6. What to look for when analyzing this layer

It is recommended to give priority to:

- `top_k`
- `scoring_func`
- `renormalize`
- `grouped_topk`
- `e_score_correction_bias`
- Does `custom_routing_function` exist?
- Whether to rely on `input_ids` hash

## 7. Related code

- [ops/fused_moe/experts_selector.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py)
- [ops/fused_moe/fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py)
- [moe-fused-analysis.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/moe-fused-analysis.md)
