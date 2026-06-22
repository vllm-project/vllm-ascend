# MoE Fused Baseline

This document summarizes the current implementation capabilities of `vllm_ascend/ops/fused_moe` as the **MoE capability baseline** for the `vllm-ascend-model-adapter` skill when adapting to the new MoE model.

The goal is not to explain universal MoE principles, but to answer these adaptation questions:

1. What is currently supported by the MoE path of `vllm-ascend`.
2. What structural assumptions does the current implementation make about the MoE layer of the new model.
3. What is the difference between the router / experts / shared expert / EP path of the new model and the existing baseline.
4. The difference is more likely to lie in upstream vLLM model adaptation, weight mapping, framework wiring, or the Ascend backend capability itself.

## 1. Code range

Main analysis objects:

- `vllm_ascend/ops/fused_moe/fused_moe.py`
- `vllm_ascend/ops/fused_moe/experts_selector.py`
- `vllm_ascend/ops/fused_moe/moe_comm_method.py`
- `vllm_ascend/ops/fused_moe/token_dispatcher.py`
- `vllm_ascend/ops/fused_moe/prepare_finalize.py`
- `vllm_ascend/ops/fused_moe/moe_mlp.py`
- `vllm_ascend/ops/fused_moe/moe_runtime_args.py`

Related dependencies:

- `vllm_ascend/utils.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/ascend_forward_context.py`

## 2. Overall conclusion

The current Ascend MoE path is not a "single fused op", but a layered pipeline:

1. `prepare`
2. `router/top-k select`
3. `token dispatch`
4. `expert MLP`
5. `token combine`
6. `finalize`

There are three core facts:

1. The current implementation has covered the mainstream execution path of **decoder-only routed MoE LLM**, especially the `topk router + SwiGLU expert + down_proj` type of structure.
2. The current implementation strongly relies on a **two-stage expert MLP hypothesis**: the first stage is `w13`/`gate_up`, the middle is `swiglu`, and the second stage is `w2`/`down`.
3. When adapting to a new model, the most common problem is usually not "Ascend does not have MoE", but:
- router rules are different;
- Expert weight layout is different;
- shared expert wiring is different;
- EP/TP path assumptions are different;
- quant/com/metadata contract not connected.

## 3. Implement entrance and main pipeline

### 3.1 Layer entrance

`AscendFusedMoE` takes over the upstream `FusedMoE` behavior in the [fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py:93) group of classes.

Key points:

- `AscendUnquantizedFusedMoEMethod.is_monolithic=False`
- `maybe_make_prepare_finalize()` directly returns `None`
- `AscendMoERunner.forward_impl()` Change to Ascend's own `forward_impl`

This means that the current implementation explicitly bypasses upstream modular-kernel prepare/finalize initialization and uses Ascend's own communication and execution stack.

### 3.2 Pipeline hub

The real pipeline hub is at [moe_comm_method.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:122):

1. `build_token_dispatch_input`
2. `token_dispatcher.token_dispatch(...)`
3. `build_mlp_compute_input`
4. `unified_apply_mlp(...)`
5. `token_dispatcher.token_combine(...)`

In other words, the current MoE capabilities can be broken down and analyzed according to 4 questions:

- How to choose an expert in router;
- How to send token to experts;
- How to calculate expert internally;
- How to merge the results back.

This is important for skills because the fit differences for new models often fall squarely into one of these four chunks.

## 4. Currently supported MoE structural assumptions

### 4.1 Existing structural assumptions

From [fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py:108) and [moe_mlp.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py:344), the current implementation of the default expert structure is:

1. `router_logits -> topk_ids/topk_weights`
2. `expert gate_up / w13`
3. `swiglu` or compatible activation
4. `expert down / w2`

Weights are organized around these objects by default:

- `layer.w13_weight`
- `layer.w2_weight`
- Optional `w13_bias`
- Optional `w2_bias`
- Optional scale / offset / scale_bias

Therefore, this baseline is best used to compare models such as:

- DeepSeek style routed experts
- `gate/up/down` expert structure like Qwen / Mixtral
- A model that shares the same router but each expert is still a two-stage MLP

### 4.2 Currently supported structures should not be directly assumed

In the following situations, you cannot think that you can run through it directly just by looking at "MoE in the warehouse":

1. expert is not a two-part form of `gate_up + swiglu + down`.
2. Expert has additional linear layers, convolutions, parallel branches or residual expert inside.
3. The router is not standard top-k gating, or requires special normalization/post-processing.
4. Shared expert is not the parallel splicing method assumed by the current implementation.
5. Expert weight naming and layout are completely different from existing loaders.

These should be determined as **MoE layer gap** first, and then decide whether it is an adapter problem or a backend problem.

## 5. Router / Gate capability baseline

### 5.1 Main entrance

The router entry is at `select_experts(...)` of [experts_selector.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py:30).

It first makes a capability determination:

- If you can use NPU fused gating, just use fused path
- Otherwise fall back to native path

### 5.2 Adapted router features

Judging from [check_npu_moe_gating_top_k](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py:140) and `_select_experts_with_fusion_ops(...)`, the current capabilities include:

- `softmax` top-k
- `sigmoid` top-k
- `sqrtsoftplus` top-k
- grouped top-k
- `renormalize`
- `e_score_correction_bias`
- `routed_scaling_factor`
- Use the hash routing specialization of `tid2eid + input_ids`
- `mix_placement` adds expert id/weight to shared expert.

### 5.3 Typical fused path

Currently there are two main types of fused gate paths:

1. `DeviceOperator.moe_gating_top_k(...)`
2. `torch.ops._C_ascend.moe_gating_top_k_hash(...)`

The second type is more like the specialized support of DeepSeek V4/hash routing.

### 5.4 fallback conditions

The following situations will cause the code to fall back to the native router path:

- `custom_routing_function` is not empty
- `scoring_func` is not in the current support set
- `sigmoid` and `renormalize=False`
- The grouping parameters do not satisfy the current fused op constraints

This shows that when skill looks at a new model, it cannot just look at "top_k=8". Also see:

- scoring function
- grouped top-k organization method
- correction bias
- Whether to customize routing function
- Whether to rely on token id hash

## 6. Token Dispatch / Combine ability baseline

### 6.1 Currently supported communication types

[setup_moe_comm_method](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:55) Current registration:

- `ALLGATHER`
- `ALLTOALL`
- `MC2`
- `FUSED_MC2`

When `ep_size == 1` is used, only `ALLGATHER` is built by default.

### 6.2 Positioning of each communication type

| Communication type | Current positioning | Typical dependencies | Adaptation meaning |
| --- | --- | --- | --- |
| `ALLGATHER` | Universal default path | `npu_moe_init_routing` + `npu_moe_token_unpermute` | The best compatibility, give priority to determine whether the new model can connect to this path first |
| `ALLTOALL` | Path to better performance in EP scenarios | all-to-all-v + grouped matmul | Perception is usually not required at the model level, but EP behavior must be aligned |
| `MC2` | Ascend specializes communication calculations in parallel paths | `npu_moe_distribute_dispatch/combine` | Relies on more forward context and distributed constraints |
| `FUSED_MC2` | More radical dispatch+ffn+combine fusion path | `_C_ascend.dispatch_ffn_combine` / `dispatch_gmm_combine_decode` | The strongest weight format and input parameter contract |

### 6.3 AllGather path

[AllGatherCommImpl](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:185) is currently the most common baseline path.

Features:

- The default compatibility is the best
- `token_dispatch` depends on `npu_moe_init_routing`
- `token_combine` depends on `npu_moe_token_unpermute`
- Some quant paths can carry quantization information directly before and after dispatch
- `apply_router_weight_on_input` only supports `topk=1`

The meaning of skill is:

- If the new model only has different expert naming or router wiring, you should usually try to get the AllGather path verification first.
- If even AllGather cannot be connected, consider whether the backend structure assumptions do not match.

### 6.4 MC2 path

[TokenDispatcherWithMC2](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:101) is one of the most important Ascend specialization dispatchers currently available.

Core features:

- Call `torch_npu.npu_moe_distribute_dispatch[_v2]`
- combine calls `torch_npu.npu_moe_distribute_combine[_v2]`
- Support `expert_map`
- Support `global_redundant_expert_num`
- Support `mc2_mask`
- Support hierarchy comm
- Support comm quant mode
- When `should_skip_allreduce_across_dp_group(...)` is true, `global_bs` takes the true upper bound and no longer passes `mc2_mask`

This means that the MC2 path depends not on a single layer, but on a more complete runtime condition:

- `mc2_mask` in forward context
- Token alignment strategy across DPs
- EP/TP group settings
- Distributed world size and max token estimation

### 6.5 Fused MC2 Path

[FusedMC2CommImpl](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:259) further integrates dispatch / ffn / combine into custom op:

- `dispatch_ffn_combine`
- `dispatch_gmm_combine_decode`

Its characteristics are:

- Stronger requirements for input signatures
- `w1` / `w2` should be passed in list form
- Floating point scenes still need to pass dummy scale tensor
- Depends on `expert_map`
- More reliance on NZ weight format

Therefore, Fused MC2 is more of a "performance enhancement path" than a "first place to bring-up new models."

## 7. Prepare / Finalize capability baseline

### 7.1 Function

[prepare_finalize.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py:40) is responsible for:

- padding
- TP split
- DP/EP gather
- Restore shape when finalize
- Bring per-token scale required for quantification when necessary

### 7.2 All2All / MC2

Common characteristics of `PrepareAndFinalizeWithAll2All` and `PrepareAndFinalizeWithMC2`:

- Do padding/split in token dimension
-Restore token arrangement when finalize
- MC2 additional dependency `_EXTRA_CTX.mc2_mask`
- MC2 additional dependency `_EXTRA_CTX.padded_num_tokens`

This means that any new model if changed:

- batch token organization method
- Token arrangement under speculative / splitfuse
- shared expert DP behavior

It may not be a problem with the router itself, but that the prepare/finalize contract is broken.

### 7.3 AllGather

`PrepareAndFinalizeWithAllGather` is compatible with two types of scenarios:

1. Processing based on DP group when not SP
2. Perform EP group processing when SP is turned on

And hidden states can be pre-quantized to:

- `W8A8`
- `MXFP8`

This shows that the MoE quantization path not only exists in the MLP core operator, but also exists in the data entry method in the prepare stage.

## 8. Expert MLP capability baseline

### 8.1 Main structure

[unified_apply_mlp](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py:344) The core structural assumption is:

1. `gmm1` calculates `gate_up`
2. Activation is usually `swiglu`
3. `gmm2` calculates `down`

This is the strongest structural premise of the current MoE baseline.

### 8.2 Non-quantized path

The non-quantified path mainly revolves around:

- `w1`
- `w2`
- `group_list`
- `group_list_type`
- `activation`

The default expert weights are organized by grouped matmul and are often transposed first.

### 8.3 Quantization path

Currently quant MoE MLP covers multiple paths:

- `W8A8`
- `W4A8`
- `MXFP8`
- `MXFP4`
- per-channel W4A8 specialization
- antiquant offset path
- Customized fused grouped matmul + swiglu quant path

Among them, `group_list` / `group_list_type` are strongly constrained parameters because of different kernel requirements:

- prefix sum
- per-expert count
- or other intermediate encoding

### 8.4 Meaning of model adaptation

If the expert structure of the new model is not the current two-stage MLP, then the problem usually should not be directly classified as "quantization is not supported", but:

- expert structure baseline does not match;
- The existing grouped matmul / swiglu contract is not established;
- Requires new upstream MoE layer modeling and even new Ascend backend support.

## 9. Weight layout and format assumptions

### 9.1 Weight post-processing

[process_weights_after_loading](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py:108) does a few key things:

- pad weight
- `w13_weight.transpose(1, 2).contiguous()`
- `w2_weight.transpose(1, 2).contiguous()`
- Force conversion to `FRACTAL_NZ` when `enable_fused_mc2`
- Otherwise go `maybe_trans_nz(...)`

This shows that the current MoE backend has clear requirements for weight layout, rather than accepting any original checkpoint layout.

### 9.2 Adaptation judgment

If new model:

- expert weight is not `w13` / `w2`
- gate and up are not combined weights
- The order of down weight dimensions is different
- shared expert weights are packaged separately
- checkpoint is another tensor packing method

Prioritize suspicion:

1. The upstream vLLM model adaptation layer did not correctly connect the weights to the current MoE contract;
2. The current baseline expert weight layout is inconsistent with the model requirements.

Don't change the backend at first reaction.

## 10. Shared Expert / Mix Placement / Dynamic EPLB

### 10.1 Shared expert

From the logic of [fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py:157) and subsequent `forward_impl`, it can be seen that the current implementation has considered:

- `n_shared_experts`
- shared expert independent forward
- Parallel computing of shared expert and routed expert
- Shared expert coordinates the reduce behavior of routed output

Therefore, shared expert is not a completely blank capability.

But please note:

- The current shared expert is built on the existing `AscendFusedMoE` organization.
- If the shared expert of the new model is residual MLP, series MoE, or the weight sharing method is special, gap analysis still needs to be done separately.

### 10.2 Mix placement

The router stage already supports `mix_placement`:

- Done by splicing shared expert entries to `topk_ids/topk_weights`

This shows that the mixed placement of "part expert routed, part expert resident" is not completely incapable.

### 10.3 Dynamic EPLB

Currently MoE supports dynamic expert load balance:

- Initialize `eplb_config`
-Update `moe_load` after forward
- routing / mlp paths all carry the `dynamic_eplb` parameter

Therefore, if the new model MoE has expert remap / redundant expert / load balance metadata, you must first check whether it can be mapped to:

- `expert_map`
- `global_redundant_expert_num`
- `log2phy`

Rather than directly thinking "expert balance is not supported".

## 11. Runtime contract

The current fused MoE runtime contract has been unified in [moe_runtime_args.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_runtime_args.py:17).

When adapting, you should pay most attention to these objects:

### 11.1 Routing contract

- `expert_map`
- `global_redundant_expert_num`
- `mc2_mask`
- `apply_router_weight_on_input`
- `log2phy`
- `pertoken_scale`

### 11.2 Weight contract

- `w1`
- `w2`
- `w1_bias`
- `w2_bias`
- `w1_scale`
- `w2_scale`
- `w1_scale_bias`
- `w2_scale_bias`
- `w1_offset`
- `w2_offset`

### 11.3 Quant contract

- `quant_type`
- `comm_quant_mode`
- MXFP parameters
- `is_per_channel_weight`

When the skill analyzes the MoE layer of the new model, it should try to translate the model requirements into this set of contracts to see which fields have been mapped and which fields have no source.

## 12. Coupling points with model runner

MoE support is not limited to the `fused_moe/` directory.

### 12.1 MoE detection

The auxiliary logic near [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:203) indicates that the framework will recursively determine whether the model has expert through the config content.

This means that if the MoE description of the new model config is very special, the first level problem may be "not being correctly determined as a MoE model".

### 12.2 DP allreduce skip logic

`should_skip_allreduce_across_dp_group(...)` will be based on:

- Is it MoE?
- Whether to enable hierarchy comm
- Whether it is a draft model

to determine the behavior in the DP dimension.

This directly affects the `global_bs` / `mc2_mask` branches of MC2 token dispatch.

### 12.3 model runner side behavior

[model_runner_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:568) is followed by this logical explanation:

-Token quantity needs to be aligned across DP
- spec decode / graph capture / mixed batch will affect metadata
- The MoE runtime also relies on input token alignment and extra fields in the forward context

In addition, the current implementation also handles routed experts capture separately, which shows that Ascend MoE has deviated from the upstream default hook.

## 13. Typical scenarios currently covered

Based on the existing code, it can be considered that these scenarios are currently covered or partially covered:

1. decoder-only routed MoE LLM
2. top-k router
3. grouped top-k
4. `softmax` / `sigmoid` / `sqrtsoftplus`
5. `e_score_correction_bias`
6. hash routing specialized path
7. shared expert
8. mix placement
9. dynamic EPLB
10. `ALLGATHER` / `ALLTOALL` / `MC2` / `FUSED_MC2`
11. EP path
12. W8A8 / W4A8 / MXFP8 / MXFP4 MoE quantization path

But "coverage" does not mean "directly compatible with any MoE architecture." The current coverage is more precisely:

**Ascend MoE implementation under a specific set of structures and specific contracts has been covered. **

## 14. Current high-risk points of differentiation

If the new model hits the following differences, MoE gap analysis is usually given priority:

1. router is not standard top-k
2. Grouped top-k rules are different
3. router requires custom routing function
4. Expert is not a two-stage form `gate_up + swiglu + down`
5. Gate/up weights are not in merged form
6. Shared expert has different structures
7. The weight layout of checkpoint expert is inconsistent with the existing loader assumptions.
8. Rely on new communication metadata
9. Rely on the new comm quant mode
10. The runtime token organization method is inconsistent with the current prepare/finalize assumption

## 15. Prioritization during adaptation

For the new MoE model, it is recommended to judge in this order:

1. `config.json` Whether to explicitly expose the expert/router/shared-expert structure.
2. Whether the modeling code can still be mapped to the current `router + experts + shared expert` three-segment organization.
3. Whether expert can be mapped to `w13/w2 + swiglu` contract.
4. Whether router can be mapped to the current `select_experts(...)` capability set.
5. Should the model go through `ALLGATHER` baseline verification first instead of directly pursuing MC2/FusedMC2.
6. Whether the quant checkpoint can be mapped to the current quant contract.
7. If all the above are true, then determine whether the EP / MC2 / fused performance path requires supplementary adaptation.

## 16. Skill usage requirements

When a skill adapts to `moe llm`, this document should be regarded as the current MoE capability baseline, and after reading the new model's `config.json`, modeling code, weight keys, and operating phenomena, write a fixed-format section:

```markdown
## MoE Gap Analysis

### 1. Current Capability
- Router capability baseline:
- Expert MLP baseline:
- Shared expert baseline:
- Communication baseline:
- Quantization baseline:

### 2. Model Requirement
- Router/gate behavior:
- Expert structure:
- Shared expert / residual MLP behavior:
- EP/TP/dispatch expectations:
- Quant / weight-layout requirements:

### 3. Gap
- Router gap:
- Expert-structure gap:
- Weight-layout gap:
- Communication/runtime-contract gap:
- Unknowns to verify:

### 4. Adaptation Plan
- Fix location:
- Minimal files to touch:
- First validation path:
- Stop / escalate condition:
```

The purpose of this section is to first determine:

- Whether the new model MoE layer can actually be connected to the existing path;
- It is still necessary to change the model adapter / weight loader of upstream vLLM;
- It is still necessary to change the vLLM frame wiring;
- Or Ascend backend itself has no corresponding ability.

## 17. Direct guidance conclusion for skill

For skills, the following judgment criteria should be used currently:

1. Treat `fused_moe` as an **existing capability baseline**, not a blank area.
2. New model MoE adaptation defaults to comparison first:
- router rules
- expert MLP structure
- shared expert structure
   - weight layout
   - EP/MC2/runtime contract
3. Prioritize classifying problems into:
- Model modeling/registration issues
- Weight mapping problem
- Existing MoE contract wiring issues
- backend capability gap
4. When bringing-up, give priority to connecting the `ALLGATHER` baseline path first, and then decide whether to pursue EP/MC2/FusedMC2.

If the MoE layer of a new model can be mapped to the current baseline, the adaptation focus is usually on the model wiring and weight mapping on the upstream vLLM side, rather than writing the Ascend MoE backend from scratch.
