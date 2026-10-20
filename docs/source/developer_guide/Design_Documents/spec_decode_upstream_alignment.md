# Speculative Decoding Upstream Alignment

## Purpose

vLLM-Ascend should use vLLM as the owner of the generic speculative-decoding
control flow. Ascend code should add platform behavior only where the generic
path cannot express NPU requirements.

This document records the audited ownership boundary and the regression gates
that keep the two runners aligned. It is based on:

- vLLM-Ascend `main` at `128f35ed1461ecb34bc8b87583a67a66bc4bd1d6`
- vLLM `main` at `a98247ab4db686ee03c66d5feb3c761e52a2f8ab`

The revisions must be refreshed whenever the supported vLLM baseline changes.

## Unification Contract

The default rule is:

1. vLLM owns draft input preparation, multi-step draft iteration, sampling,
   rejection or verification, accepted-token accounting, and output restore.
2. vLLM-Ascend subclasses or adapts those paths before or after `super()`.
3. Ascend-specific attention metadata, NPU buffers, ACL graph capture,
   PCP/DCP layout, and NPU operators remain in vLLM-Ascend behind narrow hooks.
4. A platform patch is retained only while no accepted upstream extension point
   can express the same behavior.
5. A new speculative family is added to the shared selector before either
   runner grows method-specific dispatch.

The two runners must select the same speculative method family even when their
implementation classes differ. `vllm_ascend.spec_decode.method` owns that
runner-independent classification. MRV1 dispatch, MRV2 dispatch, the 310P MRV2
router, and the MRV2 readiness gate all consume that result.

## Current Boundary

| Path | Shared or upstream owner | Ascend-specific boundary | Current action |
| --- | --- | --- | --- |
| Family selection | `vllm_ascend.spec_decode.method` | N/A | Shared by MRV1, MRV2, 310P routing, and MRV2 readiness checks. |
| MRV1 proposer lifecycle | `vllm/v1/spec_decode/*` | NPU model loading, attention metadata, ACL graph buffers, PCP/DCP coordination | Generic proposer and sampling methods delegate to upstream where their interfaces match. |
| MRV1 DFlash | Upstream DFlash proposer | NPU copy/expand kernel and persistent slot-mapping buffers | Model-specific input preparation stays behind narrow overrides. |
| MRV1 DSpark | Upstream DSpark proposer interfaces | NPU attention metadata, multi-group KV layout, and graph updates | Sampling and sequential draft generation remain upstream-owned. |
| MRV2 autoregressive | Upstream `TargetDependentARSpeculator` propose loop | Ascend metadata construction, replicated PCP, graph capture, and hidden-state broadcast | Overrides wrap upstream calls instead of reimplementing the control flow. |
| MRV2 DSpark | Upstream DSpark sampling and sequential decode | Ascend model loading, LM-head TP alignment, attention metadata, and graph handling | Draft sampling remains inherited from upstream. |
| Rejection sampling | Upstream rejection-sampler host orchestration and portable helpers | Ascend Triton kernel and NPU launch constraints | Retain only the kernel implementation that cannot run upstream on NPU. |
| KV cache metadata | vLLM block-table and slot-mapping interfaces | PCP/DCP ownership mapping, ACL graph padding, and reusable NPU buffers | MTP, Step3.5, and multi-KV-group paths share one slot-mapping write helper. |
| Method patches | vLLM runner modules | `vllm_ascend/patch/worker/patch_v2/*` | Retain only while the verified upstream revision lacks an extension point. |

The slot-mapping write contract is centralized in
`vllm_ascend.spec_decode.utils.copy_slot_mapping_to_padded_buffer`. It owns
dtype conversion, source aliasing, and graph-padding sentinel values for MTP,
Step3.5 MTP, and multi-KV-group MTP.

## NPU-Only Adaptations

The following deviations are intentional and must remain covered by tests:

- Ascend attention metadata builders and attention-state selection.
- ACL graph capture, graph parameter updates, padding, and replay.
- PCP/DCP layout conversion and context-parallel request staging.
- NPU custom operators and Triton kernels that upstream cannot lower on Ascend.
- LM-head tensor-parallel row alignment where Ascend uses fixed-size buffers.
- Model-family compatibility shims that have no generic upstream equivalent.

An adaptation must be narrow. If it changes draft input preparation, sampling,
rejection, or accepted-token accounting for every platform, it belongs upstream
instead of in a subclass or patch.

## Validation

Focused CPU tests verify family selection, MTP precedence, agreement across all
dispatch and readiness entry points, slot-mapping behavior, inheritance, hook
invocation, and delegation to upstream-owned sampling and processor paths.

NPU validation should cover the affected execution mode:

- eager, PIECEWISE, and FULL graph capture and replay where supported;
- MTP and EAGLE3 for MRV2 changes;
- DFlash, DSpark, or the affected draft family when its path changes;
- applicable TP, DP, PP, PCP, and DCP combinations;
- output correctness and draft acceptance against the pre-change baseline;
- no material latency or throughput regression.

Current hardware verification uses an Ascend A3 device and Qwen3-0.6B:

- MRV1 NGram speculative decoding completed generation on NPU.
- MRV2 `extract_hidden_states` completed generation through the upstream
  speculator on NPU.
- The Ascend slot-mapping helper was exercised with NPU tensors.

Model-backed EAGLE3, MTP, DFlash, and DSpark end-to-end runs require their draft
checkpoints and remain family-specific CI coverage rather than alternate control
flow implementations. CPU-only unit tests are not acceptance evidence for NPU
attention layouts or graph replay.
