# glm5next_kda_pcp_layer

## Description

- **Function**: An opaque dispatch boundary around the existing GLM KDA layer when PCP is enabled. It reads current request metadata at execution time so a metadata-free warmup cannot freeze prefill gather/restore decisions in a Dynamo graph. It adds no Triton kernel or new KDA arithmetic.
- **Formula**: For a PCP prefill, `Y_local = restore(F_theta(gather_causal(X_local), positions, state))`, where `F_theta` is the existing GLM KDA projection, convolution, recurrent update and output projection. Other invocations delegate to the existing layer implementation.
- **Algorithm flow**: Resolve `layer_name` in the current forward context; inspect current PCP metadata; gather and restore causal token order when required; invoke the existing KDA layer; restore rank-local output order.
- **Supported modes**: A3 verified with BF16 model activations and convolution state, FP32 recurrent state, MRV2, and DS convolution-state layout. A2 and 950PR/950DT: not validated in this change.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| hidden_states | Input | Rank-local activations, shape `[T_local, hidden_size]`; not modified in place | BF16 on A3; FP32 in CPU dispatch regression | Model-provided two-dimensional tensor |
| positions | Input | Model position argument forwarded unchanged | Model-provided integer tensor | Model runner layout |
| layer_name | Attribute | Stable registered layer key in the current forward context | Python string | N/A |
| output | Output | Projected activations in rank-local token order | Same model activation dtype | `[T_local, hidden_size]` |
| layer KV state | Input/Output, implicit context | Existing convolution/recurrent checkpoints and, for RecoverSSM, local verify records | BF16 convolution and FP32 recurrent state in validated setup | Existing paged model cache |

## Constraints

1. The layer must already exist in `forward_context.no_compile_layers`. PCP rank/group, weights, head partition and cache geometry must agree with its model configuration.
2. The existing GLM head partition requires divisible head counts. TP=1 shards KDA heads over PCP; TP>1 shards them over TP.
3. The fake implementation returns only output metadata; it does not execute collectives or update state. Real warmup delegates to the existing metadata-free KDA path, which zeroes its attention output.
4. Explicit inputs are not silently cast or mutated. Existing KDA cache updates remain implicit model state side effects. No packing buffers or NPU-to-host reads are introduced by this boundary.
5. This boundary does not add RecoverSSM support for a PCP consumer. D retains PP=1/PCP=1. Full validation used native P TP1/PCP8 and RecoverSSM D TP8; a TP4/PCP2 concurrency run encountered an AllGather timeout.

## Origin and Differences

- **Origin**: `vllm_ascend/models/glm5next/kda.py::Glm5NextLinearAttention`.
- **Differences**: Move the complete PCP layer invocation behind an upstream custom-op registration, retaining the existing `causal_conv1d`, `chunk_kda`, `recurrent_kda`, and RecoverSSM verify implementations. No launch tile, Triton constexpr, or kernel arithmetic is changed. No new environment variable is introduced.
- **Lifecycle**: Keep the boundary until an upstream request-metadata mechanism safely preserves these runtime decisions through warmup compilation. Replace it with that upstream mechanism after equivalent warmup/prefill/decode and model-level validation.

## Test Cases

1. `tests/ut/models/test_glm5next_kda_pcp.py::test_compiled_pcp_layer_reads_request_context_after_warmup` uses one compiled CPU graph across warmup, prefill and decode contexts. This tests dispatch behavior, not a new mathematical kernel.
2. `tests/ut/kv_offload/mooncake_v2/test_kda_transfer.py` independently checks global head coordinates and exact byte slices for DS/SD layouts, differing strides, native/RecoverSSM modes and invalid contracts.
3. A3 transfer validation reads complete producer states, computes CPU global-head slices independently, and compares them exactly with D state. It also checks untouched verify scratch; this is a pure-copy reference, not a new mathematical kernel test.
4. Model validation uses fixed GSM8K requests and detailed genuine speculative-decoding counters. Configurations, source revisions and observed results are recorded in [the PR experiment comment](https://github.com/vllm-project/vllm-ascend/pull/20453#issuecomment-6541117335).
5. N/A: a new single-card Triton numerical test, because no Triton arithmetic/kernel is added. Existing numerical kernel tests remain applicable. An isolated performance A/B and an all-kernel dataset compilation-identity trace have not been collected; this change does not claim those merge-readiness requirements are satisfied.

## Recompilation and side effects

Tensor shape/dtype/stride and stable `layer_name` can participate in compiler guards. PCP/TP topology and weight shapes are fixed per model instance; runtime prefill/decode metadata is read inside the opaque call. Request-dependent Python branches are no longer traced into the outer projection graph. Existing delegated kernels retain their own specialization behavior; no claim of zero dataset recompilation is made without a complete trace.

Custom-op registration is process-scoped and uses the existing upstream registration helper. Cache mutation is the existing layer behavior, ordered through the output dependency. The connector independently transfers committed convolution/recurrent bytes and leaves D-local verify scratch untouched. Temporary audit copies and shape logging used for validation are not included in the product patch.
