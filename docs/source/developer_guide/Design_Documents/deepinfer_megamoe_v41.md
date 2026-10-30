# DeepInfer MegaMoE for DeepSeek V4.1 on Ascend 950

This experimental backend runs DeepInfer's NPU `fp8_fp4_mega_moe` kernel,
including FP8 shared experts, through the Ascend MoE runner. It is opt-in:

```json
{"deepgemm_mega_moe": true, "enable_fused_mc2": 0}
```

The supported experiment is DeepSeek V4.1, BF16 activations, MXFP4 routed
weights, FP8 shared weights, TP1 and linear EP8 without EPLB replicas.
The target must load before its DSpark draft. Target and draft execute
serially and reuse symmetric storage. This implementation does not support
concurrent microbatches or weight hot reload; disable it for those workflows.
Default MoE behavior is unchanged when the option is false.

## Relationship to vLLM's DeepGEMM backend

The reference is vLLM's `DeepGemmMegaMoEExperts` in
`vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py`, reviewed at
`f81a414faed63b576768487236f98ea5e655fc57`.
The V4.1 NVIDIA model reuses the V4 MoE implementation.

| Concern | Common contract | Ascend implementation and remaining difference |
| --- | --- | --- |
| Execution | One fused dispatch, expert GEMM, activation and combine launch on local tokens | Keep the existing Ascend runner and its TP1 prepare/finalize interface; bypass ordinary expert dispatch/combine. |
| Shared experts | Kernel receives shared weights and returns routed plus shared output exactly once | Keep checkpoint registration in the existing shared module, hold a weak reference, and transfer execution ownership before post-load conversion. |
| Routing | Global expert IDs; routing scale applies only to routed experts | Apply the model's routed factor to top-k weights before the native kernel; suppress the runner's post-output scaling. |
| Weights | Kernel-specific layout and packed scales | Use the NPU dependency's public `transform_weights_for_mega_moe` once after loading. The reviewed CUDA implementation loads shards directly into kernel layout; direct-layout loading is a follow-up. |
| Scratch | Cache symmetric buffers by EP group and shape | Group-owned cache; use NPU `SymmBuffer(base=...)` for target/draft sharing. The reviewed CUDA allocator API differs. |
| Input preparation | Quantize local BF16 tokens and populate symmetric routing inputs | NPU dynamic MX quantization plus copies. CUDA's fused direct-to-buffer preparation is a remaining optimization, not implemented here. |
| Backend selection | Explicit opt-in, unsupported configurations rejected | Ascend `additional_config` selects this adapter because the upstream CUDA selector constructs a GPU-specific expert implementation. |
| Stream handling | Preserve execution ordering and buffer lifetime | An Ascend-only large-Prefill fence bounds deferred allocator reuse; this is not copied from CUDA and needs separate review. |

The adapter deliberately retains these Ascend boundaries rather than claiming
that the CUDA class can execute unchanged on NPU. No vLLM source patch is needed.

## Dependencies and reproduction scope

Hardware results used the NPU DeepGEMM source at
`3a35326271b069b8b5bf1a754435acd6cc91575b`, CANN 9.2 and
`torch_npu 2.10.0.post7`. Install a compatible NPU build of `deep_gemm` before
enabling the backend; the CUDA package is not interchangeable.

The tested CANN 9.2 Bisheng device frontend needed the included
`ops/fused_moe/deepgemm_compat/utility` header shim for libstdc++ compatibility.
For that toolchain, prepend its directory to `CPLUS_INCLUDE_PATH` in the launch
environment. The shim changes device compilation headers, not Python package
installation, and is not required or injected for other backends.

## Prefill memory mitigation

In the tested configuration, max sequences 64 and DSpark 5 produce at most
384 Decode tokens. The adapter synchronizes the current NPU stream before
main-model MoE when the global token count exceeds 512, outside profiling,
graph capture/replay and draft execution. All DP ranks use the global count,
including ranks with no local tokens. The threshold is specific to this
experiment; it is not a general Prefill/Decode classifier.

Without the fence, long Prefill accumulated 10–13 GiB per rank of
`active_pending_free` allocator blocks and repeatedly retried allocation.
The fence allows subsequent allocations to recycle those blocks. Reserved
allocator memory can remain high while reusable; no `empty_cache()` is used.
This is a measured conservative mitigation. Allocation-stack capture and
comparison with native DeepInfer are still being investigated.

## Validation available for review

Historical hardware validation used vLLM
`c4df37dfcc7f72ca41cd43801a468526ecf99a2b` and Ascend base
`9f1f27d79340e2d3a7ab8f16f0771473c65773e3`, with eight Ascend 950 cards,
DP8/TP1/EP8, maximum batch tokens 4096, maximum sequences 64,
utilization 0.85, FULL_DECODE_ONLY, DSpark 5, and HCCL buffer 1600 MiB.
Bounded replay and CANN fused MC2 were disabled.

Each throughput result is the mean of two native serving runs, concurrency
64 and 256 requests. Prefix caching remained enabled with fresh salts and
zero measured prefix hits.

| Workload | Ordinary MoE | DeepInfer fused shared experts | Change |
| --- | ---: | ---: | ---: |
| Decode, input 128/output 2048 | 5045.05 output tokens/s | 5736.37 output tokens/s | +13.70% |
| Prefill, input 8192/output 1 | 44207.86 input tokens/s | 56979.66 input tokens/s | +28.89% |

The Prefill result includes the fence: zero allocator retries, OOMs and
preemptions across eight ranks, 512 successful benchmark requests and six
passing generation smoke checks. The Decode result predates the fence and
must not be represented as a rerun of this PR. Smoke checks are not a full
model accuracy evaluation.

This PR ports the hooks to current main's unified MXFP4 method. These hardware
numbers describe the fixed-commit implementation, not a completed current-main
E2E test. Current-main hardware/accuracy and performance regression validation
remain required before merge.
