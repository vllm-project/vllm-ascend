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
The ordinary MoE backend remains selected when the option is false.
The MLA lifetime adjustment below also applies with ordinary MoE selected.

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
| Stream handling | Preserve execution ordering and buffer lifetime | Keep native stream dependencies; the separate MLA lifetime adjustment retires redundant same-stream registrations without a per-layer host synchronization. |

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

## MLA lifetime adjustment

The measured CannBotDSL submission bridge calls `record_stream` on native-call
arguments, including tensors allocated on the same stream. The tested
`torch_npu 2.10.0.post7` allocator defers their reuse until events are processed.
Stack traces identified Q projection and MLA output allocations of about
255.75 MiB each accumulating across twelve layers. These tensors had already
reached `free_requested`; retained Python references did not explain these blocks.

After native submission, the MLA wrapper records and waits on a device event,
then calls `torch_npu.npu.utils._erase_stream` for Q and the output. It does not
wait for device completion on the CPU. The previous token-threshold-based
Prefill host synchronization is removed from the MegaMoE adapter.

Safety depends on same-stream ordering, not on the Python call having returned.
In the inspected allocator revision
`3363ea59d49f3e08cfd8d4a1f532b756ccda0682`, `_erase_stream` refuses to erase when
the current stream differs from the allocation stream and retains registrations
on other streams. Registration is storage-level, not a token owned only by this
DSL call. A self-stream event wait does not establish dependencies on an unknown
side stream. The native operator must honor its supplied-stream contract.

The adjustment is skipped during graph capture. It also affects non-captured
MLA calls when MegaMoE is disabled; there is no Prefill phase or token-count gate.
This broader scope and the private allocator API require review. Skipping graph
capture alone does not prove that every Decode execution path is unaffected.

Small NPU tests confirmed that event-only handling still left pending blocks,
whereas event plus erase allowed same-stream reuse. Cross-allocation-stream
protection and other-stream registrations were preserved in the tested cases.
Removing the event itself has not been validated. Storage-alias, long-running
multistream and current-main graph regressions remain pending. A preferable
long-term location may be allocator same-stream registration handling rather
than compensation in each operator wrapper. No allocator or CannBotDSL package
modification is included here.

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

The updated lifetime handling was tested separately on the same fixed commits,
with the host fence removed and allocation stacks enabled. Eight complete
40-layer iterations reduced maximum requested pending bytes from 6.266 GiB to
0.680 GiB; the other seven iterations were approximately 0.45 GiB. Allocator
retries across those iterations dropped from four to zero. All 256 requests
(input8192/output1, concurrency64) completed, and six generation smoke checks
passed. These are instrumented diagnostics, not a formal throughput measurement
or a full accuracy evaluation. The historical throughput table above must not
be attributed to the new lifetime adjustment.

Clean formal Prefill/Decode replay remains pending: the follow-up service failed
its startup free-memory check before model execution. The main-branch unit tests
cover capture bypass, empty inputs, output preservation, and absence of the old
host fence; mocked tests do not establish the NPU lifetime contract.
