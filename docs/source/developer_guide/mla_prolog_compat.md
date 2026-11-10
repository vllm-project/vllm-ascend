# Packaged A5 NoPE MLA Prolog

GLM5.3-Flash NoPE MLA Prolog follows the A5 DeepSeek V4.1 packaging pattern:
device-adaptor selection, a Torch schema, fake outputs, PrivateUse1 dispatch and
a cached lazy external import. The scope differs: GLM fuses preprocessing and
writes KV cache; DeepSeek's packaged indexer/attention consumes prepared caches.
This is framework integration, **not a quantified speedup claim**. Validation
reports and evidence limitations are recorded below.

## Diagnostic branch: native prefill, CANNBot decode

`glm-cannbot-decode-only` is based on `5af1f31`. No new environment variable or
serve argument is needed: retain the existing launch command, including
`FULL_DECODE_ONLY`. For selected CANNBot NoPE layers, only metadata classified
as `DecodeOnly` with `num_prefills == 0` uses Prolog. Prefill, cache-hit prefill,
chunked/mixed and other non-decode states use the existing native chain for the
whole batch. Other backends and the legacy MLAPO token limit are unchanged.

This is a pre-execution route choice, not an exception fallback. Original
projection/norm parameters and prepared CANNBot weights remain loaded, and both
routes use the same KV cache. Native prefill avoids the CANNBot entry entirely.
Decode with a BF16 Indexer recomputes its query using the original A projection
and RMSNorm; no Prolog-output MXFP8-to-BF16 converter remains. Weight memory is
not restored to the all-native baseline. The public Prolog package is still
required for decode.

**Classification caveat:** MRV2 may classify prefix-cache-hit prefill with only
one uncached token per request as `DecodeOnly`; if `num_prefills` is also zero,
it retains Prolog, including inside graph replay. This branch follows attention
metadata, not a new scheduler-level prefill flag. It does not change the runner
or add device-to-host reads.

Compare against `da15e74` to isolate BF16 Indexer recomputation from the prior
input-conversion implementation; both use the decode-only policy. Keep workload,
warmup, prefix-cache hit rate and graph settings identical. Check startup for
`Cannbot MLA Prolog diagnostic policy`.
The policy and CPU contracts do not establish NPU correctness or performance:
validate native-prefill to Prolog-decode transitions, BF16/C8 cache consistency,
mixed batches and padded graph decode on the target system before relying on
the TTFT result.

## Artifact installation: pending

Target: **Linux aarch64 / Python 3.12 (cp312), A5**. Matching operator and
CANNBotDSL wheels are internal artifacts and cannot currently be exported.
Installation is deferred: this change does not invent a URL, rename an x86 wheel,
install a substitute, or change existing DeepSeek Docker dependencies. The old
DeepSeek wheel does not establish that the GLM entry is present. Before deployment,
obtain both matching aarch64 wheels and record actual filenames, SHA256 values,
CANN/torch_npu compatibility and AOT specializations.

The earlier reviewed **x86** quantized distribution was
`cannbot-arena-net==0.3.0` with `cannbot-dsl>=0.7.0,<0.8`; its metadata recorded
DSL 0.7.1 / CANN 9.2.0-beta.2. These are interface research provenance, **not an
ARM64 version lock or artifact hash**. Confirm that the supplied ARM64 wheel
exposes the same public `ops.quant_mla_prolog.quant_mla_prolog` signature and
capture-aware behavior. A matching package version alone is insufficient.

## Automatic selection

The base device adaptor returns torch_npu; the A5 adaptor selects Cannbot for
NoPE layers with MXFP8 fused QKV projection. There is no environment-variable
opt-in. RoPE layers, non-A5 devices and BF16/INT8 NoPE projections retain their
previous route. Once selected, incompatible shapes, projection methods and
cache/parallel settings fail before execution. A missing wheel fails at first
real invocation; it never silently retries native preprocessing. Preparation and
fake tracing do not load the vendor compiler.

| Property | Boundary |
| --- | --- |
| Hardware/artifact | A5 FP8-optimized profile; Linux aarch64, CPython 3.12 |
| Execution integration | Native prefill and Prolog decode; detailed hardware evidence not attached |
| Activations | BF16 before existing dynamic MXFP8 input quantization |
| Prolog projections | MXFP8, group size 32, scale algorithm 0; no bias or TP input padding |
| KPool Indexer query projection | MXFP8 as above, or the unquantized Ascend linear method with BF16 weights; no bias or TP input padding |
| Dimensions | NoPE (Dn,Dr)=(256,0), KV rank 512; hidden size 1024..8192 aligned to 256; Q rank 640/1024/1536/2048; local heads 1..128 |
| Normalization | Equal Q/KV epsilon; independent FP32 gamma snapshots |
| Main cache | Contiguous PA_BSND BF16, or FP8 E4M3 C8 with 128-element groups and 528-byte slots |
| Page size | 16..1024, aligned to 16, subject to attention backend constraints |
| Query / query norm | BF16 query; MXFP8 Indexer uses E4M3/E8M0 norm tuple; BF16 Indexer uses a natively recomputed BF16 norm |

RL weight reloads, PD/KV transfer, speculative decoding, context/sequence
parallelism, sparse KV offload, INT8 C8 and TurboQuant remain outside this
integration's validated interface scope. They are not universally forbidden by
DSL; they require separate state/cache/routing validation. No full-Q quantization
or HiF8 path is introduced. Shape admission does not prove AOT kernel coverage.

## Dispatcher and cache contract

[packaged_mla_prolog.py](https://github.com/vllm-project/vllm-ascend/blob/93e3048c969a9f216a4899adcffce2eb58b2db6f/vllm_ascend/ops/packaged_mla_prolog.py) defines
`vllm_ascend::quant_mla_prolog`. Prepared weights/scales/gammas are explicit tensor
arguments; no Python callable or layer state is captured in the schema. Its four
outputs follow the vendor convention. MX scales cross the dispatcher as zero-copy
uint8 views, converted to E8M0 inside the NPU implementation and after return.
This avoids reintroducing the vendor's documented E8M0 functionalization issue at
the outer boundary. Fake outputs match the byte-carrier contract and support
symbolic token counts without loading DSL.

**KV cache is declared `Tensor(a!)`.** The original cache crosses the mutable
boundary. Backend rank/dtype views preserve storage and offsets; noncontiguous
or non-ND cache is rejected, never copied. Functionalization that would clone
and copy back persistent cache is explicitly rejected. Failures after launch
propagate; there is no native retry on potentially modified cache.

The [central adapter](https://github.com/vllm-project/vllm-ascend/blob/93e3048c969a9f216a4899adcffce2eb58b2db6f/vllm_ascend/ops/mla_prolog.py) retains preparation
and the canonical five outputs:
`(query, query_rope, query_scale, query_norm, query_norm_scale)`.
Cannbot `qr` is normalized low-rank Query, not RoPE Query; the adapter supplies
empty BF16 `[T,N,0]` RoPE output. The [KPool Indexer](https://github.com/vllm-project/vllm-ascend/blob/93e3048c969a9f216a4899adcffce2eb58b2db6f/vllm_ascend/attention/indexer_kpool.py)
passes E4M3/E8M0 tuples directly to MXFP8 query projections. For an unquantized
BF16 query projection, SFA disables Prolog query-norm output and recomputes
`q_a_layernorm(fused_qkv_a_proj(hidden_states)[0][..., :q_lora_rank])`, following
the native input path used by PR #20332. This keeps the original MXFP8 A
projection and BF16 normalization semantics; it does not dequantize a Prolog
output, change weights, or rerun any cache-writing operation. The earlier
integer input converter is removed. Only CANNBot with a BF16 KPool Indexer
receives this exception; other PROLOG_V3 contracts remain unchanged.

Recomputation occurs only inside the selected decode Prolog path and when
Indexer query work is needed. Physical-empty batches, disabled runtime Indexers,
`skip_topk`, and zero-local-query batches skip it without disabling existing
Indexer key/cache writes. FULL-graph padding is preserved, not trimmed to the
actual-token count. Prefill/mixed batches already compute the native norm and
never execute this extra path.

The extra A projection and RMSNorm have real compute/memory costs; the source
parameters remain resident. Validate top-k, logits, graph replay and end-to-end
latency rather than assuming a speedup. In MLA layers with Q LoRA, SFA `q_proj` refers to the model
`q_b_proj`; it is separate from `indexer.wq_b`, which is not a Prolog input.

Loaded weights/scales are
transposed into independent ND snapshots once; original parameters are unchanged.
Extra resident memory and reload limitations remain.

## Graph integration is not hardware qualification

The reviewed public vendor entry detects NPU stream capture and selects its
internal graph operator. Only its ordinary checked eager branch reads indices
back to CPU. Consequently there is **no blanket enforce-eager gate** and no
private/native vendor call. The public API is used in eager and capture execution.
Ordinary compiled execution outside capture may still take the checked branch;
no host-sync-free claim is made.

The existing [MLA boundary](https://github.com/vllm-project/vllm-ascend/blob/93e3048c969a9f216a4899adcffce2eb58b2db6f/vllm_ascend/ops/mla.py) already calls
`torch.ops.vllm.mla_forward`; its fake implementation does not enter SFA, and
its runtime implementation resolves persistent cache from the forward context
before calling SFA. The packaged Prolog executes inside that opaque boundary.
Platform configuration also lists `vllm::mla_forward` among attention split ops.
This preserves the existing model/runner graph structure without duplicating it
or adding an inert model flag. Ordinary compilation around this opaque attention
boundary is distinct from directly functionalizing the packaged cache-writing op;
only the latter is explicitly rejected. Do not infer a blanket ban on piecewise
model compilation from that guard. Fake registration alone does not establish
ACLGraph correctness or prove the absence of every cache-copy transformation.

Before graph deployment, warm required specializations outside capture and
verify the **actual ARM64 public API** is capture-aware. Every nonnegative slot
must be in range and unique on **each invocation and replay**; negative padding
slots are ignored. The scheduler/slot-mapping producer owns those invariants,
as for native cache writes. One warmup check does not validate changed replay
slots. Never add CPU slot checks inside capture or bypass the public API to make
an incompatible wheel appear graph-capable.

## Validation

CPU contract tests (from the repository root):

```bash
python -m pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_mla_prolog_adapter.py tests/ut/ops/test_packaged_mla_prolog.py
python -m pytest -q --confcutdir=tests/ut/attention tests/ut/attention/test_mla_prolog_compat.py
```

These use real CPU/meta/fake tensors and test-only CPU dispatch registrations.
They cover mutation schema, lazy imports, symbolic outputs, byte reinterpretation,
cache aliasing, failure propagation and a torch.compile backend=eager side-effect
smoke test. SFA/KPool tests extract actual methods with mocked framework
dependencies; they do not import the full vLLM runtime. CPU tests are not actual
vendor execution, aarch64/cp312 validation or NPU numerical evidence. Run normal
repository UT/CI in the supported environment before merge as well.

Local validation on Windows x86_64 / Python 3.14.7 / Torch 2.14.0+cpu:
78 operator/dispatcher tests and 286 SFA/KPool/opaque-boundary tests passed,
including native-prefill/mixed routing, BF16 query recomputation, unchanged
MXFP8 tuples, source-weight preservation and skipped/empty/padded queries. This
host test environment deliberately does not stand in for the deployment target
(aarch64/Python 3.12) or its pinned Torch/CANN stack. Ruff and whitespace checks
were run on changed files; full repository pre-commit/NPU CI was not run.

After integrating upstream main at `3725f6b`, CPU comparisons found identical
results for 64 standard-provider routing cases and identical torch_npu argument
values/storage and returned data in six direct legacy Prolog cases. Upstream
MLAPO/PCP helpers and Indexer metadata/cache methods were also checked for AST
equality. Query TP row sharding and FP8 LI cache forwarding are covered by the
expanded contracts. These checks do not establish A3 NPU numerical or performance
parity; supported-environment runtime tests and hardware validation remain needed.

The author reported successful validation of runtime revision `6ade10b` on
2026-11-09, before this upstream integration. Exact run
commands/logs, graph-mode coverage, numerical results and performance numbers
were not supplied with that confirmation. This report is separate from the
local CPU tests above and does not establish every item in the following
hardware-review checklist:

1. Record exact aarch64/cp312 wheel hashes and runtime revisions; verify public
   capture dispatch and AOT/JIT warmup coverage.
2. Compare native vs fused query, dequantized query-norm, cache bytes and embedded
   FP32 scales; agree tolerances, then check KPool top-k, logits and model accuracy.
3. Capture/replay with changing inputs **and slots**, page boundaries, negative
   padding, both cache modes and query-norm enabled/disabled. Verify original
   cache identity, untouched slots and absence of stale warmup indices.
4. Verify no CPU index readback in capture, no persistent-cache clone/copy-back,
   and correct eager prefill/full-graph decode routing in the real model runner.
5. Measure extra prepared-weight memory and performance separately. FP32 fused
   versus native BF16 rounding differs; matching cache layout does not establish
   numerical equivalence or speedup.

## Release note

[PR #21302](https://github.com/vllm-project/vllm-ascend/pull/21302) carries the
[numbered release-note fragment](https://github.com/vllm-project/vllm-ascend/blob/93e3048c969a9f216a4899adcffce2eb58b2db6f/release_notes/fragments/21302.yaml).
Scope of this integration:

- Config: A5 NoPE MXFP8 MLA Prolog selects packaged CANNBot operators through the
  device adaptor rather than an environment opt-in.
- Feature: Added lazy dispatcher/fake contracts with explicit KV-cache mutation
  and eager/full-graph routing. Matching Linux aarch64/Python 3.12 artifacts and
  hardware qualification remain required before deployment.
- Fix: Replace the BF16 KPool input converter with native A-projection and
  RMSNorm recomputation during CANNBot decode, without changing Indexer weights
  or retrying cache writes. Prefill/mixed batches remain native. Detailed
  numerical and performance evidence is not attached to the author report.
