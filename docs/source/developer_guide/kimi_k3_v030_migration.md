# Kimi K3 changes ported to v0.30.0

This migration uses `cann-framework/vllm-ascend` branch `releases/v0.30.0` at `0807cef70` as its base.
The base already includes GitHub PR11 and PR5; this PR keeps only the remaining
MoE, compilation and model runner V2 changes on top of them.
The source is the following merged PRs in `wangtao43/vllm-ascend` on GitCode.
It transfers their behavior through the 0.30 interfaces rather than restoring
0.27 files or merging the entire 0.27 release branch.

## Source comparison

| GitCode PR | Source merge | Treatment on 0.30 |
| --- | --- | --- |
| 5 | `6cbcbbf71` | Keep MXFP4 shared SiTU output in BF16 for the registered Down linear to quantize. PR64 supersedes the earlier Prefill core limit: limits now belong only to the Decode ladder. |
| 35 | `af5fbc0db` | Port sparse unquantized SiTU into `_unified_apply_activation`; port grouped SiTU/MX quant into the W4A8 MXFP quant method; request expert counts from MC2 dispatch. Keep the new GMM hook architecture and its LoRA/EPLB paths. |
| 42 | `b41a64714` | Select sparse SiTU for A5 Kimi shared experts, preserving the registered Dense MLP and checkpoint parameter names. CPU, other devices and unavailable custom operators retain the native fallback. |
| 45 | `9f87b17b1` (source commit) | Included in PR48 ancestry. Keep Router GEMM and eligible MC2/SP TopK on the Router stream, with a dedicated model-owned shared communication stream. Retain 0.30 common prepare for activation padding/masks; join precomputed TopK before Dispatch. |
| 48 | `0275e9c89` | Keep 0.30's latent-input/output transforms, eager single-group GMM selection, output normalization and runtime reductions. Move the original registered router gate into an opaque MoE operation and overlap shared input gather with Router/latent down. Never project latent input twice. |
| 50 | `eb8dda36e` | Already covered: 0.30 router selection and quant-method application retain FP32 TopK weights. Add a regression; do not restore old per-method dtype policy code. |
| 51 | `1288cd435` | Add a per-forward Decode schedule on top of 0.30's shared executor and quant hooks. Both BF16 and MXFP4 shared SiTU wait for Dispatch; routed GMM1 waits for that activation; shared Down starts after GMM1. |
| 54 | `9172ba684` | Enable the text-model `support_torch_compile` boundary; keep KDA metadata dispatch opaque using `no_compile_layers`, BFG/QKV tracing protection and the FIA static blacklist entry. |
| 62 | `723ee0633` | No code needed: 0.30 no longer contains `_plan_mega_kda` or its metadata fallback logs. Do not restore that obsolete path. |
| 20 | `e689fdd0a` | Keep the 0.30 CANN MegaMoE backend, per-model communication lookup, activation binding, weight layouts and HCCL restore lifecycle. Port Kimi prefill/decode/idle agreement into the existing CPU DP collective and selector. |
| 64 | `d8d72e6e4` (source head) | Port named Decode MoE SuperKernel scopes, owning-graph optimization, draft isolation and early shared ReduceScatter. Reuse 0.30 `enable_super_kernel`; do not add the source branch's separate switch. Keep 0.30 projection and weight layouts. |
| 65 | `3a3e40fb3` | Call the native A5 TopK dispatcher directly so recipes imports cannot redirect softmax/sigmoid routing into the custom Hash alias. Included in PR64; ported once. |

The 0.30 router kernel and dtype behavior, KDA AscendC implementation, KV pool
hooks, fine-grained TP padding, hardware
capability registry, and non-Kimi communication policies remain authoritative.
No mutable module-level registry is introduced.

## Combined AttnRes and compilation support

This branch combines GitHub `cann-framework/vllm-ascend` PR5 at `81e5e8f31`
with PR6 at `cea1c3d38`. These GitHub PR numbers are separate from the GitCode
source PR numbers in the table above.

PR5 adds `attn_res_mode` values `fused`, `two_phase`, `original` and `native`.
The default is `fused`, using optional transformer/DSL operators; unavailable
operators fall back to the Torch `two_phase` implementation. The centralized
`VLLM_ASCEND_KIMI_K3_ATTNRES_FUSED_ENABLED` switch defaults to `1`; setting it
to `0` makes a requested `fused` mode use `two_phase`. The non-persistent query
buffers refresh in place after weight loading, preserving captured references.
PP/SP transport and DSpark auxiliary-state contracts are retained.

The Kimi text model again uses `@support_torch_compile`. CPU fullgraph tests
cover Dynamo and AOT functionalization for `original`, `two_phase`, and the
fused adapter with simulated kernels. Actual fused NPU kernels still require
hardware validation. Explicit `native` mode keeps the 0.30 `attn_res_fwd`
implementation: its aliased outputs remain incompatible with the AOT
functionalization path described in the reported error. Use `two_phase` or
`original` for that path; restoring the decorator does not fix the native
operator schema.

## Recipes SiTU availability

Kimi A5 SiTU model construction loads the installed recipes `custom_ops`
extension before tracing. Previously, the migrated call sites only checked
`torch.ops.custom`; a worker that never imported the extension silently used
the native fallback even when the package was installed. Missing or incompatible
packages now produce a warning naming the unavailable operators.

The two calls are `torch.ops.custom.npu_situ_and_mul_sparse` (shared experts
and eligible unquantized routed experts) and
`torch.ops.custom.grouped_situ_mx_quant` (eligible W4A8 MXFP routed experts).
Their profiling names are `SituAndMulSparse` and `GroupedSituMxQuant`.
The recipes `ops/ascendc/torch_ops_extension` package and matching CANN operator
package must be installed in every worker environment. This PR does not bundle
those external kernels. Registration alone does not override dtype, activation,
quantization, LoRA or hardware eligibility checks. MegaMoE uses its fused
pipeline and need not show these standalone operators in profiling.
Recipes registration also overwrites some `torch_npu` Python aliases. A5
softmax/sigmoid routing now uses `torch.ops.npu.npu_moe_gating_top_k` directly,
preserving native arguments, softmax renormalization and int32 expert IDs.

## Decode SuperKernel

Use the existing 0.30 `ascend_compilation_config.enable_super_kernel` flag.
Its existing default inheritance from `enable_static_kernel` is unchanged;
there is no additional Kimi SuperKernel switch. On A5 Kimi with shared
multistream and either model runner, capture optimizes only graphs that emitted
eligible Decode MoE scopes. Draft, Prefill and unsupported batches do not emit
these scopes. Other models retain their generic 0.30 policy.

Named scopes cover Router/latent down through routed Combine and latent
norm/up. Nested exclusions keep shared AllGather, shared ReduceScatter, the
final join and Add outside fusion. MoE layers sharing one model
configuration reuse one Router stream, matching the recipes model context.
Disabling shared multistream creates no Router stream. The owning ACL
graph is optimized once
after successful capture with `dcci_disable_on_kernel=[".*"]` and
`auto_op_parallel=1`; replay never repeats optimization. Capture tracking is
local to the forward context for V1 and to the owning `UpdatableGraph` capture
resources for V2, where the forward context is created and destroyed inside
the graph context. Failed captures are not optimized. Compiler
options/cache keys distinguish this policy from generic whole-model fusion.
Draft configuration keeps the v0.30 construction order: mark the returned copy
with `runner_type="draft"` without replacing the proposer's target configuration.

Actual kernel fusion and speedups require NPU profiling. A named scope does
not guarantee a single fused hardware kernel. The existing Decode eligibility,
weight layout, scale packing and native routing contracts are retained.

## Decode execution

`multistream_overlap_shared_expert` remains the existing multistream control.
Runtime metadata must explicitly indicate pure Decode. The new staged path
requires A5, MC2, SP shared experts, MXFP4 routed experts, BF16 activations,
BF16 or MXFP4 shared weights, no dynamic EPLB and no active MoE LoRA.
Other combinations retain the 0.30 shared-expert schedule.

The router uses `(AIC, AIV) = (32, 1)`. Shared Gate/Up uses `(32, 1)` with
BF16 weights and `(16, 32)` with MXFP4 weights, preserving the source PR51
split policy. These are requested limits, not a claim about measured kernel
core usage. Vector-resource waits are present for both weight types.

The sequence is:

1. Start shared AllGather on a dedicated communication stream. Router GEMM
   and TopK run consecutively on the Router stream. The main stream quantizes
   latent input before waiting for Router GEMM, then runs latent down while
   TopK may execute. Both branches can overlap AllGather.
2. Join shared AllGather before submitting routed EP communication, preserving
   0.30's cross-process-group ordering.
3. Join TopK after latent down and the original MC2 prepare. Submit shared
   Gate/Up against the event immediately before Dispatch, so Gate/Up and
   Dispatch share a release point. Gate/Up preparation/quantization remains
   outside the projection core limit.
4. Dispatch completion releases shared SiTU.
5. Shared activation completion releases routed GMM1.
6. GMM1 completion releases shared Down input quantization and projection, which may overlap routed activation,
   output quantization and GMM2.
7. Submit shared ReduceScatter on the communication stream, waiting only for
   shared Down's completion event. It can overlap routed activation, GMM2 and
   Combine. After latent up, the main stream waits for that invocation's RS
   completion event before Add.

Early shared ReduceScatter applies only to the validated MC2 Decode ladder;
all other paths retain 0.30's collective ordering. Shared input AllGather still
joins before routed EP communication. Multi-rank capture/replay must validate
the new overlap; CPU dependency tests do not establish HCCL deadlock freedom.

## Model Runner V2 validation

V2 reuses the same MC2 Dispatch and routed GMM1 callbacks as V1. Enabling V2
does not relax the Decode, hardware, SP, quantization, LoRA or EPLB checks
above. The existing `multistream_overlap_shared_expert` switch controls the
schedule independently of `enable_super_kernel`. Prefill retains its fallback
schedule without the Decode projection core limits.

Kimi and its BF16 O-projection MM/RS fallback now import the existing Ascend SP
operations. The fallback performs its own RS because the decoder skips a second
reduction when the O-projection operator is installed. Both call sites must use
runtime padding; switching only the model import leaves the fallback exposed
to unaligned token counts in compiled execution. Their custom-op boundary
keeps token padding inside the runtime operation when a compiled graph is
reused for other token counts, including requests outside the capture-size list.
For TP8, a graph traced with 320 tokens must still pad 318 tokens to 320
before sharding, rather than truncating them to 312. A graph traced with
16 tokens must also produce nonempty shards for 12, 8, 4, 2 and 1 token. Hidden-state and
padding-mask partitioning must agree on every rank. This does not change the
configured capture-size list or speculative decoding's batch descriptors.

For hardware validation, retain the deployment's model/quantization/EP/TP and
PD options, and set `VLLM_USE_V2_MODEL_RUNNER=1` before starting workers.
Merge these settings into the existing additional configuration:

```json
{
  "multistream_overlap_shared_expert": true,
  "ascend_compilation_config": {
    "enable_npugraph_ex": true,
    "enable_static_kernel": true,
    "enable_super_kernel": true
  }
}
```

First compare real-weight inference with `enable_super_kernel=false`, then
enable SK and repeat the same requests. Include default capture sizes and
explicit sizes below TP, both with and without speculative decoding. Validate
standalone and PD Decode deployments separately; include a mixed Prefill/Decode
batch to verify fallback selection. Restart workers between configurations
and retain each run's logs and compile artifacts.

Set `ASCEND_OP_COMPILE_SAVE_KERNEL_META=1` before startup to inspect scope
membership. Profiling should show shared Down releasing shared RS without
waiting for routed GMM2/Combine, and final Add waiting for both branches.
Check static-kernel names and fusion-failure reasons in `sk_meta`, rather than
inferring fusion from the task-time `static` column or requiring one kernel
per scope.

Local validation covers CPU eligibility, event dependencies, graph ownership,
failure cleanup and compiled SP padding. Multi-rank NPU capture/replay,
real-weight accuracy, deadlock freedom and latency remain hardware validation
items; CPU tests do not establish those results.

The per-forward schedule is passed through typed fused-expert/MLP inputs.
Its temporary layer reference is cleared in the existing `finally` block.
Router tensors are released at the opaque operation boundary and cross-stream
consumers are recorded for allocator lifetime tracking. The outer MoE op
resolves the layer once; it must not re-enter another layer-resolving MoE op.

## MegaMoE and dummy agreement

Kimi A5 MegaMoE uses the existing `enable_fused_mc2=2` configuration (internally
normalized to `1` after the operator availability check). The 0.30 model,
quantization, shape and capacity restrictions still apply. The config mutex
allows shared multistream only for supported Kimi A5 MegaMoE; other models,
other devices and the legacy dispatch-FFN-combine setting keep the 0.30
automatic-disable behavior. This lets one Kimi configuration use MegaMoE
for Prefill and the staged ordinary MC2 schedule for Decode.

Each participating v1 DP rank exchanges token count, candidate graph mode,
real-work flag and real-decode flag in one CPU-group collective.
Idle dummy ranks participate but their synthetic Decode flag is excluded
when real work exists. For all-dummy startup/capture runs, every rank is
considered so capture cannot accidentally select MegaMoE.

| Active workload in the communication domain | Kimi MoE | ACLGraph |
| --- | --- | --- |
| Pure prefill + idle dummy | MegaMoE if otherwise supported | NONE on all peers |
| Prefill + real Decode, or a local mixed batch | Ordinary MoE | NONE on v1; existing V2 policy |
| Pure Decode + idle dummy | Ordinary MoE | Common supported graph mode |
| Decode graph capture | Ordinary MoE | Requested capture mode |
| MRv2 coordinated pure prefill + idle dummy | MegaMoE if otherwise supported | NONE on all peers |
| MRv2 uncoordinated/draft forward, or consumer-only instance | Ordinary MoE | Existing runner policy |

A prefill is detected from computed versus prompt tokens, not from token
count alone. One-token/chunked prefill therefore cannot accidentally reuse a
Decode graph. Profile runs may warm the prefill path. Draft forwards do not
select Kimi MegaMoE.

Keep 0.30's uniform input padding when its selected backend or fine-grained
TP requires it. Unlike the old PR20 code, 0.30 already preserves the agreed
`num_tokens_across_dp` vector inside dummy execution; this fix is retained.
The policy assumes matching configuration across participating ranks.

V1 and V2 both report Kimi eligibility and backend selection at INFO level.
V1 emits `Kimi V1 MegaMoE config` from backend selection; V2 emits
`Kimi V2 MegaMoE config` after loading. Configuration messages are deduplicated
per configuration in each worker process, while selection messages use the
logger's default local-first-rank scope and are deduplicated by decision.
The configuration includes actual layouts, activation, KV role, upstream EPLB
and legacy Ascend dynamic EPLB/redundancy/mix-placement flags. Selection
includes the agreed Decode flag, graph mode, capacity overflow and draft flag.

PD colocation (`kv_both` or no KV transfer) does not itself exclude V1
MegaMoE. Pure Prefill across the participating ranks may use it; real Decode
or a Prefill/Decode mixture in that communication domain selects ordinary
MoE. The separate `mix_placement` expert-placement feature and legacy Ascend
dynamic EPLB/redundancy remain excluded by the shared weight guard. Upstream
`--enable-eplb` is accepted only with V2 by platform validation; the V2 EPLB
support below does not enable that flag on V1.

MRv2 scopes target phase metadata to `execute_model` and records real request
stages in `gather_batch_req_state`. Its existing dispatch patch exchanges
real-work, Decode, profiling, eligibility and capacity metadata in one extra
CPU-group all-reduce before upstream graph/DP coordination. This collective
also runs on idle peers; it cannot be conditional on local Prefill alone.
Consumer-only instances retain their existing coordination without this extra
collective. A `kv_both` instance can produce and consume KV; its actual batch
phase determines eligibility. Pure Prefill can use MegaMoE, while Decode and
mixed batches continue to use ordinary MoE.
Supported pure Prefill forces eager execution and pads the runner descriptor
and DP token vector to the same TP-aligned count. Real token counts remain
separate for mask construction, KV metadata and sampling. Missing phase data,
graph capture/replay and all-dummy non-profile runs cannot select MegaMoE.

The first MRv2 integration falls back to ordinary MoE for PCP, microbatching
and cross-DP o_proj/MLP TP that requires graph execution. DCP alone keeps the
same query-token layout and does not trigger this fallback. Existing weight,
activation, LoRA, EPLB and capacity checks still apply. Explicit profile runs
warm MegaMoE before KV cache sizing, including the extra capacity-sized dummy
when the scheduler token limit exceeds the MegaMoE limit. Explicit CPP
Prefill timing dummies use the same backend; unrelated graph warmups remain
ordinary MoE. Real forwards clear padded input IDs and positions so a previous
batch cannot supply stale input values to the new physical rows.

A5 MegaMoE requires `x_active_mask=None`. Although the runner preserves valid
token metadata, padded and idle rows still participate in the fused operator.
Their outputs are discarded, but their routes consume receive-buffer capacity.
Tune `mega_moe_max_tokens` using the physical padded workload and expert-load
skew; a valid-token mask does not prevent A5 receive overflow.

The dependency-free regression suite is
`python3 tests/ut/worker/test_kimi_megamoe_v2.py`. On A5, additionally run the
existing operator integration with
`torchrun --nproc_per_node=8 tests/e2e/nightly/single_node/ops/test_mega_moe.py --tokens 1 32 256 --v2-prefill`.
This covers real CPU collectives, unequal scheduled counts, an idle peer and
the padded MegaMoE numerical reference; complete model accuracy, PP/DCP,
long-running request transitions and latency still require deployment tests.
Independent P and D instances make decisions in their own communication domains.

MRv2's upstream `--enable-eplb` supports redundant expert slots with this
Prefill backend. For 224 logical experts, eight redundant experts and EP8,
the physical capacity is 232 experts, or 29 per rank. The existing router
maps logical TopK IDs to physical IDs before dispatch; the communicator uses
physical capacity, and the returned per-expert token counts feed V2's existing
load recorder. MegaMoE reuses the weight lists registered for EPLB migration
and views their byte scales as E8M0 without copying storage. Ordinary Decode
continues to use transposed views of those same live lists. Legacy Ascend
dynamic EPLB and mix placement retain their existing fallback.

The CPU regressions check selection with `kv_producer`, DP1, TP8/EP8 and
eight redundant experts, plus storage sharing across simulated in-place
weight migration. On A5, test the 29-expert weight-list contract with
`torchrun --nproc_per_node=8 tests/e2e/nightly/single_node/ops/test_mega_moe.py --experts 232 --tokens 1 32 256 --v2-prefill --eplb`.
This operator case does not run the asynchronous EPLB controller. Real-model
validation must retain `--enable-eplb --eplb-config.num_redundant_experts 8`
and run beyond the configured `step_interval` until a cycle commits, then
check accuracy, mappings, local load counts and Prefill/Decode transitions.

The multimodal K3 wrapper constructs its language model through
`init_vllm_registered_model`, which deep-copies `model_config` in
`with_hf_config`. Routed-layer quantization records belong to that inner
configuration. The wrapper propagates those records to the outer configuration
before weight processing, so the V2 runner can validate the loaded layouts.
Without this propagation, even a pure Prefill batch on a `kv_producer` sees
unknown layouts and falls back to ordinary MoE. Mixed and unsupported layouts
remain ineligible after propagation.

To diagnose a missing MegaMoE call, inspect the `Kimi V2 MegaMoE config` INFO
line emitted after model loading. It reports the normalized fused-MC2 setting,
operator availability, hardware support, actual registered routed layouts,
group size, EP/PCP sizes, KV role, upstream EPLB flag and redundant expert
count. An internal `fused_mc2=1` is expected
after the user's `enable_fused_mc2=2` is normalized. Kimi's A5 path requires
`W4A8MXFP`; an ordinary INT4/INT8 `W4A8` checkpoint does not meet this layout
contract and must not be enabled by relaxing the selector.
The `capable` flag alone does not prove execution: unknown layouts may reserve
capacity before loading, while runtime selection also requires supported
weights and an agreed Prefill phase.

The INFO phase decision distinguishes `ready`, `decode_or_unknown_phase`,
`ineligible_rank` and `token_capacity_or_padding`. The separate
`Kimi V2 MoE selection` line reports the chosen backend and forward-context
phase/graph state. Set the existing `VLLM_LOGGING_LEVEL=DEBUG` before restarting
workers for per-batch local/peer metadata, the graph-required eligibility flag,
microbatching, PCP and aligned token counts. Filter worker output with:

```bash
rg 'Kimi V[12] MegaMoE|Kimi V[12] MoE selection' worker.log
```

The initial regression suite missed the `kv_both` producer/consumer combination:
its Prefill was incorrectly excluded at context creation, phase agreement and
backend selection. A regression now executes the runner hooks across Prefill
and Decode on that role and verifies `FUSED_MC2` followed by ordinary `MC2`.

## Validation

The following tests can run directly with CPU PyTorch, without loading NPU
plugins. They execute extracted production functions and mock kernels or
collectives; they are not hardware or full-package import validation.

```bash
python tests/ut/ops/test_kimi_kda_compile.py -v
python tests/ut/ops/test_kimi_v30_moe.py -v
python tests/ut/ops/test_kimi_moe_front_overlap.py -v
python tests/ut/ops/test_kimi_v30_review.py -v
python tests/ut/ops/test_kimi_situ_loading.py -v
python tests/ut/compilation/test_kimi_moe_superkernel.py -v
python tests/ut/models/test_kimi_v2_sequence_parallel.py -v
python tests/ut/worker/test_kimi_megamoe_phase_policy.py -v
```

They cover runtime metadata changes through a single Dynamo/AOT graph,
model replacement with the same layer prefix, SiTU count layout and fallback,
FP32 route-weight preservation, single latent projection/layer resolution,
stream dependency ordering for BF16/MXFP4, and DP prefill/decode/dummy policy.

The NPU SiTU regression is:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_shared_sparse_situ.py
```

Required multi-rank acceptance before merging: real checkpoint accuracy against
0.30 eager; eager versus ACLGraph with and without `npugraph_ex`; static-kernel
capture/replay; pure prefill + idle, prefill + Decode, mixed local batches,
multiple prefill lengths, SP/TP configurations and repeated phase transitions.
Profile Dispatch/SiTU/GMM1 and shared Down ordering and compare end-to-end
latency. The local environment has no NPU or complete vLLM runtime, so these
hardware results must be supplied separately. A first-token match alone does
not establish multi-token correctness.

Local results after review: 29 CPU tests passed; changed Python files passed Ruff and AST
parsing; the migration document passed markdownlint; `git diff --check` passed.
`bash format.sh ci` was attempted twice outside the sandbox, but actionlint's
Go environment download failed first with a TLS EOF and then a timeout at
`go.dev/dl/?mode=json`. The full check did not reach completion.

## Follow-up review corrections

The initial migration missed several contracts that the focused policy tests
alone did not exercise. The follow-up includes failing-before/passing-after
regressions for the following fixes:

- MegaMoE's W4A8 MXFP loader releases the dense weights. Ordinary Decode now
  consumes zero-copy transposed views of the live MegaMoE lists, including
  the path where grouped SiTU is unavailable. The dense GMSQ branch is not
  used for released tensors. A CPU test executes the production weight loader,
  simulates release, then checks Prefill/Decode/Prefill storage ownership.
- Kimi MegaMoE execution requires registered W4A8 MXFP routed layouts,
  group size 32, supported activation and the existing EP limit (2–64), with
  no mixed layouts, legacy dynamic EPLB or mix placement. Redundant experts
  are supported with MRv2's upstream EPLB and remain excluded otherwise. These
  constraints also govern weight preparation. Layouts are recorded as an
  immutable set on the owning model config, without a global registry.
- Before weights exist, startup still reserves 0.30's configured prefill
  capacity. Unknown layouts cannot execute MegaMoE. Oversized prefill falls
  back, and ordinary MC2 fallback uses its own 512-token-per-TP-rank limit,
  rather than assuming MegaMoE's larger capacity is valid for MC2.
- PD last-prompt-token recomputation follows 0.30's Decode classification;
  it must not force NONE merely because computed tokens equal prompt minus one.
- Serial execution acquires no shared auxiliary stream. Precomputed Router
  logits and preprojected latent inputs remain explicit opaque-op inputs,
  with correct fake output width and no repeated projection.
- Shared sparse SiTU inherits the existing activation's normalized beta,
  including the default when the raw configuration value is None.

The existing model-adapter test now expects routing inside the runtime MoE
operation. Full package and NPU execution remain unverified; CPU stream/kernel
simulations do not prove HCCL stability or real-weight precision.

The follow-up reran the full-format command with a 60-second limit; it again
stalled installing actionlint and timed out. Targeted Ruff, syntax and Markdown
checks passed. The existing adapter forward test also passed in isolation.

## Execution-path alignment

Shared MXFP4 uses the sparse SiTU module in Prefill, serial and staged Decode
paths; it no longer bypasses the module through fused SiTU/quantization.
A5 routed SiTU always uses separate GMM1 plus activation/quantization, including
the optional-operator fallback, so enabling multistream does not change that
operator choice. Down input quantization remains after GMM1, matching PR51.
Gate/Up core limits are applied on the shared producer stream. Staged Decode
retains PR51's validated quant schemes and excludes custom/lookup routing,
expert weight lists and output gates.

The correct upstream is `cann-framework/vllm-ascend`. The mistaken vllm-project
PR #21051 is closed; its unrelated README-only base update is excluded.
