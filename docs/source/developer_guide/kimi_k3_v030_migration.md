# Kimi K3 changes ported to v0.30.0

This migration uses upstream `releases/v0.30.0` at `8a913c9f9` as its base (initial comparison: `56848620f`).
The source is the following merged PRs in `wangtao43/vllm-ascend` on GitCode.
It transfers their behavior through the 0.30 interfaces rather than restoring
0.27 files or merging the entire 0.27 release branch.

## Source comparison

| GitCode PR | Source merge | Treatment on 0.30 |
| --- | --- | --- |
| 35 | `af5fbc0db` | Port sparse unquantized SiTU into `_unified_apply_activation`; port grouped SiTU/MX quant into the W4A8 MXFP quant method; request expert counts from MC2 dispatch. Keep the new GMM hook architecture and its LoRA/EPLB paths. |
| 42 | `b41a64714` | Select sparse SiTU for A5 Kimi shared experts, preserving the registered Dense MLP and checkpoint parameter names. CPU, other devices and unavailable custom operators retain the native fallback. |
| 48 | `0275e9c89` | Keep 0.30's latent-input/output transforms, eager single-group GMM selection, output normalization and runtime reductions. Move the original registered router gate into an opaque MoE operation and overlap shared input gather with Router/latent down. Never project latent input twice. |
| 50 | `eb8dda36e` | Already covered: 0.30 router selection and quant-method application retain FP32 TopK weights. Add a regression; do not restore old per-method dtype policy code. |
| 51 | `1288cd435` | Add a per-forward Decode schedule on top of 0.30's shared executor and quant hooks. Both BF16 and MXFP4 shared SiTU wait for Dispatch; routed GMM1 waits for that activation; shared Down starts after GMM1. |
| 54 | `9172ba684` | Enable the Ascend text-model compile boundary; make KDA metadata dispatch opaque using the existing `no_compile_layers` registry; serialize BFG/QKV tracing for the AOT backend; append FIA to 0.30's existing static blacklist initialization. |
| 62 | `723ee0633` | No code needed: 0.30 no longer contains `_plan_mega_kda` or its metadata fallback logs. Do not restore that obsolete path. |
| 20 | `e689fdd0a` | Keep the 0.30 CANN MegaMoE backend, per-model communication lookup, activation binding, weight layouts and HCCL restore lifecycle. Port Kimi prefill/decode/idle agreement into the existing CPU DP collective and selector. |

The 0.30 router kernel and dtype behavior, KDA AscendC implementation, KV pool
hooks, Attention Residual implementation, fine-grained TP padding, hardware
capability registry, and non-Kimi communication policies remain authoritative.
No new environment variable or mutable module-level registry is introduced.

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

1. Start shared AllGather, overlapping the Router and subsequent latent down.
   Preserve 0.30's Router-before-latent-GEMM order and its TopK entry.
2. Join shared AllGather before submitting routed EP communication, preserving
   0.30's cross-process-group ordering.
3. Submit shared Gate/Up before routed TopK and Dispatch.
4. Dispatch completion releases shared SiTU/Down input preparation.
5. Shared activation completion releases routed GMM1.
6. GMM1 completion releases shared Down, which may overlap routed activation,
   output quantization and GMM2.
7. After routed finalize, submit shared ReduceScatter. The final Add waits for
   shared completion through the existing output-transform join.

**Deliberate difference from the 0.27 schedule:** shared ReduceScatter is not
moved ahead of routed finalize. The 0.30 code explicitly orders these
collectives to avoid HCCL deadlocks when ranks progress differently. Compute
can overlap; the newer communication ordering takes precedence.

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
| Prefill + real Decode, or a local mixed batch | Ordinary MoE | NONE on all peers |
| Pure Decode + idle dummy | Ordinary MoE | Common supported graph mode |
| Decode graph capture | Ordinary MoE | Requested capture mode |
| MRv2 or KV consumer | Ordinary MoE | Existing runner policy |

A prefill is detected from computed versus prompt tokens, not from token
count alone. One-token/chunked prefill therefore cannot accidentally reuse a
Decode graph. Profile runs may warm the prefill path. Draft forwards do not
select Kimi MegaMoE.

Keep 0.30's uniform input padding when its selected backend or fine-grained
TP requires it. Unlike the old PR20 code, 0.30 already preserves the agreed
`num_tokens_across_dp` vector inside dummy execution; this fix is retained.
The policy assumes matching configuration across participating ranks.
Independent P and D instances make decisions in their own communication domains.

## Validation

The following tests can run directly with CPU PyTorch, without loading NPU
plugins. They execute extracted production functions and mock kernels or
collectives; they are not hardware or full-package import validation.

```bash
python tests/ut/ops/test_kimi_kda_compile.py -v
python tests/ut/ops/test_kimi_v30_moe.py -v
python tests/ut/ops/test_kimi_v30_review.py -v
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

Local results after review: 27 CPU tests passed; changed Python files passed Ruff and AST
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
  no mixed layouts, dynamic EPLB, redundant experts or mix placement. These
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
