# Kimi DSpark C8 with QuantFlashMLA

This integration uses the existing CANN QuantFlashMLA attention and metadata
operators for Kimi DSpark C8 decode on Ascend A5 with Model Runner V2. It builds
on [PR #21569](https://github.com/vllm-project/vllm-ascend/pull/21569). It does not
introduce a model, native operator, environment variable or ModelRunner override.

## Configuration and scope

Use a compatible target and a draft checkpoint containing its own ModelSlim
attention recipe and KV scales. The draft's linear projections may remain BF16:
FP8 KV storage does not imply quantized linear weights. An unquantized draft
must not inherit the target's FP8 cache format.

A minimal eager example (replace deployment paths) is:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /path/to/target \
  --dtype bfloat16 \
  --quantization ascend \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --speculative-config '{"method":"dspark","model":"/path/to/draft-c8","quantization":"ascend","num_speculative_tokens":3,"draft_sample_method":"greedy"}'
```

Compatible DCP configurations additionally use
`--decode-context-parallel-size` and the target's required indexer configuration.
Both local/current and gathered/history query-head counts must be supported by
the delivered QuantFlashMLA operator. PCP greater than one remains excluded.
No `--block-size=128` workaround is required for the quantized staging workspace.

Both QuantFlashMLA APIs must be present and ABI-compatible with the runtime.
Image tags, directory names and wheel version strings alone do not establish
compatibility. Existing floating-writer page limits and upstream cache-spec
customization remain in effect. Real prefill attention retains its existing
path; writing FP8 context KV is not a QuantFlashMLA prefill-attention call.

With Mooncake pull P/D (`MooncakePullConnector` or `MooncakeConnectorV2`
without layerwise mode), both endpoints must use the same KV page layout and
cache dtype for each transferred layer. Deploy compatible operator packages
on both sides and verify that the local and gathered query-head counts in an
asymmetric TP/DCP deployment select the same cache layout on both endpoints.

## Integration contracts

- Resolve the draft's recipe using its local layer namespace without mutating
  target configuration or checkpoint names.
- Publish floating draft cache dtype and quantization metadata at the attention
  owner before upstream backend/page customization.
- Classify both fused projection methods; reject unsupported combinations
  instead of inventing linear weight scales.
- Keep persistent RoPE/latent scale buffers and refresh them in place on reload.
- Scatter the writer's quantized staging rows, not its optional BF16 return
  tensors. The DCP current cache uses the same actual FP8 rows.
- Jointly quantize the complete query in the existing cache units, keeping its
  descale aligned with the gathered/history or local/current head axes.
- Request LSE for the existing DCP merge. Floating attention must not slice or
  consume the quantized path's descale, including one-dimensional placeholders.
- Separate total token-buffer capacity from per-request query capacity. The
  parent's decode threshold supplies a stable metadata bound across capture and
  replay. A 256-token padded batch is not a 256-token individual query.
- Validate decode rows using existing CPU query boundaries; mixed-batch prefill
  lengths and graph padding do not set the decode bound. Actual cache lengths,
  query boundaries and used counts remain unchanged, including empty histories.

## Current-main integration checks

The PR was rebased onto Ascend main
`eafc123647e17d9c4332d92f792914aa1be585f5`. That baseline moves MLA cache-spec
construction into the attention owner. The floating-draft fix follows that
extension point instead of restoring the removed worker implementation, and
clears inherited KV quantization metadata for an unquantized draft.

The query-capacity and nonquantized-descale fixes below were transferred from
sealed device-tested sources. Operator ABI constants are class-local immutable
configuration constants, not new module-level variables.

The two standalone DCP CPU suites passed **52 tests** on the rebased source.
Targeted CPU contracts cover DCP orchestration, BF16 placeholders, quantized
scale selection, batch/padding/empty-history bounds, graph buffer reuse, mixed
prefill exclusion and the relocated cache-owner scope. They execute production
methods with mocked runtime/kernel dependencies; they are not NPU acceptance.
Ruff lint/format passed for all 17 changed Python files; Python parsing,
release-note YAML parsing and `git diff --check` passed. `bash format.sh ci`
was attempted but requires the unavailable `pre-commit` executable.

Full framework collection against the pinned vLLM revision currently encounters
an upstream `patch_kimi_k25` import incompatibility involving
`get_rope_shape_decorate`; the unmodified upstream baseline reproduces the
same collection failure. Exact rebased-revision framework and NPU reruns remain
required before merging. Do not treat the older device run as whole-tree parity.

## Eight-node device evidence (2026-11-18)

### Provenance

The tested runtime used vLLM
`4ba787ec9959e566d3b587a796dff07673324da5`, Ascend
`568d12495214ea4a06bbc4721d40ca4a01d7621c` plus the integration and fixes, and image
`sha256:b235e6cd7795575ff26289c2fac5a77e691ea07ab29c44637474e8caeb4efc30`.
The operator package was ops-transformer 9.3.0 (`20261031_220309764`). These are
provenance pins, not minimum supported versions.

Final tested source SHA256:

| File | SHA256 |
| --- | --- |
| `attention/context_parallel/mla_cp.py` | `f6544c7bce60d87b563eb0cfbbc15aca65dbe68e23a0cb042311bfe9fc116b58` |
| `attention/flashmla.py` | `4b61c1e5538ee55d7b7f3fbe66487c98a3ed4c29131b09440e7107dcaf184a9e` |
| `attention/mla_v1.py` | `3bcaa9f8cdd60896da3e4921d03728f8234477f39de54d669e2ab3f591f7c883` |

### Service and cases

- Eight hosts / 64 Ascend 950DT devices: four prefill and four decode hosts;
  TP8 per host, DP4 per role, P DCP1 / D DCP8, PP1 / PCP1, MooncakeConnectorV2.
- Full 93-layer, 896-expert W4A8_MXFP target; the same target weights and
  target FP8 KV/indexer configuration in both variants. Only draft checkpoint,
  explicit draft quantization and draft cache precision differ materially.
- Same-source five-layer Block7 BF16/C8 drafts, three speculative tokens.
  This is not the original three-layer Block5 draft configuration.
- Both roles: max model length 140000, max sequences 64; P batch tokens 16384,
  memory utilization 0.90 and eager; D batch tokens 768, memory utilization
  0.92, block/interleave 768 and FULL_DECODE_ONLY. P block size was not forced.
  EP, prefix caching and the role-specific optimization flags were retained.
- GPQA diamond: the same fixed 30 questions, thinking enabled, temperature 1.0,
  top-p 0.95, output limit 24576, client concurrency 32. Identical prompt/reference
  ordering was verified. ModelScope revision
  `1b5d04fbb4c58adc1be8ec92bc339e84c5fab826`.
- Performance: 16 requests from `GSM8K_prefix0_in16384_bs200_kimi`, each with
  16472 rendered input tokens and 1024 output tokens, client concurrency four.
  ModelScope revision `0dc81652f368433cd9b4059194d363d86ff17fec`.
- AISBench 3.1.20260630; all eight endpoints became healthy, all 32 decode ranks
  completed graph capture and actual PD transfer/request evidence was retained.

### Single paired measurement

| Metric | BF16 draft | C8 draft |
| --- | ---: | ---: |
| GPQA correct / total | 28 / 30 | 29 / 30 |
| GPQA accuracy | 93.3333% | 96.6667% |
| GPQA accepted / drafted / rounds | 81212 / 185622 / 61874 | 70083 / 156849 / 52283 |
| GPQA acceptance fraction | 43.7513% | 44.6818% |
| Performance requests successful / total | 16 / 16 | 16 / 16 |
| Performance accepted / drafted / rounds | 11788 / 16917 / 5639 | 11704 / 17181 / 5727 |
| Performance acceptance fraction | 69.6814% | 68.1218% |
| Output throughput (tokens/s) | 167.2521 | 170.4117 |
| Mean TTFT (ms) | 8677.5 | 8446.2 |
| Mean TPOT (ms) | 13.9 | 14.1 |
| Mean request latency (ms) | 22921.1 | 22884.6 |

Acceptance counters cover each complete stage including native warmup; native
performance timing excludes warmup. Counters were separated by role, deduplicated
by logical DP and not multiplied by TP. All prefill acceptance deltas were zero.
TTFT/TPOT retain AISBench's native streaming definitions, not a claim of strict
per-token ITL. Execution order was C8 then BF16, without alternating reruns.
These small samples do not establish statistical significance, general quality
non-regression or stable speedup. The 140000 setting does not mean 140000-token
requests were exercised. Actual prefix-cache hit coverage and maximum-throughput
capacity were not established.

### Earlier bounded evidence and remaining work

Earlier R1-R5 testing covered real writer boundaries, reduced-target eager
output parity, cache reuse and small graph/PD cases. A small TP2/DCP2 eager run
also matched same-topology target tokens. Those results use older source/runtime
combinations and are not additional full-model samples for the table above.

The expanded tests reproduced two integration bugs before the fixes: BF16 DCP
sliced a one-dimensional descale placeholder, and C8 graph metadata passed the
whole padded batch size as the per-request query bound. A real 64-request /
256-token metadata/current-attention probe passed after the latter repair with
zero output and causal-LSE error against its constant-value oracle.

Before leaving Draft, run exact-revision framework and NPU validation against
the current main-pinned vLLM, complete format/CI checks and obtain required owner
review. Wider datasets, repeated performance measurements, random-distribution
quality, actual long-context requests and additional topology coverage remain
separate work. Native traces, model weights, operational scripts and deployment
endpoints are not included in this PR.
