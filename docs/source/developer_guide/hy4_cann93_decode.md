# HY4 Flash CANN 9.3 decode paths

This contribution ports selected model changes from the
[run09 archive](https://github.com/peterDengcx/vllm-ascend-118/tree/codex/hy4-cann93-run09-20261123/examples/hy4flash/experiments/20261123-cann93-run09)
onto `LostFox11/vllm-ascend` commit `36c207820`, retaining its Engram prefetch support.

## Configuration

Merge the following booleans into `--additional-config` on the decode service:

| Key | Default | Behavior |
| --- | --- | --- |
| `hy4flash_cann_engram_rms` | `false` | Use CANN rstd for aligned, contiguous FP32 Engram tensors on Ascend950, dedicated KV consumer only. |
| `hy4flash_fused_ihc_pre_tail` | `false` | Fuse gate arithmetic for precise BF16 decode, 4 channels, hidden 3072, 6-query batches up to 48 tokens. |
| `hy4flash_bulk_c8_decode_reads` | `false` | Use existing bulk gather/dequantization above the fused-read threshold, preserving negative-position masking. |
| `hy4flash_fused_ihc_post` | `true` | Preserve the existing fused post path; `false` selects explicit FP32 multiply/add for comparisons. |
| `hy4flash_dfly_dp_graph` | `false` | Experimental local dense draft graph capture for DP replicas with TP/PP/PCP/DCP all one and no embedding/head collectives. |

Invalid non-boolean values raise an error. Unsupported shapes retain the original
paths. Disabling an option requires restarting and recapturing graphs.
The Engram rstd substitution is tolerance-equivalent, **not bitwise equivalent**.
The original gamma multiplication order and invalid-hash mask are preserved.

The draft graph uses a copied dense forward context; the target MoE configuration
and actual communication groups are unchanged. Explicit eager mode and non-FULL
graph modes retain eager drafting. Capture uses the Ascend communicator context.
Graph buffers are private to the draft manager; real-checkpoint graph memory and
end-to-end performance remain to be measured before recommending this option.

## Recorded experiments (2026-11-23)

Hardware: eight Ascend950PR NPUs. Software: CANN/HIXL 9.3.0,
`torch_npu 2.10.0.post7`, Mooncake 0.3.14. Model: HY4 Flash FP8 checkpoint,
BF16 computation and FP8 KV. P uses cards 0-3, TP4/DP1, eager;
D uses cards 4-7, TP1/DP4, EP, DFly5 and target FULL_DECODE_ONLY.
Input/output: 128000/1024 tokens, C32/N128, seed 1024. Independently warm
115200 shared-prefix tokens; formal cache hit ratio is exactly 90%.
No profiler is enabled. Both runs completed 128/128 requests with zero failures.

| Metric | run09 | run10 |
| --- | --- | --- |
| P max model length | 131072 | 262144 |
| D max model length | 131072 | 131072 |
| Output throughput, whole 8-card service | 856.8797 token/s | 861.0235 token/s |
| Client mean TPOT | 12.6619 ms/token | 12.6159 ms/token |
| Client mean TTFT | 21.8267 s | 21.5195 s |
| D server mean decode span | 13.0005 s | 12.9517 s |
| P mean queue time | 18.5908 s | 18.2983 s |
| P prefill phase, excluding queue | 1.7840 s | 1.7641 s |
| Draft token acceptance | 99.4186% | 99.4186% |
| Preemptions | 0 | 0 |

Run10 changes only P context capacity relative to run09; source hashes match.
The 0.48% throughput difference is a single-run observation, not proven stable
speedup. Neither run has a same-environment optimization-off baseline.
Both runs explicitly used `hy4flash_dfly_dp_graph=false`: these are **not draft
graph performance results**. Stream events batch several draft tokens; event
spacing must not be interpreted as per-token TPOT. Acceptance on synthetic
requests is not a dataset accuracy evaluation.

### Module and graph evidence

Engram module NPU-event timing, graph replay, 100 replays per group and median of
five groups, synthetic inputs at actual model dimensions:

| Tokens | Original | CANN RMS | Latency reduction |
| --- | --- | --- | --- |
| 6 | 230.823 us | 226.322 us | 1.95% |
| 24 | 269.419 us | 261.346 us | 3.00% |
| 48 | 331.663 us | 317.609 us | 4.24% |

40 full-module comparisons passed elementwise `rtol=0.02, atol=0.03125`;
minimum cosine 0.999999821, maximum absolute difference 0.0625.
1,907 of 54,743,040 elements differed bitwise. These used random module weights,
not real Engram checkpoint projections. Candidate graph/eager outputs matched
bitwise at 6/24/48 tokens, including updated inputs.

Draft graph checks used the production capture path with a reduced model
(1 layer, hidden 128, vocabulary 256), real four-rank TP1/DP4 groups and random
BF16 weights. BF16 and FP8 KV covered 96 batch cases, including 24 empty cases;
all 72 nonempty cases matched eager five-step tokens and every-step logits
bitwise. B1/B3 graphs included B2 padding, changing request mappings, positions,
rejections, KV/topk snapshots and maximum-length boundaries. This does not
establish real-checkpoint graph correctness or speedup.

## Review scope and reproducibility

The original experiment also used an IndexPool group-major specialization,
server-specific pageable shared-memory compatibility, and an explicit HIXL
resource export workaround. Those changes are excluded here: the first needs
its own numerical/performance coverage, and the others need a portable design
and separate transport review. Therefore the table records the **historical
experiment bundle**, not a benchmark of this adapted PR or an attribution to
its selected changes. Original transport connectivity passed all 32 directed
P/D combinations. No runtime source imports a server-local probe script.

Before production enablement, run a paired real-checkpoint baseline/candidate
experiment on this target branch, including its newer Engram prefetch, verify
outputs and acceptance, and record graph memory. Native whole-iHC-pre replacement
remains outside this contribution. Run focused CPU tests with:

```bash
python -m pytest -q tests/ut/models/test_hy_v4_flash_dfly_dp_graph.py \
  tests/ut/models/test_hy_v4_flash_decode_fusions.py \
  tests/ut/models/test_hy_v4_flash_reference.py \
  tests/ut/models/test_hy_v4_flash_pd.py \
  tests/ut/models/test_hy_v4_flash_engram_shared.py
```

## Contribution checks

On the dependency-matched CANN 9.3 container, CPU-only focused tests passed:
37 tests covering draft capture eligibility, isolated context, communicator
scope, eager fallback, Engram fallback, C8 threshold/padding and disabled iHC.
The initial extended selection passed 63 tests and failed three. The same three
failures reproduced on unmodified target commit `36c207820` (27 passed): two
metadata fixtures omit `query_start_loc`, and the shared-memory cleanup fixture
expects `_shared_buffers` removed by the target branch. These are recorded as
baseline failures, not passing checks. No active inference source was replaced.

Historical NPU evidence above predates this integration; this contribution has
not rerun a full-model NPU benchmark. It remains a draft pending matched
hardware/accuracy validation and repository CI.

Changed Python files pass Ruff 0.14.0 and AST parsing; the added documentation
and tests pass codespell. Full `bash format.sh ci` was attempted in the isolated
Linux checkout but hook initialization stalled on GitHub access. The local
manual-stage pre-commit fallback was also blocked during dependency setup;
standalone markdownlint installation failed with `SELF_SIGNED_CERT_IN_CHAIN`.
Full format/Markdown checks remain for CI; they are not reported as passed.
