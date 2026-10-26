# Experimental GLM KDA RecoverSSM

Enable this experimental path with the upstream `--use-replayssm` option
alongside speculative decoding. The default remains the native AscendC KDA path.

Service configuration initialization requires speculative decoding,
PP/PCP/DCP=1, `--no-enable-prefix-caching`, and Mamba cache mode `none`.
KV transfer uses `MooncakeConnectorV2` (also named `MooncakePullConnector`).
`AscendConfig.derive_and_validate()` checks these constraints when
initializing the service configuration. Model-state initialization only activates
the shared cache contract before constructing the upstream state, after
target/draft configuration construction can reset the upstream cache flag.
It does not revalidate configuration. The enable helper and backend selection
only read the explicit switch.
Boolean validation stays in `AscendConfig`; speculative-token validity stays
in upstream configuration validation.

The retained hardware evidence used MTP with five draft tokens and the settings above.
That evidence did not use PD separation. PD validation is tracked separately.

## Prefill/decode separation

Configure the pull connector with `kv_role=kv_producer` on P and
`kv_role=kv_consumer` on D. Both services need the same model weights, TP size,
speculative-window configuration, and `--use-replayssm` cache layout.
The initial integration targets one producer and one consumer with equal TP.

The connector's existing stateful-model handoff leaves the last prompt token
for D. P transfers its MLA KV pages and KDA state page covering the preceding
prompt tokens. RecoverSSM has zero speculative checkpoint blocks, so the
producer retains its single checkpoint page and the consumer loads into that
page. Native Mamba still excludes its declared speculative scratch blocks.
The Mamba transfer copies all four strided views: convolution history,
checkpoint, correction records, and key/raw-gate records. The next verify
overwrites temporary records before post-sampling commit consumes them.

This connector transfers at request level; its per-layer wait/save hooks are
no-ops. AscendStore's aligned-cache path and legacy Mooncake connectors are
outside this integration. Successful configuration or CPU transfer planning
alone does not establish NPU PD correctness or model accuracy.

Existing end-to-end validation used Model Runner V2. Removing the runner-version
configuration gate does not establish support for other runners' post-sampling
state-commit paths; those paths have not been validated.

This implementation reuses the verified vLLM revision's RecoverSSM request
partitioning, accepted-count mapping, and post-sampling coordination. The
Ascend builder supplies convolution and prefill descriptors and stages
persistent NPU graph buffers without upstream CUDA-only checks. MTP capture
includes the lifecycle fix from [PR16936](https://github.com/vllm-project/vllm-ascend/pull/16936).

Each KDA page contains convolution history, one recurrent checkpoint,
FP32 correction records, and activation-dtype key/raw-gate records. The cache
spec allocates zero per-draft checkpoint blocks. Verification leaves the
checkpoint unchanged; after rejection sampling, commit recovers the accepted
state and compacts convolution history. Native AscendC projection, short
convolution, and prefill operators are reused. No upstream vLLM source edit is
required.

The Triton verify and `KDARecoverSSMCommitContext.create/commit` contracts follow
vLLM `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`, including strided page views,
multiple layers, and sampler request mappings. The low-level commit context
retains the upstream block-boundary interface; GLM model integration does not
yet enable aligned prefix caching. GLM's four-dimensional `A_log` is flattened
when building the context. Q/K normalization follows AscendC: a nonzero vector
uses its actual squared norm, with zero vectors returning zero. This differs
from the CUDA reference's additive `1e-6` bias.

With five draft tokens, head dimension 128, BF16 activations and FP32 recurrent
state, recurrent checkpoint plus records require 70 KiB per head, compared
with six native checkpoints requiring 384 KiB. This excludes convolution,
attention caches, allocator padding, and cache-group page alignment; actual
pool memory and serving performance require measurements.

Validation must cover multi-step mathematical recurrence, native AscendC
small-norm inputs, strided pages, mixed sampler rows, convolution compaction,
graph padding, graph replay, and per-position acceptance. MRV1 results from
the permanently archived [PR19087](https://github.com/vllm-project/vllm-ascend/pull/19087)
are separate evidence and do not establish MRV2 correctness or speed.

## Triton kernel optimizations validated on Ascend950DT

The optimized RecoverSSM path retains the four-view cache contract and commits
accepted state on the main stream. It does not add an AscendC replay backend or
change the existing AscendC path selected when RecoverSSM is disabled.

1. Verification loads gate bias outside the token loop, avoids redundant masks
   on complete tiles, and skips the final token's unused state update. It
   computes the output projection from the decayed query and correction rather
   than materializing an updated state for that projection. Larger value tiles
   and head-first program ordering are selected when enough programs remain.
2. Convolution compaction places channels on the contiguous inner axis of its
   tile and increases the channel block from 256 to 4096. Both convolution-state
   layouts, ragged channels, paged strides, and the low-level boundary-state
   contract remain supported.
3. Accepted-state commit specializes tensor strides and complete-tile masks.
   The six-token window used by MTP5 has a statically expanded reverse loop;
   other window lengths retain the dynamic loop. Each request still processes
   only its accepted records. Rejected suffixes are not read.

Verification rearranges floating-point arithmetic and can produce small
numerical differences. The commit optimization retains the existing reverse
FMA order. Model-level accuracy must be evaluated separately from operator
correctness and readable-output checks.

### Hardware evidence

Validation on October 24, 2026 used eight Ascend950DT devices, CANN 9.2,
PyTorch 2.10.0, torch_npu 2.10.0.post7.dev20261020, and vLLM
`a98247ab4db686ee03c66d5feb3c761e52a2f8ab`. Each configuration used actual global
decode concurrency 90, no prefill in the measurement window, native target/MTP
graph replay, MTP5, BF16 activations, and FP32 recurrent state. RecoverSSM
commits ran on the main stream.

The following are separate before/after experiments. Each row compares the
same configuration within that experiment; the rows must not be combined as
one end-to-end speedup against the original implementation.

| Change | DP8/TP1 kernel median | DP4/TP2 kernel median |
| --- | --- | --- |
| Verify V1 to V2 | 211.768 to 128.228 us | 211.659 to 133.335 us |
| Compact before to after | 415.210 to 21.255 us | 412.246 to 21.026 us |
| Commit before to after | 658.298 to 604.718 us | 667.191 to 614.594 us |

The independent commit serving comparison used three fully drained batches
of 90 fixed long-output requests per condition, with natural EOS enabled and
`max_tokens=3072`. Common full-decode windows lasted approximately 26 to 33
seconds. Token-weighted TPOT was calculated from timestamped SSE token delivery
inside those windows. It is not a percentile of individual token gaps.

| Configuration | Before commit optimization | After | Reduction |
| --- | --- | --- | --- |
| DP8/TP1 | 19.6987 ms/token | 19.6057 ms/token | 0.4721% |
| DP4/TP2 | 22.6801 ms/token | 22.5344 ms/token | 0.6426% |

These TPOT values are medians of three trials with RecoverSSM enabled on both
sides. They do not establish an improvement over disabling RecoverSSM. The
latest profiled ten-step spans were still approximately 2.83% and 2.08% longer
than the earlier same-host disabled baselines, respectively; that comparison
is across rounds. A disabled condition was not rerun with the long-output
workload. Acceptance and other operator timings can also vary between runs.

The commit candidate matched the preceding implementation elementwise in 32
numerical cases, including zero acceptance, invalid slots, ragged dimensions,
boundary-state handling, and alternative speculative windows. Those focused
checks supplement the repository's multi-step recurrence tests. The 64 complete
serving smoke responses were readable and naturally terminated, but they are
not a formal model accuracy evaluation. Other Ascend generations have not been
revalidated for these tile choices.

### Two-head commit grouping on Ascend950DT

For non-aligned FP32 states with head dimensions 128, 32 or 64 heads, and the
six-token MTP5 window, each program handles two adjacent heads with value tiles
of 64. The live state tile remains 16384 FP32 elements. Program count, useful
state bytes, reverse FMA order and accepted-record masking are preserved.
Splitting the value dimension repeats key/gate normalization work; it does not
reduce the state bytes or promise a peak-bandwidth improvement. Other devices,
dtypes, shapes, windows and the aligned boundary-state path retain the previous
kernel.

Six service shapes, accepted lengths 1, 3 and 6, eight rotating paged-cache
slots and NPUGraph replay produced 54 measurements, each with five samples of
40 launches. Correctness checks passed for all 54 points. Compared with PR6
revision `393b0cec94481d137274c1bdd5288cffd6770b9f`, kernel time decreased by
11.5% to 13.7% for accepted lengths 1 and 3, and by 3.0% to 3.7% for length 6.
Both eight-device A5 hosts passed 67 affected NPU tests each, including the
new 32/64-head multilayer paged-state cases. One unchanged large-memory
convolution reproduction test was deselected on each host.

Fresh full-service comparisons on October 25, 2026 used Recover off, that PR6
revision, and this grouped candidate on the same host and weights within each
configuration. Each condition completed 16 readable natural-EOS outputs,
three fully drained batches of 90 long-output requests, and profiling of eight
workers for ten forwards. All measurement windows had actual global decode
concurrency 90, no prefill, target/MTP graph replay, MTP5 and single-stream
Recover commits. The table reports the median token-weighted TPOT of the three
unprofiled trials; the first SSE event of each request is excluded.

| Configuration | Recover off | PR6 before grouping | Grouped | Grouped vs PR6 | Grouped vs off |
| --- | --- | --- | --- | --- | --- |
| DP8/TP1 | 19.2695 ms/token | 19.8409 ms/token | 19.6862 ms/token | -0.780% | +2.162% |
| DP4/TP2 | 22.3328 ms/token | 22.4728 ms/token | 22.3455 ms/token | -0.567% | +0.057% |

| Configuration | Commit median before | Grouped median | Before per rank/step | Grouped per rank/step |
| --- | --- | --- | --- | --- |
| DP8/TP1 | 604.250 us | 569.269 us | 1.8568 ms | 1.7638 ms |
| DP4/TP2 | 614.265 us | 569.739 us | 1.8519 ms | 1.7023 ms |

Each Recover profile contains 2720 verify calls and 240 calls to each commit
stage, including the expected grouped kernel. Worker source hashes, graph
counters, full D90 windows, SSE accounting and absence of external NPU
processes were checked. Device cumulative times can overlap and are not a
serial critical-path decomposition. Fixed condition order, three trials and
acceptance variation limit causal interpretation of small TPOT differences.
The grouped DP4/TP2 profile retains a 35.612 ms device gap on DP2/TP0,
with a 77.223 ms CPU Event::synchronize and 37.32-39.41 ms Dispatch calls
on other DP ranks. Its raw ten-step device span is not used to override
the separate unprofiled TPOT trials; the initial delay cause is unresolved.
DP8/TP1 smoke question 13 changed from the baseline's correct answer 18 to
the candidate's incorrect answer 36. This answer difference remains unresolved;
passing operator tolerances does not establish model accuracy or the cause.
Grouped reduction layout can introduce small floating-point differences
despite retaining the reverse FMA order. Readable smoke outputs do not
establish formal model accuracy. The two
configurations also use different hosts, drivers and model configurations;
their absolute values are not an isolated DP/TP comparison.
