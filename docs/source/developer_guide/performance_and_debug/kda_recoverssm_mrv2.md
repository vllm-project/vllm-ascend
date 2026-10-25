# Experimental GLM KDA RecoverSSM

Enable this experimental path with `--additional-config '{"kda_recoverssm": true}'`
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
speculative-window configuration, and `kda_recoverssm=true` cache layout.
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
