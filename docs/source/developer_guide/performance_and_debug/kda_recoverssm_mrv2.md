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
