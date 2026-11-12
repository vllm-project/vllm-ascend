# DeepSeek V4.1 PIECEWISE execution

This draft separates DS4.1 model-runtime PIECEWISE adaptation from the
cross-layer KV pool work in [PR #21116](https://github.com/vllm-project/vllm-ascend/pull/21116).
It targets Model Runner V2 and retains the existing compiled and breakable
graph runners. NPU validation is pending.

## Execution boundaries

The existing `dsa_v41_forward` and `dsa_v41_write_kv` operators are added to
the PIECEWISE splitting list. Their eager regions can observe the current
forward context between captured graph segments.

Bounded decoder replay also uses an explicit eager boundary. A new
`dsv41_decoder_replay_forward_with_output` operator writes caller-provided
hidden-state, pre-mix and auxiliary output buffers. Capture/replay retains
these output references while the current replay rows and attention context
are resolved when the eager callback executes. Tensor output arguments are
flattened at the breakable boundary so captured callbacks can weaken each
tensor reference individually.

`run_decoder_replay` selects this operator from immutable model configuration,
including NONE warmup; it does not specialize the compiled trace on the
warmup's runtime graph mode. Other configurations keep the existing operator.
`DecoderReplayState.prepare` permits current-request compaction in PIECEWISE
while capture/profile paths retain their existing warmup behavior.

## Engram and device metadata

`EngramModelState.prepare_engram_piecewise_inputs` refreshes lookup buffers
under the active forward context before dispatch. It removes FULL-only
producer coordinates and events; dummy capture binds the fixed lookup
buffers without a request-dependent pending lookup.

`ModelAclGraphManager.run_pw_graph` calls the existing parent runner after
input preparation and finishes device-metadata submissions after their
consumers. Capture wrappers retain matching metadata tasks for compiled
warmup/capture that reuses the same metadata object, rather than carrying
tasks across unrelated requests.

## Configuration and validation

Use the existing `compilation_config.cudagraph_mode=PIECEWISE` configuration
without `--enforce-eager`. V2 needs a compiled model or the existing
`VLLM_USE_BREAKABLE_CUDAGRAPH=1` option. This draft adds no public switch or
environment variable, and makes no new FULL-graph support claim.

The draft changes model execution only. It does not enable cross-layer
PIECEWISE transfers in #21116, which rejects those configurations until
integration and hardware validation are completed.

Unit coverage includes current-request replay state, caller-owned output
buffers, NONE-warmup trace reuse, fake-tensor outputs, Engram lookup refresh
and device-metadata capture lifetime. Before readiness, validate compiled
and breakable PIECEWISE on NPU with bounded replay ON/OFF, Engram, DSpark,
precision, multiple capture sizes and successive requests. Compare latency,
throughput and HBM usage under the same workload. Cross-layer/MultiConnector
PD integration needs separate validation before that combination is enabled.
