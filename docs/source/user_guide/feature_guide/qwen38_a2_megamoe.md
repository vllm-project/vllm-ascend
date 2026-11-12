# Qwen3.8-Flash-Next A2 MegaMoE

This experimental integration uses CANN MegaMoE for prefill, mixed batches and
decode on a single 910B eight-device worker, with model runners V1 and V2.
It is disabled by default. Once enabled, `scope="all"` is the default: eager,
graph warmup and ACL graph capture select the same backend, including when
phase metadata is absent. There is no token-count selection threshold.

## Enable

Install a compatible `cann_ops_transformer` build including the 910B
`MegaMoe` operator and its communication extension in every worker environment.
The operator must support A2 intermediate sizes aligned to 128 (native I=640),
such as ops-transformer commit `9adf80d98` on
[`qwen3p8/a2-megamoe-align128`](https://gitcode.com/workice/ops-transformer/tree/qwen3p8/a2-megamoe-align128).
Older A2 operators requiring 256 alignment cannot execute this path; there is
no automatic padding or fallback to the old 768-wide weight layout.
The integration targets BF16 or dynamic INT8 W8A8 Qwen3.8-Flash-Next, registered as
`Qwen4ExpForCausalLM` or `Qwen4ExpForConditionalGeneration`, with H=2560,
intermediate size=640, 512 experts and top-k=10.

```shell
vllm serve /path/to/Qwen3.8-Flash-Next \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --additional-config '{"enable_fused_mc2":2,"a2_mega_moe_prefill":{"enabled":true,"scope":"all"}}'
```

`enable_fused_mc2=2` explicitly selects the CANN package rather than the legacy
`dispatch_ffn_combine` operator. Both options above are required. The historical
`a2_mega_moe_prefill` key is retained for compatibility; its name does not limit
the default scope. Valid scope values are `all`, `prefill` and `prefill_mixed`. Existing enabled
configurations without a scope now use all-stage execution. Token-range options
are not supported.

No compilation mode is forced by this feature. Keep or tune the model's graph
configuration separately; `FULL_DECODE_ONLY` is one candidate, not a requirement.
`scope="prefill"` restores the original pure-prefill, eager-only policy for
comparison with the all-stage policy. `scope="prefill_mixed"` also selects eager
mixed batches while leaving pure decode on the ordinary backend.

For prefill/mixed MegaMoE with ordinary decode graph replay on MRv2:

```shell
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /path/to/Qwen3.8-Flash-Next \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_fused_mc2":2,"a2_mega_moe_prefill":{"enabled":true,"scope":"prefill_mixed"}}'
```

In that configuration, pure prefill and mixed batches run MegaMoE eagerly;
pure decode uses ALLGATHER, with graph replay when a compatible graph exists
and ordinary eager execution otherwise. The whole mixed batch, including its
decode rows, uses MegaMoE; rows are not split into separate backends.

This implementation requires BF16 activations, TP8/EP8, DP1, PP1 and PCP1, static contiguous
experts, and no LoRA, speculative decoding, EPLB or mixed placement. Invalid
combinations fail during configuration validation. A3/A5 selection is unchanged.

## W8A8 checkpoints

Use the same `--additional-config` with a dynamic per-token INT8 W8A8 checkpoint.
For a ModelSlim checkpoint, add `--quantization ascend` if required by the
checkpoint metadata, while keeping `--dtype bfloat16` for activations.
The resolved routed expert scheme must be `QuantType.W8A8`; FP8/MXFP, W4A8,
and other expert schemes are rejected. Unquantized layers may coexist with
W8A8 layers in a partially quantized checkpoint.

W8A8 weights must use symmetric per-output-channel quantization: both weight
offsets must be zero and scales finite and positive. Nonzero offsets fail during
weight loading because this MegaMoE INT8 API has no weight zero-point input.
The check runs only while loading, not on the forward hot path.

Gate/up INT8 weights and their scales/offsets retain the native 640-wide
intermediate dimension before NZ conversion. W1 has 1280 output channels and
W2 has 640 input rows; W2 output-channel scales and offsets remain H-wide.
Existing packed int64 scale encoding and per-expert NZ views feed MegaMoE.
Ordinary fallback execution keeps the same native-width stacked NZ weights and
FP32 first-layer scale copy. No per-forward repacking is introduced.

## Selection

| Forward | `scope="all"` (default when enabled) | `scope="prefill"` | `scope="prefill_mixed"` |
| --- | --- | --- | --- |
| Confirmed pure prefill, eager | CANN MegaMoE | CANN MegaMoE | CANN MegaMoE |
| Mixed prefill/decode, eager | CANN MegaMoE | ALLGATHER | CANN MegaMoE |
| Pure decode, eager | CANN MegaMoE | ALLGATHER | ALLGATHER |
| Missing or ambiguous phase metadata, eager | CANN MegaMoE | ALLGATHER | ALLGATHER |
| FULL / PIECEWISE warmup and capture | CANN MegaMoE | ALLGATHER | ALLGATHER |
| Graph replay | Captured MegaMoE | Captured ALLGATHER | Captured ALLGATHER |
| Explicit eager memory-profile dummy run | CANN MegaMoE | CANN MegaMoE | CANN MegaMoE |

The two phase-scoped policies check explicit CPU counters or Qwen's complete
QSA request plan. Query length is not used to guess the phase. Unknown GDN
metadata can coexist with a complete QSA plan. `scope="prefill"` requires a pure
prefill batch; a mixed/decode plan vetoes fusion. `scope="prefill_mixed"` requires
at least one explicitly identified prefill request, or a positive `num_prefills`
counter when no QSA plan is available. A complete pure-decode, empty or ambiguous
QSA plan vetoes fusion even if another layer reports prefill.

Both phase-scoped policies keep captured execution on ALLGATHER because phase
metadata may be absent during capture and graphs cannot change backends on
replay. To use MegaMoE for prefill/mixed with `scope="prefill_mixed"`, keep those
batches eager, for example with `FULL_DECODE_ONLY` or `--enforce-eager`. Selecting
PIECEWISE capture does not enable MegaMoE inside those phase-scoped graphs.

Graph support here means ACL Graph / NPUGraph capture of the existing ACLNN
execution path; it does not use or validate the CANN TorchAir/GE converter.
The surrounding model and installed torch_npu/CANN stack must also support the
selected graph mode. Full and piecewise captures retain their communication
backend on replay; changing scope requires a worker restart and fresh capture.
Symmetric buffers must be allocated in eager warmup before capture, or execution
raises an error rather than attempting a collective allocation inside capture.

V1 refreshes its persistent active mask before replay. V2 FULL replay bypasses
Python model execution, so MC2 preparation derives its active mask inside the
graph from the runner's persistent `is_padding` input. Updating that input changes
the active rows on each replay, including all-padding dummy inputs.

Memory profiling may call MegaMoE even when no actual request has used it. With
debug logging, inspect the selector and actual MegaMoE call separately; an
allocation log alone is not evidence of real-request coverage.

## Weight layout and memory

The compatible 910B operator accepts intermediate sizes divisible by 128.
The loader keeps I=640 for both fused prefill and ordinary mixed/decode GMM:
W1 is `[H,1280]` and W2 is `[640,H]` per expert. It does not append zero
channels or pad scales/offsets to 768.

BF16 uses per-expert ND views sharing the native-width execution storage.
W8A8 uses stacked NZ weights with per-expert views for the fused call.
No complete second execution-weight copy is retained after loading; a transient
loading/conversion peak remains possible.

For 48 MoE layers and EP8, raw routed BF16 weight payload is about 28.125 GiB
per rank and INT8 payload about 14.063 GiB, excluding other model weights,
scales, NZ layout overhead and buffers. Neither backend incurs the old explicit
768-wide GMM shape expansion. BF16 fallback still uses ND TensorLists, so
matching the original width alone does not establish identical performance.
The existing fused-MC2 configuration disables shared-expert multistream overlap.
Benchmark the complete model against the original configuration before rollout.

## Communication capacity

The sender capacity covers the larger of the scheduler's `max_num_batched_tokens`
and `max_cudagraph_capture_size`, rounded up to a TP multiple, including on
decode-only workers. The reserved active mask covers that padded capacity too.
A2 does not apply a token-count policy cutoff. All EP ranks allocate identical
resources. For this model the absolute receive-route bound is
`Bmax * 8 * 10`, where Bmax is the configured per-rank sender capacity.
The A2 path uses that bound rather than the general
`mega_moe_max_tokens` reference hint, so skewed routing is not silently truncated.

Increasing the scheduler budget can substantially increase communication windows
and workspace. It is a memory budget, not a backend-selection threshold.
Forward inputs exceeding the allocated sender capacity raise an error. The
active mask excludes padding; each rank must still participate in collectives.

## Throughput tuning

Compare the disabled backend (`enable_fused_mc2=0`, `enabled=false`), the original
`scope="prefill"` policy, `scope="prefill_mixed"` and `scope="all"` under identical workload conditions.
For all-stage execution compare `--enforce-eager` with the model/runtime-supported
`cudagraph_mode` settings: `FULL_DECODE_ONLY`, `FULL`, `PIECEWISE` and
`FULL_AND_PIECEWISE`. Check the actual selected modes in runtime logs; model or
attention-backend constraints may restrict capture coverage.

Tune `max_num_batched_tokens`, concurrency, chunked prefill and graph capture
sizes together. Larger MegaMoE windows and graph pools can reduce KV capacity
and therefore concurrency. Measure end-to-end throughput, TTFT, TPOT and peak
memory under the same prompt/output lengths and arrival pattern. Select the
highest throughput configuration satisfying any latency limits; all-stage fusion
or wider graph coverage alone does not establish a performance improvement.

## Validation and rollback

CPU policy, V2 platform-entry, selector, buffer-sizing, replay-mask, BF16/W8A8
weight-producer and scale tests can run without a vLLM or NPU installation:

```shell
python -m pytest --noconftest -q tests/ut/ops/test_a2_mega_moe.py
```

An eight-card A2 standalone regression exercises native I640 BF16/W8A8 eager
execution and ACL graph replay with changed inputs, routes and padding. Run
each shape/precision in a fresh process, for example:

```shell
torchrun --nproc-per-node=8 tests/e2e/nightly/single_node/ops/test_a2_mega_moe_graph.py --tokens 32 --precision bf16
torchrun --nproc-per-node=8 tests/e2e/nightly/single_node/ops/test_a2_mega_moe_graph.py --tokens 32 --precision w8a8
```

Repeat with per-rank token counts such as 1, 8 and 512. The regression checks
changed routing and active rows against eager outputs and independently checks
per-expert counts, including skewed routing and all-padding replay. It reports
eager and graph latency but does not assert a speedup or validate complete model
precision. CPU checks do not validate CANN registration, multi-device
synchronization, NPU precision, graph replay or performance. This integration
still needs target-hardware validation. On the target machine compare
original and enabled deployments with identical inputs and scheduler settings:
pure prefill (including one-token prefix-cache tails), mixed batches, decode,
chunk boundaries, skewed experts, padding and repeated graph replay. Check every
rank's outputs and counts, TTFT, TPOT, throughput and peak memory. Cover both
model runners and all supported graph modes, restarting between policies.

Set `enable_fused_mc2=0` and `a2_mega_moe_prefill.enabled=false`, then restart the
workers, to restore the original weight-loading and communication paths. Merely
avoiding pure-prefill requests does not restore the original weight-loading configuration.
