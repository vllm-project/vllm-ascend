# Qwen3.8-Flash-Next A2 MegaMoE prefill

This experimental integration uses CANN MegaMoE for pure prefill on a single
910B eight-device worker. It is disabled by default. There is no minimum or
maximum token threshold for selecting the backend.

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
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_fused_mc2":2,"a2_mega_moe_prefill":{"enabled":true}}'
```

`enable_fused_mc2=2` explicitly selects the CANN package rather than the legacy
`dispatch_ffn_combine` operator. Both options above are required. The A2 setting
has only an `enabled` field; token-range options are not supported.

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
Ordinary mixed/decode execution keeps the same native-width stacked NZ weights
and FP32 first-layer scale copy. No per-forward repacking is introduced.

## Selection

| Forward | Routed MoE backend |
| --- | --- |
| Confirmed pure prefill, eager | CANN MegaMoE |
| Mixed prefill/decode | Original A2 selection (ALLGATHER for this EP8 model) |
| Pure decode | Original A2 selection |
| Missing or ambiguous phase metadata | Original A2 selection |
| Captured graph execution | Original A2 selection |
| Explicit memory-profile dummy run | MegaMoE can run without live phase metadata |

The phase check uses explicit prefill/decode counters or Qwen's complete QSA
request plan. Query length is not used to guess the phase. Unknown GDN metadata
can coexist with a complete QSA plan. A mixed/decode plan vetoes fusion.

The full-decode-only graph setting keeps prefill eager while decode retains its
ordinary graph path. Full and piecewise captured executions do not dynamically
switch their captured communication backend.

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

The sender capacity comes from the scheduler's `max_num_batched_tokens`, rounded
up to a TP multiple; A2 does not apply a token-count policy cutoff. All EP ranks
allocate identical resources. For this model the absolute receive-route bound is
`Bmax * 8 * 10`, where Bmax is the configured per-rank sender capacity.
The A2 prefill path uses that bound rather than the general
`mega_moe_max_tokens` reference hint, so skewed routing is not silently truncated.

Increasing the scheduler budget can substantially increase communication windows
and workspace. It is a memory budget, not a backend-selection threshold.
Forward inputs exceeding the allocated sender capacity raise an error. The
active mask excludes padding; each rank must still participate in collectives.

## Validation and rollback

CPU policy, real-selector, buffer-sizing, BF16/W8A8 weight-producer and scale tests can run
without a vLLM or NPU installation:

```shell
python -m pytest --noconftest -q tests/ut/ops/test_a2_mega_moe.py
```

These checks do not validate CANN registration, multi-device synchronization,
BF16 kernel precision, graph replay or performance. On the target machine compare
original and enabled deployments with identical inputs and scheduler settings:
pure prefill (including one-token prefix-cache tails), mixed batches, decode,
chunk boundaries, skewed experts, padding and repeated graph replay. Check every
rank's outputs and counts, TTFT, TPOT, throughput and peak memory.

Set `enable_fused_mc2=0` and `a2_mega_moe_prefill.enabled=false`, then restart the
workers, to restore the original weight-loading and communication paths. Merely
avoiding pure-prefill requests does not restore the original weight-loading configuration.
