# Whole-request Mooncake Push

`MooncakePushConnector` is an experimental non-layerwise P/D connector.
It submits P and D concurrently, reserves D pages during prefill, and writes
the complete request from a P-side background thread when both sides are ready.

Community discussion: [RFC #20790](https://github.com/vllm-project/vllm-ascend/issues/20790).

## Initial support

Use the same dense model, tokenizer/chat template, dtype, block size and TP size.
The first version requires DP=PP=PCP=DCP=1, one FP16/BF16 FullAttention group,
eager execution and synchronous scheduling.
Prefix caching, MLA, hybrid models, speculative decoding and KVPP are unsupported.
Unsupported configurations fail explicitly.

This connector uses no attention-layer send hook. Existing Pull behavior is unchanged.

## Start the engines

Replace the model and IP addresses. Verify Mooncake Ascend connectivity using the existing PD guide.

P:

```bash
vllm serve Qwen/Qwen3-0.6B --port 8100 \
  --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakePushConnector","kv_role":"kv_producer","engine_id":"prefill-push","kv_port":5600,"kv_connector_extra_config":{"push_timeout":120}}'
```

D:

```bash
vllm serve Qwen/Qwen3-0.6B --port 8200 \
  --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakePushConnector","kv_role":"kv_consumer","engine_id":"decode-push","kv_port":5700,"kv_connector_extra_config":{"push_timeout":120}}'
```

Proxy:

```bash
python examples/disaggregated_prefill_v1/mooncake_push_proxy.py \
  --prefill-url http://P_IP:8100 --decode-url http://D_IP:8200 \
  --prefill-kv-host P_IP --prefill-kv-port 5600 \
  --prefill-engine-id prefill-push --tp-size 1
```

The proxy supports `/v1/completions` and `/v1/chat/completions`, one prompt
and `n=best_of=1` per request. P's configured engine ID and control host/port
must match the proxy. For equal TP greater than one, set both engine TP sizes
and proxy `--tp-size` consistently. P control ports are `kv_port + tp_rank`.

D recomputes the final prompt token and samples every returned output token.
P's sampled token is discarded. Each attempt uses a fresh transfer UUID.

## Extra configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `push_timeout` | 60 seconds | Source/registration lifetime and D cancellation trigger; allow for prefill/queueing. |
| `push_handshake_timeout` | 10 seconds | Static handshake timeout before target block publication. |
| `push_max_pending` | 128 | Pending-registration and peer-metadata bound per worker. |

D target pages stay allocated during P computation. Include that reservation
in comparisons against Pull.

## Cancellation and recovery

D publishes target block IDs only after a model-stream initialization/zeroing fence.
If that fence fails, the target pages remain quarantined.
D cancellation retains targets until P confirms no WRITE can still access them.
A timeout requests cancellation and does not free pages by itself.

A WRITE failure without confirmed drain quarantines affected pages at both ends.
They are not returned to the block pool. Recover the affected engine processes
before resuming traffic. Process epochs reject old protocol messages.

The first implementation uses synchronous WRITE on a background thread, serializing
transfers on each rank. No Ascend performance gain is claimed without measurements.

## Validation

CPU protocol/scheduler tests live in `tests/ut/kv_offload/mooncake_v2/`;
proxy tests live in `tests/ut/proxy/`. Ascend output parity, abort/reuse pressure
and equal-TP multi-rank completion require hardware validation before production use.

See [the PD design](disaggregated_prefill.md) for existing Pull and Layerwise flows.
