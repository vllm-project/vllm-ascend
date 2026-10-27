# Decoder Append-Prefill

Decoder append-prefill reduces repeated Prefiller work in multi-turn agentic sessions. After the first turn uses normal Prefill/Decode (P/D) disaggregation, the proxy can send later turns directly to the same Decoder endpoint and its original data-parallel (DP) rank. If the Decoder still has a usable local KV prefix, it computes the uncached suffix and generates the response.

Session affinity identifies where KV may be resident. The Decoder Scheduler checks actual local cache residency when admitting each request. A cache miss is rejected before scheduling prefill, so the proxy can send the complete request through normal P/D instead. Affinity does not reserve KV blocks or guarantee a cache hit.

The feature is opt-in in the Decoder configuration and at the proxy. It is intended for workloads with growing conversation histories and relatively short outputs, where repeated Prefiller work can become a bottleneck. Its benefit depends on cache residency, request lengths, Decoder load and the cost of cache-miss fallback.

## Requirements

- A working vLLM V1 P/D-disaggregated deployment, with Decode services configured as `kv_role="kv_consumer"` and prefix caching enabled.
- `scheduler_config.decoder_append_prefill_enable=true` in every Decoder's `--additional-config`. The platform selects the append-prefill admission Scheduler for its synchronous or asynchronous scheduling mode.
- The standalone `examples/disaggregated_prefill_v1/decoder_append_prefill_proxy_server.py` proxy, started with `--enable-decoder-append-prefill`.
- A stable session identifier and the complete conversation history on every request. The proxy retains routing affinity, not conversation messages.
- Requests must omit `echo` or set `echo=false`. The proxy returns HTTP 400 for `echo=true` while append-prefill is enabled.

## Configure the Decode service

Keep the Prefill service's normal P/D connector configuration, including its `kv_producer` role. On the Decode service, enable prefix caching, retain the matching P/D connector with the `kv_consumer` role and add `decoder_append_prefill_enable` under `scheduler_config` in `--additional-config`.

For example, the following shows a synchronous Decode service using Mooncake and one internal DP rank:

```bash
vllm serve /path/to/model \
    --host 127.0.0.1 \
    --port 8200 \
    --data-parallel-size 1 \
    --enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{
      "scheduler_config": {
        "decoder_append_prefill_enable": true
      }
    }' \
    --kv-transfer-config '{
      "kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_consumer",
      "kv_port": "28000"
    }'
```

For asynchronous scheduling, replace `--no-async-scheduling` with `--async-scheduling`; keep the same `--additional-config`. The platform selects `DecoderAppendPrefillScheduler` for synchronous scheduling and `AsyncDecoderAppendPrefillScheduler` for asynchronous scheduling.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `scheduler_config.decoder_append_prefill_enable` | bool | `false` | Enables append-prefill admission and selects its Scheduler on the Decode node. |

## Start the proxy

From the vLLM Ascend repository root, start the proxy with your running services. This example uses local Prefill on port `8100` and Decode on port `8200`, with Decoder DP size `1`:

```bash
python examples/disaggregated_prefill_v1/decoder_append_prefill_proxy_server.py \
    --port 9000 \
    --prefiller-ports 8100 \
    --decoder-ports 8200 \
    --enable-decoder-append-prefill
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--port` | `8000` | Proxy listening port. |
| `--prefiller-ports` | `8001` | Prefill service ports. |
| `--decoder-ports` | `8002` | Decode service ports. |
| `--enable-decoder-append-prefill` | Disabled | Enables Decoder-local routing for follow-up turns. |

For a Decoder with multiple DP ranks, set `--decoder-data-parallel-size` to its actual DP size.

## Send a multi-turn request

Use the same `X-Session-ID` on every turn. Alternatively, provide a nonempty `conversation_id` string in the JSON body; the header takes precedence if both are present.

The following examples show a follow-up turn after an initial request has completed with the same session ID, `conversation-001`. Enable the proxy and Decoder options above, replace the example proxy address and model name, and supply the original system prompt, full user input and actual assistant reply from the preceding turn.

**curl with the session ID in the HTTP header**

```bash
curl -sS http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Session-ID: conversation-001' \
  -d '{
    "model": "your-model",
    "messages": [
      {"role": "system", "content": "<original system prompt>"},
      {"role": "user", "content": "Analyze this report: <full report>"},
      {"role": "assistant", "content": "<actual reply from the preceding turn>"},
      {"role": "user", "content": "Suggest three improvements."}
    ]
  }'
```

**JSON body with the session ID in `conversation_id`**

Alternatively, put the same session ID in the JSON body. Save the following as `request.json`; the `X-Session-ID` header can be omitted.

```json
{
  "model": "your-model",
  "conversation_id": "conversation-001",
  "messages": [
    {"role": "system", "content": "<original system prompt>"},
    {"role": "user", "content": "Analyze this report: <full report>"},
    {"role": "assistant", "content": "<actual reply from the preceding turn>"},
    {"role": "user", "content": "Suggest three improvements."}
  ]
}
```

Send this JSON as the HTTP request body:

```bash
curl -sS http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @request.json
```

Requests without a session identifier use normal P/D. The proxy rejects a request if another request with the same session identifier is still in flight.

Use `proxy_metrics.initial_route`, `final_route` and `admission_fallback` to distinguish attempted local routing from the route that completed the request. Owner, DP rank and deployment epoch are also reported. For streaming, HTTP headers describe the initial route and cannot change after streaming starts; inspect the response's `proxy_metrics` for fallback information.

## Cache misses

On an initial Decoder-local cache miss, the Scheduler rejects the request before prefill or output. The proxy retries through normal P/D with the complete conversation history and original generation budget.

## Optional Recompute and KV offload

Append-prefill works with synchronous or asynchronous scheduling and does not require Recompute or CPU offload. When combined with Recompute and `PreemptOffloadConnector`, it reuses the existing preemption and KV offload mechanisms. The connector saves a preempted request's KV to CPU and reloads it when that same request resumes.

The combined append-prefill/Recompute Scheduler coordinates KV recovery with the local cache-hit check:

- **Resuming the same request:** If the connector still has offload recovery state for the preempted request, the Scheduler allows it to wait for KV storage and restoration, even when its NPU-local prefix is no longer resident. It is not immediately rejected as a cache miss.
- **Admitting a new follow-up turn:** A later request in the same session still needs a usable prefix on its owner Decoder. A CPU copy alone does not satisfy this requirement. Once restored KV is resident on the Decoder, later turns can reuse it through normal prefix caching.
- **Handling failed recovery:** The request follows the feature's fallback or termination handling. Failed recovery does not silently turn a Decoder-local request into a full prefill on the Decoder.

To enable the combined Scheduler, configure both options in the Decode service's `--additional-config`:

```bash
--additional-config '{
  "scheduler_config": {
    "decoder_append_prefill_enable": true,
    "recompute_scheduler_enable": true
  }
}'
```

The platform selects `DecoderAppendPrefillRecomputeScheduler` with `--no-async-scheduling`, or `AsyncDecoderAppendPrefillRecomputeScheduler` with `--async-scheduling`.

To enable CPU KV storage and reload, follow [Preempt Offload Connector](preempt_offload_connector.md) to configure `MultiConnector` with the P/D connector and `PreemptOffloadConnector`, including its CPU memory budget and shared-memory settings. Keep the top-level `kv_consumer` role.

## Support and validation boundaries

- DyntraLB is unsupported because it requires decode-only Decoder work. Balance, ShortRequestFirst, profiling-chunk and batch-job policies are also not composed with the append-prefill Scheduler. ShortRequestFirst can still be configured independently on Prefill nodes.
- The current target Decode configuration uses pipeline parallel size `1`. Combinations with profiling-based chunks or larger pipeline parallel sizes need separate admission and pipeline validation.
- Restart the proxy after Decoder restarts to clear session affinity. Use one proxy worker.
