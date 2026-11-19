# Decoder Append-Prefill

## Feature Introduction

Decoder append-prefill is an opt-in multi-turn routing and local cache-admission feature for vLLM V1 Prefill/Decode (P/D) disaggregated serving in vLLM Ascend. After a session's initial turn completes through normal P/D disaggregation, the proxy routes follow-up turns directly to the same owner Decoder endpoint and data-parallel (DP) rank so the Decoder can reuse its resident KV prefix, compute only the uncached suffix, and generate the reply locally.

Offloading warm follow-up prefills from Prefiller instances reduces repeated Prefiller work and queueing in multi-turn agentic workloads, improving multi-turn throughput and follow-up Time To First Token (TTFT). Actual gains depend on KV cache residency, prompt growth, output length, Decoder load, and fallback rate, while mixed prefill/decode steps on the Decoder can increase Time Per Output Token (TPOT).

### Working Principle

```text
                       Cold turn / no session ID
Client (X-Session-ID) ──────────────────────────► Prefiller (P) ──KV transfer──► Owner Decoder + DP rank
         │                                                                                ▲
         │             Warm follow-up turn (require_decoder_cache_hit=1)                  │
         └────────────────────────────────────────────────────────────────────────────────┤
                                                                                          │
                                                     Local KV prefix hit? ──► Yes: Append-prefill + Decode
                                                               │
                                                               ▼ No (zero-output miss)
                                                     Reject before prefill ──► Proxy retries full turn via P/D
```

1. **Initial P/D turn and session binding:** Clients send a stable `X-Session-ID` HTTP header and the full conversation history on every turn. Cold sessions route through normal P/D and bind `X-Session-ID` to the selected Decoder endpoint and DP rank once the Decode stream opens. Requests without `X-Session-ID` always use P/D; concurrent in-flight requests sharing `X-Session-ID` are rejected with HTTP `409` (`session_busy`).
2. **Warm Decoder-local routing:** Within `--session-affinity-ttl`, follow-up turns go directly to the pinned Decoder and DP rank (`X-data-parallel-rank`) with `vllm_xargs.require_decoder_cache_hit=1`, bypassing the Prefiller.
3. **Local cache-hit admission:** Session affinity predicts where KV may reside without reserving KV blocks. Before scheduling prefill, the Decode Scheduler (`DecoderAppendPrefillScheduler` or `AsyncDecoderAppendPrefillScheduler`) checks actual NPU prefix residency; on a hit, it reuses cached blocks and computes only the uncached suffix.
4. **Zero-output cache-miss fallback:** When no usable local prefix remains, the Scheduler rejects the request before prefill or output (`stop_reason="decoder_cache_miss"`), and the proxy retries the complete history and original generation budget via P/D. For streaming `echo=true`, echoed input from the local attempt is buffered until admission succeeds and discarded on a miss. If the P/D fallback itself fails, non-streaming requests return HTTP `502` and streaming requests emit an error event in the stream.
5. **Optional Prefiller affinity (`--enable-prefill-affinity`):** Copies a non-empty `X-Session-ID` into `user` and hashes `model` plus conversation identity (`X-Session-ID`, else `user`, else the first user message and preceding system/developer messages; load-balanced if none) to select a stable Prefiller endpoint—not a Prefiller DP rank—for cold turns and fallbacks. Decoder-local hits bypass the Prefiller; draining an endpoint remaps selection, and Decoder owner/rank binding remains independent.
6. **Optional Recompute and CPU KV offload (`PreemptOffloadConnector`):** Append-prefill does not require Recompute or CPU offload, but when paired with `recompute_scheduler_enable=true` and `PreemptOffloadConnector`, the combined Scheduler coordinates preemption recovery with local admission:
    - **Resuming a preempted request:** Waits for CPU KV save and H2D restore when offload state exists instead of rejecting the request as a cache miss.
    - **Admitting a new follow-up turn:** Requires a resident NPU prefix on the owner Decoder; a CPU copy alone does not satisfy admission until restored to the Decoder.
    - **Failure handling:** Failed recovery never triggers full Decoder prefill or Prefiller replay on a Decoder-local route—legacy `stop_reason="recomputed"` fails closed with `decoder_offload_failed` (even before output), and KV loss after output terminates with `decoder_cache_lost_after_output`. Legacy `stop_reason="recomputed"` text replay remains only for ordinary P/D routes and is not adapted for multi-role chat continuation (it can alter history, remaining token budget, and reported usage).

### Usage Scenarios

Decoder append-prefill targets **multi-turn agentic and tool-use workloads** with growing conversation histories and relatively short per-turn outputs, where Prefiller computation becomes the bottleneck and Decoders have enough HBM KV capacity to retain active session prefixes across turns.

Keep it disabled for single-turn traffic, stateless requests without `X-Session-ID`, workloads whose inter-turn idle time routinely exceeds Decoder KV retention, or Decode-bound deployments sensitive to mixed prefill/decode TPOT.

### Constraints and Limitations

| Category | Description |
| --- | --- |
| Hardware | Inherits the Ascend NPU support of the underlying P/D connector and model; end-to-end validated on the Atlas A3 series. Validate capacity and latency on target hardware before production use. |
| Deployment Scenario | vLLM V1 P/D Decode nodes (`kv_role="kv_consumer"`) with `--enable-prefix-caching` only; Prefill, PD-mixed, and standalone deployments fail at startup. Target Decode topology uses `pipeline_parallel_size=1` (larger PP sizes require separate validation), and every Decoder must match `--decoder-data-parallel-size`. |
| Engine | vLLM Ascend (vLLM V1), supporting Model Runner V1 (MRV1) and V2 (MRV2), and synchronous or asynchronous scheduling via `decoder_append_prefill_proxy_server.py` (`/v1/chat/completions`, `/v1/completions`). |
| Model | End-to-end validated on `Qwen3-30B-A3B-Instruct-2507` (including with `PreemptOffloadConnector` on MRV2) and `DeepSeek-V4.1`. Multi-group hybrid attention or recurrent-state models admit local reuse up to a common locally resumable block boundary without external connector assistance. |
| Feature Mutual Exclusion | Incompatible on Decode nodes with `dyntra_lb_config.enabled`, `enable_balance_scheduling`, `short_request_first_config.enabled` (supported independently on Prefill nodes), and `batch_job_sched_config.enabled`. |
| Software Dependencies | Working vLLM V1 P/D KV connector (such as `MooncakeConnectorV2` or `MooncakeConnectorV1`), plus `httpx`, `fastapi`, `anyio`, and `uvicorn` for the proxy. Optional CPU offload requires `PreemptOffloadConnector` via `MultiConnector`. |
| Other Limitations | Send `X-Session-ID` (the only header used for Decoder-local routing) and full history on every turn; proxy retains affinity only, not messages. Use `--workers 1`, restart the proxy after Decoder restarts (or call `POST /reset_prefix_cache` when resetting caches), and use non-overlapping `kv_port` ranges on shared hosts. |

## Feature Usage

### Environment Preparation

- Complete basic P/D disaggregation and Mooncake setup first; see [Prefill-Decode Disaggregation (Single Node)](../../tutorials/features/pd_disaggregation_mooncake_single_node.md) or [Prefill-Decode Disaggregation (Multi-Node)](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md).
- Install proxy dependencies (`pip install httpx fastapi anyio uvicorn`) and assign non-overlapping HTTP and `kv_port` ranges when co-locating Prefill and Decode on one host.

### Usage Example

This single-host 1P1D example runs Prefill on port `8100`, Decode on port `8200` (DP size `1`), and the proxy on port `9000`.

#### Step 1: Start the Prefill service

Keep your standard P/D producer configuration (`kv_role="kv_producer"`):

```bash
vllm serve /path/to/model \
    --host 127.0.0.1 \
    --port 8100 \
    --data-parallel-size 1 \
    --enable-prefix-caching \
    --kv-transfer-config '{
      "kv_connector": "MooncakeConnectorV2",
      "kv_role": "kv_producer",
      "kv_port": 27000
    }'
```

#### Step 2: Configure and start the Decode service

Enable `--enable-prefix-caching`, keep `kv_role="kv_consumer"`, and set `scheduler_config.decoder_append_prefill_enable=true`:

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
      "kv_connector": "MooncakeConnectorV2",
      "kv_role": "kv_consumer",
      "kv_port": 28000
    }'
```

- **Asynchronous scheduling:** Replace `--no-async-scheduling` with `--async-scheduling`; the platform selects `AsyncDecoderAppendPrefillScheduler` instead of `DecoderAppendPrefillScheduler`.
- **Optional Recompute + KV offload:** Set both flags in `--additional-config` to select `DecoderAppendPrefillRecomputeScheduler` or `AsyncDecoderAppendPrefillRecomputeScheduler`, and configure `MultiConnector` with `PreemptOffloadConnector` per the [Preempt Offload Guide](preempt_offload_connector.md):

    ```bash
    --additional-config '{
      "scheduler_config": {
        "decoder_append_prefill_enable": true,
        "recompute_scheduler_enable": true
      }
    }'
    ```

#### Step 3: Start the proxy

From the vLLM Ascend repository root, start `decoder_append_prefill_proxy_server.py`:

```bash
cd /path/to/vllm-ascend
python examples/disaggregated_prefill_v1/decoder_append_prefill_proxy_server.py \
    --port 9000 \
    --prefiller-ports 8100 \
    --decoder-ports 8200 \
    --decoder-data-parallel-size 1 \
    --session-affinity-ttl 3600 \
    --enable-decoder-append-prefill \
    --enable-prefill-affinity
```

Expected startup log: `Initialized 1 prefill clients and 1 decode clients in worker <pid>.`

### Verifying the Feature

1. Check proxy readiness (`"decoder_data_parallel_size": 1`, `"active_sessions": 0`):

    ```bash
    curl -sS http://127.0.0.1:9000/healthcheck
    ```

2. Send an initial turn with `X-Session-ID: conversation-001` (replace `your-model` with the served model name). Expect HTTP `200` with `"initial_route": "pd"` and `"final_route": "pd"` in `proxy_metrics`:

    ```bash
    curl -sS http://127.0.0.1:9000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -H 'X-Session-ID: conversation-001' \
      -d '{
        "model": "your-model",
        "messages": [
          {"role": "system", "content": "<original system prompt>"},
          {"role": "user", "content": "Analyze this report: <full report>"}
        ]
      }'
    ```

3. Send a follow-up turn with the same `X-Session-ID` and complete history, including the preceding assistant reply:

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

    Expected outcomes in `proxy_metrics`:

    - **Warm local hit:** `"initial_route": "decoder_local"`, `"final_route": "decoder_local"`, `"admission_fallback": false`, and `"cached_tokens_source": "decoder"`, plus `decoder_owner`, `decoder_dp_rank`, and `decoder_engine_epoch`.
    - **Cache-miss fallback:** `"initial_route": "decoder_local"`, `"final_route": "pd"`, `"admission_fallback": true`, and `"fallback_reason": "decoder_cache_miss"`.
    - **Streaming note:** HTTP headers (`X-Proxy-Route`, `X-Decoder-Owner`, `X-Decoder-DP-Rank`, `X-Decoder-Engine-Epoch`) reflect the initial route only; check `proxy_metrics` in streamed chunks for fallback status.

## Configuration Parameters

`Required` indicates whether a parameter must be supplied explicitly; omitted parameters use defaults. Enabling the feature requires both `scheduler_config.decoder_append_prefill_enable=true` (with `--enable-prefix-caching`) on every Decoder and `--enable-decoder-append-prefill` at the proxy.

### Decode Service Parameters

Configure this field under `scheduler_config` in `--additional-config` (see [Additional Configuration](../configuration/additional_config.md)):

| Parameter | Type | Default Value | Required | Value Range | Description |
| --- | --- | --- | --- | --- | --- |
| `scheduler_config.decoder_append_prefill_enable` | bool | `false` | No | `true`, `false` | Enables local cache-hit admission and selects `DecoderAppendPrefillScheduler` or `AsyncDecoderAppendPrefillScheduler` on P/D Decode nodes (`kv_role="kv_consumer"`). Requires `--enable-prefix-caching` and proxy `--enable-decoder-append-prefill`. |

### Proxy Server Parameters

Because the base P/D proxy (`load_balance_proxy_server_example.py`) does not yet have a consolidated CLI parameter reference in the documentation, all options accepted by `examples/disaggregated_prefill_v1/decoder_append_prefill_proxy_server.py` are listed below and tagged as **`[Feature-specific]`** or **`[Inherited]`**:

| Parameter | Type | Default Value | Required | Value Range | Description |
| --- | --- | --- | --- | --- | --- |
| `--enable-decoder-append-prefill` | bool flag | `false` | No | Enabled when present; disabled when omitted | **`[Feature-specific]`** Enables Decoder-local routing for warm follow-up turns. Requires `scheduler_config.decoder_append_prefill_enable=true` on every Decoder. |
| `--decoder-data-parallel-size` | int | `1` | No | Integers ≥ `1` | **`[Feature-specific]`** Internal attention DP size of each Decoder endpoint; must match every Decoder's `--data-parallel-size`. |
| `--session-affinity-ttl` | float | `3600` | No | Values > `0` | **`[Feature-specific]`** Idle lifetime of a session binding, in seconds; size to cover typical inter-turn pauses without exceeding expected KV retention. Expired bindings use P/D; idle sessions are reclaimed periodically while in-flight sessions stay reserved. |
| `--decoder-engine-epoch` | str | `None` | No | Non-empty string, or omitted | **`[Feature-specific]`** Deployment label for logs and responses (random UUID when omitted). Does not validate backend KV residency; restart the proxy after Decoder restarts. |
| `--enable-prefill-affinity` | bool flag | `false` | No | Enabled when present; disabled when omitted | **`[Inherited]`** Uses rendezvous hashing to select stable Prefiller endpoints for initial P/D turns and fallbacks. |
| `--host` | str | `"localhost"` | No | Hostname or IP address | **`[Inherited]`** Proxy bind address. |
| `--port` | int | `8000` | No | `0`–`65535` | **`[Inherited]`** Proxy listening port (`0` selects an OS-assigned port). |
| `--prefiller-hosts` | list[str] | `["localhost"]` | No | One or more hosts | **`[Inherited]`** Space-separated Prefill hosts; count must match `--prefiller-ports`. |
| `--prefiller-ports` | list[int] | `[8001]` | No | Ports in `1`–`65535` | **`[Inherited]`** Space-separated Prefill ports; count must match `--prefiller-hosts`. |
| `--decoder-hosts` | list[str] | `["localhost"]` | No | One or more hosts | **`[Inherited]`** Space-separated Decode hosts; count must match `--decoder-ports`. |
| `--decoder-ports` | list[int] | `[8002]` | No | Ports in `1`–`65535` | **`[Inherited]`** Space-separated Decode ports; count must match `--decoder-hosts`. |
| `--workers` | int | `1` | No | Integers ≥ `1` (`1` validated) | **`[Inherited]`** `uvicorn` worker count; keep at `1` for validated operation. |
| `--max-retries` | int | `3` | No | Integers ≥ `1` | **`[Inherited]`** Maximum backend HTTP attempts before failing. |
| `--retry-delay` | float | `0.001` | No | Values ≥ `0` | **`[Inherited]`** Base exponential-backoff delay between retries, in seconds. |
| `--max-waiting-retries` | int | `3` | No | Integers ≥ `1` | **`[Inherited]`** Maximum health checks before dropping a dynamically added waiting instance. |
| `--waiting-retry-interval` | float | `10` | No | Values > `0` | **`[Inherited]`** Health-check interval for waiting instances, in seconds. |
| `--log-level` | str | `"INFO"` | No | `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"` | **`[Inherited]`** Proxy server log level. |
