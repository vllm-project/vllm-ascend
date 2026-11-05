# Disaggregated Encoder (EPD)

## Why disaggregated encoder?

EPD separates the multimodal encoder stage from the language prefill and decode
stages of a vision-language model. Each stage has a different compute profile,
so running them in independent vLLM instances gives:

1. **Independent scaling of E, P, and D.** Encoders are small compared with the
   language model; the language model can be parallelized (TP/DP/EP) without
   changing the encoder fleet, and encoder instances can be added or removed on
   their own.
2. **Better TTFT control.** Vision encoding is removed from the critical path of
   prompt prefill, and text-only requests never enter the encoder stage. The
   P/D split can still be tuned separately, as in ordinary disaggregated
   prefill.
3. **Cross-process encoder-cache reuse.** An encoder-cache entry produced by any
   E instance can be loaded by any consumer instance, so repeated images do not
   pay for vision encoding again. In an in-process encoder this reuse is
   confined to a single scheduler/worker.

## Usage

For step-by-step deployment (E+PD and E+P+D, connector choice, health checks,
startup order, and graceful shutdown), see the
[Disaggregated Encoder user guide](../../user_guide/feature_guide/epd_disaggregation.md).
The reference examples are
`examples/disaggregated_encoder/disagg_1e1pd_example.sh` (E+PD) and
`examples/epd_disaggregated/epd_disaggregated_guide.md` (E+P+D).

## How It Works

### 1. Design approach

A load-balancing proxy owns the client-facing OpenAI-compatible API. For every
request it:

1. sends the request to an encoder instance when the request contains
   multimodal items, so that the encoder cache is written before prefill;
2. optionally sends a prefill-only request (`max_tokens: 1`) to a P instance,
   which loads the encoder cache, computes the prompt KV cache, and transfers it
   to D through a prefill-decode connector;
3. forwards the request to a decode instance, which loads the encoder cache (and
   the remote KV cache in E+P+D) and generates the response.

The proxy is a reference implementation, not part of the vLLM Ascend runtime.
The runtime contract that makes EPD work is the encoder-cache connector
interface and the model-runner integration described below.

### 2. Roles and configuration

| Topology | Instance | `--ec-transfer-config` | Other key parameters |
| --- | --- | --- | --- |
| E+PD | E | connector name, `ec_role: ec_producer`, connector extra config | `--gpu-memory-utilization 0.01 --enforce-eager --no-enable-prefix-caching` |
| E+PD | PD | connector name, `ec_role: ec_consumer`, connector extra config | `--gpu-memory-utilization 0.9` |
| E+P+D | E | same as the E+PD E instance | same as the E+PD E instance |
| E+P+D | P | `ec_role: ec_consumer` plus `--kv-transfer-config` with `MooncakeLayerwiseConnector` in the `kv_producer` role | `--gpu-memory-utilization 0.7` |
| E+P+D | D | no EC configuration | `--kv-transfer-config` with `MooncakeLayerwiseConnector` in the `kv_consumer` role |

Setting `ec_role: ec_producer` marks the instance as encoder-only
(`mm_encoder_only`): it runs the multimodal encoder and the EC save path
instead of the language model decode loop, so its KV-cache allocation is
minimal. Consumer instances keep the normal generation path and only add the
EC load path.

### 3. Data plane

**Cache key.** The encoder cache is keyed by `mm_hash`, the content-derived
identifier of a multimodal feature (`request.mm_features[i].identifier`). It is
not derived from `request_id`, so:

- the same image produces the same key across requests and instances;
- different images are isolated even when they share a request;
- the same image with different preprocessing or resolution may produce a
  different key, so reuse across preprocessing configurations is not guaranteed;
- repeated items inside one request are deduplicated before encoding.

**E side.** The encoder-only instance runs the multimodal encoder. The model
runner saves each encoded result with
`ECConnector.save_caches(encoder_cache, mm_hash)`; the connector serializes it
to its backend (a safetensors file for `ECExampleConnector`, a Mooncake buffer
for `ECMooncakeConnector`, a pinned `/dev/shm` mmap for `ECCPUConnector`, a
MemCache cluster for `ECMemcacheConnector`).

**P/D side.** On the consumer the schedule proceeds as follows:

1. the scheduler's `update_state_after_alloc` records the `mm_hash` values that
   must be loaded for newly allocated requests;
2. `build_connector_meta()` collects them into the connector metadata;
3. the metadata is attached to `SchedulerOutput.ec_connector_metadata` and sent
   to the workers;
4. on the worker, `start_load_caches(encoder_cache)` loads the entries into the
   encoder cache used by the model forward pass (on Model Runner V2 this is the
   upstream `EncoderCache`);
5. the model reads the embeddings during prefill/decode.

The Ascend integration supplies the connector implementations and the model
runner lifecycle; it does not change the upstream scheduler/worker contract.

### 4. Control plane

```mermaid
sequenceDiagram
    participant C as Client
    participant PX as EPD Proxy
    participant E as Encoder (ec_producer)
    participant P as Prefill (optional)
    participant D as Decode/PD
    C->>PX: POST /v1/chat/completions (with image)
    PX->>E: POST /v1/chat/completions (child req id, max_tokens=1)
    E->>E: vision encode, save_caches(mm_hash)
    E-->>PX: 200
    opt E+P+D only
        PX->>P: POST (max_tokens=1, kv_transfer_params)
        P->>P: load_caches(mm_hash), prefill, push KV to D
        P-->>PX: 200
    end
    PX->>D: POST (original request)
    D->>D: load_caches(mm_hash); on hit skip vision, otherwise encode locally
    D-->>PX: generated response (stream or non-stream)
    PX-->>C: forwarded response
```

### 5. Model runner selection

The encoder-cache integration has two implementations behind a single
configuration:

- **Model Runner V1**: `vllm_ascend/worker/v1/model_runner.py` contains the
  complete encoder-cache implementation, including the E-node encoder-only path
  and the consumer-side load path.
- **Model Runner V2**: the runner delegates the encoder-cache contract to the
  upstream V2 interfaces. Ascend pieces live in
  `vllm_ascend/worker/worker.py`: encoder-only instances use the upstream
  `MMEncoderModelRunner`; `ensure_ec_transfer_initialized` registers the
  connector during worker distributed initialization;
  `start_worker_services()` starts connector services after model load, and
  `ensure_ec_transfer_shutdown()` stops them during worker shutdown.

Runner selection is implemented by
`vllm_ascend/mrv2_utils.py` and installed by
`vllm_ascend/patch/platform/patch_use_v2_model_runner.py`. MRV2 is the
default; `VLLM_USE_V2_MODEL_RUNNER=0` forces V1. The V2 blacklist still
contains the VL encoder graph (`compilation_config.cudagraph_mm_encoder`), so
an encoder-graph deployment falls back to V1 with a warning.

Encoder-only **pooling** models (BERT/BGE/SigLIP2 with `--runner pooling`) are
a separate feature: they are served as pure embedding models and have no EPD
topology. See the [SigLIP2 guide](../../tutorials/models/SigLIP2.md).

## Interface Design

### CLI: `--ec-transfer-config`

```json
{
  "ec_connector": "<connector name>",
  "ec_role": "ec_producer | ec_consumer | ec_both",
  "ec_connector_extra_config": {"<key>": "<value>"},
  "ec_connector_module_path": "<python module path, optional>"
}
```

- `ec_connector`: registered connector name. Ascend registers its
  implementations in `vllm_ascend/distributed/ec_transfer/__init__.py`.
- `ec_role`: producer (E), consumer (P/PD), or both (single-instance cache).
- `ec_connector_extra_config`: connector-specific keys, listed below.
- `ec_connector_module_path`: when the connector class is not found by name in
  the upstream factory (used by `ECMemcacheConnector`).

Instances that must share cache entries use the same connector and extra configuration.

### Connector extra configuration

| Connector | Keys | Notes |
| --- | --- | --- |
| `ECExampleConnector` | `shared_storage_path` | Directory on a shared filesystem; one safetensors file per `mm_hash` |
| `ECMooncakeConnector` | `mooncake_protocol`, `ascend_mooncake_bounce_arena_size` | `mooncake_protocol` must be `ascend` when set (it defaults to `ascend`); the bounce arena size is optional and is rounded to the direct-memory alignment |
| `ECCPUConnector` | `ec_cpu_bytes` | Pinned `/dev/shm` mmap size; must fit at least one encoder-cache block; single host only |
| `ECMemcacheConnector` | `ec_connector_module_path`, `recv_buffer_tokens` | `ec_connector_module_path` must point to `vllm_ascend.distributed.ec_transfer.ec_memcache_connector`; `recv_buffer_tokens` defaults to 8192 and must be between 2048 and `max_num_batched_tokens`; needs the `memcache_hybrid` package and `MMC_LOCAL_CONFIG_PATH` |

The Ascend ExampleConnector adaptation (`vllm_ascend/patch/platform/patch_example_connector.py`) pins the load device on TP>1 consumers; other paths keep upstream behavior.

### Runner selection

| Environment variable | Effect |
| --- | --- |
| unset | Model Runner V2 by default, unless the config hits the V2 blacklist |
| `VLLM_USE_V2_MODEL_RUNNER=1` | Force Model Runner V2 |
| `VLLM_USE_V2_MODEL_RUNNER=0` | Force Model Runner V1 |

### Reference proxy options

The reference proxy (`examples/disaggregated_encoder/disagg_epd_proxy.py`)
accepts these opt-in options; the defaults keep the previous behavior:

| Option | Default | Meaning |
| --- | --- | --- |
| `--encode-timeout-seconds` | `0` | Per-request encode timeout; `0` omits the per-request timeout so the shared 100000-second client timeout applies |
| `--prefill-timeout-seconds` | `0` | Per-request prefill timeout |
| `--decode-timeout-seconds` | `0` | Per-request decode timeout, including streaming decode |
| `--encoder-failure-cooldown-seconds` | `30` | Window in which a failed encoder URL is deprioritized when several encoders are configured; `0` or negative disables health-aware ordering |
| `--probe-encoders-on-startup` | off | Probe each encoder `/health` on startup (5 s per probe) and fail fast when unreachable |
| `--shutdown-drain-seconds` | `0` | Wait up to this long for in-flight requests during shutdown before closing sessions; `0` closes immediately |

## Specifications Design

### Idempotency

- Cache writes are expected to be idempotent: writing the same `mm_hash`
  again overwrites the same per-key entry (for `ECExampleConnector`, the same
  safetensors file under `shared_storage_path`). A retried encode therefore
  does not corrupt the cache.
- The key is the content hash, so two requests for the same image share one
  entry regardless of which E instance served them.
- Duplicate multimodal items inside one request are deduplicated per request.

### Retry semantics

- The reference proxy makes up to three attempts per stage request: the
  initial attempt plus up to two retries.
- Connection errors map to 502, stage timeouts to 504, and downstream 5xx
  responses to 502; all of them are retried. Every 2xx response is success.
- Downstream 4xx responses keep their status and are not retried.
- When several encoder URLs are configured, connection/timeout/5xx failures
  remove the encoder URL from selection for
  `--encoder-failure-cooldown-seconds`; single dispatch and fanout rotation
  then use only the healthy subset, and the proxy falls back to the full list
  only when every encoder is inside the cooldown window. 4xx responses are
  request-side errors and never mark an encoder unhealthy.
- Streaming requests are not retried after the first chunk is forwarded;
  non-streaming requests are retried normally. Encoder and prefill stages are
  never streamed, so they are always retried as non-streaming requests.
- When the attempts are exhausted the client receives HTTP 500.

### Timeouts

Each stage can get its own per-request timeout through
`--encode-timeout-seconds`, `--prefill-timeout-seconds`, and
`--decode-timeout-seconds` (the last one also covers streaming decode). A
value greater than zero bounds that stage request. The default `0` omits the
per-request timeout so the shared overall client timeout
(`_LEGACY_TOTAL_TIMEOUT_SECONDS = 100_000`) configured on the sessions applies.
Omitting the keyword matters: passing `timeout=None` to aiohttp would replace
the session timeout with `ClientTimeout(total=None)` and disable it.

### Concurrency and crosstalk

- In `fanout` dispatch mode the proxy derives a child request ID per
  multimodal item for encoder requests; in `single` mode the encoder request
  and the prefill request reuse the parent request ID. Because the cache key
  is `mm_hash` and not the request ID, retries and concurrent child requests
  cannot collide on the cache entry.
- The encoder stage refreshes the cache before the request reaches D, so a
  concurrent request for the same image either finds a completed entry or
  reaches the model with a cache miss. On a miss the behavior depends on the
  connector: `ECMemcacheConnector` falls back to local encoding and logs an
  `EC LOAD miss` warning, while `ECExampleConnector` requires the entry to
  exist and fails the request when it is missing (see the user guide).

## DFX Analysis

### Configuration validation

- The connector name must be registered; the upstream factory raises a clear
  error for unknown names.
- Connector-specific values are validated when the connector is created, for
  example `ec_cpu_bytes` must be a positive integer, `mooncake_protocol` must
  be `ascend`, and `recv_buffer_tokens` must be within its range.
- `ec_role` must be a role accepted by the connector; `ECCPUConnector`
  currently requires `ec_both`.

### Port conflicts

The E, P, D, and proxy services each bind their own port. The examples use
distinct, overridable ports for every instance; colliding with an already bound
port makes the instance fail at startup, so check the port list before
launching.

### Health checks

- Each vLLM instance exposes `GET /health` and returns 200 only when the engine
  is ready.
- The reference proxy (`examples/disaggregated_encoder/disagg_epd_proxy.py`)
  exposes `GET /health`, which aggregates the three URL groups and returns
  `{"proxy": "healthy", "encode_cluster": ..., "prefill_cluster": ...,
  "decode_cluster": ...}`; the HTTP status is 200 when no cluster is unhealthy
  and 503 otherwise. A cluster with no configured instances reports
  `empty`; a configured but unreachable instance reports `unhealthy`.
- The layerwise proxy used by the E+P+D guide only exposes `GET /healthcheck`,
  which reports the proxy status and connected instance counts without
  aggregating instance health.
- With `--probe-encoders-on-startup`, the proxy probes every encoder
  `GET /health` once during startup (5 s per request) and raises before
  serving when any instance is unreachable; the default is off.
- Startup order is E, then P/D, then the proxy, waiting for `/health` first.

### Observability

The reference proxy exposes Prometheus text metrics at `GET /metrics`
(implemented without extra dependencies):

| Metric | Type | Meaning |
| --- | --- | --- |
| `epd_requests_total{stage,status}` | counter | stage requests by stage and outcome (`ok`, `http_4xx`, `http_5xx`, `timeout`, `connection`, `error`) |
| `epd_stage_seconds_bucket/sum/count{stage}` | histogram | stage request latency |
| `epd_retries_total{stage}` | counter | retries performed per stage |
| `epd_encoder_requests_total` | counter | encoder requests sent |
| `epd_mm_items_total` | counter | multimodal items forwarded to encoders |
| `epd_requests_inflight` | gauge | client requests currently being processed |

`/health` keeps its existing contract; the metrics endpoint is additive.

### Shutdown drain

With `--shutdown-drain-seconds` greater than zero the shutdown hook waits
until `epd_requests_inflight` reaches zero or the window expires (logging the
remaining count), then closes the client sessions. The gauge is decremented at
the ASGI response boundary (a tracked streaming response), so a client that
disconnects before the first streamed chunk cannot leak it. With the default
`0` the sessions close immediately, matching the previous behavior.

### Logging

The proxy logs a request ID for every client request and each stage transition,
retry, and failure. Each vLLM instance keeps its own log; the E+PD example
script (`disagg_1e1pd_example.sh`) redirects them to files under `LOG_PATH`.

## Limitations

- **VL encoder graph**: `compilation_config.cudagraph_mm_encoder=True` is in
  the V2 blacklist and forces MRV1; MRV2 does not support the encoder graph.
- **EC + Pipeline Parallel** is unverified (marked unknown in the feature
  matrix).
- **ECCPUConnector** supports a single host only; it is not a cross-node EPD
  connector.
- **ECMooncakeConnector** needs a Mooncake deployment and shared storage.
- **ECExampleConnector** needs a shared filesystem with identical absolute paths
  on E and P/D.
- **E+P+D** is a supported design with example scripts, but end-to-end test
  coverage currently exists for E+PD only; MRV1 EPD and E+P+D scenarios are
  still being added.
