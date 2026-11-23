# Share perceptual encoder-cache metadata across DP schedulers

`ECMemcacheConnector` can use one shared pHash/SSIM index for all data-parallel
schedulers in a deployment. Memcache stores the resized tensors and encoder
embeddings; the index service stores pHashes, candidate keys, and ranked SSIM
scores. Hash computation, SSIM computation, and Memcache transfers remain in each
scheduler. No tensors are sent through the index RPC.

## Start vLLM normally

No separate service command is required. When a shared endpoint is configured,
the default `shared_index_mode: "embedded"` starts one background metadata thread
inside the serving DP0 scheduler's connector. Every scheduler, including DP0,
uses the same RPC interface. The thread creates and closes its own ZMQ socket;
it performs no tensor transfers, pHash/SSIM computation, or NPU initialization.
The normal vLLM environment supplies `pyzmq` and `msgpack`.

Ownership uses `parallel_config.data_parallel_index`, which remains the serving
rank even when dense models reset their compute DP rank to zero. The server is
not started in every process whose compute rank is zero.

DP0 binds the configured endpoint, so it must identify an interface on the DP0
host. For single-host DP, use `tcp://127.0.0.1:5558`. For multiple hosts, use a
reachable DP0 address such as `tcp://10.0.0.10:5558` in every scheduler. Use a free
port distinct from vLLM's DP coordination and API ports; do not use a wildcard
connect address. On Linux, `ipc:///absolute/path` is also supported for processes
on the same host. The endpoint is intended for a trusted deployment network;
the protocol does not provide authentication or encryption.

DP0 waits for the thread to bind before constructing its client. Other DPs may
start first: connection failures fall back to misses until DP0 becomes ready,
then the clients reconnect. A bind conflict fails DP0 initialization explicitly.

Closing DP0's connector stops and joins its metadata thread. Closing another
scheduler only closes its own client. If DP0 exits early or crashes, remaining
schedulers fall back to misses; there is no ownership election or independent
service process left running. A restarted DP0 starts with an empty index.

## Configure every DP identically

Add these settings to `ec_connector_extra_config.similarity_cache`:

```json
{
  "enabled": true,
  "matcher": "phash_ssim",
  "shared_index_endpoint": "tcp://127.0.0.1:5558",
  "shared_index_namespace": "qwen-vl-deployment",
  "shared_index_timeout_ms": 1000
}
```

Example connector configuration:

```bash
--ec-transfer-config '{
  "ec_connector": "ECMemcacheConnector",
  "ec_connector_module_path": "vllm_ascend.distributed.ec_transfer.ec_memcache_connector",
  "ec_role": "ec_both",
  "ec_connector_extra_config": {
    "similarity_cache": {
      "enabled": true,
      "shared_index_endpoint": "tcp://127.0.0.1:5558",
      "shared_index_namespace": "qwen-vl-deployment",
      "shared_index_timeout_ms": 1000
    }
  }
}'
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `shared_index_endpoint` | `null` | `tcp://` or `ipc://` address; `null` preserves the process-local index. |
| `shared_index_namespace` | `"default"` | Nonempty deployment name, identical on every DP. |
| `shared_index_timeout_ms` | `1000` | Positive 32-bit integer; maximum wait for each send and receive, and reconnect cooldown. |
| `shared_index_mode` | `"embedded"` | DP0 owns a background server thread. Use `"external"` only for an independently managed server. |

Do not add the DP rank to the endpoint or namespace. All DPs must also access the
same Memcache data. Multiple DP schedulers without an endpoint still use separate
local indexes and emit a warning.

## Optional independently managed mode

For an index whose lifecycle is independent of DP0, set
`"shared_index_mode": "external"` on every scheduler and start one service in the
normal vLLM environment:

```bash
python -m vllm_ascend.distributed.ec_transfer.similarity_index_service \
    --endpoint tcp://127.0.0.1:5558 \
    --namespace qwen-vl-deployment
```

In this mode, all schedulers only connect; none starts or stops the service.
The service's namespace must match `shared_index_namespace`. Stop it separately
after the schedulers have stopped. Do not run an external server on an endpoint
that an embedded DP0 is configured to bind.

## Compatibility, capacity, and failure behavior

One service serves one namespace. The first client establishes the model ID,
vision merge size, pHash/SSIM algorithm version, band count, Hamming limit, SSIM
threshold/range, and capacity settings. Every subsequent request validates these
settings, so incompatible schedulers fail explicitly rather than sharing scores.
Use a separate service and namespace for each deployment with different model
weights, model revisions, processor configuration, or embedding layout. The model
ID alone does not fingerprint the contents of a local model directory. This
feature does not change Memcache embedding keys or isolate different models'
embedding stores.

`max_capacity`, `eviction_threshold`, and `lower_watermark` now apply to the
shared index as a whole. Registration, bidirectional score updates, and eviction
are atomic. Removing metadata updates the reverse and band indexes together;
it does not delete Memcache data.

Metadata may be visible before a resized tensor or embedding is ready. Readers
check data availability and fall back to normal encoder computation when needed.
Cached SSIM results also verify embedding availability and skip the query itself.
Concurrent first encounters may still both miss while an embedding is being
computed; the shared index is not an embedding-publication barrier.

A transport timeout discards the socket and starts a cooldown. During an outage,
similarity lookups become misses and the worker still computes/saves embeddings.
An operation can wait up to twice the configured timeout (send plus receive).
The next call after cooldown reconnects. Namespace, model, and configuration
mismatches are errors, not outage fallbacks.

The service is in-memory. Restarting it clears the index, while Memcache data
remains. There is no persistent index recovery or scan of existing Memcache keys;
candidates are learned again from images that enter similarity processing.
Clients do not retain negative lookup results or old remote index snapshots.

## Focused validation

Run the lightweight CPU index tests without the repository's NPU test bootstrap:

```bash
python -m pytest --noconftest -q tests/ut/distributed/test_similarity_index.py
```

When vLLM is absent, this test module supplies a test-only package bootstrap to
skip runtime logging initialization. Production `vllm_ascend/__init__.py` remains
unchanged; the standalone server CLI is intended for the normal vLLM environment.

With the usual vLLM CPU-unit-test dependencies installed, also run the affected
matcher and connector tests:

```bash
python -m pytest -q tests/ut/distributed/test_perceptual_cache_matcher.py \
    tests/ut/distributed/test_ec_memcache_connector.py
```

Before merging, validate real DP=2 execution: save image A on DP0, wait for its
embedding, send a similar image B with a different original hash to DP1, and
check similarity-hit counters and generated output. Measure scheduler latency
and throughput with and without the shared service on the same NPU workload.
