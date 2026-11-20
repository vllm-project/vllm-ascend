# DeepSeek-V4 layerwise KV offload

DeepSeek-V4 Flash and Pro can rotate compressed history, indexer history, SWA, and compressor state through shared device buffers. AscendStore saves and restores those pages through the Memcache backend. The common planner retains the model's native cache groups and block IDs.

## Configure rotation

Set up Memcache using the [offloading guide](layerwise_and_sparse_kv_cache_offloading.md). For a single-instance deployment, pass the following as `--kv-transfer-config`.

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "backend": "memcache",
    "use_layerwise": true,
    "layerwise_num_shared_buffers": 3,
    "layerwise_prefetch_layers": 2,
    "layerwise_independent_layers": [0, 1]
  }
}
```

Omit `layerwise_num_shared_buffers` for a baseline with layerwise pooling and the native device layout. Independent layer IDs refer to the model's layer order. The value `"all"` keeps every layer resident. Use eager execution or `PIECEWISE` graphs for rotating caches.

For P/D disaggregation, configure a producer `MultiConnector` with the layerwise `AscendStoreConnector` and `MooncakePushConnector`. Keep the Decode cache resident. Push starts while the source pages are valid and the store waits for the source read before reusing their slot. Use matching P/D cache formats and context-parallel sizes. MTP draft caches remain resident. The rotating DSpark path requires a Push producer and eager draft writes. The DSV4 rotation guard rejects TurboQuant and FULL graph modes.

## Cache reuse

Use a fresh pool when upgrading from the earlier experimental A5 indexer format. The layerwise Memcache format distinguishes compatible component layouts. A5 transfers each indexer data and scale page once. Resident and rotating runs can use the same public prefix identity when the logical cache format matches. A request's continuation snapshots are scoped to that request. Retained public checkpoints are independently available for other requests.

One complete TTFT comparison uses the same model, topology, workload, and memory utilization for native and rotating layouts. Report actual block counts, HBM and external hit counters, and unprofiled TTFT. Single-instance Decode performs layerwise restores too, so report its TPOT separately.
