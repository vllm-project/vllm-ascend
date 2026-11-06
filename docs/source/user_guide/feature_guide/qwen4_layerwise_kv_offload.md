# Qwen4Exp layerwise KV cache reuse

Qwen4Exp supports experimental layerwise KV cache reuse through Memcache GVA.
It reduces device storage per block by restoring and saving rotating caches
between physical layers, preserving their storage keys and six KV cache groups.

This layout is separate from Mooncake block-key layerwise pooling and the
[sparse Decode offload deployment](layerwise_and_sparse_kv_cache_offloading.md).

## Configuration

This implementation requires the companion vLLM KVCacheSpec.reuse,
KVCacheTensor.backing_id and per-name group-intent fields in vLLM.

Use model runner V2 and configure Memcache as described in the
[service preparation guide](layerwise_and_sparse_kv_cache_offloading.md#1-preparation).
Set `MMC_LOCAL_CONFIG_PATH` to the existing service configuration. Add:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve /path/to/qwen4exp \
  --enforce-eager \
  --block-size 768 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --kv-transfer-config '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "backend": "memcache",
      "use_layerwise": true,
      "layerwise_num_shared_buffers": 2,
      "layerwise_independent_layers": [0, 1]
    }
  }'
```

Retain the model's normal quantization, TP, EP and memory options. The command
illustrates cache configuration; it does not size the model's weights or host
pool. Pure `kv_consumer` use requires `consumer_is_to_put: true`, because every
rotating owner must be saved before its storage can be reused.

`layerwise_independent_layers` uses physical decoder layer indices. Raw QSA
ring buffers and PLE remain resident regardless of this list. Main and
compressed QSA caches rotate together; GDN conv and SSM state share a slot as
a pair. The model declares GDN layers 0 and 1 as resident. Marking another
reuse-requesting layer independent is a configuration conflict and raises an
error; it does not silently disable that layer's reuse. Disable layerwise
to retain the ordinary allocation.

`layerwise_num_shared_buffers` controls reusable slots. Optional
`layerwise_prefetch_layers` controls the submission window; actual reuse still
waits for all predecessors. Disabling `use_layerwise`, or selecting a backend
without GVA reuse, retains the existing allocation path.

## Capacity and host memory

Capacity is solved against the complete physical layout before scheduling.
Worker allocation rebuilds that layout at the already selected block count;
workers cannot increase it. When TP ranks have unequal budgets, all ranks use
the minimum block count and rebuild their offsets at that capacity.

Backing sizes follow the selected block count; unused profiled budget is not
reserved as padding. The existing allocator adds alignment padding to each
backing, so raw storage bytes include that allocation overhead.

For the documented 48-layer fixture, a 12,556,015,104-byte budget with two
shared buffers and independent layers `[0, 1]` gives 3,648 manager blocks,
two backing allocations. The descriptor count follows the physical slots. The cost is 3,441,792 bytes
per block. Actual capacities depend on model dimensions, dtype and available
memory. An explicit `--num-gpu-blocks-override` exceeding the budget fails
before allocation.

Host/store capacity must hold the **logical per-layer history**, including
main and compressed QSA data and GDN conv/SSM state. It must not be sized from
the reduced device slot count. Increasing device block capacity also increases
the possible host working set. Monitor host memory and transfer latency when
sizing the pool; increased capacity does not imply increased throughput.

## Validation boundary

Active reuse requires eager execution. PP, DCP, PCP, KVPP and speculative
decoding are rejected when reuse is active. Graph replay and disaggregated
remote readers need their own end-to-end validation before deployment.
The implementation retains the existing external-reader release callback and
waits for it before overwriting a reused region.

A failed restore or incomplete save aborts the step: a completion notification
alone never authorizes reuse after a transfer failure. Scheduler block
references remain retained until every worker reports completed saves.

See the [layout design](../../developer_guide/Design_Documents/qwen4_layerwise_kv_offload.md)
for the allocation contract and tests.
