# Qwen4Exp layerwise KV cache reuse

Qwen4Exp has an experimental model-aware layout for the Memcache GVA
layerwise path. It reduces the device storage cost per block by restoring and saving
rotating owners between physical layers. It preserves each owner's logical
storage key and the original six KV cache groups.

This layout is separate from Mooncake block-key layerwise pooling and the
[sparse Decode offload deployment](layerwise_and_sparse_kv_cache_offloading.md).
It does not add GLM model support.

## Configuration

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
ring buffers and PLE remain private regardless of this list. Main and
compressed QSA caches rotate together; GDN conv and SSM state share a slot as
a pair. Independent layers retain private storage. Setting every physical
layer independent preserves the ordinary allocation descriptors.

`layerwise_num_shared_buffers` controls reusable slots. Optional
`layerwise_prefetch_layers` controls the submission window; actual reuse still
waits for all predecessors. Disabling `use_layerwise`, or selecting a backend
without GVA reuse, retains the existing allocation path.

## Capacity and host memory

Capacity is solved against the complete physical layout before scheduling.
Worker allocation rebuilds that layout at the already selected block count;
workers cannot increase it. When TP ranks have unequal budgets, all ranks use
the minimum block count and rebuild their offsets while retaining their own
reserved byte budgets.

The backing allocations reserve the profiled budget even when a block-count
override selects fewer blocks. Reducing that override does not reduce the
reserved storage. The existing transfer allocator also adds alignment padding
to each backing, so raw storage bytes include this allocation overhead.

For the documented 48-layer fixture, a 12,556,015,104-byte budget with two
shared buffers and independent layers `[0, 1]` gives 3,648 manager blocks,
seven descriptors and two backing allocations. The cost is 3,441,792 bytes
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
