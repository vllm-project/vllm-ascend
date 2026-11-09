# Mooncake Prefill KV Buffer Reuse

Prefill producers can opt into reusing a bounded ring of NPU KV buffers across
layers. Mooncake holds the full cache; each layer restores the previous chunk,
updates its cache and saves it before the next owner overwrites the buffer.
This reduces the physical KV allocation at the cost of additional Store traffic.

This implementation targets GLM-5.3-Flash with TP8 and expert parallelism, and
uses the existing cache-spec interfaces for ordinary attention models. GLM main
MLA/KDA backing packs are reused together while compressed indexer and tail
backings remain resident. With speculative decoding, KDA state and draft cache
allocations stay resident.

The implementation has CPU regression coverage. Ascend correctness, graph-mode
compatibility and end-to-end performance require hardware validation before
production use. No throughput or TTFT improvement is claimed from CPU tests.

## Configuration

First configure the services and `MOONCAKE_CONFIG_PATH` as described in
[KV Cache Pool (Ascend Store)](kv_cache_pool_ascend_store.md). Use a Mooncake NPU
client providing the session and range APIs required by
[layerwise pooling](mooncake_layerwise_adaptation_and_optimization.md).

Add the following to the **producer's** `AscendStoreConnector` configuration:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "lookup_rpc_port": "offload-p",
    "use_layerwise": true,
    "load_async": false,
    "layerwise_num_shared_buffers": 4,
    "layerwise_prefetch_layers": 2,
    "layerwise_independent_layers": [0]
  }
}
```

Use the Store connector on its own. P/D transport integration is separate: a
transport that reads these buffers must release each slot before it is reused.
Omitting `layerwise_num_shared_buffers` retains ordinary Mooncake layerwise pooling.

For the initial GLM-5.3-Flash validation, keep the model's quantization and
hardware-specific options and use `--tensor-parallel-size 8
--enable-expert-parallel --enable-chunked-prefill --mamba-cache-mode align
--enforce-eager`. EP uses the TP8 group in this configuration. Native MTP can use
`--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":3,"enforce_eager":true}'`.
KDA state remains resident so speculative acceptance can select its correct
checkpoint after forward. Native MTP caches also remain resident, participate in
external prefix loading, and complete their part of the Mooncake block object
before it becomes readable. Later proposal iterations do not advance the
completed target/offload hook sequence.
Start with PP=PCP=DCP=1 and synchronous scheduling. The existing layerwise
parallelism constraints still apply; other parallel topologies need separate
hardware validation. Set the load failure policy to `fail` for hybrid models.

`layerwise_num_shared_buffers` counts reusable main backing slots. A configured
independent layer pins its whole GLM main pack. Auxiliary backings are additional
resident allocations, so total HBM use is not simply the number of ring slots
times one layer's KV. Too few slots can serialize computation and transfers;
compare 2, 4 and 8 slots with prefetch depths 1 and 2 on the target hardware.

Warmup and profiling use immediate Mamba state copies. They do not register
live deferred copies with the connector, since dummy forward does not execute
the corresponding per-layer load hooks. Real requests retain the check that
every required deferred copy was executed.

## Snapshot Safety and Cost

Full blocks use the existing layerwise pool keys. Incomplete or not-yet-hashed
pages use immutable keys containing a request incarnation and chunk endpoint.
Only that request restores these private snapshots; other requests see committed
full prefixes. Reused recurrent state restores its exact saved endpoint,
including an unaligned prompt tail or a subsequent single-token decode step.
For example, with block size 1152, endpoints 13824, 16395 and 16396 are distinct
states. The latter two use immutable private snapshots; neither is floored to
an older checkpoint. Speculative recurrent state stays resident and is selected
by the model after acceptance. Missing exact snapshots still fail the request;
recovery never silently skips a shared recurrent restore or rolls it back from
a later state. Private snapshots become evictable Store objects;
size the Store for in-flight snapshots as well as shared prefixes. If a required
snapshot is evicted, the step fails instead of consuming overwritten NPU bytes.

GET leases are released after the step. PUT allocation is batched across requests
and groups, and the next buffer owner waits for the preceding layer's copy.
An object becomes visible only after its final group layer commits. Failed PUTs
or commits stop reuse. TP ranks synchronize between chunks so replicated-cache
readers cannot race their writer. This barrier is part of the performance cost.
Local prefix-cache hits are disabled for scratch allocations; the connector
still checks the external pool for reusable prefixes.

`apply_layerwise_kv_cache_plan` runs during cache initialization. GLM capacity
planning now materializes the shared descriptors once; runner initialization
reuses those descriptors directly. Page geometry is computed once per distinct
spec. On the per-step path, block selection is prepared once per group and load
residency class, then shared by all participating layers. Range-address generation
and transfer remain per layer. These changes reduce CPU preparation work but do
not remove network latency or the mandatory buffer fences.

## Hardware Validation

Compare the same model, weights, software versions, chunk size, scheduler and
parallel configuration with buffer reuse disabled and enabled. Use isolated
Mooncake stores for cold-cache comparisons. First keep the same
`--num-gpu-blocks-override` to compare physical allocation at equal capacity;
then remove the override to compare capacity at equal HBM budgets.

With baseline and candidate endpoints serving the same model name:

```bash
python tests/e2e/cases/features/kv_pool/verify_mooncake_offload.py \
  --baseline-url http://BASELINE:8000/v1 \
  --candidate-url http://CANDIDATE:8000/v1 \
  --model glm-5.3-flash --concurrency 4 --repeat 3 --max-tokens 1
```

The script compares generated
tokens, finish reasons and prompt lengths for cold requests, repeated prefixes,
different suffixes and mixed concurrent lengths. Ensure prompts exceed the
configured chunk size and confirm chunking and external hits in engine logs.
Repeat with one shared slot to stress overwrite ordering, with block-aligned and
partial prompt lengths, and with graph mode if that deployment requires it.

Before merge, also validate request cancellation, preemption, Store exhaustion,
failed transfers, and a non-hybrid attention model. Check that requests fail
cleanly on missing snapshots and that leases are released. Record peak HBM,
allocated KV blocks, Store capacity/traffic, TTFT, prefill throughput and p50/p99
CPU preparation latency. Profile initialization separately from per-step
`prepare_layerwise_step`, and include TP barrier and transfer-wait time.
