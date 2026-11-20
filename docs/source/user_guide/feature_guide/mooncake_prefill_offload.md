# Mooncake Prefill KV Buffer Reuse

Prefill producers can opt into reusing a bounded ring of NPU KV buffers across
layers. Mooncake holds the full cache; each layer restores the previous chunk,
updates its cache and saves it before the next owner overwrites the buffer.
This reduces the physical KV allocation at the cost of additional Store traffic.

The offload path targets GLM-5.3-Flash with TP8 and expert parallelism. GLM main
MLA/KDA components can rotate through shared slots while compressed indexer,
tail and draft caches remain resident. Target KDA state can also reuse slots
with native MTP enabled. Model-registered eligibility rules preserve GLM's
native page strides, padding and component sizes.

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

The Store connector can run alone or with the mainline `MooncakePushConnector`
in a `MultiConnector`. Both P and D must select Push and use the D-first
layerwise proxy. Bulk `MooncakeConnectorV2` / Pull cannot release source slots
during forward and is rejected with shared buffers. Omitting
`layerwise_num_shared_buffers` retains ordinary Mooncake layerwise pooling.

The offload bridge registers Push as a slot-release provider. A reused layer's
Store load waits for the predecessor's Store save and all corresponding Push
WRITEs. Push waits for destination metadata before publishing these WRITE
dependencies, so late discovery cannot submit a source read after slot release.
Target/draft readiness keeps mainline's namespace-aware layer identity;
offload maps cache names separately to the physical slot indices. Store
prefix hits become Push-readable only after their layer load and write hook.

The shared Push integration from [PR #21891](https://github.com/vllm-project/vllm-ascend/pull/21891)
uses the registered KDA cache name to publish WRITE dependencies. Missing
shared-layer save hooks fail the transfer.

P/D examples are in
`examples/disaggregated_prefill_v1/glm53_mooncake_offload/`.
Start with P PP=PCP=DCP=1; Push with shared buffers currently rejects PP/PCP
greater than one. The mainline Push restriction on P preemption still applies.
This bridge does not reimplement P/D routing or the Push protocol.

For the initial GLM-5.3-Flash validation, keep the model's quantization and
hardware-specific options and use `--tensor-parallel-size 8
--enable-expert-parallel --enable-chunked-prefill --mamba-cache-mode align
--enforce-eager`. EP uses the TP8 group in this configuration. Native MTP can use
`--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":3,"enforce_eager":true}'`.
Target KDA caches use exact execution snapshots; each transferred state page
retains its native speculative-state contents for the model's align pre-copy.
Set `layerwise_reuse_mtp_state=false` in the Store extra config to retain the
previous resident target-KDA behavior. Native MTP draft caches remain resident, participate in
external prefix loading, and complete their part of the Mooncake block object
before it becomes readable. Later proposal iterations do not advance the
completed target/offload hook sequence.
The shared-KDA path with MTP is for prefill. Requests that also perform
speculative decode commits on this instance must set
`layerwise_reuse_mtp_state=false`, since acceptance happens after the layer-save
hook. Non-speculative running steps retain exact endpoint snapshots.
Start with PP=PCP=DCP=1 and synchronous scheduling. The existing layerwise
parallelism constraints still apply; other parallel topologies need separate
hardware validation. Set the load failure policy to `fail` for hybrid models.

`layerwise_num_shared_buffers` bounds the reusable slots in each block-stride
class. A configured independent layer pins its cache components. The planner
packs components from different groups only when their block IDs are disjoint;
same-group components used by the same layer must occupy different slots.
Resident and rotating windows occupy one device allocation. Resident windows
still consume HBM, so total use is not simply the ring-slot count times one
layer's KV. Too few slots can serialize computation and transfers;
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
states. Reused KDA saves a private execution snapshot at each endpoint, including
13824 when an older request already published the same public block hash.
The latter two are never floored to an older checkpoint. Speculative verifier
placeholders beyond the real input token sequence are not new snapshot tokens.
Missing exact snapshots still fail the request;
recovery never silently skips a shared recurrent restore or rolls it back from
a later state. Private snapshots become evictable Store objects;
size the Store for in-flight snapshots as well as shared prefixes. If a required
snapshot is evicted, the step fails instead of consuming overwritten NPU bytes.

GET leases are released after the step. PUT allocation is batched across requests
and groups, and the next buffer owner waits for the preceding layer's copy.
An object becomes visible only after its final group layer commits. Failed PUTs
or commits stop reuse. TP ranks synchronize between chunks so replicated-cache
readers cannot race their writer. This barrier is part of the performance cost.
For registered component layouts, local prefix hashing remains enabled when
prefix caching is configured. Before accepting an HBM hash hit, the scheduler
asks the Store connector for a verified restart boundary. Shared scratch bytes
are restored layer by layer even if the local blocks need no new allocation.
A Store miss caps the local hit at zero. Legacy scratch layouts without this
verification continue to reject local hits.

For a repeated 16395-token prompt with block size 1152, a verified 13824-token
prefix can reuse existing local block IDs, then restore the shared KDA/MLA
payload and compute the remaining 2571 tokens. For a 16128-token prompt, the
restart limit is 14976: the last token must still run through the model, and KDA
at 16128 cannot be truncated into KDA at 14976. The Store lookup verifies that
actual earlier endpoint. A local hash alone never establishes KDA residency.

## Component Planning

Mooncake and MemCache share the eligibility registry, byte-window planner,
worker allocation and block sizing from
[PR #21256](https://github.com/vllm-project/vllm-ascend/pull/21256). The Mooncake
block-key plane opens batched GET/PUT sessions and commits immutable snapshots.

GLM first creates its native cache descriptors. The registered rules allow
target main MLA and align-mode KDA, keep indexer/tail and draft entries resident,
and exclude explicitly independent layers. The planner validates group identity,
preserves each native window's stride and size, and assigns compatible windows
to reusable slots. Other models need their own eligibility rules and native
layout validation before entering this planner; this is not automatic support
for every attention implementation.

For example, if a layer reuses one component last written by layer 1 and another
last written by layer 3, its dependency is `(1, 3)`. It waits for both Store saves
and both Push source-read completions before loading over those windows. Merely
waiting for layer 3 is insufficient: layer 3 might have had no Store work while
layer 1's asynchronous save was still reading its slot.

The physical budget counts the shared backing once. After automatic block
sizing, the planner rebuilds offsets for the chosen block count; multiplying
descriptor capacities alone would leave overlapping offsets. Unknown specs keep
their native layout. The opt-in component path currently rejects host-backed
caches and `num_gpu_blocks_override`.

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
Mooncake stores for cold-cache comparisons. Use automatic block sizing and
record both allocated block counts and physical backing bytes. Compare capacity
at equal HBM budgets, then tune memory budgets to compare equal logical capacity;
the component planner rejects `--num-gpu-blocks-override`.

With baseline and candidate endpoints serving the same model name:

```bash
python tests/SIT/cases/features/kv_pool/verify_mooncake_offload.py \
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
