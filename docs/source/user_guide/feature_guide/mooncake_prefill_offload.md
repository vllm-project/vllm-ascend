# Mooncake Prefill KV Buffer Reuse

GLM-5.3-Flash prefill producers can rotate MLA and aligned KDA caches through
shared NPU buffers. Mooncake Store holds the pages between layers and chunks.
Indexers, tails, MTP draft caches and explicitly independent layers stay
resident. Mooncake and MemCache use the same cache planner and allocator
introduced by [PR #21256](https://github.com/vllm-project/vllm-ascend/pull/21256).

## Configuration

Configure Mooncake as described in [KV Cache Pool (Ascend Store)](kv_cache_pool_ascend_store.md).
Enable the following on the producer's `AscendStoreConnector`:

```json
{
  "backend": "mooncake",
  "use_layerwise": true,
  "load_async": false,
  "layerwise_num_shared_buffers": 4,
  "layerwise_prefetch_layers": 2,
  "layerwise_independent_layers": [0]
}
```

Use `kv_role=kv_producer` and `kv_load_failure_policy=fail`. Omit
`layerwise_num_shared_buffers` to use ordinary Mooncake layerwise pooling.
GLM-5.3-Flash setup starts with TP8, expert parallelism, PP=PCP=DCP=1,
synchronous scheduling, chunked prefill and `--mamba-cache-mode align`.
Native MTP is supported during prefill. Run speculative decode on a
separate decoder with shared buffers disabled.

For disaggregated P/D, pair AscendStore with a Mooncake Push child in
`MultiConnector` on the producer. Select Push with `MooncakePushConnector`
or with `MooncakeConnectorV2` and `use_layerwise: true` in its connector
extra config. Use Push on the decoder and the D-first layerwise proxy.
Example configurations are in
`examples/disaggregated_prefill_v1/glm53_mooncake_offload/`.

## Cache lifecycle

Before overwriting a shared slot, the next layer waits for Store to save
every predecessor that owns bytes in that slot and for Push to finish its
source reads. The named KDA save hook and Push fencing follow
[PR #21891](https://github.com/vllm-project/vllm-ascend/pull/21891).

Mooncake opens batched GET/PUT sessions before the layer hooks and commits
objects after their final group layer. Private snapshots include the
request incarnation and exact token endpoint. This lets a subsequent chunk
restore its KDA state even when the same public prefix already exists.
Missing snapshots and failed writes abort the step.

The scheduler verifies the exact restart boundary in Store before
accepting a local prefix hash hit. On a hit, it can retain local block IDs
and restore shared pages layer by layer. For a full-prompt hit, it verifies
the earlier restart boundary where the model can resume computing the
prompt tail.

## Hardware validation

Compare buffer reuse disabled and enabled with the same model, scheduler,
parallelism and physical KV memory budget. Use automatic block sizing.

```bash
python tests/SIT/cases/features/kv_pool/verify_mooncake_offload.py \
  --baseline-url http://BASELINE:8000/v1 \
  --candidate-url http://CANDIDATE:8000/v1 \
  --model glm-5.3-flash --concurrency 4 --repeat 3 --max-tokens 1
```

Cover cold and warm prefixes, multiple chunks, aligned and unaligned
endpoints, native MTP and transfer failures. Record peak HBM, KV block
capacity, TTFT, throughput and Store traffic. Actual Ascend P/D validation
is pending.
