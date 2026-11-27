# HiSparse on Ascend

## Feature Introduction

HiSparse keeps a Host mirror of sparse MLA KV rows and refills only the missing
rows selected by attention. This reduces the NPU cache capacity needed for long
contexts. The experimental Ascend backend supports A2/A3 through the arch22
resolver and 950PR&950DT Products through the SIMT resolver. 950PR&950DT Products require a SIMT-capable CANN 9.2
toolchain; build the custom operators for the target architecture.

### Working Principle

HiSparse keeps recent KV rows in the NPU Resident cache and uses the NPU Hot
cache for rows refilled from Host memory. The resolver checks the selected rows
against both NPU caches. Resident and Hot
hits can be read directly. For misses, HiSparse refills only the missing rows
from the Host KV mirror into Hot before attention runs. With speculative decoding,
the resolver considers the selected rows for all query tokens of a request
together before evicting Hot rows.

### Usage Scenarios

Use HiSparse with Model Runner V2 for sparse RoPE MLA/SFA models serving long
contexts. A2/A3 support INT8 C8 KV: Main KV stores INT8 values, BF16 RoPE and
FP32 scales; Indexer KV stores INT8 values with FP16 scales. 950PR&950DT Products retain BF16 KV
and native C8 (E4M3FN values with FP32 scales), with independent Main and
Indexer KV dtype selection. The A2/A3 extension does not add a BF16 HiSparse
serving path or change the existing non-HiSparse BF16 path.

On a single Ascend 950 node with colocated Prefill and Decode, you can
[combine HiSparse with AscendStore](#combining-hisparse-with-ascendstore)
to offload historical KV to local DDR and reuse full KV blocks from MemCache
across requests.

### Constraints and Limitations

| Category | Constraint |
| --- | --- |
| Hardware and software | A2/A3 with the arch22 custom operators, or 950PR&950DT Products with a SIMT-capable CANN 9.2 toolchain. |
| Cache formats | A2/A3: INT8 C8. 950PR&950DT Products: BF16 or native E4M3FN C8. C4 and TurboQuant are outside the current support scope. |
| Parallelism and transport | Supports PCP (prefill and decode request sharding). DCP and external P/D transport are outside the current support scope. |
| Host storage | Use private Host pools per tensor-parallel rank and Core eager Host mirroring. |
| TopK | The existing SFA requirement of divisibility by 128 applies. |
| Feature mutual exclusion | HiSparse and `additional_config.sparse_kv_offload_config.enabled` cannot be enabled together. |

## Feature Usage

### Usage Example

Set `MODEL_PATH` to a compatible sparse MLA model and start the server with
Model Runner V2. The following example uses BF16 KV on 950PR&950DT Products:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve "$MODEL_PATH" \
  --block-size 128 \
--kv-cache-dtype bfloat16 \
--attention-config '{"indexer_kv_dtype":"bf16","hisparse_config":{"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4}}'
```

For native C8 on 950PR&950DT Products, replace the two dtype options with:

```bash
--kv-cache-dtype fp8 \
--attention-config '{"indexer_kv_dtype":"fp8","hisparse_config":{"eager_host_mirror":true}}'
```

For A2/A3 INT8 C8, use a compatible INT8 KV model and replace the two dtype
options with:

```bash
--kv-cache-dtype int8 \
--attention-config '{"indexer_kv_dtype":"int8","hisparse_config":{"eager_host_mirror":true}}'
```

These options select KV-cache quantization independently of model-weight
quantization; keep the weight-quantization options required by the model.

On A3 with HDK 25.5.1, select MemFabric in the HiSparse connector to provide
AIV Host-memory mapping. The SDK must include the A3 AccOffload extension.
If the extension is missing, source the CANN environment and build it with
`mfcli kernel install --soc-version A3`.

```bash
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4,"transfer_backend":"memfabric"}}'
```

For a `MultiConnector` configuration, put `transfer_backend` in the
`HiSparseConnector` child's extra configuration. This setting selects
the private Host allocator. Omitting it uses native ACL Host-memory registration.

### Combining HiSparse with AscendStore

AscendStore saves and restores Host Source and NPU Indexer pages. HiSparse
manages the Host Source, Resident, and Hot buffers.

Use the vLLM Core revision pinned by the matching vLLM Ascend build with
Ascend HiSparse support. Keep the attention options above and pass the following
JSON to `--kv-transfer-config`.

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "HiSparseConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"host_pool_gib": 4}
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"backend": "memcache", "load_async": true}
      }
    ]
  }
}
```

See [KV Cache Pool](kv_cache_pool_ascend_store.md) for MemCache setup and
[AscendStore settings](#ascendstore-settings) for the required options.
See [Monitoring AscendStore prefix reuse](#monitoring-ascendstore-prefix-reuse)
for the metrics to check after startup.
Mooncake support is deferred pending a fix for
[PUT timeout data corruption](https://github.com/kvcache-ai/Mooncake/issues/4491).

AscendStore keys for HiSparse use the `hisparse-ascendstore-v1` namespace.
This version identifies the external pool key format. The keys include the
target and draft checkpoint identities, resolved revisions, dummy-weight mode,
filename patterns for `sharded_state` and `runai_streamer_sharded`, effective
model configurations (including RoPE settings), and page layout. If the Hub revision
resolver is unavailable, pin the checkpoint commit when sharing a persistent
pool. Use a versioned directory when replacing local weights in a persistent
pool. Files overwritten in place keep the same model path.

### Verifying the Feature

Send a request and check the server's metrics endpoint for the HiSparse miss
counter. The counter can stay at zero when selected rows are already on the NPU.

```bash
curl -fsS http://127.0.0.1:8000/metrics | grep '^vllm:hisparse_cache_misses_total'
```

#### Monitoring AscendStore prefix reuse

Check the server's `/metrics` endpoint for
`vllm:external_prefix_cache_hits_total` to track prefix hits in MemCache and
`vllm:ascend_store_load_get_keys_total` to track keys read by AscendStore.
A request served from the local prefix cache may leave both counters unchanged.

## Configuration Parameters

| Parameter | Description |
| --- | --- |
| `hisparse_config` | Enables Core HiSparse for the supported sparse Attention backend. |
| `eager_host_mirror` | Keep `true` for this foundation; Core maintains the Host KV mirror. |
| `host_pool_gib` | Host pool size in GiB per tensor-parallel rank. Size it for the planned context lengths and concurrency; the example reserves 4 GiB per rank. |
| `device_buffer_size` | Per-request Hot capacity in KV rows. It must hold the deduplicated union of rows selected during speculative verification and fit the resolver's on-chip UB budget. |
| `kv_cache_dtype` / `indexer_kv_dtype` | Select main and Indexer KV formats independently from model-weight quantization. |

### AscendStore settings

| Parameter | Description |
| --- | --- |
| `kv_connector` / `kv_role` | In `--kv-transfer-config`, use `MultiConnector` with one `HiSparseConnector` and one `AscendStoreConnector`. Set `kv_role=kv_both` on the parent and both children. |
| `backend` / `load_async` | In the AscendStore child's `kv_connector_extra_config`, set `backend=memcache` and `load_async=true`. |
| `use_layerwise` / `discard_partial_chunks` | In the AscendStore child's `kv_connector_extra_config`, keep `use_layerwise=false` and `discard_partial_chunks=true` for whole-block transfers. |
| `protocol` | Set to `host_tcp` in the MemCache local configuration. |
| `hisparse_config.eager_host_mirror` | Set to `true` in `--attention-config`, including for Prefill-only saves. |
| `prefill_context_parallel_size` | Keep at `1`. |

To fall back to local computation after an external load fails, set
`kv_load_failure_policy=recompute` in `--kv-transfer-config`.
