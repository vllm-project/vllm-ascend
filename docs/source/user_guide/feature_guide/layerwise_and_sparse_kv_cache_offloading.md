# Layerwise and Sparse KV Cache Offloading Guide

Layerwise and sparse KV cache offloading are supported only in disaggregated
Prefill/Decode (PD) deployments. Colocated Prefill/Decode serving is not supported.

This guide explains how to configure:

- Layerwise KV cache offloading during the Prefill phase
- Sparse KV cache offloading during the Decode phase
- Combining both features in a disaggregated Prefill/Decode deployment

For the underlying architecture and implementation details, see
[Layerwise and Sparse KV Cache Offloading Design](../../developer_guide/Design_Documents/layerwise_and_sparse_kv_cache_offloading.md).

## Supported Models

The combined deployment currently supports the following sparse-attention
model families:

- [GLM-5.1](../../tutorials/models/GLM5.md)
- [GLM-5.2](../../tutorials/models/GLM5.2.md)
- [DeepSeek-V3.2](../../tutorials/models/DeepSeek-V3.2.md)

Other sparse-attention models have not been validated.

## 1. Preparation

### Prefill Nodes

=== "A3 series"

    Prefill offload relies on memcache service. Here we provide a simple guidance to launch memcache service. Configure `mmc-meta.conf`:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_HOST>:5000
    ock.mmc.meta_service.config_store_url = tcp://<CONFIG_STORE_HOST>:6000
    ock.mmc.meta.lease_ttl_ms = 30000
    ock.mmc.log_level = error
    ```

    Configure `mmc-local.conf` on every Prefill node:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_HOST>:5000
    ock.mmc.local_service.config_store_url = tcp://<CONFIG_STORE_HOST>:6000
    ock.mmc.log_level = error
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_sdma
    ock.mmc.local_service.dram.size = 10GB
    ```

    Start MetaService in a separate process:

    ```bash
    export MMC_META_CONFIG_PATH=/path/to/config/mmc-meta.conf
    python -c "from memcache_hybrid import MetaService; MetaService.main()"
    ```

    Prepare every Prefill node before starting vLLM:

    ```bash
    export MMC_LOCAL_CONFIG_PATH=/path/to/config/mmc-local.conf
    ```

### Decode Nodes

There is no extra work to do on decode nodes.

## 2. Layerwise KV Cache Offload on Prefill

Requirements and limitations:

- Shared-buffer Layerwise Prefill Offload requires Memcache and eager mode.
- `SfaRemoteD2HConnector` is the only supported connector to be used with prefill offload and decode offload.

For a combined deployment, Prefill TP must be greater than or equal to Decode
TP and divisible by it.

Add the following options to the Prefill launch command. `MultiConnector` lets
`AscendStoreConnector` offload layer buffers to Memcache while
`SfaRemoteD2HConnector` exposes the same buffers to Decode:

```bash
--enforce-eager \
--kv-transfer-config '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "SfaRemoteD2HConnector",
                "kv_role": "kv_producer"
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "backend": "memcache",
                    "use_layerwise": true,
                    "layerwise_num_shared_buffers": 3,
                    "layerwise_independent_layers": [0]
                }
            }
        ]
    }
}'
```

Do not set `sparse_kv_offload_config` on Prefill. The
`AscendStoreConnector` entry uses the following buffer options:

| Parameter | Description |
| :--- | :--- |
| `layerwise_num_shared_buffers` | Defaults to the number of cache-bearing layers, so omitting it does not enable cross-layer buffer reuse. Explicitly set a smaller count, such as `3`, to enable reuse. Independent layers keep dedicated buffers in addition to this shared-buffer budget; unused slots are not allocated. |
| `layerwise_independent_layers` | Layers that keep dedicated buffers. The default is `[0]`; `"all"` disables cross-layer reuse. |

The following log confirms that buffer reuse is enabled:

```text
Layerwise KV cache reuse merged ... descriptors into ... descriptors using ... buffer assignments.
```

### GLM-5.3-Flash standalone cache offload

GLM-5.3-Flash has an experimental standalone shared-buffer path with
`AscendStoreConnector`, `backend=memcache`, and `use_layerwise=true`.
One instance handles Prefill and ordinary non-speculative Decode. It requires
model runner V1, eager execution, a store save role (`kv_both` or `kv_producer`),
and DP=PP=PCP=DCP=1. Use BF16 main/indexer KV caches and
`mamba_cache_mode=align` without extra prefill checkpoint blocks.
P/D separation through `MultiConnector` is rejected before cache allocation.

Main MLA and KDA views retain their ordinary cross-group packing. For the
full 45-layer model, set `layerwise_num_shared_buffers=4` and
`layerwise_independent_layers=[0,1,2,3]`: L4-L7 share S0, L8-L11 share S1,
L12-L15 share S2, and L16-L19 share S3. L20-L23 reuse S0; L44 reuses only
S2's first KDA group pages. The four resident target layers retain
their ordinary shared backing, with distinct global block IDs for each group.
This produces five main backings and eleven resident indexer/tail backings.
The planner uses this physical layout to increase `num_blocks` within the KV
memory budget. The worker reserves an additional 2 MiB per backing for address
alignment before planning blocks. Configurations that do not reduce physical
main backings retain the ordinary layout and capacity.

Set `layerwise_prefetch_layers=4` to submit loads up to four logical layers
ahead; this is separate from the number of shared tensors. Loads and cold
writes wait for the previous owner in the same `(tensor, scheduler group)`
to finish its store save: L20 waits for L4, L21 for L5, and so on.
Only that group's allocated pages may be loaded or initialized. Shared
recurrent forwards drain their saves before the next step resets events or
copies state. These waits also protect one-token Decode steps.

Use the following configuration for a single instance:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "backend": "memcache",
    "use_layerwise": true,
    "layerwise_num_shared_buffers": 4,
    "layerwise_prefetch_layers": 4,
    "layerwise_independent_layers": [0, 1, 2, 3]
  }
}
```

Start with `VLLM_USE_V2_MODEL_RUNNER=0`, `--enforce-eager`,
`--mamba-cache-mode align`, and `--kv-cache-dtype auto`. Leave
`--speculative-config` and `--use-replayssm` unset for standalone generation.
Draft descriptors are kept resident and charged to capacity when present,
but speculative Decode commits are unsupported: sampling may roll back the
computed endpoint after the shared state has been saved.

Set a fixed `--kv-cache-memory-bytes` for capacity comparisons and leave
`--num-gpu-blocks-override` unset. Chunked Prefill and ordinary Decode save
request-scoped snapshots at the exact computed endpoint, including partial
blocks and KDA state. Snapshot data is restored before the layer consumes
its shared slot. Decode may finish a full page before its prefix hash is
published: attention pages keep their request-scoped identity until promotion,
and recurrent state restores the exact previous endpoint even after hashes
arrive. An older recurrent checkpoint cannot be reconstructed from later state.
Ordinary `kv_both` Prefill chunks retain cache-group alignment, including the
final full block before the prompt tail, so their KDA checkpoints can be
published for other requests. Only a padded final-token MTP verifier window
bypasses that split. External prefix hit rates count computation avoided across
requests; restoring this request's previous chunk is not another prefix hit.
Local prefix hashes are not treated as resident cache hits
after buffer reuse; prefix reuse is admitted only after the external store
verifies a restore point. Transfer failures stop execution because a reused
slot has no resident fallback copy. This feature does not enable the separate
sparse Decode offload deployment described below.

## 3. Sparse KV Cache Offload on Decode

Requirements and limitations:

- Sparse Decode Offload is currently not supported on 950PR&950DT Products. We are actively working on it to support it in the near future.
- Use disaggregated Prefill/Decode deployment and enable sparse offload only on Decode.
- The main SFA KV cache must use BF16. Packed main KV caches (C8 or TQ4) are
  not supported. INT8 indexer KV is supported separately through
  `--attention_config.indexer_kv_dtype int8`; it does not enable main-cache C8.
- Sparse Decode Offload supports DP and TP on Decode; CP are not supported.
- Sparse Decode Offload and `PreemptOffloadConnector` are mutually exclusive,
  including when the connector is configured inside `MultiConnector`. Use only
  one of these offload mechanisms on a Decode node.

The current decode offload use a different host pool compared to prefill. So the configuration on decode node is slightly different. Add the following options to the Decode launch command:

```bash
--additional-config '{
    "sparse_kv_offload_config": {
        "enabled": true,
        "dram_size_per_dp_GB": 110
    }
}' \
--kv-transfer-config '{
    "kv_connector": "SfaRemoteD2HConnector",
    "kv_role": "kv_consumer",
    "kv_port": 20050
}'
```

On Decode, reserve
`decode_data_parallel_size * decode_tensor_parallel_size` consecutive ports
starting from `kv_port`.

| Parameter | Description |
| :--- | :--- |
| `topk_buffer_size` | Device hot-buffer size. Defaults to `-1`, which resolves to `(1 + num_speculative_tokens) * 2048` (2048 without speculation). Explicit values are unchanged. For the default Copy-SFA path, use a multiple of 256, at least `(1 + num_speculative_tokens) * 2048` and at most 32512. |
| `dram_size_per_dp_GB` | Defaults to `110` GiB of host memory per DP rank. It must hold the full KV cache. TP ranks share this pool while different DP ranks use different pool. The whole required host memory is around `dram_size_per_dp_GB * data_parallel_size`. |
| `keep_device_kv_cache` | Defaults to `false`. Debug-only option that retains the full device KV cache when enabled. |

Note that users might encounter an error like "Sparse KV offload offload.initialize failed.". This is mainly because the contiguous host memory required by memfabric is not enough to allocate. Please refer to [memfabric-prerequisite-checks](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/kv_cache_pool_ascend_store.html#step-1-prerequisite-checks) for preparing the host memory. Or you can consider to use lower DP size to reduce the required host memory.

## 4. Start the P/D Proxy

Start Prefill and Decode with the configurations above. After both nodes are
ready, start the layerwise proxy:

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
    --host 127.0.0.1 \
    --port 9000 \
    --prefiller-hosts 127.0.0.1 \
    --prefiller-ports 8100 \
    --decoder-hosts 127.0.0.1 \
    --decoder-ports 8200
```

For multi-node deployment, advertise reachable addresses instead of
`0.0.0.0`. Send inference requests to the proxy port (`9000` in this example).
