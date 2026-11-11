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

### Selective offload for full/shared SFA layers

Enable `sparse_kv_offload_config` on Decode as above. Models
with full/shared `indexer_types` automatically select this path.
Shared layers store their main KV in host memory. Full layers and the indexer
keep their KV on the device. Use `SfaRemoteD2HConnector` for P/D transfer.
Layers with compatible block geometry share block IDs across the host and
device allocations, so each allocation is sized for one block per token range.

When a full layer produces its TopK indices, a separate stream prefetches the
missing historical KV rows for the shared layers in its group. Each shared
layer waits for its own prefetch before attention and loads the current
token's KV after that layer writes it.

This mode uses Model Runner V2 and supports one token per
request in a Decode batch. Set `--max-num-seqs` to the concurrency limit.
Each batch row has its own hot slots and the row is invalidated when its
request changes. For example, to capture one- and two-request Decode graphs:

```bash
--max-num-seqs 2 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}'
```

#### Precomputed TopK LIM and SDMA staging

The experimental native cache manager accepts the full layer's TopK and
manages each shared layer's hot slots. Build its standalone library with
CANN installed and its environment loaded. Set `SOC_VERSION` for the device
used to build and run the kernel. For the validated A3 configuration:

```bash
cmake -S csrc/attention/fused_lightning_indexer_manage/precomputed_topk \
    -B build/selective-lim -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910_9391
cmake --build build/selective-lim -j
export VLLM_ASCEND_SELECTIVE_LIM_LIBRARY="$PWD/build/selective-lim/lib/libselective_lim.so"
```

Export the library path in every Decode worker's environment before startup.
Selective offload requires this library; it does not fall back to tensor planning
or AIV Host transfers when the library is missing.
The current kernel supports `TopK <= hot capacity <= 2048`. It assigns each
captured request row to an independent vector-core owner. The owner retains cached
GM reads for TopK and cache state, and publishes indices, cache updates and
copy descriptors through batched DMA. Slot assignment within each
row preserves TopK order, including repeated selections. Select
`memfabric_transfer_protocol=sdma` in the P/D connector configuration on both nodes.

The complete shared-layer KV pool is allocated by MemFabric. TP0 gathers
missing rows into a small TP-shared Host buffer. Each rank registers its
mapping with `aclrtHostRegisterV2` using `ACL_HOST_REG_PINNED`, then reads
the shared buffer directly through SDMA H2D into its local device staging
buffer. A device scatter installs the rows in the layer's hot slots.
Buffer addresses, registration, and callback arguments remain alive for
graph replay. A callback queued after both K/V H2D copies acknowledges
completion. TP0 waits for all ranks' acknowledgements before overwriting
the shared buffer.

New KV stays in the device tail until a block fills. At the shared-group
boundary, TP0 gathers the completed suffix on the device, copies it to
pinned Host staging through SDMA D2H, and commits it to the MemFabric pool
from a Host callback. Decode joins the writeback stream and synchronizes
TP ranks at the end of the forward pass. Rollback restores historical KV
through the same shared gather and SDMA H2D path.

The additional fixed device staging and descriptor buffers are deducted
before automatic KV block planning. Shared Host staging scales with the
captured batch capacity and TopK. The native library is built separately
from the default extension. This path has been exercised on A3 with a
reduced-layer dummy-weight model. Full-model accuracy and performance,
A5 execution, and standard image integration still need validation.

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
