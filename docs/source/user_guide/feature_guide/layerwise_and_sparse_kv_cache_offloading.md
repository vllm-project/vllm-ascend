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

### Qwen4Exp packed-cache reuse

Qwen4Exp on Model Runner V2 can use the GVA layerwise path with the existing
`memcache` backend configuration. Set the connector's extra configuration to
`"layerwise_num_shared_buffers": 2` and
`"layerwise_independent_layers": [0]` to keep layer 0 independent and alternate
the remaining main/state owners between two shared buffers.

For the 48-layer Qwen4Exp configuration:

| Cache owner | Device placement | Transfer policy |
| --- | --- | --- |
| QSA main KV | Shared `r0` region | Save before reuse; restore before computation |
| GDN conv and SSM | Two views of the same shared `r0` page | Restore and save both state components |
| Compressed indexer | `r5` region in the corresponding shared main/state slot | Normal group save/load, including running-step restores |
| Raw indexer | Independent per-layer `r4` ring after all main/state slots | Preserve the existing non-prefix-cacheable lifetime |
| PLE | Original private allocation | Preserve existing state pooling |

The main/state assignments are `(0,)`, `(1, 3, ..., 47)`, and
`(2, 4, ..., 46)`. Layer `i >= 3` waits for layer `i - 2` to finish saving
before its shared buffer is overwritten. Each QSA main owner and its compressed
indexer use the same slot; its raw indexer retains its original layer ordinal.

Both enabled and disabled reuse use one physical layout:

```text
shared backing: [main/state pages | compressed-indexer pages] x M
                [raw-indexer ring pages] x N
private backing: PLE / hidden states
```

For this configuration, `N = 12`. Without reuse, `M = 12`; with two shared
buffers and independent layer 0, `M = 3`. Only the main/state slot count and
owner bindings change. Raw storage remains independent for all twelve QSA
layers. A slot reserves its compressed-indexer region even when its current
owner is GDN. Page formats, block strides, shapes, and dtypes are preserved;
region offsets and descriptor layer strides follow this common layout.
With the same HBM budget, the planner can allocate more blocks. If
`num_gpu_blocks_override` fixes the block count, it instead uses less HBM.

Ignoring alignment, if `S` is one main/state page, `C` is one compressed-indexer
page, `R` is one raw-indexer page, and `P` is the private-state cost per block,
the cost changes from `12 * (S + C) + 12 * R + P` to
`3 * (S + C) + 12 * R + P`. `S` includes the space required for either K/V or
GDN conv and SSM states, including page padding. Capacity planning, admission,
cross-rank replanning, and device allocation use the same physical layout.
The worker reserves the transfer allocator's extra alignment bytes once per
actual backing allocation, then passes the remaining HBM budget to the planner
without an additional multiplier. The capacity gain depends on these sizes
and is not a fixed multiple of the previous `num_blocks`.

The shared buffers plus independent layers must fit in the original slab
count (12 for this configuration). Omitting the shared-buffer count or setting
independent layers to `"all"` retains all main/state slots in the common layout.
Excluded persistent draft owners and Model Runner V1 are not supported by this
reuse path.

Compressed indexers remain part of the pooled QSA group. Both running-step
and external-prefix loads use the normal group transfer, including main KV
and compressed indexer entries. There is no Qwen-specific entry filter or
resident-indexer exception. The raw indexer's `prefix_cacheable=False` flag
is unchanged: raw ring contents follow the existing private-state lifecycle,
not prefix-block reuse. This mode adds no raw-state snapshot protocol.
Raw rings must survive across steps to complete compression groups, so they
retain separate per-layer storage. Compressed indexers share their main KV's
reuse slot and follow the same save-before-overwrite and restore-before-use
ordering.

Before deploying this configuration, validate multi-step decode, chunked
prefill, prefix hits, speculative decoding, request preemption, and graph execution on the target NPU
with the matching vLLM main2main version. CPU storage tests cannot establish
device transfer ordering or inference correctness.

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
