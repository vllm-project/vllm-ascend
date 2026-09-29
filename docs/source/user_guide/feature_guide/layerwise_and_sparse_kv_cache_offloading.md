# Layerwise and Sparse KV Cache Offloading Guide

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

## 1. Use the Bundled Dependencies

Use a current vLLM-Ascend image with **MemFabric Hybrid 1.3.0 and Memcache
Hybrid 1.3.0**, such as `quay.nju.edu.cn/ascend/vllm-ascend:nightly-main-a3`
for A3. The image includes the offload libraries and fused Copy-SFA operators.
No additional source builds, dependency reinstalls, `set_env.sh` scripts, or
manual `MEMFABRIC_HYBRID_EXTEND_LIB_PATH`/`LD_LIBRARY_PATH` overrides are needed.

Check the packages **inside each running container**:

```bash
python -m pip show memfabric-hybrid memcache-hybrid
```

Use the same image version on Prefill and Decode. The nightly tag is mutable:
pulling it does not update an existing container. Recreate containers from the
new image and compare their image IDs when upgrading. Do not carry over
library-path overrides from an older source-installed MemFabric environment.

MemFabric is required on both roles. Memcache is used only when Prefill
layerwise offload is enabled through `AscendStoreConnector`. A regular Prefill
worker paired with sparse Decode offload does not need a Memcache service.

On 950PR&950DT Products, use the corresponding image and select `device_urma`
on both roles as described below; the A3 examples use `sdma`.

### Memcache Configuration for Layerwise Prefill Offload

Skip this subsection when Prefill keeps its regular device KV cache.
For layerwise Prefill offload, create `mmc-meta.conf`:

```ini
ock.mmc.meta_service_url = tcp://<META_HOST>:5000
ock.mmc.meta_service.config_store_url = tcp://<CONFIG_STORE_HOST>:6000
ock.mmc.meta.lease_ttl_ms = 30000
ock.mmc.log_level = error
```

Create `mmc-local.conf` on every Prefill node:

```ini
ock.mmc.meta_service_url = tcp://<META_HOST>:5000
ock.mmc.local_service.config_store_url = tcp://<CONFIG_STORE_HOST>:6000
ock.mmc.log_level = error
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 10GB
```

The two files must use the same MetaService endpoint. The LocalService Config
Store endpoint must match the MetaService Config Store endpoint.

- Set `world_size` to the maximum supported LocalService rank count.
- Use `device_sdma` with HCCS on A3.
- Size `dram.size` for the required KV cache per Prefill rank, rounded up to GiB.

Start MetaService in a separate process, using your configuration file:

```bash
MMC_META_CONFIG_PATH="$PWD/mmc-meta.conf" \
    python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

Before starting each layerwise Prefill worker, select its configuration:

```bash
export MMC_LOCAL_CONFIG_PATH="$PWD/mmc-local.conf"
export PYTHONHASHSEED=0
```

These variables configure the service; they do not replace the image's bundled
libraries.

### Fused Copy-SFA Operators

The image's `_C_ascend` extension provides `npu_fused_lightning_indexer_manage`
and `npu_fused_scatter_copy_sparse_flash_attention`. No separate operator build
is needed. Enable them with `fused_op_type="fused_copy_sfa"` as described in
[Enable Fused LIM and Copy-SFA](#enable-fused-lim-and-copy-sfa).

Copy-SFA consumes CPU views backed by registered MemFabric memory. Ordinary
CPU allocations are not substitutes for this offload pool; the Python
integration does not copy the full host cache back to NPU.

## 2. Layerwise KV Cache Offload on Prefill

Use this mode on a dedicated Prefill node with:

- `kv_role: "kv_producer"`;
- the Memcache backend;
- an MLA, SFA, or DSA attention backend; and
- eager execution.

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
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "transfer_backend": "memfabric"
                }
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
| `layerwise_num_shared_buffers` | Number of reusable NPU buffers. Start with two to four and tune for memory and transfer bandwidth. |
| `layerwise_independent_layers` | Layers that keep dedicated buffers. The default is `[0]`; `"all"` disables cross-layer reuse. |

The `SfaRemoteD2HConnector` entry accepts the following options:

| Parameter | Description |
| :--- | :--- |
| `transfer_backend` | Transfer backend. `memfabric` is the only supported value. |
| `memfabric_transfer_protocol` | MemFabric data-path protocol: `sdma` (default) and `device_rdma` for A3 series, `device_urma` for 950PR&950DT Products. Must be set to the same value on Prefill and Decode. Invalid values abort startup. |

The following log confirms that buffer reuse is enabled:

```text
Layerwise KV cache reuse merged ... descriptors into ... descriptors using ... buffer assignments.
```

### Regular Prefill with Decode-Only Offload

To keep Prefill KV entirely on device, omit `AscendStoreConnector` and its
shared-buffer settings. Use only:

```bash
--enforce-eager \
--kv-transfer-config '{
    "kv_connector": "SfaRemoteD2HConnector",
    "kv_role": "kv_producer",
    "kv_port": 20000,
    "kv_connector_extra_config": {
        "transfer_backend": "memfabric"
    }
}'
```

Do not enable `sparse_kv_offload_config` on Prefill. Pair this producer with the
Decode configuration below and the layerwise proxy in section 4. No Memcache
MetaService or LocalService configuration is required for this mode.

## 3. Sparse KV Cache Offload on Decode

Requirements:

- use disaggregated Prefill/Decode deployment;
- enable the feature only on Decode; and
- use Model Runner V1.

Add the following options to the Decode launch command:

```bash
--additional-config '{
    "sparse_kv_offload_config": {
        "enabled": true,
        "topk_buffer_size": 4096,
        "dram_size_per_dp_GB": 128
    }
}' \
--kv-transfer-config '{
    "kv_connector": "SfaRemoteD2HConnector",
    "kv_role": "kv_consumer",
    "kv_port": 20050,
    "kv_connector_extra_config": {
        "transfer_backend": "memfabric",
        "use_layerwise": true
    }
}'
```

On Decode, reserve
`decode_data_parallel_size * decode_tensor_parallel_size` consecutive ports
starting from `kv_port`.

On 950PR&950DT Products nodes, add `"memfabric_transfer_protocol": "device_urma"` to
`kv_connector_extra_config` on both Prefill and Decode.

| Parameter | Description |
| :--- | :--- |
| `fused_op_type` | Set to `"fused_copy_sfa"` to enable fused LIM and Copy-SFA together. The default, `"none"`, uses the existing sparse offload path. |
| `topk_buffer_size` | Device hot-buffer size. It must be at least `index_topk` and divisible by `block_size`. Twice `index_topk` is a practical starting point. |
| `dram_size_per_dp_GB` | Host memory reserved per DP rank. It must hold the full KV cache. TP ranks share this pool. |
| `keep_device_kv_cache` | Debug-only option that retains the full device KV cache. Keep it `false` in production. |

### Enable Fused LIM and Copy-SFA

With the [bundled native operators](#fused-copy-sfa-operators), set the following fields in the Decode node's `--additional-config`. This
example supports MTP3:

```json
{
    "sparse_kv_offload_config": {
        "enabled": true,
        "fused_op_type": "fused_copy_sfa",
        "topk_buffer_size": 8192,
        "dram_size_per_dp_GB": 128,
        "keep_device_kv_cache": false,
        "use_fused_overlap": false
    }
}
```

Merge this object with any existing additional settings and pass
`--additional-config` once. Choose either Prefill configuration from section 2;
enable these fused operators only on Decode in a PD deployment.

The fused path has these additional requirements:

- The model must use `index_topk=2048` and a cache block size of `128`.
- Let `Q_max = 1 + num_speculative_tokens`, or `1` without speculative decoding.
  `Q_max` must be between `1` and `7`.
- `topk_buffer_size` must be a multiple of `256`, at least `Q_max * 2048`,
  and at most `16128`. The runtime allocates two additional tail blocks;
  do not add them to this setting.
- Keep `use_fused_overlap=false`; it cannot be combined with `fused_copy_sfa`.
- Use BF16 KV and indexer caches. Sparse SFA C8 and sparse LI C8 serving are
  not supported by this integration.

| Draft tokens | `Q_max` | Minimum `topk_buffer_size` |
| :--- | :--- | :--- |
| No speculative decoding | 1 | 2048 |
| MTP1 | 2 | 4096 |
| MTP3 | 4 | 8192 |
| MTP5 | 6 | 12288 |

For example, the following A3 Decode command uses GLM-5.2 W4A8 with DP2 TP8,
MTP3, and `FULL_DECODE_ONLY` target graphs. Replace the model path and size
the model length, concurrency and host-cache budget for your deployment.
The command assumes the image and package versions from section 1.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve /path/to/GLM-5.2-w4a8 \
    --host 0.0.0.0 \
    --port 8200 \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --quantization ascend \
    --block-size 128 \
    --max-model-len 131072 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16]}' \
    --additional-config '{
        "sparse_kv_offload_config": {
            "enabled": true,
            "fused_op_type": "fused_copy_sfa",
            "topk_buffer_size": 8192,
            "dram_size_per_dp_GB": 128,
            "keep_device_kv_cache": false,
            "use_fused_overlap": false
        }
    }' \
    --kv-transfer-config '{
        "kv_connector": "SfaRemoteD2HConnector",
        "kv_role": "kv_consumer",
        "kv_port": 20050,
        "kv_connector_extra_config": {
            "transfer_backend": "memfabric",
            "use_layerwise": true
        }
    }'
```

Here, `enforce_eager` applies only to the MTP draft model. The target uses the
graph mode configured by `--compilation-config`; do not add a top-level
`--enforce-eager` when using this graph example. Decode prefix caching is
disabled, and `keep_device_kv_cache=false` keeps the full main KV in the host
pool. With DP2, the example reserves `2 * 128 = 256` GiB of host KV memory.

## 4. Start the P/D Proxy

Start Prefill and Decode with the configurations above. After both nodes are
ready, start the proxy:

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

## 5. Limitations

- Shared-buffer Layerwise Prefill Offload requires Memcache and eager mode.
- Context parallelism has not been validated with Layerwise Prefill Offload.
- Sparse Decode Offload supports DP and TP; CP and PP are not supported.
- MemFabric is the only supported `SfaRemoteD2HConnector` transfer backend.
- The MemFabric data-path protocol is selected by launch configuration instead
  of hardware detection: use `sdma` (default) or `device_rdma` on A3 series and
  `device_urma` on 950PR&950DT Products, identically on Prefill and Decode.
- Layerwise buffer reuse cannot currently be combined with
  `MooncakeLayerwiseConnector` because per-buffer transfer completion gating is
  not yet implemented. Support is planned in a follow-up update.
