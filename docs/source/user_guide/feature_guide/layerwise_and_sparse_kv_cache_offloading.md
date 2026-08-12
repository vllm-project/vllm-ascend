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

## 1. Install Dependencies

The installation steps are grouped by hardware. Only A3 series is currently
supported.

### Prefill Build Dependencies

=== "A3 series"

    Prefill requires MemFabric Hybrid and Memcache Hybrid. Install them in this
    order.

    #### MemFabric Hybrid

    Install MemFabric Hybrid release 1.2 on every Prefill node. This release
    requires NPU driver `25.5.1` or later.

    ```bash
    pip uninstall -y memfabric_hybrid
    git clone -b release/1.2 https://gitcode.com/Ascend/memfabric_hybrid.git
    cd memfabric_hybrid
    bash script/build_and_pack_run.sh
    bash output/memfabric_hybrid-1.2.0_linux_aarch64.run
    ```

    #### Memcache Hybrid

    Install Memcache Hybrid after MemFabric Hybrid:

    ```bash
    git clone https://gitcode.com/Ascend/memcache.git
    cd memcache
    git submodule update --recursive --init
    git -c submodule.3rdparty/memfabric_hybrid.branch=release/1.2 \
        submodule update --remote --recursive 3rdparty/memfabric_hybrid
    bash script/build_and_pack_run.sh --build_mode RELEASE
    bash output/memcache_hybrid-*_linux_aarch64.run
    ```

    Configure `mmc-meta.conf`:

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

    The two files must use the same MetaService endpoint. The LocalService
    Config Store endpoint must match the MetaService Config Store endpoint.

    - Set `world_size` to the maximum supported LocalService rank count.
    - Use `device_sdma` with HCCS.
    - Set `dram.size` to at least the total KV cache size required by the target
      sequence length and concurrency divided by the number of Prefill ranks.
      Round the result up to a whole GiB.

    > **Note:** The configuration paths below assume Python 3.11.10. If you use
    > another Python version, replace the Python installation and
    > `site-packages` directories with those of the active environment. Locate
    > its `site-packages` directory with:
    >
    > `python -c "import site; print(site.getsitepackages())"`

    Start MetaService in a separate process:

    ```bash
    source /usr/local/memcache_hybrid/set_env.sh
    source /usr/local/memfabric_hybrid/set_env.sh
    export MMC_META_CONFIG_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/latest/config/mmc-meta.conf
    python -c "from memcache_hybrid import MetaService; MetaService.main()"
    ```

    Prepare every Prefill node before starting vLLM:

    ```bash
    source /usr/local/memcache_hybrid/set_env.sh
    source /usr/local/memfabric_hybrid/set_env.sh
    export MMC_LOCAL_CONFIG_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/latest/config/mmc-local.conf
    export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64
    export PYTHONHASHSEED=0
    ```

### Decode Build Dependencies

=== "A3 series"

    > **Important:** MemFabric Hybrid release 1.2 must be installed on both
    > Prefill and Decode nodes. Memcache Hybrid is required only on Prefill.

    Use the same MemFabric Hybrid build and installation commands shown above.
    Decode also requires Clang and OpenMP. Prepare every Decode node:

    ```bash
    source /usr/local/memfabric_hybrid/set_env.sh
    export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64
    clang --version
    ls "$(clang --print-resource-dir)/include/omp.h"
    ```

    If Clang or OpenMP is missing:

    ```bash
    apt-get update
    apt-get install -y clang libomp-dev
    ```

    If the image provides a specific Clang version, install the matching OpenMP
    package, for example `libomp-17-dev` for Clang 17.

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

The following log confirms that buffer reuse is enabled:

```text
Layerwise KV cache reuse merged ... descriptors into ... descriptors using ... buffer assignments.
```

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

| Parameter | Description |
| :--- | :--- |
| `topk_buffer_size` | Device hot-buffer size. It must be at least `index_topk` and divisible by `block_size`. Twice `index_topk` is a practical starting point. |
| `dram_size_per_dp_GB` | Host memory reserved per DP rank. It must hold the full KV cache. TP ranks share this pool. |
| `keep_device_kv_cache` | Debug-only option that retains the full device KV cache. Keep it `false` in production. |

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
- Layerwise buffer reuse cannot currently be combined with
  `MooncakeLayerwiseConnector` because per-buffer transfer completion gating is
  not yet implemented. Support is planned in a follow-up update.
