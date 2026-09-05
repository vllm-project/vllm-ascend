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

The installation steps are grouped by hardware. A3 and A5 series are supported.
On A5 nodes, set the MemFabric transfer protocol to `device_urma` as described
in sections 2 and 3.

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

The `SfaRemoteD2HConnector` entry accepts the following options:

| Parameter | Description |
| :--- | :--- |
| `transfer_backend` | Transfer backend. `memfabric` is the only supported value. |
| `memfabric_transfer_protocol` | MemFabric data-path protocol: `sdma` (default) and `device_rdma` for A3 series, `device_urma` for A5 series. Must be set to the same value on Prefill and Decode. Invalid values abort startup. |

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

On A5 nodes, add `"memfabric_transfer_protocol": "device_urma"` to
`kv_connector_extra_config` on both Prefill and Decode.

| Parameter | Description |
| :--- | :--- |
| `topk_buffer_size` | Device hot-buffer size. It must be at least `index_topk` and divisible by `block_size`. Twice `index_topk` is a practical starting point. |
| `dram_size_per_dp_GB` | Host memory reserved per DP rank. It must hold the full KV cache. TP ranks share this pool. |
| `keep_device_kv_cache` | Debug-only option that retains the full device KV cache. Keep it `false` in production. |

## 4. Fused-Overlap Sparse KV Cache Offload on Decode

Fused-overlap is an optimized Decode path for Sparse KV Cache Offload. It is
not a separate KV-transfer mode: configure the Prefill and Decode data path as
described in sections 2 and 3, then enable `use_fused_overlap` on the Decode
node.

In the regular sparse-offload path, Decode copies top-k misses from its full
host KV pool into the NPU hot buffer and then invokes sparse attention. With
fused-overlap enabled, the custom
`npu_fused_sparse_attention_overlap` operator performs miss handling and
sparse attention together. It reads full main KV from the Decode-owned host
pool, preserves the NPU selection buffer across Decode steps, and overlaps
selection-buffer updates for misses with attention computation. The operator
updates the selection buffer and its residency metadata in place; do not
attempt to manage those tensors from application code.

### Requirements and Compatibility

In addition to the requirements in section 3, the fused-overlap path has the
following requirements:

- Deploy disaggregated Prefill and Decode. Enable the feature only on Decode;
  Prefill must keep `sparse_kv_offload_config.enabled` disabled.
- Use Model Runner V1, an SFA/MLA sparse-attention model, and BF16 main SFA KV
  cache. The fused operator specialization in this release is validated for
  GLM-5.2 Decode.
- Build and install vLLM Ascend at this revision with custom operators enabled.
  The Decode process must register
  `npu_fused_sparse_attention_overlap`; a missing registration means that the
  custom operator package was not built or is not visible to the process.
- Use a PageAttention block size of 128. The model `index_topk` must be no
  greater than 2048. Set `topk_buffer_size` to a multiple of 128 and no less
  than `index_topk`.
- Context parallelism and pipeline parallelism are unsupported. The standard
  Sparse KV Cache Offload restrictions still apply, including the requirement
  that the full Decode KV history fit in the configured host pool.

The custom operator supports Decode graph capture. For serving workloads,
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` is the
recommended graph configuration. Keep the setting consistent across all
Decode replicas.

### Decode Configuration

Start from the Decode command in section 3. Replace its
`--additional-config` value with the following configuration, and retain the
same `SfaRemoteD2HConnector` configuration. The values below are examples;
size them for the target model, context length, and concurrency.

```bash
export VLLM_USE_V1=1

--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{
    "sparse_kv_offload_config": {
        "enabled": true,
        "use_fused_overlap": true,
        "topk_buffer_size": 4096,
        "dram_size_per_dp_GB": 180
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

`sparse_kv_offload_config` is the vLLM Ascend configuration namespace at this
revision. Put `use_fused_overlap` inside that object. Do not use an
unrecognized `kv_offload_decode_config` key: additional configuration is
validated and unknown keys are rejected. The `script/2p1d/` deployment files
were prepared from a separate development tree and contain that historical
name and the historical `SFAPDCpuOffloadConnector` connector; when reusing
them with this commit, use `sparse_kv_offload_config` and
`SfaRemoteD2HConnector` as shown above.

The Prefill command remains the Layerwise configuration from section 2. It
must use `MultiConnector` with `SfaRemoteD2HConnector` and
`AscendStoreConnector`, and it must retain `--enforce-eager`. The two
connectors jointly protect a reusable Prefill buffer until both the Memcache
save and the Decode remote read complete.

| Parameter | Description |
| :--- | :--- |
| `enabled` | Enables Sparse KV Cache Offload on the Decode node. It must be `true` before the fused path can be selected. |
| `use_fused_overlap` | Selects the fused-overlap Decode implementation. The default is `false`, which retains the regular top-k-miss copy followed by sparse attention path. |
| `topk_buffer_size` | Per-row NPU selection-buffer capacity in tokens. It must be at least `index_topk` and divisible by the KV cache block size. The fused operator in this release requires block size 128 and `index_topk <= 2048`. |
| `dram_size_per_dp_GB` | Host memory limit for one Decode DP rank's full main-KV pool. TP ranks in the same DP rank share this pool. Startup fails if the aligned planned pool exceeds this limit. |
| `keep_device_kv_cache` | Retains the full NPU main KV cache for PD-colocated debugging. Keep it unset or `false` for fused-overlap P/D deployment; it removes the capacity benefit of Decode offload. |

### Verification and Troubleshooting

After the first Decode request, verify that the Decode log contains a line
beginning with:

```text
[fused_overlap_offload][decode]
```

This line is emitted once per attention implementation and includes the query,
top-k, selection-buffer, and full-KV shapes. During initialization, the host
pool log reports `SparseKVOffloadManager starts CPU KV pool initialization`.
These messages confirm that the fused path has both its selection buffer and
full host KV source.

Common startup failures are configuration failures:

- If `topk_buffer_size` is smaller than `index_topk` or not divisible by the
  block size, increase it to a valid multiple of 128.
- If the planned CPU pool exceeds `dram_size_per_dp_GB`, increase the limit or
  reduce the configured context length, batch size, or concurrency.
- If the custom operator is not registered, rebuild and install the custom-op
  package for the active environment, then ensure the Decode process loads
  that installation.
- If a fused request has more than one top-k head or uses an unsupported cache
  layout, use the regular Decode offload path from section 3 instead.


## 5. Start the P/D Proxy

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

## 6. Limitations

- Shared-buffer Layerwise Prefill Offload requires Memcache and eager mode.
- Context parallelism has not been validated with Layerwise Prefill Offload.
- Sparse Decode Offload supports DP and TP; CP and PP are not supported.
- MemFabric is the only supported `SfaRemoteD2HConnector` transfer backend.
- The MemFabric data-path protocol is selected by launch configuration instead
  of hardware detection: use `sdma` (default) or `device_rdma` on A3 series and
  `device_urma` on A5 series, identically on Prefill and Decode.
- Layerwise buffer reuse cannot currently be combined with
  `MooncakeLayerwiseConnector` because per-buffer transfer completion gating is
  not yet implemented. Support is planned in a follow-up update.
