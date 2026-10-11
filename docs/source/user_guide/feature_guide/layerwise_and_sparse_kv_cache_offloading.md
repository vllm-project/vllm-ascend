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

The installation steps are grouped by hardware. A3 and 950PR&950DT Products are supported.
On 950PR&950DT Products nodes, set the MemFabric transfer protocol to `device_urma` as described
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

### Optional Fused Copy-SFA Operators

With `sparse_kv_offload_config.fused_op_type="fused_copy_sfa"`, the Python integration requires
these operators in the installed `_C_ascend` extension:

- `npu_fused_lightning_indexer_manage`
- `npu_fused_quant_lightning_indexer_manage` (LIM C8)
- `npu_fused_scatter_copy_sparse_flash_attention`

The expected interfaces are from
[vLLM-Ascend PR #16640](https://github.com/vllm-project/vllm-ascend/pull/16640),
revision `a9823977149172f1604d9f2a1937224d0b11646e`. This branch carries the
Python integration; it does not bundle these native kernels or their bindings.
An operator-enabled native build is required to run the fused_copy_sfa path.

LIM C8 uses `npu_fused_quant_lightning_indexer_manage`, which is included on
this branch. Enable it with `--attention_config.indexer_kv_dtype int8`. The
indexer then Hadamard-rotates and int8-quantizes query and key, and
`_lim_select` forwards that int8 query and the fp16 dequant scales to the C8
LIM. The 19-parameter management ABI matches BF16 LIM, so Copy-SFA and miss H2D
stay unchanged. The C8 kernel is int8-only; `indexer_kv_dtype fp8` is not a
LIM C8 serving path.

Copy-SFA receives `dram_k_rope` and `dram_kv_cache` as CPU tensor views backed
by registered MemFabric memory. Its native adapter must permit those CPU views;
the referenced PR revision needs this device-check adjustment. The remaining
inputs stay on NPU. Ordinary CPU allocations are not valid replacements for
registered DRAM, and the Python integration does not move the host cache to NPU.

For launch settings, see [Enable Fused LIM and Copy-SFA](#enable-fused-lim-and-copy-sfa).

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

## 3. Sparse KV Cache Offload on Decode

Requirements:

- use disaggregated Prefill/Decode deployment;
- enable the feature only on Decode; and
- use Model Runner V1 or V2 on Decode. Keep Decode at pipeline parallel size 1;
  Prefill can use pipeline parallelism with the producer connector.

MemFabric configuration-store ownership stays on Prefill, using the official
transfer-engine initialization. No additional store-role or preconnection
configuration is needed. Deploy the same connector protocol version on both
sides, and validate actual remote KV reads rather than relying on `/health`.

For the validated GLM-5.2 Prefill PP2 deployment on 16 NPUs, use DP1 × PP2 ×
TP8, not DP2 × PP2 × TP8. Set `VLLM_PP_LAYER_PARTITION=38,40`: each stage
must contain the full Indexer owner for its shared Indexer layers. A balanced
39/39 partition is rejected by the existing stage-boundary guard; cross-stage
TopK sharing is not supported. Decode remains DP2 × TP8 with PP size 1.

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

After installing the [optional native operators](#optional-fused-copy-sfa-operators),
set the following fields in the Decode node's `--additional-config`. This
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
`--additional-config` once. Keep the Prefill configuration from section 2;
enable these fused operators only on Decode in a PD deployment.

The fused path has these additional requirements:

- The model must use `index_topk=2048` and a cache block size of `128`.
- Let `Q_max = 1 + num_speculative_tokens`, or `1` without speculative decoding.
  `Q_max` must be between `1` and `14`.
- `topk_buffer_size` must be a multiple of `256`, at least `Q_max * 2048`,
  and at most `32512` (the largest aligned value within LIM's `32640` limit).
  The runtime allocates two additional tail blocks;
  do not add them to this setting.
- Keep `use_fused_overlap=false`; it cannot be combined with `fused_copy_sfa`.
- Use BF16 for the main KV cache. Sparse SFA C8 is not supported.
  Sparse LI C8 serving is supported for the device-resident indexer:
  add `--attention_config.indexer_kv_dtype int8` so fused Copy-SFA
  selects `npu_fused_quant_lightning_indexer_manage`.

| Draft tokens | `Q_max` | Minimum `topk_buffer_size` |
| :--- | :--- | :--- |
| No speculative decoding | 1 | 2048 |
| MTP1 | 2 | 4096 |
| MTP2 | 3 | 6144 |
| MTP3 | 4 | 8192 |
| MTP5 | 6 | 12288 |
| DSpark8 | 9 | 18432 |

For example, the following A3 Decode command uses GLM-5.2 W4A8 with DP2 TP8,
MTP3, and `FULL_DECODE_ONLY` target graphs. Replace the model path and size
the model length, concurrency and host-cache budget for your deployment.
The command assumes the dependency environment from section 1 is already set.

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

Model Runner V2 uses the same sparse-offload configuration, including MTP and
`fused_copy_sfa`: set `VLLM_USE_V2_MODEL_RUNNER=1`. For an eager V2 launch,
replace the graph compilation option with a top-level `--enforce-eager`.

### DSpark with Sparse Decode Offload

GLM MLA DSpark requires Model Runner V2 on both Prefill and Decode. Configure
the same draft checkpoint and speculative-token count on both nodes, for example:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /path/to/target \
    --speculative-config '{"method":"dspark","model":"/path/to/GLM-MLA-draft","num_speculative_tokens":8}' \
    ...
```

Keep Prefill eager and disable local vLLM prefix caching on both nodes. Auxiliary capture
layers are resolved from the draft checkpoint; no separate layer-ID setting is
needed. Prefill projects the target's prompt features locally and writes its
own draft KV. The SFA producer connector transfers those pages alongside target
KV, and Decode waits for both transfers before decoding. Hidden features are
not sent to Decode for prompt-KV reconstruction.

The Prefill Memcache layerwise store can reuse external prefixes independently
of the local prefix-caching switch. It saves draft KV under checkpoint-specific
companion keys, without placing the live draft pages in target scratch buffers.
A prefix is usable only when every target stage and every final-stage draft TP
rank has saved it. Prefill restores draft pages into the new request's NPU block
table, projects only the remaining prompt features, then transfers the completed
target and draft KV to Decode. The final Eagle recomputation block is retained
even on a complete store hit. Missing draft companions make the prefix a miss;
load failures after lookup fail closed rather than using uninitialized draft KV.

Enable sparse offload only on Decode, with the consumer connector and
`fused_copy_sfa` configuration above. For draft8, set `topk_buffer_size` to at
least `18432`; `20480` is an example with extra hot-cache capacity. These are
kernel/layout bounds, not a draft-checkpoint whitelist or a guarantee that every
width has been validated end to end. Both nodes must use the same cache block
size. Remote DSpark draft-KV transfer does not support PCP or DCP.

Only the target's main KV is offloaded to the host. The loaded draft supplies
its cache-layer ownership, and its full context KV stays in device memory.
Layerwise Prefill Offload excludes draft pages from its shared scratch pool.
The paired prefix-store path also keeps draft save/load separate from target
layer hooks, including when target buffers are not reused. Other store paths
retain their original registration and addressing.
Account for this context-dependent HBM cost when sizing long-context serving;
target offload alone does not establish 1M-context DSpark support.

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
- Sparse Decode Offload supports DP and TP on Decode; CP and Decode-side PP
  are not supported. Prefill-side PP is supported with the producer connector.
- MemFabric is the only supported `SfaRemoteD2HConnector` transfer backend.
- The MemFabric data-path protocol is selected by launch configuration instead
  of hardware detection: use `sdma` (default) or `device_rdma` on A3 series and
  `device_urma` on 950PR&950DT Products, identically on Prefill and Decode.
- Layerwise buffer reuse cannot currently be combined with
  `MooncakeLayerwiseConnector` because per-buffer transfer completion gating is
  not yet implemented. Support is planned in a follow-up update.
