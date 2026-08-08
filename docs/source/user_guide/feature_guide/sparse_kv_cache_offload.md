# Sparse KV Cache Offload Guide

## Overview

This guide shows how to use Sparse KV Cache Offload, a long sequence inference optimization technique.

In long sequence inference scenario, KV cache size has become one of the inference bottlenecks. Although Layerwise KV Cache Offload proposed in RFC ([#33398](https://github.com/vllm-project/vllm/issues/33398)) can save KV cache NPU memory usage during prefill, we found out that Layerwise KV Cache Offload is not suitable for decode: time consumption of loading full KV Cache is too long to be overlapped by decoding, leads to a significant increase in TPOT.

To solve the problems above, we propose sparse KV cache offload, a sparse attention based long sequence inference optimization technique. Take [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) as an example, although we still need to store full KV cache, only topk (2048) of them are needed in sparse attention. By offloading the full KV cache to host, we are able to save most of the KV Cache NPU memory usage. And since we only need to onload the needed topk KV cache in each decode step, the h2d data transmission size is small (about ~2MB for each batch), thus h2d loading time is relatively low and make it available in decode phase.

For more information about Layerwise KV Cache Offload and Sparse KV Cache Offload, please refer to our RFC ([#48203](https://github.com/vllm-project/vllm/issues/48203)).

## Supported Scenarios

- Sparse KV Cache Offload only support sparse attention based model, such as [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) and [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2).
- Currently Sparse KV Cache Offload only support PD disagregate scenario, and it can be only used in D node.
- Currently Sparse KV Cache Offload only support Tensor Parallel (TP). Context Parallel (CP) and Pipeline Parallel (PP) are not supported now.
- Currently Sparse KV Cache Offload only support model_runner_v1, we will support model_runner_v2 in the future.

Some other features that can be used together with Sparse KV Cache Offload are as follows:

| Eager | Graph | SpecDecode <br> (MTP) | Li C8 |
| ----- | ------| -----                 | ----- |
| ✅    | ✅   | ✅                   | ✅      |

## How to use Sparse KV Cache offload

### Environment setup

Since some third party dependencies needed by sparse KV Cache offload are not included in published images yet, you may need to install them manually before using sparse KV Cache offload:

- Update memfabric-hybrid

    ```bash
    # uninstall old memfabric_hybrid in your image
    pip uninstall -y memfabric_hybrid
    # clone and build from src
    git clone https://gitcode.com/Ascend/memfabric_hybrid.git -b release/1.2
    cd memfabric_hybrid
    bash script/build_and_pack_run.sh
    # install new version memfabric_hybrid
    bash output/memfabric_hybrid-1.2.0_linux_aarch64.run
    # set library path env
    export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64
    ```

    You can refer to the [memfabric installation doc](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/doc/installation.md) for more information if you encounter any problems.

    This dependency will be removed after sparse KV cache offload related code is merged to memfabric master branch.

    **NOTE:** Current memfabric_hybrid version requires NPU driver version >= `25.5.1`. Use `npu-smi info` to check your driver version, you may need to update it if it's version is too low.

- Install CPP compilation dependencies (optional)

    If your image doesn't have clang installed, install clang and openmp library.

    ```bash
    # check whether clang is installed
    clang --version
    # install clang and openmp
    apt install clang
    apt-get install libomp-dev
    ```

    If your image has already installed clang, check whether openmp library is already installed too. If not, install the specific version according to your clang version.
    ```bash
    # check whether openmp is already installed by finding omp.h
    ls $(clang --print-resource-dir)/include/omp.h
    # install the specific version openmp,
    # for example your image already have clang17, then you need to install omp-17
    apt-get install libomp-17-dev
    ```

    If your image already have clang and openmp library installed, you can skip this step.

    This dependency will be removed after we use custom NPU operator to replace current `lru_resident_compact` and `compute_lru_resident_addrs` written by cpp.

### Configuration

You can enable Sparse KV Cache Offload by setting `sparse_kv_offload_config` in `additional-config`. You also need to specify `SFAPDCpuOffloadConnector` in `kv-transfer-config` for PD KV transfer. Refer to the following example:

```bash
vllm serve zai-org/GLM-5.2 \
    --host 0.0.0.0 \
    --port 8005 \
    --served-model-name model \
    --tensor-parallel-size 16 \
    --max-num-seqs 4 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --enforce-eager \
    --gpu-memory-utilization 0.7 \
    --quantization ascend \
    --no-enable-prefix-caching \
    --additional-config '{"sparse_kv_offload_config": {"enabled": true, "topk_buffer_size": 4096, "dram_size_per_dp_GB": 128}}' \
    --kv-transfer-config "{
        \"kv_connector\": \"SFAPDCpuOffloadConnector\",
        \"kv_buffer_device\": \"npu\",
        \"kv_role\": \"kv_consumer\",
        \"kv_parallel_size\": 1,
        \"kv_port\": 20050,
        \"kv_rank\": 1,
        \"kv_connector_extra_config\": {\"use_layerwise\": true}
    }"
```

`sparse_kv_offload_config` parameters meaning and usage:

- `enabled` (bool, default: `False`): Whether to enable sparse KV Cache offload.
- `topk_buffer_size` (int, default: `4096`): Size of the device hot KV buffer, should be greater than the model's `index_topk`. Increase this size will increase topk KV hit rate thus reduce tpot, but will bring higher device memory usage. Normally we set it to 2 * `index_topk` to achieve a balance between topk KV hit rate and device memory usage.
- `dram_size_per_dp_GB` (bool, default: `128`): Reserved host memory size in Gigabyte per DP group. An error will be raised if this reserved size is not enough to contain the whole host KV cache buffer. All ranks in one TP group will share one host KV cache buffer, so the total host memory usage will be `dram_size_per_dp_GB` * `data_parallel_size`.
- `keep_device_kv_cache` (bool, optional, default: `False`): Whether to allocate the device KV Cache buffer. If enabled, we will still allocate device KV Cache, thus we can't save KV cache device memory usage, and can not improve sequence length or batch_size. Only reserved for PD-colocate debugging, you **SHOULD NOT** enable it in actual production environment.
