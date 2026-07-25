# Sparse KV Cache Offload Guide

## Overview
This guide shows how to use Sparse KV Cache Offload, a long sequence inference optimization technique.

In long sequence inference scenario, KV cache size has become one of the inference bottlenecks. Although Layerwise KV Cache Offload proposed in RFC ([#33398](https://github.com/vllm-project/vllm/issues/33398)) can save KV cache NPU memory usage during prefill, we found out that Layerwise KV Cache Offload is not suitable for decode: time consumption of loading full KV Cache is too long to be overlapped by decoding, leads to a significant increase in TPOT.

To solve the problems above, we propose sparse KV cache offload, a sparse attention based long sequence inference optimization technique. Take [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) as an example, although we still need to store full KV cache, only topk (2048) of them are needed in sparse attention. By offloading the full KV cache to host, we are able to save most of the KV Cache NPU memory usage. And since we only need to onload the needed topk KV cache in each decode step, the h2d data transmission size is small (about ~2MB for each batch), thus h2d loading time is relatively low and make it available in decode phase.

For for information about Layerwise KV Cache Offload and Sparse KV Cache Offload, please refer to our RFC ([#48203](https://github.com/vllm-project/vllm/issues/48203)).

## Supported Scenarios

- Sparse KV Cache Offload only support sparse attention based model, such as [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) and [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2).
- Currently Sparse KV Cache Offload only support PD disagregate scenario, and it can be only used in D node.
- Currently Sparse KV Cache Offload only support Tensor Parallel (TP).

Some other features that can be used together with Sparse KV Cache Offload are as follows:

|         | Eager | Graph | SpecDecode <br> (MTP) | Li C8 |
| ------- | ----- | ------| -----                 | ----- |
| **PCP** | ✅    | ✅   | ✅                   | ✅      |
| **DCP** | ✅    | ✅   | ✅                   | ✅      |

## How to use KV Cache offload

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
        \"kv_port\": ${KV_PORT},
        \"kv_rank\": ${KV_RANK},
        \"kv_connector_extra_config\": {\"use_layerwise\": true}
    }"
```
