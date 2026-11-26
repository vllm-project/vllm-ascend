# Qwen3.8-27B

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

Qwen3.8-27B is the 27-billion-parameter dense member of the Qwen3.8 family, the most capable generation in the Qwen open-model family to date. Built on the architectural foundation of Qwen3.5, it shares the same hybrid-attention backbone as the 2.4T MoE flagship: of its 64 layers, only 16 run full (gated) attention (`full_attention_interval: 4`) while the other 48 run linear attention (Gated DeltaNet) with a constant recurrent state. It ships with a built-in MTP (Multi-Token Prediction) draft head and a native 262,144-token context window.

Qwen3.8-27B delivers substantial gains over Qwen3.5/Qwen3.6 across coding, professional work, research, and long-horizon agentic tasks. It features stronger autonomous planning, more reliable end-to-end task completion, and broader downstream compatibility with popular harnesses and development tools.

This document focuses on text serving on Ascend NPUs. It describes the main validation steps for the model, including supported features, prerequisites, installation, online service deployment, Prefill-Decode (PD) disaggregation, functional verification, accuracy and performance evaluation, performance tuning, and FAQs.

This document is validated and written based on **vLLM-Ascend 0.23.0**. The current model (Qwen3.8-27B) is first supported in this version.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_features.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

## 3 Prerequisites

### 3.1 Model Weight

The following model weights are available:

|  Weight Version                              | Hardware Requirements | Download Links |
|----------------------------------------------|-----------------------|----------------|
| `Qwen3.8-27B` (BF16 version)                 | 1 950DT Products (96GB × 8) node or 1 950PR Products (128GB × 8) node or 1 Atlas 800I A3 (64GB × 16) node <br>or 1 Atlas 800I A2 (64GB × 8) node | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B) |
| `Qwen3.8-27B-w8a8` (Quantized version)       | 1 Atlas 800I A3 (64GB × 16) node or 1 Atlas 800I A2 (64GB × 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8) |
| `Qwen3.8-27B-w8a8-mxfp8` (Quantized version) | 1 950DT Products (96GB × 8) node or 1 950PR Products (128GB × 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8-mxfp8) |
| `Qwen3.8-27B-w8a8c8-mxfp8` (Quantized version) | 1 950DT Products (96GB × 8) node or 1 950PR Products (128GB × 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8c8-mxfp8) |
| `Qwen3.8-27B-w8a8-310p` (Quantized version)  | 1 Atlas 300I Duo  | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8-310p) |

- `Qwen3.8-27B-w8a8c8-mxfp8` (W8A8 MXFP8 quantization with C8 MXFP8 KV cache): serve it with `--kv-cache-dtype mxfp8`; recommended for long-sequence scenarios.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

> **Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

## 4 Installation

### 4.1 Docker Image Installation

{% include "getting_started/installation/ascend_docker_runtime.inc.md" %}

Select an image based on your machine type and start the docker container on your node. Refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image).

=== "950PR&950DT Products"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:qwen3.8-a5
    export NAME=vllm-ascend

    docker run --rm \
        -e ASCEND_VISIBLE_DEVICES=0-7 \
        --name $NAME \
        --net=host \
        --shm-size=1g \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
        -v /etc/hixlep/:/etc/hixlep/ \
        -v /root/.cache:/root/.cache \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
        -v /usr/lib64:/usr/lib64 \
        -it $IMAGE bash
    ```

=== "Atlas 800I A3"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:qwen3.8-a3
    export NAME=vllm-ascend

    docker run --rm \
        -e ASCEND_VISIBLE_DEVICES=0-15 \
        --name $NAME \
        --net=host \
        --shm-size=1g \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

=== "Atlas 800I A2"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:qwen3.8-a2
    export NAME=vllm-ascend

    docker run --rm \
        -e ASCEND_VISIBLE_DEVICES=0-7 \
        --name $NAME \
        --shm-size=1g \
        --net=host \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

=== "Atlas 300I Duo"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:v0.23.0-310p
    export NAME=vllm-ascend

    docker run --rm \
        -e ASCEND_VISIBLE_DEVICES=0-7 \
        --name $NAME \
        --shm-size=1g \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

After entering the container, verify that vLLM and vLLM-Ascend can be imported:

```shell
python -c "import vllm, vllm_ascend; print('vllm and vllm_ascend are ready')"
```

Expected output:

```output
vllm and vllm_ascend are ready
```

### 4.2 Source Code Installation

You can also build and install `vllm-ascend` from source. Refer to [set up using Python](../../getting_started/installation.md#installation-existing-cann-install).

If you want to deploy a multi-node service, install the same version of vLLM and vLLM-Ascend on each node.

!!! note

    On Atlas 300I Duo, you may need to uninstall `triton-ascend` and `triton` to avoid dependency conflicts:

    ```bash
    pip uninstall -y triton-ascend triton
    ```

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and medium-scale inference scenarios. On Atlas 300I Duo, at least 2 devices are required.

Before starting the service:

- Replace the model path, parallel sizes and service port with values from the target environment.

=== "950DT Products"

    The following example is for 950DT Products.

    ```bash
    #!/bin/sh
    # Load model from ModelScope to speed up download
    export MODEL_PATH=Eco-Tech/Qwen3.8-27B-w8a8c8-mxfp8
    export VLLM_USE_MODELSCOPE=True
    export HCCL_BUFFSIZE=512
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    sysctl -w vm.swappiness=0 || true
    sysctl -w kernel.numa_balancing=0 || true
    sysctl -w kernel.sched_migration_cost_ns=50000 || true  # no-op on EEVDF (Linux 6.6+), which removed this parameter
    # Size of the shared buffer (in MB) used by HCCL for NPU-to-NPU collective communication
    # To reduce memory fragmentation and avoid out of memory

    # Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.8-27B-w8a8c8-mxfp8) or a local directory path
    # Ensure the model path matches the directory recorded during download

    vllm serve $MODEL_PATH \
        --host 0.0.0.0 \
        --port 8000 \
        --data-parallel-size 1 \
        --tensor-parallel-size 1 \
        --quantization ascend \
        --kv-cache-dtype mxfp8 \
        --served-model-name qwen3.8 \
        --max-num-seqs 32 \
        --max-model-len 133120 \
        --max-num-batched-tokens 16384 \
        --trust-remote-code \
        --allowed-local-media-path / \
        --mm-processor-cache-gb 0 \
        --mm-encoder-tp-mode data \
        --mm-processor-cache-type shm \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.85 \
        --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
        --compilation-config '{"cudagraph_capture_sizes": [1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128],"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding": true, "enable_flashcomm1": true}'
    ```

    Key Parameter Descriptions:

    - `--data-parallel-size 1` and `--tensor-parallel-size 1` are common settings for data parallelism (DP) and tensor parallelism (TP) sizes.
    - `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
    - `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
    - `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
        - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`.
        - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
        - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on HBM memory (activation value usage) will be greater.
    - `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak HBM memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak HBM memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the HBM memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
    - `--quantization ascend` indicates that quantization is used. To disable quantization, remove this option.
    - `--kv-cache-dtype mxfp8` enables the C8 (MXFP8 KV cache) and is required by the `Qwen3.8-27B-w8a8c8-mxfp8` weights. It stores the KV cache of the full-attention layers as FP8 (E4M3) with E8M0 scales, and runs the full-attention layers on the dedicated QFA (quant flash attention) kernel. The W8A8C8 weights bring performance gains in long-sequence scenarios. With this option, `--block-size` must be a multiple of `512` (it defaults to `512` if unset).
    - For short-sequence scenarios (e.g., 16k input length), the W8A8 weights (e.g., `Qwen3.8-27B-w8a8-mxfp8`) are recommended instead. When serving a W8A8 weight, remove `--kv-cache-dtype mxfp8` from the startup command. Note that C8 quantization is only supported on 950PR&950DT Products.
    - `--enable-prefix-caching` enables prefix caching, which is the default behavior. To disable it, use `--no-enable-prefix-caching`. For workloads with long shared prefixes, finer-grained hybrid prefix matching can be enabled with `--enable-prefix-caching --prefix-match-unit 16`. The match unit must divide every KV cache group's block size. A smaller unit can reuse more of a partial physical page, at the cost of additional hash metadata, copy-on-write operations, and an extra prefill split at the final matching boundary.
    - `--speculative-config` uses `qwen3_5_mtp` for `Qwen3.8-27B` because it shares the same MTP head design as `Qwen3.5-27B`.
    - `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are `"cudagraph_mode"` and `"cudagraph_capture_sizes"`, which have the following meanings:
        - `"cudagraph_mode"`: represents the specific graph mode. Currently, `"PIECEWISE"` and `"FULL_DECODE_ONLY"` are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, `"FULL_DECODE_ONLY"` is recommended.
        - `"cudagraph_capture_sizes"`: specifies the batch size tiers at which graphs are captured. The default value is `[1, 2, 4, 8, 16, 24, 32, 40,..., --max-num-seqs]`. In the graph mode, the batch size of each captured graph is fixed, and an actual batch size between two adjacent capture sizes is automatically padded up to the next capture size. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.
    - `--allowed-local-media-path` allows the API server to read local images or videos from the specified directory when processing multimodal inputs. In this example, `/` whitelists the entire file system. This is security-sensitive and should only be enabled in trusted environments.
    - `--mm-processor-cache-gb` sets the size (in GiB) of the multimodal processor cache, which is used to avoid re-processing past multimodal inputs. The cache is duplicated in each API server process and engine core process, so the total host memory usage is the cache size multiplied by the number of these processes. A value of `0` disables this cache completely.
    - `--mm-encoder-tp-mode` indicates how to optimize multimodal encoder inference using tensor parallelism (TP). `data` splits the batched encoder input data across TP ranks while each rank hosts the full encoder weights, and is recommended when multimodal inputs are used. The default mode `weights` splits the encoder weights across TP ranks instead.
    - `--mm-processor-cache-type` sets the cache type used by the multimodal preprocessor and mapper. `shm` uses a shared-memory FIFO cache, while the default `lru` uses a mirrored LRU cache.
    - `--additional-config` contains Ascend-specific optimization options:
        - `"enable_cpu_binding": true` enables Ascend-native CPU binding, which pins the worker process, key runtime threads, memory pages, and NPU IRQs to CPU ranges close to the driven NPU, reducing cross-NUMA traffic and thread preemption and improving latency stability. It is enabled by default starting from vllm-ascend v0.18.0rc1.
        - `"enable_flashcomm1": true` enables the FlashComm communication optimization, which splits the tensor-parallel AllReduce into Reduce-Scatter and All-Gather to reduce communication overhead and improve prefill throughput.

=== "950PR Products"

    The following example is for 950PR Products. Quantized versions need `--quantization ascend`. On 950PR Products, `Qwen3.8-27B-w8a8` is the current recommended weight; switch to `Qwen3.8-27B-w8a8-mxfp8` later after further optimization.

    ```bash
    #!/bin/sh
    # Load model from ModelScope to speed up download
    export VLLM_USE_MODELSCOPE=True
    # Size of the shared buffer (in MB) used by HCCL for NPU-to-NPU collective communication
    export HCCL_BUFFSIZE=512
    # To reduce memory fragmentation and avoid out of memory
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    sysctl -w vm.swappiness=0 || true
    sysctl -w kernel.numa_balancing=0 || true
    sysctl -w kernel.sched_migration_cost_ns=50000 || true  # no-op on EEVDF (Linux 6.6+), which removed this parameter

    # Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.8-27B-w8a8) or a local directory path
    # Ensure the model path matches the directory recorded during download
    export MODEL_PATH=Eco-Tech/Qwen3.8-27B-w8a8

    vllm serve $MODEL_PATH \
        --host 0.0.0.0 \
        --port 8000 \
        --data-parallel-size 1 \
        --tensor-parallel-size 1 \
        --quantization ascend \
        --served-model-name qwen3.8 \
        --max-num-seqs 32 \
        --max-model-len 256000 \
        --max-num-batched-tokens 16384 \
        --trust-remote-code \
        --allowed-local-media-path / \
        --mm-processor-cache-gb 0 \
        --mm-encoder-tp-mode data \
        --mm-processor-cache-type shm \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.9 \
        --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
        --compilation-config '{"cudagraph_capture_sizes": [1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128],"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding": true, "enable_flashcomm1": true}'
    ```

    Key Parameter Descriptions:

    - `--data-parallel-size 1` and `--tensor-parallel-size 1` are common settings for data parallelism (DP) and tensor parallelism (TP) sizes.
    - `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
    - `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
    - `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
        - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`.
        - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
        - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on HBM memory (activation value usage) will be greater.
    - `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak HBM memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak HBM memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the HBM memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
    - `--quantization ascend` indicates that quantization is used. To disable quantization, remove this option.
    - `--enable-prefix-caching` enables prefix caching, which is the default behavior. To disable it, use `--no-enable-prefix-caching`. For workloads with long shared prefixes, finer-grained hybrid prefix matching can be enabled with `--enable-prefix-caching --prefix-match-unit 16`. The match unit must divide every KV cache group's block size. A smaller unit can reuse more of a partial physical page, at the cost of additional hash metadata, copy-on-write operations, and an extra prefill split at the final matching boundary.
    - `--speculative-config` uses `qwen3_5_mtp` for `Qwen3.8-27B` because it shares the same MTP head design as `Qwen3.5-27B`.
    - `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are `"cudagraph_mode"` and `"cudagraph_capture_sizes"`, which have the following meanings:
        - `"cudagraph_mode"`: represents the specific graph mode. Currently, `"PIECEWISE"` and `"FULL_DECODE_ONLY"` are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, `"FULL_DECODE_ONLY"` is recommended.
        - `"cudagraph_capture_sizes"`: specifies the batch size tiers at which graphs are captured. The default value is `[1, 2, 4, 8, 16, 24, 32, 40,..., --max-num-seqs]`. In the graph mode, the batch size of each captured graph is fixed, and an actual batch size between two adjacent capture sizes is automatically padded up to the next capture size. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.
    - `--allowed-local-media-path` allows the API server to read local images or videos from the specified directory when processing multimodal inputs. In this example, `/` whitelists the entire file system. This is security-sensitive and should only be enabled in trusted environments.
    - `--mm-processor-cache-gb` sets the size (in GiB) of the multimodal processor cache, which is used to avoid re-processing past multimodal inputs. The cache is duplicated in each API server process and engine core process, so the total host memory usage is the cache size multiplied by the number of these processes. A value of `0` disables this cache completely.
    - `--mm-encoder-tp-mode` indicates how to optimize multimodal encoder inference using tensor parallelism (TP). `data` splits the batched encoder input data across TP ranks while each rank hosts the full encoder weights, and is recommended when multimodal inputs are used. The default mode `weights` splits the encoder weights across TP ranks instead.
    - `--mm-processor-cache-type` sets the cache type used by the multimodal preprocessor and mapper. `shm` uses a shared-memory FIFO cache, while the default `lru` uses a mirrored LRU cache.
    - `--additional-config` contains Ascend-specific optimization options:
        - `"enable_cpu_binding": true` enables Ascend-native CPU binding, which pins the worker process, key runtime threads, memory pages, and NPU IRQs to CPU ranges close to the driven NPU, reducing cross-NUMA traffic and thread preemption and improving latency stability. It is enabled by default starting from vllm-ascend v0.18.0rc1.
        - `"enable_flashcomm1": true` enables the FlashComm communication optimization, which splits the tensor-parallel AllReduce into Reduce-Scatter and All-Gather to reduce communication overhead and improve prefill throughput.

=== "Atlas 800I A3 / Atlas 800I A2"

    The following example is for Atlas 800I A3 / Atlas 800I A2.

    ```bash
    #!/bin/sh
    # Load model from ModelScope to speed up download
    export MODEL_PATH=Eco-Tech/Qwen3.8-27B-w8a8
    export VLLM_USE_MODELSCOPE=True
    export HCCL_BUFFSIZE=512
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    sysctl -w vm.swappiness=0 || true
    sysctl -w kernel.numa_balancing=0 || true
    sysctl -w kernel.sched_migration_cost_ns=50000 || true  # no-op on EEVDF (Linux 6.6+), which removed this parameter
    # Size of the shared buffer (in MB) used by HCCL for NPU-to-NPU collective communication
    # To reduce memory fragmentation and avoid out of memory

    # Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.8-27B-w8a8) or a local directory path
    # Ensure the model path matches the directory recorded during download

    vllm serve $MODEL_PATH \
        --host 0.0.0.0 \
        --port 8000 \
        --data-parallel-size 1 \
        --tensor-parallel-size 2 \
        --quantization ascend \
        --served-model-name qwen3.8 \
        --max-num-seqs 32 \
        --max-model-len 133120 \
        --max-num-batched-tokens 16384 \
        --trust-remote-code \
        --allowed-local-media-path / \
        --dtype bfloat16 \
        --reasoning-parser qwen3 \
        --mm-processor-cache-gb 0 \
        --mm-encoder-tp-mode data \
        --mm-processor-cache-type shm \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.85 \
        --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
        --compilation-config '{"cudagraph_capture_sizes": [1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128],"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true}'
    ```

    Key Parameter Descriptions:

    - `--data-parallel-size 1` and `--tensor-parallel-size 2` are common settings for data parallelism (DP) and tensor parallelism (TP) sizes.
    - `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
    - `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
    - `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
        - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`.
        - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
        - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on HBM memory (activation value usage) will be greater.
    - `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak HBM memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak HBM memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the HBM memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
    - `--quantization ascend` indicates that quantization is used. To disable quantization, remove this option.
    - `--enable-prefix-caching` enables prefix caching, which is the default behavior. To disable it, use `--no-enable-prefix-caching`. For workloads with long shared prefixes, finer-grained hybrid prefix matching can be enabled with `--enable-prefix-caching --prefix-match-unit 16`. The match unit must divide every KV cache group's block size. A smaller unit can reuse more of a partial physical page, at the cost of additional hash metadata, copy-on-write operations, and an extra prefill split at the final matching boundary.
    - `--speculative-config` uses `qwen3_5_mtp` for `Qwen3.8-27B` because it shares the same MTP head design as `Qwen3.5-27B`.
    - `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are `"cudagraph_mode"` and `"cudagraph_capture_sizes"`, which have the following meanings:
        - `"cudagraph_mode"`: represents the specific graph mode. Currently, `"PIECEWISE"` and `"FULL_DECODE_ONLY"` are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, `"FULL_DECODE_ONLY"` is recommended.
        - `"cudagraph_capture_sizes"`: specifies the batch size tiers at which graphs are captured. The default value is `[1, 2, 4, 8, 16, 24, 32, 40,..., --max-num-seqs]`. In the graph mode, the batch size of each captured graph is fixed, and an actual batch size between two adjacent capture sizes is automatically padded up to the next capture size. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.
    - `--allowed-local-media-path` allows the API server to read local images or videos from the specified directory when processing multimodal inputs. In this example, `/` whitelists the entire file system. This is security-sensitive and should only be enabled in trusted environments.
    - `--dtype bfloat16` sets the data type for model weights and activations. `bfloat16` is used on Atlas 800I A3 / Atlas 800I A2; `float16` is required on Atlas 300I Duo instead.
    - `--reasoning-parser qwen3` enables the reasoning output parser for the OpenAI-compatible API. It splits the model's thinking content from the final answer and returns it in the `reasoning_content` field of `chat.completions` responses.
    - `--mm-processor-cache-gb` sets the size (in GiB) of the multimodal processor cache, which is used to avoid re-processing past multimodal inputs. The cache is duplicated in each API server process and engine core process, so the total host memory usage is the cache size multiplied by the number of these processes. A value of `0` disables this cache completely.
    - `--mm-encoder-tp-mode` indicates how to optimize multimodal encoder inference using tensor parallelism (TP). `data` splits the batched encoder input data across TP ranks while each rank hosts the full encoder weights, and is recommended when multimodal inputs are used. The default mode `weights` splits the encoder weights across TP ranks instead.
    - `--mm-processor-cache-type` sets the cache type used by the multimodal preprocessor and mapper. `shm` uses a shared-memory FIFO cache, while the default `lru` uses a mirrored LRU cache.
    - `--additional-config` contains Ascend-specific optimization options:
        - `"enable_cpu_binding": true` enables Ascend-native CPU binding, which pins the worker process, key runtime threads, memory pages, and NPU IRQs to CPU ranges close to the driven NPU, reducing cross-NUMA traffic and thread preemption and improving latency stability. It is enabled by default starting from vllm-ascend v0.18.0rc1.

=== "Atlas 300I Duo"

    Currently only the **TP** scenario is supported. Choose **TP=2** or **TP=4** according to the available devices. Replace `MODEL_PATH` with a ModelScope model id or a local directory path. The quantized version needs to start with the `--quantization ascend` parameter.

    === "Qwen3.8-27B-w8a8"

        Startup Command:

        ```bash
        #!/bin/sh
        # Load model from ModelScope to speed up download
        export VLLM_USE_MODELSCOPE=True

        # Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.8-27B-w8a8) or a local directory path
        # Ensure the model path matches the directory recorded during download
        export MODEL_PATH=Eco-Tech/Qwen3.8-27B-w8a8-310p

        vllm serve $MODEL_PATH \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 4 \
            --served-model-name qwen3.8 \
            --max-num-seqs 128 \
            --max-model-len 16384 \
            --trust-remote-code \
            --gpu-memory-utilization 0.90 \
            --mamba-ssm-cache-dtype float16 \
            --dtype float16 \
            --speculative-config '{"method": "qwen3_5_mtp","num_speculative_tokens":1}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [2,16]}' \
            --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'
        ```

    Key Parameter Descriptions:

    - `--tensor-parallel-size` sets the tensor parallel size. Choose **TP=2** or **TP=4** according to the available devices.
    - `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request. Configure it based on the actual workload and available memory; `Qwen3.8-27B` supports up to 262144.
    - `--max-num-seqs` indicates the maximum number of concurrent requests.
    - `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Configure this value according to the actual device memory; setting it too high may cause OOM. The default value is `0.9`.
    - `--mamba-ssm-cache-dtype` sets the data type of the Mamba SSM cache. On Atlas 300I Duo, only `float16` is supported.
    - `--dtype float16` must be set on Atlas 300I Duo. These devices only support the FP16 data type.
    - `--speculative-config` uses `qwen3_5_mtp` for `Qwen3.8-27B` because it shares the same MTP head design as `Qwen3.5-27B`. On Atlas 300I Duo, it is recommended to set `num_speculative_tokens` to `1`.
    - `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are `"cudagraph_mode"` and `"cudagraph_capture_sizes"`, which have the following meanings:
        - `"cudagraph_mode"`: represents the specific graph mode. Currently, `"PIECEWISE"` and `"FULL_DECODE_ONLY"` are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, `"FULL_DECODE_ONLY"` is recommended.
        - `"cudagraph_capture_sizes"`: specifies the batch size tiers at which graphs are captured. When tensor parallelism (TP) is enabled, hardware event-id constraints allow at most two capture sizes (for example, `[1, 8]`).
        With MTP enabled, calculate each capture size as `n * (num_speculative_tokens + 1)`, where `n` is a capture size for the deployment without MTP. For example, when `num_speculative_tokens` is `1`, the non-MTP sizes `[1,2,4,8]` become `[2,4,8,16]`.
    - `--additional-config` with `"ascend_compilation_config": {"enable_npugraph_ex": false}` is required on Atlas 300I Duo because `enable_npugraph_ex` is not supported on this platform.

Wait until the engine finishes loading weights and graph capture. A successful startup includes output similar to the following:

```output
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 5.2 Prefill-Decode Disaggregation (Atlas 800I A3) {: #52-prefill-decode-disaggregation-atlas-800i-a3 }

PD disaggregation separates Prefill and Decode into different service groups. Prefill groups process large prompt chunks with a large `--max-num-batched-tokens` in eager mode; Decode groups serve token generation with a small `--max-num-batched-tokens`, `FULL_DECODE_ONLY` aclgraph, and the recompute scheduler. A Pull proxy forwards each request to Prefill first and then to Decode, and Decode pulls the KV cache from Prefill through Mooncake.

The launch scripts below tune kernel parameters with `sysctl`. This requires `root` **and** write permission on `/proc/sys` — inside a container, being `root` (`id -u` returning `0`) is not sufficient by default: a non-privileged container mounts `/proc/sys` read-only. Use a privileged container (`--privileged`), or configure these kernel parameters on the host beforehand. Verify with a dry-run write before starting:

```shell
id -u          # must print 0 for root
sysctl -w vm.swappiness=0 && echo OK   # dry-run the write permission; "OK" means /proc/sys is writable
```

The scripts run under `set -euo pipefail`; a `sysctl` call that fails (for example `kernel.sched_migration_cost_ns`, which the EEVDF scheduler removed in Linux 6.6+) aborts the launch, so the scripts guard each tuning call with `|| true`.

This example deploys **3P1D on one 16-NPU Atlas 800I A3 host** for the 64K-input / 1K-output workload with a 0% prefix-cache hit rate. Each service group uses TP2 on 2 NPUs. "3P1D" describes four service groups; it does not mean that P and D must occupy separate hosts.

| Service group | Physical devices | TP | DP | HTTP port | Mooncake KV port | `engine_id` |
|---------------|------------------|----|----|-----------|------------------|-------------|
| Prefill P0    | 8,9              | 2  | 1  | 8081      | 58010            | 0           |
| Prefill P1    | 10,11            | 2  | 1  | 8082      | 58020            | 1           |
| Prefill P2    | 12,13            | 2  | 1  | 8083      | 58030            | 2           |
| Decode D0     | 14,15            | 2  | 1  | 8088      | 58080            | 7           |

We recommend using Mooncake for deployment. Refer to [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for the general PD disaggregation workflow. This example uses `MooncakeConnectorV1` with the standard Pull proxy in [Section 5.4](#54-request-forwarding).

#### 5.2.1 Prefill Service Group

Create `run_p.sh` on the A3 host. P0, P1, and P2 use the same script with different `ASCEND_RT_VISIBLE_DEVICES`, HTTP port, Mooncake `kv_port`, and `engine_id`. The values below are for P0.

```shell
#!/usr/bin/env bash
set -euo pipefail

unset ftp_proxy
unset https_proxy
unset http_proxy

MODEL_PATH="<LOCAL_W8A8_MODEL_PATH>"   # e.g. Qwen3.8-27B-w8a8

export ASCEND_RT_VISIBLE_DEVICES=8,9
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
sysctl -w vm.swappiness=0 || true
sysctl -w kernel.numa_balancing=0 || true
sysctl -w kernel.sched_migration_cost_ns=50000 || true  # no-op on EEVDF (Linux 6.6+), which removed this parameter
export TASK_QUEUE_ENABLE=2

# HCCL_OP_EXPANSION_MODE="AIV" is only used on the Decode side; unset any
# value inherited from a previous Decode/co-located session in this terminal,
# otherwise long-input Prefill slows down.
unset HCCL_OP_EXPANSION_MODE

# Single-host deployment; use the host service IP for multi-host deployment
export HCCL_IF_IP="127.0.0.1"
export HCCL_BUFFSIZE=512
export ASCEND_BUFFER_POOL="0:0"
# e.g. enp48s3u1u1; find the active NIC with "ip a" (do not use "lo",
# or Mooncake transfer setup fails)
export NETWORK_CARD_NAME="<SERVICE_NETWORK_INTERFACE>"
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"

vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 8081 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --served-model-name "qwen3.8" \
    --max-model-len 133120 \
    --max-num-batched-tokens 66000 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --quantization ascend \
    --enforce-eager \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
    --additional-config '{"enable_cpu_binding": true}' \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_producer",
        "kv_port": "58010",
        "engine_id": "0",
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 2},
            "decode":  {"dp_size": 1, "tp_size": 2}
        }
    }'
```

Run the script once per Prefill group in separate terminals:

```shell
bash run_p.sh   # P0; repeat with P1/P2 values for the other two groups
```

If a Prefill service does not become ready, confirm that its two devices are free, and that its HTTP port and Mooncake KV port are unused.

#### 5.2.2 Decode Service Group

Create `run_d.sh` on the same host. Use the same checkpoint, tokenizer, `--served-model-name`, and MTP configuration as Prefill.

```shell
#!/usr/bin/env bash
set -euo pipefail

unset ftp_proxy
unset https_proxy
unset http_proxy

MODEL_PATH="<LOCAL_W8A8_MODEL_PATH>"

export ASCEND_RT_VISIBLE_DEVICES=14,15
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
sysctl -w vm.swappiness=0 || true
sysctl -w kernel.numa_balancing=0 || true
sysctl -w kernel.sched_migration_cost_ns=50000 || true  # no-op on EEVDF (Linux 6.6+), which removed this parameter
export TASK_QUEUE_ENABLE=1

export HCCL_IF_IP="127.0.0.1"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=512
export ASCEND_BUFFER_POOL="0:0"
# e.g. enp48s3u1u1; find the active NIC with "ip a" (do not use "lo",
# or Mooncake transfer setup fails)
export NETWORK_CARD_NAME="<SERVICE_NETWORK_INTERFACE>"
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"

vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 8088 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --served-model-name "qwen3.8" \
    --max-model-len 133120 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --quantization ascend \
    --compilation-config '{"cudagraph_capture_sizes": [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
    --additional-config '{"enable_cpu_binding": true, "scheduler_config": {"recompute_scheduler_enable": true}}' \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",
        "kv_port": "58080",
        "engine_id": "7",
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 2},
            "decode":  {"dp_size": 1, "tp_size": 2}
        }
    }'
```

```shell
bash run_d.sh
```

#### 5.2.3 Proxy

After all P and D services report ready, start the standard Pull proxy on the same host. Run the command from the directory containing `load_balance_proxy_server_example.py` (see [Section 5.4](#54-request-forwarding)):

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy

python load_balance_proxy_server_example.py \
    --port 8199 \
    --host 0.0.0.0 \
    --prefiller-hosts 127.0.0.1 127.0.0.1 127.0.0.1 \
    --prefiller-ports 8081 8082 8083 \
    --decoder-hosts 127.0.0.1 \
    --decoder-ports 8088
```

Send client requests to `http://<HOST_IP>:8199`.

**Key Parameter Descriptions:**

The P:D group ratio and the parameter values in the two scripts should be adapted to the workload:

- The P:D group ratio follows the workload. Workloads dominated by long, uncached inputs (e.g., long-context summarization over unique documents) need more Prefill groups; workloads with a high prefix-cache hit rate spend most wall time in decode and shift capacity toward Decode (e.g., 1P2D).
- On Prefill, `--max-num-batched-tokens` should be no smaller than the effective uncached input length (within the HBM activation budget), so a prompt is prefilled in a single chunk instead of being split into multiple rounds. The example uses `66000` for a 64K-token 0%-cache-hit input budget; scale it down when prefix caching absorbs most of the input, because a cold request of the full length can transiently exceed the activation budget and cause OOM.
- `--max-num-seqs` on Prefill bounds concurrent prefill batches per group. Keep it small for long-input workloads so each chunk gets the full HBM activation budget; larger values are reasonable for short inputs.
- Qwen3.8-27B is a hybrid-attention model: only 16 of 64 layers use full attention, and the other 48 Gated DeltaNet layers carry a recurrent state. To enable prefix caching in PD mode, replace `--no-enable-prefix-caching` with `--enable-prefix-caching` on P and D, and keep the choice identical on both roles; the recurrent-state cache is aligned with the KV blocks automatically, no extra flag is needed.
- The FLA AscendC prefill kernel for the linear-attention layers is enabled automatically on hardware profiles that support it, so no extra configuration is needed for prefill throughput.
- `HCCL_OP_EXPANSION_MODE="AIV"` is set only on Decode. Do not export it (directly or through a shared terminal) when launching Prefill; long-input prefill slows down with AIV.
- Keep `--served-model-name`, the checkpoint, and the MTP `method`/`num_speculative_tokens` identical on P and D; change them only together.
- On Decode, the `cudagraph_capture_sizes` are coupled to MTP: with `num_speculative_tokens = N`, each capture size must be a multiple of `N + 1` (the example uses `4`-multiples for `num_speculative_tokens = 3`). If you change `num_speculative_tokens` or disable MTP, regenerate the list accordingly (e.g., `N = 4` → `5, 10, 15, ...`); otherwise the actual decode batch size cannot hit a captured graph, aclgraph silently falls back to eager, and TPOT degrades.
- `engine_id` must be unique across all services in the same PD cluster. Each service must also use its own HTTP port and Mooncake KV port.
- If multiple PD deployments run on the same host, stagger their HTTP and Mooncake KV ports.

### 5.3 Prefill-Decode Disaggregation (950PR&950DT Products) {: #53-prefill-decode-disaggregation-950pr950dt-products }

This example deploys **4P1D on one 950-series host** with the `Qwen3.8-27B-w8a8c8-mxfp8` checkpoint and C8 (`--kv-cache-dtype mxfp8`). Each group is single-NPU TP1, following the DP-first principle for this model size. P1-P4 use the same script with different `ASCEND_RT_VISIBLE_DEVICES` (0/1/2/3), HTTP port (9090/9092/9094/9096), Mooncake `kv_port` (58010/58020/58030/58040), and `engine_id` (3/4/5/6). The values below are for P1.

| Service group | NPUs | TP | DP | HTTP port | Mooncake KV port | `engine_id` |
| ------------- | ---- | -- | -- | --------- | ---------------- | ----------- |
| Prefill P1-P4 | 0-3 | 1 | 1 | 9090/9092/9094/9096 | 58010/58020/58030/58040 | 3/4/5/6 |
| Decode D | 4 | 1 | 1 | 10000 | 58080 | 2 |

#### 5.3.1 Prefill Node

```shell
#!/usr/bin/env bash
set -euo pipefail

unset ftp_proxy
unset https_proxy
unset http_proxy

MODEL_PATH="<LOCAL_W8A8C8_MODEL_PATH>"   # e.g. Qwen3.8-27B-w8a8c8-mxfp8

export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
sysctl -w vm.swappiness=0 || true
sysctl -w kernel.numa_balancing=0 || true
sysctl -w kernel.sched_migration_cost_ns=50000 || true  # no-op on EEVDF (Linux 6.6+), which removed this parameter
export TASK_QUEUE_ENABLE=1

# Single-host deployment; use the host service IP for multi-host deployment
export HCCL_IF_IP="127.0.0.1"
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export HCCL_BUFFSIZE=1024
export ASCEND_BUFFER_POOL="0:0"
# e.g. enp48s3u1u1; find the active NIC with "ip a" (do not use "lo",
# or Mooncake transfer setup fails)
export NETWORK_CARD_NAME="<SERVICE_NETWORK_INTERFACE>"

vllm serve "${MODEL_PATH}" \
    --served-model-name "qwen3.8" \
    --host 0.0.0.0 \
    --port 9090 \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype mxfp8 \
    --max-model-len 153600 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
    --trust-remote-code \
    --async-scheduling \
    --quantization ascend \
    --enforce-eager \
    --allowed-local-media-path / \
    --mm-processor-cache-gb 3 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --additional-config '{"enable_cpu_binding": true}' \
    --distributed-executor-backend mp \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_producer",
        "kv_port": "58010",
        "engine_id": "3",
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 1},
            "decode":  {"dp_size": 1, "tp_size": 1}
        }
    }'
```

#### 5.3.2 Decode Node

```shell
#!/usr/bin/env bash
set -euo pipefail

unset ftp_proxy
unset https_proxy
unset http_proxy

MODEL_PATH="<LOCAL_W8A8C8_MODEL_PATH>"

export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
sysctl -w vm.swappiness=0 || true
sysctl -w kernel.numa_balancing=0 || true
sysctl -w kernel.sched_migration_cost_ns=50000 || true  # no-op on EEVDF (Linux 6.6+), which removed this parameter
export TASK_QUEUE_ENABLE=1

export HCCL_IF_IP="127.0.0.1"
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export HCCL_BUFFSIZE=1024
export ASCEND_BUFFER_POOL="0:0"
# e.g. enp48s3u1u1; find the active NIC with "ip a" (do not use "lo",
# or Mooncake transfer setup fails)
export NETWORK_CARD_NAME="<SERVICE_NETWORK_INTERFACE>"

vllm serve "${MODEL_PATH}" \
    --served-model-name "qwen3.8" \
    --host 0.0.0.0 \
    --port 10000 \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype mxfp8 \
    --max-model-len 153600 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.90 \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
    --trust-remote-code \
    --async-scheduling \
    --quantization ascend \
    --allowed-local-media-path / \
    --mm-processor-cache-gb 3 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --compilation-config '{"cudagraph_capture_sizes": [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding": true, "scheduler_config": {"recompute_scheduler_enable": true}}' \
    --distributed-executor-backend mp \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",
        "kv_port": "58080",
        "engine_id": "2",
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 1},
            "decode":  {"dp_size": 1, "tp_size": 1}
        }
    }'
```

#### 5.3.3 Proxy

After all services report ready, start the standard Pull proxy on the same host:

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy

python load_balance_proxy_server_example.py \
    --port 8090 \
    --host 0.0.0.0 \
    --prefiller-hosts 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 \
    --prefiller-ports 9090 9092 9094 9096 \
    --decoder-hosts 127.0.0.1 \
    --decoder-ports 10000
```

Send client requests to `http://<HOST_IP>:8090`.

**C8 requirements for PD:**

- P and D must use the same checkpoint revision, tokenizer, quantization metadata, and `--kv-cache-dtype mxfp8`.
- C8 (`--kv-cache-dtype mxfp8`) is supported on 950PR&950DT Products only. With it enabled, `--block-size` must be a multiple of 512 (it defaults to 512 if unset).
- After the PD data path is up, check outputs for repeated punctuation or abnormal repetition, which indicate a KV-transfer or quantization mismatch.

### 5.4 Request Forwarding {: #54-request-forwarding }

Use the proxy paired with the actual connector, rather than choosing it by hardware name. The standard proxy script is in `examples/disaggregated_prefill_v1/`; run the command from that directory. Replace the example hosts with reachable service IPs and use each service's HTTP port. `--served-model-name` is a server-side CLI parameter configured identically when launching P and D; the client is not involved in configuring it, but each client request must set its `model` field to that same served-model name (for example `"model": "qwen3.8"` in the request body).

| Connector | Request Order | Proxy | Scope |
| --------- | ------------- | ----- | ----- |
| `MooncakeConnectorV1` | Proxy calls P, then D; D pulls KV | `load_balance_proxy_server_example.py` | A3 and 950PR&950DT PD recipes in [Section 5.2](#52-prefill-decode-disaggregation-atlas-800i-a3) and [Section 5.3](#53-prefill-decode-disaggregation-950pr950dt-products) |

Do not use a DP RPC port or a Mooncake KV-transfer port as a proxy backend. If requests reach the proxy but return no output, verify the P and D `/health` endpoints from the proxy host, confirm that the configured backend addresses exactly match each service's `--host` and HTTP port, and then run the end-to-end checks in [Section 6](#6-functional-verification).

## 6 Functional Verification {: #6-functional-verification }

After the service is started, the model can be invoked by sending a prompt. Two API interfaces are supported: `completions` and `chat/completions`. Use the `--served-model-name` you configured (`qwen3.8` for `Qwen3.8-27B`).

**Completions API:**

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0.7
    }'
```

**Chat Completions API:**

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8",
        "messages": [
            {"role": "user", "content": "The future of AI is"}
        ],
        "max_completion_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "reasoning_effort": "xhigh",
        "chat_template_kwargs": {
            "enable_thinking": true,
            "preserve_thinking": true
        }
    }'
```

Thinking mode is on by default and can be disabled by setting `enable_thinking` to `false` in `chat_template_kwargs`. The example above keeps thinking mode on, and `preserve_thinking` retains the reasoning context of historical messages.

Set `reasoning_effort` in the top-level Chat Completions request body to adjust
the model's reasoning depth. The example above uses `xhigh`; omit the field to
use the default `xhigh`. The model officially supports these levels:

- `xhigh`: default; deeper reasoning for complex tasks.
- `medium`: balances accuracy and speed.
- `low`: prioritizes speed and cost.

Expected Result: The service returns HTTP 200 OK. The JSON response contains the `choices` field with generated text. Example output for the completions API (content truncated for brevity):

```json
{
    "id": "cmpl-xxxxxxxxxxxxx",
    "object": "text_completion",
    "created": 1780971952,
    "model": "qwen3.8",
    "choices": [
        {
            "index": 0,
            "text": "The future of AI is a rapidly evolving landscape with breakthroughs in natural language understanding, multimodal reasoning, and autonomous agents. As models grow more capable and efficient...",
            "logprobs": null,
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 4,
        "total_tokens": 54,
        "completion_tokens": 50
    }
}
```

## 7 Accuracy Evaluation

Here is an accuracy evaluation method.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result. Here are the results of `Qwen3.8-27B`, `Qwen3.8-27B-w8a8` and `Qwen3.8-27B-w8a8-mxfp8` in `vllm-ascend:v0.23.0` for reference only.

| dataset | model | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| GPQA Diamond | Qwen3.8-27B | accuracy | gen | 90.40 |
| GPQA Diamond | Qwen3.8-27B-w8a8 | accuracy | gen | 89.90 |
| GPQA Diamond | Qwen3.8-27B-w8a8-mxfp8 | accuracy | gen | 89.39 |

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The current documentation focuses on the rapid adaptation and validation of the Qwen3.8-27B model on Ascend NPUs. Performance tuning results have not yet been fully verified. Recommended configurations for typical scenarios (e.g., long context, low latency, and high throughput) will be supplemented and updated here once the corresponding validation is completed. In the meantime, please refer to [Section 9.2](#92-tuning-guidelines) for general tuning guidance.
>
> **Atlas 300I Duo**: Currently only the TP scenario is supported. Choose **TP=2** or **TP=4** according to the available devices. With **TP=4**, `--max-model-len` can support **128k** and **256k** long-sequence scenarios.
>
> **950PR&950DT Products**: The `Qwen3.8-27B-w8a8c8-mxfp8` model weight easily fits in a single NPU (96 GB HBM per NPU on 950DT Products, 128 GB on 950PR Products). Following the **DP-first** principle, **TP=1 is the recommended default** for most scenarios, and the remaining NPUs should be allocated to DP for parallel request batches. The C8 weights are recommended for long-sequence scenarios and are served with `--kv-cache-dtype mxfp8`; for short-sequence scenarios (e.g., 16k input), use `Qwen3.8-27B-w8a8-mxfp8` without `--kv-cache-dtype mxfp8`. `--max-model-len` can support up to **262144** in the same TP=1 + DP=8 layout.

### 9.2 Tuning Guidelines {: #92-tuning-guidelines }

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

#### 9.2.2 Model-Specific Optimizations

##### Optimizations Enabled by Default

The following optimizations are enabled by default and require no additional configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
| --- | --- | --- |
| Chunked Prefill | The vLLM V1 scheduler splits long prefill inputs into chunks, with each step processing at most `--max-num-batched-tokens` tokens. | Reduces per-step memory peaks, enabling larger batch sizes and higher throughput. |
| Full Decode ACL Graph | Captures and replays the entire decode graph at once using `--compilation-config '"cudagraph_mode":"FULL_DECODE_ONLY"'`. | Reduces per-step operator dispatch overhead, stabilizing decode latency. |
| CPU Binding | Binds worker threads to dedicated CPU cores via `--additional-config '"enable_cpu_binding":true'`. | Reduces CPU scheduling jitter and stabilizes decode latency. |

##### Optimizations That Require Explicit Enabling

| Optimization Technique | Applicable Scenarios | Enablement Method | Technical Principle | Precautions |
| --- | --- | --- | --- | --- |
| Prefix Cache | Scenarios with repeated prefixes (e.g., shared system prompts, few-shot examples) | `--enable-prefix-caching` | Reuses cached KV state for repeated prefixes, avoiding redundant prefill computation. | The benefit depends on the prefix cache hit rate. |
| Qwen3.5 MTP Speculative Decoding | Decode-heavy workloads with acceptable acceptance rate | `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'` | Uses the built-in MTP draft head to speculate multiple tokens per step and verifies them in a single forward pass. | Tune the speculative token count for the target workload; benefit depends on the acceptance rate. |

## 10 FAQ

For common environment, installation, and general parameter issues, refer to the [Public FAQs](../../faqs.md).
