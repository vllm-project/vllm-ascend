# Qwen3.5-27B & Qwen3.6-27B

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

Qwen3.5-27B and Qwen3.6-27B are dense hybrid Mamba-Transformer language models in the Qwen3.5/Qwen3.6 family, integrating breakthroughs in architectural efficiency, reinforcement learning scale, and global accessibility. They share the same hybrid attention design (GDN + full attention), so deployment on Ascend NPUs follows the same pattern for both models. They are suitable for general-purpose text generation tasks such as dialogue, content creation, and code generation running on Ascend NPUs.

This document will demonstrate the main validation steps for the models, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, as well as accuracy and performance evaluation.

It is **strongly recommended to use the `v0.30.0` version** of `vllm-ascend`. As a minimum-version requirement, `Qwen3.5-27B` is first supported in `vllm-ascend:v0.17.0rc1`, and `Qwen3.6-27B` is first supported in `vllm-ascend:v0.18.0rc1`. Support for Atlas 300I DUO and 950DT Products starts from `vllm-ascend:v0.23.0rc1`.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

|  Weight Version | Hardware Requirements | Download Links |
|-----------------|-----------------------|----------------|
| `Qwen3.5-27B` (BF16 version) | 1 Atlas 800I A3 (64GB × 16) node or 1 Atlas 800I A2 (64GB × 8) node or Atlas 300I DUO | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3.5-27B) \| [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-27B) |
| `Qwen3.5-27B-w8a8` (Quantized version) | 1 Atlas 800I A3 (64GB × 16) node or 1 Atlas 800I A2 (64GB × 8) node or Atlas 300I DUO | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.5-27B-w8a8-mtp) |
| `Qwen3.6-27B` (BF16 version) | 1 950DT Products(96GB × 8) node or 1 Atlas 800I A3 (64GB × 16) node or 1 Atlas 800I A2 (64GB × 8) node or Atlas 300I DUO | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3.6-27B) \| [Hugging Face](https://huggingface.co/Qwen/Qwen3.6-27B) |
| `Qwen3.6-27B-w8a8` (Quantized version) | 1 Atlas 800I A3 (64GB × 16) node or 1 Atlas 800I A2 (64GB × 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-27B-w8a8) |
| `Qwen3.6-27B-w8a8-mxfp8` (Quantized version) | 1 950DT Products (96GB × 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-27B-w8a8-mxfp8) |
| `Qwen3.6-27B-w8a8c8-mxfp8` (Quantized version) | 1 950DT Products (96GB × 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-27B-w8a8c8-mxfp8) |
| `Qwen3.6-27B-w8a8-310p`(Quantized version) | 1 Atlas 300I DUO | [ModelScope](https://modelscope.cn/models/Eco-Tech/Qwen3.6-27B-W8A8-310P) |

- `Qwen3.6-27B-w8a8c8-mxfp8` (W8A8 MXFP8 quantization with C8 MXFP8 KV cache): serve it with `--kv-cache-dtype mxfp8`; recommended for long-sequence scenarios.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

### 3.2 Verify Multi-node Communication

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image).

It is **recommended to use the `v0.30.0` version** of the `vllm-ascend` image to ensure the best compatibility and access to the latest features. As a minimum-version requirement, use `vllm-ascend:v0.17.0rc1` (or a later version) for `Qwen3.5-27B`, and `vllm-ascend:v0.18.0rc1` (or a later version) for `Qwen3.6-27B`. For `Qwen3.6-27B` on Atlas 800I A3, please use the matching `v0.18.0rc1-a3` (or a later `-a3`) image. For Atlas 300I DUO, use `vllm-ascend:v0.23.0rc1-310p` (or a later `-310p`) image.

=== "Atlas 800I A3"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net=host \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci8 \
        --device /dev/davinci9 \
        --device /dev/davinci10 \
        --device /dev/davinci11 \
        --device /dev/davinci12 \
        --device /dev/davinci13 \
        --device /dev/davinci14 \
        --device /dev/davinci15 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

=== "Atlas 800I A2"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net=host \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

=== "Atlas 300I DUO"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -p 8000:8000 \
        -it $IMAGE bash
    ```

=== "950DT Products"

    Start the docker image on each node.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
    export NAME=vllm-ascend

    docker run --rm \
        --name $NAME \
        --net=host \
        --shm-size=1g \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci_manager \
        --device /dev/hisi_hdc \
        --device /dev/ummu \
        --device /dev/uburma \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
        -v /etc/hixlep/:/etc/hixlep/ \
        -v /root/.cache:/root/.cache \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
        -v /usr/lib64:/usr/lib64 \
        -it $IMAGE bash
    ```

After a successful docker run, you can verify the running container service by executing the `docker ps` command. The expected result is that the container `vllm-ascend` is listed with status `Up`, confirming the docker installation is successful.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

1. Clone the repository and install `vllm-ascend` from source:

    ```bash
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    pip install -e .
    ```

    For the complete installation steps, refer to [installation](../../getting_started/installation.md).

    !!! note

        On Atlas 300I DUO, you may need to uninstall `triton-ascend` and `triton` to avoid dependency conflicts:

        ```bash
        pip uninstall -y triton-ascend triton
        ```

2. If you want to deploy a multi-node environment, you need to set up the environment on each node.

To verify the source code installation, run the following command and confirm the displayed version matches the one you installed:

```bash
pip show vllm-ascend
```

Expected result: The version information of `vllm-ascend` is displayed, confirming a successful installation.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and medium-scale inference scenarios. The `Qwen3.5-27B`, `Qwen3.5-27B-w8a8`, `Qwen3.6-27B`, and `Qwen3.6-27B-w8a8` models can all be deployed on 1 Atlas 800I A3 (64GB × 16), 1 Atlas 800I A2 (64GB × 8). On Atlas 300I DUO, at least 2 devices are required. The `Qwen3.6-27B`, `Qwen3.6-27B-w8a8-mxfp8` and `Qwen3.6-27B-w8a8c8-mxfp8` models can all be deployed on 1 950DT Products (96GB × 8). The quantized versions need to start with the `--quantization ascend` parameter.

Both `Qwen3.5-27B` and `Qwen3.6-27B` share the same MTP head design, so the `qwen3_5_mtp` speculative decoding method can be used for both.

=== "Atlas 800I A3 / Atlas 800I A2"

    The following examples are for Atlas 800I A3 / Atlas 800I A2. Quantized versions need `--quantization ascend`.

    === "Qwen3.5-27B-w8a8"

        Startup Command:

        ```bash
        #!/bin/sh
        # Load model from ModelScope to speed up download
        export MODEL_PATH=Eco-Tech/Qwen3.5-27B-w8a8-mtp
        export VLLM_USE_MODELSCOPE=True
        export HCCL_BUFFSIZE=512
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        sysctl -w vm.swappiness=0
        sysctl -w kernel.numa_balancing=0
        sysctl kernel.sched_migration_cost_ns=50000
        # To reduce memory fragmentation and avoid out of memory
        # Size of the shared buffer (in MB) used by HCCL for NPU-to-NPU collective communication

        # Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.5-27B-w8a8-mtp) or a local directory path
        # Ensure the model path matches the directory recorded during download

        vllm serve $MODEL_PATH \
            --host 0.0.0.0 \
            --port 8000 \
            --data-parallel-size 1 \
            --tensor-parallel-size 2 \
            --quantization ascend \
            --served-model-name qwen3.5 \
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
            --gpu-memory-utilization 0.85 \
            --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
            --compilation-config '{"cudagraph_capture_sizes": [1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128],"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --additional-config '{"enable_cpu_binding":true}'
        ```

    === "Qwen3.6-27B-w8a8"

        Startup Command:

        ```bash
        #!/bin/sh
        # Load model from ModelScope to speed up download
        export MODEL_PATH=Eco-Tech/Qwen3.6-27B-w8a8
        export VLLM_USE_MODELSCOPE=True
        export HCCL_BUFFSIZE=512
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        sysctl -w vm.swappiness=0
        sysctl -w kernel.numa_balancing=0
        sysctl kernel.sched_migration_cost_ns=50000
        # To reduce memory fragmentation and avoid out of memory
        # Size of the shared buffer (in MB) used by HCCL for NPU-to-NPU collective communication

        # Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.6-27B-w8a8) or a local directory path
        # Ensure the model path matches the directory recorded during download

        vllm serve $MODEL_PATH \
            --host 0.0.0.0 \
            --port 8000 \
            --data-parallel-size 1 \
            --tensor-parallel-size 2 \
            --quantization ascend \
            --served-model-name qwen3.6 \
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
            --gpu-memory-utilization 0.85 \
            --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
            --compilation-config '{"cudagraph_capture_sizes": [1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128],"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --additional-config '{"enable_cpu_binding":true}'
        ```

    Key Parameter Descriptions:

    - `--data-parallel-size 1` and `--tensor-parallel-size 2` are common settings for data parallelism (DP) and tensor parallelism (TP) sizes.
    - `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request. The Qwen3.6-27B model supports up to 262144.
    - `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
    - `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
        - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
        - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
        - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on HBM memory (activation value usage) will be greater.
    - `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak HBM memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak HBM memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the HBM memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
    - Prefix caching is enabled by default, and the example above uses the default. Add `--no-enable-prefix-caching` to disable it if needed. For workloads with long shared prefixes, finer-grained hybrid prefix matching can be enabled with `--enable-prefix-caching --prefix-match-unit 16`. The match unit must divide every KV cache group's block size. A smaller unit can reuse more of a partial physical page, at the cost of additional hash metadata, copy-on-write operations, and an extra prefill split at the final matching boundary.
    - `--quantization ascend` indicates that quantization is used. To disable quantization, remove this option.
    - `--speculative-config` uses `qwen3_5_mtp` for both `Qwen3.5-27B` and `Qwen3.6-27B` because they share the same MTP head design.
    - `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are `"cudagraph_mode"` and `"cudagraph_capture_sizes"`, which have the following meanings:
        - `"cudagraph_mode"`: represents the specific graph mode. Currently, `"PIECEWISE"` and `"FULL_DECODE_ONLY"` are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, `"FULL_DECODE_ONLY"` is recommended.
        - `"cudagraph_capture_sizes"`: specifies the batch size tiers at which graphs are captured. The default value is `[1, 2, 4, 8, 16, 24, 32, 40,..., --max-num-seqs]`. In the graph mode, the batch size of each captured graph is fixed, and an actual batch size between two adjacent capture sizes is automatically padded up to the next capture size. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.
    - `--allowed-local-media-path` allows the API server to read local images or videos from the specified directory when processing multimodal inputs. In this example, `/` whitelists the entire file system. This is security-sensitive and should only be enabled in trusted environments.
    - `--dtype bfloat16` sets the data type for model weights and activations. `bfloat16` is used on Atlas 800I A3 / Atlas 800I A2; `float16` is required on Atlas 300I DUO instead.
    - `--reasoning-parser qwen3` enables the reasoning output parser for the OpenAI-compatible API. It splits the model's thinking content from the final answer and returns it in the `reasoning_content` field of `chat.completions` responses.
    - `--mm-processor-cache-gb` sets the size (in GiB) of the multimodal processor cache, which is used to avoid re-processing past multimodal inputs. The cache is duplicated in each API server process and engine core process, so the total host memory usage is the cache size multiplied by the number of these processes. A value of `0` disables this cache completely.
    - `--mm-encoder-tp-mode` indicates how to optimize multimodal encoder inference using tensor parallelism (TP). `data` splits the batched encoder input data across TP ranks while each rank hosts the full encoder weights, and is recommended when multimodal inputs are used. The default mode `weights` splits the encoder weights across TP ranks instead.
    - `--mm-processor-cache-type` sets the cache type used by the multimodal preprocessor and mapper. `shm` uses a shared-memory FIFO cache, while the default `lru` uses a mirrored LRU cache.
    - `--additional-config` contains Ascend-specific optimization options:
        - `"enable_cpu_binding": true` enables Ascend-native CPU binding, which pins the worker process, key runtime threads, memory pages, and NPU IRQs to CPU ranges close to the driven NPU, reducing cross-NUMA traffic and thread preemption and improving latency stability. It is enabled by default starting from vllm-ascend v0.18.0rc1.

=== "Atlas 300I DUO"

    Currently only the **TP** scenario is supported. Choose **TP=2** or **TP=4** according to the available devices. Replace `MODEL_PATH` with a ModelScope model id or a local directory path.The quantized versions need to start with the `--quantization ascend` parameter.

    === "Qwen3.5-27B-w8a8"

        Startup Command:

        ```bash
        #!/bin/sh
        # Load model from ModelScope to speed up download
        export MODEL_PATH=Eco-Tech/Qwen3.5-27B-w8a8-mtp
        export VLLM_USE_MODELSCOPE=True

        # Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.5-27B-w8a8-mtp) or a local directory path
        # Ensure the model path matches the directory recorded during download

        vllm serve $MODEL_PATH \
            --host 127.0.0.1 \
            --port 8000 \
            --tensor-parallel-size 4 \
            --served-model-name qwen3.5 \
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

    === "Qwen3.6-27B-w8a8"

        Startup Command:

        ```bash
        #!/bin/sh
        # Load model from ModelScope to speed up download
        export MODEL_PATH=Eco-Tech/Qwen3.6-27B-w8a8-310p
        export VLLM_USE_MODELSCOPE=True

        # Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.6-27B-w8a8) or a local directory path
        # Ensure the model path matches the directory recorded during download

        vllm serve $MODEL_PATH \
            --host 127.0.0.1 \
            --port 8000 \
            --tensor-parallel-size 4 \
            --served-model-name qwen3.6 \
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
    - `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request. Configure it based on the actual workload and available memory; The Qwen3.6-27B model supports up to 262144.
    - `--max-num-seqs` indicates the maximum number of concurrent requests.
    - `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Configure this value according to the actual device memory; setting it too high may cause OOM. The default value is `0.9`.
    - `--mamba-ssm-cache-dtype` sets the data type of the Mamba SSM cache. On Atlas 300I DUO, only `float16` is supported.
    - `--dtype float16` must be set on Atlas 300I DUO. These devices only support the FP16 data type.
    - `--speculative-config` uses `qwen3_5_mtp` for both `Qwen3.5-27B` and `Qwen3.6-27B` because they share the same MTP head design. On Atlas 300I DUO, it is recommended to set `num_speculative_tokens` to `1`.
    - `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are `"cudagraph_mode"` and `"cudagraph_capture_sizes"`, which have the following meanings:
        - `"cudagraph_mode"`: represents the specific graph mode. Currently, `"PIECEWISE"` and `"FULL_DECODE_ONLY"` are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, `"FULL_DECODE_ONLY"` is recommended.
        - `"cudagraph_capture_sizes"`: specifies the batch size tiers at which graphs are captured. When tensor parallelism (TP) is enabled, hardware event-id constraints allow at most two capture sizes (for example, `[1, 8]`). With MTP enabled, calculate each capture size as `n * (num_speculative_tokens + 1)`, where `n` is a capture size for the deployment without MTP. For example, when `num_speculative_tokens` is `1`, the non-MTP sizes `[1,2,4,8]` become `[2,4,8,16]`.
    - `--additional-config` with `"ascend_compilation_config": {"enable_npugraph_ex": false}` is required on Atlas 300I DUO because `enable_npugraph_ex` is not supported on this platform.

=== "950DT Products"

    The following example is for 950DT Products. Quantized versions need `--quantization ascend`.

    === "Qwen3.6-27B-w8a8c8-mxfp8"

        Startup Command:

        ```bash
        #!/bin/sh
        # Load model from ModelScope to speed up download
        export MODEL_PATH=Eco-Tech/Qwen3.6-27B-w8a8c8-mxfp8
        export VLLM_USE_MODELSCOPE=True
        export HCCL_BUFFSIZE=1024
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        sysctl -w vm.swappiness=0
        sysctl -w kernel.numa_balancing=0
        sysctl kernel.sched_migration_cost_ns=50000
        # To reduce memory fragmentation and avoid out of memory
        # Size of the shared buffer (in MB) used by HCCL for NPU-to-NPU collective communication

        # Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.6-27B-w8a8c8-mxfp8) or a local directory path
        # Ensure the model path matches the directory recorded during download

        vllm serve $MODEL_PATH \
            --host 0.0.0.0 \
            --port 8000 \
            --data-parallel-size 1 \
            --tensor-parallel-size 1 \
            --quantization ascend \
            --kv-cache-dtype mxfp8 \
            --served-model-name qwen3.6 \
            --max-num-seqs 32 \
            --max-model-len 153600 \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --allowed-local-media-path / \
            --mm-processor-cache-gb 0 \
            --mm-encoder-tp-mode data \
            --mm-processor-cache-type shm \
            --gpu-memory-utilization 0.90 \
            --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
            --compilation-config '{"cudagraph_capture_sizes": [1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128],"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --additional-config '{"enable_cpu_binding": true, "enable_flashcomm1": true}'
        ```

    Key Parameter Descriptions:

    - `--data-parallel-size 1` and `--tensor-parallel-size 1` are common settings for data parallelism (DP) and tensor parallelism (TP) sizes.
    - `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request. The Qwen3.6-27B model supports up to 262144.
    - `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
    - `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
        - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
        - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
        - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on HBM memory (activation value usage) will be greater.
    - `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak HBM memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak HBM memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the HBM memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
    - Prefix caching is enabled by default, and the example above uses the default. Add `--no-enable-prefix-caching` to disable it if needed. For workloads with long shared prefixes, finer-grained hybrid prefix matching can be enabled with `--enable-prefix-caching --prefix-match-unit 16`. The match unit must divide every KV cache group's block size. A smaller unit can reuse more of a partial physical page, at the cost of additional hash metadata, copy-on-write operations, and an extra prefill split at the final matching boundary.
    - `--quantization ascend` indicates that quantization is used. To disable quantization, remove this option.
    - `--kv-cache-dtype mxfp8` enables the C8 (MXFP8 KV cache) and is required by the `Qwen3.6-27B-w8a8c8-mxfp8` weights. It stores the KV cache of the full-attention layers as FP8 (E4M3) with E8M0 scales, and runs the full-attention layers on the dedicated QFA (quant flash attention) kernel. The W8A8C8 weights bring performance gains in long-sequence scenarios. With this option, `--block-size` must be a multiple of `512` (it defaults to `512` if unset).
    - For short-sequence scenarios (e.g., 16K input length), the W8A8 weights (e.g., `Qwen3.6-27B-w8a8-mxfp8`) are recommended instead. When serving a W8A8 weight, remove `--kv-cache-dtype mxfp8` from the startup command. Note that C8 quantization is only supported on 950PR&950DT Products.
    - `--speculative-config` uses `qwen3_5_mtp` for `Qwen3.6-27B` because it shares the same MTP head design as `Qwen3.5-27B`.
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

Common Issues Tip: If you encounter issues, please refer to the [Public FAQs](../../faqs.md) for troubleshooting.

Service Verification:

If the service starts successfully, the following startup log will be displayed:

```text
(APIServer pid=<pid>) INFO:     Started server process [<pid>]
(APIServer pid=<pid>) INFO:     Waiting for application startup.
(APIServer pid=<pid>) INFO:     Application startup complete.
```

For functional testing (e.g., `completions` and `chat.completions` curl examples with expected responses), please refer to [Section 6](#6-functional-verification).

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt. Two API interfaces are supported: `completions` and `chat.completions`. Use the `--served-model-name` you configured (`qwen3.5` for `Qwen3.5-27B` or `qwen3.6` for `Qwen3.6-27B`).

**Completions API:**

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0
    }'
```

> **Note**: For `Qwen3.6-27B`, set `"model": "qwen3.6"` in the request body.

**Chat Completions API:**

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5",
        "messages": [
            {"role": "user", "content": "The future of AI is"}
        ],
        "max_completion_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95
    }'
```

> **Note**: For `Qwen3.6-27B`, set `"model": "qwen3.6"` in the request body.

Expected Result: The service returns HTTP 200 OK. The JSON response contains the `choices` field with generated text. Example output for the completions API (content truncated for brevity):

```json
{
    "id": "cmpl-xxxxxxxxxxxxx",
    "object": "text_completion",
    "created": 1780971952,
    "model": "qwen3.5",
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

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result. Here is the result of `Qwen3.5-27B-w8a8` in `vllm-ascend:v0.17.0rc1` for reference only. The accuracy result of `Qwen3.6-27B-w8a8` can be obtained in the same way and is not listed here.

**Accuracy Evaluation Config File:**

```bash
# Example configuration: benchmarks/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="your_model_path",
        model="qwen3.5",
        request_rate=0,
        retry=2,
        host_ip="127.0.0.1",
        host_port=8000,
        max_out_len=32768,
        batch_size=32,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

> For `Qwen3.6-27B-w8a8`, change `model` to `qwen3.6` and `path` to the corresponding model weight path.

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| gsm8k | - | accuracy | gen | 96.74 |

### 7.2 Using Language Model Evaluation Harness

Using the `gsm8k` dataset as an example test dataset, run the accuracy evaluation for `Qwen3.5-27B-w8a8` in online mode.

1. For `lm_eval` installation, please refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).
2. Run `lm_eval` to execute the accuracy evaluation.

```shell
# For Qwen3.5-27B-w8a8
export VLLM_USE_MODELSCOPE=True
vllm serve Eco-Tech/Qwen3.5-27B-w8a8-mtp \
    --served-model-name qwen3.5 \
    --trust-remote-code \
    --quantization ascend \
    --tensor-parallel-size 2 \
    --max-model-len 133000 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching

# Run lm_eval in another terminal
lm_eval \
  --model local-completions \
  --model_args model=qwen3.5,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to [Section 9.2](#92-tuning-guidelines) for tuning based on actual conditions.
>
> **Parallelism Strategy**: `Qwen3.5-27B-w8a8` and `Qwen3.6-27B-w8a8` are only ~30 GB and easily fit in a single NPU (64 GB HBM per NPU). Following the **DP-first** principle, **TP=2 is the recommended default** for most scenarios, and the remaining NPUs should be allocated to DP for parallel request batches. **TP=8 is only recommended for ultra-long context (256k+) scenarios**, where it shards the KV cache across 8 NPUs to maximize the available context window per rank. For `Qwen3.6-27B-w8a8`, you can also raise `--max-model-len` up to 262144 in the same TP/DP layout.
>
> **Atlas 300I DUO**: Currently only the TP scenario is supported. Choose **TP=2** or **TP=4** according to the available devices. With **TP=4**, `--max-model-len` can support **128k** and **256k** long-sequence scenarios.
>
> **950DT Products**: The `Qwen3.6-27B-w8a8c8-mxfp8` model weight easily fits in a single NPU (96 GB HBM per NPU). Following the **DP-first** principle, **TP=1 is the recommended default** for most scenarios, and the remaining NPUs should be allocated to DP for parallel request batches. The C8 weights are recommended for long-sequence scenarios and are served with `--kv-cache-dtype mxfp8`; for short-sequence scenarios (e.g., 16K input), use `Qwen3.6-27B-w8a8-mxfp8` without `--kv-cache-dtype mxfp8`. `--max-model-len` can support up to **262144** in the same TP=1 + DP=8 layout.

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
|----------|----------------|-------------|----------------|---------------------|
| High Throughput<br>(128k context) | Single-Node (A2) | 8 (A2) | Qwen3.5-27B-w8a8 / Qwen3.6-27B-w8a8 | TP=2 + DP=4 fully utilizes all 8 NPUs for parallel request batches |
| High Throughput<br>(128k context) | Single-Node (A3) | 16 (A3) | Qwen3.5-27B-w8a8 / Qwen3.6-27B-w8a8 | TP=2 + DP=8 fully utilizes all 16 NPUs for parallel request batches |
| Low Latency<br>(128k context) | Single-Node (A3) | 16 (A3) | Qwen3.5-27B-w8a8 / Qwen3.6-27B-w8a8 | TP=2 + DP=8 reduces per-layer Allreduce overhead for small interactive batches |
| Long Context<br>(256k+ context) | Single-Node (A3) | 16 (A3) | Qwen3.5-27B-w8a8 / Qwen3.6-27B-w8a8 | TP=8 + DP=2 shards the KV cache across 8 NPUs to maximize the available context window |
| High Throughput<br>(128k context) | Single-Node (950DT Products) | 8 (950DT Products) | Qwen3.6-27B-w8a8c8-mxfp8 | TP=1 + DP=8 fully utilizes all 8 NPUs for parallel request batches |
| Long Context<br>(256k+ context) | Single-Node (950DT Products) | 8 (950DT Products) | Qwen3.6-27B-w8a8c8-mxfp8 | TP=1 + DP=8 maximizes the available context window while keeping all 8 NPUs busy |

> `*Total NPUs` indicates the total number of NPUs used across all nodes. 1 Atlas 800I A3 node = 16 NPUs, 1 Atlas 800I A2 node = 8 NPUs, 1 950DT Products node = 8 NPUs.

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | NPUs | TP | DP | Max Num Seqs | Max Num Batched Tokens | Max Model Len | MTP Speculation Num | Async Scheduling |
|----------|---------------|-------|----|----|----|-------------|--------------------|---------------------|------------------|
| High Throughput (128k) | Single-Node (A2) | 8 | 2 | 4 | 32 | 16384 | 133000 | 3 | On |
| High Throughput (128k) | Single-Node (A3) | 16 | 2 | 8 | 32 | 16384 | 133000 | 3 | On |
| Low Latency (128k) | Single-Node (A3) | 16 | 2 | 8 | 4 | 4096 | 133000 | 3 | On |
| Long Context (256k+) | Single-Node (A3) | 16 | 8 | 2 | 8 | 8192 | 262144 | 3 | On |
| High Throughput (128k) | Single-Node (950DT Products) | 8 | 1 | 8 | 32 | 16384 | 133000 | 3 | On |
| Long Context (256k+) | Single-Node (950DT Products) | 8 | 1 | 8 | 32 | 16384 | 262144 | 3 | On |

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Section 5](#5-online-service-deployment).

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md).
