# Qwen3.8-27B

## 1 Introduction

Qwen3.8-27B is the 27-billion-parameter dense member of the Qwen3.8 family, the most capable generation in the Qwen open-model family to date. Built on the architectural foundation of Qwen3.5, it shares the same hybrid-attention backbone as the 2.4T MoE flagship: of its 64 layers, only 16 run full (gated) attention (`full_attention_interval: 4`) while the other 48 run linear attention (Gated DeltaNet) with a constant recurrent state. It is a native vision-language model — the architecture is `Qwen3_5ForConditionalGeneration` and `config.json` carries a `vision_config` — that understands images and videos, and it ships with a built-in MTP (Multi-Token Prediction) draft head and a native 262,144-token context window extensible up to 1,000,000 tokens.

Delivering substantial gains over Qwen3.5/Qwen3.6 across coding, professional work, research, and long-horizon agentic tasks, Qwen3.8-27B features stronger autonomous planning, more reliable end-to-end task completion, and broader downstream compatibility with popular harnesses and development tools. Thinking mode is on by default and can be disabled per request; reasoning depth is tunable via `reasoning_effort` (`xhigh`/`medium`/`low`), and reasoning context from historical messages is retained via `preserve_thinking`.

This document focuses on text serving on Ascend NPUs. It describes the main validation steps for the model, including supported features, prerequisites, installation, multi-node deployment, functional verification, accuracy and performance evaluation, performance tuning, and FAQs.

This document is validated and written based on **vLLM-Ascend 0.23.0**. The current model (Qwen3.8-27B) is first supported in this version.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_features.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

> **Note**: The support matrix records the maximum verified capability for this model. Adjust `--max-model-len`, `--max-num-seqs`, and `--max-num-batched-tokens` based on your service workload and available KV cache.

## 3 Prerequisites

### 3.1 Model Weight

The following model weights are available:

- `Qwen3.8-27B` (BF16 version): requires 1 Ascend950DT series (96GB × 8) node or 1 Atlas 800 A3 (64GB × 16) node. [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B)
- `Qwen3.8-27B-w8a8` (Quantized version): requires 1 Atlas 800 A3 (64GB × 16) node. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8)
- `Qwen3.8-27B-w8a8-mxfp8` (Quantized version): requires 1 Ascend950DT series (96GB × 8) node. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8-mxfp8)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy the model in a multi-node environment, verify the communication environment according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

=== "A3 series"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:qwen3.8-a3
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

=== "Ascend950DT series"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:qwen3.8-a5
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

After entering the container, verify that vLLM and vLLM-Ascend can be imported:

```shell
python -c "import vllm, vllm_ascend; print('vllm and vllm_ascend are ready')"
```

### 4.2 Source Code Installation

You can also build and install `vllm-ascend` from source. Refer to [set up using Python](../../installation.md#set-up-using-python).

If you want to deploy a multi-node service, install the same version of vLLM and vLLM-Ascend on each node.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and medium-scale inference scenarios.

Before starting the service:

- Replace the model path, parallel sizes and service port with values from the target environment.

=== "A3 series"

    The following example is for Atlas 800 A3. Quantized versions need `--quantization ascend`.

    ```shell
    #!/bin/sh
    # Load model from ModelScope to speed up download
    export VLLM_USE_MODELSCOPE=True
    # To reduce memory fragmentation and avoid out of memory
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    # Size of the shared buffer (in MB) used by HCCL for NPU-to-NPU collective communication
    export HCCL_BUFFSIZE=512
    # Whether OpenMP threads are bound to specific CPU cores
    export OMP_PROC_BIND=false
    # Number of OpenMP threads available for parallel regions
    export OMP_NUM_THREADS=1

    # Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.8-27B-w8a8) or a local directory path
    export MODEL_PATH=Eco-Tech/Qwen3.8-27B-w8a8

    vllm serve $MODEL_PATH \
        --host 0.0.0.0 \
        --port 8000 \
        --data-parallel-size 1 \
        --tensor-parallel-size 2 \
        --quantization ascend \
        --served-model-name qwen3.8 \
        --max-num-seqs 32 \
        --max-model-len 131072 \
        --max-num-batched-tokens 16384 \
        --trust-remote-code \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.85 \
        --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true}'
    ```

    Key Parameter Descriptions:

    - `--data-parallel-size 1` and `--tensor-parallel-size 2` are common settings for data parallelism (DP) and tensor parallelism (TP) sizes.
    - `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
    - `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
    - `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
        - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
        - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
        - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on HBM memory (activation value usage) will be greater.
    - `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak HBM memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak HBM memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the HBM memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
    - `--quantization ascend` indicates that quantization is used. To disable quantization, remove this option.
    - `--enable-prefix-caching` enables automatic prefix caching.
    - `--speculative-config` uses `qwen3_5_mtp` for `Qwen3.8-27B` because it shares the same MTP head design as `Qwen3.5-27B`.
    - `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are `"cudagraph_mode"` and `"cudagraph_capture_sizes"`, which have the following meanings:
        - `"cudagraph_mode"`: represents the specific graph mode. Currently, `"PIECEWISE"` and `"FULL_DECODE_ONLY"` are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, `"FULL_DECODE_ONLY"` is recommended.
        - `"cudagraph_capture_sizes"`: represents different levels of graph modes. The default value is `[1, 2, 4, 8, 16, 24, 32, 40,..., --max-num-seqs]`. In the graph mode, the input for graphs at different levels is fixed, and inputs between levels are automatically padded to the next level. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.

=== "Ascend950DT series"

    The following example is for Ascend950DT series. Quantized versions need `--quantization ascend`.

    ```bash
    #!/bin/sh
    # Load model from ModelScope to speed up download
    export VLLM_USE_MODELSCOPE=True
    # To reduce memory fragmentation and avoid out of memory
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    # Size of the shared buffer (in MB) used by HCCL for NPU-to-NPU collective communication
    export HCCL_BUFFSIZE=512
    # Whether OpenMP threads are bound to specific CPU cores
    export OMP_PROC_BIND=false
    # Number of OpenMP threads available for parallel regions
    export OMP_NUM_THREADS=1

    # Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.8-27B-w8a8-mxfp8) or a local directory path
    export MODEL_PATH=Eco-Tech/Qwen3.8-27B-w8a8-mxfp8

    vllm serve $MODEL_PATH \
        --host 0.0.0.0 \
        --port 8000 \
        --data-parallel-size 1 \
        --tensor-parallel-size 1 \
        --quantization ascend \
        --served-model-name qwen3.8 \
        --max-num-seqs 32 \
        --max-model-len 131072 \
        --max-num-batched-tokens 16384 \
        --trust-remote-code \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.85 \
        --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true}'
    ```

    Key Parameter Descriptions:

    - `--data-parallel-size 1` and `--tensor-parallel-size 1` are common settings for data parallelism (DP) and tensor parallelism (TP) sizes.
    - `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
    - `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
    - `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
        - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
        - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
        - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on HBM memory (activation value usage) will be greater.
    - `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak HBM memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak HBM memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the HBM memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
    - `--quantization ascend` indicates that quantization is used. To disable quantization, remove this option.
    - `--enable-prefix-caching` enables automatic prefix caching.
    - `--speculative-config` uses `qwen3_5_mtp` for `Qwen3.8-27B` because it shares the same MTP head design as `Qwen3.5-27B`.
    - `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are `"cudagraph_mode"` and `"cudagraph_capture_sizes"`, which have the following meanings:
        - `"cudagraph_mode"`: represents the specific graph mode. Currently, `"PIECEWISE"` and `"FULL_DECODE_ONLY"` are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, `"FULL_DECODE_ONLY"` is recommended.
        - `"cudagraph_capture_sizes"`: represents different levels of graph modes. The default value is `[1, 2, 4, 8, 16, 24, 32, 40,..., --max-num-seqs]`. In the graph mode, the input for graphs at different levels is fixed, and inputs between levels are automatically padded to the next level. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt. Two API interfaces are supported: `completions` and `chat/completions`. Use the `--served-model-name` you configured (`qwen3.8` for `Qwen3.8-27B`).

**Completions API:**

```bash
curl http://localhost:8000/v1/completions \
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
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8",
        "messages": [
            {"role": "user", "content": "The future of AI is"}
        ],
        "max_completion_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

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

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result. Here are the results of `Qwen3.8-27B`, `Qwen3.8-27B-w8a8` and `Qwen3.8-27B-w8a8-mxfp8` in `vllm-ascend:v0.23.0rc1` for reference only.

| dataset | model | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| GPQA Diamond | Qwen3.8-27B | accuracy | gen | 90.40 |
| GPQA Diamond | Qwen3.8-27B-w8a8 | accuracy | gen | 89.90 |
| GPQA Diamond | Qwen3.8-27B-w8a8-mxfp8 | accuracy | gen | 89.39 |

## 8 Performance Evaluation

### 8.1 Install AISBench

Run AISBench in a separate environment or container so that the load generator does not affect the serving processes. Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation).

### 8.2 Performance Service Configuration

Use the deployment in Section 5 as the baseline. Change the following values on the node:

| Parameter | Standard deployment | Performance test |
| --- | ---: | ---: |
| `--max-model-len` | 131072 | 250000 |
| `--max-num-batched-tokens` | 16384 | 8192 |
| `--gpu-memory-utilization` | 0.85 | 0.95 |

### 8.3 Run the Tests

Run performance evaluation of `Qwen3.8-27B` as an example. Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

```shell
vllm bench serve \
  --model Qwen/Qwen3.8-27B \
  --served-model-name qwen3.8 \
  --base-url http://<server_ip>:8000 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --num-prompts 16 \
  --max-concurrency 4 \
  --save-result \
  --result-dir ./
```

Record the node count, DP/TP topology, context length, concurrency, reasoning effort, and weight revision together with the result.

### 8.4 Enabled Optimizations

| Feature | Description |
| --- | --- |
| Chunked Prefill | Splits long prefill inputs into chunks to reduce per-step memory peaks. |
| W8A8 | Uses Ascend quantization for the validated checkpoint. |
| Lazy Safetensors | Avoids prefetching the complete NFS checkpoint. |
| MTP | Uses three speculative tokens with the `qwen3_5_mtp` method. |
| ACL Graph | Uses `FULL_DECODE_ONLY` replay. |
| CPU Binding | Reduces cross-core scheduling overhead. |

## 9 Performance Tuning

Use the deployment values above as a baseline. Adjust `max-model-len`, `max-num-seqs`, `max-num-batched-tokens`, and `gpu-memory-utilization` together for the target workload.

Refer to the [performance tuning guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md) and the [feature matrix](../../user_guide/support_matrix/feature_matrix.md) for additional guidance.

## 10 FAQ

For common environment, installation, and general parameter issues, refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
