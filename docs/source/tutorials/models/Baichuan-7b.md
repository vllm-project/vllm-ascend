# Baichuan-7B

## 1 Introduction

Baichuan-7B is a 7 billion parameter large language model developed by Baichuan AI. It is an open-source model designed for various natural language processing tasks including text generation, question answering, and conversation.

This document will demonstrate the main validation steps for Baichuan-7B in the vLLM-Ascend environment, including supported features, environment preparation, single-node deployment, as well as accuracy and performance evaluation. By tailoring service-level configurations to fit different use cases, you can ensure optimal performance across various scenarios.

Baichuan-7B is supported in vLLM-Ascend for Atlas A2 Series inference products. This document is validated based on **vLLM-Ascend v0.21.0**. All **v0.21.0 and later versions** can run stably.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

The Baichuan-7B model weight is available for download. It is recommended to download the model weight to a shared directory accessible to all nodes.

| Model | Hardware Requirement | Download |
|-------|---------------------|----------|
| Baichuan-7B (BF16) | 1 Atlas A2 inference products (64GB × 1) | [Download](https://modelscope.cn/models/baichuan-inc/baichuan-7B) |

### 3.2 Verify Multi-node Communication

If you need to deploy a multi-node environment, verify the multi-node communication according to [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Baichuan-7B.

**Docker Pull:**

```bash
docker pull quay.io/ascend/vllm-ascend:v0.23.0rc1
```

**Docker Run:**

Start the docker image on your node.

```bash
export IMAGE=quay.io/ascend/vllm-ascend:v0.23.0rc1

docker run --rm \
--name vllm-ascend-env \
--shm-size=500g \
--net=host \
--privileged=true \
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
-v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/sbin:/usr/local/sbin \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
-v /root/.cache:/root/.cache \
-itd \
--entrypoint=bash \
$IMAGE
```

The default workdir is `/workspace`. vLLM and vLLM-Ascend are installed as Python packages in site-packages.

Installation Verification:
After starting the container, run the following command to verify the installation:

```bash
docker ps | grep vllm-ascend-env
```

Expected result: The container is listed with status Up. You can also verify the vllm-ascend version inside the container:

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, matching the pulled image version.

### 4.2 Source Code Installation

If you prefer to build from source instead of using the Docker image, install vLLM-Ascend following the [Installation Guide](../../installation.md).

To verify the source installation:

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, confirming a successful installation.

!!! note

    If deploying a multi-node environment, set up the environment on each node.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and small-to-medium scale inference scenarios.

**Start the server:**
> The following command is an example configuration. Adjust the parameters based on your actual scenario.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_BUFFSIZE=600
export HCCL_CONNECT_TIMEOUT=600
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1

vllm serve /path/to/baichuan-7B \
    --served-model-name baichuan \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 65536 \
    --no-enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8031 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

!!! note

    - [vLLM Serving Arguments documentation](https://docs.vllm.com.cn/en/latest/cli/serve/?h=block+size#arguments) — Additional parameter details for vLLM serve commands.
    - [Environment Variables](../../user_guide/configuration/env_vars.md) — Ascend-specific environment variables (`HCCL_*`, etc.).

**Service Verification:**

If the service starts successfully, the following startup log will be displayed:

```text
(APIServer pid=<pid>) INFO:     Started server process [<pid>]
(APIServer pid=<pid>) INFO:     Waiting for application startup.
(APIServer pid=<pid>) INFO:     Application startup complete.
```

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt.

**Completions API:**

```bash
curl http://localhost:8031/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "baichuan",
        "prompt": "给我介绍一下人工智能的发展历史。",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512
    }'
```

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

## 7 Accuracy Evaluation

### Using AISBench

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

The following is an example configuration for the accuracy evaluation config file:

**Accuracy Evaluation Config File:**

```python
# Example configuration: benchmarks/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py
from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        attr="service",
        type=VLLMCustomAPI,
        abbr="vllm-api-general",
        path="baichuan-7B",
        model="baichuan",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8031,
        url="",
        max_out_len=3500,
        batch_size=10,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0.6,
            ignore_eos=False,
            top_p=0.95,
        ),
    )
]
```

**Run the accuracy evaluation using the C-Eval dataset as an example:**

```bash
ais_bench --models vllm_api_general --datasets ceval_stem_gen_5_shot_chat_prompt --debug
```

> The --models parameter value corresponds to the abbr field in the configuration file above. Adjust max_out_len, batch_size, and dataset tasks based on your scenario.

**Accuracy Results:**

The following accuracy results were validated on Atlas A2 Series with BF16 precision:

| dataset | version | metric | mode | vllm-api-general |
|----- | ----- | ----- | ----- | -----|
| ceval-computer_network | ecf7cb | accuracy | gen | 5.26 |

## 8 Performance Evaluation

### Using vLLM Benchmark

Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

**Run performance evaluation of Baichuan-7B as an example:**

```bash
vllm bench serve \
    --base-url http://localhost:8031 \
    --model /path/to/baichuan-7B \
    --trust_remote_code \
    --served-model-name baichuan \
    --random-input-len 2048 \
    --random-output-len 2048 \
    --num-prompts 64 \
    --request-rate 1 \
    --max-concurrency 16 \
    --ignore_eos \
    --endpoint /v1/completions
```

After several minutes, you will get the performance evaluation result including metrics such as:
- Requests per second (RPS): 0.16
- Time per output token (TPOT): 47.71 ms
- Token throughput (tokens/s): 639.04
- Time to First Token (TTFT): 277.83 ms

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios.

#### Configuration Parameters

| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| `tensor_parallel_size` | 1 | Single card deployment for 7B model |
| `max_model_len` | 4096 | Maximum sequence length |
| `max_num_seqs` | 16 | Maximum concurrent sequences |
| `max_num_batched_tokens` | 65536 | Maximum batched tokens |
| `gpu_memory_utilization` | 0.9 | GPU memory utilization ratio |
| `dtype` | bfloat16 | Data type for model weights |
| `cudagraph_mode` | FULL_DECODE_ONLY | CUDA graph optimization mode |

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

#### 9.2.2 Memory Tuning

If you encounter OOM (Out of Memory) issues, consider the following adjustments:

1. Reduce `--max-model-len` to decrease memory usage for long sequences
2. Reduce `--max-num-seqs` to limit concurrent requests
3. Reduce `--gpu-memory-utilization` from 0.9 to 0.8 or lower

#### 9.2.3 Throughput Tuning

For better throughput performance:

1. Increase `--max-num-seqs` for higher concurrency
2. Increase `--max-num-batched-tokens` for larger batch processing
3. Enable prefix caching by removing `--no-enable-prefix-caching` if your workload has repeated prompts

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [vLLM-Ascend FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This section only covers issues specific to Baichuan-7B model.

### Q1: How to handle trust_remote_code warning?

A: Baichuan-7B requires `--trust-remote-code` flag to be set for custom model code. Always ensure you download models from trusted sources like ModelScope or Hugging Face.

### Q2: What is the minimum hardware requirement?

A: Baichuan-7B can run on a single Atlas A2 inference card with 64GB memory. For production use, 8-card configuration (64GB × 8) is recommended for better performance.
