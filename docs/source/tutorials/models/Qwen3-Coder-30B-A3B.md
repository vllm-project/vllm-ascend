# Qwen3-Coder-30B-A3B

## 1 Introduction

Qwen3-Coder-30B-A3B is a Mixture-of-Experts (MoE) model in the Qwen3 Coder series, sharing the same architecture as Qwen3-30B-A3B with 30.5B total parameters and 3.3B activated per token. Built upon the Qwen3 base architecture, it delivers significant optimizations in agentic coding, extended context support of up to 1M tokens, and versatile function calling capabilities.

This document will demonstrate the main validation steps for Qwen3-Coder-30B-A3B in the vLLM-Ascend environment, including supported features, environment preparation, single-node deployment, as well as accuracy and performance evaluation.

The Qwen3-Coder-30B-A3B model is first supported in **v0.10.0rc1**. This document is validated and written based on **vLLM-Ascend v0.22.1rc**. All **v0.22.1rc and later versions** can run stably. To use the latest features, it is recommended to use the latest release candidate or official version.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

The following model variants are available. It is recommended to download the model weight to a shared directory accessible to all nodes.

| Model                               | Hardware Requirement                                                                             | Download                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| Qwen3-Coder-30B-A3B-Instruct (BF16) | Atlas 800I A3 (64G, 1\~2 cards)<br>Atlas 800I A2 (64G, 2\~4 cards) | [Download](https://www.modelscope.cn/models/Qwen/Qwen3-Coder-30B-A3B-Instruct)          |
| Qwen3-Coder-30B-A3B-Instruct-W8A8   | Atlas 800I A3 (64G, 1\~2 cards)<br>Atlas 800I A2 (64G, 2\~4 cards)                               | [Download](https://www.modelscope.cn/models/Eco-Tech/Qwen3-Coder-30B-A3B-Instruct-w8a8) |
| Eagle3 Draft Model                  | NA                                                                                               | [Download](https://huggingface.co/AngelSlim/Qwen3-a3B_eagle3)                           |

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

If the W8A8 quantized weights are not available for direct download, you can obtain them by quantizing the BF16 model using **msmodelslim**. Refer to the [Quantization Guide](../../user_guide/feature_guide/quantization.md) for details. All model paths in this document should be adjusted to your actual local paths.

!!! note

    Qwen3-Coder-30B-A3B-W8A8 adopts a hybrid quantization strategy (ordered by model structure):

    - **Embedding layer**: BF16 (no quantization)
    - **Q/K normalization** (q_norm, k_norm): BF16 (weights and biases)
    - **Attention projections** (q/k/v/o_proj): Static W8A8 with pre-computed per-tensor scales; biases kept in BF16
    - **MoE routing gate** (mlp.gate): BF16
    - **MoE expert projections** (gate/up/down_proj): Dynamic W8A8 where input scales are computed on-the-fly during inference

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3 MoE models.

**Docker Pull:**

```bash
docker pull quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
```

**Docker Run:**

=== "Atlas 800I A3"

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3

    docker run \
        --name vllm-ascend-env \
        --shm-size=128g \
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
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /home:/home \
        -v /data:/data \
        -v /tmp:/tmp \
        -v /mnt:/mnt \
        -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
        -v /root:/host_root \
        -it -d $IMAGE bash
    ```

    !!! note

        A3 has 8 NPUs with dual-die design (16 chips total: `/dev/davinci[0-15]`).
        If you are on a shared machine, map only the chips you need (e.g., `/dev/davinci[0-7]` for NPU 0-3).

=== "Atlas 800I A2"

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run \
        --name vllm-ascend-env \
        --shm-size=128g \
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
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /home:/home \
        -v /data:/data \
        -v /tmp:/tmp \
        -v /mnt:/mnt \
        -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
        -v /root:/host_root \
        -it -d $IMAGE bash
    ```

The default workdir is `/workspace`. vLLM and vLLM-Ascend are installed as Python packages in site-packages.

**Installation Verification:**

After starting the container, run the following command to verify the installation:

```bash
docker ps | grep vllm-ascend-env
```

Expected result: The container is listed with status `Up`. You can also verify the vllm-ascend version inside the container:

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, matching the pulled image version.

### 4.2 Source Code Installation

If you prefer not to use the Docker image, you can build from source. Install vLLM from source first:

1. Clone and install vLLM:

   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   pip install -e .
   ```

2. Clone and install the vLLM-Ascend repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   pip install -e .
   ```

**Installation Verification:**

```bash
pip show vllm vllm-ascend
```

Expected result: The version information for both packages is displayed, confirming a successful installation.

!!! note

    If deploying a multi-node environment, set up the environment on each node.

For more details, please refer to the [Installation Guide](../../installation.md).

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and small-to-medium scale inference scenarios. For the Qwen3-Coder-30B-A3B MoE model, Expert Parallelism (EP) is required to distribute experts across NPUs.

> The following command is an example configuration. Adjust the parameters based on your actual scenario.

**Atlas 800I A2/A3:**

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE="AIV"  # not needed on A2
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve your_model_path \
    --served-model-name qwen3-coder \
    --trust-remote-code \
    --max-num-seqs 100 \
    --max-model-len 40960 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --quantization ascend \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_flashcomm1": true, "weight_nz_mode": 2}' \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --speculative-config '{"method": "eagle3", "model": "your_eagle3_model_path", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 3}'
```

!!! note

    - `ASCEND_RT_VISIBLE_DEVICES`: must be set to the NPU chip IDs allocated to your environment (e.g., `0,1,2,3` for 4 chips).
    - `--port`: adjust to avoid conflicts with other services running on the same machine.
    - `--no-enable-prefix-caching`: disabled by default as prefix caching effectiveness for this model on Ascend NPUs has not been fully characterized. You can try enabling it to evaluate the cache hit rate for your workload.
    - `--quantization ascend`: required for W8A8 quantized models. Remove this parameter when using BF16 weights.

!!! tip

    For parameter details, refer to:

    - [vLLM CLI documentation](https://docs.vllm.ai/en/stable/cli/) — standard serve parameters (`--host`, `--port`, `--max-model-len`, etc.)
    - [Environment Variables](../../user_guide/configuration/env_vars.md) — Ascend-specific environment variables (`HCCL_*`, etc.)
    - [Additional Configuration](../../user_guide/configuration/additional_config.md) — `--additional-config` format and options

**Service Verification:**

After the service is started, verify it is running by sending a prompt. Refer to [Section 6](#6-functional-verification) for a usage example.

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt.

**Chat Completions API:**

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-coder",
        "messages": [
            {"role": "user", "content": "Give me a short introduction to large language models."}
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_completion_tokens": 4096
    }'
```

!!! note

    Adjust the following fields based on your deployment:

    - **URL** (`http://localhost:8000`): Replace `localhost` and `8000` with your server IP and the `--port` value from the `vllm serve` command.
    - **`model`**: Must match the `--served-model-name` value from the `vllm serve` command (e.g., `qwen3-coder`).

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

## 7 Accuracy Evaluation

### Using AISBench

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

The following is an example configuration for the accuracy evaluation config file, demonstrated using the GSM8K dataset:

```python
# Example configuration: benchmarks/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="your_model_path",
        model="qwen3-coder",
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8000,
        max_out_len=32768,
        batch_size=32,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0.6,
            top_k=20,
            top_p=0.95,
        ),
    )
]
```

Run the accuracy evaluation using the `gsm8k` dataset as an example:

```shell
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_4_shot_cot_str --mode all --dump-eval-details --debug
```

The following table lists the `--datasets` parameter for each evaluation dataset:

| Dataset       | `--datasets` Parameter               |
| ------------- | ------------------------------------ |
| GSM8K         | `gsm8k_gen_4_shot_cot_str`           |
| GPQA-Diamond  | `gpqa_gen_0_shot_cot_chat_prompt`    |
| AIME 2024     | `aime2024_gen_0_shot_str`            |
| LiveCodeBench | `livecodebench_0_shot_chat_v4_v5_v6` |

> The `--models` parameter value corresponds to the configuration file name (e.g., `vllm_api_general_chat` for `vllm_api_general_chat.py`). Adjust `max_out_len`, `batch_size`, and dataset tasks based on your scenario.

For dataset preparation, please refer to the [AISBench Datasets Guide](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/get_started/datasets.md).

!!! note

    vLLM-Ascend also supports the following evaluation tools:

    - [lm_eval](../../developer_guide/evaluation/using_lm_eval.md)
    - [OpenCompass](../../developer_guide/evaluation/using_opencompass.md)
    - [EvalScope](../../developer_guide/evaluation/using_evalscope.md)

**Accuracy Results (Atlas 800I A3, vLLM-Ascend v0.22.1rc, W8A8):**

| Dataset       | Metric                | Score  |
| ------------- | --------------------- | ------ |
| GSM8K         | accuracy (4-shot CoT) | 90.14% |
| GPQA-Diamond  | accuracy (0-shot CoT) | 53.54% |
| LiveCodeBench | pass@1 (0-shot)       | 38.60% |
| AIME 2024     | accuracy (0-shot)     | 33.33% |

## 8 Performance Evaluation

### Using AISBench

For setup details, please refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation).

First, configure the model for streaming performance testing (`ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py`):

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-stream-chat',
        path="your_model_path",
        model="qwen3-coder",
        stream=True,
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8000,
        max_out_len=1500,
        batch_size=32,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=True,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

> Key differences from the accuracy config: `stream=True`, `ignore_eos=True` (ensures output reaches `max_out_len` for consistent TPOT measurement), and `batch_size` controls concurrency.

Then, configure the synthetic dataset distribution (`ais_bench/datasets/synthetic/synthetic_config.py`). Adjust the configuration based on your actual scenario. Note that random synthetic data is not suitable for benchmarking scenarios where prefix caching is enabled, as random inputs produce zero cache hit rate.

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 200,
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": 3500, "MaxValue": 3500}
        },
        "Output": {
            "Method": "uniform",
            "Params": {"MinValue": 1500, "MaxValue": 1500}
        }
    }
}
```

Then run the performance evaluation:

```shell
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf --debug
```

> The `--models` value should match the `abbr` in your model config file. Use `--num-prompts` to limit the number of test requests.

### Using vLLM Benchmark

Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

Take the `serve` subcommand as an example. The `--random-output-len` parameter controls the number of output tokens per request; adjust it based on your target scenario (e.g., 2048 for short outputs, 32768 for long outputs).

```shell
vllm bench serve \
    --model your_model_path \
    --served-model-name qwen3-coder \
    --port 8000 \
    --dataset-name random \
    --random-input 200 \
    --random-output-len 2048 \
    --num-prompts 200 \
    --request-rate 1 \
    --save-result \
    --result-dir ./
```

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

| Scenario        | Deployment Mode   | *Total NPUs      | Weight Version | Key Considerations                                               |
| --------------- | ----------------- | ---------------- | -------------- | ---------------------------------------------------------------- |
| High Throughput | Single-Node (TP1) | 1 (A3)<br>2 (A2) | W8A8           | Single-card deployment maximizes concurrent request processing   |
| Low Latency     | Single-Node (TP4) | 2 (A3)<br>4 (A2) | W8A8           | Multi-card TP reduces per-token latency with expert parallelism  |
| Long Context    | Single-Node (TP4) | 2 (A3)<br>4 (A2) | W8A8           | Reduces concurrent sequences to accommodate longer max-model-len |

> `*Total NPUs` indicates the total number of NPUs used across all nodes. On Atlas 800I A3, each NPU contains two dies (chips), so TP4 requires 4 chips = 2 NPUs.

#### Table 2: Detailed Node Configuration

| Scenario        | NPUs   | TP  | max-model-len | max-num-seqs | FUSED_MC2 | EP  | hf-overrides |
| --------------- | ------ | --- | ------------- | ------------ | --------- | --- | ------------ |
| High Throughput | 1 (A3) | 1   | 37364         | 100          | Off       | Off | -            |
| Low Latency     | 2 (A3) | 4   | 37364         | 100          | Off       | On  | -            |
| Long Context    | 2 (A3) | 4   | 131072        | 14           | Off       | On  | -            |

> For detailed parameter descriptions, please refer to the deployment examples in Section 5.

**Low Latency Configuration:**

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve your_model_path \
    --served-model-name qwen3-coder \
    --trust-remote-code \
    --max-num-seqs 100 \
    --max-model-len 37364 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_flashcomm1": true, "weight_nz_mode": 2}' \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --speculative-config '{"method": "eagle3","model": "your_eagle3_model_path", "num_speculative_tokens": 3}'
```

!!! tip

    Example AISBench settings for this configuration:

    - `request_rate`: 0
    - `batch_size`: 32
    - Input/Output length: 2048/2048 or 3500/1500

**High Throughput Configuration:**

```shell
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve your_model_path \
    --served-model-name qwen3-coder \
    --trust-remote-code \
    --max-num-seqs 100 \
    --max-model-len 37364 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 1 \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"weight_nz_mode": 2}' \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --speculative-config '{"method": "eagle3","model": "your_eagle3_model_path", "num_speculative_tokens": 3}'
```

!!! tip

    Example AISBench settings for this configuration:

    - `request_rate`: 0
    - `batch_size`: 32
    - Input/Output length: 2048/2048 or 3500/1500

**Long Context Configuration:**

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve your_model_path \
    --served-model-name qwen3-coder \
    --trust-remote-code \
    --max-num-seqs 14 \
    --max-model-len 131072 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_flashcomm1": true, "weight_nz_mode": 2}' \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --speculative-config '{"method": "eagle3","model": "your_eagle3_model_path", "num_speculative_tokens": 3}'
```

!!! tip

    Example AISBench settings for this configuration:

    - `request_rate`: 0
    - `batch_size`: 32
    - Input/Output length: 65536/1024 or 131072/1024

### 9.2 Tuning Guidelines

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This chapter only covers model-specific issues.

### Q: How do I enable long context (beyond 256K)?

Qwen3-Coder-30B-A3B natively supports 256K token context length. For contexts beyond 256K, YaRN rope scaling is required to extend up to 1M. Enable YaRN via `--hf-overrides`:

```bash
--hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":4,"original_max_position_embeddings":262144}}'
```

For contexts within the native 256K range, no additional configuration is needed. Just set `--max-model-len` to your desired length.

### Q: What makes Qwen3-Coder different from Qwen3-30B-A3B?

Qwen3-Coder-30B-A3B shares the same MoE architecture (30.5B/3.3B) as the base Qwen3-30B-A3B but is specifically fine-tuned for coding tasks, with optimizations for agentic coding, function calling, and extended context support up to 1M tokens.
