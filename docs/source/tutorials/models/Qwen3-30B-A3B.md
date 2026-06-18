# Qwen3-30B-A3B

## 1 Introduction

Qwen3-30B-A3B is a Mixture-of-Experts (MoE) model in the Qwen3 series, featuring 30.5B total parameters with 3.3B activated per token. The sparse MoE architecture enables efficient training and inference, delivering strong performance across reasoning, instruction-following, and agent capabilities while maintaining lower computational cost compared to dense models of similar capability.

This document will demonstrate the main validation steps for Qwen3-30B-A3B in the vLLM-Ascend environment, including supported features, environment preparation, single-node deployment, as well as accuracy and performance evaluation.

The Qwen3-30B-A3B model is first supported in v0.8.4rc2. This document is validated and written based on **vLLM-Ascend v0.21.0**. All **v0.21.0 and later versions** can run stably. To use the latest features, it is recommended to use v0.21.0 or a later version.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

The following model variants are available. It is recommended to download the model weight to a shared directory accessible to all nodes.

| Model | Hardware Requirement | Download |
|-------|---------------------|----------|
| Qwen3-30B-A3B (BF16) | Atlas 800I A3 (64G, 1~2 cards)<br>Atlas 800I A2 (64G, 2~4 cards) | [Download](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B) |
| Qwen3-30B-A3B-W8A8 | Atlas 800I A3 (64G, 1~2 cards)<br>Atlas 800I A2 (64G, 2~4 cards) | [Download](https://www.modelscope.cn/models/Eco-Tech/Qwen3-30B-A3B-w8a8) |

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

If the W8A8 quantized weights are not available for direct download, you can obtain them by quantizing the BF16 model using **msmodelslim**. Refer to the [Quantization Guide](../../user_guide/feature_guide/quantization.md) for details. All model paths in this document should be adjusted to your actual local paths.

:::{note}
Qwen3-30B-A3B-W8A8 adopts a hybrid quantization strategy (ordered by model structure):

- **Embedding layer**: BF16 (no quantization)
- **Q/K normalization** (q_norm, k_norm): BF16
- **Attention projections** (q/k/v/o_proj): Static W8A8 with pre-computed per-tensor scales
- **MoE routing gate** (mlp.gate): BF16
- **MoE expert projections** (gate/up/down_proj): Dynamic W8A8 where input scales are computed on-the-fly during inference
:::

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3 MoE models.

**Docker Pull:**

```bash
docker pull quay.io/ascend/vllm-ascend:|vllm_ascend_version|
```

**Docker Run:**

```{code-block} bash
   :substitutions:

# Update --device according to your device (Atlas A2: /dev/davinci[0-7], Atlas A3: /dev/davinci[0-15]).
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

docker run \
    --name vllm-ascend-env \
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
    -it -d $IMAGE bash
```

**Docker Run (Atlas 800I A5):**

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

docker run --runtime=runc -u root -it -d --name vllm-ascend-env \
    --net=host --privileged=true --shm-size=2g \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc \
    --device=/dev/ummu --device=/dev/uburma \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/log/npu/:/usr/slog \
    -v /root/host:/root/host \
    -v /mnt:/mnt \
    -v /data:/data \
    -v /home/:/home/ \
    -v /etc/hixlep:/etc/hixlep \
    $IMAGE bash
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

If you prefer not to use the Docker image, you can build from source:

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
pip show vllm-ascend
```

Expected result: The version information is displayed, confirming a successful installation.

:::{note}
If deploying a multi-node environment, set up the environment on each node.
:::

For more details, please refer to the [Installation Guide](../../installation.md).

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and small-to-medium scale inference scenarios. For the Qwen3-30B-A3B MoE model, Expert Parallelism (EP) is required to distribute experts across NPUs.

> The following command is an example configuration. Adjust the parameters based on your actual scenario.

**Atlas 800I A2/A3:**

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_MODELSCOPE=True
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve your_model_path \
    --served-model-name qwen3 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 32768 \
    --quantization ascend \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.95 \
    --port 8000
```

**Atlas 800I A5:**

```bash
export ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600
export HCCL_ALGO=level0:fullmesh
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=1
export TRITON_DISABLE_FFTS=1

vllm serve your_model_path \
    --host 0.0.0.0 \
    --served-model-name qwen3 \
    --trust-remote-code \
    --max-num-seqs 200 \
    --max-model-len 40960 \
    --max-num-batched-tokens 40960 \
    --tensor-parallel-size 1 \
    --port 8000 \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --gpu-memory-utilization 0.9 \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config": {"fuse_qknorm_rope": false}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[3, 9, 18, 27, 54, 108]}' \
    --speculative-config '{"method": "eagle3", "model": "your_eagle3_model_path", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 2}'
```

:::{tip}
For parameter details, refer to:

- [vLLM CLI documentation](https://docs.vllm.ai/en/stable/cli/) — standard serve parameters (`--host`, `--port`, `--max-model-len`, etc.)
- [Environment Variables](../../user_guide/configuration/env_vars.md) — Ascend-specific environment variables (`VLLM_ASCEND_ENABLE_NZ`, `HCCL_*`, etc.)
- [Additional Configuration](../../user_guide/configuration/additional_config.md) — `--additional-config` format and options
:::

**Service Verification:**

After the service is started, verify it is running by sending a prompt. Refer to [Section 6](#functional-verification) for a usage example.

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt.

**Chat Completions API:**

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3",
        "messages": [
            {"role": "user", "content": "Give me a short introduction to large language models."}
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_completion_tokens": 4096
    }'
```

:::{note}
Adjust the following fields based on your deployment:

- **URL** (`http://localhost:8000`): Replace `localhost` and `8000` with your server IP and the `--port` value from the `vllm serve` command.
- **`model`**: Must match the `--served-model-name` value from the `vllm serve` command (e.g., `qwen3`).
:::

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

## 7 Accuracy Evaluation

### Using AISBench

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

**Install from source:**

```bash
git clone https://github.com/AISBench/benchmark.git
cd benchmark
pip install -e .
```

The following is an example configuration for the accuracy evaluation config file:

```python
# Example configuration: benchmarks/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="your_model_path",
        model="qwen3",
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8000,
        max_out_len=8192,
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

> The `--models` parameter value corresponds to the `abbr` field in the configuration file above. Adjust `max_out_len`, `batch_size`, and dataset tasks based on your scenario. Since Qwen3 is a thinking model, `max_out_len` should be at least 8192 to avoid truncation during reasoning.

For dataset preparation, please refer to the [AISBench Datasets Guide](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/get_started/datasets.md).

:::{note}
vLLM-Ascend also supports the following evaluation tools:

- [lm_eval](../../developer_guide/evaluation/using_lm_eval.md)
- [OpenCompass](../../developer_guide/evaluation/using_opencompass.md)
- [EvalScope](../../developer_guide/evaluation/using_evalscope.md)
:::

**Accuracy Results (Atlas 800I A3, vLLM-Ascend v0.21.0, W8A8):**

| Dataset | Metric | Score |
|---------|--------|-------|
| GSM8K | accuracy (4-shot CoT) | 92.87% |
| GPQA-Diamond | accuracy (0-shot CoT) | TBD |
| LiveCodeBench | pass@1 (0-shot) | TBD |
| AIME 2024 | accuracy (0-shot) | TBD |

## 8 Performance

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
        model="qwen3",
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
    --served-model-name qwen3 \
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

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
|----------|----------------|-------------|----------------|------------------------|
| High Throughput | Single-Node (TP1) | 1 (A3)<br>2 (A2) | W8A8 | Single-card deployment maximizes concurrent request processing |
| Low Latency | Single-Node (TP4) | 2 (A3)<br>4 (A2) | W8A8 | Multi-card TP reduces per-token latency with expert parallelism |
| Long Context | Single-Node (TP4) | 2 (A3)<br>4 (A2) | W8A8 | Reduces concurrent sequences to accommodate longer max-model-len |

> `*Total NPUs` indicates the total number of NPUs used across all nodes.

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | NPUs | TP | DP | FUSED_MC2 | EP Switch | Async Scheduling |
|----------|---------------|-------|----|----|-----------|-----------|------------------|
| High Throughput | Single-Node | 1 | 1 | 1 | Off | Off | On |
| Low Latency | Single-Node | 2 | 4 | 1 | Off | On | On |
| Long Context | Single-Node | 2 | 4 | 1 | Off | On | On |

> For detailed parameter descriptions, please refer to the deployment examples in Section 5.

**Low Latency Configuration:**

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve your_model_path \
    --served-model-name qwen3 \
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
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --speculative-config '{"method": "eagle3","model": "your_eagle3_model_path", "num_speculative_tokens": 3}'
```

:::{tip}
Example AISBench settings for this configuration:

- `request_rate`: 0
- `batch_size`: 32
- Input/Output length: 2048/2048 or 3500/1500
:::

**High Throughput Configuration:**

```shell
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=2

vllm serve your_model_path \
    --served-model-name qwen3 \
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
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --speculative-config '{"method": "eagle3","model": "your_eagle3_model_path", "num_speculative_tokens": 3}'
```

:::{tip}
Example AISBench settings for this configuration:

- `request_rate`: 0
- `batch_size`: 32
- Input/Output length: 2048/2048 or 3500/1500
:::

**Long Context Configuration:**

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve your_model_path \
    --served-model-name qwen3 \
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
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --speculative-config '{"method": "eagle3","model": "your_eagle3_model_path", "num_speculative_tokens": 3}' \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":4,"original_max_position_embeddings":32768}}'
```

:::{tip}
Example AISBench settings for this configuration:

- `request_rate`: 0
- `batch_size`: 32
- Input/Output length: 65536/1024 or 131072/1024
:::

### 9.2 Tuning Guidelines

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This chapter only covers model-specific issues.
