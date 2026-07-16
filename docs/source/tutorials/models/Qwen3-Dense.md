# Qwen3-Dense (Qwen3-0.6B/1.7B/4B/8B/14B/32B, W8A8, W4A8, W4A4)

## 1 Introduction

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support. The Dense variants covered in this document include Qwen3-0.6B, 1.7B, 4B, 8B, 14B, and 32B, along with their quantized versions (W8A8, W4A8, W4A4) optimized for Ascend NPU deployment.

This document will demonstrate the main validation steps for Qwen3 Dense models in the vLLM-Ascend environment, including supported features, environment preparation, model quantization, single-node and multi-node deployment, as well as accuracy and performance evaluation. By tailoring service-level configurations to fit different use cases, you can ensure optimal performance across various scenarios.

The Qwen3 Dense models are first supported in v0.8.4rc2. W8A8 quantization was first supported in v0.8.4rc2, W4A8 quantization is supported since v0.9.1rc2, and W4A4 is supported since v0.11.0rc1. This document is validated and written based on **vLLM-Ascend v0.21.0**. All **v0.21.0 and later versions** can run stably. To use the latest features, it is recommended to use the latest release candidate or official version.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

The following model variants are available. It is recommended to download the model weight to a shared directory accessible to all nodes.

**BF16 Versions:**

| Model | Hardware Requirement | Download |
|-------|---------------------|----------|
| Qwen3-0.6B | 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8) | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-0.6B) |
| Qwen3-1.7B | 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8) | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-1.7B) |
| Qwen3-4B | 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8) | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-4B) |
| Qwen3-8B | 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8) | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-8B) |
| Qwen3-14B | 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8) | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-14B) |
| Qwen3-32B | 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8) | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-32B) |

**Quantized Versions:**

| Model | Quantization | Hardware Requirement | Download |
|-------|-------------|---------------------|----------|
| Qwen3-8B-W4A8 | W4A8 | 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8) | [Download](https://www.modelscope.cn/models/vllm-ascend/Qwen3-8B-W4A8) |
| Qwen3-32B-W4A4 | W4A4 | 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8) | [Download](https://www.modelscope.cn/models/vllm-ascend/Qwen3-32B-W4A4) |
| Qwen3-32B-W8A8 | W8A8 | 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8) | [Download](https://www.modelscope.cn/models/vllm-ascend/Qwen3-32B-W8A8) |
| Qwen3-32B-W8A8 (LLM-Compressor) | W8A8 compressed-tensors | 4 Atlas 300I (310P) devices | [Download](https://huggingface.co/ramblingpolymath/Qwen3-32B-W8A8) |

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

### 3.2 Verify Multi-node Communication

If you need to deploy a multi-node environment, verify the multi-node communication according to [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3 Dense models.

**Docker Pull:**

```bash

docker pull quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
```

**Docker Run:**

Start the docker image on your each node.

=== "A3 series"

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3

    docker run --rm \
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

    !!! note

        A3 has 8 NPUs with dual-die design (16 chips total: `/dev/davinci[0-15]`).
        If you are on a shared machine, map only the chips you need (e.g., `/dev/davinci[0-7]` for NPU 0-3).

=== "A2 series"

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run --rm \
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
        -it $IMAGE bash
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

Atlas 800I A2/A3:

Qwen3-32B-W8A8:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

vllm serve your_model_path \
    --served-model-name qwen3 \
    --trust-remote-code \
    --async-scheduling \
    --quantization ascend \
    --distributed-executor-backend mp \
    --tensor-parallel-size 4 \
    --max-model-len 5500 \
    --max-num-batched-tokens 40960 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --port <port> \
    --gpu-memory-utilization 0.9 \
    --additional-config '{"enable_flashcomm1": true}'
```

For LLM-Compressor `compressed-tensors` checkpoints such as
`ramblingpolymath/Qwen3-32B-W8A8` on Atlas 300I (310P), the quantization
method is read from the model config. Do not add `--quantization ascend`.
Use `float16` on 310P. The validated serving path uses contiguous ND weights
with `npu_quant_matmul_dequant` and `FULL_DECODE_ONLY` ACLGraph with one fixed
capture size. Keep eager mode for accuracy baselines or graph-isolation
debugging. NZ matmul is a performance exploration path and should be validated
separately on the target CANN/runtime stack before production use.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

vllm serve ramblingpolymath/Qwen3-32B-W8A8 \
    --served-model-name qwen3-w8a8 \
    --trust-remote-code \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --max-model-len 40960 \
    --max-num-batched-tokens 1536 \
    --max-num-seqs 8 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8]}' \
    --port 8000
```

For smaller smoke or accuracy runs, reduce `--max-model-len` and
`--max-num-seqs`. Add `--enforce-eager` when reproducing the eager accuracy
baseline or debugging graph issues. Multi-size graph capture such as
`[1,2,4,8]` can hit `507903 Insufficient_Event_Resources` on the validated
CANN/runtime stack; use one fixed capture size such as `[8]` for ACLGraph.

```bash
vllm serve ramblingpolymath/Qwen3-32B-W8A8 \
    --served-model-name qwen3-w8a8 \
    --trust-remote-code \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --max-num-batched-tokens 1536 \
    --max-num-seqs 1 \
    --enforce-eager \
    --port 8000
```

Atlas 800I A2/A3：

Qwen3-32B-W4A4:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=1024
vllm serve your_model_path \
    --port 8004 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --served-model-name qwen3 \
    --distributed_executor_backend "mp" \
    --max-model-len 40960 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --compilation-config '{"cudagraph_capture_sizes": [64]}' \
    --additional-config '{"enable_flashcomm1": true, "ascend_compilation_config": {"fuse_norm_quant": false}}'
```

Atlas 800I A2/A3：
Qwen3-8B-W4A8:  

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
vllm serve your_model_path \
    --served-model-name qwen3 \
    --max-model-len 4096 \
    --port 20001 \
    --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}' \
    --quantization ascend
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

**Chat Completions API:**

```bash
curl http://localhost:<port>/v1/chat/completions \
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

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

## 7 Accuracy Evaluation

### 7.1 LLM-Compressor Qwen3-32B-W8A8 on 310P

The LLM-Compressor `ramblingpolymath/Qwen3-32B-W8A8` checkpoint was validated
with real weights on four Atlas 300I (310P) devices, TP4, and `dtype=float16`.
Scores use the corrected scoring pass: final-answer extraction misses caused by
answer formatting are verified and counted when the answer is correct.

| Runtime path | Mode | Dataset | Metric | Corrected score | Reference |
|--------------|------|---------|--------|-----------------|-----------|
| Eager | Non-thinking | GPQA-Diamond full, 0-shot | mean accuracy | 56.06% | Qwen3-32B non-thinking official: BF16 54.6%, AWQ-INT4 53.1% |
| Eager | Thinking | GPQA-Diamond full, 0-shot | mean accuracy | 67.68% | Qwen3-32B thinking official: BF16 68.4%, AWQ-INT4 69.0% |
| `FULL_DECODE_ONLY` ACLGraph, capture size `[8]` | Non-thinking | GPQA-Diamond full, 0-shot | mean accuracy | 52.53% | Qwen3-32B non-thinking official: BF16 54.6%, AWQ-INT4 53.1% |
| `FULL_DECODE_ONLY` ACLGraph, capture size `[8]` | Thinking | GPQA-Diamond full, 0-shot | mean accuracy | 66.67% | Qwen3-32B thinking official: BF16 68.4%, AWQ-INT4 69.0% |

The official LLM-Compressor W8A8 score is not published, so the BF16 and
AWQ-INT4 Qwen3-32B numbers are used as the alignment references.

### 7.2 Using AISBench

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

The following is an example configuration for the accuracy evaluation config file:

**Accuracy Evaluation Config File:**

```bash
# Example configuration: benchmarks/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="your_model_path",
        model="qwen3",
        request_rate = 0,
        retry = 2,
        host_ip = "127.0.0.1",
        host_port = 2001,
        max_out_len = 32768,
        batch_size = 32,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0.6,
            top_k = 20,
            top_p = 0.95,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

**Run the accuracy evaluation using the aime2025 dataset as an example:**

```bash
ais_bench --models vllm_api_general_chat --datasets aime2025_gen_0_shot_chat_prompt --debug
```

> The --models parameter value corresponds to the abbr field in the configuration file above. Adjust max_out_len, batch_size, and dataset tasks based on your scenario.

## 8 Performance Evaluation

### Using AISBench

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

The following is an example configuration for the accuracy evaluation config file:

```bash
# Example configuration: benchmarks/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat",
        path="your_model_path",
        model="qwen3",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        host_ip="127.0.0.1",
        host_port=8004,
        max_out_len=1500,
        batch_size=90,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos = True
        ),
    )
]
```

**Run the performance evaluation using the GSM8K dataset as an example:**

```bash
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf --num-prompts 360
```

### Using vLLM Benchmark

Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take `serve` as an example:

```shell
vllm bench serve \
    --model your_model_path \
    --served-model-name qwen3 \
    --port <port> \
    --dataset-name random \
    --random-input 200 \
    --num-prompts 200 \
    --request-rate 1 \
    --save-result \
    --result-dir ./
```

After several minutes, you will get the performance evaluation result.

## 9 Performance Tuning

For the LLM-Compressor Qwen3-32B-W8A8 checkpoint on 310P, the current validated
runtime path uses `npu_quant_matmul_dequant` with `FULL_DECODE_ONLY` ACLGraph
and one fixed capture size. Eager mode remains the baseline and graph fallback.
NZ matmul and PIECEWISE ACLGraph showed useful performance potential in short
probes, but require separate long-running stability validation before they
should be used as the default path. With TP4, `--max-model-len 40960`,
and `--max-num-seqs 8`, the full 40,960-token model context was validated.

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
|----------|----------------|-------------|----------------|---------------------|
| High Throughput | Single-Node (TP4) | 4 (A3) | W8A8 | 4-card TP maximizes concurrent request processing |
| Long Context | Single-Node (TP4) | 4 (A3) | W8A8 | 4-card TP extends context window for long sequences |
| Low Latency | Single-Node (TP8) | 8 (A3) | W8A8 | 8-card TP reduces per-token latency for interactive responses |

> `*Total NPUs` indicates the total number of NPUs used across all nodes.

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | #NPUs | TP | DP | FUSED_MC2 | EP Switch | Async Scheduling |
|----------|---------------|-------|----|----|-------------|--------------|--------------|
| High Throughput | Single-Node | 4 | 4 | 1 | Off | Off | On |
| Long Context | Single-Node | 4 | 4 | 1 | Off | Off | On |
| Low Latency | Single-Node | 8 | 8 | 1 | Off | Off | On |

For detailed parameter descriptions, please refer to the deployment examples in [Section 5](#5-online-service-deployment)

<u>High Throughput Configuration:</u>

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

vllm serve your_model_path \
  --served-model-name qwen3 \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --tensor-parallel-size 4 \
  --max-model-len 5500 \
  --max-num-batched-tokens 40960 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,64,72,76,80,96,100,120,140,144,160,192,216,240,252,288,320,336,360,384,400,408,416,420,432,480,540,576,600]}' \
  --additional-config '{"weight_prefetch_config":{"enabled":true}, "enable_flashcomm1": true}' \
  --host <host_ip> \
  --port <port> \
  --block-size 128 \
  --gpu-memory-utilization 0.9
```

<u>Long Context Configuration:</u>

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

vllm serve your_model_path \
  --host <host_ip> \
  --port <port> \
  --served-model-name qwen3 \
  --trust-remote-code \
  --seed 1024 \
  --max-model-len 135000 \
  --max-num-batched-tokens 40960 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend "mp" \
  --async-scheduling \
  --no-enable-prefix-caching \
  --speculative-config '{"method": "eagle3", "model":"your_eagle3_model_path", "num_speculative_tokens": 3}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":131072}}' \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --additional-config '{"enable_flashcomm1": true}'
```

<u>Low Latency Configuration:</u>

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

vllm serve your_model_path \
  --served-model-name qwen3 \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --tensor-parallel-size 8 \
  --max-model-len 5500 \
  --max-num-batched-tokens 40960 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,72,76,80,96,100,120,140,144,160,192,216,240,252,288,320,336,360,384,400,408,416,420,432,480,540,576,600]}' \
  --speculative-config '{"method": "eagle3", "model":"your_eagle3_model_path", "enforce_eager": true, "num_speculative_tokens": 3}' \
  --port <port> \
  --block-size 128 \
  --gpu-memory-utilization 0.9
```

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [vLLM-Ascend FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This section only covers issues specific to Qwen3 Dense models.
