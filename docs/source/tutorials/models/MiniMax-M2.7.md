# MiniMax-M2.7

## 1 Introduction

MiniMax-M2.7 is MiniMax's flagship large language model, reinforced for high-value scenarios such as code generation, agentic tool calling/search, and complex office workflows, with an emphasis on reasoning efficiency and end-to-end speed on challenging tasks. It supports both M2.5 and M2.7 model versions.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

This document is validated and written based on **vLLM-Ascend v0.13.0**. The current model (MiniMax-M2.7/M2.5) is fully supported in this version, and all **v0.13.0 and later versions** can run stably. To use the latest features (e.g., PD separation, EAGLE3 speculative decoding), it is recommended to use v0.13.0 or a later version.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

The following model weights and EAGLE3 weights are available on ModelScope. Search for the corresponding model name on [ModelScope](https://modelscope.cn) to obtain the latest weight files.

| Model | Description | Recommended Hardware | Source |
|-------|-------------|---------------------|--------|
| `MiniMax-M2.7` / `MiniMax-M2.5` | FP8 checkpoint | 1× Atlas 800 A3 (64G × 16) or 1× Atlas 800I A2 (64G × 8) | [ModelScope](https://modelscope.cn) |
| `MiniMax-M2.7-w8a8-QuaRot` / `MiniMax-M2.5-w8a8-QuaRot` | W8A8 quantized version | 1× Atlas 800 A3 (64G × 16) or 1× Atlas 800I A2 (64G × 8) | [ModelScope](https://modelscope.cn) |
| `Eagle3` (MiniMax-M2.5/M2.7) | Speculative decoding head model | Matches the base model node count | [ModelScope](https://modelscope.cn) |

It is recommended to download the model weights to a shared directory, such as `/mnt/sfs_turbo/.cache/`. The current release automatically detects the MiniMax-M2 FP8 checkpoint, disables FP8 quantization kernels on NPU, and loads the weights by dequantizing to BF16. This behavior may be removed once public BF16 weights are available.

### 3.2 Verify Multi-node Communication (Optional)

If you need to deploy a multi-node environment, verify the multi-node communication according to [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the container on your node. For the available image tags and published versions, refer to [Using Docker](../../installation.md#set-up-using-docker).

**A3 series**

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

# Run the container using the defined variables
# Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance
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
-v /mnt/sfs_turbo/.cache:/home/cache \
-it $IMAGE bash
```

**A2 series**

Create and run `minimax-docker-run.sh`.

Notes:

- The default configuration assumes an **Atlas 800I A2 8-NPU** node and sets `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. Update it based on your hardware.
- Map your model weight directory into the container (the example maps it to `/opt/data/verification/`).

```{code-block} bash
#!/bin/sh
NAME=minimax
DEVICES="0,1,2,3,4,5,6,7"
IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|

docker run -itd -u 0 --ipc=host --privileged \
  -e VLLM_USE_MODELSCOPE=True \
  -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
  -e ASCEND_RT_VISIBLE_DEVICES=$DEVICES \
  --name $NAME \
  --net=host \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --shm-size=1200g \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home/:/home/ \
  -v /opt/data/verification/:/opt/data/verification/ \   # Map the model weights here
  -v /root/.cache:/root/.cache \
  -v /mnt/performance/:/mnt/performance/ \
  -it $IMAGE bash

# Start and enter the container
# bash minimax-docker-run.sh
# docker exec -it minimax bash
```

**Verification:**

After starting the container, verify the installation with:

```bash
# Check that the container is running
docker ps | grep $NAME

# Verify that NPU devices are visible inside the container
docker exec $NAME npu-smi info
```

Expected result: `docker ps` shows the container with status "Up", and `npu-smi info` lists the expected number of NPU devices.

### 4.2 Source Code Installation

If you prefer to build from source instead of using the Docker image, install vLLM-Ascend following the [Installation Guide](../../installation.md).

To verify the source installation:

```bash
python -c "import vllm_ascend; print(vllm_ascend.__version__)"
```

## 5 Online Service Deployment

:::{note}
In this tutorial, we assume you have downloaded the model weights. Replace `/path/to/weight/` with your actual model weight path.
:::

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and low-to-medium throughput production scenarios.

**Common Issues Tip:** If you encounter OOM, HCCL port conflicts, or other startup issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting. For MiniMax-specific issues, refer to [Chapter 10 FAQ](#10-faq).

#### A3 (single node)

Below is a recommended startup configuration for short-context conditions (e.g., 3.5k input / 1.5k output) to achieve good performance.

Notes:

- If you only care about short-context low latency, you can set `--max-model-len 32768`, `--tensor-parallel-size 16`, and `--data-parallel-size 1`.
- `export VLLM_ASCEND_BALANCE_SCHEDULING=1` enhances scheduling capacity between prefill and decode. This works best with a larger `--data-parallel-size` and can increase performance when concurrency approaches `data-parallel-size × max-num-seqs`.

```{code-block} bash
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1

vllm serve /path/to/weight/MiniMax-M2.7-w8a8-QuaRot \
    --served-model-name "MiniMax-M2.7" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}' \
    --enable-expert-parallel \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --max-num-seqs 48 \
    --max-model-len 40690 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 \
    --speculative_config '{"enforce_eager": true, "method": "eagle3", "model": "/path/to/weight/Eagle3/", "num_speculative_tokens": 3}'
```

Remarks:

- `minimax_m2_append_think` keeps `<think>...</think>` inside `content`.
- If you mainly rely on the reasoning semantics of `/v1/responses`, it is recommended to use `--reasoning-parser minimax_m2` instead.
- To achieve better performance on long-context scenarios (e.g., 128k or 64k), we recommend the following adjustments, and you can remove `export VLLM_ASCEND_BALANCE_SCHEDULING=1`:

```{code-block} bash
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --decode-context-parallel-size 1 \
    --prefill-context-parallel-size 2 \
    --cp-kv-cache-interleave-size 128 \
    --max-num-seqs 16 \
    --max-model-len 138000 \
    --max-num-batched-tokens 65536 \
    --gpu-memory-utilization 0.85 \
    --speculative_config '{"enforce_eager": true, "method": "eagle3", "model": "/path/to/weight/Eagle3/", "num_speculative_tokens": 1}'
```

- If you need to test with `curl` and tool calling, add the following to the startup command:

```{code-block} bash
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
```

#### A2 (single node)

```{code-block} bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=512
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1

vllm serve /path/to/weight/MiniMax-M2.7-w8a8-QuaRot \
    --served-model-name MiniMax-M2.7 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --quantization ascend \
    --enable-expert-parallel \
    --max-num-seqs 32 \
    --seed 1024 \
    --max-num-batched-tokens 32768 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,16,20,32,80,96,128,200,256,320]}' \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-force-include-usage \
    --additional-config '{"enable_cpu_binding":true}' \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
    --speculative_config '{"method": "eagle3", "model": "/path/to/weight/Eagle3/",  "num_speculative_tokens":3}'
```

Remarks:

- `--max-num-seqs` parameter can be adjusted according to actual request conditions.
- `--max-num-batched-tokens 32768` is applicable to input sequence lengths of 32k or longer.
- `--max-num-batched-tokens 16384` is applicable to input sequence lengths of 16k.
- `--max-num-batched-tokens 6144` is applicable to short sequence input scenarios such as 2k and 3.5k.

#### Key Parameter Reference

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| `--tensor-parallel-size` | Number of NPUs for tensor parallelism | 4–16 (A3) / 8 (A2) |
| `--data-parallel-size` | Number of data parallel replicas | 1–4 |
| `--max-num-seqs` | Maximum concurrent sequences | 16–48 |
| `--max-model-len` | Maximum model context length | 32768–138000 |
| `--max-num-batched-tokens` | Maximum tokens per batch | 6144–65536 |
| `--gpu-memory-utilization` | Fraction of NPU memory for KV cache | 0.85–0.92 |
| `--speculative_config` | EAGLE3 speculative decoding settings | `"num_speculative_tokens": 1–3` |

#### Environment Variable Reference

| Variable | Description | Common Value |
|----------|-------------|-------------|
| `HCCL_OP_EXPANSION_MODE` | HCCL operation expansion mode | `"AIV"` |
| `HCCL_BUFFSIZE` | HCCL communication buffer size (MB) | `512` (A2) / `1024` (A3) |
| `PYTORCH_NPU_ALLOC_CONF` | PyTorch NPU memory allocator config | `expandable_segments:True` |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | Enable fused MC2 kernel | `1` |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | Enable FlashComm v1 communication optimization | `1` |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | Enable balanced prefill/decode scheduling | `1` (short context) |
| `TASK_QUEUE_ENABLE` | Enable task queue for improved scheduling | `1` |
| `LD_PRELOAD` | Preload jemalloc for memory optimization | `/usr/lib/.../libjemalloc.so.2` |
| `HCCL_INTRA_PCIE_ENABLE` | Enable intra-node PCIe communication (A2) | `1` |
| `HCCL_INTRA_ROCE_ENABLE` | Enable intra-node RoCE communication (A2) | `0` |

### 5.2 Multi-Node PD Separation Deployment

PD (Prefill-Decode) separation splits the Prefill and Decode phases across different nodes, improving throughput and resource utilization for high-concurrency scenarios.

:::{note}
PD separation deployment requires careful tuning of DP/TP ratios between Prefill and Decode nodes. For detailed configuration guidance, refer to [Distributed DP Server With Large-Scale Expert Parallelism](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/large_scale_ep.html).
:::

**Common Issues Tip:** For PD separation specific issues such as KV transfer timeouts or Mooncake connection errors, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). For MiniMax-specific PD separation issues, refer to [Chapter 10 FAQ](#10-faq).

> **TODO (Model Owner):** Please fill in the specific PD separation startup commands, launch scripts, and KV transfer configuration for MiniMax-M2.7.

## 6 Functional Verification

Once your server is started, you can query the model with input prompts.

**Note:**

- `<node_ip>`: The IP address of the node where the server is running (e.g., localhost for single-node).
- `<port>`: The port number specified in the server startup command (e.g., `8000`).

### Using curl

```bash
curl http://<node_ip>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.7",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "stream": false,
    "temperature": 0.8,
    "max_tokens": 200
  }'
```

Expected result: HTTP 200 with a JSON response containing a `choices` field with the model's reply text.

### Using OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="na")

resp = client.chat.completions.create(
    model="MiniMax-M2.7",
    messages=[{"role": "user", "content": "你好，请介绍一下你自己，并展示一次工具调用的参数格式。"}],
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

Expected result: The response should contain a coherent self-introduction and tool call parameter format in the `content` field.

### Tool Calling Verification

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.7",
    "messages": [{"role": "user", "content": "请查询上海的天气。"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get weather by city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 512
  }'
```

Expected result: HTTP 200 with a JSON response containing a `tool_calls` field with the function name and arguments.

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

For details, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

### 7.2 Using Language Model Evaluation Harness

Using the `gsm8k` dataset as an example test dataset, run the accuracy evaluation for `MiniMax-M2.7-W8A8` in online mode.

1. For `lm_eval` installation, please refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).
2. Run `lm_eval` to execute the accuracy evaluation:

```shell
lm_eval \
  --model local-completions \
  --model_args model=/path/to/weight/MiniMax-M2.7-w8a8-QuaRot,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 Performance

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Run performance evaluation for `MiniMax-M2.7-W8A8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

Take the `serve` subcommand as an example:

```shell
export VLLM_USE_MODELSCOPE=True
vllm bench serve \
  --model /path/to/weight/MiniMax-M2.7-w8a8-QuaRot \
  --dataset-name random \
  --random-input 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./
```

> **TODO (Model Owner):** Please fill in the specific performance metrics (throughput, TPOT, TTFT) for MiniMax-M2.7 under various deployment configurations.

## 9 Performance Tuning

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

### 9.1 Recommended Configurations

#### Scenario Overview

| Scenario | Deployment Mode | Total NPUs | Weight Version | Key Considerations |
|----------|----------------|------------|----------------|------------------------|
| High Throughput (32K → 1K) | 1P1D deployment | 16 (A3) | MiniMax-M2.7-W8A8 | Increase `max-num-seqs` and `data-parallel-size` |
| Long Context (128K → 2K) | Single-node | 16 (A3) | MiniMax-M2.7-W8A8 | Enable context parallelism, reduce `data-parallel-size` |
| Low Latency | Single-node | 8–16 | MiniMax-M2.7-W8A8 | Reduce `max-num-batched-tokens`, enable FullGraph |

#### Detailed Configuration

| Scenario | Configuration | NPUs | TP | DP | BS | Concurrency | Max Context | MTP (EAGLE3) | FUSED_MC2 | FlashComm1 | Async Scheduling |
|----------|---------------|------|----|----|----|-------------|-------------|--------------|-----------|------------|------------------|
| High Throughput (32K→1K) | A3 Single-node | 16 | 4 | 4 | 32 | 48 | 40k | 3 | On | On | On |
| Long Context (128K→2K) | A3 Single-node | 16 | 8 | 1 | 8 | 16 | 138k | 1 | On | On | Off |
| Low Latency (3.5K→1.5K) | A3 Single-node | 16 | 16 | 1 | 8 | 16 | 32k | 3 | On | On | On |
| A2 (8-NPU) | A2 Single-node | 8 | 8 | 1 | 16 | 32 | 32k | 3 | Off | On | On |

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods.

Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

#### 9.2.2 Model-Specific Optimizations

##### Optimizations Enabled by Default

The following optimizations are enabled by default and require no additional configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
| ---------------------- | ------------------- | ------------------- |
| FullGraph Optimization | Captures and replays the entire decoding graph at once using `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` | Significantly reduces scheduling latency, stabilizes multi-device performance |
| CPU Binding | Uses `--additional-config '{"enable_cpu_binding":true}'` to bind CPU cores | Reduces cross-core scheduling overhead, improving decode latency stability |
| Multi-thread Weight Loading | Uses `--model-loader-extra-config '{"enable_multithread_load":true}'` for parallel weight loading | Reduces model loading time |

##### Optimizations That Require Explicit Enabling

| Optimization Technique | Applicable Scenarios | Enablement Method | Technical Principle | Precautions |
| ---------------------- | -------------------- | ----------------- | ------------------- | ----------- |
| FlashComm v1 | High-concurrency, TP scenarios | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | Decomposes traditional Allreduce into Reduce-Scatter and All-Gather | Threshold protection: only takes effect when the actual number of tokens exceeds the threshold |
| Fused MC2 | TP ≥ 4 scenarios | `export VLLM_ASCEND_ENABLE_FUSED_MC2=1` | Fuses multiple communication and computation operations | Recommended for A3; not applicable for A2 |
| Balanced Scheduling | High DP scenarios | `export VLLM_ASCEND_BALANCE_SCHEDULING=1` | Enhances scheduling capacity between prefill and decode | Works best when concurrency ≈ DP × max-num-seqs. Disable for long-context scenarios |
| EAGLE3 Speculative Decoding | All scenarios | `--speculative_config '{"method": "eagle3", "model": "/path/to/Eagle3/", "num_speculative_tokens": 3}'` | Uses a draft model to predict future tokens | 1–3 tokens for long context; 3 tokens for short context |
| jemalloc Preload | All scenarios | `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2` | Replaces default memory allocator to reduce fragmentation | Ensure jemalloc is installed in the container |

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This chapter only covers MiniMax-M2.7/M2.5 model-specific issues.

- **Q: What should I do if the output is garbled in EP mode?**

  A: It is recommended to keep `--enable-expert-parallel` and `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`.

- **Q: Why is the `reasoning` field often empty after using `minimax_m2_append_think`?**

  A: This is expected. The parser keeps `<think>...</think>` inside `content`. If you mainly rely on the reasoning semantics of `/v1/responses`, use `--reasoning-parser minimax_m2` instead.

- **Q: Startup fails with HCCL port conflicts (address already bound). What should I do?**

  A: Clean up old processes and restart: `pkill -f "vllm serve"`.

- **Q: How to handle OOM or unstable startup?**

  A: Reduce `--max-num-seqs` and `--max-num-batched-tokens` first. If needed, reduce concurrency and load-testing pressure. Also consider lowering `--gpu-memory-utilization` (e.g., from 0.9 to 0.85).

- **Q: How should I choose `--reasoning-parser`?**

  A: This guide uses `minimax_m2_append_think` so that `<think>...</think>` is kept in `content`. If you mainly rely on the reasoning semantics of `/v1/responses`, consider using `--reasoning-parser minimax_m2`.

- **Q: Which ports must be accessible?**

  A: At minimum, expose the serving port (e.g., `8000`). For multi-node deployment, also ensure HCCL communication ports and DP RPC ports are accessible.
