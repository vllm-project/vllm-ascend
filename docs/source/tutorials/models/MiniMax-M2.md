# MiniMax-M2

## 1 Introduction

MiniMax-M2 is MiniMax's flagship large language model series, including **MiniMax-M2.5** and **MiniMax-M2.7**. It is reinforced for high-value scenarios such as code generation, agentic tool calling/search, and complex office workflows, with an emphasis on reasoning efficiency and end-to-end speed on challenging tasks.

This document will show the main verification steps for both MiniMax-M2.5 and MiniMax-M2.7, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

This document is written based on the latest vLLM-Ascend version. Both MiniMax-M2.5 and MiniMax-M2.7 are fully supported. To use the latest features (e.g., PD separation, EAGLE3 speculative decoding), it is recommended to use the latest version.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

The following model weights and EAGLE3 weights are available on ModelScope. Search for the corresponding model name on [ModelScope](https://modelscope.cn) to obtain the latest weight files.

| Model | Description | Recommended Hardware | Source |
|-------|-------------|---------------------|--------|
| `MiniMax-M2.7-w8a8-QuaRot` | M2.7 W8A8 quantized version | 1× Atlas 800 A3 (64G × 16) or 1× Atlas 800I A2 (64G × 8) | [MiniMax-M2.7-w8a8-QuaRot](https://www.modelscope.ai/models/vllm-ascend/MiniMax-M2.7-w8a8-QuaRot) |
| `MiniMax-M2.5-w8a8-QuaRot` | M2.5 W8A8 quantized version | 1× Atlas 800 A3 (64G × 16) or 1× Atlas 800I A2 (64G × 8) | [MiniMax-M2.5-w8a8-QuaRot](https://modelscope.cn/models/Eco-Tech/MiniMax-M2.5-w8a8-QuaRot) |
| `MiniMax-M2.7-w8a8c8-QuaRot` | M2.7 W8A8C8 quantized version | 1× Atlas 800 A3 (64G × 16) or 1× Atlas 800I A2 (64G × 8) | [MiniMax-M2.7-w8a8c8-QuaRot](https://www.modelscope.ai/models/vllm-ascend/MiniMax-M2.7-w8a8c8-QuaRot) |
| `Eagle3` (M2.7) | M2.7 speculative decoding head model | Matches the base model node count | [MiniMax-M2.7-eagle-model](https://modelscope.cn/models/Eco-Tech/MiniMax-M2.7-eagle-model-short) |
| `Eagle3` (M2.5) | M2.5 speculative decoding head model | Matches the base model node count | [MiniMax-M2.5-eagle-model](https://modelscope.cn/models/vllm-ascend/MiniMax-M2.5-eagle-model-0318) |

It is recommended to download the model weights to a shared directory, such as `/root/.cache/`.

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
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

**A2 series**

Map your model weight directory into the container (the example maps it to `/root/.cache/`).

```{code-block} bash
#!/bin/sh
NAME=minimax
IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|

docker run -itd -u 0 --ipc=host \
  -e VLLM_USE_MODELSCOPE=True \
  -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
  --name $NAME \
  --net=host \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --shm-size=1200g \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -it $IMAGE bash
```

Save the script as `minimax-docker-run.sh`, then start and enter the container:

```bash
bash minimax-docker-run.sh
docker exec -it minimax bash
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

- If you only care about short-context low latency, you can set `--max-model-len 32768`, `--tensor-parallel-size 4`, and `--data-parallel-size 4`.

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

export VLLM_ASCEND_BALANCE_SCHEDULING=0

vllm serve /path/to/weight/MiniMax-M2.7-w8a8-QuaRot \
    --served-model-name "MiniMax-M2.7" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --additional-config '{"enable_cpu_binding":true,
                          "enable_fused_mc2":true,
                          "enable_flashcomm1":true,
                          "weight_nz_mode":true}' \
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
- To achieve better performance on long-context scenarios (e.g., 128k or 64k), we recommend the following adjustments:

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

> **Note**: The above parameters are validated in a specific test environment for reference only. Please adjust `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, and `--gpu-memory-utilization` based on your actual input/output length, concurrency, and hardware configuration.

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
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.85 \
    --additional-config '{"enable_cpu_binding":true,
                          "enable_flashcomm1":true}' \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
    --speculative_config '{"method": "eagle3", "model": "/path/to/weight/Eagle3/",  "num_speculative_tokens":3}'
```

> **Note**: The above parameters are validated in a specific test environment for reference only. Please adjust `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, and `--gpu-memory-utilization` based on your actual input/output length, concurrency, and hardware configuration.

- If you need to test with `curl` and tool calling, add the following to the startup command:

```{code-block} bash
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
```

### 5.2 Multi-Node PD Separation Deployment

PD (Prefill-Decode) separation splits the Prefill and Decode phases across different nodes for better throughput. The following 1P1D configuration is validated for 128k input/output scenarios with `MiniMax-M2.7-W8A8`.

**Hardware**: 2× Atlas 800 A3 (64G × 16), one for Prefill, one for Decode.

**Common Issues Tip:** For PD separation specific issues such as KV transfer timeouts or Mooncake connection errors, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). For MiniMax-specific PD separation issues, refer to [Chapter 10 FAQ](#10-faq).

First, prepare `launch_online_dp.py` on each node:

```python
import argparse
import multiprocessing
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size-local", type=int, default=-1)
    parser.add_argument("--dp-rank-start", type=int, default=0)
    parser.add_argument("--dp-address", type=str, required=True)
    parser.add_argument("--dp-rpc-port", type=str, default=12345)
    parser.add_argument("--vllm-start-port", type=int, default=9000)
    return parser.parse_args()

args = parse_args()
dp_size, tp_size = args.dp_size, args.tp_size
dp_size_local = args.dp_size_local if args.dp_size_local != -1 else dp_size

def run_command(visible_devices, dp_rank, vllm_engine_port):
    subprocess.run([
        "bash", "./run_dp_template.sh",
        visible_devices, str(vllm_engine_port),
        str(dp_size), str(dp_rank), args.dp_address,
        args.dp_rpc_port, str(tp_size),
    ], check=True)

if __name__ == "__main__":
    for i in range(dp_size_local):
        dp_rank = args.dp_rank_start + i
        vllm_port = args.vllm_start_port + i
        visible_devices = ",".join(str(x) for x in range(i * tp_size, (i + 1) * tp_size))
        p = multiprocessing.Process(target=run_command, args=(visible_devices, dp_rank, vllm_port))
        p.start()
        p.join()
```

Then prepare `run_dp_template.sh` on each node.

**Prefill node** (set `nic_name` and `local_ip` to your own):

```bash
unset http_proxy https_proxy ftp_proxy

nic_name="<your_nic_name>"
local_ip="<your_ip>"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export PYTHONHASHSEED=0

export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve /path/to/weight/MiniMax-M2.7-w8a8-QuaRot \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --served-model-name minimax \
    --max-model-len 200000 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --quantization ascend \
    --enforce-eager \
    --speculative_config '{"method": "eagle3", "model": "/path/to/weight/Eagle3/", "num_speculative_tokens": 3}' \
    --additional-config '{"enable_cpu_binding":true}' \
    --kv-transfer-config \
        '{"kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "35880",
        "engine_id": "0",
        "kv_connector_extra_config": {
             "use_ascend_direct": true,
             "prefill": {"dp_size": 2, "tp_size": 8},
             "decode":  {"dp_size": 2, "tp_size": 8}
        }}'
```

**Decode node** (set `nic_name` and `local_ip` to your own):

```bash
unset http_proxy https_proxy ftp_proxy

nic_name="<your_nic_name>"
local_ip="<your_ip>"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_BUFFSIZE=2048
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export PYTHONHASHSEED=0

export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve /path/to/weight/MiniMax-M2.7-w8a8-QuaRot \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --served-model-name minimax \
    --max-model-len 200000 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 16 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --quantization ascend \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative_config '{"method": "eagle3", "model": "/path/to/weight/Eagle3/", "num_speculative_tokens": 3}' \
    --additional-config '{"enable_cpu_binding":true}' \
    --kv-transfer-config \
        '{"kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": "56900",
        "engine_id": "1",
        "kv_connector_extra_config": {
             "use_ascend_direct": true,
             "prefill": {"dp_size": 2, "tp_size": 8},
             "decode":  {"dp_size": 2, "tp_size": 8}
        }}'
```

Once the scripts are ready, start the servers on each node.

**Prefill node:**

```bash
python launch_online_dp.py \
    --dp-size 2 --tp-size 8 \
    --dp-size-local 2 --dp-rank-start 0 \
    --dp-address <prefill_ip> --dp-rpc-port 12321 \
    --vllm-start-port 7000
```

**Decode node:**

```bash
python launch_online_dp.py \
    --dp-size 2 --tp-size 8 \
    --dp-size-local 2 --dp-rank-start 0 \
    --dp-address <decode_ip> --dp-rpc-port 12321 \
    --vllm-start-port 7100
```

#### Request Forwarding

Run the proxy on any machine that can reach both nodes. You can get the proxy script from the repository: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).

```bash
unset http_proxy https_proxy

python load_balance_proxy_server_example.py \
    --port 8009 \
    --host <prefill_ip> \
    --prefiller-hosts \
       <prefill_ip> <prefill_ip> \
    --prefiller-ports \
       7000 7001 \
    --decoder-hosts \
       <decode_ip> <decode_ip> \
    --decoder-ports \
       7100 7101
```

The service is then accessible at `http://<proxy_ip>:8009`.

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

> **Note**: Post-processing parameters (e.g., `max_tokens`, `temperature`, `stop` tokens) should match those defined in the model weight's `generation_config.json`. The recommended maximum output length for GPQA-diamond and AIME2025 is 64k (65536 tokens).

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

## 8 Performance Evaluation

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

## 9 Performance Tuning

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

### 9.1 Recommended Configurations

The following configurations are validated on the self-test report (AR20260326132822) and are categorized by use case.

| Scenario | Input/Output | Deployment | NPUs | P Config | D Config | Max Batched Tokens | Max Num Seqs (P/D) | Max Model Len | EAGLE3 | FUSED_MC2 | FlashComm1 | Async Scheduling |
|----------|-------------|------------|------|----------|----------|-------------------|----------------|---------------|--------|-----------|------------|------------------|
| Short Seq High Throughput | 3.5K → 1.5K | 1P2D PD separation | 24 (A3) | DP8TP2EP16 | DP32TP1EP32 | 16384 | 128 / 128 | 32k | 3 | On | On | On |
| Short Seq Low Latency | 3.5K → 1.5K | 1P2D PD separation | 24 (A3) | DP4TP4EP16 | DP8TP4EP32 | 16384 | 128 / 128 | 32k | 3 | On | On | On |
| Long Seq High Throughput | 128K → 1K <br>（90% cache hit） | 1P1D PD separation | 16 (A3) | DP2TP8EP16 | DP2TP8EP16 | 16384 | 64 / 16 | 200k | 3 | On | On | On |
| Long Seq Low Latency | 128K → 1K <br>（90% cache hit） | 1P2D PD separation | 24 (A3) | DP2TP8EP16 | DP4TP8EP32 | 16384 | 64 / 16 | 200k | 3 | On | On | On |

> **Note**: The prefix cache hit rate for short-sequence tests is 0%; for long-sequence tests it is 90%. Adjust `max-num-seqs`, `max-model-len`, and `max-num-batched-tokens` based on your actual workload.

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
| FlashComm v1 | High-concurrency, TP scenarios | `--additional-config '{"enable_flashcomm1": true}'` | Decomposes traditional Allreduce into Reduce-Scatter and All-Gather | Threshold protection: only takes effect when the actual number of tokens exceeds the threshold |
| Fused MC2 | TP ≥ 4 scenarios | `--additional-config '{"enable_fused_mc2": true}'` | Fuses multiple communication and computation operations | Recommended for A3; not applicable for A2 |
| Balanced Scheduling | High DP scenarios | `export VLLM_ASCEND_BALANCE_SCHEDULING=1` | Enhances scheduling capacity between prefill and decode | Currently disabled by default (`0`). Set to `1` only when concurrency ≈ DP × max-num-seqs. Disable for long-context scenarios |
| EAGLE3 Speculative Decoding | All scenarios | `--speculative_config '{"method": "eagle3", "model": "/path/to/Eagle3/", "num_speculative_tokens": 3}'` | Uses a draft model to predict future tokens | 1–3 tokens for long context; 3 tokens for short context |
| jemalloc Preload | All scenarios | `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2` | Replaces default memory allocator to reduce fragmentation | Ensure jemalloc is installed in the container |

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This chapter only covers MiniMax-M2 (M2.5/M2.7) model-specific issues.

- **Q: Does C8 quantization support EAGLE3 speculative decoding?**

  A: Not yet. C8 quantization with EAGLE3 is currently unsupported.

- **Q: Which `--reasoning-parser` is recommended for tool calling tasks?**

  A: For tool calling tasks, it is recommended to use `--reasoning-parser minimax_m2_append_think`.

- **Q: Why is the `reasoning` field often empty after using `minimax_m2_append_think`?**

  A: This is expected. The parser keeps `<think>...</think>` inside `content`. If you mainly rely on the reasoning semantics of `/v1/responses`, use `--reasoning-parser minimax_m2` instead.

- **Q: Startup fails with HCCL port conflicts (address already bound). What should I do?**

  A: Check whether another process is already occupying the port (e.g., `lsof -i :<port>` or `ss -tlnp | grep <port>`). If a port conflict is found, switch to a different port with `--port`, or terminate the specific process occupying that port.

- **Q: How to handle OOM or unstable startup?**

  A: Refer to the upstream vLLM guide on [out-of-memory troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/#out-of-memory). In short: reduce `--max-num-seqs` and `--max-num-batched-tokens` first, lower `--gpu-memory-utilization` (e.g., from 0.9 to 0.85), or decrease the number of concurrent requests.

- **Q: How should I choose `--reasoning-parser`?**

  A: This guide uses `minimax_m2_append_think` so that `<think>...</think>` is kept in `content`. If you mainly rely on the reasoning semantics of `/v1/responses`, consider using `--reasoning-parser minimax_m2`.

- **Q: Which ports must be accessible?**

  A: At minimum, expose the serving port (e.g., `8000`). For multi-node deployment, also ensure HCCL communication ports and DP RPC ports are accessible.
