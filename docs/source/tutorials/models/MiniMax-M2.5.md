# MiniMax-M2.5

## Introduction

MiniMax‑M2.5 is MiniMax’s flagship large language model, reinforced for high‑value scenarios such as code generation, agentic tool calling/search, and complex office workflows, with an emphasis on reasoning efficiency and end‑to‑end speed on challenging tasks.

This document provides a unified deployment guide for `MiniMax-M2.5` on vLLM Ascend, covering both:

- **A3 single-node** deployment (Atlas 800 A3)
- **A2 dual-node** deployment (2× Atlas 800I A2)

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weights

- `MiniMax-M2.5` (fp8 checkpoint): recommended to use **1× Atlas 800 A3** or **2× Atlas 800I A2** nodes. Download the model weights from [MiniMax/MiniMax-M2.5](https://modelscope.cn/models/MiniMax/MiniMax-M2.5).
- `MiniMax-M2.5-w8a8-QuaRot` : Download the model weights from [Eco-Tech/MiniMax-M2.5-w8a8-QuaRot](https://modelscope.cn/models/Eco-Tech/MiniMax-M2.5-w8a8-QuaRot).
- `Eagle3` : Download the model weights from [vllm-ascend/MiniMax-M2.5-eagel-model](https://modelscope.cn/models/vllm-ascend/MiniMax-M2.5-eagel-model-0318).

It is recommended to download the model weights to a shared directory, such as `/mnt/sfs_turbo/.cache/`. The current release automatically detects the MiniMax-M2 fp8 checkpoint, disables fp8 quantization kernels on NPU, and loads the weights by dequantizing to bf16. This behavior may be removed once public bf16 weights are available.

### Installation

You can use the official docker image to run `MiniMax-M2.5` directly.

Select an image based on your machine type and start the container on your node. See [using docker](../../installation.md#set-up-using-docker).

## Run with Docker

### A3 (single node)

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

### A2 (dual node, run on both nodes)

Create and run `minimax25-docker-run.sh` on **both** A2 nodes.

Notes:

- The default configuration assumes an **Atlas 800I A2 8-NPU** node and sets `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. Update it based on your hardware.
- Map your model weight directory into the container (the example maps it to `/opt/data/verification/`).

```{code-block} bash
#!/bin/sh
NAME=minimax2_5
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
# bash minimax25-docker-run.sh
# docker exec -it minimax2_5 bash
```

## Online Inference on Multi-NPU

### A3 (single node)

Below is a recommended startup configuration for short-context condition like 3.5k/1.5k to reach a good performance.

Notes:

- If you only care about short-context low latency, you can explicitly set `--max-model-len 32768`. You may also set `tensor-parallel-size` to 16 and set `data-parallel-size` to 1.
- `export VLLM_ASCEND_BALANCE_SCHEDULING=1` is used to enhance scheduling capacity between prefill and decode. This will work remarkably with a larger `data-parallel-size`. This can increace performance when concurrency gets closer to values equals to `data-parallel-size` times `max-num-seqs`.

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

vllm serve /path/to/weight/MiniMax-M2.5-w8a8-QuaRot \
    --served-model-name "MiniMax-M2.5" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --quantization ascend \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}' \
    --enable-expert-parallel \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --max-num-seqs 48 \
    --max-model-len 40690 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 \
    --speculative_config '{"enforce_eager": true, "method": "eagle3", "model": "/path/to/weight/Eagle3/", "num_speculative_tokens": 3}' \
```

Remarks:

- `minimax_m2_append_think` keeps `<think>...</think>` inside `content`.
- If you mainly rely on the reasoning semantics of `/v1/responses`, it is recommended to use `--reasoning-parser minimax_m2` instead.
- To receive a better performance on long-context like 128k or 64k, we recommend to do changes as shown below, and you can remove `export VLLM_ASCEND_BALANCE_SCHEDULING=1`.

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
    --speculative_config '{"enforce_eager": true, "method": "eagle3", "model": "/path/to/weight/Eagle3/", "num_speculative_tokens": 1}' \
```

- If you will to test with `curl` command, you can add following commands addition to start up command above.

```{code-block} bash
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
```

### A2 (dual node, tp=8 + dp=2)

Since cross-node tensor parallelism (TP) can be unstable, the dual-node guide uses a **tp=8 + dp=2** setup (8 NPUs per node, 16 NPUs total).

#### Node0 (primary) startup script

Edit `minimax25_service_node0.sh` inside the node0 container, and replace the placeholders with your actual values:

- `{PrimaryNodeIP}`: the primary node's IP address (public/cluster network)
- `{NIC}`: the NIC name for the public/cluster network (check via `ifconfig`, e.g., `enp67s0f0np0`)
- `VLLM_TORCH_PROFILER_DIR`: optional, directory to store profiling outputs

```{code-block} bash
# Primary node (node0)
export HCCL_IF_IP={PrimaryNodeIP}
export GLOO_SOCKET_IFNAME="{NIC}"
export TP_SOCKET_IFNAME="{NIC}"
export HCCL_SOCKET_IFNAME="{NIC}"
export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

# profiling (optional)
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_TORCH_PROFILER_DIR="{profiling_dir}"

vllm serve /opt/data/verification/models/MiniMax-M2.5/ \
  --served-model-name "minimax25" \
  --host {PrimaryNodeIP} \
  --port 20004 \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 0 \
  --data-parallel-address {PrimaryNodeIP} \
  --data-parallel-rpc-port 2347 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 65536 \
  --gpu-memory-utilization 0.92 \
  --enable-expert-parallel \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --mm_processor_cache_type="shm" \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding":true}'
```

#### Node1 (secondary) startup script

Edit `minimax25_service_node1.sh` inside the node1 container:

- `{SecondaryNodeIP}`: the secondary node's IP address
- `{PrimaryNodeIP}`: the primary node's IP address (same as node0)
- `{NIC}`: same as above

```{code-block} bash
# Secondary node (node1)
export HCCL_IF_IP={SecondaryNodeIP}
export GLOO_SOCKET_IFNAME="{NIC}"
export TP_SOCKET_IFNAME="{NIC}"
export HCCL_SOCKET_IFNAME="{NIC}"
export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

# profiling (optional)
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_TORCH_PROFILER_DIR="{profiling_dir}"

vllm serve /opt/data/verification/models/MiniMax-M2.5/ \
  --served-model-name "minimax25" \
  --host {SecondaryNodeIP} \
  --port 20004 \
  --headless \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address {PrimaryNodeIP} \
  --data-parallel-rpc-port 2347 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 65536 \
  --gpu-memory-utilization 0.92 \
  --enable-expert-parallel \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --mm_processor_cache_type="shm" \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding":true}'
```

#### Startup order

Start the service on both nodes:

```{code-block} bash
# node0
bash minimax25_service_node0.sh

# node1
bash minimax25_service_node1.sh
```

After node0 prints `service start` in logs, you can verify the service.

## Verify the Service

### A3 (single node)

Test with an OpenAI-compatible client:

```{code-block} python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="na")

resp = client.chat.completions.create(
    model="MiniMax-M2.5",
    messages=[{"role": "user", "content": "你好，请介绍一下你自己，并展示一次工具调用的参数格式。"}],
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

Or send a request using curl:

```{code-block} bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.5",
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

### A2 (dual node)

Run the following from any machine that can reach the primary node (replace `{PrimaryNodeIP}` with the real IP):

```{code-block} bash
curl http://{PrimaryNodeIP}:20004/v1/chat/completions \
  -H "Content-type: application/json" \
  -d '{
    "model": "minimax25",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "stream": false,
    "ignore_eos": true,
    "temperature": 0.8,
    "top_p": 0.8,
    "max_tokens": 200
  }'
```

## Performance Reference

### A3 (single node, tp=16, 4k/1k@bs16)

#### Results

**Baseline** (`3.5k/1k@bs=217`)

| Metric | Result |
| --- | --- |
| Success/Failure | `217/0` |
| Mean TTFT | `10316.56 ms` |
| Mean TPOT | `34.28 ms` |
| Output tok/s | `4803.81` |
| Total tok/s | `16096.59` |

**Long-context reference** (`190k/1k@bs=4`)

| Metric | Result |
| --- | --- |
| Output tok/s | `37.12` |
| Mean TTFT | `2002.37 ms` |
| Mean TPOT | `105.54 ms` |
| Mean ITL | `105.54 ms` |

### A2 (dual node, 190k/1k, concurrency=4, 16 prompts)

#### Benchmark method

Use vLLM bench for the **190k/1k, concurrency=4, 16 prompts** scenario:

```{code-block} bash
vllm bench serve --backend vllm \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 175104 \   # Input: 190×1024 tokens with 90% prefix repetition
  --prefix-repetition-suffix-len 19440 \    # Input: 190×1024 tokens minus the prefix length above
  --prefix-repetition-output-len 1024 \     # Output: 1024 tokens
  --prefix-repetition-num-prefixes 1 \
  --num-prompts 16 \
  --max-concurrency 4 \
  --ignore-eos \
  --model minimax25 \
  --tokenizer {model_path} \
  --endpoint /v1/completions \
  --request-rate inf \
  --seed 1000 \
  --host {service_ip} \
  --port 20004
```

#### Results

**190k/1k, concurrency=4, 16 prompts**

| Metric | Result |
| --- | --- |
| TTFT (avg) | 3305.25 ms |
| TPOT (avg) | 109.83 ms |
| Output throughput | 35.29 tok/s |
| Prefix hit rate | 85% |

## FAQ

- **Q: What should I do if the output is garbled in EP mode?**

  A: It is recommended to keep `--enable-expert-parallel` and `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`.

- **Q: Why is the `reasoning` field often empty after using `minimax_m2_append_think`?**

  A: This is expected. The parser keeps `<think>...</think>` inside `content`. If you mainly rely on the reasoning semantics of `/v1/responses`, use `--reasoning-parser minimax_m2` instead.

- **Q: Startup fails with HCCL port conflicts (address already bound). What should I do?**

  A: Clean up old processes and restart: `pkill -f "vllm serve /models/MiniMax-M2.5"`.

- **Q: How to handle OOM or unstable startup?**

  A: Reduce `--max-num-seqs` and `--max-num-batched-tokens` first. If needed, reduce concurrency and load-testing pressure (e.g., `max-concurrency` / `num-prompts`).

- **Q: Why not use cross-node tp=16?**

  A: The referenced practice noted that cross-node TP may be unstable, so `tp=8, dp=2` is recommended for dual-node deployment.

- **Q: How should I choose `--reasoning-parser`?**

  A: This guide uses `minimax_m2_append_think` so that `<think>...</think>` is kept in `content`. If you mainly rely on the reasoning semantics of `/v1/responses`, consider using `--reasoning-parser minimax_m2`.

- **Q: Which ports must be accessible?**

  A: At minimum, expose the serving port (e.g., `20004`) and the data-parallel RPC port (e.g., `2347`), and ensure the two nodes can reach each other over the network.
