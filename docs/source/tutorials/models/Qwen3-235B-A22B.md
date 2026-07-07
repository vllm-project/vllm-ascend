# Qwen3-235B-A22B

## 1 Introduction

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support. Qwen3-235B-A22B is the largest MoE variant, featuring 235B total parameters with 22B activated per token.

This document will demonstrate the main validation steps for Qwen3-235B-A22B in the vLLM-Ascend environment, including supported features, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

The Qwen3-235B-A22B model is first supported in **v0.8.4rc2**. This document is validated and written based on **vLLM-Ascend v0.21.0**. All **v0.21.0 and later versions** can run stably. To use the latest features, it is recommended to use the latest release candidate or official version.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

The following model variants are available. It is recommended to download the model weight to a shared directory accessible to all nodes.

**BF16 Version:**

| Model | Hardware Requirement | Download |
|-------|---------------------|----------|
| Qwen3-235B-A22B (BF16) | 1 Atlas 800I A3 (64G × 16), 1 Atlas 800I A2 (64G × 8)| [Download](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B) |

**Quantized Version (Pre-converted):**

| Model | Quantization | Hardware Requirement | Download |
|-------|-------------|---------------------|----------|
| Qwen3-235B-A22B-W8A8 | W8A8 | 1 Atlas 800I A3 (64G × 16), 1 Atlas 800I A2 (64G × 8)| [Download](https://modelers.cn/models/Modelers_Park/Qwen3-235B-A22B-w8a8) |

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

### 3.2 Model Quantization

**Install msmodelslim:**

```shell
# 1. Clone the msmodelslim repository.
git clone https://gitcode.com/Ascend/msmodelslim.git

# 2. Enter the msmodelslim directory and run the installation script.
cd msmodelslim
bash install.sh

# The following message indicates that msmodelslim has been installed successfully.
Successfully installed msmodelslim-{version}
```

**Run quantization:**

```shell
cd example/Qwen3-MOE
# Run the following command to quantize the model.
python3 quant_qwen_moe_w8a8.py --model_path /path/to/your/Qwen3-235B-A22B \
    --save_path /path/to/your/Qwen3-235B-A22B-W8A8-rot \
    --anti_dataset ../common/qwen3-moe_anti_prompt_50.json \
    --calib_dataset ../common/qwen3-moe_calib_prompt_50.json \
    --trust_remote_code True \
    --rot
```

### 3.3 Verify Multi-node Communication

If you need to deploy a multi-node environment, verify the multi-node communication according to [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3 MoE models.

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

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve your_model_path \
    --host <host_ip> \
    --port <port> \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --seed 1024 \
    --quantization ascend \
    --served-model-name qwen3 \
    --max-num-seqs 32 \
    --max-model-len 131072 \
    --max-num-batched-tokens 8096 \
    --enable-expert-parallel \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_flashcomm1": true}' \
    --async-scheduling
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

### 5.2 Multi-Node PD Separation Deployment

PD (Prefill-Decode) separation splits the Prefill and Decode phases across different nodes for better throughput. The following example shows the parameter configuration for a three-node A3 PD disaggregation scenario (one Prefill node + two Decode nodes):

For the detailed deployment guide, please refer to [Prefill-Decode Disaggregation Mooncake Verification](../features/pd_disaggregation_mooncake_multi_node.md).

**Hardware**: 3 × Atlas 800 A3 (64G × 16), one for Prefill, two for Decode.

First, prepare `launch_online_dp.py` on each node:

```python
import argparse
import multiprocessing
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True, help="Data parallel size.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--dp-size-local", type=int, default=-1, help="Local data parallel size.")
    parser.add_argument("--dp-rank-start", type=int, default=0, help="Starting rank for data parallel.")
    parser.add_argument("--dp-address", type=str, required=True, help="IP address for data parallel master node.")
    parser.add_argument("--dp-rpc-port", type=str, default=12345, help="Port for data parallel master node.")
    parser.add_argument("--vllm-start-port", type=int, default=9000, help="Starting port for the engine.")
    return parser.parse_args()

args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
dp_size_local = args.dp_size_local
if dp_size_local == -1:
    dp_size_local = dp_size
dp_rank_start = args.dp_rank_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port

def run_command(visible_devices, dp_rank, vllm_engine_port):
    command = [
        "bash",
        "./run_dp_template.sh",
        visible_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        dp_rpc_port,
        str(tp_size),
    ]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    template_path = "./run_dp_template.sh"
    if not os.path.exists(template_path):
        print(f"Template file {template_path} does not exist.")
        sys.exit(1)

    processes = []
    num_cards = dp_size_local * tp_size
    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        vllm_engine_port = vllm_start_port + i
        visible_devices = ",".join(str(x) for x in range(i * tp_size, (i + 1) * tp_size))
        process = multiprocessing.Process(target=run_command, args=(visible_devices, dp_rank, vllm_engine_port))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
```

Then prepare `run_dp_template.sh` on each node.

**Prefill node** (set `nic_name` and `local_ip` to your own):

```bash
nic_name="<your_nic_name>"
local_ip="<your_ip>"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve "/data/weights/Qwen3-235B-A22B-w8a8-rot" \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --served-model-name qwen3_235b \
    --max-model-len 40960 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 24 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --no-enable-prefix-caching \
    --enforce-eager \
    --additional-config '{"enable_flashcomm1": true, "enable_fused_mc2": 1}' \
    --kv-transfer-config \
        '{"kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "engine_id": "0",
        "kv_connector_extra_config": {
             "use_ascend_direct": true,
             "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 8,
                    "tp_size": 4
             }
        }
        }'
```

**Decode node 0** (set `nic_name` and `local_ip` to your own):

```bash
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
export TASK_QUEUE_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export VLLM_TORCH_PROFILER_WITH_STACK=0
export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve "/data/weights/Qwen3-235B-A22B-w8a8-rot" \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --served-model-name qwen3_235b \
    --max-model-len 40960 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 128 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --no-enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_flashcomm1": true, "enable_fused_mc2": 2}' \
    --kv-transfer-config \
        '{"kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": "30100",
        "engine_id": "1",
        "kv_connector_extra_config": {
             "use_ascend_direct": true,
             "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 8,
                    "tp_size": 4
             }
        }
        }'
```

**Decode node 1** (set `nic_name` and `local_ip` to your own):

```bash
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
export TASK_QUEUE_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export VLLM_TORCH_PROFILER_WITH_STACK=0
export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve "/data/weights/Qwen3-235B-A22B-w8a8-rot" \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --served-model-name qwen3_235b \
    --max-model-len 40960 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 128 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --no-enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_flashcomm1": true, "enable_fused_mc2": 2}' \
    --kv-transfer-config \
        '{"kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": "30100",
        "engine_id": "1",
        "kv_connector_extra_config": {
             "use_ascend_direct": true,
             "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 8,
                    "tp_size": 4
             }
        }
        }'
```

Once the scripts are ready, start the servers on each node.

**Prefill node:**

```bash
python launch_online_dp.py \
    --dp-size 2 --tp-size 8 \
    --dp-size-local 2 --dp-rank-start 0 \
    --dp-address <prefill_ip> --dp-rpc-port 54951 \
    --vllm-start-port 9123
```

**Decode node 0:**

```bash
python launch_online_dp.py \
    --dp-size 8 --tp-size 4 \
    --dp-size-local 4 --dp-rank-start 0 \
    --dp-address <decode_ip> --dp-rpc-port 54951 \
    --vllm-start-port 9123
```

**Decode node 1:**

```bash
python launch_online_dp.py \
    --dp-size 8 --tp-size 4 \
    --dp-size-local 4 --dp-rank-start 4 \
    --dp-address <decode_ip> --dp-rpc-port 54951 \
    --vllm-start-port 9123
```

**Request Forwarding:**

Run the proxy on any machine that can reach both nodes. You can get the proxy script from the repository: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).

```bash
unset http_proxy https_proxy

python load_balance_proxy_server_example.py \
  --port 38085 \
  --host <prefill_ip> \
  --prefiller-hosts \
    <prefill_ip> <prefill_ip> \
  --prefiller-ports \
    9123 9124 \
  --decoder-hosts \
    <decode0_ip> <decode0_ip> <decode0_ip> <decode0_ip> \
    <decode1_ip> <decode1_ip> <decode1_ip> <decode1_ip> \
  --decoder-ports \
    9123 9124 9125 9126 \
    9123 9124 9125 9126 \
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

After the service is started, the model can be invoked by sending a prompt:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

## 7 Accuracy Evaluation

### Using AISBench

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

**Run the accuracy evaluation using the aime2024 dataset as an example:**

```bash
ais_bench --models vllm_api_general_chat --datasets aime2024_gen_0_shot_chat_prompt --debug
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
        model="qwen",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        host_ip="localhost",
        host_port=20002,
        max_out_len=1500,
        batch_size=140,
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
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf --num-prompts 560
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
    --dataset-name random \
    --random-input 200 \
    --num-prompts 200 \
    --request-rate 1 \
    --save-result \
    --result-dir ./
```

After several minutes, you will get the performance evaluation result.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
|----------|----------------|-------------|----------------|---------------------|
| High Throughput | Single-Node (TP4, DP4) | 16 (A3) | W8A8 | DP and TP distribute MoE experts across 16 NPUs for maximum throughput |
| High Throughput | PD Disaggregation (3 nodes) | 48 (3×A3) | W8A8 | 3-node PD separation balances prefill and decode resources for high throughput |
| Low Latency | Single-Node (TP16) | 16 (A3) | W8A8 | 16-NPU TP minimizes per-token latency with speculative decoding |
| Long Context | Single-Node (TP8, CP2) | 16 (A3) | W8A8 | 16-NPU TP with Context Parallelism extends context to 135K tokens |

> `*Total NPUs` indicates the total number of NPUs used across all nodes.

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | #NPUs | TP | DP | MTP Speculation Num | FUSED_MC2 | EP Switch | Async Scheduling |
|----------|---------------|-------|----|-------------|--------------------|-----------|-----------|--------------|
| High Throughput | Single-Node | 16 | 4 | 4 | none | On | On  | On |
| Low Latency | Single-Node | 16 | 16 | 1 | 3 | Off | On | On |
| Long Context | Single-Node | 16 | 8 | 1 | none | On | On | Off |

> For additional parameter details, please refer to the deployment examples in [Section 5.1](#51-single-node-online-deployment)

<u>Single-node PD Hybrid — High Throughput:</u>

Single-node PD hybrid deployment optimized for maximum throughput on Atlas 800I A3 (64G × 16):

```bash
export HCCL_IF_IP=<node_ip>
export GLOO_SOCKET_IFNAME=<ifname>
export TP_SOCKET_IFNAME=<ifname>
export HCCL_SOCKET_IFNAME=<ifname>

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

vllm serve your_model_path \
    --served-model-name qwen3 \
    --host <host_ip> \
    --port <port> \
    --async-scheduling \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --data-parallel-size-local 4 \
    --data-parallel-start-rank 0 \
    --data-parallel-address <node_ip> \
    --data-parallel-rpc-port <rpc_port> \
    --enable-expert-parallel \
    --max-num-seqs 128 \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --quantization ascend \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1": true, "enable_fused_mc2": 1}'
```

<u>Single-node PD Hybrid — Low Latency:</u>

Single-node PD hybrid deployment optimized for low latency with speculative decoding (Eagle3):

```bash
export HCCL_IF_IP=<node_ip>
export GLOO_SOCKET_IFNAME=<ifname>
export TP_SOCKET_IFNAME=<ifname>
export HCCL_SOCKET_IFNAME=<ifname>

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

vllm serve your_model_path \
    --served-model-name qwen3 \
    --host <host_ip> \
    --port <port> \
    --async-scheduling \
    --tensor-parallel-size 16 \
    --data-parallel-size 1 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address <node_ip> \
    --data-parallel-rpc-port <rpc_port> \
    --enable-expert-parallel \
    --max-num-seqs 128 \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --quantization ascend \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"method": "eagle3", "model":"your_eagle3_model_path", "num_speculative_tokens": 3}' \
    --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1": true}'
```

<u>Single-node PD Hybrid — Long Context:</u>

Single-node PD hybrid deployment optimized for long context with Context Parallelism and yarn rope-scaling:

```bash
export HCCL_IF_IP=<node_ip>
export GLOO_SOCKET_IFNAME=<ifname>
export TP_SOCKET_IFNAME=<ifname>
export HCCL_SOCKET_IFNAME=<ifname>

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

vllm serve your_model_path \
    --served-model-name qwen3 \
    --host <host_ip> \
    --port <port> \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --decode-context-parallel-size 2 \
    --prefill-context-parallel-size 2 \
    --enable-expert-parallel \
    --cp-kv-cache-interleave-size 128 \
    --max-num-seqs 32 \
    --max-model-len 135000 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --quantization ascend \
    --no-enable-prefix-caching \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":131072}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1": true, "enable_fused_mc2": 1}'
```

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [vLLM-Ascend FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs/). This section only covers issues specific to Qwen3-235B-A22B.

### Q: What hardware is required for Qwen3-235B-A22B?

For BF16: 1 Atlas 800I A3 (64G × 16) node, 1 Atlas 800I A2 (64G × 8) node, or 2 Atlas 800I A2 (32G × 8) nodes. For W8A8 quantized version, the hardware requirements are similar.

### Q: How do I enable long context beyond 40K?

Use yarn rope-scaling. For vLLM >= v0.12.0: `--hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}'`. For older versions, use `--rope_scaling`. Model variants like Qwen3-235B-A22B-Instruct-2507 natively support long contexts and don't need this parameter.

### Q: When should I use PD disaggregation vs single-node deployment?

Single-node deployment is simpler and recommended when the model fits within a single node. PD disaggregation separates Prefill and Decode across nodes, enabling higher throughput for large-scale serving. For Qwen3-235B-A22B, three A3 nodes with PD disaggregation can achieve ~3× the throughput of single-node deployment.

### Q: What is the difference between `enable_fused_mc2=1` and `=2`?

Value `1` enables the base MoE fused operator, suitable for typical EP configurations. Value `2` enables an alternative fusion strategy optimized for large-scale EP (e.g., EP32 in PD disaggregation scenarios). Both are experimental and currently only support W8A8 quantization on Atlas A3 servers.

### Q: When should I use Expert Parallelism?

Expert Parallelism (EP) should always be enabled for Qwen3-235B-A22B (an MoE model) via `--enable-expert-parallel`. It distributes FFN experts across NPUs to reduce per-device computation. EP works alongside TP, where MoE layers use EP and non-MoE layers use TP.

### Q: How do I choose between Context Parallelism and PD Disaggregation?

Context Parallelism (CP) splits the KV cache of a single request across multiple NPUs, suitable for long context scenarios on a single node. PD Disaggregation separates Prefill and Decode across nodes, suitable for high-throughput serving with many concurrent requests.
