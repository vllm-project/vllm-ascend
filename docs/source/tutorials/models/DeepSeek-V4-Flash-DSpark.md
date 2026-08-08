# DeepSeek-V4-Flash DSpark

## 1 Introduction

This guide covers deployment of DeepSeek-V4-Flash with DSpark speculative decoding. Specifically for the DeepSeek-V4-Flash-DSpark-0731 weights released by DeepSeek on July 31.

## 2 Prerequisites

### 2.1 Model Weight

Use quantized `DeepSeek-V4-Flash-0731-w8a8` weight. Download it from [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8) and place it in a shared directory, for example `/path/to/DeepSeek-V4-Flash-0731-w8a8`.

### 2.2 Container Image

**Attention**: DSpark is supported on both A2 and A3 in vLLM Ascend `v0.25.0` and later. Use the image `quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731` for A2 or the image `quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731-a3` for A3.

Start the container as described in [Using Docker](../../installation.md#set-up-using-docker). For multi-node deployment, also complete [multi-node communication verification](../../installation.md#verify-multi-node-communication).

## 3 A3 Deployment Configurations

### 3.1 Single-Node Deployment

This configuration uses the quantized `DeepSeek-V4-Flash-0731-w8a8` weights. Set the local directory containing the weights `/path/to/DeepSeek-V4-Flash-0731-w8a8`.

```shell
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

vllm serve /path/to/DeepSeek-V4-Flash-0731-w8a8 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 10240 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 64 \
    --data-parallel-size 4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --quantization ascend \
    --port 8900 \
    --block-size 32 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{
        "ascend_compilation_config": {
            "enable_npugraph_ex": true,
            "enable_static_kernel": false
        },
        "enable_cpu_binding": true,
        "enable_dsa_cp": true,
        "enable_flashcomm1": true,
        "multistream_overlap_shared_expert": true
    }'
```

### 3.2 1P1D PD Separation Deployment

DSpark uses Mooncake for KV transfer. Before deployment, prepare the following scripts on both the Prefill and Decode nodes.

#### 3.2.1 Prepare `launch_online_dp.py` on Each Node

Prepare the script `launch_online_dp.py` on each node.

```python
import argparse
import multiprocessing
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dp-size",
        type=int,
        required=True,
        help="Data parallel size."
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size."
    )
    parser.add_argument(
        "--dp-size-local",
        type=int,
        default=-1,
        help="Local data parallel size."
    )
    parser.add_argument(
        "--dp-rank-start",
        type=int,
        default=0,
        help="Starting rank for data parallel."
    )
    parser.add_argument(
        "--dp-address",
        type=str,
        required=True,
        help="IP address for data parallel master node."
    )
    parser.add_argument(
        "--dp-rpc-port",
        type=str,
        default=12345,
        help="Port for data parallel master node."
    )
    parser.add_argument(
        "--vllm-start-port",
        type=int,
        default=9000,
        help="Starting port for the engine."
    )
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
        process = multiprocessing.Process(target=run_command,
                                        args=(visible_devices, dp_rank,
                                                vllm_engine_port))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
```

Parameter descriptions:

|Parameter|Type|Required|Default|Description|
|---------|----|--------|-------|-----------|
|`--dp-size`|int|Yes|-|Data parallel size (total number of DP ranks across all nodes).|
|`--tp-size`|int|No|1|Tensor parallel size within each DP rank.|
|`--dp-size-local`|int|No|(same as `--dp-size`)|Number of DP ranks on the current node. If not set, defaults to `--dp-size`.|
|`--dp-rank-start`|int|No|0|Starting rank offset for data parallel ranks on this node.|
|`--dp-address`|str|Yes|-|IP address of the data parallel master node.|
|`--dp-rpc-port`|str|No|12345|RPC port for data parallel master communication.|
|`--vllm-start-port`|int|No|9000|Starting port for each vLLM engine instance on this node.|

#### 3.2.2 Run the Prefill and Decode Nodes

1. Prefill node

```shell
nic_name="xxxx" # change to your own nic name
local_ip=xx.xx.xx.1 # change to your own ip

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export ASCEND_RT_VISIBLE_DEVICES=$1

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

vllm serve /path/to/DeepSeek-V4-Flash-0731-w8a8 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name dsv4-spark \
    --max-model-len 1048576 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --no-disable-hybrid-kv-cache-manager \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --speculative-config '{"num_speculative_tokens": 5,"method": "dspark","enforce_eager": true}' \
    --trust-remote-code \
    --block-size 32 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --enforce-eager \
    --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true,  "enable_dsa_cp": true, "enable_flashcomm1": true}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 4,
                        "tp_size": 4
                },
                "decode": {
                        "dp_size": 16,
                        "tp_size": 1
                }
        }
    }'
```

2. Decode node

```shell
nic_name="xxxx" # change to your own nic name
local_ip=xx.xx.xx.2 # change to your own ip

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve /path/to/DeepSeek-V4-Flash-0731-w8a8 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name dsv4 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 256 \
    --max-num-seqs 32 \
    --block-size 32 \
    --no-disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --tokenizer-mode deepseek_v4 \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 5,"method": "dspark","enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_consumer",
    "kv_port": "30100",
    "engine_id": "1",
    "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 4,
                        "tp_size": 4
                },
                "decode": {
                        "dp_size": 16,
                        "tp_size": 1
                }
        }
    }' \
    --additional-config '{
        "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
        },
        "enable_cpu_binding":true,
        "multistream_overlap_shared_expert":true
    }'
```

3. Start the server with the following command on each node.

1. Prefill node

```shell
# change ip to your own
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
```

2. Decode node

```shell
# change ip to your own
python launch_online_dp.py --dp-size 16 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
```

4. Deploy the P-D disaggregation proxy.

Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

### 3.3 Key Deployment Parameters

- `--speculative-config`: DSpark requires Prefill and Decode to use the same `num_speculative_tokens` value.
- `--no-disable-hybrid-kv-cache-manager`: keeps the hybrid KV cache manager enabled for this model configuration.
- `MooncakeHybridConnector`: transfers KV cache between Prefill and Decode nodes.
- `enable_shared_expert_dp: true`: enables data parallelism for shared experts.
- `recompute_scheduler_enable: true`: enable this only on Decode nodes when recomputation is required because KV cache is insufficient.
- `enable_flashcomm1` enables the FlashComm communication optimization.
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`: Controls the retention interval, in tokens, for prefix-cache checkpoints of hybrid attention layers. It is applicable to DeepSeek-V4 and takes effect only when prefix caching is enabled. Under KV-cache pressure, it can improve the effective prefix-cache hit rate for reusable long prefixes. The value must be a non-negative multiple of `--block-size`; for DeepSeek-V4-Flash, 128 times `--block-size` is recommended. Set it to `4096` when `--block-size` is `32`, or `16384` when `--block-size` is `128`.

For proxy deployment and verification, see [Prefill-Decode Disaggregation (DeepSeek)](../features/pd_disaggregation_mooncake_multi_node.md).

## 4 Accuracy Evaluation

| Dataset | Version | Metric | Mode | Result | Configuration |
| --- | --- | --- | --- | --- | --- |
| GPQA | v0.25.1rc | accuracy | gen | 90.4 | A3 1P1D DSpark w8w8 |
| SWE Multilingual | v0.25.1rc | accuracy | gen | 68.33 | A3 1P1D DSpark w8w8 |
