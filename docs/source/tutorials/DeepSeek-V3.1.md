# DeepSeek-V3.1

## Introduction

DeepSeek-V3.1 is a hybrid model that supports both thinking mode and non-thinking mode. Compared to the previous version, this upgrade brings improvements in multiple aspects:

- Hybrid thinking mode: One model supports both thinking mode and non-thinking mode by changing the chat template.

- Smarter tool calling: Through post-training optimization, the model's performance in tool usage and agent tasks has significantly improved.

- Higher thinking efficiency: DeepSeek-V3.1-Think achieves comparable answer quality to DeepSeek-R1-0528, while responding more quickly.

The `DeepSeek-V3.1` model is first supported in `vllm-ascend:v0.9.1rc3`

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight
- `DeepSeek-V3.1`(BF16 version): [Download model weight](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3.1)
- `DeepSeek-V3.1-w8a8`(Quantized version without mtp): [Download model weight](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.1-w8a8).
- `DeepSeek-V3.1_w8a8mix_mtp`(Quantized version with mix mtp): [Download model weight](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V3.1-w8a8). Please modify `torch_dtype` from `float16` to `bfloat16` in `config.json`.
- `Method of Quantify`: [msmodelslim](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/DeepSeek/README.md#deepseek-v31-w8a8-%E6%B7%B7%E5%90%88%E9%87%8F%E5%8C%96-mtp-%E9%87%8F%E5%8C%96). You can use these methods to quantify the model.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../installation.md#verify-multi-node-communication).

### Installation

You can using our official docker image to run `DeepSeek-V3.1` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

```{code-block} bash
   :substitutions:
# Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.
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
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /mnt/sfs_turbo/.cache:/root/.cache \
-it $IMAGE bash
```

If you want to deploy multi-node environment, you need to set up environment on each node.

## Deployment

### Single-node Deployment

- Quantized model `DeepSeek-V3.1_w8a8mix_mtp` can be deployed on 1 Atlas 800 A3 (64G × 16).

Run the following script to execute online inference.

```shell
#!/bin/sh
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is install on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

# AIV
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_ENABLE_MLAPO=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export DISABLE_L2_CACHE=1

vllm serve /weights/DeepSeek-V3.1_w8a8mix_mtp \
--host 0.0.0.0 \
--port 8015 \
--data-parallel-size 4 \
--tensor-parallel-size 4 \
--quantization ascend \
--seed 1024 \
--served-model-name deepseek_v3 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 8192 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"torchair_graph_config":{"enabled":false}}'
```

### Multi-node Deployment

- `DeepSeek-V3.1_w8a8mix_mtp`: require at least 2 Atlas 800 A2 (64G × 8).

Run the following scripts on two nodes respectively.

**Node 0**

```shell
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="xxxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is install on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

# AIV
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

vllm serve /weights/DeepSeek-V3.1_w8a8mix_mtp \
--host 0.0.0.0 \
--port 8004 \
--data-parallel-size 4 \
--data-parallel-size-local 2 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 4 \
--quantization ascend \
--seed 1024 \
--served-model-name deepseek_v3 \
--enable-expert-parallel \
--max-num-seqs 20 \
--max-model-len 8192 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.94 \
--speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"torchair_graph_config":{"enabled":false}}'
```

**Node 1**

```shell
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is install on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

# AIV
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

vllm serve /weights/DeepSeek-V3.1_w8a8mix_mtp \
--host 0.0.0.0 \
--port 8004 \
--headless \
--data-parallel-size 4 \
--data-parallel-size-local 2 \
--data-parallel-start-rank 2 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 4 \
--quantization ascend \
--seed 1024 \
--served-model-name deepseek_v3 \
--enable-expert-parallel \
--max-num-seqs 20 \
--max-model-len 8192 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.94 \
--speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"torchair_graph_config":{"enabled":false}}'
```

### Prefill-Decode Disaggregation

We recommend using Mooncake for deployment: [Mooncake](./multi_node_pd_disaggregation_mooncake.md).

Take Atlas 800 A3 (64G × 16) for example, we recommend to deploy 2P1D (4 nodes) rather than 1P1D (2 nodes), because there is no enough NPU memory to serve high concurrency in 1P1D case.
 - `DeepSeek-V3.1_w8a8mix_mtp 2P1D Layerwise` require 4 Atlas 800 A3 (64G × 16).

To run the vllm-ascend `Prefill-Decode Disaggregation` service, you need to deploy a `launch_dp_program.py` script and a `run_dp_template.sh` script on each node and deploy a `proxy.sh` script on prefill master node to forward requests.

1. `launch_dp_program.py` script for each node:

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

2. Prefill Node 0 `run_dp_template.sh` script

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="141.xx.xx.1"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is install on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_VERSION="0.11.0"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120


export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICE=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

vllm serve /weights/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name deepseek_v3 \
  --max-model-len 40000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --additional-config '{"recompute_scheduler_enable":true,"enable_shared_expert_dp": true}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_producer",
  "kv_port": "30000",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

3. Prefill Node 1 `run_dp_template.sh` script

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="141.xx.xx.2"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is install on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_VERSION="0.11.0"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120


export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICE=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

vllm serve /weights/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name deepseek_v3 \
  --max-model-len 40000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --additional-config '{"recompute_scheduler_enable":true,"enable_shared_expert_dp": true}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_producer",
  "kv_port": "30100",
  "engine_id": "1",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

4. Decode Node 0 `run_dp_template.sh` script

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="141.xx.xx.3"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is install on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_VERSION="0.11.0"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120


export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=600
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICE=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

vllm serve /weights/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name deepseek_v3 \
  --max-model-len 40000 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 40 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94 \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"lm_head_tensor_parallel_size":16}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_consumer",
  "kv_port": "30200",
  "engine_id": "2",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

5. Decode Node 1 `run_dp_template.sh` script

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="141.xx.xx.4"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is install on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_VERSION="0.11.0"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120


export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=600
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICE=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

vllm serve /weights/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name deepseek_v3 \
  --max-model-len 40000 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 40 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94 \
  --quantization ascend \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"lm_head_tensor_parallel_size":16}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnector",
  "kv_role": "kv_consumer",
  "kv_port": "30300",
  "engine_id": "3",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 32,
                    "tp_size": 1
             }
      }
  }'
```

6. run server for each node

```shell
# p0
python launch_dp_program.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 141.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
# p1
python launch_dp_program.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 141.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
# d0
python launch_dp_program.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address 141.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
# d1
python launch_dp_program.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 16 --dp-address 141.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
```

7. Prefill master node `proxy.sh` scripts

```shell
python load_balance_proxy_server_example.py \
  --port 1999 \
  --host 141.xx.xx.1 \
  --prefiller-hosts \
    141.xx.xx.1 \
    141.xx.xx.1 \
    141.xx.xx.2 \
    141.xx.xx.2 \
  --prefiller-ports \
    7100 7101 7100 7101 \
  --decoder-hosts \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.3 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
    141.xx.xx.4 \
  --decoder-ports \
    7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115 \
    7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115 \
```

8. run proxy

Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_layerwise\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py) or [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
cd vllm-ascend/examples/disaggregated_prefill_v1/
bash proxy.sh
```

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v3",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0
    }'
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench
1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `DeepSeek-V3.1_w8a8mix_mtp` in `vllm-ascend:0.11.0rc1` for reference only.

| dataset | version | metric | mode | vllm-api-general-chat | note |
|----- | ----- | ----- | ----- | -----| ----- |
| ceval | - | accuracy | gen | 90.94 | 1 Atlas 800 A3 (64G × 16) |
| gsm8k | - | accuracy | gen | 96.28 | 1 Atlas 800 A3 (64G × 16) |

### Using Language Model Evaluation Harness
Not test yet.

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `DeepSeek-V3.1_w8a8mix_mtp` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve --model vllm-ascend/DeepSeek-V3.1_w8a8mix_mtp  --dataset-name random --random-input 1024 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.
