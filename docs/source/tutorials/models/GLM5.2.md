# GLM-5.2

## Introduction

[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) uses a Mixture-of-Experts (MoE) architecture and targets complex systems engineering and long-horizon agentic tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `GLM-5.2`(BF16 version): requires 2 Atlas 800 A3 (128G × 8) node or 4 Atlas 800 A2 (64G × 8) node.[Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-5.2).
- `GLM-5.2-w8a8`: requires 1 Atlas 800 A3 (128G × 8) node or 2 Atlas 800 A2 (64G × 8) node.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w8a8).
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

You can use our official docker image to run GLM-5.2 directly.

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

Start the docker image on your each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:glm5.2-a3
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

::::
::::{tab-item} A2 series
:sync: A2

Start the docker image on each of your nodes.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:glm5.2
docker run --rm \
    --name vllm-ascend \
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

::::
:::::

If you want to deploy multi-node environment, you need to set up environment on each node.

## Deployment

The deployment scenarios validated for this release are organized by context window size (below 1M / 1M), hardware (Atlas 800 A3 / A2), and deployment mode (single-node, multi-node co-located, Prefill-Decode disaggregation). All startup scripts below are the verified reference commands; key parameters are explained after each scenario.

### Context Below 1M

#### Atlas 800 A3

##### Single-Node Deployment

- Quantized model `glm-5.2-w8a8` can be deployed on 1 Atlas 800 A3 (64G × 16) .

Run the following script to execute online inference.

```{code-block} bash
   :substitutions:
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_MLAPO=1
vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8 \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 2 \
--tensor-parallel-size 8 \
--enable-expert-parallel \
--seed 1024 \
--served-model-name glm-52 \
--max-num-seqs 48 \
--max-model-len 20480 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--quantization ascend \
--async-scheduling \
--additional-config '{"multistream_overlap_shared_expert":true}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'

```

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario.

**Model-specific parameters:**

- `--enable-expert-parallel`: Must be enabled for the MoE architecture of GLM-5.2.
- `--quantization ascend`: Enables Ascend quantization for the w8a8 quantized weights.
- `--data-parallel-size 2` / `--tensor-parallel-size 8`: DP2 TP8 parallelism layout, recommended to balance memory capacity and compute efficiency for the w8a8 weights. For low-latency scenarios, use `dp1tp16` instead and turn off expert parallel, at the cost of lower throughput.
- `--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'`: Enables Multi-Token Prediction (MTP) speculative decoding with the DeepSeek-style MTP draft head of GLM-5.2. `num_speculative_tokens` (3-5) controls how many tokens are speculated per step; `enforce_eager: true` is required because GLM-5.2 does not support graph-mode speculative decoding.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables graph capture for the decode phase only, improving decode performance by reducing kernel launch overhead.
- `--async-scheduling`: Enables asynchronous scheduling to improve scheduling efficiency and overlap computation with scheduling decisions.

**`--additional-config` fields (Ascend-specific optimizations):**

- `"multistream_overlap_shared_expert": true`: Overlaps shared-expert computation on an additional stream, improving decode efficiency.


##### Multi-Node Co-Located Deployment

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

- `glm-5.2-w8a8`: can be deployed on 2 Atlas 800 A3 (64G × 16).

Run the following scripts on two nodes respectively.

**node 0**

```{code-block} bash
   :substitutions:
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_LAUNCH_BLOCKING=0

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8 \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 12980 \
--tensor-parallel-size 16 \
--seed 1024 \
--served-model-name glm-52 \
--max-num-seqs 48 \
--max-model-len 64000 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.93 \
--quantization ascend \
--enable-prefix-caching \
--async-scheduling \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"multistream_overlap_shared_expert":true}' \
--speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}'
```

**node 1**

```{code-block} bash
   :substitutions:
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_LAUNCH_BLOCKING=0

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8 \
--host 0.0.0.0 \
--port 8077 \
--headless \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-start-rank 1 \
--data-parallel-rpc-port 12980 \
--data-parallel-address $node0_ip \
--tensor-parallel-size 16 \
--seed 1024 \
--served-model-name glm-52 \
--max-num-seqs 48 \
--max-model-len 64000 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.93 \
--quantization ascend \
--enable-prefix-caching \
--async-scheduling \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"multistream_overlap_shared_expert":true}' \
--speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}'
```

Key Parameter Descriptions (in addition to [Single-Node Deployment](#single-node-deployment)):

**Multi-node network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`: Network interface configuration for multi-node communication. Set `nic_name` to the network interface name (obtained via `ifconfig`) and `local_ip` to the current node's IP address. These must be correctly configured on each node for successful multi-node communication.
- `--data-parallel-size 2`: Total number of data parallel ranks across all nodes (1 rank per node in this scenario).
- `--data-parallel-size-local 1`: Number of data parallel ranks on the current node.
- `--data-parallel-start-rank`: Starting rank offset for data parallel ranks on this node. Node 0 uses `0`, node 1 uses `1`.
- `--data-parallel-address`: IP address of the data parallel master node (node 0). Must match the `local_ip` of the master node.
- `--data-parallel-rpc-port 12980`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: Indicates a non-master node (used on node 1). Do not use on node 0.

##### Co-located Deployment on 4 Nodes (200k context)

In a co-located (mixed) deployment, prefill and decode run together on the same nodes, in contrast to the disaggregated setup below. The following templates deploy `GLM-5.2` across 4 nodes with `DP4 TP8` (`data-parallel-size-local=1` per node), a 200k context window, and MTP (`num_speculative_tokens=5`). Node 0 hosts the API server and is the DP master; Node 1 to Node 3 run with `--headless`. Prefix caching is disabled (`--no-enable-prefix-caching`) in this configuration. All IPs, NIC names, ports and weight paths are placeholders.

**Node 0** (API server / DP master):

```bash
#!/usr/bin/bash

nic_name="<NIC_NAME>"
local_ip=$(hostname -I | awk -F " " '{print $1}')
echo "$local_ip"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_RPC_TIMEOUT=360000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_EXEC_TIMEOUT=200
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200


vllm serve <MODEL_PATH> \
  --max_model_len 200000 \
  --max-num-batched-tokens 4096 \
  --served-model-name glm-52 \
  --seed 1024 \
  --api-server-count 1 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 32 \
  --data-parallel-size 4 \
  --data-parallel-size-local 1 \
  --data-parallel-address $local_ip \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --port 7000 \
  --safetensors-load-strategy 'prefetch' \
  --block-size 128 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --async-scheduling \
  --additional-config '{"multistream_overlap_shared_expert": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}'
```

**Node 1** (headless, `--data-parallel-start-rank 1`):

```bash
#!/usr/bin/bash

nic_name="<NIC_NAME>"
local_ip=$(hostname -I | awk -F " " '{print $1}')
node0_ip="<NODE0_IP>"
echo "$local_ip"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_RPC_TIMEOUT=360000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_EXEC_TIMEOUT=200
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200

vllm serve <MODEL_PATH> \
  --max_model_len 200000 \
  --max-num-batched-tokens 4096 \
  --headless \
  --served-model-name glm-52 \
  --seed 1024 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 32 \
  --safetensors-load-strategy 'prefetch' \
  --data-parallel-size 4 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address $node0_ip \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --port 7000 \
  --block-size 128 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --async-scheduling \
  --additional-config '{"multistream_overlap_shared_expert": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}'
```

Node 2 and Node 3 use the same script as Node 1, with `--data-parallel-start-rank` set to `2` and `3` respectively (and `node0_ip` pointing to Node 0).

##### Prefill-Decode Disaggregation

We'd like to show the deployment guide of `GLM-5` on multi-node environment with 1P1D for better performance.

Prefill-Decode disaggregation can be deployed on 4 Atlas 800 A3 (64G × 32).

Before you start, please

1. prepare the script `launch_online_dp.py` on each node:

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

2. prepare the script `run_dp_template.sh` on each node.

    To support a 200k context window on the stage of prefill, the parameter `"layer_sharding": ["q_b_proj"]` needs to be added to `--additional_config` on each prefill node.
    1. Prefill node 0

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400
        export ASCEND_AGGREGATE_ENABLE=1
        export ASCEND_TRANSPORT_PRINT=1
        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8 \
            --host 0.0.0.0 \
            --port $2 \
            --safetensors-load-strategy prefetch \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name glm-52 \
            --max-model-len 135000 \
            --speculative-config '{"num_speculative_tokens": 5, "method":"deepseek_mtp", "enforce_eager": true}' \
            --additional-config '{"enable_dsa_cp": true}' \
            --max-num-batched-tokens 4096 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --async-scheduling \
            --quantization ascend \
            --gpu-memory-utilization 0.95 \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "engine_id": "0",
            "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {
                                "dp_size": 2,
                                "tp_size": 16
                        },
                        "decode": {
                                "dp_size": 8,
                                "tp_size": 4
                        }
                }
            }'

        ```

    2. Prefill node 1

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400
        export ASCEND_AGGREGATE_ENABLE=1
        export ASCEND_TRANSPORT_PRINT=1
        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        
        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8 \
            --host 0.0.0.0 \
            --port $2 \
            --safetensors-load-strategy prefetch \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name glm-52 \
            --max-model-len 135000 \
            --speculative-config '{"num_speculative_tokens": 5, "method":"deepseek_mtp", "enforce_eager": true}' \
            --additional-config '{"enable_dsa_cp": true}' \
            --max-num-batched-tokens 4096 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --async-scheduling \
            --quantization ascend \
            --gpu-memory-utilization 0.95 \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "engine_id": "0",
            "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {
                                "dp_size": 2,
                                "tp_size": 16
                        },
                        "decode": {
                                "dp_size": 8,
                                "tp_size": 4
                        }
                }
            }'
        ```

    3. Decode node 0

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=500
        export ASCEND_AGGREGATE_ENABLE=1
        export ASCEND_TRANSPORT_PRINT=1
        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export DYNAMIC_EPLB=1
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export VLLM_ASCEND_ENABLE_MLAPO=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8 \
            --host 0.0.0.0 \
            --port $2 \
            --safetensors-load-strategy prefetch \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name glm-52 \
            --max-model-len 135000 \
            --max-num-batched-tokens 164 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --speculative-config '{"num_speculative_tokens": 5, "method":"deepseek_mtp", "enforce_eager": true}' \
            --additional-config '{"recompute_scheduler_enable": true}' \
            --trust-remote-code \
            --max-num-seqs 48 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "30100",
            "engine_id": "1",
            "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {
                                "dp_size": 2,
                                "tp_size": 16
                        },
                        "decode": {
                                "dp_size": 8,
                                "tp_size": 4
                        }
                }
            }'
        ```

    4. Decode node 1

         ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip
            
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=500
        export ASCEND_AGGREGATE_ENABLE=1
        export ASCEND_TRANSPORT_PRINT=1
        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export DYNAMIC_EPLB=1
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export VLLM_ASCEND_ENABLE_MLAPO=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8 \
            --host 0.0.0.0 \
            --port $2 \
            --safetensors-load-strategy prefetch \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name glm-52 \
            --max-model-len 135000 \
            --max-num-batched-tokens 164 \
            --speculative-config '{"num_speculative_tokens": 5, "method":"deepseek_mtp", "enforce_eager": true}' \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"recompute_scheduler_enable": true}' \
            --trust-remote-code \
            --max-num-seqs 48 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "30100",
            "engine_id": "1",
            "kv_connector_extra_config": {
                         "use_ascend_direct": true,
                         "prefill": {
                                 "dp_size": 2,
                                 "tp_size": 16
                         },
                         "decode": {
                                 "dp_size": 8,
                                 "tp_size": 4
                         }
                 }
             }'
         ```

Once the preparation is done, you can start the server with the following command on each node:

1. Prefill node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 2 --tp-size 16  --dp-size-local 1 --dp-rank-start 0 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    ```

2. Prefill node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 2 --tp-size 16  --dp-size-local 1 --dp-rank-start 1 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    ```

3. Decode node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    ```

4. Decode node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 4 --dp-rank-start 4 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    ```

To set up request forwarding, run the following script on any machine. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
unset http_proxy
unset https_proxy

python load_balance_proxy_server_example.py \
    --port 8000 \
    --host 0.0.0.0 \
    --prefiller-hosts \
      $node_p0_ip \
      $node_p1_ip \
    --prefiller-ports \
      9081 9081 \
    --decoder-hosts \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
    --decoder-ports \
      9900 9901 9902 9903 \
      9900 9901 9902 9903
```

Key Parameter Descriptions (in addition to [Single-Node Deployment](#single-node-deployment) and [Multi-Node Co-Located Deployment](#multi-node-co-located-deployment)):

**`launch_online_dp.py` parameters:**

|Parameter|Type|Required|Default|Description|
|---------|----|--------|-------|-----------|
|`--dp-size`|int|Yes|-|Data parallel size (total number of DP ranks across all nodes).|
|`--tp-size`|int|No|1|Tensor parallel size within each DP rank.|
|`--dp-size-local`|int|No|(same as `--dp-size`)|Number of DP ranks on the current node. If not set, defaults to `--dp-size`.|
|`--dp-rank-start`|int|No|0|Starting rank offset for data parallel ranks on this node.|
|`--dp-address`|str|Yes|-|IP address of the data parallel master node (node 0).|
|`--dp-rpc-port`|str|No|12345|RPC port for data parallel master communication.|
|`--vllm-start-port`|int|No|9000|Starting port for each vLLM engine instance on this node. Each DP rank's engine port = `vllm_start_port` + local rank index.|

**Prefill node-specific configurations:**

- `VLLM_ASCEND_ENABLE_FUSED_MC2=1`: Enables the `dispatch_ffn_combine`/`mega_moe` fused operators on prefill and decode nodes. 
- `ASCEND_A3_ENABLE=1`: A3-specific communication aggregation optimization (`ACL_OP_INIT_MODE=1` is described in [Multi-Node Deployment](#multi-node-co-located-deployment)).
- `--speculative-config '{"num_speculative_tokens": 5, ...}'`: MTP speculation on both prefill and decode nodes in the templates above (the parameter itself is described in [Single-Node Deployment](#single-node-deployment)).

**Decode node-specific configurations:**

- `--max-num-batched-tokens 164`: Small batch token limit on decode nodes — decode processes one token per sequence per step, so batch tokens should be close to `max-num-seqs`.
- `--additional-config '{"recompute_scheduler_enable": true}'`: Enables the recomputation scheduler. When the decode node KV cache is insufficient, requests are sent back to the prefill node to recompute the KV cache. Recommended on both prefill and decode nodes in PD scenarios.
- `DYNAMIC_EPLB=1`: Enables dynamic expert-parallel load balancing on decode nodes.

**Mooncake KV transfer configuration (`--kv-transfer-config`):**

- `"kv_connector": "MooncakeConnectorV1"`: Uses Mooncake as the KV cache transfer connector between prefill and decode nodes.
- `"kv_role": "kv_producer"` / `"kv_consumer"`: `kv_producer` on prefill nodes, `kv_consumer` on decode nodes.
- `"kv_port"`: Port for Mooncake KV transfer communication. Use different port ranges for prefill (`30000`) and decode (`30100`) node groups.
- `"use_ascend_direct": true`: Enables Ascend direct transfer for KV cache, reducing latency.
- `"prefill"` / `"decode"` sections: Specify the `dp_size` and `tp_size` of the prefill and decode node groups respectively. These must match the actual deployment topology (`prefill: dp2 tp16`, `decode: dp8 tp4` in this scenario).

**Request forwarding (proxy):**

- The proxy command maps every prefill engine endpoint (2 prefiller hosts × 1 port per node) and every decode engine endpoint (8 decoder hosts/ports) to a single entry point on port `8000`.
- The proxy program can be found in [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).

Please refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py) for further explanation and restrictions of the environment variables above.

#### Atlas 800 A2

##### Multi-Node Co-Located Deployment (A2)

- `glm-5.2-w8a8`: can be deployed on 2 Atlas 800 A2 (64G × 32).

**node 0**

```{code-block} bash
   :substitutions:
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_RPC_TIMEOUT=360000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_EXEC_TIMEOUT=200
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
#export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
#export USE_MULTI_GROUPS_KV_CACHE=1
#export USE_MULTI_BLOCK_POOL=1
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8 \
--max_model_len 40000 \
--max-num-batched-tokens 4096 \
--served-model-name glm-52 \
--seed 1024 \
--gpu-memory-utilization 0.95 \
--api-server-count 1 \
--max-num-seqs 16 \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 8 \
--enable-expert-parallel \
--quantization ascend \
--port 7000 \
--safetensors-load-strategy 'prefetch' \
--block-size 128 \
--async-scheduling \
--additional-config '{"multistream_overlap_shared_expert": true}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}'
```

**node 1**

```{code-block} bash
   :substitutions:
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_RPC_TIMEOUT=360000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_EXEC_TIMEOUT=200
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
#export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
#export USE_MULTI_GROUPS_KV_CACHE=1
#export USE_MULTI_BLOCK_POOL=1
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8 \
--max_model_len 40000 \
--max-num-batched-tokens 4096 \
--served-model-name glm-52 \
--seed 1024 \
--gpu-memory-utilization 0.95 \
--max-num-seqs 16 \
--headless \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-start-rank 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 8 \
--enable-expert-parallel \
--quantization ascend \
--port 7000 \
--safetensors-load-strategy 'prefetch' \
--block-size 128 \
--async-scheduling \
--additional-config '{"multistream_overlap_shared_expert": true}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}'
```

Key Parameter Descriptions (in addition to [Single-Node Deployment](#single-node-deployment) and [Multi-Node Co-Located Deployment](#multi-node-co-located-deployment)):

The A2 series uses a different optimization stack than A3: FlashComm1 and DSA-CP are not used here.

**A2-specific environment variables:**

- `CPU_AFFINITY_CONF=1`: Enables CPU core affinity binding for worker processes.
- `ACL_OP_INIT_MODE=1`: ACL operator initialization mode to speed up operator compilation.
- `VLLM_RPC_TIMEOUT=360000` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000` / `HCCL_EXEC_TIMEOUT=200` / `HCCL_CONNECT_TIMEOUT=120` / `VLLM_ENGINE_READY_TIMEOUT_S=1200`: Timeout settings for multi-node startup and model execution on the slower A2 platform. Increase them if the engine fails to become ready in time.

##### Prefill-Decode Disaggregation (A2)

On Atlas 800 A2, where each node exposes 8 cards, the same global P/D topology (Prefill `DP4 TP8`, Decode `DP8 TP4`) is split across 8 nodes: 4 prefill nodes hosting 1 DP rank each (8 cards per rank), and 4 decode nodes hosting 2 DP ranks each (4 cards per rank). The `launch_online_dp.py` above is reused as-is. The prefill side enables FlashComm1 and DSA CP; the decode side enables MLAPO and `DYNAMIC_EPLB` with a `FULL_DECODE_ONLY` graph. Both sides enable prefix caching and MTP (`num_speculative_tokens=3`). All IPs, NIC names, ports and weight paths below are placeholders.

`run_dp_template.sh` for the prefill nodes:

```bash
#!/usr/bin/bash
nic_name="<NIC_NAME>"
local_ip="<CURRENT_NODE_IP>"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_HOST_IP=$local_ip
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=256
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
export ASCEND_RT_VISIBLE_DEVICES=$1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve <MODEL_PATH> \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name glm-52 \
  --max-model-len 115168 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.95 \
  --quantization ascend \
  --async-scheduling \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --kv-transfer-config \
  '{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector",
    "kv_connector_extra_config": {
      "use_ascend_direct": true,
      "prefill": {
        "dp_size": 4,
        "tp_size": 8
      },
      "decode": {
        "dp_size": 8,
        "tp_size": 4
      }
    }
  }' \
  --additional-config \
  '{
    "multistream_overlap_shared_expert": true,
    "enable_dsa_cp": true
  }' \
  --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp", "enforce_eager": true}'
```

`run_dp_template.sh` for the decode nodes:

```bash
#!/usr/bin/bash

nic_name="<NIC_NAME>"
local_ip="<CURRENT_NODE_IP>"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_HOST_IP=$local_ip
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=500
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export DYNAMIC_EPLB=1
export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve <MODEL_PATH> \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name glm-52 \
  --max-model-len 135168 \
  --max-num-batched-tokens 164 \
  --trust-remote-code \
  --max-num-seqs 48 \
  --gpu-memory-utilization 0.92 \
  --async-scheduling \
  --quantization ascend \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --kv-transfer-config \
  '{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_consumer",
    "kv_port": "30100",
    "engine_id": "1",
    "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector",
    "kv_connector_extra_config": {
      "use_ascend_direct": true,
      "prefill": {
        "dp_size": 4,
        "tp_size": 8
      },
      "decode": {
        "dp_size": 8,
        "tp_size": 4
      }
    }
  }' \
  --compilation-config \
  '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config \
  '{
    "multistream_overlap_shared_expert": true,
    "recompute_scheduler_enable": true
  }' \
  --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp", "enforce_eager": true}'
```

Once the preparation is done, start the server with the following commands:

1. Prefill nodes — run on `$node_p0_ip`, `$node_p1_ip`, `$node_p2_ip`, `$node_p3_ip` with `--dp-rank-start` `0/1/2/3`:

    ```shell
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 2 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 3 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    ```

2. Decode nodes — run on `$node_d0_ip`, `$node_d1_ip`, `$node_d2_ip`, `$node_d3_ip` with `--dp-rank-start` `0/2/4/6`:

    ```shell
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 0 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 2 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 4 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 6 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    ```

For request forwarding on this 8-node A2 layout, use 4 prefiller hosts (1 endpoint each) and 4 decoder hosts (2 endpoints each) in the Request Forwarding command below.

To set up request forwarding, run the following script on any machine. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
unset http_proxy
unset https_proxy

python load_balance_proxy_server_example.py \
    --port 8000 \
    --host 0.0.0.0 \
    --prefiller-hosts \
      $node_p0_ip \
      $node_p1_ip \
      $node_p2_ip \
      $node_p3_ip \
    --prefiller-ports \
      9081 9081 \
      9081 9081 \
    --decoder-hosts \
      $node_d0_ip \
      $node_d0_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d2_ip \
      $node_d2_ip \
      $node_d3_ip \
      $node_d3_ip \
    --decoder-ports \
      9900 9901 9900 9901 \
      9900 9901 9900 9901
```

Key Parameter Descriptions (in addition to [Prefill-Decode Disaggregation](#prefill-decode-disaggregation)):

This 8-node A2 layout splits the global P/D topology across 8 Atlas 800 A2 nodes: 4 prefill nodes hosting 1 DP rank each (8 cards per rank, `DP4 TP8`) and 4 decode nodes hosting 2 DP ranks each (4 cards per rank, `DP8 TP4`). The `launch_online_dp.py` script is the same as in [Prefill-Decode Disaggregation](#prefill-decode-disaggregation).

**Decode node-specific configurations (A2):**

- `VLLM_ASCEND_ENABLE_MLAPO=1`: MLAPO fusion on decode nodes for memory-bandwidth-bound token generation.

**Common environment variables (A2 PD):**

- `VLLM_HOST_IP=$local_ip`: Host IP advertised by the engine for cross-node communication.
- `ASCEND_AGGREGATE_ENABLE=1` / `ASCEND_TRANSPORT_PRINT=1`: Communication aggregation and transport debug printing.
- `VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000`: Timeout for automatically releasing the prefiller's KV cache for a particular request.

**Mooncake KV transfer configuration (`--kv-transfer-config`):**

- The configuration is the same as [Prefill-Decode Disaggregation](#prefill-decode-disaggregation), with topology `prefill: dp4 tp8` / `decode: dp8 tp4`. A2-specific fields: `"kv_connector": "MooncakeConnector"` with `"kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector"` and `"engine_id"` (`"0"` for prefill, `"1"` for decode).

Please refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py) for further explanation and restrictions of the environment variables above.

### 1M Context Deployment

Recommended configurations for serving `GLM-5.2` with a 1M context window on Atlas 800 A3 (64G x 16) and quantized GLM-5.2(W4A8C8) weights:

| Mode | Hardware | Parallelism | Context |
| ---- | -------- | ----------- | ------- |
| Single-node co-located | 1 Atlas 800 A3 (64G x 16) | `DP1 PP1 TP16 PCP1 DCP16` | `1024000` |
| Dual-node co-located | 2 Atlas 800 A3 (64G x 16) | `DP4 PP1 TP8 PCP1 DCP8` | `1024000` |
| 1P1D PD disaggregation | 1 prefiller with 2 A3 nodes + 1 decoder with 2 A3 nodes | Prefill `DP4 PP1 TP8 PCP1 DCP8`, Decode `DP4 PP1 TP8 PCP1 DCP8` | `1024000` |

#### Atlas 800 A3

##### Single-Node 1M Deployment

Recommended command:

```shell
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=20
export HCCL_BUFFSIZE=768
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1

vllm serve <MODEL_PATH> \
  --seed 1024 \
  --host 0.0.0.0 \
  --port 9000 \
  --served-model-name glm-52 \
  --max-model-len 1024000 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.80 \
  --api-server-count 1 \
  --max-num-seqs 32 \
  --data-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 16 \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size 16 \
  --cp-kv-cache-interleave-size 128 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 16, 128]}' \
  --additional-config '{"enable_flashcomm1": true, "enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true, "multistream_overlap_shared_expert": true,"enable_sparse_c8": true, "enable_cpu_binding": true}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
  --quantization ascend \
  --enable-expert-parallel \
  --safetensors-load-strategy prefetch
```

Key Parameter Descriptions (in addition to [Single-Node Deployment](#single-node-deployment)):

**1M-specific environment variables:**

- `VLLM_ASCEND_ENABLE_NZ=1`: Enables NZ format memory layout for the C8 quantized tensors, required for the 1M context deployment.
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`: Uses the spawn start method for multi-process workers (required in this scenario).

**1M-specific vllm serve parameters:**

- `--data-parallel-size 1` / `--pipeline-parallel-size 1` / `--tensor-parallel-size 16`: Single-node parallelism layout `DP1 PP1 TP16`.
- `--prefill-context-parallel-size 1` / `--decode-context-parallel-size 16`: Decode context parallelism (DCP) of 16 for the decode phase; prefill uses PCP 1.
- `--cp-kv-cache-interleave-size 128`: KV cache interleave size for context parallelism.

##### Dual-Node Co-Located 1M Deployment

Recommended command for both co-located nodes:

```shell
nic_name="<NIC_NAME>"
local_ip="<CURRENT_NODE_IP>"
node_0_ip="<NODE0_IP>"
# Node 0: data_parallel_start_rank=0, server_role_args="--api-server-count 1"
# Node 1: data_parallel_start_rank=2, server_role_args="--headless"
data_parallel_start_rank=0
server_role_args="--api-server-count 1"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=20
export HCCL_BUFFSIZE=768
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1

vllm serve <MODEL_PATH> \
  --seed 1024 \
  --host 0.0.0.0 \
  --port 9000 \
  --served-model-name glm-52 \
  --max-model-len 1024000 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.75 \
  ${server_role_args} \
  --max-num-seqs 8 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank $data_parallel_start_rank \
  --data-parallel-address $node_0_ip \
  --data-parallel-rpc-port 16591 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 8 \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 128 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_flashcomm1": true, "enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true}, "multistream_overlap_shared_expert": true,"enable_sparse_c8": true, "enable_cpu_binding": true}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
  --quantization ascend \
  --enable-expert-parallel \
  --safetensors-load-strategy prefetch
```

Key Parameter Descriptions (in addition to [Single-Node 1M Deployment](#single-node-1m-deployment)):

**Multi-node configuration:**

- `--data-parallel-size 4` / `--data-parallel-size-local 2` / `--data-parallel-start-rank`: Node 0 uses `data_parallel_start_rank=0` with `server_role_args="--api-server-count 1"`; node 1 uses `data_parallel_start_rank=2` with `server_role_args="--headless"`.
- `--data-parallel-address $node_0_ip` / `--data-parallel-rpc-port 16591`: Data parallel master node IP and RPC port, identical on both nodes.
- `--tensor-parallel-size 8` / `--decode-context-parallel-size 8`: `DP4 TP8 DCP8` layout — each node hosts 2 DP ranks × TP8.

##### PD Disaggregation 1M Deployment

Recommended command for both prefiller nodes:

```shell
nic_name="<NIC_NAME>"
local_ip="<CURRENT_PREFILL_NODE_IP>"
node_p0_ip="<PREFILL_NODE0_IP>"
# Prefiller node 0: data_parallel_start_rank=0, server_role_args="--api-server-count 1"
# Prefiller node 1: data_parallel_start_rank=2, server_role_args="--headless"
data_parallel_start_rank=0
server_role_args="--api-server-count 1"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=20
export HCCL_BUFFSIZE=768
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480

vllm serve <MODEL_PATH> \
  --seed 1024 \
  --host 0.0.0.0 \
  --port 9081 \
  --served-model-name glm-52 \
  --max-model-len 1024000 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.75 \
  ${server_role_args} \
  --max-num-seqs 8 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank $data_parallel_start_rank \
  --data-parallel-address $node_p0_ip \
  --data-parallel-rpc-port 16591 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 8 \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 128 \
  --enforce-eager \
  --additional-config '{"enable_flashcomm1": true, "enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true},"multistream_overlap_shared_expert": true,"enable_sparse_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": true}' \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": true}' \
  --quantization ascend \
  --enable-expert-parallel \
  --safetensors-load-strategy prefetch \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
      "use_ascend_direct": true,
      "prefill": {
        "dp_size": 4,
        "tp_size": 8
      },
      "decode": {
        "dp_size": 4,
        "tp_size": 8
      }
    }
  }'
```

Recommended command for both decoder nodes:

```shell
nic_name="<NIC_NAME>"
local_ip="<CURRENT_DECODE_NODE_IP>"
node_d0_ip="<DECODE_NODE0_IP>"
# Decoder node 0: data_parallel_start_rank=0, server_role_args="--api-server-count 1"
# Decoder node 1: data_parallel_start_rank=2, server_role_args="--headless"
data_parallel_start_rank=0
server_role_args="--api-server-count 1"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=20
export HCCL_BUFFSIZE=768
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480

vllm serve <MODEL_PATH> \
  --seed 1024 \
  --host 0.0.0.0 \
  --port 9900 \
  --served-model-name glm-52 \
  --max-model-len 1024000 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.75 \
  ${server_role_args} \
  --max-num-seqs 32 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank $data_parallel_start_rank \
  --data-parallel-address $node_d0_ip \
  --data-parallel-rpc-port 16600 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 8 \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 128 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_flashcomm1": true, "enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true}, "multistream_overlap_shared_expert": true, "enable_sparse_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": true}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
  --quantization ascend \
  --enable-expert-parallel \
  --safetensors-load-strategy prefetch \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_consumer",
    "kv_port": "30100",
    "engine_id": "1",
    "kv_connector_extra_config": {
      "use_ascend_direct": true,
      "prefill": {
        "dp_size": 4,
        "tp_size": 8
      },
      "decode": {
        "dp_size": 4,
        "tp_size": 8
      }
    }
  }'
```

Key Parameter Descriptions (in addition to [Prefill-Decode Disaggregation](#prefill-decode-disaggregation) and [Single-Node 1M Deployment](#single-node-1m-deployment)):

**Prefill nodes (1M):**

- `recompute_scheduler_enable` is enabled on both prefill and decode nodes in this scenario (see [Prefill-Decode Disaggregation](#prefill-decode-disaggregation) for the parameter description).
- `--kv-transfer-config`: Mooncake as `kv_producer` (`kv_port: 30000`) with topology `prefill: dp4 tp8` / `decode: dp4 tp8` (matching the 1M P/D topology, in contrast to `dp2 tp16`/`dp8 tp4` in the sub-1M PD scenario).

**Decode nodes (1M):**

- `--max-num-batched-tokens 16384`: Decode nodes store the large 1M KV cache received from prefill nodes.
- `--kv-transfer-config`: Mooncake as `kv_consumer` (`kv_port: 30100`), same configuration as [Prefill-Decode Disaggregation](#prefill-decode-disaggregation).

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-52",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

### Using Language Model Evaluation Harness

Not tested yet.

## Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## Performance Tuning

### Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to [Tuning Guidelines](#tuning-guidelines) for tuning based on actual conditions.

The tables below provide recommended parameter configurations for different deployment scenarios. All scenarios are categorized by use case (Low Latency, High Throughput, Long Context) and correspond to the deployment modes documented in [Deployment](#deployment).

#### Table 1: Scenario Overview

> `*Total NPUs` indicates the total number of NPUs used across all nodes. 1 node = 1 Atlas 800 A3 server (64G × 16 NPUs) or 1 Atlas 800 A2 server (64G × 8 NPUs).

|Scenario|Deployment Mode|*Total NPUs|Weight Version|Key Considerations|
|--------|---------------|-----------|--------------|------------------|
|Low Latency<br>(64K input)|Dual-Node Co-Located (A3), [Multi-Node Co-Located Deployment](#multi-node-co-located-deployment)|32 (A3)|w4a8c8|dp2 tp16, MTP3, max-num-seqs 8, max-model-len 135000, FlashComm1, DSA CP|
|Low Latency<br>(128K input)|Dual-Node Co-Located (A3), [Multi-Node Co-Located Deployment](#multi-node-co-located-deployment)|32 (A3)|w4a8c8|dp2 tp16, MTP3, max-num-seqs 8, max-model-len 135000, FlashComm1, DSA CP|
|High Throughput<br>(64K input)|Dual-Node Co-Located (A3), [Multi-Node Co-Located Deployment](#multi-node-co-located-deployment)|32 (A3)|w4a8c8|dp4 tp8, fused MC2, MTP3, max-num-seqs 16, max-model-len 66000, FlashComm1, DSA CP|
|High Throughput|PD Disaggregation (A3), [Prefill-Decode Disaggregation](#prefill-decode-disaggregation)|4 nodes (A3)|w4a8c8|P: dp4 tp8 (max-num-seqs 64, max-num-batched-tokens 8192, MTP1); D: dp32 tp1 (max-num-seqs 32, max-num-batched-tokens 164, MTP5), max-model-len 133120, dedicated P/D nodes, Mooncake KV transfer|
|Long Context<br>(1M)|PD Disaggregation (A3), [PD Disaggregation 1M Deployment](#pd-disaggregation-1m-deployment)|4 nodes (A3)|w4a8c8|P/D: dp4 tp8, DCP8, max-model-len 1040000, Mooncake KV transfer|

#### Table 2: Detailed Node Configuration

> The TP/DP columns show the values **per node** as configured in the Deployment scripts (a node hosting 2 DP ranks of TP8, or 1 DP rank of TP16, uses 16 NPUs).

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#deployment)** chapter.

|Scenario|Configuration|NPUs (per node)|TP|DP (per node)|Max Num Seqs|Max Num Batched Tokens|Max Model Len|MTP Spec Num|
|--------|-------------|-----|--|--|------------|----------------------|--------------|-------------|
|Low Latency 64K (A3)|Dual-Node (per node), [Multi-Node Co-Located Deployment](#multi-node-co-located-deployment)|16|16|1|8|8192|135000|3|
|Low Latency 128K (A3)|Dual-Node (per node), [Multi-Node Co-Located Deployment](#multi-node-co-located-deployment)|16|16|1|8|8192|135000|3|
|High Throughput 64K (A3)|Dual-Node (per node), [Multi-Node Co-Located Deployment](#multi-node-co-located-deployment)|16|8|2|16|8192|66000|3|
|High Throughput (A3)|PD — Server-P Node, [Prefill-Decode Disaggregation](#prefill-decode-disaggregation)|16|8|2|64|8192|133120|1|
|High Throughput (A3)|PD — Server-D Node, [Prefill-Decode Disaggregation](#prefill-decode-disaggregation)|16|1|16|32|164|133120|5|
|Long Context 1M (A3)|PD — Server-P Node, [PD Disaggregation 1M Deployment](#pd-disaggregation-1m-deployment)|16|8|2|8|16384|1040000|1|
|Long Context 1M (A3)|PD — Server-D Node, [PD Disaggregation 1M Deployment](#pd-disaggregation-1m-deployment)|16|8|2|32|128|1040000|3|

> On PD decode nodes, `--max-num-batched-tokens` is configured as `(MTP Spec Num + 1) × Max Num Seqs` — each sequence generates one target token plus the speculated MTP tokens per decode step.
>
> For complete startup commands and detailed parameter descriptions, please refer to the deployment examples and Key Parameter Descriptions in [Deployment](#deployment).

#### Table 3: Performance-Related Parameter Tuning Guide

|Parameter|Low Latency|High Throughput|Long Context|Description|
|---------|-----------|---------------|-------------|-----------|
|`--max-num-batched-tokens`|Lower (4096–8192)|Higher for prefill (8192)|Higher for prefill (16384)|Controls batch size per step. Lower values reduce per-step latency; higher values improve prefill throughput.|
|`--gpu-memory-utilization`|0.92–0.95|0.90–0.95|0.75–0.93|NPU memory fraction. 1M scenarios must reserve memory for the huge KV cache, so use lower values (0.75–0.80 on prefill/co-located).|
|`num_speculative_tokens` (MTP)|3–5|3–5|1 (prefill) / 3 (decode)|MTP speculation count. Higher values improve decode throughput at the cost of memory for the draft model KV cache. Use `1` on prefill nodes in PD mode.|
|`enable_dsa_cp`|Optional|Enable (prefill nodes)|Enable (prefill nodes)|DSA context parallelism accelerates long-context prefill. Decoupled from FlashComm1 since v0.21.0.|
|`enable_sparse_sfa_c8` / `enable_sparse_li_c8`|`false` / `true`|`true` / `true` (PD, 1M)|`true` / `true`|SFA optimizations for the C8 quantized model. `enable_sparse_li_c8` is always recommended; `enable_sparse_sfa_c8` (SFA DCP, since v0.23.0) benefits long-context prefill.|
|`enable_balance_scheduling`|Enable (single-node)|Enable|Disable in PD mode|Improves output throughput and reduces TPOT in the v1 scheduler. TTFT may degrade in some scenarios; not recommended when Prefill-Decode is separated.|
|`VLLM_ASCEND_ENABLE_FLASHCOMM1`|1|1 (prefill nodes)|1 (prefill nodes)|Communication optimization. Conflicts with `layer_sharding` containing `o_proj`.|
|`VLLM_ASCEND_ENABLE_MLAPO`|—|1 (A2 P/D nodes)|—|Fusion operator that significantly improves performance but consumes more NPU memory. On A3 used on decode nodes only.|
|`cudagraph_mode`|FULL_DECODE_ONLY|FULL_DECODE_ONLY (decode)|FULL_DECODE_ONLY (decode)|Graph capture for the decode phase only. Prefill nodes in PD mode use `--enforce-eager` instead.|

#### Common Parameters

Common parameters used across all deployment scenarios are described here once:

- `--enable-chunked-prefill` / `--enable-prefix-caching`: Recommended for long-context and multi-user scenarios — chunked prefill splits long prompts to improve TTFT; prefix caching reuses KV cache for shared prefixes and reduces KV transfer volume in PD scenarios.
- `--enforce-eager`: Disables graph capture. Required on prefill nodes in PD scenarios (PD separation with MTP requires eager mode on the prefill path).
- `HCCL_BUFFSIZE`: HCCL communication buffer size. Larger values improve communication throughput at the cost of additional memory: `200` (single-node), `400` (multi-node A3), `256`–`2560` (A2), `768` (1M context).
- `OMP_NUM_THREADS` / `OMP_PROC_BIND`: CPU thread settings to avoid thread oversubscription — `1` by default, `10` on A2, `20` for the 1M context workload.
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`: Reduces NPU memory fragmentation.
- `TASK_QUEUE_ENABLE=1`: Enables the task queue for more efficient batch scheduling (used on decode nodes, A2, and 1M scenarios).
- `--api-server-count 1` / `--safetensors-load-strategy 'prefetch'` / `--seed 1024` / `--trust-remote-code`: Number of OpenAI-compatible API server instances, weight prefetching for faster startup, fixed seed, and remote code loading.

### Tuning Guidelines

For general performance tuning methods, refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md).

For detailed feature descriptions and configuration options, refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md).

For environment variable descriptions and constraints, refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py).

## FAQ

- **Q: How to enable function calling for GLM-5.2?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```
