# GLM-5.3 (Experimental)

## 1 Introduction

[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) uses the same base model as GLM-5.2 — every gain comes from post-training. Compared with GLM-5.2, it is much better at complex coding and long-horizon tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, multi-node deployment, accuracy and performance evaluation.

!!! warning

    **Current status and constraints**

    - The multi-node co-located examples were tested on the official Docker images `quay.io/ascend/vllm-ascend:v0.23.0-a3` and `quay.io/ascend/vllm-ascend:v0.23.0`. The [Prefill-Decode disaggregation example](#52-prefill-decode-disaggregation) uses a separate **0.30.0RC reference configuration** with PP2 and optional MemCache KV pooling on A3.
    - The features listed in [Supported Features](#2-supported-features) are only those enabled by the verified deployment commands in this document, and do **not** imply that all features are supported for GLM-5.3. This is an early-access version; performance optimization and reliability validation are still in progress (see [Declaration](#10-declaration)).
    - The co-located scripts are based on **v0.23.0**; Section 5.2 has its own version requirements. Check configuration compatibility before using either example with another release or the main branch.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

|  Weight Version          | Hardware Requirements                                         | Download Links |
|--------------------------|---------------------------------------------------------------|----------------|
|  `GLM-5.3-w8a8c8`        | 2 Atlas 800 A3 (128GB × 8) node or 4 Atlas 800 A2 (64GB × 8) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-w8a8c8) |

- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

- You can use our official docker image to run GLM-5.3 directly.

=== "A3 series"

    Start the docker image on each node.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
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

=== "A2 series"

    Start the docker image on each of your nodes.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
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

If you want to deploy multi-node environment, you need to set up environment on each node.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md).

## 5 Online Service Deployment

### 5.1 Multi-node Deployment

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

!!! warning

    - The scripts in Section 5.1 were tested on **v0.23.0**. Parameters may have changed in the main branch. For Section 5.2, follow the version requirements in that section.

Common Issues Tip: If you encounter issues, Refer to [Public FAQs](../../faqs.md).

#### 5.1.1 Context Below 1M

=== "A3 series"
    -  `GLM-5.3-w8a8c8`: can be deployed on 2 Atlas 800 A3 (64GB × 16).

    Run the following scripts on two nodes respectively.

    **node 0**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="xxx"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_TRANSFER_TIMEOUT=600
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=3600
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_MLAPO=1

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --safetensors-load-strategy prefetch \
        --api-server-count 1 \
        --data-parallel-size 8 \
        --data-parallel-start-rank 0 \
        --data-parallel-size-local 4 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm-5 \
        --tool-call-parser glm47 \
        --reasoning-parser glm47 \
        --enable-auto-tool-choice \
        --max-num-seqs 6 \
        --max-model-len 202752 \
        --max-num-batched-tokens 4096 \
        --trust-remote-code \
        --gpu-memory-utilization 0.90 \
        --quantization ascend \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "enable_fused_mc2": 1, "enable_flashcomm1": true}'  \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

    **node 1**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="xxx"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_TRANSFER_TIMEOUT=600
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=3600
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_MLAPO=1

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --headless \
        --data-parallel-size 8 \
        --data-parallel-start-rank 4 \
        --data-parallel-size-local 4 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm-5 \
        --tool-call-parser glm47 \
        --reasoning-parser glm47 \
        --enable-auto-tool-choice \
        --max-num-seqs 6 \
        --max-model-len 202752 \
        --max-num-batched-tokens 4096 \
        --trust-remote-code \
        --gpu-memory-utilization 0.92 \
        --quantization ascend \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --async-scheduling \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "enable_fused_mc2": 1, "enable_flashcomm1": true}' \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

=== "A2 series"

    - `GLM-5.3-w8a8c8`: can be deployed on 4 Atlas 800 A2 (64GB × 8).

    Run the following scripts on four nodes respectively.

    **node 0**

    ```shell
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
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=200
    export HCCL_CONNECT_TIMEOUT=120
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ACL_OP_INIT_MODE=1
    export CPU_AFFINITY_CONF=1
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1200

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --max-model-len 135000 \
        --data-parallel-size 4 \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank 0 \
        --data-parallel-address "${node0_ip}" \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm-5 \
        --reasoning-parser glm47 \
        --safetensors-load-strategy prefetch \
        --max-num-seqs 128 \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --quantization ascend \
        --gpu-memory-utilization 0.92 \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_flashcomm1": true}' \
        --enable-prefix-caching \
        --async-scheduling \
        --api-server-count 1
    ```

    **node 1-3**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="xxx"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxx"

    # node1: dp_start_rank=1, node2: dp_start_rank=2, node3: dp_start_rank=3
    dp_start_rank=1

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export VLLM_RPC_TIMEOUT=360000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=200
    export HCCL_CONNECT_TIMEOUT=120
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ACL_OP_INIT_MODE=1
    export CPU_AFFINITY_CONF=1
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1200

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --headless \
        --max-model-len 135000 \
        --data-parallel-size 4 \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank ${dp_start_rank} \
        --data-parallel-address "${node0_ip}" \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm-5 \
        --reasoning-parser glm47 \
        --safetensors-load-strategy prefetch \
        --max-num-seqs 128 \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --quantization ascend \
        --gpu-memory-utilization 0.92 \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true,"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_flashcomm1": true}' \
        --enable-prefix-caching \
        --async-scheduling
    ```

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. max-model-len and max-num-seqs need to be set according to the actual usage scenario.

**Multi-node network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`: Network interface configuration for multi-node communication. Set `nic_name` to the network interface name (obtained via `ifconfig`) and `local_ip` to the current node's IP address. These must be correctly configured on each node for successful multi-node communication.
- `--data-parallel-size 8`: Total number of data parallel ranks across all nodes (4 ranks per node in this scenario).
- `--data-parallel-size-local 4`: Number of data parallel ranks on the current node.
- `--data-parallel-start-rank`: Starting rank offset for data parallel ranks on this node. Node 0 uses `0`, node 1 uses `4`.
- `--data-parallel-address`: IP address of the data parallel master node (node 0). Must match the `local_ip` of the master node.
- `--data-parallel-rpc-port 12980`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: Indicates a non-master node (used on node 1-3). Do not use on node 0.

**A2-specific environment variables:**

- `CPU_AFFINITY_CONF=1`: Enables CPU core affinity binding for worker processes.
- `ACL_OP_INIT_MODE=1`: ACL operator initialization mode to speed up operator compilation.
- `VLLM_RPC_TIMEOUT=360000` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000` / `HCCL_EXEC_TIMEOUT=200` / `HCCL_CONNECT_TIMEOUT=120` / `VLLM_ENGINE_READY_TIMEOUT_S=1200`: Timeout settings for multi-node startup and model execution on the slower A2 platform. Increase them if the engine fails to become ready in time.

**Notice:**
This scenario enables `additional_config.enable_fused_mc2=1` (fused `dispatch_ffn_combine`/`mega_moe` operators). Fused MC2 conflicts with `multistream_overlap_shared_expert` — the two optimizations must not be enabled at the same time (the runtime forcibly disables `multistream_overlap_shared_expert` when fused MC2 is on).

#### 5.1.2 1M Context Deployment

The 1M context scenarios have not yet tested for `GLM-5.3`. If you want to deploy, please refer to scripts in [GLM-5.2 1M Context Deployment](https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/models/GLM5.2.html#m-context-deployment).

### 5.2 Prefill-Decode Disaggregation

This A3 reference serves GLM-5.3 W8A8C8 with a **200,000-token** maximum
sequence length for the 198K workload. It uses one Prefill engine spanning
P0/P1 with TP16 and PP2, and sixteen TP2 Decode engines on D0/D1.
Section 5.3 adds MemCache KV pooling to the same deployment.

#### 5.2.1 Topology and Prerequisites

| Role | Nodes | Global DP | TP | PP | Placement | Devices per role |
| --- | --- | --- | --- | --- | --- | --- |
| Prefill | P0, P1 | 1 | 16 | 2 | One PP stage per node | 32 |
| Decode | D0, D1 | 16 | 2 | 1 | Eight DP ranks per node | 32 |

Each container exposes devices 0-15. P0 exposes the only Prefill API;
P1 runs the second PP stage without an API server. Decode ranks 0-7 run
on D0 and ranks 8-15 on D1, with API ports 8000-8007 on each node.

!!! warning "Version requirements"

    These commands use a **0.30.0RC reference configuration**, separately
    from the v0.23.0 co-located examples in Section 5.1. Use matching
    vLLM/vLLM-Ascend builds with the V1 model runner, multi-node `mp` PP,
    `glm47` parsers, and `MooncakeConnectorV2` available. Check the
    release-specific options before using another version or main.

1. Prepare the same weights and software on all four nodes. Set `MODEL_PATH`
   and `PYTHON_LIB_DIR` in each role script to the installed paths. Install
   `fastokens` on Prefill nodes for `VLLM_USE_FASTOKENS=1`.
2. Verify [multi-node communication](../../getting_started/installation.md#installation-multi-node-interconnect).
   Set `LOCAL_IP="<PREFILL_NODE_IP>"` on each Prefill node and
   `LOCAL_IP="<DECODE_NODE_IP>"` on each Decode node to that node's IP.
   Set `NIC_NAME="<NETWORK_INTERFACE>"` to its matching interface and
   `MODEL_PATH="<YOUR_MODEL_PATH>"` to its model weight directory.
3. Reserve the PP master port 7060, Decode DP RPC port 16600, API ports,
   proxy port 8000, and the topology-derived Mooncake transfer/handshake ports.
   Prefill and Decode run on separate node groups.
4. Keep the 78-layer PP partition `42,36` consistent in the Prefill
   environment and both roles' connector topology. Both roles explicitly
   use int8 KV cache and int8 indexer KV cache for these W8A8C8 weights.

#### 5.2.2 Prepare the Scripts

Before you start, please

1. prepare the script `launch_online_dp.py` on each Decode node:

    ```python
    import argparse
    import multiprocessing
    import subprocess
    import sys


    def run_command(args, local_rank):
        dp_rank = args.dp_rank_start + local_rank
        devices = ",".join(str(i) for i in range(
            local_rank * args.tp_size, (local_rank + 1) * args.tp_size))
        subprocess.run([
            "bash", args.script, devices, str(args.vllm_start_port + local_rank),
            str(args.dp_size), str(dp_rank), args.dp_address,
            str(args.dp_rpc_port), str(args.tp_size),
        ], check=True)


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dp-size", type=int, required=True)
        parser.add_argument("--tp-size", type=int, required=True)
        parser.add_argument("--dp-size-local", type=int, required=True)
        parser.add_argument("--dp-rank-start", type=int, required=True)
        parser.add_argument("--dp-address", required=True)
        parser.add_argument("--dp-rpc-port", type=int, default=16600)
        parser.add_argument("--vllm-start-port", type=int, default=8000)
        parser.add_argument("--script", default="./run_dp_template.sh")
        args = parser.parse_args()
        if min(args.dp_size, args.tp_size, args.dp_size_local) <= 0:
            parser.error("Parallel sizes must be positive")
        if args.dp_rank_start < 0 or args.dp_rank_start + args.dp_size_local > args.dp_size:
            parser.error("Local ranks must be within the global DP group")
        if args.dp_size_local * args.tp_size > 16:
            parser.error("This A3 configuration exposes 16 devices per node")
        processes = []
        for local_rank in range(args.dp_size_local):
            process = multiprocessing.Process(target=run_command, args=(args, local_rank))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        return 1 if any(process.exitcode != 0 for process in processes) else 0


    if __name__ == "__main__":
        sys.exit(main())
    ```

2. prepare the script `run_dp_template.sh` on each node.

    1. Prefill node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib

        # Set the current node IP and shared P0 master IP before starting.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000

        NODE_P0_IP="$LOCAL_IP"

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export VLLM_USE_FASTOKENS=1
        export VLLM_PP_LAYER_PARTITION="42,36"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr "$NODE_P0_IP" \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 0 \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --enforce-eager \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "enable_dsa_cp":true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "glm53-prefill",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                    "decode": {"dp_size": 16, "tp_size": 2}
                }
            }'
        ```

    2. Prefill node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib

        # Set the current node IP and shared P0 master IP before starting.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000

        NODE_P0_IP="<PREFILL_NODE0_IP>"

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export VLLM_USE_FASTOKENS=1
        export VLLM_PP_LAYER_PARTITION="42,36"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr "$NODE_P0_IP" \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 1 \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --enforce-eager \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "enable_dsa_cp":true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "glm53-prefill",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                    "decode": {"dp_size": 16, "tp_size": 2}
                }
            }' \
            --headless
        ```

    3. Decode node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib

        # Arguments $1-$7 are supplied by launch_online_dp.py.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000


        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 256 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_fused_mc2": 1
            }' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_consumer",
                "kv_port": "30100",
                "engine_id": "glm53-decode-dp'"$4"'",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                    "decode": {"dp_size": 16, "tp_size": 2}
                }
            }'
        ```

    4. Decode node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib

        # Arguments $1-$7 are supplied by launch_online_dp.py.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000


        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 256 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_fused_mc2": 1
            }' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_consumer",
                "kv_port": "30100",
                "engine_id": "glm53-decode-dp'"$4"'",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                    "decode": {"dp_size": 16, "tp_size": 2}
                }
            }'
        ```

#### 5.2.3 Start the Engines

Once the preparation is done, you can start the server with the following command on each node:

1. Prefill node 0

    ```shell
    bash run_dp_template.sh
    ```

2. Prefill node 1

    ```shell
    bash run_dp_template.sh
    ```

3. Decode node 0

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

4. Decode node 1

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 8 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

#### 5.2.4 Start the Proxy

Replace each node's IP and installation-path placeholders before starting.
Run the four commands in Section 5.2.3 in separate terminals. Keep the
master processes running while the other nodes join; start both nodes of
each group before waiting for the group to become healthy.

After the Prefill API on P0:8000 and all sixteen Decode APIs answer
`curl http://<node_ip>:<port>/v1/models`, start the
[PD load-balancing proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)
from the matching checkout on a separate host that can reach all P/D
API endpoints. The proxy listens on port 8000; using a separate host
avoids a port conflict with P0:8000 and the Decode APIs on 8000-8007.
Register P0 once and all sixteen Decode endpoints. P1 is not a separate
Prefill API endpoint.

```shell
P0_IP="<P0_IP>"
D0_IP="<D0_IP>"
D1_IP="<D1_IP>"

python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 --port 8000 \
    --prefiller-hosts "$P0_IP" --prefiller-ports 8000 \
    --decoder-hosts "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
        "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
        "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
        "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
    --decoder-ports 8000 8001 8002 8003 8004 8005 8006 8007 \
        8000 8001 8002 8003 8004 8005 8006 8007
```

Key parameters:

- Prefill uses TP16/PP2 over two nodes with the `mp` executor and
  `VLLM_PP_LAYER_PARTITION="42,36"`. Its batch budget is 16,384 tokens,
  maximum sequence count is 64, and it runs in eager mode with one MTP token.
- Decode uses DP16/TP2, a 256-token batch budget, 64 sequences, five MTP
  tokens with an eager draft model, and `FULL_DECODE_ONLY` graph mode.
  `recompute_scheduler_enable=true` allows recomputation scheduling.
- Both roles retain chunked prefill, prefix caching, async scheduling,
  seed 1024, memory utilization 0.92, and fused MC2. Prefill also enables
  DSA CP and FlashComm1. Both use `glm47` tool and reasoning parsers and
  the served model name `glm5`.
- `MooncakeConnectorV2` declares the same Prefill DP1/PP2/TP16 and
  Decode DP16/TP2 topology on both sides. The non-pooled base KV ports
  are 30000 on Prefill and 30100 on Decode; reserve derived worker ports.
- The examples give Prefill and each Decode engine distinct `engine_id`
  values so P-to-D peer registration cannot conflict.

### 5.3 MemCache KV Cache Pool Deployment

This extends the same four-node PP/PD topology with `MultiConnector`:
`MooncakeConnectorV2` transfers KV cache from Prefill to Decode, and
`AscendStoreConnector` provides non-layerwise MemCache pooling. Retain
the launcher, PP placement, and proxy routing from Section 5.2.

#### 5.3.1 Prepare MemCache

Complete [MemCache backend setup](../../user_guide/feature_guide/kv_pool.md#scenario-2-memcache-backend),
including the hardware/CANN prerequisites, `memfabric-hybrid` and
`memcache-hybrid` installation, configuration files, and metadata service.
All P/D instances must access the same pool; connector arguments do not
start the MemCache service. Set `MEMCACHE_ROOT` to the installed
`memcache_hybrid` directory and `PYTHON_LIB_DIR` to the library directory
of the Python installation actually used by vLLM on each node.

#### 5.3.2 Prepare the Scripts

Before you start, please

1. prepare the script `launch_online_dp.py` on each Decode node, using the
   launcher in Section 5.2.2.

2. prepare the pooled script `run_dp_template.sh` on each node.

    1. Prefill node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid

        # Set the current node IP and shared P0 master IP before starting.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000

        NODE_P0_IP="$LOCAL_IP"

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export VLLM_USE_FASTOKENS=1
        export VLLM_PP_LAYER_PARTITION="42,36"

        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr "$NODE_P0_IP" \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 0 \
            --enable-chunked-prefill \
            --async-scheduling \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --enforce-eager \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "enable_dsa_cp":true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "glm53-prefill",
                "kv_connector_extra_config": {
                    "connectors":[
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_producer",
                        "kv_port": "30000",
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                            "decode": { "dp_size": 16, "tp_size": 2}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_producer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port":"0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                    ]
                }
            }'
        ```

    2. Prefill node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid

        # Set the current node IP and shared P0 master IP before starting.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000

        NODE_P0_IP="<PREFILL_NODE0_IP>"

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export VLLM_USE_FASTOKENS=1
        export VLLM_PP_LAYER_PARTITION="42,36"

        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr "$NODE_P0_IP" \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 1 \
            --enable-chunked-prefill \
            --async-scheduling \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --enforce-eager \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "enable_dsa_cp":true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "glm53-prefill",
                "kv_connector_extra_config": {
                    "connectors":[
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_producer",
                        "kv_port": "30000",
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                            "decode": { "dp_size": 16, "tp_size": 2}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_producer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port":"0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                    ]
                }
            }' \
            --headless
        ```

    3. Decode node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid

        # Arguments $1-$7 are supplied by launch_online_dp.py.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000


        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export ASCEND_RT_VISIBLE_DEVICES=$1

        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 256 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_fused_mc2": 1
            }' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_consumer",
                "kv_port": "30200",
                "engine_id": "glm53-decode-dp'"$4"'",
                "kv_connector_extra_config": {
                    "connectors":[
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_consumer",
                        "kv_port": "30200",
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                            "decode": { "dp_size": 16, "tp_size": 2}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_consumer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port":"0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                    ]
                }
            }'
        ```

    4. Decode node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid

        # Arguments $1-$7 are supplied by launch_online_dp.py.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000


        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export ASCEND_RT_VISIBLE_DEVICES=$1

        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 256 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_fused_mc2": 1
            }' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_consumer",
                "kv_port": "30200",
                "engine_id": "glm53-decode-dp'"$4"'",
                "kv_connector_extra_config": {
                    "connectors":[
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_consumer",
                        "kv_port": "30200",
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                            "decode": { "dp_size": 16, "tp_size": 2}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_consumer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port":"0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                    ]
                }
            }'
        ```

The pooled configuration uses base KV ports 30000 on Prefill and
30200 on Decode. The outer `engine_id` is unique per engine and is
inherited by the child connectors. This differs from the supplied pooled
reference's repeated `"0"` IDs to avoid Mooncake peer identity conflicts.
`lookup_rpc_port="0"` is retained: Ascend Store uses an IPC endpoint that
also includes the DP rank, and P/D run on separate hosts. It is not a TCP
listener on port zero. `backend="memcache"` and `use_layerwise=false`
are set on both sides.

#### 5.3.3 Start the Engines

Start the MemCache metadata service and complete its configuration on all
four nodes before starting vLLM.

Once the preparation is done, you can start the server with the following command on each node:

1. Prefill node 0

    ```shell
    bash run_dp_template.sh
    ```

2. Prefill node 1

    ```shell
    bash run_dp_template.sh
    ```

3. Decode node 0

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

4. Decode node 1

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 8 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

#### 5.3.4 Start the Proxy and Verify Pooling

1. Start the MemCache metadata service and complete its configuration on
   all four nodes before starting vLLM.
2. Start P0, P1, D0, and D1 with the commands in
   Section 5.3.3. Both Prefill scripts already specify port 8000.
3. Wait for P0:8000 and all sixteen Decode APIs to become healthy. Start
   the proxy from Section 5.2.4 on a separate host, using the same
   Prefill endpoint and Decode ports 8000-8007. Its public port is 8000.
4. Send requests to the proxy using the served model name `glm5`:

```shell
curl "http://<proxy_ip>:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm5",
        "messages": [{"role": "user", "content": "Explain prefill and decode in an LLM."}],
        "max_tokens": 128,
        "temperature": 0
    }'
```

For either mode, check P/D logs for successful KV transfer. For pooling,
also check MemCache initialization, then warm the pool with a prompt longer
than one cache block and repeat requests with the same prefix. Inspect
Ascend Store/MemCache saves, lookups, loads, and hit metrics. A successful
response or a local prefix-cache hit alone does not demonstrate shared pool
reuse; verify a pool load when the prefix is absent from the engine's local
cache. Keep thinking enabled as described above. These references do not
establish A2, 1M-context, or other parallel-layout support, and do not add
accuracy or performance claims for the 198K workload.

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-5",
        "messages":[
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "temperature": 0
    }'
```

Expected result should be have this:

```text
"message":{"role":"assistant","content":"I'm GLM, a large language model developed by Z.ai. I'm designed to understand and generate human-like text based on the conversations we have together. My role involves processing diverse text data to help answer questions and provide assistance across many topics.\n\nI don't store your personal data, and I'm continually learning to improve my capabilities. Is there something specific I can help you with today?","refusal":null,"annotations":null,"audio":null,"function_call":oning":"Let me analyze this question about my identity. First, I should acknowledge that this is a fundamental question about who and what I am. The key aspects are my identity as GLM, a large language model by Z.ai, and my core capabilities. I should explain my primary function of text processing and be transparent about my nature as an AI system. It's also important to clarify my role in helping users and my ability to engage with various topics. Mention my text processing abilities and learning from diverse datasets, but avoid making claims about consciousness or emotions. The response should be structured logically, starting with my basic identity and moving on to my capabilities and purpose. I'll organize this information in a clear, straightforward manner that addresses the user's query directly."}
```

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result. Here are the results of `GLM-5.3-w8a8c8` in `vllm-ascend:v0.23.0` for reference only.

| dataset | model | hardware | metric | mode | vllm-api-general-chat |
| ----- | ----- | ----- | ----- | ----- | ----- |
| GPQA Diamond | GLM-5.3-w8a8c8 | A3 | accuracy | gen | 92.42 |
| GPQA Diamond | GLM-5.3-w8a8c8 | A2 | accuracy | gen | 90.40 |

### 7.2 Using Language Model Evaluation Harness

Not tested yet.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 FAQ

- **Q: How to enable function calling for GLM-5.3?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --enable-auto-tool-choice \
  ```

- **Q: Does GLM-5.3 support `enable_thinking: false`?**

  A: No, GLM-5.3 does not support `enable_thinking`.

## 10 Declaration

- The current version is only for early experience, and performance optimization is still in progress.
- The service reliability has not been fully validated, and it is not recommended for direct use in production environments.
