# GLM-5.3 (Experimental)

## 1 Introduction

[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) uses the same base model as GLM-5.2 — every gain comes from post-training. Compared with GLM-5.2, it is much better at complex coding and long-horizon tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, multi-node deployment, accuracy and performance evaluation.

!!! warning

    **Current status and constraints**

    - The multi-node co-located examples were tested on the official Docker images `quay.io/ascend/vllm-ascend:v0.23.0-a3` and `quay.io/ascend/vllm-ascend:v0.23.0`. The [Prefill-Decode disaggregation example](#52-prefill-decode-disaggregation) uses a separate **0.30.0RC reference configuration** with MemCache KV pooling on A3.
    - The features listed in [Supported Features](#2-supported-features) are only those enabled by the verified deployment commands in this document, and do **not** imply that all features are supported for GLM-5.3. This is an early-access version; performance optimization and reliability validation are still in progress (see [Declaration](#10-declaration)).
    - The co-located scripts are based on **v0.23.0**; Section 5.2 has its own version requirements. Check configuration compatibility before using either example with another release or the main branch.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

|  Weight Version          | Hardware Requirements                                         | Download Links |
|--------------------------|---------------------------------------------------------------|----------------|
|  `GLM-5.3-w8a8c8`        | 2 Atlas 800 A3 (128GB × 8) node or 4 Atlas 800 A2 (64GB × 32) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-w8a8c8) |

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

The multi-node co-located examples are organized by context window size (below 1M) and hardware (Atlas 800 A3 / A2). Section 5.2 adds a separate A3 Prefill-Decode disaggregation reference with MemCache KV pooling. Version requirements and key parameters are described with each scenario.

!!! note

    Do not set `enable_thinking: false` / `thinking: false` for GLM-5.3, otherwise the output quality may degrade.

!!! warning

    - The scripts in Section 5.1 were tested on **v0.23.0**. Parameters may have changed in the main branch. For Section 5.2, follow the version requirements in that section.

### 5.1 Multi-node Deployment

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

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
        --port 8077 \
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
        --reasoning-parser glm45 \
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
        --port 8077 \
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
        --reasoning-parser glm45 \
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

    - `GLM-5.3-w8a8c8`: can be deployed on 4 Atlas 800 A2 (64GB × 32).

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
        --port 8077 \
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
        --port 8077 \
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

#### 5.2.1 Deployment on 4 Atlas 800 A3 with MemCache KV Pooling

This reference uses GLM-5.3 W8A8C8 weights, a maximum sequence length of **200,000 tokens**, and separate Prefill and Decode instances. `MultiConnector` combines `MooncakeConnectorV1` for P-to-D KV transfer with `AscendStoreConnector` for MemCache KV pooling.

!!! warning "Version requirements"

    These commands are adapted from a **0.30.0RC reference configuration**, not the v0.23.0 co-located environment above. Use a matching vLLM/vLLM-Ascend environment with the V1 model runner, `glm47` parsers, and both connectors available. Do not assume that the v0.23.0 images in Section 4 support this configuration. Check release-specific options when using a different version; in particular, DSA CP and scheduler configuration can change between releases.

The following placement uses four A3 nodes, each exposing 16 NPU devices to its container. Prefill and Decode use separate DP groups.

| Role | Nodes | Global DP | TP per DP rank | DP ranks per node | Devices per role |
| --- | --- | --- | --- | --- | --- |
| Prefill | P0, P1 | 4 | 8 | 2 | 32 |
| Decode | D0, D1 | 8 | 4 | 4 | 32 |

Prerequisites:

1. Prepare the same GLM-5.3 W8A8C8 weights and compatible software on every node. Install `fastokens` on Prefill nodes for `VLLM_USE_FASTOKENS=1`. Set `MODEL_PATH` to the local model directory. The `--quantization ascend` option loads the Ascend quantization configuration from the weights; do not copy the co-located KV-cache overrides into this reference without checking compatibility.
2. Verify [multi-node communication](../../getting_started/installation.md#installation-multi-node-interconnect). Set each node's `local_ip` and `nic_name` to its reachable address and matching network interface.
3. Complete [MemCache backend setup](../../user_guide/feature_guide/kv_pool.md#scenario-2-memcache-backend), including the hardware/CANN prerequisites, `memfabric-hybrid` and `memcache-hybrid` installation, configuration files, and metadata service. All P/D instances must access the same pool. The connector configuration below does not start the MemCache service.
4. Ensure the API, DP RPC, and KV-transfer ports are available and reachable. Each instance needs a distinct engine ID and non-conflicting local ports. The scripts below derive these from the role and global DP rank.

Before you start, prepare the following scripts on each node, following the same launcher/template layout as the [GLM-5.2 PD deployment](GLM5.2.md#513-prefill-decode-disaggregation).

1. Prepare the script `launch_online_dp.py` on each node:

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
        parser.add_argument("--dp-rpc-port", type=str, default="12321", help="Port for data parallel master node.")
        parser.add_argument("--vllm-start-port", type=int, default=8000, help="Starting port for the engine.")
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
    gpus_per_dp_rank = tp_size

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
        num_cards = dp_size_local * gpus_per_dp_rank

        for i in range(dp_size_local):
            dp_rank = dp_rank_start + i
            vllm_engine_port = vllm_start_port + i
            visible_devices = ",".join(str(x) for x in range(i * gpus_per_dp_rank, (i + 1) * gpus_per_dp_rank))
            process = multiprocessing.Process(
                target=run_command,
                args=(visible_devices, dp_rank, vllm_engine_port)
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
        sys.exit(1 if any(process.exitcode != 0 for process in processes) else 0)
    ```

2. Prepare the script `run_dp_template.sh` on each node. Use the Prefill template on P0/P1 and the Decode template on D0/D1. Set `local_ip`, `nic_name`, and the installation paths separately on each node. `MEMCACHE_ROOT` is the installed `memcache_hybrid` package directory; `PYTHON_LIB_DIR` is the library directory of the Python installation used by vLLM.

    1. Prefill nodes 0 and 1

        ```shell
        #!/usr/bin/env bash
        set -euo pipefail

        # Set these values separately on each node.
        local_ip="<current_node_ip>"
        nic_name="<current_node_nic>"
        MODEL_PATH=/path/to/GLM-5.3-W8A8C8
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid
        PYTHON_LIB_DIR=/path/to/python/lib

        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH:-}"
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export VLLM_HOST_IP="${local_ip}"
        export HCCL_IF_IP="${local_ip}"
        export GLOO_SOCKET_IFNAME="${nic_name}"
        export TP_SOCKET_IFNAME="${nic_name}"
        export HCCL_SOCKET_IFNAME="${nic_name}"
        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE=AIV
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1
        export VLLM_USE_V2_MODEL_RUNNER=0
        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1

        export VLLM_USE_FASTOKENS=1

        vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens "8192" \
            --trust-remote-code \
            --max-num-seqs "64" \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy prefetch \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --enforce-eager \
            --api-server-count 8 \
            --additional-config '{"enable_dsa_cp":true,"enable_fused_mc2":1,"enable_flashcomm1":true}' \
            --speculative-config '{"num_speculative_tokens":1,"method":"deepseek_mtp","enforce_eager":true}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_producer",
                "kv_port": 30000,
                "engine_id": "glm53-prefill-'"$4"'",
                "kv_connector_extra_config": {
                    "connectors": [
                        {
                            "kv_connector": "MooncakeConnectorV1",
                            "kv_role": "kv_producer",
                            "kv_port": 30000,
                            "kv_connector_extra_config": {
                                "use_ascend_direct": true,
                                "prefill": {"dp_size": 4, "tp_size": 8},
                                "decode": {"dp_size": 8, "tp_size": 4}
                            }
                        },
                        {
                            "kv_connector": "AscendStoreConnector",
                            "kv_role": "kv_producer",
                            "kv_connector_extra_config": {
                                "lookup_rpc_port": '$((37000 + $4))',
                                "backend": "memcache",
                                "use_layerwise": false
                            }
                        }
                    ]
                }
            }'
        ```

    2. Decode nodes 0 and 1

        ```shell
        #!/usr/bin/env bash
        set -euo pipefail

        # Set these values separately on each node.
        local_ip="<current_node_ip>"
        nic_name="<current_node_nic>"
        MODEL_PATH=/path/to/GLM-5.3-W8A8C8
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid
        PYTHON_LIB_DIR=/path/to/python/lib

        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH:-}"
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export VLLM_HOST_IP="${local_ip}"
        export HCCL_IF_IP="${local_ip}"
        export GLOO_SOCKET_IFNAME="${nic_name}"
        export TP_SOCKET_IFNAME="${nic_name}"
        export HCCL_SOCKET_IFNAME="${nic_name}"
        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE=AIV
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1
        export VLLM_USE_V2_MODEL_RUNNER=0
        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10

        vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens "256" \
            --trust-remote-code \
            --max-num-seqs "32" \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy prefetch \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"recompute_scheduler_enable":true,"enable_fused_mc2":1,"ascend_compilation_config":{"enable_static_kernel":false}}' \
            --speculative-config '{"num_speculative_tokens":5,"method":"deepseek_mtp","enforce_eager":true}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_consumer",
                "kv_port": 30200,
                "engine_id": "glm53-decode-'"$4"'",
                "kv_connector_extra_config": {
                    "connectors": [
                        {
                            "kv_connector": "MooncakeConnectorV1",
                            "kv_role": "kv_consumer",
                            "kv_port": 30200,
                            "kv_connector_extra_config": {
                                "use_ascend_direct": true,
                                "prefill": {"dp_size": 4, "tp_size": 8},
                                "decode": {"dp_size": 8, "tp_size": 4}
                            }
                        },
                        {
                            "kv_connector": "AscendStoreConnector",
                            "kv_role": "kv_consumer",
                            "kv_connector_extra_config": {
                                "lookup_rpc_port": '$((37100 + $4))',
                                "backend": "memcache",
                                "use_layerwise": false
                            }
                        }
                    ]
                }
            }'
        ```

Once the preparation is done, start the servers with the following command on each node. Keep all ranks running while the other nodes join; start the proxy only after every API endpoint is healthy.

1. Prefill node 0

    ```shell
    node_p0_ip="<P0_ip>"
    python launch_online_dp.py \
        --dp-size 4 \
        --tp-size 8 \
        --dp-size-local 2 \
        --dp-rank-start 0 \
        --dp-address "$node_p0_ip" \
        --dp-rpc-port 16591 \
        --vllm-start-port 9081
    ```

2. Prefill node 1

    ```shell
    node_p0_ip="<P0_ip>"
    python launch_online_dp.py \
        --dp-size 4 \
        --tp-size 8 \
        --dp-size-local 2 \
        --dp-rank-start 2 \
        --dp-address "$node_p0_ip" \
        --dp-rpc-port 16591 \
        --vllm-start-port 9081
    ```

3. Decode node 0

    ```shell
    node_d0_ip="<D0_ip>"
    python launch_online_dp.py \
        --dp-size 8 \
        --tp-size 4 \
        --dp-size-local 4 \
        --dp-rank-start 0 \
        --dp-address "$node_d0_ip" \
        --dp-rpc-port 16600 \
        --vllm-start-port 9900
    ```

4. Decode node 1

    ```shell
    node_d0_ip="<D0_ip>"
    python launch_online_dp.py \
        --dp-size 8 \
        --tp-size 4 \
        --dp-size-local 4 \
        --dp-rank-start 4 \
        --dp-address "$node_d0_ip" \
        --dp-rpc-port 16600 \
        --vllm-start-port 9900
    ```

From the matching vLLM-Ascend checkout, start the [P/D load-balancing proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py) on a host that can reach all 12 API endpoints. Set the four host variables to the corresponding node addresses, including repeated hosts for instances on the same node.

```shell
P0="<P0_ip>"
P1="<P1_ip>"
D0="<D0_ip>"
D1="<D1_ip>"

python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 \
    --port 8000 \
    --prefiller-hosts "$P0" "$P0" "$P1" "$P1" \
    --prefiller-ports 9081 9082 9081 9082 \
    --decoder-hosts "$D0" "$D0" "$D0" "$D0" "$D1" "$D1" "$D1" "$D1" \
    --decoder-ports 9900 9901 9902 9903 9900 9901 9902 9903
```

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. Adjust `max-model-len` and `max-num-seqs` according to the workload and available memory.

**Multi-node network and data parallel configuration:**

- `local_ip` / `nic_name`: The current node's reachable IP address and matching network interface. `node_p0_ip` / `node_d0_ip` identify the masters of the separate Prefill and Decode DP groups.
- `launch_online_dp.py` starts one external-DP instance per local rank. It passes `$1` = visible devices, `$2` = API port, `$3` = global DP size, `$4` = global DP rank, `$5` = DP master address, `$6` = DP RPC port, and `$7` = TP size to `run_dp_template.sh`.
- Prefill uses `DP4 TP8`, with ranks 0–1 on P0 and 2–3 on P1. Decode uses `DP8 TP4`, with ranks 0–3 on D0 and 4–7 on D1. Each node uses devices 0–15 without overlap between local instances.
- Every instance exposes an API endpoint for the proxy. Do not add `--headless` to these external-DP templates.

**Prefill node-specific configurations (P0/P1):**

- `VLLM_USE_FASTOKENS=1` / `--api-server-count 8`: Enable the installed `fastokens` backend and start eight API server processes per Prefill instance.
- `--enforce-eager`: Run Prefill in eager mode, with an 8,192-token batch budget and at most 64 sequences.
- `enable_dsa_cp` / `enable_flashcomm1` / `enable_fused_mc2`: Retain the reference's DSA CP, FlashComm1, and fused MC2 settings.
- `--speculative-config`: Use one MTP token and an eager draft model on Prefill.

**Decode node-specific configurations (D0/D1):**

- `FULL_DECODE_ONLY` / `enable_static_kernel=false`: Retain the reference's decode graph configuration, with a 256-token batch budget and at most 32 sequences.
- `--speculative-config`: Use five MTP tokens and an eager draft model on Decode.
- `recompute_scheduler_enable=true`: Enable recomputation scheduling only on Decode, so requests can return to Prefill when decode KV cache is insufficient.
- `OMP_PROC_BIND=false` / `OMP_NUM_THREADS=10`: Reference CPU settings; size CPU resources for the local worker count before tuning them.

**KV transfer and MemCache pooling configuration:**

- `MultiConnector`: Combine `MooncakeConnectorV1` for P-to-D transfer with `AscendStoreConnector` for MemCache pooling. Set both child roles to `kv_producer` on Prefill and `kv_consumer` on Decode.
- `prefill` / `decode`: Keep the same global topology (`DP4 TP8` / `DP8 TP4`) in every Mooncake connector configuration.
- `kv_port`: Base handshake port, 30000 on Prefill and 30200 on Decode. Mooncake offsets worker ports using the DP/TP ranks; reserve the resulting ranges on each host.
- `engine_id` / `lookup_rpc_port`: Distinct per-instance engine IDs and integer store lookup identifiers derived from the role and global DP rank.
- `backend=memcache` / `use_layerwise=false`: Use non-layerwise MemCache pooling on both sides. Complete the pool configuration and start its metadata service before starting the vLLM instances.

**Request forwarding and verification:**

- Register all four Prefill and eight Decode API endpoints with the proxy, including repeated host addresses for local instances.
- Send requests to the proxy using the served model name `glm5`:

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

Check P/D logs for successful KV transfer and MemCache initialization. To verify pool reuse, send requests with a shared prefix and inspect the Ascend Store/MemCache hit and load metrics; one successful response alone does not demonstrate a pool hit. Keep thinking enabled as described above. This reference does not establish support for A2, 1M context, or other parallel layouts.

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
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```

- **Q: Does GLM-5.3 support `enable_thinking: false`?**

  A: No, GLM-5.3 does not support `enable_thinking`.

## 10 Declaration

- The current version is only for early experience, and performance optimization is still in progress.
- The service reliability has not been fully validated, and it is not recommended for direct use in production environments.
