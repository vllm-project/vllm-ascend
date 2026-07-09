# DeepSeek-V3.2

## 1 Introduction

DeepSeek-V3.2 is a sparse attention model. The main architecture is similar to DeepSeek-V3.1, but with a sparse attention mechanism, which is designed to explore and validate optimizations for training and inference efficiency in long-context scenarios.

The `DeepSeek-V3.2` model is first supported in `vllm-ascend:v0.13.0rc1`, and all **v0.13.0rc1 and later versions** can run stably. To use the latest features (e.g., PD separation, MTP), it is recommended to use v0.13.0rc1 or a later version.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `DeepSeek-V3.2-Exp-W8A8` (Quantized version): requires **1 Atlas 800 A3 (64G × 16) node** or **2 Atlas 800 A2 (64G × 8) nodes**. [Download model weight](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.2-Exp-W8A8)
- `DeepSeek-V3.2-w8a8` (Quantized version): requires **1 Atlas 800 A3 (64G × 16) node** or **2 Atlas 800 A2 (64G × 8) nodes**. [Download model weight](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.2-W8A8/)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

You can use our official docker image to run `DeepSeek-V3.2` directly.

=== "A3 series"

    Start the docker image on your each node.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
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

    Start the docker image on your each node.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
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

To verify the successful installation of the environment, please refer to [installation](../../installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

### 4.2 Source Code Installation

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## 5 Online Service Deployment

:::{note}
In this tutorial, we suppose you downloaded the model weight to `/root/.cache/`. Feel free to change it to your own path.
:::

### Single-node Deployment

- Quantized model `DeepSeek-V3.2-w8a8` can be deployed on 1 Atlas 800 A3 (64G × 16).

Run the following script to execute online inference.

```shell
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_ENABLE_MLAPO=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 \
--host 0.0.0.0 \
--port 8000 \
--data-parallel-size 2 \
--tensor-parallel-size 8 \
--quantization ascend \
--seed 1024 \
--served-model-name deepseek_v3_2 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 8192 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'

```

Common Issues Tip: If you encounter issues, Refer to [FAQs](../../faqs.md).

### 5.2 Multi-Node Data Parallel Deployment

- `DeepSeek-V3.2-w8a8`: require at least 2 Atlas 800 A2 (64G × 8).

Common Issues Tip: If you encounter issues, Refer to [FAQs](../../faqs.md).

Run the following scripts on two nodes respectively.

=== "A3 series"

**Node0**

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
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export VLLM_USE_V1=1
    export HCCL_BUFFSIZE=200
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 \
    --host 0.0.0.0 \
    --port 8077 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12890 \
    --tensor-parallel-size 16 \
    --quantization ascend \
    --seed 1024 \
    --served-model-name deepseek_v3_2 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.92 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
    ```

    **Node1**

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
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export VLLM_USE_V1=1
    export HCCL_BUFFSIZE=200
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 \
    --host 0.0.0.0 \
    --port 8077 \
    --headless \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12890 \
    --tensor-parallel-size 16 \
    --quantization ascend \
    --seed 1024 \
    --served-model-name deepseek_v3_2 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.92 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
    ```

=== "A2 series"

    **Node0**

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
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=100
    export VLLM_USE_V1=1
    export HCCL_BUFFSIZE=200
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export HCCL_CONNECT_TIMEOUT=120
    export HCCL_INTRA_PCIE_ENABLE=1
    export HCCL_INTRA_ROCE_ENABLE=0

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 \
    --host 0.0.0.0 \
    --port 8077 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 8 \
    --quantization ascend \
    --seed 1024 \
    --served-model-name deepseek_v3_2 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.92 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[8, 16, 24, 32, 40, 48]}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'

    ```

    **Node1**

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
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=100
    export VLLM_USE_V1=1
    export HCCL_BUFFSIZE=200
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export HCCL_CONNECT_TIMEOUT=120
    export HCCL_INTRA_PCIE_ENABLE=1
    export HCCL_INTRA_ROCE_ENABLE=0

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 \
    --host 0.0.0.0 \
    --port 8077 \
    --headless \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 8 \
    --quantization ascend \
    --seed 1024 \
    --served-model-name deepseek_v3_2 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.92 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[8, 16, 24, 32, 40, 48]}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'

    ```

**Notice:**
The parameters are explained as follows:

- For multi-node deployment, we recommend using `dp2tp16` (A3) or `dp2tp8` (A2) with expert parallel enabled.

### 5.3 Multi-Node PD Separation Deployment

We recommend using Mooncake for deployment: [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md).

In the standard single-node deployment mode, Prefill (prompt processing) and Decode (token generation) tasks run on the same set of NPUs. PD (Prefill-Decode) separation addresses this by running Prefill and Decode on dedicated node groups, each configured independently:

- **Prefill nodes** focus on high-throughput prompt processing, optimized for compute and communication.
- **Decode nodes** focus on low-latency token generation, optimized for memory bandwidth.

This architecture is recommended for production deployments with concurrent multi-user workloads, where stable latency and high throughput are both required.

We'd like to show the deployment guide of `DeepSeek-V3.2` on multi-node environment with 1P1D for better performance.

Common Issues Tip: If you encounter issues, Refer to [FAQs](../../faqs.md).

- `DeepSeek-V3.1-w8a8-mtp-QuaRot 1P1D Layerwise` require 4 Atlas 800 A3 (64G × 16).

To run the vllm-ascend `Prefill-Decode Disaggregation` service, you need to deploy a `launch_online_dp.py` script and a `run_dp_template.sh` script on each node and deploy a `proxy.sh` script on prefill master node to forward requests.

[launch_online_dp.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py)

Parameter descriptions:

|Parameter|Type|Required|Default|Description|
|---------|----|--------|-------|-----------|
|`--dp-size`|int|Yes|-|Data parallel size (total number of DP ranks across all nodes).|
|`--tp-size`|int|No|1|Tensor parallel size within each DP rank.|
|`--dp-size-local`|int|No|(same as `--dp-size`)|Number of DP ranks on the current node. If not set, defaults to `--dp-size`.|
|`--dp-rank-start`|int|No|0|Starting rank offset for data parallel ranks on this node.|
|`--dp-address`|str|Yes|-|IP address of the data parallel master node (node 0).|
|`--dp-rpc-port`|str|No|12345|RPC port for data parallel master communication.|
|`--vllm-start-port`|int|No|9000|Starting port for each vLLM engine instance on this node. Each DP rank's engine port = `vllm_start_port` + local rank index.|

1. `run_dp_template.sh` script

=== "Node 0(Prefill)"

    ```shell
    nic_name="enp48s3u1u1" # change to your own nic name
    local_ip=141.61.39.105 # change to your own ip

    export HCCL_OP_EXPANSION_MODE="AIV"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_USE_V1=1
    export HCCL_BUFFSIZE=256

    export ASCEND_AGGREGATE_ENABLE=1
    export ASCEND_TRANSPORT_PRINT=1
    export ACL_OP_INIT_MODE=1
    export ASCEND_A3_ENABLE=1
    # Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request.
    export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480

    export ASCEND_RT_VISIBLE_DEVICES=$1

    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1


    vllm serve /root/.cache/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --enable-expert-parallel \
        --speculative-config '{"num_speculative_tokens": 2, "method":"deepseek_mtp"}' \
        --profiler-config \
        '{"profiler": "torch",
        "torch_profiler_dir": "./vllm_profile",
        "torch_profiler_with_stack": false}' \
        --seed 1024 \
        --served-model-name deepseek_v3.2 \
        --max-model-len 68000 \
        --max-num-batched-tokens 32560 \
        --trust-remote-code \
        --max-num-seqs 64 \
        --gpu-memory-utilization 0.82 \
        --quantization ascend \
        --enforce-eager \
        --no-enable-prefix-caching \
        --additional-config '{"layer_sharding": ["q_b_proj", "o_proj"], "enable_dsa_cp": true}' \
        --kv-transfer-config \
        '{"kv_connector": "MooncakeLayerwiseConnector",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "kv_connector_extra_config": {
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

=== "Node 1(Prefill)"

    ```shell
    nic_name="enp48s3u1u1" # change to your own nic name
    local_ip=141.61.39.113 # change to your own ip

    export HCCL_OP_EXPANSION_MODE="AIV"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_USE_V1=1
    export HCCL_BUFFSIZE=256

    export ASCEND_AGGREGATE_ENABLE=1
    export ASCEND_TRANSPORT_PRINT=1
    export ACL_OP_INIT_MODE=1
    export ASCEND_A3_ENABLE=1
    # Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request.
    export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480

    export ASCEND_RT_VISIBLE_DEVICES=$1

    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1


    vllm serve /root/.cache/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --enable-expert-parallel \
        --speculative-config '{"num_speculative_tokens": 2, "method":"deepseek_mtp"}' \
        --profiler-config \
        '{"profiler": "torch",
        "torch_profiler_dir": "./vllm_profile",
        "torch_profiler_with_stack": false}' \
        --seed 1024 \
        --served-model-name deepseek_v3.2 \
        --max-model-len 68000 \
        --max-num-batched-tokens 32560 \
        --trust-remote-code \
        --max-num-seqs 64 \
        --gpu-memory-utilization 0.82 \
        --quantization ascend \
        --enforce-eager \
        --no-enable-prefix-caching \
        --additional-config '{"layer_sharding": ["q_b_proj", "o_proj"], "enable_dsa_cp": true}' \
        --kv-transfer-config \
        '{"kv_connector": "MooncakeLayerwiseConnector",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "kv_connector_extra_config": {
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

=== "Node 0(Decode)"

    ```shell
    nic_name="enp48s3u1u1" # change to your own nic name
    local_ip=141.61.39.117 # change to your own ip

    export HCCL_OP_EXPANSION_MODE="AIV"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    #Mooncake
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10

    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_USE_V1=1
    export HCCL_BUFFSIZE=256


    export ASCEND_AGGREGATE_ENABLE=1
    export ASCEND_TRANSPORT_PRINT=1
    export ACL_OP_INIT_MODE=1
    export ASCEND_A3_ENABLE=1
    # Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request.
    export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480

    export TASK_QUEUE_ENABLE=1

    export ASCEND_RT_VISIBLE_DEVICES=$1


    vllm serve /root/.cache/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --enable-expert-parallel \
        --speculative-config '{"num_speculative_tokens": 2, "method":"deepseek_mtp"}' \
        --profiler-config \
        '{"profiler": "torch",
        "torch_profiler_dir": "./vllm_profile",
        "torch_profiler_with_stack": false}' \
        --seed 1024 \
        --served-model-name deepseek_v3.2 \
        --max-model-len 68000 \
        --max-num-batched-tokens 12 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[3, 6, 9, 12]}' \
        --trust-remote-code \
        --max-num-seqs 4 \
        --gpu-memory-utilization 0.95 \
        --no-enable-prefix-caching \
        --quantization ascend \
        --kv-transfer-config \
        '{"kv_connector": "MooncakeLayerwiseConnector",
        "kv_role": "kv_consumer",
        "kv_port": "30100",
        "kv_connector_extra_config": {
                    "prefill": {
                            "dp_size": 2,
                            "tp_size": 16
                    },
                    "decode": {
                            "dp_size": 8,
                            "tp_size": 4
                    }
            }
        }' \
        --additional-config '{"recompute_scheduler_enable" : true}'
    ```

=== "Node 1(Decode)"

    ```shell
    nic_name="enp48s3u1u1" # change to your own nic name
    local_ip=141.61.39.181 # change to your own ip

    export HCCL_OP_EXPANSION_MODE="AIV"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    #Mooncake
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10

    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_USE_V1=1
    export HCCL_BUFFSIZE=256

    export ASCEND_AGGREGATE_ENABLE=1
    export ASCEND_TRANSPORT_PRINT=1
    export ACL_OP_INIT_MODE=1
    export ASCEND_A3_ENABLE=1
    # Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request.
    export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480

    export TASK_QUEUE_ENABLE=1

    export ASCEND_RT_VISIBLE_DEVICES=$1


    vllm serve /root/.cache/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --enable-expert-parallel \
        --speculative-config '{"num_speculative_tokens": 2, "method":"deepseek_mtp"}' \
        --profiler-config \
        '{"profiler": "torch",
        "torch_profiler_dir": "./vllm_profile",
        "torch_profiler_with_stack": false}' \
        --seed 1024 \
        --served-model-name deepseek_v3.2 \
        --max-model-len 68000 \
        --max-num-batched-tokens 12 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY",  "cudagraph_capture_sizes":[3, 6, 9, 12]}' \
        --trust-remote-code \
        --max-num-seqs 4 \
        --gpu-memory-utilization 0.95 \
        --no-enable-prefix-caching \
        --quantization ascend \
        --kv-transfer-config \
        '{"kv_connector": "MooncakeLayerwiseConnector",
        "kv_role": "kv_consumer",
        "kv_port": "30100",
        "kv_connector_extra_config": {
                    "prefill": {
                            "dp_size": 2,
                            "tp_size": 16
                    },
                    "decode": {
                            "dp_size": 8,
                            "tp_size": 4
                    }
            }
        }' \
        --additional-config '{"recompute_scheduler_enable" : true}'
    ```

Once the preparation is done, you can start the server with the following command on each node:
Refer to [Distributed DP Server With Large-Scale Expert Parallelism](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/large_scale_ep.html) to get the detailed boot method.

1. Prefill node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 0 --dp-address 141.61.39.105 --dp-rpc-port 12890 --vllm-start-port 9100
    ```

2. Prefill node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 1 --dp-address 141.61.39.105 --dp-rpc-port 12890 --vllm-start-port 9100
    ```

3. Decode node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address 141.61.39.117 --dp-rpc-port 12777 --vllm-start-port 9100
    ```

4. Decode node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 4 --dp-rank-start 4 --dp-address 141.61.39.117 --dp-rpc-port 12777 --vllm-start-port 9100
    ```

**Notice:**
To support a long context window on the stage of prefill, the parameter `"layer_sharding": ["q_b_proj", "o_proj"]` and `"enable_dsa_cp": true` needs to be added to `--additional-config` on each prefill node.

### Request Forwarding

To set up request forwarding, run the following script on any machine. You can get the proxy program in the repository's examples: [load_balance_proxy_layerwise_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py)

```shell
unset http_proxy
unset https_proxy

python load_balance_proxy_layerwise_server_example.py \
    --port 8000 \
    --host 141.61.39.105 \
    --prefiller-hosts \
       141.61.39.105 \
       141.61.39.113 \
    --prefiller-ports \
       9100 \
       9100 \
    --decoder-hosts \
      141.61.39.117 \
      141.61.39.117 \
      141.61.39.117 \
      141.61.39.117 \
      141.61.39.181 \
      141.61.39.181 \
      141.61.39.181 \
      141.61.39.181 \
    --decoder-ports \
      9100 9101 9102 9103 \
      9100 9101 9102 9103 \
```

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

**Note**:

- `<node0_ip>`: The IP address of the node where the server is running (e.g., localhost). For PD-separated deployment, use the host IP of the node where the proxy script resides.
- `<port>`: The port number specified in the server startup command (e.g., 8000). For PD-separated deployment, use the port configured in the proxy script.

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v3.2",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

**Expected Result**:

```json
{"id":"019eab54ead036b23e53f3a709e09289","object":"chat.completion","created":1780990929,"model":"deepseek_v3.2","choices":[{"index":0,"message":{"role":"assistant","content":"The future of AI is **not a single destination, but a complex, multi-faceted trajectory** that will reshape nearly every aspect of human society, technology, and our understanding of intelligence itself. It can be understood through several interconnected lenses:\n\n### "},"finish_reason":"length"}],"usage":{"prompt_tokens":9,"completion_tokens":50,"total_tokens":59,"completion_tokens_details":{"reasoning_tokens":0},"prompt_tokens_details":{"cached_tokens":0},"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":9},"system_fingerprint":""}
```

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

### Using Language Model Evaluation Harness

As an example, take the `gsm8k` dataset as a test dataset, and run accuracy evaluation of `DeepSeek-V3.2-W8A8` in online mode.

1. Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for `lm_eval` installation.

2. Run `lm_eval` to execute the accuracy evaluation.

    ```shell
    lm_eval \
    --model local-completions \
    --model_args model=/root/.cache/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
    --tasks gsm8k \
    --output_path ./
    ```

3. After execution, you can get the result.

## 8 Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

The performance result is:  

**Hardware**: A3-752T, 4 node

**Deployment**: 1P1D, Prefill node: DP2+TP16, Decode Node: DP8+TP4

**Input/Output**: 64k/3k

**Performance**: 533tps, TPOT 32ms

### Using vLLM Benchmark

Run performance evaluation of `DeepSeek-V3.2-W8A8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=True
vllm bench serve --model /root/.cache/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

## 9 Performance Tuning

The function call feature is supported from v0.13.0rc1 on. Please use the latest version.

Refer to [DeepSeek-V3.2 Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2.html#tool-calling-example) for details.

**Notice:** `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the [Deployment](#deployment) chapter.

## 10 FAQ

- Common Issues Tip: If you encounter issues, Refer to [FAQs](../../faqs.md).

- **Q: How to resolve empty output or garbled characters in PD separated deployment?**

  A: This is a known issue. Please ensure `--no-enable-prefix-caching` is set and use the latest version of vllm-ascend.

- **Q: How to configure the model weight path correctly?**

  A: The model weight path in the deployment commands (e.g., `/root/.cache/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot`) should be adjusted to your actual download path. Refer to the [Model Weight](#model-weight) section for details.