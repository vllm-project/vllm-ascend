# GLM-5.2

## 1 Introduction

[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) uses a Mixture-of-Experts (MoE) architecture and targets complex systems engineering and long-horizon agentic tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `GLM-5.2`(BF16 version): requires 2 Atlas 800 A3 (128GB × 8) node or 4 Atlas 800 A2 (64GB × 8) node.[Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-5.2).
- `GLM-5.2-w8a8`: requires 1 Atlas 800 A3 (128GB × 8) node or 2 Atlas 800 A2 (64GB × 8) node.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w8a8).
- `GLM-5.2-w4a8c8`: requires 1 Atlas 800 A3 (128GB × 8) node or 2 Atlas 800 A2 (64GB × 8) node.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w4a8c8).
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

- You can use our official docker image to run GLM-5.2 directly.

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

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## 5 Deployment

The deployment scenarios validated for this release are organized by context window size (below 1M / 1M), hardware (Atlas 800 A3 / A2), and deployment mode (single-node, multi-node co-located, Prefill-Decode disaggregation). All startup scripts below are the verified reference commands; key parameters are explained after each scenario.

### 5.1 Context Below 1M

#### 5.1.1 Atlas 800 A3

##### 5.1.1.1 Single-Node Deployment

- Quantized model `GLM-5.2-w4a8c8` can be deployed on 1 Atlas 800 A3 (64GB × 16) .

Run the following script to execute online inference.

```shell
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
--host 0.0.0.0 \
--port 8077 \
--safetensors-load-strategy prefetch \
--api-server-count 1 \
--data-parallel-size 2 \
--enable-expert-parallel \
--tensor-parallel-size 8 \
--seed 1024 \
--served-model-name glm-5 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice \
--max-num-seqs 12 \
--max-model-len 135000 \
--max-num-batched-tokens 8192 \
--trust-remote-code \
--gpu-memory-utilization 0.92 \
--quantization ascend \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"enable_dsa_cp": true,"enable_sparse_li_c8": true,"enable_balance_scheduling": true,"multistream_overlap_shared_expert":true,"enable_fused_mc2":0}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'

```

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario.

**Model-specific parameters:**

- `--enable-expert-parallel`: Must be enabled for the MoE architecture of GLM-5.2.
- `--quantization ascend`: Enables Ascend quantization for the w4a8c8 quantized weights.
- `--data-parallel-size 2` / `--tensor-parallel-size 8`: DP2 TP8 parallelism layout, recommended to balance memory capacity and compute efficiency for the w4a8c8 weights. For low-latency scenarios, use `dp1tp16` instead and turn off expert parallel, at the cost of lower throughput.
- `--tool-call-parser glm47` / `--reasoning-parser glm45` / `--enable-auto-tool-choice`: Enable function calling for GLM-5.2.
- `--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'`: Enables Multi-Token Prediction (MTP) speculative decoding with the DeepSeek-style MTP draft head of GLM-5.2. `num_speculative_tokens` (3-5) controls how many tokens are speculated per step; `enforce_eager: true` is required because GLM-5.2 does not support graph-mode speculative decoding.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables graph capture for the decode phase only, improving decode performance by reducing kernel launch overhead.
- `additional_config.enable_fused_mc2=0`: Disables the fused `dispatch_ffn_combine`/`mega_moe` operators in this scenario because they conflict with `multistream_overlap_shared_expert`; turn them on in multi-node scenarios where they are beneficial.

**`--additional-config` fields (Ascend-specific optimizations):**

- `"enable_dsa_cp": true`: Enables DSA context parallelism to accelerate long-context prefill. With DSA-CP enabled, `layer_sharding` cannot include `o_proj`.
- `"enable_sparse_li_c8": true`: Sparse attention optimizations of the C8 quantized model. `enable_sparse_li_c8` accelerates the layer-index (LI) sparse attention and is recommended to keep `true`; If the GPU memory is insufficient due to a long sequence length, you are advised to enable `enable_sparse_sfa_c8`.
- `"enable_balance_scheduling": true`: Balance scheduling improves output throughput and reduces TPOT in the v1 scheduler. This config field replaces the removed environment setting; TTFT may degrade in some scenarios, and it is not recommended when Prefill-Decode is separated.
- `"multistream_overlap_shared_expert": true`: Overlaps shared-expert computation on an additional stream, improving decode efficiency.

##### 5.1.1.2 Multi-Node Co-Located Deployment

- `GLM-5.2-w4a8c8`: can be deployed on 2 Atlas 800 A3 (64GB × 16).

Run the following scripts on two nodes respectively.

**node 0**

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_BUFFSIZE=400
export HCCL_IF_IP=$local_ip
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
--host 0.0.0.0 \
--port 8077 \
--safetensors-load-strategy prefetch \
--api-server-count 1 \
--data-parallel-size 4 \
--data-parallel-start-rank 0 \
--data-parallel-size-local 2 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 12980 \
--tensor-parallel-size 8 \
--enable-expert-parallel \
--seed 1024 \
--served-model-name glm-52 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice \
--max-num-seqs 16 \
--max-model-len 66000 \
--max-num-batched-tokens 8192 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--quantization ascend \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"enable_dsa_cp": true,"enable_sparse_li_c8": true,"enable_balance_scheduling": true,"enable_fused_mc2":1}'  \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
```

**node 1**

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_BUFFSIZE=400
export HCCL_IF_IP=$local_ip
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
--host 0.0.0.0 \
--port 8077 \
--headless \
--safetensors-load-strategy prefetch \
--data-parallel-size 4 \
--data-parallel-start-rank 2 \
--data-parallel-size-local 2 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 12980 \
--tensor-parallel-size 8 \
--enable-expert-parallel \
--seed 1024 \
--served-model-name glm-52 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice \
--max-num-seqs 16 \
--max-model-len 66000 \
--max-num-batched-tokens 8192 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--quantization ascend \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"enable_dsa_cp": true,"enable_sparse_li_c8": true,"enable_balance_scheduling": true,"enable_fused_mc2":1}'  \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
```

Key Parameter Descriptions (in addition to [Single-Node Deployment](#5111-single-node-deployment)):

**Multi-node network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`: Network interface configuration for multi-node communication. Set `nic_name` to the network interface name (obtained via `ifconfig`) and `local_ip` to the current node's IP address. These must be correctly configured on each node for successful multi-node communication.
- `--data-parallel-size 4`: Total number of data parallel ranks across all nodes (2 ranks per node in this scenario).
- `--data-parallel-size-local 2`: Number of data parallel ranks on the current node.
- `--data-parallel-start-rank`: Starting rank offset for data parallel ranks on this node. Node 0 uses `0`, node 1 uses `2`.
- `--data-parallel-address`: IP address of the data parallel master node (node 0). Must match the `local_ip` of the master node.
- `--data-parallel-rpc-port 12980`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: Indicates a non-master node (used on node 1). Do not use on node 0.

**Notice:**
This scenario enables `additional_config.enable_fused_mc2=1` (fused `dispatch_ffn_combine`/`mega_moe` operators). Fused MC2 conflicts with `multistream_overlap_shared_expert` — the two optimizations must not be enabled at the same time (the runtime forcibly disables `multistream_overlap_shared_expert` when fused MC2 is on).

##### 5.1.1.3 Prefill-Decode Disaggregation

We'd like to show the deployment guide of `GLM-5.2` on multi-node environment with 1P1D for better performance.

Prefill-Decode disaggregation can be deployed on 4 Atlas 800 A3 (64GB × 16).

**Deployment topology:**

|Node group|Nodes|Parallelism|Engine ports|
|----------|-----|-----------|------------|
|Prefill|2 (node 0/1)|`DP4 TP8` (2 ranks per node)|9081/9082 per node|
|Decode|2 (node 0/1)|`DP32 TP1` (16 ranks per node)|9900-9915 per node|

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

    1. Prefill node 0

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        export HCCL_BUFFSIZE=400
        export HCCL_IF_IP=$local_ip
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_SOCKET_IFNAME=$nic_name
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        export GLOO_SOCKET_IFNAME=$nic_name
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
            --host 0.0.0.0 \
            --port $2 \
            --safetensors-load-strategy prefetch \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens":1, "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 133120 \
            --additional-config '{"enable_dsa_cp":true,"enable_sparse_li_c8": true,"c8_enable_reshape_optim":true,"enable_fused_mc2":1}' \
            --max-num-batched-tokens 8192 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --quantization ascend \
            --gpu-memory-utilization 0.92 \
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
                                "dp_size": 4,
                                "tp_size": 8
                        },
                        "decode": {
                                "dp_size": 32,
                                "tp_size": 1
                        }
                }
            }'

        ```

    2. Prefill node 1

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        export HCCL_BUFFSIZE=400
        export HCCL_IF_IP=$local_ip
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_SOCKET_IFNAME=$nic_name
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        export GLOO_SOCKET_IFNAME=$nic_name
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
            --host 0.0.0.0 \
            --port $2 \
            --safetensors-load-strategy prefetch \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens":1, "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 133120 \
            --additional-config '{"enable_dsa_cp":true, "enable_sparse_li_c8": true,"c8_enable_reshape_optim":true,"enable_fused_mc2":1}' \
            --max-num-batched-tokens 8192 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --quantization ascend \
            --gpu-memory-utilization 0.92 \
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
                                "dp_size": 4,
                                "tp_size": 8
                        },
                        "decode": {
                                "dp_size": 32,
                                "tp_size": 1
                        }
                }
            }'
        ```

    3. Decode node 0

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        #Mooncake
        export HCCL_BUFFSIZE=256
        export HCCL_IF_IP=$local_ip
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_SOCKET_IFNAME=$nic_name
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        export GLOO_SOCKET_IFNAME=$nic_name
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
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
            --served-model-name glm-5 \
            --max-model-len 133120 \
            --max-num-batched-tokens 164 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --additional-config '{"recompute_scheduler_enable":true,"enable_sparse_li_c8":true,"enable_fused_mc2":1}' \
            --trust-remote-code \
            --max-num-seqs 32 \
            --gpu-memory-utilization 0.92 \
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
                                "dp_size": 4,
                                "tp_size": 8
                        },
                        "decode": {
                                "dp_size": 32,
                                "tp_size": 1
                        }
                }
            }'
        ```

    4. Decode node 1

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        #Mooncake
        export HCCL_BUFFSIZE=256
        export HCCL_IF_IP=$local_ip
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_SOCKET_IFNAME=$nic_name
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        export GLOO_SOCKET_IFNAME=$nic_name
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
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
            --served-model-name glm-5 \
            --max-model-len 133120 \
            --max-num-batched-tokens 164 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --additional-config '{"recompute_scheduler_enable":true,"enable_sparse_li_c8":true,"enable_fused_mc2":1}' \
            --trust-remote-code \
            --max-num-seqs 32 \
            --gpu-memory-utilization 0.92 \
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
                                "dp_size": 4,
                                "tp_size": 8
                        },
                        "decode": {
                                "dp_size": 32,
                                "tp_size": 1
                        }
                }
            }'
        ```

Once the preparation is done, you can start the server with the following command on each node:

1. Prefill node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 4 --tp-size 8  --dp-size-local 2 --dp-rank-start 0 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    ```

2. Prefill node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 4 --tp-size 8  --dp-size-local 2 --dp-rank-start 2 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    ```

3. Decode node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    ```

4. Decode node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 16 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
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
      $node_p0_ip \
      $node_p1_ip \
      $node_p1_ip \
    --prefiller-ports \
      9081 9082 \
      9081 9082 \
    --decoder-hosts \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
    --decoder-ports \
      9900 9901 9902 9903 9904 9905 9906 9907 9908 9909 9910 9911 9912 9913 9914 9915 \
      9900 9901 9902 9903 9904 9905 9906 9907 9908 9909 9910 9911 9912 9913 9914 9915
```

Key Parameter Descriptions (in addition to [Single-Node Deployment](#5111-single-node-deployment) and [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)):

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

- `additional_config.enable_fused_mc2=1`: Enables the `dispatch_ffn_combine`/`mega_moe` fused operators. Note: fused MC2 conflicts with `multistream_overlap_shared_expert`.
- `--speculative-config '{"num_speculative_tokens": 1, ...}'`: Minimal MTP speculation during prefill (decode nodes use a higher count, see below).

**Decode node-specific configurations:**

- `--max-num-batched-tokens 164`: Small batch token limit on decode nodes — decode processes one token per sequence per step, so batch tokens should be close to `max-num-seqs`.
- `--speculative-config '{"num_speculative_tokens": 5, ...}'`: Higher MTP speculation count on decode nodes to maximize decode throughput.
- `--additional-config '{"recompute_scheduler_enable": true}'`: Enables the recomputation scheduler. When the decode node KV cache is insufficient, requests are sent back to the prefill node to recompute the KV cache. Recommended on both prefill and decode nodes in PD scenarios.

**Mooncake KV transfer configuration (`--kv-transfer-config`):**

- `"kv_connector": "MooncakeConnectorV1"`: Uses Mooncake as the KV cache transfer connector between prefill and decode nodes.
- `"kv_role": "kv_producer"` / `"kv_consumer"`: `kv_producer` on prefill nodes, `kv_consumer` on decode nodes.
- `"kv_port"`: Port for Mooncake KV transfer communication. Use different port ranges for prefill (`30000`) and decode (`30100`) node groups.
- `"use_ascend_direct": true`: Enables Ascend direct transfer for KV cache, reducing latency.
- `"prefill"` / `"decode"` sections: Specify the `dp_size` and `tp_size` of the prefill and decode node groups respectively. These must match the actual deployment topology (`prefill: dp4 tp8`, `decode: dp32 tp1`).

**Request forwarding (proxy):**

- The proxy command maps every prefill engine endpoint (4 prefiller hosts × 2 ports per node) and every decode engine endpoint (32 decoder hosts/ports) to a single entry point on port `8000`.
- The proxy program can be found in [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).

Please refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py) for further explanation and restrictions of the environment variables above.

##### 5.1.1.4 Multi-Node Deployment over RoCE

This section describes the additional configuration for A3 deployments that use RoCE between physical nodes, such as deployments across SuperPods. Apply these settings in addition to the corresponding A3 deployment commands above.

1. **Logical SuperPod configuration**

    Set a unique `HCCL_LOGIC_SUPERPOD_ID` for each physical node. All vLLM processes on the same physical node must use the same value.

2. **Expert Parallel communication configuration**

    The supported MC2 communication mode depends on whether the Expert Parallel (EP) communication domain spans physical nodes:

    - If the EP communication domain is contained within a physical node, no change to the existing configuration is required. Fused MC2 remains supported.
    - If the EP communication domain spans physical nodes, Fused MC2 is not supported. Hierarchical MC2 communication over RoCE is required and must be enabled by adding `"mc2_comm_alg": "hierarchy"` to `--additional-config`.

**Example**

For the 1P1D (2+2) scenario, configure the logical SuperPod ID as follows:

| Physical node | Configuration |
|---------------|---------------|
| Prefill node 0 | `export HCCL_LOGIC_SUPERPOD_ID=0` |
| Prefill node 1 | `export HCCL_LOGIC_SUPERPOD_ID=1` |
| Decode node 0 | `export HCCL_LOGIC_SUPERPOD_ID=2` |
| Decode node 1 | `export HCCL_LOGIC_SUPERPOD_ID=3` |

Apply the following Fused MC2 setting on every node:

```shell
--additional-config '{..., "enable_fused_mc2": 0, "mc2_comm_alg": "hierarchy"}'
```

Merge `"mc2_comm_alg": "hierarchy"` into the existing `--additional-config` JSON object. The following pattern uses `...` to represent the existing fields:

```text
--additional-config '{..., "mc2_comm_alg": "hierarchy"}'
```

#### 5.1.2 Atlas 800 A2

##### 5.1.2.1 Multi-Node Co-Located Deployment

- `GLM-5.2-w4a8c8`: can be deployed on 2 Atlas 800 A2 (64GB × 8). A single Atlas 800 A2 node (8 × 64GB) cannot fit the w4a8c8 weights, so the 2-node deployment is the minimum configuration for the A2 series.

**node 0**

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxx"

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_IF_IP=$local_ip
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
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
--additional-config '{"multistream_overlap_shared_expert": true}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}'
```

**node 1**

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxx"

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_IF_IP=$local_ip
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
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
--additional-config '{"multistream_overlap_shared_expert": true}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}'
```

Key Parameter Descriptions (in addition to [Single-Node Deployment](#5111-single-node-deployment) and [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)):

The A2 series uses a different optimization stack than A3; DSA-CP is not used here.

**A2-specific environment variables:**

- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000`: Timeout settings for multi-node startup and model execution on the slower A2 platform. Increase them if the engine fails to become ready in time.

##### 5.1.2.2 Prefill-Decode Disaggregation

On Atlas 800 A2, where each node exposes 8 cards, the same global P/D topology (Prefill `DP4 TP8`, Decode `DP8 TP4`) is split across 8 nodes: 4 prefill nodes hosting 1 DP rank each (8 cards per rank), and 4 decode nodes hosting 2 DP ranks each (4 cards per rank). The `launch_online_dp.py` above is reused as-is. The prefill side uses DSA CP; the decode side enables MLAPO and `DYNAMIC_EPLB` with a `FULL_DECODE_ONLY` graph. Both sides enable prefix caching and MTP (`num_speculative_tokens=1` on prefill, `3` on decode). All IPs, NIC names, ports and weight paths below are placeholders.

Please refer to the [KV Cache Pool (Ascend Store) Deployment Guide](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/feature_guide/kv_pool.html) for the KV Cache Pool startup method and the Mooncake configuration file.

`run_dp_template.sh` for the prefill nodes:

```bash
#!/usr/bin/bash
nic_name="<NIC_NAME>"
local_ip="<CURRENT_NODE_IP>"

export HCCL_BUFFSIZE=256
export HCCL_IF_IP=$local_ip
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME=$nic_name
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib:/usr/local/lib:$LD_LIBRARY_PATH
export GLOO_SOCKET_IFNAME=$nic_name
export MOONCAKE_CONFIG_PATH="/mnt/share/scripts/mooncake.json"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --seed 1024 \
    --enable-chunked-prefill \
    --served-model-name glm-5 \
    --max-model-len 256000 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.95 \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "kv_connector_extra_config": {
                    "prefill": {
                        "dp_size": 4,
                        "tp_size": 8
                    },
                    "decode": {
                        "dp_size": 8,
                        "tp_size": 4
                    }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "backend": "mooncake"
                }
            }
        ]
    }
    }' \
    --additional-config '{"enable_dsa_cp": true, "multistream_overlap_shared_expert": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}' \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp", "enforce_eager":true}'
```

`run_dp_template.sh` for the decode nodes:

```bash
#!/usr/bin/bash

nic_name="<NIC_NAME>"
local_ip="<CURRENT_NODE_IP>"

export HCCL_BUFFSIZE=2560
export HCCL_IF_IP=$local_ip
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME=$nic_name
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib:/usr/local/lib:$LD_LIBRARY_PATH
export GLOO_SOCKET_IFNAME=$nic_name
export MOONCAKE_CONFIG_PATH="/mnt/share/scripts/mooncake.json"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --seed 1024 \
    --served-model-name glm-5 \
    --max-model-len 256000 \
    --max-num-batched-tokens 256 \
    --trust-remote-code \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.95 \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_consumer",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_consumer",
                "kv_port": "30100",
                "kv_connector_extra_config": {
                    "prefill": {
                        "dp_size": 4,
                        "tp_size": 8
                    },
                    "decode": {
                        "dp_size": 8,
                        "tp_size": 4
                    }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "load_async": true,
                    "backend": "mooncake"
                }
            }
        ]
    }
    }' \
     --compilation-config \
    '{
        "cudagraph_mode": "FULL_DECODE_ONLY",
    }' \
    --additional-config '{"multistream_overlap_shared_expert": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "recompute_scheduler_enable": true, "enable_mlapo": true}' \
    --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp", "enforce_eager":true}'
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

Key Parameter Descriptions (in addition to [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation)):

This 8-node A2 layout splits the global P/D topology across 8 Atlas 800 A2 nodes: 4 prefill nodes hosting 1 DP rank each (8 cards per rank, `DP4 TP8`) and 4 decode nodes hosting 2 DP ranks each (4 cards per rank, `DP8 TP4`). The `launch_online_dp.py` script is the same as in [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation).

**Prefill node-specific configurations (A2):**

- `--additional-config '{"enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, ...}'`: Both SFA C8 optimizations are enabled in this scenario (vs. `enable_sparse_sfa_c8: false` in the A3 co-located scenarios).

**Decode node-specific configurations (A2):**

- `additional_config.enable_mlapo=true`: MLAPO fusion on decode nodes for memory-bandwidth-bound token generation.
- `--max-num-batched-tokens 256`: Small batch token limit on decode nodes.

**Multi-connector KV transfer configuration (`--kv-transfer-config`):**

- `"kv_connector": "MultiConnector"`: Combines multiple KV transfer connectors.
- `MooncakeConnectorV1`: Mooncake KV cache transfer between prefill and decode nodes (same role as in [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation)).
- `AscendStoreConnector`: The KV Cache Pool (Ascend Store) connector, available since v0.23.0 — see the [KV Cache Pool (Ascend Store) Deployment Guide](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/feature_guide/kv_pool.html).
- `"kv_load_failure_policy": "recompute"`: When a KV block fails to load from the KV pool, the request falls back to recomputation instead of failing.
- `"load_async": true` (decode side): Asynchronously loads KV cache from the pool on decode nodes.
- `"backend": "mooncake"`: The Ascend Store backend used by the connector.

**Environment variables for the A2 PD scenario:**

- `PYTHONHASHSEED=0`: Fixed hash seed for reproducible distributed scheduling.
- `MOONCAKE_CONFIG_PATH="/mnt/share/scripts/mooncake.json"`: Path to the Mooncake configuration file (must exist on each node).
- `HCCL_INTRA_ROCE_ENABLE=1`: Enables HCCL intra-node communication over RoCE.

Please refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py) for further explanation and restrictions of the environment variables above.

### 5.2 1M Context Deployment

Recommended configurations for serving `GLM-5.2` with a 1M context window on Atlas 800 A3 (64GB x 16) and quantized GLM-5.2(W4A8C8) weights:

| Mode | Hardware | Parallelism | Context |
| ---- | -------- | ----------- | ------- |
| Single-node co-located | 1 Atlas 800 A3 (64GB x 16) | `DP1 PP1 TP16 PCP1 DCP16` | `1024000` |
| Dual-node co-located | 2 Atlas 800 A3 (64GB x 16) | `DP4 PP1 TP8 PCP1 DCP8` | `1024000` |
| 1P1D PD disaggregation | 1 prefiller with 2 A3 nodes + 1 decoder with 2 A3 nodes | Prefill `DP4 PP1 TP8 PCP1 DCP8`, Decode `DP4 PP1 TP8 PCP1 DCP8` | `1024000` |

The 1M context scenarios are validated on Atlas 800 A3 only; the A2 series is not validated for 1M context.

!!! warning
    DCP and Sparse Flash Attention C8 (`enable_sparse_sfa_c8`, also referred to as `sfa_c8`) are experimental features in v0.23.0. Enabling them together has known issues in this release, including performance degradation, and is not recommended. The following 1M deployment examples enable DCP and leave `enable_sparse_sfa_c8` disabled.

#### 5.2.1 Single-Node 1M Deployment

Recommended command:

```shell
export HCCL_BUFFSIZE=768
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

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
  --additional-config '{"enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true}, "multistream_overlap_shared_expert": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true,"weight_nz_mode":1}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
  --quantization ascend \
  --enable-expert-parallel \
  --safetensors-load-strategy prefetch
```

Key Parameter Descriptions (in addition to [Single-Node Deployment](#5111-single-node-deployment)):

**1M-specific environment variables:**

- `additional_config.weight_nz_mode=1`: Enables NZ format memory layout for the C8 quantized tensors, required for the 1M context deployment.

**1M-specific vllm serve parameters:**

- `--data-parallel-size 1` / `--pipeline-parallel-size 1` / `--tensor-parallel-size 16`: Single-node parallelism layout `DP1 PP1 TP16`.
- `--prefill-context-parallel-size 1` / `--decode-context-parallel-size 16`: Decode context parallelism (DCP) of 16 for the decode phase; prefill uses PCP 1.
- `--cp-kv-cache-interleave-size 128`: KV cache interleave size for context parallelism.

#### 5.2.2 Dual-Node Co-Located 1M Deployment

Recommended command for both co-located nodes:

```shell
nic_name="<NIC_NAME>"
local_ip="<CURRENT_NODE_IP>"
node_0_ip="<NODE0_IP>"
# Node 0: data_parallel_start_rank=0, server_role_args="--api-server-count 1"
# Node 1: data_parallel_start_rank=2, server_role_args="--headless"
data_parallel_start_rank=0
server_role_args="--api-server-count 1"

export HCCL_BUFFSIZE=768
export HCCL_IF_IP=$local_ip
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

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
  --additional-config '{"enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true}, "multistream_overlap_shared_expert": true,"enable_sparse_li_c8": true, "enable_cpu_binding": true,"weight_nz_mode":1}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
  --quantization ascend \
  --enable-expert-parallel \
  --safetensors-load-strategy prefetch
```

Key Parameter Descriptions (in addition to [Single-Node 1M Deployment](#521-single-node-1m-deployment)):

**Multi-node configuration:**

- `--data-parallel-size 4` / `--data-parallel-size-local 2` / `--data-parallel-start-rank`: Node 0 uses `data_parallel_start_rank=0` with `server_role_args="--api-server-count 1"`; node 1 uses `data_parallel_start_rank=2` with `server_role_args="--headless"`.
- `--data-parallel-address $node_0_ip` / `--data-parallel-rpc-port 16591`: Data parallel master node IP and RPC port, identical on both nodes.
- `--tensor-parallel-size 8` / `--decode-context-parallel-size 8`: `DP4 TP8 DCP8` layout — each node hosts 2 DP ranks × TP8.

#### 5.2.3 PD Disaggregation 1M Deployment

Recommended command for both prefiller nodes:

```shell
nic_name="<NIC_NAME>"
local_ip="<CURRENT_PREFILL_NODE_IP>"
node_p0_ip="<PREFILL_NODE0_IP>"
# Prefiller node 0: data_parallel_start_rank=0, server_role_args="--api-server-count 1"
# Prefiller node 1: data_parallel_start_rank=2, server_role_args="--headless"
data_parallel_start_rank=0
server_role_args="--api-server-count 1"

export HCCL_BUFFSIZE=768
export HCCL_IF_IP=$local_ip
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

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
  --additional-config '{"enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true}, "multistream_overlap_shared_expert": true,"enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": true,"weight_nz_mode":1}' \
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

export HCCL_BUFFSIZE=768
export HCCL_IF_IP=$local_ip
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve <MODEL_PATH> \
  --seed 1024 \
  --host 0.0.0.0 \
  --port 9900 \
  --served-model-name glm-52 \
  --max-model-len 1024000 \
  --max-num-batched-tokens 128 \
  --gpu-memory-utilization 0.93 \
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
  --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true},"multistream_overlap_shared_expert": true,"enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": true,"weight_nz_mode":1}' \
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

Recommended proxy command:

```shell
unset http_proxy
unset https_proxy

python load_balance_proxy_server_example.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts <PREFILL_NODE0_IP> \
  --prefiller-ports 9081 \
  --decoder-hosts <DECODE_NODE0_IP> \
  --decoder-ports 9900
```

Key Parameter Descriptions (in addition to [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation) and [Single-Node 1M Deployment](#521-single-node-1m-deployment)):

**Prefill nodes (1M):**

- `--additional-config '{"recompute_scheduler_enable": true, ...}'`: The recompute scheduler is enabled on both prefill and decode nodes in this scenario.
- `--kv-transfer-config`: Mooncake connector as `kv_producer` with `prefill: dp4 tp8` / `decode: dp4 tp8` (matching the 1M P/D topology, in contrast to `dp32 tp1` decode in the sub-1M PD scenario).

**Decode nodes (1M):**

- `--max-num-batched-tokens 128`: Decode nodes store the large 1M KV cache received from prefill nodes.
- `--kv-transfer-config`: Mooncake connector as `kv_consumer` (`kv_port: 30100`).

## 6 Functional Verification

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

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

### 7.2 Using Language Model Evaluation Harness

Not tested yet.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to [Tuning Guidelines](#92-tuning-guidelines) for tuning based on actual conditions.

The tables below provide recommended parameter configurations for different deployment scenarios. All scenarios are categorized by use case (Low Latency, High Throughput, Long Context) and correspond to the deployment modes documented in [Deployment](#5-deployment).

#### 9.1.1 Table 1: Scenario Overview

> `*Total NPUs` indicates the total number of NPUs used across all nodes. 1 node = 1 Atlas 800 A3 server (64GB × 16 NPUs) or 1 Atlas 800 A2 server (64GB × 8 NPUs).

|Scenario|Deployment Mode|*Total NPUs|Weight Version|Key Considerations|
|--------|---------------|-----------|--------------|------------------|
|Low Latency<br>(64K input)|Dual-Node Co-Located (A3), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|32 (A3)|w4a8c8|dp2 tp16, MTP3, max-num-seqs 8, max-model-len 135000, DSA CP|
|Low Latency<br>(128K input)|Dual-Node Co-Located (A3), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|32 (A3)|w4a8c8|dp2 tp16, MTP3, max-num-seqs 8, max-model-len 135000, DSA CP|
|High Throughput<br>(64K input)|Dual-Node Co-Located (A3), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|32 (A3)|w4a8c8|dp4 tp8, fused MC2, MTP3, max-num-seqs 16, max-model-len 66000, DSA CP|
|High Throughput|PD Disaggregation (A3), [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation)|4 nodes (A3)|w4a8c8|P: dp4 tp8 (max-num-seqs 64, max-num-batched-tokens 8192, MTP1); D: dp32 tp1 (max-num-seqs 32, max-num-batched-tokens 164, MTP5), max-model-len 133120, dedicated P/D nodes, Mooncake KV transfer|
|Long Context<br>(1M)|PD Disaggregation (A3), [PD Disaggregation 1M Deployment](#523-pd-disaggregation-1m-deployment)|4 nodes (A3)|w4a8c8|P/D: dp4 tp8, DCP8, max-model-len 1040000, Mooncake KV transfer|

#### 9.1.2 Table 2: Detailed Node Configuration

> The TP/DP columns show the values **per node** as configured in the Deployment scripts (a node hosting 2 DP ranks of TP8, or 1 DP rank of TP16, uses 16 NPUs).

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#5-deployment)** chapter.

|Scenario|Configuration|NPUs (per node)|TP|DP (per node)|Max Num Seqs|Max Num Batched Tokens|Max Model Len|MTP Spec Num|
|--------|-------------|-----|--|--|------------|----------------------|--------------|-------------|
|Low Latency 64K (A3)|Dual-Node (per node), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|16|16|1|8|8192|135000|3|
|Low Latency 128K (A3)|Dual-Node (per node), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|16|16|1|8|8192|135000|3|
|High Throughput 64K (A3)|Dual-Node (per node), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|16|8|2|16|8192|66000|3|
|High Throughput (A3)|PD — Server-P Node, [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation)|16|8|2|64|8192|133120|1|
|High Throughput (A3)|PD — Server-D Node, [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation)|16|1|16|32|164|133120|5|
|Long Context 1M (A3)|PD — Server-P Node, [PD Disaggregation 1M Deployment](#523-pd-disaggregation-1m-deployment)|16|8|2|8|16384|1040000|1|
|Long Context 1M (A3)|PD — Server-D Node, [PD Disaggregation 1M Deployment](#523-pd-disaggregation-1m-deployment)|16|8|2|32|128|1040000|3|

> On PD decode nodes, `--max-num-batched-tokens` is configured as `(MTP Spec Num + 1) × Max Num Seqs` — each sequence generates one target token plus the speculated MTP tokens per decode step.
>
> For complete startup commands and detailed parameter descriptions, please refer to the deployment examples and Key Parameter Descriptions in [Deployment](#5-deployment).

#### 9.1.3 Table 3: Performance-Related Parameter Tuning Guide

|Parameter|Low Latency|High Throughput|Long Context|Description|
|---------|-----------|---------------|-------------|-----------|
|`--max-num-batched-tokens`|Lower (4096–8192)|Higher for prefill (8192)|Higher for prefill (16384)|Controls batch size per step. Lower values reduce per-step latency; higher values improve prefill throughput.|
|`--gpu-memory-utilization`|0.92–0.95|0.90–0.95|0.75–0.93|NPU memory fraction. 1M scenarios must reserve memory for the huge KV cache, so use lower values (0.75–0.80 on prefill/co-located).|
|`num_speculative_tokens` (MTP)|3–5|3–5|1 (prefill) / 3 (decode)|MTP speculation count. Higher values improve decode throughput at the cost of memory for the draft model KV cache. Use `1` on prefill nodes in PD mode.|
|`enable_dsa_cp`|Optional|Enable (prefill nodes)|Enable (prefill nodes)|DSA context parallelism accelerates long-context prefill.|
|`enable_sparse_sfa_c8` / `enable_sparse_li_c8`|`false` / `true`|`true` / `true` (PD)|`false` / `true`|`enable_sparse_sfa_c8` is an experimental SFA optimization for the C8 quantized model; do not combine it with DCP in v0.23.0 because of known issues. `enable_sparse_li_c8` is independent and remains recommended.|
|`enable_balance_scheduling`|Enable (single-node)|Enable|Disable in PD mode|Improves output throughput and reduces TPOT in the v1 scheduler. TTFT may degrade in some scenarios; not recommended when Prefill-Decode is separated.|
|`additional_config.enable_mlapo`|—|1 (A2 P/D nodes)|—|Fusion operator that significantly improves performance but consumes more NPU memory. On A3 used on decode nodes only.|
|`cudagraph_mode`|FULL_DECODE_ONLY|FULL_DECODE_ONLY (decode)|FULL_DECODE_ONLY (decode)|Graph capture for the decode phase only. Prefill nodes in PD mode use `--enforce-eager` instead.|

### 9.2 Tuning Guidelines

For general performance tuning methods, refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md).

For detailed feature descriptions and configuration options, refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md).

For environment variable descriptions and constraints, refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py).

## 10 FAQ

- **Q: How to enable function calling for GLM-5.2?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```
