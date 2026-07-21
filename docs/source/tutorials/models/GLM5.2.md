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
- `GLM-5.2-w4a8c8`: requires 1 Atlas 800 A3 (128G × 8) node or 2 Atlas 800 A2 (64G × 8) node.[Download model weight](https://modelscope.cn/models/Eco-Tech/GLM-5.2-w4a8c8).
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

- You can use our official docker image to run GLM-5.2 directly.
- [KV Cache Pool (Ascend Store) Deployment Guide](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/feature_guide/kv_pool.html)

=== "A3 series"

    Start the docker image on your each node.

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

## Deployment

### Single-node Deployment

- Quantized model `GLM-5.2-w4a8c8` can be deployed on 1 Atlas 800 A3 (64G × 16) .

Run the following script to execute online inference.

```shell
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
--host 0.0.0.0 \
--port 8077 \
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
--additional-config '{"enable_dsa_cp": true,"enable_sparse_sfa_c8": false, "enable_sparse_li_c8": true,"enable_balance_scheduling": true,"multistream_overlap_shared_expert":true}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'

```

**Notice:**
The parameters are explained as follows:

- For single-node deployment, we recommend using `dp1tp16` and turn off expert parallel in low-latency scenarios.

### Multi-node Deployment

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

=== "A3 series"

    - `GLM-5.2-w4a8c8`: can be deployed on 2 Atlas 800 A3 (64G × 16).

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
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
    --host 0.0.0.0 \
    --port 8077 \
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
    --additional-config '{"enable_dsa_cp": true,"enable_sparse_sfa_c8": false, "enable_sparse_li_c8": true,"enable_balance_scheduling": true,"fuse_muls_add":true,"multistream_overlap_shared_expert":true,"c8_enable_reshape_optim":false,    "enable_reduce_sample": "True"}'  \
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

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
    --host 0.0.0.0 \
    --port 8077 \
    --headless \
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
    --additional-config '{"enable_dsa_cp": true,"enable_sparse_sfa_c8": false, "enable_sparse_li_c8": true,"enable_balance_scheduling": true,"fuse_muls_add":true,"multistream_overlap_shared_expert":true,"c8_enable_reshape_optim":false,     "enable_reduce_sample": "True"}'  \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
    ```

=== "A2 series"

    - `GLM-5.2-w4a8c8`: can be deployed on 2 Atlas 800 A2 (64G × 32).

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

    export VLLM_VERSION=0.21.0
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
    --block-size 128 \
    --async-scheduling \
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

    export VLLM_VERSION=0.21.0
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
    --async-scheduling \
    --additional-config '{"multistream_overlap_shared_expert": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

### Prefill-Decode Disaggregation

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

    1. Prefill node 0

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export HCCL_OP_EXPANSION_MODE="AIV"
        
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400
        
        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
            --host 0.0.0.0 \
            --port $2 \
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
            --additional-config '{"recompute_scheduler_enable" : false,"multistream_overlap_shared_expert": true, "enable_dsa_cp":true,"enable_sparse_sfa_c8": false, "enable_sparse_li_c8": true,"c8_enable_reshape_optim":false}' \
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

        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export HCCL_OP_EXPANSION_MODE="AIV"
        
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400
        
        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
            --host 0.0.0.0 \
            --port $2 \
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
            --additional-config '{"recompute_scheduler_enable" : false,"multistream_overlap_shared_expert": true, "enable_dsa_cp":true,"enable_sparse_sfa_c8": false, "enable_sparse_li_c8": true,"c8_enable_reshape_optim":false}' \
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

        export HCCL_OP_EXPANSION_MODE="AIV"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        
        #Mooncake
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=256
        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
            --host 0.0.0.0 \
            --port $2 \
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
            --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert":true,"enable_sparse_sfa_c8": false, "enable_sparse_li_c8": true}' \
            --trust-remote-code \
            --max-num-seqs 32 \
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

        export HCCL_OP_EXPANSION_MODE="AIV"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        
        #Mooncake
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=256
        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
            --host 0.0.0.0 \
            --port $2 \
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
            --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert":true,"enable_sparse_sfa_c8": false, "enable_sparse_li_c8": true}' \
            --trust-remote-code \
            --max-num-seqs 32 \
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
    python launch_online_dp.py --dp-size 4 --tp-size 8  --dp-size-local 1 --dp-rank-start 0 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    ```

2. Prefill node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 4 --tp-size 8  --dp-size-local 1 --dp-rank-start 1 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    ```

3. Decode node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 4 --dp-rank-start 0 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    ```

4. Decode node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 4 --dp-rank-start 4 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
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

#### Deployment on 8 Atlas 800 A2

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
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib:/usr/local/lib:$LD_LIBRARY_PATH
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH="/mnt/share/scripts/mooncake.json"
export HCCL_INTRA_ROCE_ENABLE=1
export ACL_OP_INIT_MODE=1

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
    --async-scheduling \
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
    --additional-config '{"enable_flashcomm1": true, "enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm": false, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": false}' \
    --profiler-config \
    '{
        "profiler": "torch", 
        "torch_profiler_dir": "/mnt/share/xxx/prof", 
        "torch_profiler_with_stack": false
    }' \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp", "enforce_eager":true}'
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

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=2560
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib:/usr/local/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/mooncake:$LD_LIBRARY_PATH

export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH="/mnt/share/scripts/mooncake.json"
export HCCL_INTRA_ROCE_ENABLE=1

export ACL_OP_INIT_MODE=1

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
    --async-scheduling \
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
        "cudagraph_capture_sizes": [4,8,16,24,32,40,48,56,64,96,128,160,192,224,256,298,320,352,384]
    }' \
    --profiler-config \
    '{
        "profiler": "torch", 
        "torch_profiler_dir": "/mnt/share/xxx/prof", 
        "torch_profiler_with_stack": false
    }' \
    --additional-config '{"enable_flashcomm1": false, "enable_dsa_cp": false, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm": false, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": true}' \
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

**Notice:**

Some configurations for optimization are shown below:

- `VLLM_ASCEND_ENABLE_FLASHCOMM1`: Enable FlashComm optimization to reduce communication and computation overhead on prefill node. With FlashComm enabled, layer_sharding list cannot include o_proj as an element.
- `VLLM_ASCEND_ENABLE_FUSED_MC2`: Enable the dispatch_ffn_combine fused operator.

Please refer to the following python file for further explanation and restrictions of the environment variables above: [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py)

### 1M Context Configuration

Recommended configurations for serving `GLM-5.2` with a 1M context window on Atlas 800 A3 (64G x 16) and quantized GLM-5.2(W4A8C8) weights:

| Mode | Hardware | Parallelism | Context |
| ---- | -------- | ----------- | ------- |
| Single-node co-located | 1 Atlas 800 A3 (64G x 16) | `DP1 PP1 TP16 PCP1 DCP16` | `1024000` |
| Dual-node co-located | 2 Atlas 800 A3 (64G x 16) | `DP4 PP1 TP8 PCP1 DCP8` | `1024000` |
| 1P1D PD disaggregation | 1 prefiller with 2 A3 nodes + 1 decoder with 2 A3 nodes | Prefill `DP4 PP1 TP8 PCP1 DCP8`, Decode `DP4 PP1 TP8 PCP1 DCP8` | `1024000` |

#### Single-Node 1M Deployment

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
  --additional-config '{"enable_flashcomm1": true, "enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm": false, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": false}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
  --quantization ascend \
  --enable-expert-parallel \
  --safetensors-load-strategy prefetch
```

#### Dual-Node Co-Located 1M Deployment

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
  --additional-config '{"enable_flashcomm1": true, "enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm": false, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": false}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
  --quantization ascend \
  --enable-expert-parallel \
  --safetensors-load-strategy prefetch
```

#### PD Disaggregation 1M Deployment

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
  --additional-config '{"enable_flashcomm1": true, "enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm": false, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": true}' \
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
  --additional-config '{"enable_flashcomm1": false, "enable_dsa_cp": false, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm": false, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": true}' \
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

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/) for more details.

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#deployment)** chapter.

## FAQ

- **Q: How to enable function calling for GLM-5.2?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```
