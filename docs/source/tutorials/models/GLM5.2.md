# GLM-5.2

## 1 Introduction

[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) uses a Mixture-of-Experts (MoE) architecture and targets complex systems engineering and long-horizon agentic tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Model Weight

- `GLM-5.2`(BF16 version): requires 2 Atlas 800 A3 (128GB × 8) node or 4 Atlas 800 A2 (64GB × 8) node.[Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-5.2).
- `GLM-5.2-w8a8`: requires 1 Atlas 800 A3 (128GB × 8) node or 2 Atlas 800 A2 (64GB × 8) node.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w8a8).
- `GLM-5.2-w8a8c8`(Quantized version for Atlas 800 A3): requires 2 Atlas 800 A3 (64GB × 16) node.[Download model weight](https://modelers.cn/models/Eco-Tech/GLM-5.2-w8a8c8).
- `GLM-5.2-w4a8c8`: requires 1 Atlas 800 A3 (128GB × 8) node or 2 Atlas 800 A2 (64GB × 8) node.[Download model weight](https://modelscope.cn/models/Eco-Tech/GLM-5.2-w4a8c8).
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

## 4 Installation

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

## 5 Deployment

### 5.1 Context Below 1M

#### 5.1.1 Single-node Deployment

- Quantized model `GLM-5.2-w4a8c8` can be deployed on 1 Atlas 800 A3 (64GB × 16) .

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

#### 5.1.2 Multi-node Deployment

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

=== "A3 series"

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

    - `GLM-5.2-w4a8c8`: can be deployed on 2 Atlas 800 A2 (64GB × 32).

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
    export TASK_QUEUE_ENABLE=1
    export CPU_AFFINITY_CONF=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1200

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
    export TASK_QUEUE_ENABLE=1
    export CPU_AFFINITY_CONF=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1200

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

#### 5.1.3 Prefill-Decode Disaggregation

We'd like to show the deployment guide of `GLM-5.2` on multi-node environment with 1P1D for better performance.

In the PD disaggregation scenario, Mooncake is used as the KV cache transfer connector between the prefill and decode nodes. Please refer to [KV Cache Pool (Ascend Store) Deployment Guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/kv_pool.md) for the Mooncake configuration.

##### 5.1.3.1 Deployment on 4 Atlas 800 A3

Prefill-Decode disaggregation with the `GLM-5.2-w8a8c8` weights can be deployed on 4 Atlas 800 A3 (64GB × 16).

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
        parser.add_argument("--dp-size", type=int, required=True, help="Data parallel size.")
        parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
        parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size.")
        parser.add_argument("--dp-size-local", type=int, default=-1, help="Local data parallel size.")
        parser.add_argument("--dp-rank-start", type=int, default=0, help="Starting rank for data parallel.")
        parser.add_argument("--dp-address", type=str, required=True, help="IP address for data parallel master node.")
        parser.add_argument("--dp-rpc-port", type=str, default="12321", help="Port for data parallel master node.")
        parser.add_argument("--vllm-start-port", type=int, default=8000, help="Starting port for the engine.")
        return parser.parse_args()

    args = parse_args()
    dp_size = args.dp_size
    tp_size = args.tp_size
    pp_size = args.pp_size
    dp_size_local = args.dp_size_local
    if dp_size_local == -1:
        dp_size_local = dp_size
    dp_rank_start = args.dp_rank_start
    dp_address = args.dp_address
    dp_rpc_port = args.dp_rpc_port
    vllm_start_port = args.vllm_start_port
    gpus_per_dp_rank = tp_size * pp_size

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
            str(pp_size),
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

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

        # pp=2
        export VLLM_PP_LAYER_PARTITION="41,37"

        vllm serve <MODEL_PATH> \
            --host 0.0.0.0 \
            --port $2 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr $local_ip \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 0 \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 202752 \
            --safetensors-load-strategy 'prefetch' \
            --additional-config '{"fuse_muls_add":true, "enable_dsa_cp":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "c8_enable_reshape_optim":true}' \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --enable-prefix-caching \
            --max-num-seqs 64 \
            --quantization ascend \
            --gpu-memory-utilization 0.85 \
            --api-server-count 1 \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "0",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "41,37"},
                    "decode": { "dp_size": 16, "tp_size": 2}
                }
            }'

        ```

    2. Prefill node 1

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        # The value of node_p0_ip must be consistent with the value of local_ip set in prefill node 0 (p0, PP master node)
        node_p0_ip="xxxx"

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

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

        # pp=2
        export VLLM_PP_LAYER_PARTITION="41,37"

        vllm serve <MODEL_PATH> \
            --host 0.0.0.0 \
            --port $2 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr $node_p0_ip \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 1 \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 202752 \
            --safetensors-load-strategy 'prefetch' \
            --additional-config '{"fuse_muls_add":true, "enable_dsa_cp":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "c8_enable_reshape_optim":true}' \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --enable-prefix-caching \
            --max-num-seqs 64 \
            --quantization ascend \
            --gpu-memory-utilization 0.85 \
            --headless \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "0",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "41,37"},
                    "decode": { "dp_size": 16, "tp_size": 2}
                }
            }'
        ```

    3. Decode node 0

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        # d0: api server on this node; d1: --headless
        server_role_args="--api-server-count 1"

        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
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

        export VLLM_ASCEND_ENABLE_MLAPO=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

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
            --served-model-name glm-5 \
            --max-model-len 202752 \
            --safetensors-load-strategy 'prefetch' \
            --max-num-batched-tokens 192 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}' \
            --trust-remote-code \
            --max-num-seqs 32 \
            --gpu-memory-utilization 0.90 \
            ${server_role_args} \
            --async-scheduling \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_consumer",
                "kv_port": "30200",
                "engine_id": "2",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "41,37"},
                    "decode": { "dp_size": 16, "tp_size": 2}
                }
            }'
        ```

    4. Decode node 1

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        # d0: api server on this node; d1: --headless
        server_role_args="--headless"

        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
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

        export VLLM_ASCEND_ENABLE_MLAPO=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

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
            --served-model-name glm-5 \
            --max-model-len 202752 \
            --safetensors-load-strategy 'prefetch' \
            --max-num-batched-tokens 192 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}' \
            --trust-remote-code \
            --max-num-seqs 32 \
            --gpu-memory-utilization 0.90 \
            ${server_role_args} \
            --async-scheduling \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_consumer",
                "kv_port": "30200",
                "engine_id": "2",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "41,37"},
                    "decode": { "dp_size": 16, "tp_size": 2}
                }
            }'
        ```

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
    # change ip to your own
    python launch_online_dp.py --dp-size 16 --tp-size 2 --pp-size 1 --dp-size-local 8 --dp-rank-start 0 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    ```

4. Decode node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 16 --tp-size 2 --pp-size 1 --dp-size-local 8 --dp-rank-start 8 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
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
    --prefiller-ports \
      9081 \
    --decoder-hosts \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
      $node_d0_ip \
    --decoder-ports \
      9900 9901 9902 9903 9904 9905 9906 9907
```

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario.

**PP2 prefill node-specific configurations (p0/p1):**

- `VLLM_PP_LAYER_PARTITION="41,37"` / `--pipeline-parallel-size 2` / `--nnodes 2` / `--node-rank`: The prefill engine is split as PP2 over the two prefill nodes — the 78 layers are partitioned as `41/37`. `--node-rank` is `0` on prefill node 0 (p0) and `1` on prefill node 1 (p1).
- `--distributed-executor-backend mp` / `--master-addr` / `--master-port 7060`: PP runs over two nodes with the `mp` executor (no Ray required). `--master-addr` is `$local_ip` on prefill node 0 (p0, PP master) and `$node_p0_ip` on prefill node 1 (p1).
- `VLLM_ASCEND_ENABLE_FUSED_MC2=1`: Enables the fused `dispatch_ffn_combine`/`mega_moe` operators.
- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`: Enables FlashComm optimization to reduce communication and computation overhead on prefill nodes.
- `--enforce-eager`: The prefill side runs in eager mode (the `FULL_DECODE_ONLY` graph capture is used on the decode side instead).
- `--speculative-config '{"num_speculative_tokens": 1, ...}'`: Minimal MTP speculation during prefill (decode nodes use a higher count, see below).
- `fuse_muls_add` / `enable_dsa_cp` / `enable_sparse_sfa_c8` / `enable_sparse_li_c8` / `c8_enable_reshape_optim` (in `--additional-config`): Mul-Add fusion, DSA context parallelism for long-context prefill, the SFA/LI sparse attention optimizations and the reshape optimization of the C8 quantized model.

**Decode node-specific configurations (d0/d1):**

- `VLLM_ASCEND_ENABLE_MLAPO=1`: MLAPO fusion on decode nodes for memory-bandwidth-bound token generation.
- `--max-num-batched-tokens 192`: Small batch token limit on decode nodes — decode processes one target token plus the 5 speculated MTP tokens per sequence per step (`(5 + 1) × 32`).
- `--speculative-config '{"num_speculative_tokens": 5, ...}'`: Higher MTP speculation count on decode nodes to maximize decode throughput.
- `recompute_scheduler_enable=true` (in `--additional-config`): Enables the recomputation scheduler. When the decode node KV cache is insufficient, requests are sent back to the prefill node to recompute the KV cache.
- `--async-scheduling`: Async scheduling on the decode side.

**Mooncake KV transfer configuration (`--kv-transfer-config`):**

- `"kv_connector": "MooncakeConnectorV1"`: Uses Mooncake as the KV cache transfer connector between prefill and decode nodes.
- `"kv_role": "kv_producer"` / `"kv_consumer"`: `kv_producer` on prefill nodes, `kv_consumer` on decode nodes.
- `"kv_port"`: Port for Mooncake KV transfer communication. Use different port ranges for prefill (`30000`) and decode (`30200`) node groups.
- `"use_ascend_direct": true`: Enables Ascend direct transfer for KV cache, reducing latency.
- `"engine_id"`: Fixed engine id of the Mooncake connector per role group: `0` for the prefill engine (shared by the two PP ranks) and `2` for the decode engines.
- `"prefill"` / `"decode"` sections: Specify the parallelism of the prefill and decode node groups respectively. These must match the actual deployment topology (`prefill: dp_size 1, pp_size 2, tp_size 16, pp_layer_partition "41,37"`, `decode: dp_size 16, tp_size 2`).

**Request forwarding (proxy):**

- The proxy program can be found in [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).

Please refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py) for further explanation and restrictions of the environment variables above.

##### 5.1.3.2 Deployment on 8 Atlas 800 A2

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
- `VLLM_ASCEND_ENABLE_FUSED_MC2`: Enable the dispatch_ffn_combine/mega_moe fused operator.

Please refer to the following python file for further explanation and restrictions of the environment variables above: [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py)

### 5.2 1M Context Configuration

Recommended configurations for serving `GLM-5.2` with a 1M context window on Atlas 800 A3 (64GB x 16) and quantized GLM-5.2(W4A8C8) weights:


#### 5.2.1 Single-Node 1M Deployment

- Quantized model `GLM-5.2-w4a8c8` can be deployed on 1 Atlas 800 A3 (64GB × 16) for the 1M context.

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

#### 5.2.2 Dual-Node Co-Located 1M Deployment

- `GLM-5.2-w4a8c8` can be deployed on 2 Atlas 800 A3 (64GB × 16) for the 1M context.

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

#### 5.2.3 PD Disaggregation 1M Deployment

PD disaggregation for the 1M context with the `GLM-5.2-w8a8c8` weights can be deployed on 4 Atlas 800 A3 (64GB × 16).

Before you start, please

prepare the script `launch_online_dp.py` on each node (used by the prefill and decode nodes below to pass the engine port and DP parameters):

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

prepare the script `run_dp_template.sh` on each node.

1. Prefill node 0

    ```shell
    nic_name="<NIC_NAME>" # change to your own nic name
    local_ip="<CURRENT_PREFILL_NODE_IP>" # change to your own ip
    # p0: api server on this node; p1: --headless
    server_role_args="--api-server-count 1"

    export VLLM_ASCEND_ENABLE_FUSED_MC2=1
    export HCCL_OP_EXPANSION_MODE="AIV"

    export VLLM_ENGINE_READY_TIMEOUT_S=100000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=204
    export HCCL_CONNECT_TIMEOUT=7200
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export MF_GROUP_JOIN_MAX_TIMEOUT=1200

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

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

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
        --prefill-context-parallel-size 1 \
        --decode-context-parallel-size 16 \
        --cp-kv-cache-interleave-size 128 \
        --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
        --seed 1024 \
        --served-model-name glm-5 \
        --max-model-len 1048576 \
        --safetensors-load-strategy 'prefetch' \
        --additional-config '{"fuse_muls_add":true, "multistream_overlap_shared_expert": true, "enable_dsa_cp":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "c8_enable_reshape_optim":true, "mega_moe_max_tokens": 8192}' \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --enable-prefix-caching \
        --max-num-seqs 64 \
        --quantization ascend \
        --gpu-memory-utilization 0.85 \
        ${server_role_args} \
        --enforce-eager \
        --enable-auto-tool-choice \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --kv-transfer-config '{
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "engine_id": "0",
            "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 2, "tp_size": 16},
                "decode": { "dp_size": 4, "tp_size": 8}
            }
        }'
    ```

2. Prefill node 1

    DP rank `1`, engine port `9082`. Same script as p0 with `server_role_args="--headless"` (non-master node, no API server); `--data-parallel-address` (`$5`) still points to prefill node 0 (DP master node).

    ```shell
    nic_name="<NIC_NAME>" # change to your own nic name
    local_ip="<CURRENT_PREFILL_NODE_IP>" # change to your own ip
    # p0: api server on this node; p1: --headless
    server_role_args="--headless"

    export VLLM_ASCEND_ENABLE_FUSED_MC2=1
    export HCCL_OP_EXPANSION_MODE="AIV"

    export VLLM_ENGINE_READY_TIMEOUT_S=100000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=204
    export HCCL_CONNECT_TIMEOUT=7200
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export MF_GROUP_JOIN_MAX_TIMEOUT=1200

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

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

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
        --prefill-context-parallel-size 1 \
        --decode-context-parallel-size 16 \
        --cp-kv-cache-interleave-size 128 \
        --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
        --seed 1024 \
        --served-model-name glm-5 \
        --max-model-len 1048576 \
        --safetensors-load-strategy 'prefetch' \
        --additional-config '{"fuse_muls_add":true, "multistream_overlap_shared_expert": true, "enable_dsa_cp":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "c8_enable_reshape_optim":true, "mega_moe_max_tokens": 8192}' \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --enable-prefix-caching \
        --max-num-seqs 64 \
        --quantization ascend \
        --gpu-memory-utilization 0.85 \
        ${server_role_args} \
        --enforce-eager \
        --enable-auto-tool-choice \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --kv-transfer-config '{
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "engine_id": "0",
            "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 2, "tp_size": 16},
                "decode": { "dp_size": 4, "tp_size": 8}
            }
        }'
    ```

3. Decode node 0

    ```shell
    nic_name="<NIC_NAME>" # change to your own nic name
    local_ip="<CURRENT_DECODE_NODE_IP>" # change to your own ip

    # d0: api server on this node; d1: --headless
    server_role_args="--api-server-count 1"

    export VLLM_ASCEND_ENABLE_FUSED_MC2=1
    export HCCL_OP_EXPANSION_MODE="AIV"

    export VLLM_ENGINE_READY_TIMEOUT_S=100000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=204
    export HCCL_CONNECT_TIMEOUT=7200
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export MF_GROUP_JOIN_MAX_TIMEOUT=1200

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    #Mooncake
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1

    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=400

    export ACL_OP_INIT_MODE=1
    export ASCEND_A3_ENABLE=1
    export TASK_QUEUE_ENABLE=1

    export VLLM_ASCEND_ENABLE_MLAPO=1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

    vllm serve <MODEL_PATH> \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --enable-expert-parallel \
        --prefill-context-parallel-size 1 \
        --decode-context-parallel-size 8 \
        --cp-kv-cache-interleave-size 128 \
        --seed 1024 \
        --served-model-name glm-5 \
        --max-model-len 1048576 \
        --safetensors-load-strategy 'prefetch' \
        --max-num-batched-tokens 192 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
        --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable":true, "multistream_overlap_shared_expert":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}' \
        --trust-remote-code \
        --max-num-seqs 32 \
        --gpu-memory-utilization 0.90 \
        ${server_role_args} \
        --async-scheduling \
        --quantization ascend \
        --enable-auto-tool-choice \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --kv-transfer-config '{
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "30200",
            "engine_id": "2",
            "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 2, "tp_size": 16},
                "decode": { "dp_size": 4, "tp_size": 8}
            }
        }'
    ```

4. Decode node 1

    ```shell
    nic_name="<NIC_NAME>" # change to your own nic name
    local_ip="<CURRENT_DECODE_NODE_IP>" # change to your own ip

    # d0: api server on this node; d1: --headless
    server_role_args="--headless"

    export VLLM_ASCEND_ENABLE_FUSED_MC2=1
    export HCCL_OP_EXPANSION_MODE="AIV"

    export VLLM_ENGINE_READY_TIMEOUT_S=100000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=204
    export HCCL_CONNECT_TIMEOUT=7200
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export MF_GROUP_JOIN_MAX_TIMEOUT=1200

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    #Mooncake
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1

    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=400

    export ACL_OP_INIT_MODE=1
    export ASCEND_A3_ENABLE=1
    export TASK_QUEUE_ENABLE=1

    export VLLM_ASCEND_ENABLE_MLAPO=1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

    vllm serve <MODEL_PATH> \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --enable-expert-parallel \
        --prefill-context-parallel-size 1 \
        --decode-context-parallel-size 8 \
        --cp-kv-cache-interleave-size 128 \
        --seed 1024 \
        --served-model-name glm-5 \
        --max-model-len 1048576 \
        --safetensors-load-strategy 'prefetch' \
        --max-num-batched-tokens 192 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
        --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable":true, "multistream_overlap_shared_expert":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}' \
        --trust-remote-code \
        --max-num-seqs 32 \
        --gpu-memory-utilization 0.90 \
        ${server_role_args} \
        --async-scheduling \
        --quantization ascend \
        --enable-auto-tool-choice \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --kv-transfer-config '{
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "30200",
            "engine_id": "2",
            "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 2, "tp_size": 16},
                "decode": { "dp_size": 4, "tp_size": 8}
            }
        }'
    ```

Once the preparation is done, start the server on each node:

1. Prefill node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 0 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    ```

2. Prefill node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 1 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9082
    ```

3. Decode node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    ```

4. Decode node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 2 --dp-rank-start 2 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    ```

To set up request forwarding, run the following script on any machine.

```shell
unset http_proxy
unset https_proxy

python load_balance_proxy_server_example.py \
    --port 8000 \
    --host 0.0.0.0 \
    --prefiller-hosts \
      $node_p0_ip \
    --prefiller-ports \
      9081 \
    --decoder-hosts \
      $node_d0_ip \
      $node_d0_ip \
    --decoder-ports \
      9900 9901
```

Key Parameter Descriptions (in addition to [Prefill-Decode Disaggregation](#513-prefill-decode-disaggregation) and [Single-Node 1M Deployment](#521-single-node-1m-deployment)):

The `run_dp_template.sh` templates use the positional parameters passed by `launch_online_dp.py`: `$1` = visible devices, `$2` = engine port, `$3` = data-parallel-size, `$4` = data-parallel-rank, `$5` = data-parallel-address, `$6` = data-parallel-rpc-port, `$7` = tensor-parallel-size.

**1M-specific environment variables (both prefiller and decoder nodes):**

- `VLLM_ENGINE_READY_TIMEOUT_S=100000` / `VLLM_RPC_TIMEOUT=3600000` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000` / `HCCL_EXEC_TIMEOUT=204` / `HCCL_CONNECT_TIMEOUT=7200` / `ASCEND_CONNECT_TIMEOUT=10000` / `ASCEND_TRANSFER_TIMEOUT=10000` / `MF_GROUP_JOIN_MAX_TIMEOUT=1200`: Timeout settings for the slow multi-node startup and model execution of the 1M context scenario. Increase them if the engine fails to become ready in time.

**Prefill nodes (1M):**

- `--speculative-config '{"num_speculative_tokens": 1, ...}'`: Minimal MTP speculation during prefill.
- `mega_moe_max_tokens=8192` (in `--additional-config`): Per-rank token capacity after dispatch in the fused MC2/MegaMoe path.
- `--kv-transfer-config`: Mooncake connector as `kv_producer` with `prefill: dp2 tp16` / `decode: dp4 tp8`.

**Decode nodes (1M):**

- `--max-num-batched-tokens 192`: Small batch token limit on decode nodes — decode nodes store the large 1M KV cache received from prefill nodes.
- `VLLM_ASCEND_ENABLE_MLAPO=1`: MLAPO fusion on decode nodes for memory-bandwidth-bound token generation.
- `--kv-transfer-config`: Mooncake connector as `kv_consumer` (`kv_port: 30200`).

Please refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py) for further explanation and restrictions of the environment variables above.

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

Expected Result:

The service returns HTTP 200 OK. The JSON response contains the `choices` field with the generated text, along with usage statistics:

```json
{
    "id": "chatcmpl-90e6de0720743e72",
    "object": "text_completion",
    "created": 1784891079,
    "model": "glm-52",
    "choices": [
        {
            "index": 0,
            "text": "here,and it's not just about chatbots. It's about AI agents",
            "logprobs":null,
            "finish_reason”:"length",
            "stop_reason":null,
            "token_ids":null,
            "prompt_logprobs":null,
            "prompt_token_ids":null,
            "routed_experts":null
        }
    ],
    "service_tier":null,
    "system fingerprint":"vllm-0.26.0-tp16-ep-2a76151d",
    "usage":{
        "prompt tokens :5,
        "total tokens :21,
        "completion tokens":16,
        "prompt tokens details :null
        },
        "ky transfer params":null
    }
}
```

## 7 Accuracy Evaluation

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

| dataset | version | metric | mode | vllm-api-general-chat | note |
| ----- | ----- | ----- | ----- | ----- | ----- |
| AIME2026 | - | accuracy | gen | 93.33 | 4 Atlas 800 A3 (64GB × 16) |
| GPQA | - | accuracy | gen | 90.4 | 8 Atlas 800 A3 (64GB × 16) |

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#5-deployment)** chapter.

## 9 Performance Tuning

### 9.1 Tested Performance Cases

#### Table 1: Prefill-Only Test Cases

|Scenario|Configuration|NPUs|TP|DP|PP|Max Model Len|MTP Speculation Num|
|--------|-------------|-----|--|--|--|-------------------|--------------------|
|Decode-only|Server-P Node|32 (A3)|16|1|2|198k|1|
|Decode-only|Server-D Node|32 (A3)|16|2|1|198k|5|
|Prefill-only|Server-P Node|32 (A3)|16|1|2|198k|1|
|Prefill-only|Server-D Node|32 (A3)|16|2|1|198k|5|
|high concurrency|Server-P Node|32 (A3)|16|1|2|198k|1|
|high concurrency|Server-D Node|32 (A3)|16|2|1|198k|5|
|high concurrency(1M)|Server-P Node|32 (A3)|2|16|-|198k|1|
|high concurrency(1M)|Server-D Node|32 (A3)|4|8|-|198k|5|

### 9.2 Recommended Configurations

#### Table 2: Optimizations Requiring Explicit Enablement

|Optimization|Scenario|Enablement|Principle (Benefits)|Notes|
|------------|--------|----------|---------------------|-----|
|FlashComm_v1|A3 prefill nodes / co-located nodes|`export VLLM_ASCEND_ENABLE_FLASHCOMM1=1`|Splits AllReduce into Reduce-Scatter and All-Gather, improving prefill throughput and reducing communication latency|Not available when `layer_sharding` includes `o_proj`|
|Fused MC2|A3 prefill nodes|`export VLLM_ASCEND_ENABLE_FUSED_MC2=1`|Replaces ALLTOALL+MC2 with the `dispatch_ffn_combine`/`dispatch_gmm_combine_decode` operators, reducing MoE communication overhead and improving MoE inference performance|`dispatch_ffn_combine` only for w8a8, EP≤32, non-MTP, non-dynamic-EPLB; conflicts with `multistream_overlap_shared_expert` (the latter is auto-disabled)|
|MLAPO|A3 co-located / PD decode nodes; A2 P/D nodes|`export VLLM_ASCEND_ENABLE_MLAPO=1`|Fuses the MLA preprocess operations, significantly improving decode performance|Consumes more NPU memory; in PD scenarios enable on decode nodes only|
|DSA CP|A3 prefill nodes; long context|`--additional-config '{"enable_dsa_cp": true}'`|DSA context parallelism accelerates long-context prefill, reducing TTFT for long prompts|In the reference configs, enabled on co-located nodes and PD prefill nodes; PD decode nodes use decode context parallelism instead|
|Sparse SFA C8|A3 (w8a8c8); long-context prefill|`--additional-config '{"enable_sparse_sfa_c8": true}'`|Sparse Flash Attention skips unnecessary attention computation of the C8 quantized model, accelerating long-context prefill|Experimental in v0.23.0. On the w8a8c8 weights it can be combined with DCP/context parallelism (as in the reference configs in this document); on the w4a8c8 weights enabling both together has known issues and is not recommended|
|Sparse LI C8|A3 (w8a8c8)|`--additional-config '{"enable_sparse_li_c8": true}'`|Sparse attention optimization reduces computation of the C8 quantized model, improving throughput|Independent of `enable_sparse_sfa_c8`|
|Recompute Scheduler|A3 decode nodes|`--additional-config '{"recompute_scheduler_enable": true}'`|Recomputes KV cache on prefill nodes when decode KV cache is insufficient, avoiding decode-side OOM and improving throughput|Set to `false` on prefill nodes|
|Multistream Overlap Shared Expert|A3|`--additional-config '{"multistream_overlap_shared_expert": true}'`|Overlaps shared-expert computation on an additional stream, hiding its latency and improving decode performance|Auto-disabled when `VLLM_ASCEND_ENABLE_FUSED_MC2=1`|

## 10 FAQ

- **Q: How to enable function calling for GLM-5.2?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```
