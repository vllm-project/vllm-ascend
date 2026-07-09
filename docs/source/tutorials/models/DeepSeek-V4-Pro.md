# DeepSeek-V4-Pro

## 1 Introduction

DeepSeek-V4 introduces several key upgrades over DeepSeek-V3:

- The Manifold-Constrained Hyper-Connections (mHC) to strengthen conventional residual connections.
- A hybrid attention architecture, which greatly improves long-context efficiency through Compress-4-Attention and Compress-128-Attention. For the Mixture-of-Experts (MoE) components, it still adopts the DeepSeekMoE architecture, with only minor adjustments.

DeepSeek-V4-Pro, the maximum reasoning effort mode of DeepSeek-V4, significantly advances the knowledge capabilities of open-source models, firmly establishing itself as the best open-source model available today. It achieves top-tier performance in coding benchmarks and significantly bridges the gap with leading closed-source models on reasoning and agentic tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

> **Note**: Please replace the version placeholder above with your actual validation version.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `DeepSeek-V4-Pro-w4a8-mtp` (Quantized version): requires 2 Atlas 800 A3 (128G × 8) nodes or 4 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy a multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

=== "A3 series"

    Start the docker image on each node.

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    docker run --rm \
        --name vllm-ascend \
        --shm-size=512g \
        --net=host \
        --privileged=true \
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
        -v /etc/hccn.conf:/etc/hccn.conf \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

=== "A2 series"

    Start the docker image on each node.

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    docker run --rm \
        --name vllm-ascend \
        --shm-size=512g \
        --net=host \
        --privileged=true \
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
        -v /etc/hccn.conf:/etc/hccn.conf \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

After a successful docker run, you can verify the running container service by executing the `docker ps` command.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy a multi-node environment, you need to set up the environment on each node.

## 5 Online Service Deployment

!!! note

    In this tutorial, we suppose you downloaded the model weight to `/root/.cache/modelscope/hub/models/vllm-ascend/`. Feel free to change it to your own path.

    It is recommended that the following service code be encapsulated in a .sh script file and executed in Bash mode.

### 5.1 Multi-Node Online Deployment

The quantized model `DeepSeek-V4-Pro-w4a8-mtp` requires at least 2 Atlas 800 A3 (128G × 8) nodes or 4 Atlas 800 A2 (64G × 8) nodes. Run the following scripts on each node respectively.

=== "A2 series"

    **Node0**

    ```bash
    local_ip="xxx"
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export IFNAME="xxx"
    export GLOO_SOCKET_IFNAME="$IFNAME"
    export TP_SOCKET_IFNAME="$IFNAME"
    export HCCL_SOCKET_IFNAME="$IFNAME"
    export HCCL_BUFFSIZE=512
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ACL_OP_INIT_MODE=1
    export VLLM_ENGINE_READY_TIMEOUT_S=3600
    export HCCL_OP_EXPANSION_MODE="AIV"

    export TASK_QUEUE_ENABLE=1
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

    export HCCL_CONNECT_TIMEOUT=7200
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export VLLM_RPC_TIMEOUT=1800000

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
      --host 0.0.0.0 \
      --port 10010 \
      --max-model-len 135000 \
      --max-num-batched-tokens 4096 \
      --served-model-name dsv4 \
      --gpu-memory-utilization 0.9 \
      --max-num-seqs 16 \
      --data-parallel-size 4 \
      --tensor-parallel-size 8 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 0 \
      --data-parallel-address $node0_ip  \
      --enable-expert-parallel \
      --quantization ascend \
      --no-enable-prefix-caching \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --async-scheduling \
      --safetensors-load-strategy 'prefetch' \
      --block-size 128 \
      --speculative-config '{
         "num_speculative_tokens": 1,
         "method": "mtp",
         "enforce_eager": true
      }' \
      --additional-config '{
         "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
         },
         "enable_cpu_binding": true,
         "enable_shared_expert_dp": true,
         "multistream_overlap_shared_expert":true
      }' \
      --compilation-config '{
         "cudagraph_mode":"FULL_DECODE_ONLY"
      }' \
      --model-loader-extra-config '{
         "enable_multithread_load": "true",
         "num_threads": 128
      }'
    ```

    **Node1-Node3**

    ```bash
    local_ip="xxx"
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export IFNAME="xxx"
    export GLOO_SOCKET_IFNAME="$IFNAME"
    export TP_SOCKET_IFNAME="$IFNAME"
    export HCCL_SOCKET_IFNAME="$IFNAME"
    export HCCL_BUFFSIZE=512
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ACL_OP_INIT_MODE=1
    export VLLM_ENGINE_READY_TIMEOUT_S=3600
    export HCCL_OP_EXPANSION_MODE="AIV"

    export TASK_QUEUE_ENABLE=1
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

    export HCCL_CONNECT_TIMEOUT=7200
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export VLLM_RPC_TIMEOUT=1800000

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
      --host 0.0.0.0 \
      --port 10010 \
      --max-model-len 135000 \
      --max-num-batched-tokens 4096 \
      --served-model-name dsv4 \
      --gpu-memory-utilization 0.9 \
      --max-num-seqs 16 \
      --data-parallel-size 4 \
      --tensor-parallel-size 8 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 1 \
      --data-parallel-address $node0_ip  \
      --enable-expert-parallel \
      --quantization ascend \
      --no-enable-prefix-caching \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --async-scheduling \
      --safetensors-load-strategy 'prefetch' \
      --block-size 128 \
      --headless \
      --speculative-config '{
         "num_speculative_tokens": 1,
         "method": "mtp",
         "enforce_eager": true
      }' \
      --additional-config '{
         "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
         },
         "enable_cpu_binding": true,
         "enable_shared_expert_dp": true,
         "multistream_overlap_shared_expert":true
      }' \
      --compilation-config '{
         "cudagraph_mode":"FULL_DECODE_ONLY"
      }' \
      --model-loader-extra-config '{
         "enable_multithread_load": "true",
         "num_threads": 128
      }'
    ```

=== "A3 series"

    **Node0**

    ```bash
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
    export HCCL_BUFFSIZE=2048
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export TASK_QUEUE_ENABLE=1
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
      --safetensors-load-strategy 'prefetch' \
      --max-model-len 135000  \
      --max-num-batched-tokens 4096 \
      --served-model-name dsv4 \
      --gpu-memory-utilization 0.9 \
      --max-num-seqs 32 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 0 \
      --data-parallel-address $node0_ip \
      --data-parallel-rpc-port 13399 \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --quantization ascend \
      --port 8900 \
      --host 0.0.0.0 \
      --block-size 128 \
      --async-scheduling \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
      --additional-config '
        {"ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
            },
        "enable_cpu_binding": true,
        "multistream_overlap_shared_expert":true}'
    ```

    **Node1**

    ```bash
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
    export HCCL_BUFFSIZE=2048
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export TASK_QUEUE_ENABLE=1
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
      --safetensors-load-strategy 'prefetch' \
      --max-model-len 135000  \
      --max-num-batched-tokens 4096 \
      --served-model-name dsv4 \
      --gpu-memory-utilization 0.9 \
      --max-num-seqs 32 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 1 \
      --data-parallel-address $node0_ip \
      --data-parallel-rpc-port 13399 \
      --headless \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --quantization ascend \
      --port 8900 \
      --host 0.0.0.0 \
      --block-size 128 \
      --async-scheduling \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
      --additional-config '
        {"ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
            },
        "enable_cpu_binding": true,
        "multistream_overlap_shared_expert":true}'
    ```

Key Parameter Descriptions:

- `--data-parallel-start-rank` specifies the starting data parallel rank of the current node. Each node must be set to a unique value (e.g., Node0 = 0, Node1 = 1).
- `--data-parallel-address` specifies the IP address of the data parallel master node (Node0). It must be consistent across all nodes.
- `--headless` (used on non-master nodes) disables the API server on the node, since only the master node serves requests.
- `--max-model-len` specifies the maximum context length. Adjust it according to your actual scenario.
- `--speculative-config` configures the MTP (Multi-Token Prediction) speculative decoding to accelerate inference.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full ACL graph execution in the decode phase to reduce scheduling latency.
- `--async-scheduling` enables asynchronous scheduling to overlap CPU scheduling with NPU computation.
- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` enables the FlashComm communication optimization.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

Service Verification:

```shell
curl http://<node0_ip>:8900/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dsv4",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 256,
        "temperature": 0
    }'
```

Expected Result:

The service returns HTTP 200 OK with a JSON response containing the `choices` field.

### 5.2 Multi-Node PD Separation Deployment

We recommend using Mooncake for deployment: [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md).

In the standard deployment mode, Prefill (prompt processing) and Decode (token generation) tasks run on the same set of NPUs. PD (Prefill-Decode) separation addresses this by running Prefill and Decode on dedicated node groups, each configured independently. This architecture is recommended for production deployments with concurrent multi-user workloads, where stable latency and high throughput are both required.

The following sections describe PD separation deployment on both Atlas 800 A3 (128G × 8) and Atlas 800 A2 (64G × 8) multi-node environments.

#### 5.2.1 A3 Series PD Separation Deployment

This section shows the deployment guide of DeepSeek-V4-Pro on Atlas 800 A3 (128G × 8) multi-node environment with 1P1D for better performance.

Before you start, please:

1. Prepare the script `launch_online_dp.py` on each node.

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

2. Prepare the script `run_dp_template.sh` on each node.

    1. Prefill node 0

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.1 # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"
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
        export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name auto \
            --max-model-len 131072 \
            --max-num-batched-tokens 4096 \
            --max-num-seqs 16 \
            --no-disable-hybrid-kv-cache-manager \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --safetensors-load-strategy 'prefetch' \
            --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
            --trust-remote-code \
            --gpu-memory-utilization 0.92 \
            --quantization ascend \
            --block-size 128 \
            --enforce-eager \
            --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
            --additional-config '{"enable_cpu_binding": true, "enable_dsa_cp": true}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_producer",
            "kv_port": "30200",
            "engine_id": "1",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 2,
                                "tp_size": 16
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 2
                        }
                }
            }'
        ```

    2. Prefill node 1

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.2 # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"
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
        export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name auto \
            --max-model-len 131072 \
            --max-num-batched-tokens 4096 \
            --max-num-seqs 16 \
            --no-disable-hybrid-kv-cache-manager \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --safetensors-load-strategy 'prefetch' \
            --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
            --trust-remote-code \
            --gpu-memory-utilization 0.92 \
            --quantization ascend \
            --block-size 128 \
            --enforce-eager \
            --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
            --additional-config '{"enable_cpu_binding": true, "enable_dsa_cp": true}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_producer",
            "kv_port": "30200",
            "engine_id": "1",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 2,
                                "tp_size": 16
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 2
                        }
                }
            }'
        ```

    3. Decode node (Same as another D node)

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.3/4 # change to your own ip

        export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
        export HCCL_OP_EXPANSION_MODE="AIV"
        export TASK_QUEUE_ENABLE=1
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=2000
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

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name auto \
            --max-model-len 131072 \
            --max-num-batched-tokens 120 \
            --max-num-seqs 60 \
            --async-scheduling \
            --block-size 128 \
            --no-enable-prefix-caching \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --no-disable-hybrid-kv-cache-manager \
            --safetensors-load-strategy 'prefetch' \
            --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
            --trust-remote-code \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --speculative-config '{"num_speculative_tokens": 1, "method":"mtp", "enforce_eager": true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_consumer",
            "kv_port": "30800",
            "engine_id": "8",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 2,
                                "tp_size": 16
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 2
                        }
                }
            }' \
            --additional-config '{
                "ascend_compilation_config":{
                    "enable_npugraph_ex":true,
                    "enable_static_kernel":false
                },
            "enable_cpu_binding":true,
            "recompute_scheduler_enable":true
            }'
        ```

3. Start the server with the following command on each node.

    1. Prefill node 0

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    2. Prefill node 1

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 1 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    3. Decode node 0

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 0 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    4. Decode node 1

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 8 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

4. Deploy the P-D disaggregation proxy.

    Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

#### 5.2.2 A2 Series PD Separation Deployment

This section shows the deployment guide of DeepSeek-V4-Pro on Atlas 800 A2 (64G × 8) multi-node environment with 1P1D for better performance.

Before you start, please:

1. Prepare the script `launch_online_dp.py` on each node.

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

2. Prepare the script `run_dp_template.sh` on each node.

    1. Prefill node (4 P nodes share the same script)

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.1/2/3/4 # change to your own ip

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

        sysctl -w vm.swappiness=0
        sysctl -w kernel.numa_balancing=0
        sysctl kernel.sched_migration_cost_ns=50000

        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
        export ASCEND_RT_VISIBLE_DEVICES=$1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
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
            --max-model-len 133072 \
            --max-num-batched-tokens 4096 \
            --max-num-seqs 16 \
            --no-disable-hybrid-kv-cache-manager \
            --trust-remote-code \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --enforce-eager \
            --no-enable-prefix-caching \
            --speculative-config '{"num_speculative_tokens": 1, "method":"mtp", "enforce_eager": true}' \
            --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true, "enable_dsa_cp": true}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "engine_id": "0",
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
          }'
        ```

    2. Decode node (4 D nodes share the same script)

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.5/6/7/8 # change to your own ip

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

        sysctl -w vm.swappiness=0
        sysctl -w kernel.numa_balancing=0
        sysctl kernel.sched_migration_cost_ns=50000

        export ASCEND_RT_VISIBLE_DEVICES=$1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
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
            --max-model-len 133072 \
            --max-num-batched-tokens 120 \
            --max-num-seqs 60 \
            --async-scheduling \
            --block-size 128 \
            --no-disable-hybrid-kv-cache-manager \
            --trust-remote-code \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --safetensors-load-strategy 'prefetch' \
            --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
            --no-enable-prefix-caching \
            --speculative-config '{"num_speculative_tokens": 1, "method":"mtp", "enforce_eager": true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_consumer",
            "kv_port": "30100",
            "engine_id": "1",
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
          }' \
            --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}, "enable_cpu_binding":true, "recompute_scheduler_enable":true}'
        ```

3. Start the server with the following command on each node.

    1. Prefill node 0

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    2. Prefill node 1

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    3. Prefill node 2

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 2 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    4. Prefill node 3

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 3 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    5. Decode node 0

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 0 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    6. Decode node 1

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 2 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    7. Decode node 2

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 4 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    8. Decode node 3

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 6 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

4. Deploy the P-D disaggregation proxy.

    Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

Key Parameter Descriptions:

- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`: enables the communication optimization function on the prefill nodes.
- `VLLM_ASCEND_ENABLE_FUSED_MC2=1`: enables the Fused MC2 fusion operator to accelerate communication on prefill nodes (A3 series).
- `recompute_scheduler_enable: true`: enables the recomputation scheduler. When the KV Cache of the decode node is insufficient, requests will be sent to the prefill node to recompute the KV Cache. In the PD separation scenario, enable this configuration only on decode nodes.
- `MooncakeHybridConnector`: the KV transfer connector used for PD separation, transferring KV Cache between prefill and decode nodes.
- `enable_shared_expert_dp: true`: enables data parallelism for shared experts, applicable to MoE models.

Deployment Verification:

After the PD separation service is fully started, send a request through the proxy port on the prefill master node to verify that Prefill and Decode nodes are working correctly together. Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) for the proxy verification method.

Common Issues Tip: If you encounter issues with PD separation deployment, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

In <node0_ip>:<port>, use the IP address and port number of the primary node. If the primary and standby nodes are separated, use the IP address and port number of the proxy node.

```shell
curl http://<node0_ip>:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dsv4",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 256,
        "temperature": 0
    }'
```

Expected Result:

The service returns HTTP 200 OK with a JSON response containing the `choices` field.

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

| dataset | version | metric | mode | vllm-api-general-chat | note |
| ----- | ----- | ----- | ----- | ----- | ----- |
| GPQA | - | accuracy | gen | 89.90 | 1 Atlas 800 A3 (128G × 8) |
| GSM8K | - | accuracy | gen | 96.21 | 1 Atlas 800 A3 (128G × 8) |

## 8 Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

> `*Total NPUs` indicates the total number of NPUs used across all nodes.

|Scenario|Deployment Mode|*Total NPUs|Weight Version|Key Considerations|
|--------|---------------|-----------|---------------|-------------------|
|High Throughput|Single-Node Mixed|32 (A3)|DeepSeek-V4-Pro-w4a8-mtp|Use dp2 tp16 to balance memory capacity and compute efficiency|
|High Throughput|1P1D deployment|64 (A3)|DeepSeek-V4-Pro-w4a8-mtp|dp16 tp2 or dp2 tp16, depending on memory and concurrency|
|Long Context (1M)|Single-Node Mixed|32 (A3)|DeepSeek-V4-Pro-w4a8-mtp|Use dp2 tp16 to balance memory capacity and compute efficiency|
|Long Context (1M)|1P1D deployment|64 (A3)|DeepSeek-V4-Pro-w4a8-mtp|dp2 tp16 on both P and D nodes; balanced latency and throughput|

#### Table 2: Detailed Node Configuration

|Scenario|Configuration|NPUs|TP|DP|Max Num Seqs|Max Num Batched Tokens|Max Model Len|MTP Speculation Num|
|--------|-------------|-----|--|--|------------|----------------------|--------------|--------------------|
|Multi-Node (A3)|Node0 / Node1|8|16|2|32|4096|135000|1|
|PD Separation (A3)|Prefill Node|8|16|2|16|4096|131072|1|
|PD Separation (A3)|Decode Node|8|2|16|60|120|131072|1|

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Chapter 5](#5-online-service-deployment).

**Notice:**

`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the [Deployment](#5-online-service-deployment) chapter.

Currently, we support 4K prefix cache hit in an experimental manner. You only need to change the value of --block-size from 128 to 32 in the service.

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html); this chapter only covers model-specific issues.
