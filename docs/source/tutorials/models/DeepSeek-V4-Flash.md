# DeepSeek-V4-Flash

## 1 Introduction

DeepSeek-V4 introduces several key upgrades over DeepSeek-V3:

- The Manifold-Constrained Hyper-Connections (mHC) to strengthen conventional residual connections.
- A hybrid attention architecture, which greatly improves long-context efficiency through Compress-4-Attention and Compress-128-Attention. For the Mixture-of-Experts (MoE) components, it still adopts the DeepSeekMoE architecture, with only minor adjustments.

DeepSeek-V4-Flash is the lightweight variant of the DeepSeek-V4 family, suitable for high-throughput and low-latency serving scenarios.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

> **Note**: Please replace the version placeholder above with your actual validation version.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `DeepSeek-V4-Flash-w8a8-mtp` (Quantized version): requires 1 Atlas 800 A3 (128G × 8) node or 1 Atlas 800 A2 (64G × 8) node. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp)

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
    # deepseek-v4-flash uses the following image
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    
    # deepseek-v4-flash-dspark uses the following image
    export IMAGE=quay.io/ascend/vllm-ascend:nightly-main
    
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

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node. The quantized model `DeepSeek-V4-Flash-w8a8-mtp` can be deployed on 1 Atlas 800 A3 (128G × 8) or 1 Atlas 800 A2 (64G × 8).

=== "A2 series"

    Run the following script to execute online inference.

    ```shell
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export HCCL_BUFFSIZE=1024
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
        --max-model-len 133120 \
        --max-num-batched-tokens 8192 \
        --served-model-name dsv4 \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 32 \
        --data-parallel-size 1 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --safetensors-load-strategy 'prefetch' \
        --no-enable-prefix-caching \
        --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
        --quantization ascend \
        --port 8900 \
        --block-size 128 \
        --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --async-scheduling \
        --additional-config '
        {"ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
            },
        "enable_cpu_binding": true,
        "enable_dsa_cp": true,
        "multistream_overlap_shared_expert":true}'
    ```

=== "A2 series with dspark"

    Run the following script to execute online inference.

    ```shell
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export HCCL_BUFFSIZE=1024
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE=AIV

    vllm serve /root/.cache/modelscope/hub/models/UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test \
        --max-model-len 800000 \
        --max-num-batched-tokens 8192 \
        --served-model-name dsv4-dspark \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 32 \
        --data-parallel-size 1 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
        --quantization ascend \
        --port 8000 \
        --block-size 128 \
        --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, "enforce_eager": true}'  \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
    ```
    tps more than 50+ ,its reach  2X speed of dsv4f with mtp

=== "A3 series"

    Run the following script to execute online inference.

    ```shell
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export HCCL_BUFFSIZE=1024
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
        --max-model-len 1048576 \
        --max-num-batched-tokens 10240 \
        --served-model-name dsv4 \
        --gpu-memory-utilization 0.9 \
        --api-server-count 1 \
        --max-num-seqs 64 \
        --data-parallel-size 4 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --safetensors-load-strategy 'prefetch' \
        --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
        --quantization ascend \
        --port 8900 \
        --block-size 128 \
        --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --async-scheduling \
        --additional-config '
        {"ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
            },
        "enable_cpu_binding": true,
        "multistream_overlap_shared_expert":true}'
    ```

Key Parameter Descriptions:

- `--max-model-len` specifies the maximum context length - that is, the sum of input and output tokens for a single request. Adjust it according to your actual scenario.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
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

In the standard single-node deployment mode, Prefill (prompt processing) and Decode (token generation) tasks run on the same set of NPUs. This can lead to two issues:

1. **Prefill preemption interrupts Decode**: Prefill is a compute-intensive task that processes the entire input context at once, while Decode generates tokens one by one. When a new user request arrives, its Prefill phase can preempt and interrupt ongoing Decode tasks, causing jitter and higher time-per-output-token (TPOT) latency.
2. **Inflexible resource allocation**: Prefill and Decode have fundamentally different computational characteristics — Prefill is compute-bound and memory-bandwidth-intensive, while Decode is memory-bandwidth-bound. Running them on the same hardware forces a compromise that satisfies neither optimally.

PD (Prefill-Decode) separation addresses these issues by running Prefill and Decode on dedicated node groups, each configured independently. This architecture is recommended for production deployments with concurrent multi-user workloads, where stable latency and high throughput are both required.

The following sections describe PD separation deployment on both Atlas 800 A3 (128G × 8) and Atlas 800 A2 (64G × 8) multi-node environments.

#### 5.2.1 A3 Series PD Separation Deployment

This section shows the deployment guide of DeepSeek-V4-Flash on Atlas 800 A3 (128G × 8) multi-node environment with 1P1D for better performance.

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
        export HCCL_BUFFSIZE=2560
        export TASK_QUEUE_ENABLE=1
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
        export HCCL_OP_EXPANSION_MODE="AIV"
        export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
        export ASCEND_RT_VISIBLE_DEVICES=$1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
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
            --max-num-batched-tokens 8192 \
            --max-num-seqs 16 \
            --no-disable-hybrid-kv-cache-manager \
            --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
            --no-enable-prefix-caching \
            --safetensors-load-strategy 'prefetch' \
            --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
            --trust-remote-code \
            --block-size 128 \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --enforce-eager \
            --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true,  "enable_dsa_cp": true}' \
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

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
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
            --max-num-batched-tokens 120 \
            --max-num-seqs 60 \
            --async-scheduling \
            --block-size 128 \
            --no-disable-hybrid-kv-cache-manager \
            --no-enable-prefix-caching \
            --safetensors-load-strategy 'prefetch' \
            --trust-remote-code \
            --tokenizer-mode deepseek_v4 \
            --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
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
                "multistream_overlap_shared_expert":true,
                "recompute_scheduler_enable":true
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

#### 5.2.2 A2 Series PD Separation Deployment

This section shows the deployment guide of DeepSeek-V4-Flash on Atlas 800 A2 (64G × 8) multi-node environment with 4\*1P 1\*4D for better performance.

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
        parser.add_argument("--dp-size", type=int, required=True, help="Data parallel size.")
        parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
        parser.add_argument("--dp-size-local", type=int, default=-1, help="Local data parallel size.")
        parser.add_argument("--dp-rank-start", type=int, default=0, help="Starting rank for data parallel.")
        parser.add_argument("--dp-address", type=str, required=True, help="IP address for data parallel master node.")
        parser.add_argument("--dp-rpc-port", type=str, default=12345, help="Port for data parallel master node.")
        parser.add_argument("--vllm-start-port", type=int, default=9000, help="Starting port for the engine.")
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
            process = multiprocessing.Process(target=run_command, args=(visible_devices, dp_rank, vllm_engine_port))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
    ```

2. Prepare the script `run_dp_template.sh` on each node.

    1. Prefill node (4 P nodes share the same script)

        For each P instance, only these two configuration values need to be modified: `kv_port` and `engine_id`. The `engine_id` should start from 0 and increment sequentially, while the `kv_port` (e.g., `30100`) must be unique for each P instance, such as 30000, 30100, etc.

        ```shell
        unset ftp_proxy
        unset https_proxy
        unset http_proxy
        rm -rf ~/ascend/log

        nic_name="xxxxxx" #eg."enp67s0f0np0"
        local_ip=`hostname -I|awk -F " " '{print$1}'`

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
        export TASK_QUEUE_ENABLE=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
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
            --max-model-len 135000 \
            --max-num-batched-tokens 4096 \
            --max-num-seqs 16 \
            --block-size 128 \
            --enforce-eager \
            --async-scheduling \
            --no-disable-hybrid-kv-cache-manager \
            --enable-prefix-caching \
            --trust-remote-code \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true}' \
            --speculative-config '{"num_speculative_tokens": 1, "method": "mtp","enforce_eager": true}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "engine_id": "0",
            "kv_connector_extra_config": {
                        "prefill": {
                            "dp_size": 8,
                            "tp_size": 1
                        },
                        "decode": {
                            "dp_size": 32,
                            "tp_size": 1
                        }
                }
            }'
        ```

    2. Decode node (4 D nodes share the same script)

        ```shell
        unset ftp_proxy
        unset https_proxy
        unset http_proxy
        rm -rf ~/ascend/log

        nic_name="xxxxxx" #eg."enp67s0f0np0"
        local_ip=`hostname -I|awk -F " " '{print$1}'`

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

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
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
            --max-model-len 135000 \
            --max-num-batched-tokens 60 \
            --max-num-seqs 30 \
            --async-scheduling \
            --block-size 128 \
            --no-disable-hybrid-kv-cache-manager \
            --no-enable-prefix-caching \
            --trust-remote-code \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --speculative-config '{"num_speculative_tokens": 1, "method": "mtp","enforce_eager": true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_consumer",
            "kv_port": "30400",
            "engine_id": "4",
            "kv_connector_extra_config": {
                        "prefill": {
                            "dp_size": 8,
                            "tp_size": 1
                        },
                        "decode": {
                            "dp_size": 32,
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
               "multistream_overlap_shared_expert":true,
               "recompute_scheduler_enable":true
            }'
        ```

3. Start the server with the following command on each node.

    1. Prefill node

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 8 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12321 --vllm-start-port 7100
        ```

        For each P instance, only the `--dp-address` parameter differs and must be configured as the IP address of the service within the same subnet as the other instances.

    2. Decode node

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start x --dp-address x.x.x.x --dp-rpc-port 12321 --vllm-start-port 7100
        ```

        For each D instance, only the `--dp-rank-start` parameter differs, which should be configured as 0, 8, 16, and 24 respectively. Each instance's `--dp-address` must be set to the IP address of the main D node, which is the IP of the Decode instance with `--dp-rank-start` set to 0.

4. Deploy the P-D disaggregation proxy.

    The proxy is also implemented by referring to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md).

Key Parameter Descriptions:

- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`: enables the communication optimization function on the prefill nodes.
- `recompute_scheduler_enable: true`: enables the recomputation scheduler. When the KV Cache of the decode node is insufficient, requests will be sent to the prefill node to recompute the KV Cache. In the PD separation scenario, enable this configuration only on decode nodes.
- `MooncakeHybridConnector`: the KV transfer connector used for PD separation, transferring KV Cache between prefill and decode nodes.
- `enable_shared_expert_dp: true`: enables data parallelism for shared experts, applicable to MoE models.

Deployment Verification:

After the PD separation service is fully started, send a request through the proxy port on the prefill master node to verify that Prefill and Decode nodes are working correctly together. Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) for the proxy verification method.

Common Issues Tip: If you encounter issues with PD separation deployment, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

#### 5.2.3 Ultra-Long Sequence Deployment

For ultra-long sequence scenarios, support can be achieved by adjusting the PD (Prefill/Decode) ratio and the model parallelism strategy. For example, in a 1M sequence scenario, a 1\*4P-1\*4D ratio can be used, with the model parallelism set to DP4TP8 mode.

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
| GPQA | - | accuracy | gen | 88.17 | 1 Atlas 800 A3 (128G × 8) |
| GSM8K | - | accuracy | gen | 96.30 | 1 Atlas 800 A3 (128G × 8) |

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
|High Throughput|Single-Node Mixed|16 (A3)|DeepSeek-V4-Flash-w8a8-mtp|Use dp4 tp4 to balance memory capacity and compute efficiency|
|High Throughput|1P1D deployment|32 (A3)|DeepSeek-V4-Flash-w8a8-mtp|dp16 tp1 on both P and D nodes; balanced latency and throughput|
|Long Context (1M)|Single-Node (A3)|8 (A3)|DeepSeek-V4-Flash-w8a8-mtp|Use dp4 tp4 to balance memory capacity and compute efficiency|
|Long Context (1M)|1P1D deployment|32 (A3)|DeepSeek-V4-Flash-w8a8-mtp|dp16 tp1 on both P and D nodes; balanced latency and throughput|

#### Table 2: Detailed Node Configuration

|Scenario|Configuration|NPUs|TP|DP|Max Num Seqs|Max Num Batched Tokens|Max Model Len|MTP Speculation Num|
|--------|-------------|-----|--|--|------------|----------------------|--------------|--------------------|
|High Throughput (A3)|Server / Single Machine|8|4|4|64|10240|1048576|1|
|Long Context (1M, A3)|Server / Single Machine|8|4|4|64|10240|1048576|1|
|PD Separation (A3)|Server-P Node|8|4|4|16|8192|1048576|1|
|PD Separation (A3)|Server-D Node|8|1|16|60|120|1048576|1|

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
