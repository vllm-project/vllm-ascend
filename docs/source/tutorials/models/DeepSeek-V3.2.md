# DeepSeek-V3.2

## Introduction

DeepSeek-V3.2 is a sparse attention model. The main architecture is similar to DeepSeek-V3.1, but with a sparse attention mechanism, which is designed to explore and validate optimizations for training and inference efficiency in long-context scenarios.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `DeepSeek-V3.2-Exp`(BF16 version): require 2 Atlas 800 A3 (64G × 16) nodes or 4 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://modelers.cn/models/Modelers_Park/DeepSeek-V3.2-Exp-BF16)
- `DeepSeek-V3.2-Exp-w8a8`(Quantized version): require 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://modelers.cn/models/Modelers_Park/DeepSeek-V3.2-Exp-w8a8)
- `DeepSeek-V3.2`(BF16 version): require 2 Atlas 800 A3 (64G × 16) nodes or 4 Atlas 800 A2 (64G × 8) nodes. Model weight in BF16 not found now.
- `DeepSeek-V3.2-w8a8`(Quantized version): require 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.2-W8A8/)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

### Installation

You can use our official docker image to run `DeepSeek-V3.2` directly.

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

Start the docker image on your each node.

```{code-block} bash
   :substitutions:

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

::::
::::{tab-item} A2 series
:sync: A2

Start the docker image on your each node.

```{code-block} bash
   :substitutions:

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

::::
:::::

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

## Deployment

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
--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'

```

### Multi-node Deployment

- `DeepSeek-V3.2-w8a8`: require at least 2 Atlas 800 A2 (64G × 8).

Run the following scripts on two nodes respectively.

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

**Node0**

```{code-block} bash
   :substitutions:
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
--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
```

**Node1**

```{code-block} bash
   :substitutions:
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
--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
```

::::
::::{tab-item} A2 series
:sync: A2

**Node0**

```{code-block} bash
   :substitutions:
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
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

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
--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'

```

**Node1**

```{code-block} bash
   :substitutions:
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
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

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
--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'

```

::::
:::::

### Prefill-Decode Disaggregation

We'd like to show the deployment guide of `DeepSeek-V3.2` on multi-node environment with 1P1D for better performance.

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
        export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000

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
            --served-model-name dsv3 \
            --max-model-len 68000 \
            --max-num-batched-tokens 32550 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.82 \
            --quantization ascend \
            --enforce-eager \
            --no-enable-prefix-caching \
            --additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
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
        export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000

        export ASCEND_RT_VISIBLE_DEVICES=$1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

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
            --served-model-name dsv3 \
            --max-model-len 68000 \
            --max-num-batched-tokens 32550 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.82 \
            --quantization ascend \
            --enforce-eager \
            --no-enable-prefix-caching \
            --additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
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
        export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000

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
            --served-model-name dsv3 \
            --max-model-len 68000 \
            --max-num-batched-tokens 12 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[3, 6, 9, 12]}' \
            --trust-remote-code \
            --max-num-seqs 4 \
            --gpu-memory-utilization 0.95 \
            --no-enable-prefix-caching \
            --async-scheduling \
            --quantization ascend \
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
            }' \
            --additional-config '{"recompute_scheduler_enable" : true}'
        ```

    4. Decode node 1

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
        export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000

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
            --served-model-name dsv3 \
            --max-model-len 68000 \
            --max-num-batched-tokens 12 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY",  "cudagraph_capture_sizes":[3, 6, 9, 12]}' \
            --trust-remote-code \
            --async-scheduling \
            --max-num-seqs 4 \
            --gpu-memory-utilization 0.95 \
            --no-enable-prefix-caching \
            --quantization ascend \
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
            }' \
            --additional-config '{"recompute_scheduler_enable" : true}'
        ```

Once the preparation is done, you can start the server with the following command on each node:

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

### Request Forwarding

To set up request forwarding, run the following script on any machine. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
unset http_proxy
unset https_proxy

python load_balance_proxy_server_example.py \
    --port 8000 \
    --host 0.0.0.0 \
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

## Functional Verification

Once your server is started, you can query the model with input prompts:

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

## Accuracy Evaluation

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

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

The performance result is:  

**Hardware**: A3-752T, 4 node

**Deployment**: 1P1D, Prefill node: DP2+TP16, Decode Node: DP8+TP4

**Input/Output**: 64k/3k

**Performance**: 533tps, TPOT 32ms

### Using vLLM Benchmark

Run performance evaluation of `DeepSeek-V3.2-W8A8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model /root/.cache/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

## Function Call

The function call feature is supported from v0.13.0rc1 on. Please use the latest version.

Refer to [DeepSeek-V3.2 Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2.html#tool-calling-example) for details.
