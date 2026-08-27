# Qwen3.8-2.4T-A95B

## 1 Introduction

Qwen3.8-2.4T-A95B is a large-scale MoE model with 2400 billion total
parameters and 95 billion activated parameters. It is built on the Qwen3.5
architecture and improves coding, professional work, scientific research, and
long-horizon agent tasks.

This document describes the main validation steps for the model, including
supported features, prerequisites, installation, multi-node deployment,
functional verification, accuracy and performance evaluation, performance
tuning, and FAQs. The validated configurations cover Atlas A3 and Atlas A2
deployments.

This document is validated and written based on **vLLM-Ascend 0.23.0**. The
current model (Qwen3.8-2.4T-A95B) is first supported in this version.

## 2 Supported Features

Refer to [Supported Models](../../user_guide/support_matrix/supported_models.md)
to get the model support matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get feature
configuration details.

## 3 Prerequisites

### 3.1 Model Weight

The following model weights are available:

- `Qwen3.8-2.4T-A95B` (FP16/BF16): approximately 4.89 TB of storage and weight
  memory. [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3.8-2.4T-A95B).
- `Qwen3.8-2.4T-A95B-w8a8`: approximately 2.33 TiB of storage and weight
  memory. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-2.4T-A95B-w8a8).
- `Qwen3.8-2.4T-A95B-w4a8`: approximately 1.21 TiB of storage and weight
  memory. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-2.4T-A95B-w4a8).

This guide includes the following validated deployment configurations:

| Platform | Weight | Deployment | Topology |
| --- | --- | --- | --- |
| 4 × Atlas 800 A3 (64GB × 16) | W8A8 | Mixed Prefill/Decode deployment | DP4/TP16/EP64 |
| 8 × Atlas 800 A2 (64GB × 8) | W4A8 | Mixed Prefill/Decode deployment | DP8/TP8/EP64 |

The checkpoint and tokenizer directories must be available at the same paths
on all serving nodes. The W8A8 deployment uses the lazy Safetensors strategy to
avoid prefetching the complete checkpoint from shared storage.

It is recommended to download the model weight to the shared directory of
multiple nodes, such as `/root/.cache/`.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy the model in a multi-node environment, verify the
communication environment according to
[verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your
node. Refer to [using docker](../../installation.md#set-up-using-docker).

=== "A3 series"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:qwen3.8-a3
    export NAME=vllm-ascend

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

    After entering the container, verify that vLLM and vLLM-Ascend can be imported:

    ```shell
    python -c "import vllm, vllm_ascend; print('vllm and vllm_ascend are ready')"
    ```

=== "A2 series"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:v0.23.0
    export NAME=vllm-ascend

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

    After entering the container, verify that vLLM and vLLM-Ascend can be imported:

    ```shell
    python -c "import vllm, vllm_ascend; print('vllm and vllm_ascend are ready')"
    ```

### 4.2 Source Code Installation

You can also build and install `vllm-ascend` from source. Refer to
[set up using Python](../../installation.md#set-up-using-python).

If you want to deploy a multi-node service, install the same version of vLLM
and vLLM-Ascend on each node.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Multi-Node Deployment

The validated mixed deployment runs one DP rank per node, uses tensor
parallelism within each node, and spans expert parallelism across all nodes.
Select the tab that matches your platform:

- **A3 series**: four Atlas 800 A3 (64GB × 16) nodes, DP4/TP16/EP64, with the
  W8A8 checkpoint.
- **A2 series**: eight Atlas 800 A2 (64GB × 8) nodes, DP8/TP8/EP64, with the
  W4A8 checkpoint.

Before starting the service:

- Replace the model path, node count, parallel sizes, local IP address, network
  interface, service port, and RPC port with values from the target
  environment.
- `NIC_NAME` must be the interface that owns `LOCAL_IP`.
- Start Node 0 first. `NODE0_IP` on every worker must equal `LOCAL_IP` on Node
  0.
- Assign every worker a unique `DP_START_RANK`.

=== "A3 series"

    === "Node 0"

        ```shell
        # Values that must be adapted to the target environment.
        export MODEL_PATH=<QWEN3_8_MODEL_PATH>
        export TOKENIZER_PATH=<QWEN3_8_TOKENIZER_PATH>
        export LOCAL_IP=<NODE0_LOCAL_IP>
        export NIC_NAME=<NODE0_NIC_NAME>
        export PORT=<SERVICE_PORT>
        export RPC_PORT=<DP_RPC_PORT>
        export DP_SIZE=4
        export TP_SIZE=16

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
        export HCCL_BUFFSIZE=1024
        export HCCL_BUFFSIZE_EP=2048
        export HCCL_IF_IP=$LOCAL_IP
        export HCCL_INTRA_ROCE_ENABLE=0
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_SOCKET_IFNAME=$NIC_NAME
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        export GLOO_SOCKET_IFNAME=$NIC_NAME
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export OPENBLAS_NUM_THREADS=1

        vllm serve $MODEL_PATH \
            --host 0.0.0.0 \
            --port $PORT \
            --served-model-name qwen3.8 \
            --tokenizer $TOKENIZER_PATH \
            --trust-remote-code \
            --quantization ascend \
            --safetensors-load-strategy lazy \
            --tensor-parallel-size $TP_SIZE \
            --data-parallel-size $DP_SIZE \
            --data-parallel-size-local 1 \
            --data-parallel-address $LOCAL_IP \
            --data-parallel-rpc-port $RPC_PORT \
            --enable-prefix-caching \
            --enable-expert-parallel \
            --max-model-len 131072 \
            --max-num-seqs 8 \
            --max-num-batched-tokens 16384 \
            --gpu-memory-utilization 0.85 \
            --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":1}'
        ```

    === "Nodes 1-3"

        Run this command on every worker. Set `LOCAL_IP` and `NIC_NAME` to the
        current node and set `DP_START_RANK` to `1`, `2`, or `3`.

        ```shell
        # Values that must be adapted to the target environment.
        export MODEL_PATH=<QWEN3_8_MODEL_PATH>
        export TOKENIZER_PATH=<QWEN3_8_TOKENIZER_PATH>
        export LOCAL_IP=<WORKER_LOCAL_IP>
        export NODE0_IP=<NODE0_LOCAL_IP>
        export NIC_NAME=<WORKER_NIC_NAME>
        export PORT=<SERVICE_PORT>
        export RPC_PORT=<DP_RPC_PORT>
        export DP_SIZE=4
        export DP_START_RANK=<1_OR_2_OR_3>
        export TP_SIZE=16

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
        export HCCL_BUFFSIZE=1024
        export HCCL_BUFFSIZE_EP=2048
        export HCCL_IF_IP=$LOCAL_IP
        export HCCL_INTRA_ROCE_ENABLE=0
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_SOCKET_IFNAME=$NIC_NAME
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        export GLOO_SOCKET_IFNAME=$NIC_NAME
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export OPENBLAS_NUM_THREADS=1

        vllm serve $MODEL_PATH \
            --headless \
            --host 0.0.0.0 \
            --port $PORT \
            --served-model-name qwen3.8 \
            --tokenizer $TOKENIZER_PATH \
            --trust-remote-code \
            --quantization ascend \
            --safetensors-load-strategy lazy \
            --tensor-parallel-size $TP_SIZE \
            --data-parallel-size $DP_SIZE \
            --data-parallel-size-local 1 \
            --data-parallel-start-rank $DP_START_RANK \
            --data-parallel-address $NODE0_IP \
            --data-parallel-rpc-port $RPC_PORT \
            --enable-prefix-caching \
            --enable-expert-parallel \
            --max-model-len 131072 \
            --max-num-seqs 8 \
            --max-num-batched-tokens 16384 \
            --gpu-memory-utilization 0.85 \
            --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":1}'
        ```

    The following values differ between the master and worker nodes:

    | Setting | Node 0 | Nodes 1-3 | Description |
    | --- | --- | --- | --- |
    | `LOCAL_IP` | Node 0 IP | Current worker IP | Each node uses its own communication IP. |
    | `NODE0_IP` | Not required | Node 0 IP | Workers use this address to join the DP group. |
    | `--headless` | Omitted | Enabled | Workers do not expose an API endpoint. |
    | `--data-parallel-address` | `$LOCAL_IP` | `$NODE0_IP` | Always resolves to Node 0. |
    | `--data-parallel-start-rank` | `0` by default | `1`, `2`, or `3` | Every node must own a unique DP rank. |

    Key deployment parameters:

    | Parameter | Description |
    | --- | --- |
    | `--tensor-parallel-size 16` | Uses all 16 NPUs in one A3 node for tensor parallelism. |
    | `--data-parallel-size 4` | Creates four global DP ranks across four nodes. |
    | `--data-parallel-size-local 1` | Runs one DP rank on the current node. |
    | `--data-parallel-start-rank` | Selects the global DP rank of a worker node. |
    | `--enable-expert-parallel` | Enables expert parallelism for the MoE layers. |
    | `--max-model-len 131072` | Sets the maximum combined input and output length. |
    | `--quantization ascend` | Enables Ascend W8A8 quantization. |
    | `--safetensors-load-strategy lazy` | Avoids prefetching the complete 2.33 TiB checkpoint from NFS. |
    | `--max-num-seqs 8` | Sets eight active sequences for each DP group and 32 across DP4. |
    | `--max-num-batched-tokens 16384` | Controls the scheduler token budget. |
    | `--gpu-memory-utilization 0.85` | Reserves HBM headroom for runtime memory. |
    | `--enable-prefix-caching` | Enables automatic prefix caching. |
    | `--speculative-config` | Enables one Qwen3.5 MTP speculative token. |
    | `--compilation-config` | Uses `FULL_DECODE_ONLY` ACL Graph replay. |
    | `--additional-config` | Enables CPU binding and Fused MC2. |

    Common Issues Tip: If a worker exits immediately, confirm that Node 0 is
    already running, `--data-parallel-address` resolves to Node 0, all nodes
    use the same RPC port, and every worker uses a unique DP start rank.

=== "A2 series"

    The validated mixed deployment uses eight Atlas 800 A2 (64GB × 8) nodes
    with the `Qwen3.8-2.4T-A95B-w4a8` checkpoint. Data parallelism spans the
    eight nodes, each node runs one DP rank, tensor parallelism uses all 8
    NPUs in the node, and the resulting topology is DP8/TP8/EP64.

    The W4A8 checkpoint is used on A2 so that enough HBM remains for the
    large activation footprint of the 95B activated parameters. Because the
    per-token MoE intermediate buffers are large, `--max-num-batched-tokens`
    is reduced to 4096 and `--gpu-memory-utilization` is raised to 0.92
    compared with the A3 configuration. The smaller token budget keeps the
    MoE activation peak low enough to leave KV cache room for a 262144
    context length.

    === "Node 0"

        ```shell
        # Values that must be adapted to the target environment.
        export MODEL_PATH=<QWEN3_8_MODEL_PATH>
        export TOKENIZER_PATH=<QWEN3_8_TOKENIZER_PATH>
        export LOCAL_IP=<NODE0_LOCAL_IP>
        export NIC_NAME=<NODE0_NIC_NAME>
        export PORT=<SERVICE_PORT>
        export RPC_PORT=<DP_RPC_PORT>
        export DP_SIZE=8
        export TP_SIZE=8

        export VLLM_ENGINE_READY_TIMEOUT_S=7200
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
        export HCCL_BUFFSIZE=1024
        export HCCL_BUFFSIZE_EP=2048
        export HCCL_CONNECT_TIMEOUT=1800
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_IF_IP=$LOCAL_IP
        export HCCL_INTRA_PCIE_ENABLE=1
        export HCCL_INTRA_ROCE_ENABLE=0
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_SOCKET_IFNAME=$NIC_NAME
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export GLOO_SOCKET_IFNAME=$NIC_NAME
        export TP_SOCKET_IFNAME=$NIC_NAME
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1

        vllm serve $MODEL_PATH \
            --host 0.0.0.0 \
            --port $PORT \
            --served-model-name qwen3.8 \
            --tokenizer $TOKENIZER_PATH \
            --trust-remote-code \
            --quantization ascend \
            --safetensors-load-strategy lazy \
            --tensor-parallel-size $TP_SIZE \
            --data-parallel-size $DP_SIZE \
            --data-parallel-size-local 1 \
            --data-parallel-address $LOCAL_IP \
            --data-parallel-rpc-port $RPC_PORT \
            --enable-prefix-caching \
            --enable-expert-parallel \
            --max-model-len 262144 \
            --max-num-seqs 8 \
            --max-num-batched-tokens 4096 \
            --gpu-memory-utilization 0.92 \
            --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":1}'
        ```

    === "Nodes 1-7"

        Run this command on every worker. Set `LOCAL_IP` and `NIC_NAME` to the
        current node and set `DP_START_RANK` to a unique value from `1` to
        `7`.

        ```shell
        # Values that must be adapted to the target environment.
        export MODEL_PATH=<QWEN3_8_MODEL_PATH>
        export TOKENIZER_PATH=<QWEN3_8_TOKENIZER_PATH>
        export LOCAL_IP=<WORKER_LOCAL_IP>
        export NODE0_IP=<NODE0_LOCAL_IP>
        export NIC_NAME=<WORKER_NIC_NAME>
        export PORT=<SERVICE_PORT>
        export RPC_PORT=<DP_RPC_PORT>
        export DP_SIZE=8
        export DP_START_RANK=<1_TO_7>
        export TP_SIZE=8

        export VLLM_ENGINE_READY_TIMEOUT_S=7200
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
        export HCCL_BUFFSIZE=1024
        export HCCL_BUFFSIZE_EP=2048
        export HCCL_CONNECT_TIMEOUT=1800
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_IF_IP=$LOCAL_IP
        export HCCL_INTRA_PCIE_ENABLE=1
        export HCCL_INTRA_ROCE_ENABLE=0
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_SOCKET_IFNAME=$NIC_NAME
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export GLOO_SOCKET_IFNAME=$NIC_NAME
        export TP_SOCKET_IFNAME=$NIC_NAME
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1

        vllm serve $MODEL_PATH \
            --headless \
            --host 0.0.0.0 \
            --port $PORT \
            --served-model-name qwen3.8 \
            --tokenizer $TOKENIZER_PATH \
            --trust-remote-code \
            --quantization ascend \
            --safetensors-load-strategy lazy \
            --tensor-parallel-size $TP_SIZE \
            --data-parallel-size $DP_SIZE \
            --data-parallel-size-local 1 \
            --data-parallel-start-rank $DP_START_RANK \
            --data-parallel-address $NODE0_IP \
            --data-parallel-rpc-port $RPC_PORT \
            --enable-prefix-caching \
            --enable-expert-parallel \
            --max-model-len 262144 \
            --max-num-seqs 8 \
            --max-num-batched-tokens 4096 \
            --gpu-memory-utilization 0.92 \
            --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":1}'
        ```

    The following values differ between the master and worker nodes:

    | Setting | Node 0 | Nodes 1-7 | Description |
    | --- | --- | --- | --- |
    | `LOCAL_IP` | Node 0 IP | Current worker IP | Each node uses its own communication IP. |
    | `NODE0_IP` | Not required | Node 0 IP | Workers use this address to join the DP group. |
    | `--headless` | Omitted | Enabled | Workers do not expose an API endpoint. |
    | `--data-parallel-address` | `$LOCAL_IP` | `$NODE0_IP` | Always resolves to Node 0. |
    | `--data-parallel-start-rank` | `0` by default | `1` to `7` | Every node must own a unique DP rank. |

    Key deployment parameters:

    | Parameter | Description |
    | --- | --- |
    | `--tensor-parallel-size 8` | Uses all 8 NPUs in one A2 node for tensor parallelism. |
    | `--data-parallel-size 8` | Creates eight global DP ranks across eight nodes. |
    | `--data-parallel-size-local 1` | Runs one DP rank on the current node. |
    | `--data-parallel-start-rank` | Selects the global DP rank of a worker node. |
    | `--enable-expert-parallel` | Enables expert parallelism for the MoE layers. |
    | `--max-model-len 262144` | Sets the maximum combined input and output length. |
    | `--quantization ascend` | Enables Ascend W4A8 quantization for the validated checkpoint. |
    | `--safetensors-load-strategy lazy` | Avoids prefetching the complete checkpoint from NFS. |
    | `--max-num-seqs 8` | Sets eight active sequences for each DP group. |
    | `--max-num-batched-tokens 4096` | Controls the scheduler token budget. Larger values increase the MoE activation peak and shrink the KV cache available for the 262144 context. |
    | `--gpu-memory-utilization 0.92` | Reserves HBM headroom for the weights and activation peak on 64GB nodes. |
    | `--enable-prefix-caching` | Enables automatic prefix caching. |
    | `--speculative-config` | Enables one Qwen3.5 MTP speculative token. |
    | `--compilation-config` | Uses `FULL_DECODE_ONLY` ACL Graph replay. |
    | `--additional-config` | Enables CPU binding and Fused MC2. |

    Common Issues Tip: If a worker exits immediately, confirm that Node 0 is
    already running, `--data-parallel-address` resolves to Node 0, all nodes
    use the same RPC port, and every worker uses a unique DP start rank.

### 5.2 Prefill-Decode Disaggregation

PD disaggregation separates Prefill and Decode into different service groups.
Prefill nodes process prompt chunks, Decode nodes serve token generation, and
a proxy forwards requests between them.

Detailed commands and configuration for PD disaggregation will be provided in
a future update. Stay tuned.

## 6 Functional Verification

After all DP ranks are ready, send a text request to the Node 0 API endpoint.
The open-weight model always uses thinking mode and does not support
multimodal input.

```shell
curl http://<server_ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8",
    "messages": [
      {"role": "user", "content": "Write a Python function to merge two sorted linked lists."}
    ],
    "temperature": 1.0,
    "top_p": 0.95,
    "stream": true,
    "chat_template_kwargs": {
      "enable_thinking": true,
      "preserve_thinking": true
    }
  }'
```

Expected Result: Each response begins with reasoning wrapped in
`<think>...</think>`, followed by the final answer. The service returns HTTP
200 and a `choices` field containing generated text. Worker nodes are headless
and do not accept HTTP requests directly.

The model officially supports these reasoning effort levels:

- `xhigh`: default; deeper reasoning for complex tasks.
- `medium`: balances accuracy and speed.
- `low`: prioritizes speed and cost.

## 7 Accuracy Evaluation

### 7.1 Prepare the Evaluation Environment

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for the
evaluation environment and dataset preparation.

### 7.2 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md)
   for details.

2. After execution, you can get the result. Here are the GPQA Diamond results
   of
   [Qwen3.8-2.4T-A95B-w4a8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-2.4T-A95B-w4a8)
   and
   [Qwen3.8-2.4T-A95B-w8a8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-2.4T-A95B-w8a8)
   for reference only.

| dataset | model | metric | mode | vllm-api-general-chat |
| --- | --- | --- | --- | ---: |
| GPQA Diamond | Qwen3.8-2.4T-A95B-w4a8 | accuracy | gen | 91.92% |
| GPQA Diamond | Qwen3.8-2.4T-A95B-w8a8 | accuracy | gen | 93.75% |

## 8 Performance Evaluation

### 8.1 Install AISBench

Run AISBench in a separate environment or container so that the load generator
does not affect the serving processes. Refer to
[Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation).

### 8.2 Performance Service Configuration

Use the deployment in Section 5 as the baseline. Change the following values
on all four nodes:

| Parameter | Standard deployment | Performance test |
| --- | ---: | ---: |
| `--max-model-len` | 131072 | 250000 |
| `--max-num-batched-tokens` | 16384 | 8192 |
| `--gpu-memory-utilization` | 0.85 | 0.95 |

### 8.3 Run the Tests

Run performance evaluation of `Qwen3.8-2.4T-A95B` as an example. Refer to
[vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

```shell
vllm bench serve \
  --model Qwen/Qwen3.8-2.4T-A95B \
  --served-model-name qwen3.8 \
  --base-url http://<server_ip>:8000 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --num-prompts 16 \
  --max-concurrency 4 \
  --save-result \
  --result-dir ./
```

Record the A3 node count, TP/PP/EP topology, context length, concurrency,
reasoning effort, and weight revision together with the result.

## 9 Performance Tuning

### 9.1 Recommended Configurations

The following configuration is validated in a specific test environment and
is for reference only. The optimal configuration depends on maximum
input/output length, request concurrency, prefix cache hit rate, quantization,
and workload characteristics. Tune the parameters in Section 9.2 based on your
actual workload.

**Table 1: Scenario Overview**

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Topology |
| --- | --- | ---: | --- | --- |
| Mixed Prefill/Decode | Four Atlas 800 A3 nodes | 64 | W8A8 | DP4/TP16/EP64 |

> `*Total NPUs` indicates the total number of NPUs used across all nodes.

**Table 2: Detailed Configuration**

| Max Num Seqs | Max Model Len | Max Num Batched Tokens | GPU Memory Utilization | MTP Tokens | Prefix Cache | Main Optimizations |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 8 per DP | 131072 | 16384 | 0.85 | 1 | On | Full decode ACL Graph, Fused MC2, CPU binding |

> For complete startup commands and parameter descriptions, refer to the
> deployment example in Chapter 5.

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Refer to
[Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md)
for general tuning methods, and refer to the
[Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for feature
descriptions.

Recommended tuning order:

1. Set the deployment topology first. The validated configuration uses four
   A3 nodes with DP4/TP16/EP64.
2. Choose the maximum context length with `--max-model-len`. Long context
   increases KV cache usage, so reduce `--max-num-seqs` or
   `--gpu-memory-utilization` if OOM occurs.
3. Tune `--max-num-batched-tokens`. Larger values usually improve prefill
   throughput but increase activation memory. Decode-heavy workloads usually
   need smaller values.
4. Tune `--max-num-seqs` according to service concurrency. Requests above
   this value wait in the queue, and the waiting time is counted in TTFT and
   TPOT.
5. Tune `--gpu-memory-utilization`. Increase it to provide more KV cache, but
   leave headroom for runtime memory fluctuation and expert imbalance.
6. Tune `--speculative-config`. MTP can improve decode throughput, but the
   optimal `num_speculative_tokens` depends on acceptance rate and workload.
7. Tune ACL Graph capture. `FULL_DECODE_ONLY` is recommended for decode. If
   `cudagraph_capture_sizes` is set manually, include common decode batch
   sizes.

#### 9.2.2 Model-Specific Optimizations

| Optimization | Enablement | Benefit | Notes |
| --- | --- | --- | --- |
| Chunked Prefill | Enabled by the vLLM V1 scheduler | Splits long prefill inputs to reduce per-step memory peaks. | Tune with `--max-num-batched-tokens`. |
| Prefix Cache | `--enable-prefix-caching` | Reuses KV state for repeated prefixes. | The benefit depends on the prefix cache hit rate. |
| Qwen3.5 MTP speculative decoding | `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'` | Improves decode throughput when the acceptance rate is good. | Tune the speculative token count for the target workload. |
| Full decode ACL Graph | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | Reduces operator dispatch overhead during decode. | Used by the validated configuration. |
| Fused MC2 | `--additional-config '{"enable_fused_mc2":1}'` | Enables fused MoE communication and computation. | Used by the validated configuration. |
| Lazy Safetensors | `--safetensors-load-strategy lazy` | Avoids prefetching the complete checkpoint from shared storage. | Used for the W8A8 checkpoint. |
| CPU binding | `--additional-config '{"enable_cpu_binding":true}'` | Reduces CPU scheduling jitter. | Explicitly enabled in the deployment commands. |

## 10 FAQ

For common environment, installation, and general parameter issues, refer to
the [Public FAQs](../../faqs.md).

### Q1: Which inputs are supported by the open-weight model?

Qwen3.8-2.4T-A95B is a text-only model. It does not support multimodal input.

### Q2: Can thinking mode be disabled?

No. Keep `enable_thinking=true`. The response starts with reasoning in a
`<think>` block and then returns the final answer.

### Q3: How should the parallel sizes be selected?

Select TP, PP, DP, and EP together according to model compatibility, weight
memory, KV-cache capacity, and communication performance. Validate the
complete topology on the target cluster.
