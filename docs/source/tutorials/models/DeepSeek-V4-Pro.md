# DeepSeek-V4-Pro

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.
    
    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

DeepSeek-V4 introduces several key upgrades over DeepSeek-V3:

- The Manifold-Constrained Hyper-Connections (mHC) to strengthen conventional residual connections.
- A hybrid attention architecture, which greatly improves long-context efficiency through Compress-4-Attention and Compress-128-Attention. For the Mixture-of-Experts (MoE) components, it still adopts the DeepSeekMoE architecture, with only minor adjustments.

DeepSeek-V4-Pro, the maximum reasoning effort mode of DeepSeek-V4, significantly advances the knowledge capabilities of open-source models, firmly establishing itself as the best open-source model available today. It achieves top-tier performance in coding benchmarks and significantly bridges the gap with leading closed-source models on reasoning and agentic tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

This document is written based on the vLLM-Ascend v0.30.0 release. This model is supported in this release.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

|  Weight Version             | Hardware Requirements                                               | Download Links |
|-----------------------------|---------------------------------------------------------------------|----------------|
| `DeepSeek-V4-Pro-w4a8-mtp`  | 2 Atlas 800I A3 (128GB × 8) nodes or 4 Atlas 800I A2 (64GB × 8) nodes | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp) |
| `DeepSeek-V4-Pro-0813-w4a8` | 2 Atlas 800I A3 (128GB × 8) nodes or 4 Atlas 800I A2 (64GB × 8) nodes | [ModelScope](https://modelscope.cn/models/Eco-Tech/DeepSeek-V4-Pro-0813-w4a8)    |
| `DeepSeek-V4-Pro-0813-w4a4c8` | 950PR&950DT Products | Quantize locally with [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_pro_w4a4c8.yaml) |

- `DeepSeek-V4-Pro-0813-w4a8` (Official release with DSpark after quantized): This checkpoint includes the DSpark draft weights, so no separate draft-model path is required.

- `DeepSeek-V4-Pro-0813-w4a4c8` (W4A4 MXFP4 mixed-precision quantization): For how the ultra-low-bit quantization preserves accuracy, see the [W4A4C8 FAQ](#q-how-do-the-w4a4c8-ultra-low-bit-quantized-weights-preserve-accuracy).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy a multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image-selection).

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

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md).

If you want to deploy a multi-node environment, you need to set up the environment on each node.

## 5 Online Service Deployment

!!! note

    In this tutorial, we suppose you downloaded the model weight to `/root/.cache/modelscope/hub/models/vllm-ascend/`. Feel free to change it to your own path.

    It is recommended that the following service code be encapsulated in a .sh script file and executed in Bash mode.

### 5.1 Multi-Node Online Deployment

The quantized model `DeepSeek-V4-Pro-w4a8-mtp` requires at least 2 Atlas 800I A3 (128GB × 8) nodes or 4 Atlas 800I A2 (64GB × 8) nodes. Run the following scripts on each node respectively.

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

    export HCCL_CONNECT_TIMEOUT=7200
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export VLLM_RPC_TIMEOUT=1800000

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
      --host 0.0.0.0 \
      --port 8000 \
      --data-parallel-address $node0_ip  \
      --data-parallel-size 4 \
      --data-parallel-size-local 1 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --served-model-name dsv4 \
      --max-model-len 135000 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 16 \
      --gpu-memory-utilization 0.9 \
      --block-size 128 \
      --no-enable-prefix-caching \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --attention_config.indexer_kv_dtype int8 \
      --model-loader-extra-config '{
         "enable_multithread_load": true,
         "num_threads": 128
      }' \
      --quantization ascend \
      --speculative-config '{
         "num_speculative_tokens": 1,
         "method": "mtp",
         "enforce_eager": true
       }' \
      --compilation-config '{
         "cudagraph_mode":"FULL_DECODE_ONLY"
      }' \
      --additional-config '{
         "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
         },
         "enable_cpu_binding": true,
         "enable_shared_expert_dp": true,
         "enable_flashcomm1": true,
         "multistream_overlap_shared_expert":true
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

    export HCCL_CONNECT_TIMEOUT=7200
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export VLLM_RPC_TIMEOUT=1800000
    
    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
      --host 0.0.0.0 \
      --port 8000 \
      --headless \
      --data-parallel-address $node0_ip  \
      --data-parallel-size 4 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 1 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --served-model-name dsv4 \
      --max-model-len 135000 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 16 \
      --gpu-memory-utilization 0.9 \
      --block-size 128 \
      --no-enable-prefix-caching \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --attention_config.indexer_kv_dtype int8 \
      --model-loader-extra-config '{
         "enable_multithread_load": true,
         "num_threads": 128
      }' \
      --quantization ascend \
      --speculative-config '{
         "num_speculative_tokens": 1,
         "method": "mtp",
         "enforce_eager": true
       }' \
      --compilation-config '{
         "cudagraph_mode":"FULL_DECODE_ONLY"
      }' \
      --additional-config '{
         "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
         },
         "enable_cpu_binding": true,
         "enable_shared_expert_dp": true,
         "enable_flashcomm1": true,
         "multistream_overlap_shared_expert":true
      }'
    ```

=== "A3 series with DSpark"

    Use the official `DeepSeek-V4-Pro-0813-w4a8` checkpoint on both nodes. The example keeps the existing DP2/TP16 topology and changes only the DSpark-related and required runtime parameters.

    **Node0**

    ```bash
    # nic_name is the network interface that owns local_ip on the current node.
    nic_name="xxx"
    local_ip="xxx"

    # node0_ip must be the local_ip of Node0 on every node.
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_BUFFSIZE=1024
    export HCCL_CONNECT_TIMEOUT=7200
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export VLLM_RPC_TIMEOUT=1800000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export TASK_QUEUE_ENABLE=1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
    export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

    vllm serve /path/to/DeepSeek-V4-Pro-0813-w4a8 \
      --host 0.0.0.0 \
      --port 8000 \
      --data-parallel-address $node0_ip \
      --data-parallel-rpc-port 13399 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --served-model-name dsv4 \
      --max-model-len 135000 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 16 \
      --gpu-memory-utilization 0.9 \
      --block-size 32 \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --attention_config.indexer_kv_dtype int8 \
      --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
      --quantization ascend \
      --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{
        "ascend_compilation_config": {
            "enable_npugraph_ex": true,
            "enable_static_kernel": false
        },
        "enable_cpu_binding": true,
        "enable_flashcomm1": true,
        "multistream_overlap_shared_expert": true
      }'
    ```

    **Node1**

    Use the same environment variables and command as Node0, with the following DP changes:

    ```bash
    nic_name="xxx"
    local_ip="xxx"
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_BUFFSIZE=1024
    export HCCL_CONNECT_TIMEOUT=7200
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export VLLM_RPC_TIMEOUT=1800000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export TASK_QUEUE_ENABLE=1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
    export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

    vllm serve /path/to/DeepSeek-V4-Pro-0813-w4a8 \
      --host 0.0.0.0 \
      --port 8000 \
      --headless \
      --data-parallel-address $node0_ip \
      --data-parallel-rpc-port 13399 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 1 \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --served-model-name dsv4 \
      --max-model-len 135000 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 16 \
      --gpu-memory-utilization 0.9 \
      --block-size 32 \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --attention_config.indexer_kv_dtype int8 \
      --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
      --quantization ascend \
      --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{
        "ascend_compilation_config": {
            "enable_npugraph_ex": true,
            "enable_static_kernel": false
        },
        "enable_cpu_binding": true,
        "enable_flashcomm1": true,
        "multistream_overlap_shared_expert": true
      }'
    ```

Key Parameter Descriptions:

- `--data-parallel-size` sets the global number of data parallel ranks, and `--data-parallel-size-local` sets the number of DP ranks on the current node.
- `--data-parallel-start-rank` specifies the starting data parallel rank of the current node. Set a unique value on each headless node (e.g., Node1 = 1). On Node0, omit `--data-parallel-start-rank` even when the start rank is 0, to avoid enabling hybrid load balancing with headless remote engines.
- `--data-parallel-address` specifies the IP address of the data parallel master node (Node0). It must be consistent across all nodes.
- `--data-parallel-rpc-port` is the DP RPC port. Use the same value on all nodes and ensure the port is available.
- `--tensor-parallel-size` sets the tensor parallel size within each DP rank. Configure it together with the DP sizes according to the deployment topology and available NPUs.
- `--enable-expert-parallel` enables expert parallelism for MoE layers. Do not mix MoE tensor parallelism and expert parallelism in the same MoE layer.
- `--headless` (used on non-master nodes) disables the API server on the node, since only the master node serves requests.
- `--max-model-len` specifies the maximum context length. Adjust it according to your actual scenario.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` is the maximum number of tokens processed in one scheduler step. A larger value can improve prefill efficiency but consumes more activation memory.
- `--tokenizer-mode deepseek_v4`, `--tool-call-parser deepseek_v4`, `--enable-auto-tool-choice`, and `--reasoning-parser deepseek_v4` enable the DeepSeek-V4 tokenizer behavior, automatic tool calling, and reasoning-output parsing.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- `--block-size` sets the KV cache block size. To enable the experimental 4k prefix cache hit support, change it from `128` to `32`.
- `--quantization ascend` enables Ascend quantization for the W4A8 model.
- `--model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}'` selects the multi-thread weight iterator. `enable_multithread_load` must be a JSON boolean and `num_threads` must be a positive integer.
- `--safetensors-load-strategy prefetch` is an alternative that warms checkpoint files into the OS page cache before the normal iterator loads them. Do not combine it with multi-thread loading: the default loader rejects `prefetch`, `eager`, or `torchao` when `enable_multithread_load` is `true`. The examples in this document use only multi-thread loading.
- `--speculative-config` configures speculative decoding. Use `mtp` for the preview MTP checkpoint and `dspark` for `DeepSeek-V4-Pro-0813-w4a8`. For DSpark, use the value declared by the checkpoint; the example uses five speculative tokens, and all ranks must use the same value.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full ACL graph execution in the decode phase to reduce scheduling latency.
- `--additional-config` enables Ascend-specific optimizations. `enable_npugraph_ex` enables enhanced ACL graph execution, `enable_static_kernel: false` keeps static-kernel compilation disabled, `enable_cpu_binding` enables Ascend-native CPU binding, `enable_shared_expert_dp` enables data parallelism for shared experts, and `multistream_overlap_shared_expert` overlaps shared expert computation for better MoE throughput.
- `enable_flashcomm1: true` in `--additional-config` enables the FlashComm1 communication optimization. It is auto-enabled whenever `enable_dsa_cp` is enabled.
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` retains prefix-cache checkpoints every 4096 tokens. It takes effect only when prefix caching is enabled and must be a non-negative multiple of `--block-size`; `4096` matches the DSpark example's block size of `32`.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQs](../../faqs.md) for troubleshooting.

Service Verification:

```shell
curl http://<node0_ip>:8000/v1/chat/completions \
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

The following sections describe PD separation deployment on both Atlas 800I A3 (128GB × 8) and Atlas 800I A2 (64GB × 8) multi-node environments.

#### 5.2.1 A3 Series PD Separation Deployment

This section shows a deployment with one Prefill pool and one Decode pool on Atlas 800I A3 (128GB × 8). The example uses two physical nodes for each pool.

Use the official `DeepSeek-V4-Pro-0813-w4a8` checkpoint on all nodes. It includes the DSpark draft weights, so no separate draft-model path is required. Replace the model path, IP addresses, and network interface names below with your environment's values.

Before you start, please:

1. Prepare `launch_online_dp.py` on each Decode node.

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
            "--pcp-size",
            type=int,
            default=1,
            help="Prefill context parallel size of each DP rank."
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
            default="12321",
            help="Port for data parallel master node."
        )
        parser.add_argument(
            "--vllm-start-port",
            type=int,
            default=7100,
            help="Starting port for the engine."
        )
        return parser.parse_args()

    args = parse_args()
    dp_size = args.dp_size
    tp_size = args.tp_size
    pcp_size = args.pcp_size
    dp_size_local = args.dp_size_local
    if dp_size_local == -1:
        dp_size_local = dp_size
    dp_rank_start = args.dp_rank_start
    dp_address = args.dp_address
    dp_rpc_port = args.dp_rpc_port
    vllm_start_port = args.vllm_start_port

    # Each local DP instance uses TP x PCP devices.
    devices_per_dp_rank = tp_size * pcp_size

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
            str(pcp_size),
        ]
        subprocess.run(command, check=True)

    if __name__ == "__main__":
        template_path = "./run_dp_template.sh"
        if not os.path.exists(template_path):
            print(f"Template file {template_path} does not exist.")
            sys.exit(1)

        processes = []
        for i in range(dp_size_local):
            dp_rank = dp_rank_start + i
            vllm_engine_port = vllm_start_port + i
            device_start = i * devices_per_dp_rank
            device_end = (i + 1) * devices_per_dp_rank
            visible_devices = ",".join(str(x) for x in range(device_start, device_end))
            process = multiprocessing.Process(target=run_command,
                                            args=(visible_devices, dp_rank,
                                                    vllm_engine_port))
            processes.append(process)
            process.start()

        exit_code = 0
        for process in processes:
            process.join()
            if process.exitcode != 0:
                exit_code = 1
        sys.exit(exit_code)
    ```

    Parameter descriptions:

    |Parameter|Type|Required|Default|Description|
    |---------|----|--------|-------|-----------|
    |`--dp-size`|int|Yes|-|Data parallel size (total number of DP ranks across all nodes).|
    |`--tp-size`|int|No|1|Tensor parallel size within each DP rank.|
    |`--pcp-size`|int|No|1|Prefill context parallel size within each DP rank; passed as `$8` / `PCP_SIZE` to `--prefill-context-parallel-size`. Each local DP instance uses `TP × PCP` devices.|
    |`--dp-size-local`|int|No|(same as `--dp-size`)|Number of DP ranks on the current node. If not set, defaults to `--dp-size`.|
    |`--dp-rank-start`|int|No|0|Starting rank offset for data parallel ranks on this node.|
    |`--dp-address`|str|Yes|-|IP address of the data parallel master node.|
    |`--dp-rpc-port`|str|No|12321|RPC port for data parallel master communication.|
    |`--vllm-start-port`|int|No|7100|Starting port for each vLLM engine instance on this node.|

2. Prepare `run_prefill.sh` on both Prefill nodes.

    Set `local_ip` to the current node's communication IP and `MASTER_ADDR` to Prefill node 0's communication IP on both nodes.

    ```shell
    local_ip="xx.xx.xx.x"
    nic_name="xxxx" # NIC that owns local_ip
    MASTER_ADDR="xx.xx.xx.1" # Prefill node 0 on both nodes

    NODE_RANK="${1:?Usage: bash run_prefill.sh <0|1>}"
    HEADLESS_ARGS=()
    if [ "$NODE_RANK" = "1" ]; then
        HEADLESS_ARGS=(--headless)
    fi

    export HCCL_IF_IP="$local_ip"
    export GLOO_SOCKET_IFNAME="$nic_name"
    export TP_SOCKET_IFNAME="$nic_name"
    export HCCL_SOCKET_IFNAME="$nic_name"
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_BUFFSIZE=1024
    export HCCL_CONNECT_TIMEOUT=6000
    export HCCL_EXEC_TIMEOUT=204
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_PP_LAYER_PARTITION="32,29"

    vllm serve /path/to/DeepSeek-V4-Pro-0813-w4a8 \
        --tensor-parallel-size 2 \
        --prefill-context-parallel-size 8 \
        --host 0.0.0.0 \
        --port 7100 \
        "${HEADLESS_ARGS[@]}" \
        --nnodes 2 \
        --node-rank "$NODE_RANK" \
        --master-addr "$MASTER_ADDR" \
        --master-port 12321 \
        --data-parallel-size 1 \
        --pipeline-parallel-size 2 \
        --decode-context-parallel-size 1 \
        --served-model-name dsv4 \
        --enable-expert-parallel \
        --max-model-len 150000 \
        --max-num-batched-tokens 4096 \
        --max-num-seqs 16 \
        --block-size 32 \
        --gpu-memory-utilization 0.92 \
        --enable-chunked-prefill \
        --no-disable-hybrid-kv-cache-manager \
        --enable-prefix-caching \
        --enforce-eager \
        --quantization ascend \
        --tokenizer-mode deepseek_v4 \
        --attention_config.indexer_kv_dtype int8 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --trust-remote-code \
        --seed 1024 \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
        --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
        --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true,"enable_fused_mc2":1}' \
        --kv-transfer-config \
        '{
          "kv_connector": "MooncakeConnectorV2",
          "kv_role": "kv_producer",
          "kv_port": 14579,
          "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 2, "pp_size": 2, "pcp_size": 8},
            "decode": {"dp_size": 16, "tp_size": 2, "pp_size": 1, "pcp_size": 1}
          }
        }'
    ```

3. Prepare `run_dp_template.sh` on both Decode nodes, in the same directory as `launch_online_dp.py`.

    Set `local_ip` and `nic_name` separately on each node. The launcher supplies the visible devices, service port, and parallel parameters.

    ```shell
    local_ip="xx.xx.xx.x"
    nic_name="xxxx" # NIC that owns local_ip

    export HCCL_IF_IP="$local_ip"
    export GLOO_SOCKET_IFNAME="$nic_name"
    export TP_SOCKET_IFNAME="$nic_name"
    export HCCL_SOCKET_IFNAME="$nic_name"
    export ASCEND_RT_VISIBLE_DEVICES="$1"
    export SERVICE_PORT="$2"
    export DP_SIZE="$3"
    export DP_RANK="$4"
    export DP_MASTER_IP="$5"
    export DP_RPC_PORT="$6"
    export TP_SIZE="$7"
    export PCP_SIZE="$8"
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_BUFFSIZE=1800
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_EXEC_TIMEOUT=2000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve /path/to/DeepSeek-V4-Pro-0813-w4a8 \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-address "$DP_MASTER_IP" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --data-parallel-size "$DP_SIZE" \
        --data-parallel-rank "$DP_RANK" \
        --tensor-parallel-size "$TP_SIZE" \
        --prefill-context-parallel-size "$PCP_SIZE" \
        --decode-context-parallel-size 1 \
        --served-model-name dsv4 \
        --enable-expert-parallel \
        --max-model-len 150000 \
        --max-num-batched-tokens 180 \
        --max-num-seqs 30 \
        --block-size 32 \
        --gpu-memory-utilization 0.95 \
        --enable-chunked-prefill \
        --no-disable-hybrid-kv-cache-manager \
        --no-enable-prefix-caching \
        --async-scheduling \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --quantization ascend \
        --tokenizer-mode deepseek_v4 \
        --attention_config.indexer_kv_dtype int8 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --trust-remote-code \
        --seed 1024 \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
        --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true}}' \
        --kv-transfer-config \
        '{
          "kv_connector": "MooncakeConnectorV2",
          "kv_role": "kv_consumer",
          "kv_port": 14579,
          "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 2, "pp_size": 2, "pcp_size": 8},
            "decode": {"dp_size": 16, "tp_size": 2, "pp_size": 1, "pcp_size": 1}
          }
        }'
    ```

4. Start the two Prefill nodes and two Decode nodes in separate terminals on the corresponding machines.

    ```shell
    # Prefill node 0
    bash run_prefill.sh 0

    # Prefill node 1: headless
    bash run_prefill.sh 1

    # Decode node 0
    python launch_online_dp.py --dp-size 16 --tp-size 2 --pcp-size 1 --dp-size-local 8 --dp-rank-start 0 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100

    # Decode node 1
    python launch_online_dp.py --dp-size 16 --tp-size 2 --pcp-size 1 --dp-size-local 8 --dp-rank-start 8 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
    ```

5. Deploy the P-D disaggregation proxy.

    Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

#### 5.2.2 A2 Series PD Separation Deployment

This section shows a deployment with one Prefill pool and one Decode pool on Atlas 800I A2 (64GB × 8). The example uses four physical nodes for each pool.

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

        export HCCL_OP_EXPANSION_MODE="AIV"
        export TASK_QUEUE_ENABLE=1

        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=6000

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=1024
        export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
        export VLLM_USE_V2_MODEL_RUNNER=1

        sysctl -w vm.swappiness=0
        sysctl -w kernel.numa_balancing=0
        sysctl kernel.sched_migration_cost_ns=50000

        export ASCEND_RT_VISIBLE_DEVICES=$1
        
        # Ensure the model path matches the directory recorded during download
        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-0813-w4a8 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name dsv4 \
            --max-model-len 150000 \
            --max-num-batched-tokens 4096 \
            --max-num-seqs 16 \
            --gpu-memory-utilization 0.92 \
            --block-size 32 \
            --no-enable-prefix-caching \
            --no-disable-hybrid-kv-cache-manager \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --attention_config.indexer_kv_dtype int8 \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --trust-remote-code \
            --quantization ascend \
            --enforce-eager \
            --speculative-config '{"num_speculative_tokens": 7, "method":"dspark", "enforce_eager": true}' \
            --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_flashcomm1":true,"enable_fused_mc2":1}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "kv_connector_extra_config": {
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

    2. Decode node (4 D nodes share the same script)

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.5/6/7/8 # change to your own ip

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
        export HCCL_BUFFSIZE=1800
        export VLLM_USE_V2_MODEL_RUNNER=1

        sysctl -w vm.swappiness=0
        sysctl -w kernel.numa_balancing=0
        sysctl kernel.sched_migration_cost_ns=50000

        export ASCEND_RT_VISIBLE_DEVICES=$1
        
        # Ensure the model path matches the directory recorded during download
        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-0813-w4a8 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name dsv4 \
            --max-model-len 150000 \
            --max-num-batched-tokens 256 \
            --max-num-seqs 32 \
            --async-scheduling \
            --gpu-memory-utilization 0.92 \
            --block-size 32 \
            --no-enable-prefix-caching \
            --no-disable-hybrid-kv-cache-manager \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --attention_config.indexer_kv_dtype int8 \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --trust-remote-code \
            --quantization ascend \
            --speculative-config '{"num_speculative_tokens": 7, "method":"dspark", "enforce_eager": true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}, "enable_cpu_binding":true, "multistream_overlap_shared_expert":true, "scheduler_config":{"recompute_scheduler_enable":true}}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_consumer",
            "kv_port": "30100",
            "kv_connector_extra_config": {
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
        python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    6. Decode node 1

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    7. Decode node 2

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 2 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    8. Decode node 3

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 3 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

4. Deploy the P-D disaggregation proxy.

    Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

Key Parameter Descriptions:

- `--no-disable-hybrid-kv-cache-manager` keeps the hybrid KV cache manager enabled. DeepSeek-V4 KV Pool deployments require this flag; otherwise, the service may OOM during startup.
- `--enforce-eager` forces eager execution on prefill nodes instead of graph compilation.
- `--trust-remote-code` allows the model repository's custom code to be loaded. Only use trusted model repositories.
- `--prefill-context-parallel-size` configures prefill context parallelism (PCP) on Model Runner V2 (MRv2), replacing the DSA-CP configuration used on MRv1. The A3 DSpark example uses PCP size `8` on Prefill and `1` on Decode (`1` disables PCP). The Decode launcher passes `--pcp-size` through `$8` / `PCP_SIZE` to this option. Keep `kv_connector_extra_config.prefill.pcp_size` and `decode.pcp_size` consistent with the PCP size of the corresponding pool.
- `--kv-transfer-config` configures KV cache transfer between the prefill producer and decode consumer in PD separation.
- `kv_connector_extra_config.prefill.dp_size/tp_size` and `decode.dp_size/tp_size` must match the actual global DP, TP, PCP and DCP layout on the prefill and decode sides.
- `additional_config.enable_fused_mc2=1`: enables the Fused MC2 fusion operator to accelerate communication on Prefill nodes (A3 series).
- `scheduler_config.recompute_scheduler_enable: true`: enables the recomputation scheduler. When the KV Cache of the Decode node is insufficient, requests are sent to Prefill for KV Cache recomputation. Enable it only on PD Decode nodes; the legacy top-level key is deprecated.
- `--speculative-config`: Prefill and Decode must use the same DSpark `num_speculative_tokens` value. Use the value declared by the checkpoint; this example uses `7`. The nested `enforce_eager` applies to DSpark draft execution, while the top-level `--enforce-eager` keeps the Prefill target model in eager mode.
- `MooncakeConnectorV2`: required for KV cache transfer when pipeline parallelism (PP) is enabled in this deployment. The A3 example enables PP on Prefill (`--pipeline-parallel-size 2`), so it uses `MooncakeConnectorV2` instead of the previously used `MooncakeHybridConnector`.

Deployment Verification:

After the PD separation service is fully started, send a request through the proxy port on the prefill master node to verify that Prefill and Decode nodes are working correctly together. Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) for the proxy verification method.

Common Issues Tip: If you encounter issues with PD separation deployment, please refer to the [Public FAQs](../../faqs.md) for troubleshooting.

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

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

| dataset | version | metric | mode | vllm-api-general-chat | note |
| ----- | ----- | ----- | ----- | ----- | ----- |
| GPQA | - | accuracy | gen | 91.41 | 0831 official W4A8 weight with DSpark enabled |
| GPQA | - | accuracy | gen | 89.90 | 2 Atlas 800I A3 (128GB × 8) |
| GSM8K | - | accuracy | gen | 96.21 | 2 Atlas 800I A3 (128GB × 8) |
| GSM8K | - | accuracy | gen | 97.54 | 950PR&950DT Products, w4a4c8 |

> **Note**: Dataset evaluation results fluctuate between runs due to sampling and runtime non-determinism. When a dataset is evaluated multiple times, the average score is reported.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are based on the Chapter 5 deployment examples and specific test environments. They are for reference only and are not globally optimal. In particular, the long-context configuration is a suggested starting point for inputs around 1M tokens and must be validated against the actual input/output length, prefix cache hit rate, concurrency, memory usage, and deployment environment. Refer to Section 9.2 for further tuning.

#### Table 1: Scenario Overview

> `*Total NPUs` indicates the total number of NPUs used across all nodes.

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
| --- | --- | --- | --- | --- |
| Short Sequence / High Throughput | 1P1D deployment (2 Prefill + 2 Decode A3 nodes) | 64 (A3) | DeepSeek-V4-Pro-0813-w4a8 | Prefill uses DP1/TP2/PP2/PCP8 with 16 sequences, while Decode uses DP16/TP2/PP1/PCP1 with 30 sequences and 180 batched tokens. Enable FUSED_MC2 on Prefill to prioritize throughput. |
| Long Context (~1M Input, Reference) | 1P1D deployment (2 Prefill + 2 Decode A3 nodes) | 64 (A3) | DeepSeek-V4-Pro-0813-w4a8 | For this separate long-context reference, use DP2/TP16 on both sides, reduce both sides to 2 sequences, and reduce Decode batched tokens to 16 to reserve memory for a 1M-token context. |

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | NPUs | TP | DP | Max Num Seqs | Max Num Batched Tokens | Max Model Len | DSpark Speculation Num | FUSED_MC2 | EP Switch | FC | DSA-CP (MRv1) / PCP (MRv2) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Short Sequence / High Throughput | Prefill Pool | 32 (2 A3 nodes) | 2 | 1 | 16 | 4096 | 150000 | 5 | On | On | On | PCP=8 |
| Short Sequence / High Throughput | Decode Pool | 32 (2 A3 nodes) | 2 | 16 | 30 | 180 | 150000 | 5 | Off | On | Off | PCP=1 |
| Long Context (~1M Input, Reference) | Prefill Pool | 32 (2 A3 nodes) | 16 | 2 | 2 | 4096 | 1048576 | 5 | Off | On | On | PCP=1 |
| Long Context (~1M Input, Reference) | Decode Pool | 32 (2 A3 nodes) | 16 | 2 | 2 | 16 | 1048576 | 5 | Off | On | Off | PCP=1 |

> `FC` reports the FlashComm1.
>
> `DSA-CP (MRv1) / PCP (MRv2)` describes context parallelism according to the model runner version: MRv1 uses the `enable_dsa_cp` switch, while MRv2 uses `--prefill-context-parallel-size` to specify the PCP size. `PCP=8` means eight-way context parallelism; `PCP=1` means no context parallelism. For the long-context Prefill reference, enable FlashComm1 and disable context parallelism (`enable_dsa_cp: false` on MRv1, or PCP size `1` on MRv2).
>
> FUSED_MC2 is disabled in the long-context reference configuration because it increases memory usage. If sufficient memory is available and higher performance is the priority, enabling FUSED_MC2 on Prefill is recommended.
>
> `--max-model-len 1048576` is the total context budget for prompt and generated tokens, not an additional output budget on top of a 1M-token prompt. Reduce `--max-num-seqs` or the actual input length if memory is insufficient.
>
> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Chapter 5](#5-online-service-deployment).

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md); this chapter only covers model-specific issues.

### Q: How do the W4A4C8 ultra-low-bit quantized weights preserve accuracy?

A: For 950PR&950DT Products, the W4A4C8 weights of `DeepSeek-V4-Pro-0813` are quantized with [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_pro_w4a4c8.yaml). To preserve accuracy at such a low bit-width, a mixed-precision strategy is used: only the modules that dominate the parameter count are quantized to 4 bit, while precision-sensitive modules fall back to higher precision or are not quantized at all.

| Module | Quantization configuration |
| ------ | -------------------------- |
| MoE FFN routed experts | W4A4 MXFP4 |
| Attention linear layers | W8A8 MXFP8 |
| FFN shared experts | W8A8 MXFP8 |
| Router gates, compressor projections (`wgate`/`wkv`), indexer projections, and MTP layers | Not quantized (kept in high precision) |

- **Quantization algorithm**: The weights and activations use the MXFP4/MXFP8 microscaling formats. Each tensor is divided into blocks of 32 elements, and each block shares a single scaling factor obtained with symmetric min-max calibration on a mixed calibration dataset. Block-wise scaling adapts to the local dynamic range of each tensor, so the quantization error is far lower than that of uniform INT4 quantization.
- **C8 (FP8 KV cache)**: C8 is a dynamic quantization applied to the KV cache during inference. On 950PR&950DT Products, the KV cache is quantized to FP8 (not INT8). It is unrelated to the parameters stored in the weight checkpoint and is controlled only by the serving configuration. DeepSeek-V4-Flash and DeepSeek-V4-Pro share exactly the same C8 scheme:
    - **Attention KV cache** (FP8 by default on 950PR&950DT Products; switch back to BF16 with `--kv-cache-dtype bfloat16`): the compressed KV latent, which serves as both K and V, is dynamically quantized per 64-element tile to FP8 (E4M3) with a shared E8M0 scale per tile; the RoPE part of the KV cache stays in BF16, and Q is not quantized.
    - **Indexer cache** (enabled with `--attention_config.indexer_kv_dtype fp8`): both indexer Q and K are first rotated by a Hadamard matrix to spread outliers, and then dynamically quantized per token to FP8 (E4M3) with per-token FP32 scales.

For W4A4C8 accuracy scores, see [Accuracy Evaluation](#7-accuracy-evaluation).
