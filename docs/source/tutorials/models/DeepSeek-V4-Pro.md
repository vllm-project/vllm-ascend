# DeepSeek-V4-Pro

## 1 Introduction

DeepSeek-V4 introduces several key upgrades over DeepSeek-V3:

- The Manifold-Constrained Hyper-Connections (mHC) to strengthen conventional residual connections.
- A hybrid attention architecture, which greatly improves long-context efficiency through Compress-4-Attention and Compress-128-Attention. For the Mixture-of-Experts (MoE) components, it still adopts the DeepSeekMoE architecture, with only minor adjustments.

DeepSeek-V4-Pro, the maximum reasoning effort mode of DeepSeek-V4, significantly advances the knowledge capabilities of open-source models, firmly establishing itself as the best open-source model available today. It achieves top-tier performance in coding benchmarks and significantly bridges the gap with leading closed-source models on reasoning and agentic tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `DeepSeek-V4-Pro-0813-w4a8` (Official release with DSpark after quantized): download the production weight from [ModelScope](https://modelscope.cn/models/Eco-Tech/DeepSeek-V4-Pro-0813-w4a8). This checkpoint includes the DSpark draft weights, so no separate draft-model path is required.

- `DeepSeek-V4-Pro-w4a8-mtp` (Quantized version): requires 2 Atlas 800 A3 (128GB × 8) nodes or 4 Atlas 800 A2 (64GB × 8) nodes. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp)

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

The quantized model `DeepSeek-V4-Pro-w4a8-mtp` requires at least 2 Atlas 800 A3 (128GB × 8) nodes or 4 Atlas 800 A2 (64GB × 8) nodes. Run the following scripts on each node respectively.

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

    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

    export HCCL_CONNECT_TIMEOUT=7200
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export VLLM_RPC_TIMEOUT=1800000

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
      --host 0.0.0.0 \
      --port 10010 \
      --data-parallel-address $node0_ip  \
      --data-parallel-size 4 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 0 \
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

    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

    export HCCL_CONNECT_TIMEOUT=7200
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export VLLM_RPC_TIMEOUT=1800000

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
      --host 0.0.0.0 \
      --port 10010 \
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

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
      --host 0.0.0.0 \
      --port 8900 \
      --data-parallel-address $node0_ip \
      --data-parallel-rpc-port 13399 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 0 \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --served-model-name dsv4 \
      --max-model-len 135000  \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 32 \
      --gpu-memory-utilization 0.9 \
      --block-size 128 \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
      --quantization ascend \
      --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '
        {"ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
            },
        "enable_cpu_binding": true,
        "enable_flashcomm1": true,
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

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
      --host 0.0.0.0 \
      --port 8900 \
      --headless \
      --data-parallel-address $node0_ip \
      --data-parallel-rpc-port 13399 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 1 \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --served-model-name dsv4 \
      --max-model-len 135000  \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 32 \
      --gpu-memory-utilization 0.9 \
      --block-size 128 \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
      --quantization ascend \
      --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '
        {"ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
            },
        "enable_cpu_binding": true,
        "enable_flashcomm1": true,
        "multistream_overlap_shared_expert":true}'
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
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
    export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

    vllm serve /path/to/DeepSeek-V4-Pro-0813-w4a8 \
      --host 0.0.0.0 \
      --port 8900 \
      --data-parallel-address $node0_ip \
      --data-parallel-rpc-port 13399 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 0 \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --served-model-name dsv4-pro \
      --max-model-len 135000 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 16 \
      --gpu-memory-utilization 0.9 \
      --block-size 32 \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
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
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
    export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

    vllm serve /path/to/DeepSeek-V4-Pro-0813-w4a8 \
      --host 0.0.0.0 \
      --port 8900 \
      --headless \
      --data-parallel-address $node0_ip \
      --data-parallel-rpc-port 13399 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 1 \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --served-model-name dsv4-pro \
      --max-model-len 135000 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 16 \
      --gpu-memory-utilization 0.9 \
      --block-size 32 \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
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
- `--data-parallel-start-rank` specifies the starting data parallel rank of the current node. Each node must be set to a unique value (e.g., Node0 = 0, Node1 = 1).
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
- `--block-size` sets the KV cache block size. To enable the experimental 4K prefix cache hit support, change it from `128` to `32`.
- `--quantization ascend` enables Ascend quantization for the W4A8 model.
- `--model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}'` selects the multi-thread weight iterator. `enable_multithread_load` must be a JSON boolean and `num_threads` must be a positive integer.
- `--safetensors-load-strategy prefetch` is an alternative that warms checkpoint files into the OS page cache before the normal iterator loads them. Do not combine it with multi-thread loading: the default loader rejects `prefetch`, `eager`, or `torchao` when `enable_multithread_load` is `true`. The examples in this document use only multi-thread loading.
- `--speculative-config` configures speculative decoding. Use `mtp` for the preview MTP checkpoint and `dspark` for `DeepSeek-V4-Pro-0813-w4a8`. For DSpark, use the value declared by the checkpoint; the example uses five speculative tokens, and all ranks must use the same value.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full ACL graph execution in the decode phase to reduce scheduling latency.
- `--additional-config` enables Ascend-specific optimizations. `enable_npugraph_ex` enables enhanced ACL graph execution, `enable_static_kernel: false` keeps static-kernel compilation disabled, `enable_cpu_binding` enables Ascend-native CPU binding, `enable_shared_expert_dp` enables data parallelism for shared experts, and `multistream_overlap_shared_expert` overlaps shared expert computation for better MoE throughput.
- `enable_flashcomm1: true` in `--additional-config` enables the FlashComm1 communication optimization. This is the recommended replacement for the deprecated `VLLM_ASCEND_ENABLE_FLASHCOMM1` environment variable. Configure it explicitly whenever `enable_dsa_cp` is enabled.
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` retains prefix-cache checkpoints every 4096 tokens. It takes effect only when prefix caching is enabled and must be a non-negative multiple of `--block-size`; `4096` matches the DSpark example's block size of `32`.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQs](../../faqs.md) for troubleshooting.

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

The following sections describe PD separation deployment on both Atlas 800 A3 (128GB × 8) and Atlas 800 A2 (64GB × 8) multi-node environments.

#### 5.2.1 A3 Series PD Separation Deployment

This section shows a deployment with one Prefill pool and one Decode pool on Atlas 800 A3 (128GB × 8). The example uses two physical nodes for each pool.

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

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name auto \
            --max-model-len 131072 \
            --max-num-batched-tokens 4096 \
            --max-num-seqs 16 \
            --gpu-memory-utilization 0.92 \
            --block-size 128 \
            --no-disable-hybrid-kv-cache-manager \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --trust-remote-code \
            --quantization ascend \
            --enforce-eager \
            --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
            --additional-config '{"enable_cpu_binding":true,"enable_dsa_cp":true,"enable_flashcomm1":true}' \
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

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name auto \
            --max-model-len 131072 \
            --max-num-batched-tokens 4096 \
            --max-num-seqs 16 \
            --gpu-memory-utilization 0.92 \
            --block-size 128 \
            --no-disable-hybrid-kv-cache-manager \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --trust-remote-code \
            --quantization ascend \
            --enforce-eager \
            --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
            --additional-config '{"enable_cpu_binding":true,"enable_dsa_cp":true,"enable_flashcomm1":true}' \
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
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name auto \
            --max-model-len 131072 \
            --max-num-batched-tokens 120 \
            --max-num-seqs 60 \
            --gpu-memory-utilization 0.9 \
            --block-size 128 \
            --no-enable-prefix-caching \
            --no-disable-hybrid-kv-cache-manager \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --trust-remote-code \
            --quantization ascend \
            --speculative-config '{"num_speculative_tokens": 1, "method":"mtp", "enforce_eager": true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --additional-config '{
                "ascend_compilation_config":{
                    "enable_npugraph_ex":true,
                    "enable_static_kernel":false
                },
            "enable_cpu_binding":true,
            "scheduler_config":{"recompute_scheduler_enable":true}
            }' \
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

##### A3 DSpark Configuration

The official `DeepSeek-V4-Pro-0813-w4a8` checkpoint supports DSpark without a separate draft-model path. Reuse `launch_online_dp.py` above and replace the MTP `run_dp_template.sh` with the following role-specific templates.

1. Prefill nodes

    ```shell
    nic_name="xxxx" # change to the NIC that owns local_ip
    local_ip="xx.xx.xx.x" # use the IP of the current Prefill node

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_TRANSFER_TIMEOUT=600
    export HCCL_EXEC_TIMEOUT=204
    export HCCL_CONNECT_TIMEOUT=6000
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1
    export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
    export ASCEND_RT_VISIBLE_DEVICES=$1

    vllm serve /path/to/DeepSeek-V4-Pro-0813-w4a8 \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --tensor-parallel-size $7 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name dsv4-pro \
        --max-model-len 150000 \
        --max-num-batched-tokens 4096 \
        --max-num-seqs 16 \
        --gpu-memory-utilization 0.94 \
        --block-size 32 \
        --no-disable-hybrid-kv-cache-manager \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
        --trust-remote-code \
        --quantization ascend \
        --enforce-eager \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
        --additional-config '{"enable_cpu_binding":true,"enable_dsa_cp":true,"enable_flashcomm1":true}' \
        --kv-transfer-config \
        '{"kv_connector":"MooncakeHybridConnector",
          "kv_role":"kv_producer",
          "kv_port":"30100",
          "engine_id":"1",
          "kv_connector_extra_config":{
              "prefill":{"dp_size":4,"tp_size":8},
              "decode":{"dp_size":16,"tp_size":2}
          }
        }'
    ```

2. Decode nodes

    ```shell
    nic_name="xxxx" # change to the NIC that owns local_ip
    local_ip="xx.xx.xx.x" # use the IP of the current Decode node

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_TRANSFER_TIMEOUT=600
    export HCCL_EXEC_TIMEOUT=204
    export HCCL_CONNECT_TIMEOUT=1200
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1800
    export TASK_QUEUE_ENABLE=1
    export ASCEND_RT_VISIBLE_DEVICES=$1

    vllm serve /path/to/DeepSeek-V4-Pro-0813-w4a8 \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --tensor-parallel-size $7 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name dsv4-pro \
        --max-model-len 150000 \
        --max-num-batched-tokens 96 \
        --max-num-seqs 8 \
        --gpu-memory-utilization 0.95 \
        --block-size 32 \
        --no-enable-prefix-caching \
        --no-disable-hybrid-kv-cache-manager \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
        --trust-remote-code \
        --quantization ascend \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{
          "ascend_compilation_config": {
              "enable_npugraph_ex": true,
              "enable_static_kernel": false
          },
          "enable_cpu_binding": true,
          "multistream_overlap_shared_expert": true,
          "scheduler_config": {"recompute_scheduler_enable": true}
        }' \
        --kv-transfer-config \
        '{"kv_connector":"MooncakeHybridConnector",
          "kv_role":"kv_consumer",
          "kv_port":"30800",
          "engine_id":"8",
          "kv_connector_extra_config":{
              "prefill":{"dp_size":4,"tp_size":8},
              "decode":{"dp_size":16,"tp_size":2}
          }
        }'
    ```

3. Start two Prefill nodes and two Decode nodes. Replace the addresses with the Prefill and Decode master-node IPs respectively.

    ```shell
    # Prefill node 0
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100

    # Prefill node 1
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 2 --dp-rank-start 2 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100

    # Decode node 0
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 0 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100

    # Decode node 1
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 8 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
    ```

4. Deploy the P-D disaggregation proxy as described in [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md).

#### 5.2.2 A2 Series PD Separation Deployment

This section shows a deployment with one Prefill pool and one Decode pool on Atlas 800 A2 (64GB × 8). The example uses four physical nodes for each pool.

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

        export ASCEND_RT_VISIBLE_DEVICES=$1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
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
            --max-model-len 131072 \
            --max-num-batched-tokens 4096 \
            --max-num-seqs 16 \
            --gpu-memory-utilization 0.9 \
            --no-enable-prefix-caching \
            --no-disable-hybrid-kv-cache-manager \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --trust-remote-code \
            --quantization ascend \
            --enforce-eager \
            --speculative-config '{"num_speculative_tokens": 1, "method":"mtp", "enforce_eager": true}' \
            --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_dsa_cp":true,"enable_flashcomm1":true}' \
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
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name dsv4 \
            --max-model-len 131072 \
            --max-num-batched-tokens 120 \
            --max-num-seqs 60 \
            --gpu-memory-utilization 0.9 \
            --block-size 128 \
            --no-enable-prefix-caching \
            --no-disable-hybrid-kv-cache-manager \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --trust-remote-code \
            --quantization ascend \
            --speculative-config '{"num_speculative_tokens": 1, "method":"mtp", "enforce_eager": true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}, "enable_cpu_binding":true, "scheduler_config":{"recompute_scheduler_enable":true}}' \
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

- `--no-disable-hybrid-kv-cache-manager` keeps the hybrid KV cache manager enabled. DeepSeek-V4 KV Pool deployments require this flag; otherwise, the service may OOM during startup.
- `--enforce-eager` forces eager execution on prefill nodes instead of graph compilation.
- `--trust-remote-code` allows the model repository's custom code to be loaded. Only use trusted model repositories.
- `enable_dsa_cp: true` enables DSA context parallelism on Prefill nodes. DSA-CP depends on FlashComm1, so the same `--additional-config` object must also set `"enable_flashcomm1": true`.
- `--kv-transfer-config` configures KV cache transfer between the prefill producer and decode consumer in PD separation.
- `kv_connector_extra_config.prefill.dp_size/tp_size` and `decode.dp_size/tp_size` must match the actual global DP and TP layout on the prefill and decode sides.
- `VLLM_ASCEND_ENABLE_FUSED_MC2=1`: enables the Fused MC2 fusion operator to accelerate communication on prefill nodes (A3 series).
- `scheduler_config.recompute_scheduler_enable: true`: enables the recomputation scheduler. When the KV Cache of the Decode node is insufficient, requests are sent to Prefill for KV Cache recomputation. Enable it only on PD Decode nodes; the legacy top-level key is deprecated.
- `--speculative-config`: Prefill and Decode must use the same DSpark `num_speculative_tokens` value. Use the value declared by the checkpoint; this example uses `5`. The nested `enforce_eager` applies to DSpark draft execution, while the top-level `--enforce-eager` keeps the Prefill target model in eager mode.
- `MooncakeHybridConnector`: the KV transfer connector used for PD separation, transferring KV Cache between prefill and decode nodes.

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

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

| dataset | version | metric | mode | vllm-api-general-chat | note |
| ----- | ----- | ----- | ----- | ----- | ----- |
| GPQA | - | accuracy | gen | 91.41 | 0831 official W4A8 weight with DSpark enabled |
| GPQA | - | accuracy | gen | 89.90 | 2 Atlas 800 A3 (128GB × 8) |
| GSM8K | - | accuracy | gen | 96.21 | 2 Atlas 800 A3 (128GB × 8) |

## 8 Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are based on the Chapter 5 deployment examples and specific test environments. They are for reference only and are not globally optimal. In particular, the long-context configuration is a suggested starting point for inputs around 1M tokens and must be validated against the actual input/output length, prefix cache hit rate, concurrency, memory usage, and deployment environment. Refer to Section 9.2 for further tuning.

#### Table 1: Scenario Overview

> `*Total NPUs` indicates the total number of NPUs used across all nodes.

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
| --- | --- | --- | --- | --- |
| Short Sequence / High Throughput | 1P1D deployment (2 Prefill + 2 Decode A3 nodes) | 64 (A3) | DeepSeek-V4-Pro-0813-w4a8 | Compared with the long-context configuration, use higher DP and concurrency: Prefill uses DP4/TP8 with 16 sequences, while Decode uses DP16/TP2 with 8 sequences. Enable FUSED_MC2 on Prefill to prioritize throughput. |
| Long Context (~1M Input, Reference) | 1P1D deployment (2 Prefill + 2 Decode A3 nodes) | 64 (A3) | DeepSeek-V4-Pro-0813-w4a8 | Compared with the short-sequence configuration, use DP2/TP16 on both sides, reduce both sides to 2 sequences, and reduce Decode batched tokens to 16 to reserve memory for a 1M-token context. |

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | NPUs | TP | DP | Max Num Seqs | Max Num Batched Tokens | Max Model Len | DSpark Speculation Num | FUSED_MC2 | EP Switch | FC / DSA-CP Switch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Short Sequence / High Throughput | Prefill Pool | 32 (2 A3 nodes) | 8 | 4 | 16 | 4096 | 150000 | 5 | On | On | On / On |
| Short Sequence / High Throughput | Decode Pool | 32 (2 A3 nodes) | 2 | 16 | 8 | 96 | 150000 | 5 | Off | On | Off / Off |
| Long Context (~1M Input, Reference) | Prefill Pool | 32 (2 A3 nodes) | 16 | 2 | 2 | 4096 | 1048576 | 5 | Off | On | On / Off |
| Long Context (~1M Input, Reference) | Decode Pool | 32 (2 A3 nodes) | 16 | 2 | 2 | 16 | 1048576 | 5 | Off | On | Off / Off |

> `FC / DSA-CP Switch` reports the FlashComm1 and DSA context-parallel switches respectively. For the long-context Prefill configuration, set `"enable_flashcomm1": true` and `"enable_dsa_cp": false` in `--additional-config`. On Decode, configure recomputation as `"scheduler_config": {"recompute_scheduler_enable": true}`. Keep the role-specific `--kv-transfer-config` as the final argument, using `"kv_port": "30100"` and `"engine_id": "1"` for Prefill, and `"kv_port": "30800"` and `"engine_id": "8"` for Decode.
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
