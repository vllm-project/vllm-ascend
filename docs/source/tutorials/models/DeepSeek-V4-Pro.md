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

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

|  Weight Version             | Hardware Requirements                                               | Download Links |
|-----------------------------|---------------------------------------------------------------------|----------------|
| `DeepSeek-V4-Pro-w4a8-mtp`  | 2 Atlas 800 A3 (128GB × 8) nodes or 4 Atlas 800 A2 (64GB × 8) nodes | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp) |
| `DeepSeek-V4-Pro-0813-w4a8` |                                                                     | [ModelScope](https://modelscope.cn/models/Eco-Tech/DeepSeek-V4-Pro-0813-w4a8)    |
| `DeepSeek-V4-Pro-0813-w4a4c8` | 950PR&950DT Products | Quantize locally with [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_pro_w4a4c8.yaml) |
| `DeepSeek-V4-Pro-0813` (original) | 2 Ascend 950DT (96GB × 8) servers for the mixed deployment in Section 5.1; 4 Prefill + 4 Decode servers for the 2P1D example in Section 5.2.3 | [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) |

- `DeepSeek-V4-Pro-0813-w4a8` (Official release with DSpark after quantized): This checkpoint includes the DSpark draft weights, so no separate draft-model path is required.

- For Ascend 950DT servers, use the original [`DeepSeek-V4-Pro-0813`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) weights released by DeepSeek on Hugging Face. The Attention weights use MXFP8, while the MoE weights use MXFP4 with 4-bit weights and 8-bit activation computation (W4A8). No Ascend-specific quantization or weight conversion is required. Both the mixed-deployment example in Section 5.1 and the 2P1D example in Section 5.2.3 use Ascend 950DT servers (96GB × 8) with this checkpoint.

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

=== "950DT Products"

    Start the Docker container on each server.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
    export NAME=vllm-ascend

    docker run --rm \
        --name $NAME \
        --net=host \
        --shm-size=512g \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci_manager \
        --device /dev/hisi_hdc \
        --device /dev/ummu \
        --device /dev/uburma \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
        -v /etc/hixlep/:/etc/hixlep/ \
        -v /root/.cache:/root/.cache \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
        -itd $IMAGE bash
    ```

After a successful docker run, you can verify the running container service by executing the `docker ps` command.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md).

If you want to deploy a multi-node environment, you need to set up the environment on each node.

## 5 Online Service Deployment

!!! note

    The A2/A3 examples assume that the model weights are stored in `/root/.cache/modelscope/hub/models/vllm-ascend/`. The Ascend 950DT examples use the original `DeepSeek-V4-Pro-0813` Hugging Face weights at `/root/.cache/DeepSeek-V4-Pro-0813`. Feel free to change these paths to match your environment.

    It is recommended that the following service code be encapsulated in a .sh script file and executed in Bash mode.

### 5.1 Multi-Node Online Deployment

The quantized model `DeepSeek-V4-Pro-w4a8-mtp` requires at least 2 Atlas 800 A3 (128GB × 8) nodes or 4 Atlas 800 A2 (64GB × 8) nodes. The Ascend 950DT mixed-deployment example uses 2 servers (96GB × 8) with the original Hugging Face weights. Run the following scripts on each node respectively.

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
    
    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
      --host 0.0.0.0 \
      --port 8000 \
      --data-parallel-address $node0_ip \
      --data-parallel-rpc-port 13399 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
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
      --attention_config.indexer_kv_dtype int8 \
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
    
    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp \
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
      --max-model-len 135000  \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 32 \
      --gpu-memory-utilization 0.9 \
      --block-size 128 \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --attention_config.indexer_kv_dtype int8 \
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

=== "950DT Products"

    Use the original `DeepSeek-V4-Pro-0813` checkpoint on both servers. Set `node0_ip` to the service IP of Node0 on both servers. Each server hosts one DP rank whose 8 NPUs run prefill context parallel (PCP8), so the engine uses `DP2/TP1` with `--prefill-context-parallel-size 8`.

    **Node0**

    ```bash
    unset https_proxy
    unset http_proxy
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

    nic_name="xxx"              # change to your NIC name
    local_ip="xx.xx.xx.1"       # change to your IP
    node0_ip="xx.xx.xx.1"       # service IP of Node0

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_BUFFSIZE=1024

    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_OP_EXPANSION_MODE=CCU_SCHED
    export TASK_QUEUE_ENABLE=1

    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export HCCL_EXEC_TIMEOUT=2040
    export HCCL_CONNECT_TIMEOUT=1200

    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

    vllm serve /root/.cache/DeepSeek-V4-Pro-0813 \
        --host 0.0.0.0 \
        --served-model-name dsv4 \
        --port 8000 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 6987 \
        --tensor-parallel-size 1 \
        --prefill-context-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 16 \
        --async-scheduling \
        --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
        --trust-remote-code \
        --block-size 32 \
        --kv-cache-dtype fp8 \
        --attention_config.indexer_kv_dtype fp8 \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --gpu-memory-utilization 0.95 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{
          "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
          },
          "enable_cpu_binding":true,
          "enable_fused_mc2":0,
          "enable_shared_expert_dp":true,
          "multistream_overlap_shared_expert":true
        }'
    ```

    **Node1**

    Use the same environment variables as Node0, with the following DP changes (start rank 1 and `--headless`):

    ```bash
    unset https_proxy
    unset http_proxy
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

    nic_name="xxx"              # change to your NIC name
    local_ip="xx.xx.xx.2"       # change to your IP
    node0_ip="xx.xx.xx.1"       # service IP of Node0

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_BUFFSIZE=1024

    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_OP_EXPANSION_MODE=CCU_SCHED
    export TASK_QUEUE_ENABLE=1

    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export HCCL_EXEC_TIMEOUT=2040
    export HCCL_CONNECT_TIMEOUT=1200

    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

    vllm serve /root/.cache/DeepSeek-V4-Pro-0813 \
        --served-model-name dsv4 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank 1 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 6987 \
        --tensor-parallel-size 1 \
        --prefill-context-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 16 \
        --async-scheduling \
        --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
        --trust-remote-code \
        --block-size 32 \
        --kv-cache-dtype fp8 \
        --attention_config.indexer_kv_dtype fp8 \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --gpu-memory-utilization 0.95 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{
          "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
          },
          "enable_cpu_binding":true,
          "enable_fused_mc2":0,
          "enable_shared_expert_dp":true,
          "multistream_overlap_shared_expert":true
        }' \
        --headless
    ```

Key Parameter Descriptions:

- `--data-parallel-size` sets the global number of data parallel ranks, and `--data-parallel-size-local` sets the number of DP ranks on the current node.
- `--data-parallel-start-rank` specifies the starting data parallel rank of the current node. Set a unique value on each headless node (e.g., Node1 = 1). On Node0, omit `--data-parallel-start-rank` even when the start rank is 0, to avoid enabling hybrid load balancing with headless remote engines.
- `--data-parallel-address` specifies the IP address of the data parallel master node (Node0). It must be consistent across all nodes.
- `--data-parallel-rpc-port` is the DP RPC port. Use the same value on all nodes and ensure the port is available.
- `--tensor-parallel-size` sets the tensor parallel size within each DP rank. Configure it together with the DP sizes according to the deployment topology and available NPUs.
- On Ascend 950DT, DeepSeek-V4 does not currently support standalone tensor parallelism (TP-only). TP partitions only the `wq_b`, `wo_a`, and `wo_b` linear layers, so data parallelism is recommended. For long-context prefill, the 950DT examples combine `DP2/TP1` with `--prefill-context-parallel-size 8`, which spreads the prompt across the 8 NPUs of each DP rank. Do not combine it with `enable_dsa_cp`, which is mutually exclusive with prefill context parallelism and unsupported by the V2 model runner.
- `--enable-expert-parallel` enables expert parallelism for MoE layers. Do not mix MoE tensor parallelism and expert parallelism in the same MoE layer.
- `--headless` (used on non-master nodes) disables the API server on the node, since only the master node serves requests.
- `--max-model-len` specifies the maximum context length. Adjust it according to your actual scenario.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` is the maximum number of tokens processed in one scheduler step. A larger value can improve prefill efficiency but consumes more activation memory.
- `--tokenizer-mode deepseek_v4`, `--tool-call-parser deepseek_v4`, `--enable-auto-tool-choice`, and `--reasoning-parser deepseek_v4` enable the DeepSeek-V4 tokenizer behavior, automatic tool calling, and reasoning-output parsing.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- `--block-size` sets the KV cache block size. To enable the experimental 4k prefix cache hit support, change it from `128` to `32`.
- `--quantization ascend` enables Ascend quantization for the W4A8 model used in the A2 and A3 configurations. The original DeepSeek-V4-Pro-0813 weights used on Ascend 950DT do not require this option.
- `--model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}'` selects the multi-thread weight iterator. `enable_multithread_load` must be a JSON boolean and `num_threads` must be a positive integer.
- `--safetensors-load-strategy prefetch` is an alternative that warms checkpoint files into the OS page cache before the normal iterator loads them. Do not combine it with multi-thread loading: the default loader rejects `prefetch`, `eager`, or `torchao` when `enable_multithread_load` is `true`. The examples in this document use only multi-thread loading.
- `--speculative-config` configures speculative decoding. Use `mtp` for the preview MTP checkpoint and `dspark` for the `DeepSeek-V4-Pro-0813` checkpoints. For DSpark, use the value declared by the checkpoint; the examples use five speculative tokens, and all ranks must use the same value.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full ACL graph execution in the decode phase to reduce scheduling latency.
- `--additional-config` enables Ascend-specific optimizations. `enable_npugraph_ex` enables enhanced ACL graph execution, `enable_static_kernel: false` keeps static-kernel compilation disabled, `enable_cpu_binding` enables Ascend-native CPU binding, `enable_shared_expert_dp` enables data parallelism for shared experts, and `multistream_overlap_shared_expert` overlaps shared expert computation for better MoE throughput. `enable_fused_mc2: 0` disables the fused MC2 dispatch/combine path.
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

The following sections describe PD separation deployment on Atlas 800 A3 nodes (128GB × 8), Atlas 800 A2 nodes (64GB × 8), and Ascend 950DT servers (96GB × 8).

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

2. Select one of the following configurations and prepare `run_dp_template.sh` on each node.

=== "A3 series with MTP"

    Both Prefill nodes use the same template. Set `local_ip` to the IP address of the current node.

    **Prefill nodes**

    ```shell
    nic_name="xxxx" # change to your own nic name
    local_ip=xx.xx.xx.x # use the IP of the current Prefill node

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
    export ASCEND_RT_VISIBLE_DEVICES=$1
    # Ensure the model path matches the directory recorded during download
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
        --gpu-memory-utilization 0.92 \
        --block-size 128 \
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
        --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
        --additional-config '{"enable_cpu_binding":true,"enable_dsa_cp":true,"enable_fused_mc2":1,"enable_flashcomm1":true}' \
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

    Both Decode nodes also use one shared template. Set `local_ip` to the IP address of the current node.

    **Decode nodes**

    ```shell
    nic_name="xxxx" # change to your own nic name
    local_ip=xx.xx.xx.x # use the IP of the current Decode node

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
    
    # Ensure the model path matches the directory recorded during download
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
        --attention_config.indexer_kv_dtype int8 \
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

    Start the two Prefill nodes and two Decode nodes. The two Prefill commands are both required because each node owns a different DP rank range.

    ```shell
    # Prefill node 0
    python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100

    # Prefill node 1
    python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 1 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100

    # Decode node 0
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 0 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100

    # Decode node 1
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 8 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
    ```

=== "A3 series with DSpark"

    The official `DeepSeek-V4-Pro-0813-w4a8` checkpoint supports DSpark without a separate draft-model path. Reuse `launch_online_dp.py` above and use the following role-specific templates.

    **Prefill nodes**

    ```shell
    nic_name="xxxx" # change to the NIC that owns local_ip
    local_ip="xx.xx.xx.x" # use the IP of the current Prefill node

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
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
    export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
    export ASCEND_RT_VISIBLE_DEVICES=$1
    
    # Ensure the model path matches the directory recorded during download
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
        --served-model-name dsv4 \
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
        --attention_config.indexer_kv_dtype int8 \
        --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
        --trust-remote-code \
        --quantization ascend \
        --enforce-eager \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
        --additional-config '{"enable_cpu_binding":true,"enable_dsa_cp":true,"enable_fused_mc2":1,"enable_flashcomm1":true}' \
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

    **Decode nodes**

    ```shell
    nic_name="xxxx" # change to the NIC that owns local_ip
    local_ip="xx.xx.xx.x" # use the IP of the current Decode node

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
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
    
    # Ensure the model path matches the directory recorded during download
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
        --served-model-name dsv4 \
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
        --attention_config.indexer_kv_dtype int8 \
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

    Start the two Prefill nodes and two Decode nodes. Replace the addresses with the Prefill and Decode master-node IPs respectively.

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

3. Deploy the P-D disaggregation proxy.

    Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

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
        
        # Ensure the model path matches the directory recorded during download
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
            --attention_config.indexer_kv_dtype int8 \
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
        
        # Ensure the model path matches the directory recorded during download
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
            --attention_config.indexer_kv_dtype int8 \
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

#### 5.2.3 Ascend 950DT Series PD Separation Deployment

This section shows a 2P1D deployment on 8 Ascend 950DT servers (96GB × 8) with the `DeepSeek-V4-Pro-0813` checkpoint. The Prefill side (2P) runs 2 Prefill instances, and each instance is a single logical engine with `DP1/TP1/PP2` plus `--prefill-context-parallel-size 8` spread over 2 servers (`--nnodes 2`): one PP stage runs on each server. The Decode side (1D) uses 4 servers as one Decode pool containing 32 DP ranks with the `DP32/TP1` configuration, and each Decode server starts 8 local DP ranks. Each Prefill instance declares `"prefill": {"dp_size": 1, "tp_size": 1, "pp_size": 2, "pp_layer_partition": "32,29"}` in `--kv-transfer-config`, while the Decode side declares `"decode": {"dp_size": 32, "tp_size": 1}`. Assign a unique `kv_port` and `engine_id` to every Prefill instance and to the Decode pool.

Before you start, make sure that `/etc/hixlep/` is mounted into the container as shown in Section 4.1. In every script below, set `nic_name` and `local_ip` to the NIC name and IP address of the current node before starting the service.

1. Prepare the script `launch_online_dp.py` on each Decode node. The Prefill engines are started directly by `vllm serve` on the Prefill servers, so this script is not required there.

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

2. Prepare the Prefill startup scripts. There are 2 Prefill instances, and each one is a PP2 engine spread across 2 servers: the master server (`--node-rank 0`) starts the engine and its API server, while the worker server (`--node-rank 1`) joins the same engine with `--headless`. `--master-addr` must point to the master server's service IP so that the worker can join the cross-node process group, and both servers must use identical model, TP, PP, PCP, and `VLLM_PP_LAYER_PARTITION` settings. `--port $2` takes the engine port from the second command-line argument of the script, so start the script with a placeholder first argument, for example `bash run_p_node0.sh 0 8000`.

    1. `run_p_node0.sh` on the master server of Prefill instance 1:

        ```bash
        #!/usr/bin/env bash
        unset https_proxy
        unset http_proxy
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

        nic_name="xxx"              # change to your NIC name
        local_ip="xx.xx.xx.x"       # change to your IP
        node0_ip="xx.xx.xx.x"       # service IP of this instance's master server

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_BUFFSIZE=1024

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_OP_EXPANSION_MODE=CCU_SCHED
        export TASK_QUEUE_ENABLE=1

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
        export HCCL_EXEC_TIMEOUT=2040
        export HCCL_CONNECT_TIMEOUT=1200

        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
        export VLLM_PP_LAYER_PARTITION="32,29"
        export VLLM_USE_FASTOKENS=1
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

        vllm serve /root/.cache/DeepSeek-V4-Pro-0813 \
            --host 0.0.0.0 \
            --served-model-name dsv4 \
            --port $2 \
            --nnodes 2 \
            --node-rank 0 \
            --master-addr $node0_ip \
            --tensor-parallel-size 1 \
            --prefill-context-parallel-size 8 \
            --data-parallel-size 1 \
            --pipeline-parallel-size 2 \
            --enable-expert-parallel \
            --seed 1024 \
            --max-model-len 1048576 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 16 \
            --no-disable-hybrid-kv-cache-manager \
            --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
            --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
            --trust-remote-code \
            --block-size 32 \
            --kv-cache-dtype fp8 \
            --attention_config.indexer_kv_dtype fp8 \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --gpu-memory-utilization 0.95 \
            --enforce-eager \
            --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/profiling", "torch_profiler_with_stack": false}' \
            --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":0, "multistream_overlap_shared_expert": true, "enable_shared_expert_dp":true}' \
            --kv-transfer-config '{
                "kv_connector":"MooncakeConnectorV2",
                "kv_role":"kv_producer",
                "kv_port":"30000",
                "engine_id":"0",
                "kv_connector_extra_config":{
                    "prefill":{"dp_size":1,"tp_size":1,"pp_size":2,"pp_layer_partition":"32,29"},
                    "decode":{"dp_size":32,"tp_size":1}
                }
            }'
        ```

    2. `run_p_node1.sh` on the worker server of Prefill instance 1: use the same content as Prefill node 0 with these changes: set `local_ip` to the IP of this node, keep `node0_ip` as the master server's service IP, change `--node-rank` to `1`, and append `--headless` to `vllm serve` so that this node runs only its PP stage without starting an API server.

    3. Prefill instance 2: prepare `run_p_node2.sh` (master server) and `run_p_node3.sh` (worker server) the same way, with these changes: set `local_ip` to the IP of each server, set `node0_ip` to this instance's own master server IP, and assign a unique `kv_port` and `engine_id`, for example `"kv_port":"30200"` and `"engine_id":"2"`. Keep all other settings identical so that both Prefill instances present the same `prefill` topology to the Decode pool.

3. Prepare `run_dp_template.sh` on each Decode node. `launch_online_dp.py` passes the visible devices, port, DP size, DP rank, DP address, DP RPC port, and TP size as `$1` through `$7`.

    ```bash
    #!/usr/bin/env bash
    unset https_proxy
    unset http_proxy
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

    nic_name="xxx"              # change to your NIC name
    local_ip="xx.xx.xx.x"       # change to your IP

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_BUFFSIZE=2048

    export ASCEND_RT_VISIBLE_DEVICES=$1
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=1

    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export HCCL_EXEC_TIMEOUT=2040
    export HCCL_CONNECT_TIMEOUT=1200

    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_USE_FASTOKENS=1
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

    vllm serve /root/.cache/DeepSeek-V4-Pro-0813 \
        --host 0.0.0.0 \
        --served-model-name dsv4 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 512 \
        --max-num-seqs 32 \
        --async-scheduling \
        --block-size 32 \
        --kv-cache-dtype fp8 \
        --attention_config.indexer_kv_dtype fp8 \
        --no-enable-prefix-caching \
        --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
        --trust-remote-code \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --gpu-memory-utilization 0.92 \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":false}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/profiling", "torch_profiler_with_stack": false}' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeConnectorV2",
            "kv_role":"kv_consumer",
            "kv_port":"30100",
            "engine_id":"1",
            "kv_connector_extra_config":{
                "prefill":{"dp_size":1,"tp_size":1,"pp_size":2,"pp_layer_partition":"32,29"},
                "decode":{"dp_size":32,"tp_size":1}
            }
        }' \
        --additional-config '{
            "ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "recompute_scheduler_enable":true,
            "enable_shared_expert_dp":true,
            "combine_quant_mode":4
        }'
    ```

4. Start the service. Set `local_ip` in every script to the IP of the current node before starting. For each Prefill instance, start the master server (`--node-rank 0`) before its worker server. The second argument is the engine port passed to `--port $2`.

    1. Prefill instance 1, master server:

        ```shell
        bash run_p_node0.sh 0 8000
        ```

    2. Prefill instance 1, worker server (headless):

        ```shell
        bash run_p_node1.sh 0 8000
        ```

    3. Prefill instance 2, master server:

        ```shell
        bash run_p_node2.sh 0 8000
        ```

    4. Prefill instance 2, worker server (headless):

        ```shell
        bash run_p_node3.sh 0 8000
        ```

    5. Decode node 0

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 --dp-address xx.xx.xx.5 --dp-rpc-port 12321 --vllm-start-port 8000
        ```

    6. Decode node 1

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 8 --dp-address xx.xx.xx.5 --dp-rpc-port 12321 --vllm-start-port 8000
        ```

    7. Decode node 2

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 16 --dp-address xx.xx.xx.5 --dp-rpc-port 12321 --vllm-start-port 8000
        ```

    8. Decode node 3

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 24 --dp-address xx.xx.xx.5 --dp-rpc-port 12321 --vllm-start-port 8000
        ```

5. Deploy the P-D disaggregation proxy.

    Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) and register all Prefill and Decode engine endpoints started above. Each Prefill instance exposes one API server on port `8000` of its master server, and each Decode node exposes ports `8000` through `8007`. Use the DP master address (`--dp-address` above, e.g. `xx.xx.xx.5`) as the IP of Decode node 0.

Key Parameter Descriptions:

- `--no-disable-hybrid-kv-cache-manager` keeps the hybrid KV cache manager enabled. DeepSeek-V4 KV Pool deployments require this flag; otherwise, the service may OOM during startup.
- `--pipeline-parallel-size 2` / `--nnodes 2` / `--node-rank` / `--prefill-context-parallel-size 8`: each Prefill instance is a single `DP1/TP1/PP2` engine distributed over 2 servers (`--nnodes 2`), with one PP stage per server; prefill context parallel further splits long-context computation across the 8 NPUs of each PP stage. The worker server of each instance (`--node-rank 1`) runs headless. `--master-addr` is the service IP of the instance's master server and is required for the cross-node process group.
- `VLLM_PP_LAYER_PARTITION="32,29"` partitions the 61 target layers into 32/29 across the 2 PP stages. The same partition must be declared in `kv_connector_extra_config.prefill.pp_layer_partition` so that the Decode side lays out KV cache consistently.
- `--enforce-eager` forces eager execution for the Prefill target model; graph compilation is only used on Decode (`FULL_DECODE_ONLY`).
- `--kv-cache-dtype fp8` and `--attention_config.indexer_kv_dtype fp8` use FP8 KV cache on both sides. The P and D sides must use the same KV data types so that the transferred KV cache is directly reusable.
- `--speculative-config`: Prefill and Decode must use the same DSpark `num_speculative_tokens` value (`5` here). The nested `enforce_eager` applies to DSpark draft execution: `true` on Prefill (eager) and `false` on Decode (graph mode).
- `MooncakeConnectorV2`: the KV transfer connector required by the KVPP PD disaggregation path, transferring KV cache between the prefill producer and decode consumer. `kv_connector_extra_config.prefill/decode` must match the actual global topology on both sides.
- Prefix caching: the Prefill side keeps prefix caching enabled (the default) and sets `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` to retain prefix-cache checkpoints every 4096 tokens. The Decode side uses `--no-enable-prefix-caching` because its KV cache is transferred from the Prefill side and decode-time prefix hits are not expected.
- `--async-scheduling` enables asynchronous scheduling on the Decode side to overlap CPU scheduling with NPU computation.
- `--model-loader-extra-config='{"enable_multithread_load":true,"num_threads":128}'` loads weights with 128 threads on both sides.
- `enable_fused_mc2: 0` disables the Fused MC2 fusion operator on Prefill; `multistream_overlap_shared_expert: true` overlaps shared-expert computation with MoE communication instead, and `enable_shared_expert_dp: true` places shared experts on DP ranks.
- `recompute_scheduler_enable: true`: enables the recomputation scheduler on Decode nodes. When the KV cache of a Decode node is insufficient, preempted requests return to the Prefill side for KV cache recomputation. Enable it only on PD Decode nodes; the legacy top-level key in `--additional-config` still works but `scheduler_config.recompute_scheduler_enable` is the preferred form.
- `ascend_compilation_config.enable_npugraph_ex: true` enables the extended NPU graph mode on Decode; `enable_static_kernel: false` keeps dynamic kernels, which is required for the speculative decode shapes used here.
- `combine_quant_mode: 4` forces the MoE MC2 combine stage on Decode nodes to use FP8 (MXFP float8_e4m3) communication quantization.
- `--profiler-config` (optional): writes torch profiler traces to `torch_profiler_dir` for performance analysis. Remove it in production if tracing is not needed.

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
| GPQA | - | accuracy | gen | 89.90 | 2 Atlas 800 A3 (128GB × 8) |
| GSM8K | - | accuracy | gen | 96.21 | 2 Atlas 800 A3 (128GB × 8) |
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
| Short Sequence / High Throughput | 1P1D deployment (2 Prefill + 2 Decode A3 nodes) | 64 (A3) | DeepSeek-V4-Pro-0813-w4a8 | Compared with the long-context configuration, use higher DP and concurrency: Prefill uses DP4/TP8 with 16 sequences, while Decode uses DP16/TP2 with 8 sequences. Enable FUSED_MC2 on Prefill to prioritize throughput. |
| Long Context (~1M Input, Reference) | 1P1D deployment (2 Prefill + 2 Decode A3 nodes) | 64 (A3) | DeepSeek-V4-Pro-0813-w4a8 | Compared with the short-sequence configuration, use DP2/TP16 on both sides, reduce both sides to 2 sequences, and reduce Decode batched tokens to 16 to reserve memory for a 1M-token context. |
| High Throughput | Multi-node mixed deployment (2 Ascend 950DT servers) | 16 (Ascend 950DT) | DeepSeek-V4-Pro-0813 | Use DP2/TP1 with prefill context parallel (PCP8) on 2 Ascend 950DT servers with the original Hugging Face weights. |
| Long Context (~1M Input) | 2P1D deployment (4 Prefill + 4 Decode Ascend 950DT servers; each Prefill instance spans 2 servers) | 64 (Ascend 950DT) | DeepSeek-V4-Pro-0813 | Run 2 Prefill instances, each a DP1/TP1/PP2 engine with prefill context parallel (PCP8) over 2 servers, and one Decode pool with DP32/TP1. |

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | NPUs | TP | DP | Max Num Seqs | Max Num Batched Tokens | Max Model Len | DSpark Speculation Num | FUSED_MC2 | EP Switch | FC / DSA-CP Switch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Short Sequence / High Throughput | Prefill Pool | 32 (2 A3 nodes) | 8 | 4 | 16 | 4096 | 150000 | 5 | On | On | On / On |
| Short Sequence / High Throughput | Decode Pool | 32 (2 A3 nodes) | 2 | 16 | 8 | 96 | 150000 | 5 | Off | On | Off / Off |
| Long Context (~1M Input, Reference) | Prefill Pool | 32 (2 A3 nodes) | 16 | 2 | 2 | 4096 | 1048576 | 5 | Off | On | On / Off |
| Long Context (~1M Input, Reference) | Decode Pool | 32 (2 A3 nodes) | 16 | 2 | 2 | 16 | 1048576 | 5 | Off | On | Off / Off |
| Multi-node mixed (Ascend 950DT) | Node0 / Node1 | 16 (2 servers) | 1 | 2 | 16 | 8192 | 1048576 | 5 | Off | Off | Off / Off |
| PD Separation (Ascend 950DT) | Prefill Pool | 32 (4 servers, 2 instances) | 1 | 1 | 16 | 8192 | 1048576 | 5 | Off | Off | Off / Off |
| PD Separation (Ascend 950DT) | Decode Pool | 32 (4 servers) | 1 | 32 | 32 | 512 | 1048576 | 5 | Off | Off | Off / Off |

> `FC / DSA-CP Switch` reports the FlashComm1 and DSA context-parallel switches respectively. For the A3/A2 long-context Prefill configurations, set `"enable_flashcomm1": true` and `"enable_dsa_cp": false` in `--additional-config`, and configure recomputation on Decode as `"scheduler_config": {"recompute_scheduler_enable": true}`. The role-specific `--kv-transfer-config` (connector, `kv_port`, `engine_id`, and `prefill`/`decode` topology) differs per example; use the exact values from Section 5.2.1 (A3), Section 5.2.2 (A2), or Section 5.2.3 (Ascend 950DT).
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
