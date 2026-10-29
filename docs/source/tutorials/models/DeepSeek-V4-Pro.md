# DeepSeek-V4-Pro

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

DeepSeek-V4 introduces several key upgrades over DeepSeek-V3:

- The Manifold-Constrained Hyper-Connections (mHC) to strengthen conventional residual connections.
- A hybrid attention architecture, which greatly improves long-context efficiency through Compress-4-Attention and Compress-128-Attention. For the Mixture-of-Experts (MoE) components, it still adopts the DeepSeekMoE architecture, with only minor adjustments.

DeepSeek-V4-Pro, the maximum reasoning effort mode of DeepSeek-V4, significantly advances the knowledge capabilities of open-source models, firmly establishing itself as the best open-source model available today. It achieves top-tier performance in coding benchmarks and significantly bridges the gap with leading closed-source models on reasoning and agentic tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, multi-node deployment, accuracy and performance evaluation.

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
| `DeepSeek-V4-Pro-0813-w4a4c8` | 950PR&950DT products | Quantize locally with [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_pro_w4a4c8.yaml) |
| `DeepSeek-V4-Pro-0813` | 950DT products | [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) |

- `DeepSeek-V4-Pro-0813-w4a8` (Official release with DSpark after quantization): This checkpoint includes the DSpark draft weights, so no separate draft-model path is required.

- `DeepSeek-V4-Pro-0813`: For Ascend 950DT products, use the original [`DeepSeek-V4-Pro-0813`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) weights released by DeepSeek on Hugging Face. The Attention weights use MXFP8, while the MoE weights use MXFP4 with 4-bit weights and 8-bit activation computation (W4A8). No Ascend-specific quantization or weight conversion is required. Both the mixed-deployment example in [Section 5.1](#51-multi-node-online-deployment) and the 2P1D example in [Section 5.2](#52-multi-node-pd-separation-deployment) use Ascend 950DT products (96GB × 8) with this checkpoint.

- `DeepSeek-V4-Pro-0813-w4a4c8` (W4A4 MXFP4 mixed-precision quantization): For how the ultra-low-bit quantization preserves accuracy, see the [W4A4C8 FAQ](#q-how-do-the-w4a4c8-ultra-low-bit-quantized-weights-preserve-accuracy).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy a multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image-selection).

=== "950DT products"

    Start the docker image on each node.

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
        -v /root/.cache:/root/.cache \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
        -itd $IMAGE bash
    ```

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

    The A2/A3 examples assume that the model weights are stored in `/root/.cache/modelscope/hub/models/vllm-ascend/`. The examples for 950DT products use the original `DeepSeek-V4-Pro-0813` Hugging Face weights at `/root/.cache/DeepSeek-V4-Pro-0813`. Feel free to change these paths to match your environment.

    It is recommended that the following service code be encapsulated in a .sh script file and executed in Bash mode.

### 5.1 Multi-Node Online Deployment

The quantized model `DeepSeek-V4-Pro-w4a8-mtp` requires at least 2 Atlas 800I A3 (128GB × 8) nodes or 4 Atlas 800I A2 (64GB × 8) nodes. The Ascend 950DT mixed-deployment example uses 2 nodes (96GB × 8) with the original Hugging Face weights. Run the following scripts on each node respectively.

=== "950DT products with DSpark"

    Use the original `DeepSeek-V4-Pro-0813` checkpoint on both nodes. Set `node0_ip` to the service IP of Node0 on both nodes. Each node hosts one DP rank whose 8 NPUs run prefill context parallel (PCP8), so the engine uses `DP2/TP1` with `--prefill-context-parallel-size 8`.

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
    export VLLM_USE_FASTOKENS=1

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
        --max-num-batched-tokens 2048 \
        --max-num-seqs 16 \
        --async-scheduling \
        --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
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
    # nic_name is the network interface that owns local_ip on the current node.
    nic_name="xxx"
    local_ip="xxx"

    # node0_ip must be the local_ip of Node0 on every node.
    node0_ip="xxxx"

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
    export VLLM_USE_FASTOKENS=1

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
        --max-num-batched-tokens 2048 \
        --max-num-seqs 16 \
        --async-scheduling \
        --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
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
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_EXEC_TIMEOUT=2040
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export TASK_QUEUE_ENABLE=1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

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
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_EXEC_TIMEOUT=2040
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export TASK_QUEUE_ENABLE=1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

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

=== "A2 series"

    Use the `DeepSeek-V4-Pro-w4a8-mtp` checkpoint on all four nodes. The example keeps the existing `DP4/TP8` topology and changes only the MTP-related and required runtime parameters.

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
    # nic_name is the network interface that owns local_ip on the current node.
    nic_name="xxx"
    local_ip="xxx"

    # node0_ip must be the local_ip of Node0 on every node.
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
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

Key Parameter Descriptions:

- `--data-parallel-size` sets the global number of data parallel ranks, and `--data-parallel-size-local` sets the number of DP ranks on the current node.
- `--data-parallel-start-rank` specifies the starting data parallel rank of the current node. Set a unique value on each headless node (e.g., Node1 = 1). On Node0, omit `--data-parallel-start-rank` even when the start rank is 0, to avoid enabling hybrid load balancing with headless remote engines.
- `--data-parallel-address` specifies the IP address of the data parallel master node (Node0). It must be consistent across all nodes.
- `--data-parallel-rpc-port` is the DP RPC port. Use the same value on all nodes and ensure the port is available.
- `--tensor-parallel-size` sets the tensor parallel size within each DP rank. Configure it together with the DP sizes according to the deployment topology and available NPUs.
- `--prefill-context-parallel-size` configures prefill context parallelism (PCP) on Model Runner V2 (MRv2), which spreads a long prompt across multiple NPUs during prefill. Besides speeding up prefill, enabling PCP also shards the o_proj weights across the PCP group, providing partial tensor parallelism even when `--tensor-parallel-size` is `1`. The 950DT DP mixed-deployment uses `DP2/TP1` with PCP `8`, and the o_proj weight sharding also relieves its memory pressure. Do not combine it with `enable_dsa_cp`, which is mutually exclusive with PCP and unsupported by the V2 model runner.
- `--enable-expert-parallel` enables expert parallelism for MoE layers. Do not mix MoE tensor parallelism and expert parallelism in the same MoE layer.
- `--headless` (used on non-master nodes) disables the API server on the node, since only the master node serves requests.
- `--max-model-len` specifies the maximum context length. The example values are intended to allow startup with the documented hardware and configuration. To tune this value, check the startup log `Maximum concurrency for L tokens per request: Cx`: `L × C` provides a rough KV-cache capacity reference for increasing the context length with the same configuration. Use the smallest estimate across all ranks and, for PD-disaggregated deployment, across both Prefill and Decode pools.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` is the maximum number of tokens processed in one scheduler step. A larger value can improve prefill efficiency but consumes more activation memory.
- `--tokenizer-mode deepseek_v4`, `--tool-call-parser deepseek_v4`, `--enable-auto-tool-choice`, and `--reasoning-parser deepseek_v4` enable the DeepSeek-V4 tokenizer behavior, automatic tool calling, and reasoning-output parsing.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- `--block-size` sets the KV cache block size. To enable the experimental 4k prefix cache hit support, change it from `128` to `32`.
- `--quantization ascend` enables Ascend quantization for the W4A8 model used in the A2 and A3 configurations. The original DeepSeek-V4-Pro-0813 weights used on 950DT products do not require this option.
- `--model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}'` selects the multi-thread weight iterator. `enable_multithread_load` must be a JSON boolean and `num_threads` must be a positive integer.
- `--safetensors-load-strategy prefetch` is an alternative that warms checkpoint files into the OS page cache before the normal iterator loads them. Do not combine it with multi-thread loading: the default loader rejects `prefetch`, `eager`, or `torchao` when `enable_multithread_load` is `true`. The examples in this document use only multi-thread loading.
- `--speculative-config` configures speculative decoding. Use `mtp` for the preview MTP checkpoint and `dspark` for the `DeepSeek-V4-Pro-0813` checkpoint. For DSpark, use the value declared by the checkpoint; the example uses 5 speculative tokens, and all ranks must use the same value.
- `--kv-cache-dtype` and `--attention_config.indexer_kv_dtype` set the KV cache data type. The 950DT example uses FP8 KV cache with both options set to `fp8`.
- `--no-disable-hybrid-kv-cache-manager` keeps the hybrid KV cache manager enabled. DeepSeek-V4 deployments require this flag; otherwise, the service may OOM during startup.
- `--async-scheduling` enables asynchronous scheduling to overlap scheduler overhead with model execution.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full ACL graph execution in the decode phase.
- `--additional-config` enables Ascend-specific optimizations. `enable_npugraph_ex` enables enhanced ACL graph execution, `enable_static_kernel: false` keeps static-kernel compilation disabled, `enable_cpu_binding` enables Ascend-native CPU binding, `enable_shared_expert_dp` enables data parallelism for shared experts, and `multistream_overlap_shared_expert` overlaps shared expert computation for better MoE throughput. `enable_fused_mc2: 0` disables the Fused MC2 dispatch/combine path.

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

The following tabs describe PD separation deployment on Ascend 950DT nodes (96GB × 8), Atlas 800I A3 nodes (128GB × 8), and Atlas 800I A2 nodes (64GB × 8).

Select the tab that matches the deployment hardware.

=== "950DT products"

    #### 950DT products {: #dsv4pro-5-2-950dt }

    This section shows a 2P1D deployment on 8 Ascend 950DT nodes (96GB × 8) with the `DeepSeek-V4-Pro-0813` checkpoint. The Prefill side (2P) runs 2 Prefill instances, and each instance is a single logical engine with `DP1/TP1/PP2` plus `--prefill-context-parallel-size 8` spread over 2 nodes (`--nnodes 2`): one PP stage runs on each node. The Decode side (1D) uses 4 nodes as one Decode pool containing 32 DP ranks with the `DP32/TP1` configuration, and each Decode node starts 8 local DP ranks. Each Prefill instance declares `"prefill": {"dp_size": 1, "tp_size": 1, "pp_size": 2, "pp_layer_partition": "32,29"}` in `--kv-transfer-config`, while the Decode side declares `"decode": {"dp_size": 32, "tp_size": 1}`. Assign a unique `kv_port` to every Prefill instance and to the Decode pool.

    Before you start, set `nic_name` and `local_ip` in every script below to the NIC name and IP address of the current node before starting the service.

    1. Prepare the script `launch_online_dp.py` on each Decode node. The Prefill engines are started directly by `vllm serve` on the Prefill nodes, so this script is not required there. The launcher is shared by all hardware tabs and is provided in [Common Script and Parameter Descriptions](#dsv4pro-5-2-common).

    2. Prepare the Prefill startup scripts. There are 2 Prefill instances, and each one is a PP2 engine spread across 2 nodes: the master node (`--node-rank 0`) starts the engine and its API server, while the worker node (`--node-rank 1`) joins the same engine with `--headless`. `--master-addr` must point to the master node's service IP so that the worker can join the cross-node process group, and both nodes must use identical model, TP, PP, PCP, and `VLLM_PP_LAYER_PARTITION` settings. `--port $1` takes the engine port from the first command-line argument of the script, for example `bash run_p_node0.sh 8000`.

        1. `run_p_node0.sh` on the master node of Prefill instance 1:

            ```bash
            #!/usr/bin/env bash
            # nic_name is the network interface that owns local_ip on the current node.
            nic_name="xxx"
            local_ip="xxx"

            # node0_ip must be the local_ip of the master node of this instance.
            node0_ip="xxxx"

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
            export VLLM_PP_LAYER_PARTITION="32,29"
            export VLLM_USE_FASTOKENS=1
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

            vllm serve /root/.cache/DeepSeek-V4-Pro-0813 \
                --host 0.0.0.0 \
                --served-model-name dsv4 \
                --port $1 \
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
                --gpu-memory-utilization 0.94 \
                --enforce-eager \
                --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":0,"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true}' \
                --kv-transfer-config '{
                    "kv_connector":"MooncakeConnectorV2",
                    "kv_role":"kv_producer",
                    "kv_port":"30000",
                    "kv_connector_extra_config":{
                        "prefill":{"dp_size":1,"tp_size":1,"pp_size":2,"pp_layer_partition":"32,29"},
                        "decode":{"dp_size":32,"tp_size":1}
                    }
                }'
            ```

        2. `run_p_node1.sh` on the worker node of Prefill instance 1: use the same content as Prefill node 0 with these changes: set `local_ip` to the IP of this node, keep `node0_ip` as the master node's service IP, change `--node-rank` to `1`, and append `--headless` to `vllm serve` so that this node runs only its PP stage without starting an API server.

        3. Prefill instance 2: prepare `run_p_node2.sh` (master node) and `run_p_node3.sh` (worker node) the same way, with these changes: set `local_ip` to the IP of each node, set `node0_ip` to this instance's own master node IP, and assign a unique `kv_port`, for example `"kv_port":"30200"`. Keep all other settings identical so that both Prefill instances present the same `prefill` topology to the Decode pool.

    3. Prepare `run_dp_template.sh` on each Decode node. `launch_online_dp.py` passes the visible devices, port, DP size, DP rank, DP address, DP RPC port, TP size, and PCP size as `$1` through `$8`; this template ignores `$8` because Decode engines do not use PCP.

        ```bash
        #!/usr/bin/env bash
        # nic_name is the network interface that owns local_ip on the current node.
        nic_name="xxx"
        local_ip="xxx"

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
            --kv-transfer-config '{
                "kv_connector":"MooncakeConnectorV2",
                "kv_role":"kv_consumer",
                "kv_port":"30100",
                "kv_connector_extra_config":{
                    "prefill":{"dp_size":1,"tp_size":1,"pp_size":2,"pp_layer_partition":"32,29"},
                    "decode":{"dp_size":32,"tp_size":1}
                }
            }' \
            --additional-config '{
                "ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},
                "enable_cpu_binding":true,
                "multistream_overlap_shared_expert":true,
                "scheduler_config":{"recompute_scheduler_enable":true},
                "enable_shared_expert_dp":true,
                "combine_quant_mode":4
            }'
        ```

    4. Start the service. Set `local_ip` in every script to the IP of the current node before starting. For each Prefill instance, start the master node (`--node-rank 0`) before its worker node. The first argument is the engine port passed to `--port $1`.

        1. Prefill instance 1, master node:

            ```shell
            bash run_p_node0.sh 8000
            ```

        2. Prefill instance 1, worker node (headless):

            ```shell
            bash run_p_node1.sh 8000
            ```

        3. Prefill instance 2, master node:

            ```shell
            bash run_p_node2.sh 8000
            ```

        4. Prefill instance 2, worker node (headless):

            ```shell
            bash run_p_node3.sh 8000
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

        Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) and register all Prefill and Decode engine endpoints started above. Each Prefill instance exposed one API server on port `8000` of its master node, and each Decode node exposes ports `8000` through `8007`. Use the DP master address (`--dp-address` above, e.g. `xx.xx.xx.5`) as the IP of Decode node 0.

    Key Parameter Descriptions:

    - `--kv-cache-dtype fp8` sets the KV cache dtype to FP8, and `--attention_config.indexer_kv_dtype fp8` sets the index-side KV dtype to FP8. The P and D sides must use the same KV data types so that the transferred KV cache is directly reusable.
    - `--pipeline-parallel-size 2` / `--nnodes 2` / `--node-rank` / `--master-addr` distribute each Prefill engine (`DP1/TP1/PP2`) over 2 nodes with one PP stage per node; the worker node (`--node-rank 1`) runs headless, and `--master-addr` is the service IP of the instance's master node for the cross-node process group.
    - `VLLM_PP_LAYER_PARTITION="32,29"` partitions the 61 target layers across the 2 PP stages; the same partition must be declared in `kv_connector_extra_config.prefill.pp_layer_partition`.
    - `--prefill-context-parallel-size 8` spreads the prompt across the 8 NPUs of each PP stage.
    - `--kv-transfer-config` uses `MooncakeConnectorV2` for KV transfer; it is required because pipeline parallelism is enabled.
    - `--additional-config`: `enable_fused_mc2: 0` disables Fused MC2 and instead overlaps shared-expert computation with MoE communication via `multistream_overlap_shared_expert: true`, with `enable_shared_expert_dp: true` placing shared experts on DP ranks; `combine_quant_mode: 4` forces the MoE MC2 combine stage on Decode nodes to use FP8 (MXFP float8_e4m3) communication quantization.

=== "A3 series"

    #### A3 series {: #dsv4pro-5-2-a3 }

    This section shows a deployment with one Prefill pool and one Decode pool on Atlas 800I A3 (128GB × 8). The example uses two physical nodes for each pool.

    Use the official `DeepSeek-V4-Pro-0813-w4a8` checkpoint on all nodes. It includes the DSpark draft weights, so no separate draft-model path is required. Replace the model path, IP addresses, and network interface names below with your environment's values.

    Before you start, please:

    1. Prepare `launch_online_dp.py` on each Decode node. The launcher is shared by all hardware tabs and is provided in [Common Script and Parameter Descriptions](#dsv4pro-5-2-common).

    2. Prepare the Prefill startup scripts. There is one Prefill instance, a PP2 engine spread across 2 nodes: the master node (`--node-rank 0`) starts the engine and its API server, while the worker node (`--node-rank 1`) joins the same engine with `--headless`. `--master-addr` must point to the master node's service IP so that the worker can join the cross-node process group, and both nodes must use identical model, TP, PP, PCP, and `VLLM_PP_LAYER_PARTITION` settings. `--port $1` takes the engine port from the first command-line argument of the script, for example `bash run_p_node0.sh 8000`.

        1. `run_p_node0.sh` on the master node:

            ```bash
            #!/usr/bin/env bash
            # nic_name is the network interface that owns local_ip on the current node.
            nic_name="xxx"
            local_ip="xxx"

            # node0_ip must be the local_ip of the master node of this instance.
            node0_ip="xxxx"

            export HCCL_IF_IP="$local_ip"
            export GLOO_SOCKET_IFNAME="$nic_name"
            export TP_SOCKET_IFNAME="$nic_name"
            export HCCL_SOCKET_IFNAME="$nic_name"
            export VLLM_USE_V2_MODEL_RUNNER=1
            export HCCL_OP_EXPANSION_MODE="AIV"
            export HCCL_BUFFSIZE=1024
            export HCCL_CONNECT_TIMEOUT=1200
            export HCCL_EXEC_TIMEOUT=2040
            export VLLM_RPC_TIMEOUT=3600000
            export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export VLLM_PP_LAYER_PARTITION="32,29"

            vllm serve /path/to/DeepSeek-V4-Pro-0813-w4a8 \
                --tensor-parallel-size 2 \
                --prefill-context-parallel-size 8 \
                --host 0.0.0.0 \
                --port $1 \
                --nnodes 2 \
                --node-rank 0 \
                --master-addr $node0_ip \
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
                  "kv_port": 30000,
                  "kv_connector_extra_config": {
                    "prefill": {"dp_size": 1, "tp_size": 2, "pp_size": 2, "pcp_size": 8},
                    "decode": {"dp_size": 16, "tp_size": 2, "pp_size": 1, "pcp_size": 1}
                  }
                }'
            ```

        2. `run_p_node1.sh` on the worker node: use the same content as `run_p_node0.sh` with these changes: set `local_ip` to the IP of this node, keep `node0_ip` as the master node's service IP, change `--node-rank` to `1`, append `--headless` to `vllm serve` so that this node runs only its PP stage without starting an API server, and assign a unique `kv_port`, for example `"kv_port":"30200"`.

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
        export HCCL_EXEC_TIMEOUT=2040
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
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
              "kv_port": 30100,
              "kv_connector_extra_config": {
                "prefill": {"dp_size": 1, "tp_size": 2, "pp_size": 2, "pcp_size": 8},
                "decode": {"dp_size": 16, "tp_size": 2, "pp_size": 1, "pcp_size": 1}
              }
            }'
        ```

    4. Start the service in separate terminals on the corresponding machines. Start the Prefill master node before its worker node; the first argument is the engine port passed to `--port $1`.

        1. Prefill master node

            ```shell
            bash run_p_node0.sh 8000
            ```

        2. Prefill worker node: headless

            ```shell
            bash run_p_node1.sh 8000
            ```

        3. Decode node 0

            ```shell
            python launch_online_dp.py --dp-size 16 --tp-size 2 --pcp-size 1 --dp-size-local 8 --dp-rank-start 0 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

        4. Decode node 1

            ```shell
            python launch_online_dp.py --dp-size 16 --tp-size 2 --pcp-size 1 --dp-size-local 8 --dp-rank-start 8 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

    5. Deploy the P-D disaggregation proxy.

        Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

    Key Parameter Descriptions:

    - `--kv-cache-dtype` keeps the default BF16 KV dtype, and `--attention_config.indexer_kv_dtype int8` sets the index-side KV dtype to INT8. The P and D sides must use the same KV data types so that the transferred KV cache is directly reusable.
    - `--pipeline-parallel-size 2` / `--nnodes 2` / `--node-rank` / `--master-addr` distribute the Prefill engine (`DP1/TP2/PP2`) over 2 nodes with one PP stage per node; the worker node (`--node-rank 1`) runs headless.
    - `--prefill-context-parallel-size 8` on Prefill; Decode passes `--pcp-size 1` to the launcher, keeping PCP disabled on the Decode pool. Keep `kv_connector_extra_config.prefill.pcp_size` and `decode.pcp_size` consistent with the PCP size of the corresponding pool.
    - `--kv-transfer-config` uses `MooncakeConnectorV2` for KV transfer; it is required because pipeline parallelism is enabled.
    - `--additional-config`: `enable_flashcomm1: true` and `enable_fused_mc2: 1` enable the Fused MC2 fusion operator with FlashComm to accelerate communication on Prefill nodes.

=== "A2 series"

    #### A2 series {: #dsv4pro-5-2-a2 }

    This section shows a deployment with one Prefill pool and one Decode pool on Atlas 800I A2 (64GB × 8). The example uses four physical nodes for each pool.

    Before you start, please:

    1. Prepare the script `launch_online_dp.py` on each node. The launcher is shared by all hardware tabs and is provided in [Common Script and Parameter Descriptions](#dsv4pro-5-2-common).

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
            export VLLM_USE_V2_MODEL_RUNNER=1


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
            python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

        2. Prefill node 1

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

        3. Prefill node 2

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 2 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

        4. Prefill node 3

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 3 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

        5. Decode node 0

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

        6. Decode node 1

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

        7. Decode node 2

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 2 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

        8. Decode node 3

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 3 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

    4. Deploy the P-D disaggregation proxy.

        Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

    Key Parameter Descriptions:

    - `--kv-cache-dtype` keeps the default BF16 KV dtype, and `--attention_config.indexer_kv_dtype int8` sets the index-side KV dtype to INT8. The P and D sides must use the same KV data types so that the transferred KV cache is directly reusable.
    - `--kv-transfer-config` uses `MooncakeHybridConnector` for KV transfer because pipeline parallelism is not enabled.
    - `--additional-config`: `enable_flashcomm1: true` and `enable_fused_mc2: 1` enable the Fused MC2 fusion operator with FlashComm; the Prefill nodes additionally set `enable_shared_expert_dp: true`.
    - No `--prefill-context-parallel-size`: PCP is not used because each DP rank already runs full `TP8`.

**Common key parameter descriptions**

- `--no-disable-hybrid-kv-cache-manager` keeps the hybrid KV cache manager enabled. DeepSeek-V4 KV Pool deployments require this flag; otherwise, the service may OOM during startup.
- `--enforce-eager` forces eager execution on prefill nodes instead of graph compilation.
- `--trust-remote-code` allows the model repository's custom code to be loaded. Only use trusted model repositories.
- `--kv-transfer-config` configures KV cache transfer between the prefill producer and decode consumer in PD separation.
- `kv_connector_extra_config.prefill.dp_size/tp_size` and `decode.dp_size/tp_size` must match the actual parallel layout on the prefill and decode sides.
- `scheduler_config.recompute_scheduler_enable: true`: enables the recomputation scheduler. When the KV Cache of the Decode node is insufficient, requests are sent to Prefill for KV Cache recomputation. Enable it only on PD Decode nodes; the legacy top-level key is deprecated.
- `--speculative-config`: Prefill and Decode must use the same DSpark `num_speculative_tokens` value. Use the value declared by the checkpoint; the 950DT and A3 examples use `5`, while the A2 example uses `7`. The nested `enforce_eager` applies to DSpark draft execution, while the top-level `--enforce-eager` keeps the Prefill target model in eager mode.
- `ascend_compilation_config.enable_npugraph_ex: true` enables the extended NPU graph mode on Decode for graph compilation (Prefill runs eager mode instead); `enable_static_kernel: false` keeps dynamic-shape kernels instead of static-shape kernels for aclgraph execution.
- `--prefill-context-parallel-size` configures prefill context parallelism (PCP) on Model Runner V2 (MRv2), replacing the DSA-CP configuration used on MRv1. Besides speeding up long-context prefill, enabling PCP also shards the o_proj weights across the PCP group, providing partial tensor parallelism even when `--tensor-parallel-size` is `1` and relieving memory pressure. Each hardware tab configures PCP differently; see the tab-specific descriptions.
- `--quantization ascend` enables Ascend quantization for the W4A8 model used in the A2 and A3 configurations. The original DeepSeek-V4-Pro-0813 weights used on 950DT products do not require this option.

The hardware-specific parameter differences are listed inside each tab above.

<a id="dsv4pro-5-2-common"></a>

**Common Script and Parameter Descriptions**

The three hardware tabs above share the same `launch_online_dp.py` launcher, shown below. Every local DP instance occupies `TP × PCP` NPUs; when `--pcp-size` is not specified, PCP defaults to `1`.

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
        required=True,
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
        default=8000,
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

Parameter descriptions of `launch_online_dp.py`:

|Parameter|Type|Required|Default|Description|
|---------|----|--------|-------|-----------|
|`--dp-size`|int|Yes|-|Data parallel size (total number of DP ranks across all nodes).|
|`--tp-size`|int|No|1|Tensor parallel size within each DP rank.|
|`--pcp-size`|int|No|1|Prefill context parallel size within each DP rank; passed as `$8` / `PCP_SIZE` to `run_dp_template.sh`. Each local DP instance uses `TP × PCP` devices.|
|`--dp-size-local`|int|No|(same as `--dp-size`)|Number of DP ranks on the current node. If not set, defaults to `--dp-size`.|
|`--dp-rank-start`|int|Yes|-|Starting rank offset for data parallel ranks on this node.|
|`--dp-address`|str|Yes|-|IP address of the data parallel master node.|
|`--dp-rpc-port`|str|No|12321|RPC port for data parallel master communication.|
|`--vllm-start-port`|int|No|8000|Starting port for each vLLM engine instance on this node.|

Deployment Verification:

After the PD separation service is fully started and every engine reports `Application startup complete`, send a request through the proxy port on the prefill master node to verify that Prefill and Decode nodes are working correctly together. Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) for the proxy verification method.

Common Issues Tip: If you encounter issues with PD separation deployment, please refer to the [Public FAQs](../../faqs.md) for troubleshooting.

### 5.3 KV Cache Pool Deployment

This section provides PD separation deployments that add a KV Cache Pool for
cross-engine KV reuse. Each hardware series uses the pool backend that suits its
interconnect: the A3 series and 950DT products use the MemCache backend, and the
A2 series uses the Mooncake backend. Topology, connector roles, and pool settings
are explained in each hardware tab; apply only the settings for the selected platform.

Select the tab that matches the deployment hardware.

=== "950DT products"

    #### 950DT products {: #dsv4pro-5-3-950dt }

    This section extends the 2P1D deployment of [Section 5.2](#dsv4pro-5-2-950dt) with a Memcache KV Cache Pool on the same 8 Ascend 950DT nodes (96GB × 8) and the `DeepSeek-V4-Pro-0813` checkpoint. `MooncakeConnectorV2` still transfers KV cache from Prefill to Decode, while `AscendStoreConnector` stores reusable KV cache in Memcache so that requests sharing a prefix can skip Prefill recomputation. Both connectors are configured as children of `MultiConnector`. Prefill enables layerwise Memcache access with `use_layerwise: true` and `layerwise_prefetch_layers: 3`; Decode uses the default non-layerwise mode.

    This example uses the embedded LocalService mode: every vLLM process loads `mmc-local.conf` from `MMC_LOCAL_CONFIG_PATH` and joins the pool itself, so no standalone Memcache process is required. Before starting vLLM, install `memfabric-hybrid` and `memcache-hybrid` and prepare the configuration files below. For details, refer to the [Memcache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend). All Prefill and Decode nodes must use the same MetaService endpoint.

    ##### Prepare the Memcache Services

    Create `mmc-meta.conf` on the node that runs MetaService:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
    ock.mmc.meta_service.config_store_url = tcp://<META_SERVICE_IP>:6000
    ock.mmc.meta_service.metrics_url = http://<META_SERVICE_IP>:8000
    ock.mmc.log_level = info
    ```

    Create `mmc-local.conf` on all Prefill and Decode nodes for the LocalService embedded in vLLM:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
    ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:6000
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_urma
    ock.mmc.local_service.dram.size = 40GB
    ock.mmc.local_service.max.dram.size = 2048GB
    ```

    Use `device_urma` for 950DT products Device UB; for UBOE, replace it with `device_uboe`. `world_size` is the maximum number of LocalServices supported by the pool, not the number of NPUs used by this example; changing it after LocalServices connect requires restarting MetaService. `max.dram.size` must be at least the largest `dram.size` contributed by any LocalService. Because this example uses the embedded LocalService mode, `dram.size` must be greater than `0GB`, and no standalone LocalService process is started.

    Start MetaService with `mmc-meta.conf`:

    ```shell
    export MMC_META_CONFIG_PATH="<MEMCACHE_CONFIG_DIR>/mmc-meta.conf"
    python -c "from memcache_hybrid import MetaService; MetaService.main()"
    ```

    The startup scripts below export `PYTHONHASHSEED=0` and `MMC_LOCAL_CONFIG_PATH`. `PYTHONHASHSEED=0` must match across all engines. `MMC_LOCAL_CONFIG_PATH` points to the absolute path of `mmc-local.conf`, for example `/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config/mmc-local.conf`; change it to the actual path in your environment.

    ##### Prepare the Prefill Startup Scripts

    Reuse `launch_online_dp.py` from [Section 5.2](#dsv4pro-5-2-950dt) on the Decode nodes. The Prefill engines are still started directly by `vllm serve`, with the environment variables and `MultiConnector` configuration required by the pool.

    1. `run_p_node0.sh` on the master node of Prefill instance 1:

        ```bash
        #!/usr/bin/env bash
        # nic_name is the network interface that owns local_ip on the current node.
        nic_name="xxx"
        local_ip="xxx"

        # node0_ip must be the local_ip of the master node of this instance.
        node0_ip="xxxx"

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
        export VLLM_PP_LAYER_PARTITION="32,29"
        export VLLM_USE_FASTOKENS=1
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

        # Parameters required by the Memcache KV Cache Pool. Change the config
        # path to the actual path of mmc-local.conf in your environment.
        export PYTHONHASHSEED=0
        export MMC_LOCAL_CONFIG_PATH=/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config/mmc-local.conf

        vllm serve /root/.cache/DeepSeek-V4-Pro-0813 \
            --host 0.0.0.0 \
            --served-model-name dsv4 \
            --port $1 \
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
            --enable-prefix-caching \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --gpu-memory-utilization 0.94 \
            --enforce-eager \
            --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":0,"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true}' \
            --kv-transfer-config '{
                "kv_connector":"MultiConnector",
                "kv_role":"kv_producer",
                "kv_port":"30000",
                "engine_id":"0",
                "kv_connector_extra_config":{
                    "connectors":[
                        {
                            "kv_connector":"MooncakeConnectorV2",
                            "kv_role":"kv_producer",
                            "kv_port":"30000",
                            "kv_connector_extra_config":{
                                "prefill":{"dp_size":1,"tp_size":1,"pp_size":2,"pp_layer_partition":"32,29"},
                                "decode":{"dp_size":32,"tp_size":1}
                            }
                        },
                        {
                            "kv_connector":"AscendStoreConnector",
                            "kv_role":"kv_producer",
                            "kv_connector_extra_config":{
                                "backend":"memcache",
                                "use_layerwise":true,
                                "layerwise_prefetch_layers":3
                            }
                        }
                    ]
                }
            }'
        ```

    2. `run_p_node1.sh` on the worker node of Prefill instance 1: use the same content as `run_p_node0.sh` with these changes: set `local_ip` to the IP of this node, keep `node0_ip` as the master node's service IP, change `--node-rank` to `1`, and append `--headless` to `vllm serve` so that this node runs only its PP stage without starting an API server.

    3. Prefill instance 2: prepare `run_p_node2.sh` (master node) and `run_p_node3.sh` (worker node) the same way, with these changes: set `local_ip` to the IP of each node, set `node0_ip` to this instance's own master node IP, and assign a unique `kv_port` to the outer `MultiConnector` and its `MooncakeConnectorV2` child, for example `"kv_port":"30200"`. Keep all other settings identical so that both Prefill instances present the same `prefill` topology to the Decode pool.

    ##### Prepare the Decode Startup Script

    Prepare `run_dp_template.sh` on each Decode node. Compared with [Section 5.2](#dsv4pro-5-2-950dt), only the Memcache environment variables and `--kv-transfer-config` change; `launch_online_dp.py` still passes the visible devices, port, DP size, DP rank, DP address, DP RPC port, TP size, and PCP size as `$1` through `$8`.

    ```bash
    #!/usr/bin/env bash
    # nic_name is the network interface that owns local_ip on the current node.
    nic_name="xxx"
    local_ip="xxx"

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

    # Parameters required by the Memcache KV Cache Pool. Change the config
    # path to the actual path of mmc-local.conf in your environment.
    export PYTHONHASHSEED=0
    export MMC_LOCAL_CONFIG_PATH=/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config/mmc-local.conf

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
        --kv-transfer-config '{
            "kv_connector":"MultiConnector",
            "kv_role":"kv_consumer",
            "kv_port":"30100",
            "engine_id":"1",
            "kv_connector_extra_config":{
                "connectors":[
                    {
                        "kv_connector":"MooncakeConnectorV2",
                        "kv_role":"kv_consumer",
                        "kv_port":"30100",
                        "kv_connector_extra_config":{
                            "prefill":{"dp_size":1,"tp_size":1,"pp_size":2,"pp_layer_partition":"32,29"},
                            "decode":{"dp_size":32,"tp_size":1}
                        }
                    },
                    {
                        "kv_connector":"AscendStoreConnector",
                        "kv_role":"kv_consumer",
                        "kv_connector_extra_config":{
                            "backend":"memcache"
                        }
                    }
                ]
            }
        }' \
        --additional-config '{
            "ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "scheduler_config":{"recompute_scheduler_enable":true},
            "enable_shared_expert_dp":true,
            "combine_quant_mode":4
        }'
    ```

    ##### Startup and Verification

    Start the Memcache MetaService first, because every embedded LocalService connects to it during engine startup. Prefill and Decode have no required startup order relative to each other; wait until every engine reports `Application startup complete` before deploying the proxy.

    1. Start MetaService on the node containing `mmc-meta.conf`. Wait until its logs show that the Meta Service has launched.

    2. Prefill nodes. For each Prefill instance, start the master node (`--node-rank 0`) before its worker node. The first argument is the engine port passed to `--port $1`.

        1. Prefill instance 1, master node:

            ```shell
            bash run_p_node0.sh 8000
            ```

        2. Prefill instance 1, worker node (headless):

            ```shell
            bash run_p_node1.sh 8000
            ```

        3. Prefill instance 2, master node:

            ```shell
            bash run_p_node2.sh 8000
            ```

        4. Prefill instance 2, worker node (headless):

            ```shell
            bash run_p_node3.sh 8000
            ```

    3. Decode nodes. Run each command from the directory containing `launch_online_dp.py` and `run_dp_template.sh`:

        1. Decode node 0

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 --dp-address xx.xx.xx.5 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

        2. Decode node 1

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 8 --dp-address xx.xx.xx.5 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

        3. Decode node 2

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 16 --dp-address xx.xx.xx.5 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

        4. Decode node 3

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 24 --dp-address xx.xx.xx.5 --dp-rpc-port 12321 --vllm-start-port 8000
            ```

    4. Deploy the P-D disaggregation proxy. Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) and register all Prefill and Decode engine endpoints started above, the same way as [Section 5.2](#dsv4pro-5-2-950dt).

    Deployment Verification:

    After the PD separation service is fully started, send a request through the proxy to verify that Prefill and Decode nodes are working correctly together. Then send the same request again to verify the KV Cache Pool: check the Prefill logs for Memcache lookup, get, put, and cache-hit activity, and confirm that Decode logs still show normal Prefill-to-Decode KV transfer. An HTTP 200 response alone does not confirm pool reuse. For details, refer to the [KV Cache Pool guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend).

=== "A3 series"

    #### A3 series {: #dsv4pro-5-3-a3 }

    This configuration extends the Atlas 800I A3 deployment in [Section 5.2](#dsv4pro-5-2-a3) with a
    Memcache KV Cache Pool. `MooncakeConnectorV2` transfers KV cache from Prefill to
    Decode and supports the pipeline parallelism used on Prefill.
    `AscendStoreConnector` stores reusable KV cache in Memcache. Both connectors
    are configured as children of `MultiConnector`.

    Use the official `DeepSeek-V4-Pro-0813-w4a8` checkpoint on all nodes. This example
    uses MRv2 and DSpark. Prefill enables layerwise Memcache access with
    `use_layerwise: true` and `layerwise_prefetch_layers: 3`; Decode uses
    `use_layerwise: false`.

    Before starting vLLM, follow the
    [Memcache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend)
    to install `memfabric-hybrid` and `memcache-hybrid`, configure
    `mmc-meta.conf`, `mmc-local.conf`, and `mmc-local-standalone.conf`, and start the
    Memcache MetaService. The
    Prefill and Decode nodes must use the same MetaService endpoint.

    This example uses separated Memcache deployment so that the pool reserves host
    memory before vLLM loads the model. Prepare one MetaService configuration and
    two LocalService configurations as described below.

    ##### Prepare the Memcache Services

    Create `mmc-meta.conf` on the node that runs MetaService:

    ```ini
    ock.mmc.log_level = info
    ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
    ock.mmc.meta_service.config_store_url = tcp://<META_SERVICE_IP>:6000
    ock.mmc.meta_service.metrics_url = http://<META_SERVICE_IP>:7000
    ock.mmc.evict_threshold_high = 70
    ock.mmc.evict_threshold_low = 60
    ```

    Create `mmc-local.conf` on all Prefill and Decode nodes for the LocalService
    embedded in vLLM:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
    ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:6000
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_sdma
    ock.mmc.local_service.dram.size = 0GB
    ock.mmc.local_service.max.dram.size = 1024GB
    ock.mmc.local_service.storage.enabled = false
    ```

    Create `mmc-local-standalone.conf` on all four nodes for the standalone Memcache
    process. It uses the same endpoints, world size, and protocol, but contributes
    the host DRAM used by the pool:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
    ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:6000
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_sdma
    ock.mmc.local_service.dram.size = 120GB
    ock.mmc.local_service.max.dram.size = 1024GB
    ock.mmc.local_service.storage.enabled = true
    ubsio.disk.path = <DISK_PATH>
    ubsio.mem.size_in_gb = 600
    ubsio.standalone.device_count = 1
    ubsio.standalone.force_new_disk = true
    ```

    `world_size` is the maximum number of LocalServices that can join the pool, not
    the DP size. Changing it after LocalServices connect requires restarting
    MetaService. `device_sdma` is recommended for A3 HCCS and requires LingQu
    Computing Network 1.5 or later. `max.dram.size` must be at least the largest
    `dram.size` contributed by any LocalService.

    Each node runs one standalone LocalService initialized with `store.init(0)`.
    Its `120GB` DRAM contribution is per node, rather than per NPU.
    `ubsio.standalone.device_count = 1` matches this single DRAM-enabled process.
    The `600GB` configured by `ubsio.mem.size_in_gb` is a separate host-memory pool
    for UBS IO, not the storage-device capacity. Reserve host memory for both
    allocations as well as the operating system and vLLM.

    `<DISK_PATH>` is the cache device path used in this example. Set `ubsio.disk.path` to
    the dedicated, unmounted cache device on each node. With
    `ubsio.standalone.force_new_disk = true`, UBS IO rebuilds **its cache layout and
    metadata** on the specified block device at startup, invalidating the existing
    cache.
    The MetaService eviction watermarks are `70` (high) and `60` (low).

    Start MetaService with `mmc-meta.conf`:

    ```shell
    export MMC_META_CONFIG_PATH="<MEMCACHE_CONFIG_DIR>/mmc-meta.conf"
    python3 -c 'from memcache_hybrid import MetaService; MetaService.main()'
    ```

    Save the following script as `memcache_standalone.py` on all Prefill and
    Decode nodes:

    ```python
    import time

    from memcache_hybrid import DistributedObjectStore


    store = DistributedObjectStore()
    result = store.init(0)
    if result != 0:
        raise RuntimeError(f"Failed to initialize Memcache: {result}")
    print("Successfully initialized Memcache")

    while True:
        time.sleep(1)
    ```

    After MetaService logs `Meta Service launched successfully`, start the standalone
    Memcache process on all Prefill and Decode nodes. Wait until each process prints
    `Successfully initialized Memcache` to confirm successful standalone
    initialization:

    ```shell
    export MMC_LOCAL_CONFIG_PATH="<MEMCACHE_CONFIG_DIR>/mmc-local-standalone.conf"
    python3 memcache_standalone.py
    ```

    For additional parameters and capacity constraints, refer to
    [Memcache and vLLM separated deployment](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#Memcache-and-vLLM-Separated-Deployment).

    Reuse `launch_online_dp.py` from [Section 5.2](#dsv4pro-5-2-a3). Set `MEMCACHE_INSTALL_PATH` to
    the `Location` reported by `pip show memcache_hybrid`, and set
    `MEMCACHE_CONFIG_DIR` to the absolute directory containing the configuration
    files above on all nodes.

    ##### Start the Prefill Nodes with Pooling

    Prepare the Prefill startup scripts. There is one Prefill instance, a PP2 engine spread across 2 nodes: the master node (`--node-rank 0`) starts the engine and its API server, while the worker node (`--node-rank 1`) joins the same engine with `--headless`. `--master-addr` must point to the master node's service IP so that the worker can join the cross-node process group, and both nodes must use identical model, TP, PP, PCP, and `VLLM_PP_LAYER_PARTITION` settings. `--port $1` takes the engine port from the first command-line argument of the script, for example `bash run_p_node0.sh 8000`. Set `MEMCACHE_INSTALL_PATH` to the `Location` reported by `pip show memcache_hybrid`, and set `MEMCACHE_CONFIG_DIR` to the absolute directory containing the configuration files above on both nodes.

    1. `run_p_node0.sh` on the master node:

        ```bash
        #!/usr/bin/env bash
        MEMCACHE_INSTALL_PATH="<MEMCACHE_INSTALL_PATH>"
        MEMCACHE_CONFIG_DIR="<MEMCACHE_CONFIG_DIR>"

        # nic_name is the network interface that owns local_ip on the current node.
        nic_name="xxx"
        local_ip="xxx"

        # node0_ip must be the local_ip of the master node of this instance.
        node0_ip="xxxx"

        export HCCL_IF_IP="$local_ip"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export TP_SOCKET_IFNAME="$nic_name"
        export HCCL_SOCKET_IFNAME="$nic_name"
        # Parameters required by the Memcache KV Cache Pool.
        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export LD_LIBRARY_PATH="$MEMCACHE_INSTALL_PATH/memcache_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:/usr/local/lib/"
        export MMC_LOCAL_CONFIG_PATH="$MEMCACHE_CONFIG_DIR/mmc-local.conf"

        export VLLM_USE_V2_MODEL_RUNNER=1
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_BUFFSIZE=1024
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_EXEC_TIMEOUT=2040
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export VLLM_PP_LAYER_PARTITION="32,29"


        vllm serve /path/to/DeepSeek-V4-Pro-0813-w4a8 \
            --tensor-parallel-size 2 \
            --prefill-context-parallel-size 8 \
            --host 0.0.0.0 \
            --port $1 \
            --nnodes 2 \
            --node-rank 0 \
            --master-addr $node0_ip \
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
              "kv_connector": "MultiConnector",
              "kv_role": "kv_producer",
              "kv_port": 30000,
              "kv_connector_extra_config": {
                "connectors": [
                  {
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_producer",
                    "kv_port": 30000,
                    "kv_connector_extra_config": {
                      "prefill": {
                        "dp_size": 1,
                        "tp_size": 2,
                        "pp_size": 2,
                        "pcp_size": 8
                      },
                      "decode": {
                        "dp_size": 16,
                        "tp_size": 2,
                        "pp_size": 1,
                        "pcp_size": 1
                      }
                    }
                  },
                  {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {
                      "lookup_rpc_port": "0",
                      "backend": "memcache",
                      "use_layerwise": true,
                      "layerwise_prefetch_layers": 3
                    }
                  }
                ]
              }
            }'
        ```

    2. `run_p_node1.sh` on the worker node: use the same content as `run_p_node0.sh` with these changes: set `local_ip` to the IP of this node, keep `node0_ip` as the master node's service IP, change `--node-rank` to `1`, append `--headless` to `vllm serve` so that this node runs only its PP stage without starting an API server, and assign a unique `kv_port`, for example `"kv_port":"30200"`.

    After the Decode engines are ready, start the Prefill master node before its
    worker node in separate terminals on the corresponding machines. The first
    argument is the engine port passed to `--port $1`:

    1. Prefill master node

        ```shell
        bash run_p_node0.sh 8000
        ```

    2. Prefill worker node: headless

        ```shell
        bash run_p_node1.sh 8000
        ```

    ##### Start the Decode Nodes with Pooling

    On both Decode nodes, save the following script as `run_dp_template.sh` in the
    same directory as `launch_online_dp.py`. Set `local_ip`, `nic_name`,
    `MEMCACHE_INSTALL_PATH`, and `MEMCACHE_CONFIG_DIR` for the local environment.

    ```shell
    MEMCACHE_INSTALL_PATH="<MEMCACHE_INSTALL_PATH>"
    MEMCACHE_CONFIG_DIR="<MEMCACHE_CONFIG_DIR>"

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
    # Parameters required by the Memcache KV Cache Pool.
    export PYTHONHASHSEED=0
    export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH="$MEMCACHE_INSTALL_PATH/memcache_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:/usr/local/lib/"
    export MMC_LOCAL_CONFIG_PATH="$MEMCACHE_CONFIG_DIR/mmc-local.conf"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_BUFFSIZE=2200
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_EXEC_TIMEOUT=2040
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
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
          "kv_connector": "MultiConnector",
          "kv_role": "kv_consumer",
          "kv_port": 30100,
          "kv_connector_extra_config": {
            "connectors": [
              {
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_consumer",
                "kv_port": 30100,
                "kv_connector_extra_config": {
                  "prefill": {
                    "dp_size": 1,
                    "tp_size": 2,
                    "pp_size": 2,
                    "pcp_size": 8
                  },
                  "decode": {
                    "dp_size": 16,
                    "tp_size": 2,
                    "pp_size": 1,
                    "pcp_size": 1
                  }
                }
              },
              {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                  "lookup_rpc_port": "0",
                  "backend": "memcache",
                  "use_layerwise": false
                }
              }
            ]
          }
        }'
    ```

    Start the Decode engines in separate terminals on the corresponding machines:

    ```shell
    # Decode node 0
    python launch_online_dp.py --dp-size 16 --tp-size 2 --pcp-size 1 --dp-size-local 8 --dp-rank-start 0 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 8000

    # Decode node 1
    python launch_online_dp.py --dp-size 16 --tp-size 2 --pcp-size 1 --dp-size-local 8 --dp-rank-start 8 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 8000
    ```

    ##### Startup and Verification

    Start the components in this order:

    1. Memcache MetaService.
    2. Standalone Memcache processes on all four nodes. Wait for successful initialization.
    3. Decode engines. Wait until every engine reports `Application startup complete`.
    4. Both Prefill nodes. Wait until the Prefill service is ready.
    5. Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

    Send the same request twice with a reusable prefix. Confirm that Prefill logs
    show Memcache lookup, get, put, and hit activity, and that Decode logs still
    show normal Prefill-to-Decode KV transfer.

    Key pooling parameters:

    - `MultiConnector` combines Prefill-to-Decode transfer through `MooncakeConnectorV2` with KV reuse through `AscendStoreConnector`. Both child connectors use the producer role on Prefill and the consumer role on Decode.
    - `backend: "memcache"` selects the Memcache KV Cache Pool backend. All nodes use the same MetaService endpoint.
    - Prefill uses `use_layerwise: true` with `layerwise_prefetch_layers: 3`. Decode keeps `use_layerwise: false`.

    For Memcache configuration, capacity tuning, and troubleshooting, refer to the
    [KV Cache Pool guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend).

=== "A2 series"

    #### A2 series {: #dsv4pro-5-3-a2 }

    This configuration extends the Atlas 800I A2 deployment in [Section 5.2](#dsv4pro-5-2-a2) with a
    KV Cache Pool on the Mooncake backend. `MooncakeHybridConnector` transfers KV cache
    from Prefill to Decode and controls HBM-to-host-memory tiering through
    `enable_tiered_storage` and `hbm_to_ram_threshold_ratio`.
    `AscendStoreConnector` stores reusable KV cache in the Mooncake pool with
    `tiered_pooling_mode: "async_bypass"`. Both connectors are configured as children
    of `MultiConnector`.

    Use the official `DeepSeek-V4-Pro-0813-w4a8` checkpoint on all nodes. This example
    uses MRv2 and DSpark with `num_speculative_tokens: 7`. Unlike the A3 example, the
    pool runs on the Mooncake backend and both roles keep `use_layerwise: false`.

    ##### Prepare the Mooncake Service

    Follow the [Mooncake backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-1-mooncake-backend)
    to install the Mooncake software (`mooncake >= 0.3.11`). Create `mooncake.json` on
    every Prefill and Decode node; all nodes must point `master_server_address` at the
    same master:

    ```json
    {
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "ascend",
        "device_name": "",
        "master_server_address": "<MASTER_IP>:50099",
        "global_segment_size": "24GB",
        "preferred_segment": false,
        "prefer_alloc_in_same_node": true
    }
    ```

    Start `mooncake_master` on the master node:

    ```shell
    mooncake_master \
        --port 50099 \
        --default_kv_lease_ttl 11000 \
        --eviction_high_watermark_ratio 0.9 \
        --eviction_ratio 0.1
    ```

    On every Prefill and Decode node, set `MOONCAKE_CONFIG_PATH` to the full path of
    `mooncake.json` and preload jemalloc. The pooled startup scripts below export both
    from `MOONCAKE_CONFIG_DIR`.

    Reuse `launch_online_dp.py` from [Section 5.2](#dsv4pro-5-2-a2).

    ##### Start the Prefill Nodes with Pooling

    Save the following script as `run_dp_template.sh` on the four Prefill nodes (they
    share the same script). Set `local_ip` and `nic_name` for the current node, and set
    `MOONCAKE_CONFIG_DIR` to the absolute directory containing `mooncake.json`:

    ```shell
    MOONCAKE_CONFIG_DIR="<MOONCAKE_CONFIG_DIR>"

    local_ip=xx.xx.xx.1/2/3/4 # change to your own ip
    nic_name="xxxx" # change to your own nic name

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=204
    export HCCL_INTRA_ROCE_ENABLE=1
    export PYTHONHASHSEED=0
    export ASCEND_CONNECT_TIMEOUT=16000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export HCCL_RDMA_TIMEOUT=17
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export MOONCAKE_CONFIG_PATH="$MOONCAKE_CONFIG_DIR/mooncake.json"
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2

    vllm serve /path/to/DeepSeek-V4-Pro-0813-w4a8 \
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
      --max-model-len 150000 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 16 \
      --no-disable-hybrid-kv-cache-manager \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
      --trust-remote-code \
      --gpu-memory-utilization 0.92 \
      --quantization ascend \
      --block-size 32 \
      --enforce-eager \
      --speculative-config '{"num_speculative_tokens": 7, "method": "dspark","enforce_eager": true}' \
      --additional-config '{"enable_cpu_binding": true, "enable_fused_mc2": 1, "enable_flashcomm1":true}' \
      --kv-transfer-config \
      '{"kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "kv_connector_extra_config": {
          "connectors": [
            {
              "kv_connector": "MooncakeHybridConnector",
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
                },
                "enable_tiered_storage": true,
                "hbm_to_ram_threshold_ratio": 0.85
              }
            },
            {
              "kv_connector": "AscendStoreConnector",
              "kv_role": "kv_producer",
              "kv_connector_extra_config": {
                "lookup_rpc_port": "0",
                "backend": "mooncake",
                "use_layerwise": false,
                "tiered_pooling_mode": "async_bypass"
              }
            }
          ]
        }
      }'
    ```

    Start the Prefill engines on the corresponding machines:

    ```shell
    # Prefill node 0
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 8000

    # Prefill node 1
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 8000

    # Prefill node 2
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 2 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 8000

    # Prefill node 3
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 3 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 8000
    ```

    ##### Start the Decode Nodes with Pooling

    Save the following script as `run_dp_template.sh` on the four Decode nodes. Set
    `local_ip`, `nic_name`, and `MOONCAKE_CONFIG_DIR` for the local environment:

    ```shell
    MOONCAKE_CONFIG_DIR="<MOONCAKE_CONFIG_DIR>"

    local_ip=xx.xx.xx.5/6/7/8 # change to your own ip
    nic_name="xxxx" # change to your own nic name

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=204
    export HCCL_INTRA_ROCE_ENABLE=1
    export PYTHONHASHSEED=0
    export ASCEND_CONNECT_TIMEOUT=16000
    export ASCEND_TRANSFER_TIMEOUT=10000
    export HCCL_RDMA_TIMEOUT=17
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024
    export TASK_QUEUE_ENABLE=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export MOONCAKE_CONFIG_PATH="$MOONCAKE_CONFIG_DIR/mooncake.json"
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
    export HCCL_HOST_SOCKET_PORT_RANGE=auto
    export HCCL_NPU_SOCKET_PORT_RANGE=auto

    vllm serve /path/to/DeepSeek-V4-Pro-0813-w4a8 \
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
        --max-model-len 150000 \
        --max-num-batched-tokens 256 \
        --max-num-seqs 32 \
        --async-scheduling \
        --block-size 32 \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
        --trust-remote-code \
        --speculative-config '{"num_speculative_tokens": 7,"method": "dspark","enforce_eager": true}' \
        --gpu-memory-utilization 0.92 \
        --quantization ascend \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-transfer-config \
        '{"kv_connector": "MultiConnector",
          "kv_role": "kv_consumer",
          "kv_port": "30100",
          "kv_connector_extra_config": {
            "connectors": [
              {
                "kv_connector": "MooncakeHybridConnector",
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
                  },
                  "enable_tiered_storage": true,
                  "hbm_to_ram_threshold_ratio": 0.85
                }
              },
              {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                  "lookup_rpc_port": "0",
                  "backend": "mooncake",
                  "use_layerwise": false,
                  "tiered_pooling_mode": "async_bypass"
                }
              }
            ]
          }
        }' \
        --additional-config '{
            "ascend_compilation_config":{
                  "enable_npugraph_ex":true,
                  "enable_static_kernel":false
            },
           "enable_cpu_binding":true,
           "multistream_overlap_shared_expert":true,
           "scheduler_config":{"recompute_scheduler_enable":true}
        }'
    ```

    Start the Decode engines on the corresponding machines:

    ```shell
    # Decode node 0
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 8000

    # Decode node 1
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 8000

    # Decode node 2
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 2 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 8000

    # Decode node 3
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 3 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 8000
    ```

    ##### Startup and Verification

    Start the components in this order:

    1. `mooncake_master` on the master node. Wait until it is listening on `<MASTER_IP>:50099` before starting the engines.
    2. The Prefill engines on the four Prefill nodes.
    3. The Decode engines on the four Decode nodes.
    4. Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

    Send the same request twice with a reusable prefix. Confirm that the vLLM logs
    show Mooncake pool and tiered-storage activity, and that Decode logs still show
    normal Prefill-to-Decode KV transfer.

    Key pooling parameters:

    - `MultiConnector` combines Prefill-to-Decode transfer through `MooncakeHybridConnector` with KV reuse through `AscendStoreConnector`. Both child connectors use the producer role on Prefill and the consumer role on Decode.
    - `backend: "mooncake"` selects the Mooncake KV Cache Pool backend. All Prefill and Decode nodes must point `master_server_address` at the same master.
    - `enable_tiered_storage: true` with `hbm_to_ram_threshold_ratio: 0.85` controls HBM-to-host-memory tiering on the transfer connector.
    - `AscendStoreConnector` uses `tiered_pooling_mode: "async_bypass"`, and both roles keep `use_layerwise: false` — unlike the A3 example, which enables layerwise Memcache access on Prefill.

    For Mooncake configuration, capacity tuning, and troubleshooting, refer to the
    [KV Cache Pool guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-1-mooncake-backend).

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

Replace `<node0_ip>:<port>` with the IP address and port of the node that serves requests: for the mixed deployment in [Section 5.1](#51-multi-node-online-deployment), use the primary node (Node0) that exposes the API server; for the PD separation deployment in [Section 5.2](#52-multi-node-pd-separation-deployment), use the proxy node instead, since all requests are routed to the Prefill and Decode pools through the proxy.

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
| GPQA | - | accuracy | gen | 89.90 | 2 Atlas 800I A3 (128GB × 8) |
| GPQA | - | accuracy | gen | 92.93 | 8 Ascend 950DT nodes (96GB × 8) |
| GSM8K | - | accuracy | gen | 96.21 | 2 Atlas 800I A3 (128GB × 8) |
| GSM8K | - | accuracy | gen | 97.54 | 950PR&950DT products, w4a4c8 |

> **Note**: Dataset evaluation results fluctuate between runs due to sampling and runtime non-determinism. When a dataset is evaluated multiple times, the average score is reported.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are based on the [Section 5](#5-online-service-deployment) deployment examples and specific test environments. They are for reference only and are not globally optimal. In particular, the long-context configuration is a suggested starting point for inputs around 1M tokens and must be validated against the actual input/output length, prefix cache hit rate, concurrency, memory usage, and deployment environment. Refer to [Section 9.2](#92-tuning-guidelines) for further tuning.

#### Table 1: Scenario Overview

> `*Total NPUs` indicates the total number of NPUs used across all nodes.

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
| --- | --- | --- | --- | --- |
| Short Sequence / High Throughput | 1P1D deployment (2 Prefill + 2 Decode A3 nodes) | 64 (A3) | DeepSeek-V4-Pro-0813-w4a8 | Prefill uses DP1/TP2/PP2/PCP8 with 16 sequences, while Decode uses DP16/TP2/PP1/PCP1 with 30 sequences and 180 batched tokens. Enable Fused MC2 on Prefill to prioritize throughput. |
| Long Context (~1M Input, Reference) | 1P1D deployment (2 Prefill + 2 Decode A3 nodes) | 64 (A3) | DeepSeek-V4-Pro-0813-w4a8 | For this separate long-context reference, use DP2/TP16 on both sides, reduce both sides to 2 sequences, and reduce Decode batched tokens to 16 to reserve memory for a 1M-token context. |
| Long Context (~1M Input) | 2P1D deployment (4 Prefill + 4 Decode Ascend 950DT nodes; each Prefill instance spans 2 nodes) | 64 (Ascend 950DT) | DeepSeek-V4-Pro-0813 | Run 2 Prefill instances, each a DP1/TP1/PP2 engine with prefill context parallel (PCP8) over 2 nodes, and one Decode pool with DP32/TP1. |

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | NPUs | TP | DP | Max Num Seqs | Max Num Batched Tokens | Max Model Len | DSpark Speculation Num | Fused MC2 | EP Switch | FC | DSA-CP (MRv1) / PCP (MRv2) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Short Sequence / High Throughput | Prefill Pool | 32 (2 A3 nodes) | 2 | 1 | 16 | 4096 | 150000 | 5 | On | On | On | PCP=8 |
| Short Sequence / High Throughput | Decode Pool | 32 (2 A3 nodes) | 2 | 16 | 30 | 180 | 150000 | 5 | Off | On | Off | PCP=1 |
| Long Context (~1M Input, Reference) | Prefill Pool | 32 (2 A3 nodes) | 16 | 2 | 2 | 4096 | 1048576 | 5 | Off | On | On | PCP=1 |
| Long Context (~1M Input, Reference) | Decode Pool | 32 (2 A3 nodes) | 16 | 2 | 2 | 16 | 1048576 | 5 | Off | On | Off | PCP=1 |
| Long Context (~1M Input) | Prefill Pool | 32 (4 950DT nodes, 2 instances) | 1 | 1 | 16 | 8192 | 1048576 | 5 | Off | Off | Off | PCP=8 |
| Long Context (~1M Input) | Decode Pool | 32 (4 950DT nodes) | 1 | 32 | 32 | 512 | 1048576 | 5 | Off | Off | Off | PCP=1 |

> `FC` reports the FlashComm1.
>
> `DSA-CP (MRv1) / PCP (MRv2)` describes context parallelism according to the model runner version: MRv1 uses the `enable_dsa_cp` switch, while MRv2 uses `--prefill-context-parallel-size` to specify the PCP size. `PCP=8` means eight-way context parallelism; `PCP=1` means no context parallelism. For the long-context Prefill reference, enable FlashComm1 and disable context parallelism (`enable_dsa_cp: false` on MRv1, or PCP size `1` on MRv2). The long-context example for 950DT products runs on the V2 model runner and uses `PCP=8` on Prefill; `enable_dsa_cp` is mutually exclusive with PCP and must stay off there.
>
> Fused MC2 is disabled in the long-context reference configuration because it increases memory usage. If sufficient memory is available and higher performance is the priority, enabling Fused MC2 on Prefill is recommended.
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

A: For 950PR&950DT products, the W4A4C8 weights of `DeepSeek-V4-Pro-0813` are quantized with [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_pro_w4a4c8.yaml). To preserve accuracy at such a low bit-width, a mixed-precision strategy is used: only the modules that dominate the parameter count are quantized to 4 bit, while precision-sensitive modules fall back to higher precision or are not quantized at all.

| Module | Quantization configuration |
| ------ | -------------------------- |
| MoE FFN routed experts | W4A4 MXFP4 |
| Attention linear layers | W8A8 MXFP8 |
| FFN shared experts | W8A8 MXFP8 |
| Router gates, compressor projections (`wgate`/`wkv`), indexer projections, and MTP layers | Not quantized (kept in high precision) |

- **Quantization algorithm**: The weights and activations use the MXFP4/MXFP8 microscaling formats. Each tensor is divided into blocks of 32 elements, and each block shares a single scaling factor obtained with symmetric min-max calibration on a mixed calibration dataset. Block-wise scaling adapts to the local dynamic range of each tensor, so the quantization error is far lower than that of uniform INT4 quantization.
- **C8 (FP8 KV cache)**: C8 is a dynamic quantization applied to the KV cache during inference. On 950PR&950DT products, the KV cache is quantized to FP8 (not INT8). It is unrelated to the parameters stored in the weight checkpoint and is controlled only by the serving configuration. DeepSeek-V4-Flash and DeepSeek-V4-Pro share exactly the same C8 scheme:
    - **Attention KV cache** (FP8 by default on 950PR&950DT products; switch back to BF16 with `--kv-cache-dtype bfloat16`): the compressed KV latent, which serves as both K and V, is dynamically quantized per 64-element tile to FP8 (E4M3) with a shared E8M0 scale per tile; the RoPE part of the KV cache stays in BF16, and Q is not quantized.
    - **Indexer cache** (enabled with `--attention_config.indexer_kv_dtype fp8`): both indexer Q and K are first rotated by a Hadamard matrix to spread outliers, and then dynamically quantized per token to FP8 (E4M3) with per-token FP32 scales.

For W4A4C8 accuracy scores, see [Accuracy Evaluation](#7-accuracy-evaluation).
