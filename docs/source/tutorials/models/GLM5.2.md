# GLM-5.2 & GLM-5.3

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) uses a Mixture-of-Experts (MoE) architecture and targets complex systems engineering and long-horizon agentic tasks. [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) shares its base model with GLM-5.2 but has different post-training. This page covers both models; commands and validation results are version-specific unless explicitly stated otherwise.

This document shows the verification steps for both models, including supported features, environment preparation, single-node and multi-node deployment, PP2 Prefill-Decode disaggregation, MemCache KV pooling, 1M DCP, accuracy, and performance evaluation. Shared deployment settings follow the GLM-5.3 configurations. Checkpoint identity and measured results remain model-specific.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

Download the checkpoint from [Eco-Tech on Modelers](https://modelers.cn/user/Eco-Tech?model_name=glm) for the hardware shown below. A3 and A5 checkpoints use different quantization formats; do not interchange them. The scripts use `/root/.cache/modelscope/hub/models/vllm-ascend/<checkpoint-name>` as a local mount path, not as a download source. Preserve the exact case of the repository name when naming the local directory.

|  Weight Version          | Hardware Requirements                                             | Download Links |
|--------------------------|-------------------------------------------------------------------|----------------|
|  `GLM-5.2-w8a8c8`        | **A3**: 2 Atlas 800I A3 (128GB × 8) nodes; **A2**: 4 Atlas 800I A2 (64GB × 8) nodes | [Modelers](https://modelers.cn/models/Eco-Tech/GLM-5.2-w8a8c8) |
|  `GLM-5.2-W8A8C8-mxfp8` | **A5**: 950PR&950DT Products | [Modelers](https://modelers.cn/models/Eco-Tech/GLM-5.2-W8A8C8-mxfp8) |
|  `GLM-5.3-w8a8c8`        | **A3**: Atlas 800I A3 (128GB × 8); **A2**: Atlas 800I A2 | [Modelers](https://modelers.cn/models/Eco-Tech/GLM-5.3-w8a8c8) |
|  `GLM-5.3-W8A8C8-mxfp8` | **A5**: 950PR&950DT Products | [Modelers](https://modelers.cn/models/Eco-Tech/GLM-5.3-W8A8C8-mxfp8) |
|  `GLM-5.3-W4A4C8-mxfp4` | **A5**: 1 950DT Product (96GB × 8) | Locally validated checkpoint; prepare with [msmodelslim](https://gitcode.com/Ascend/msmodelslim) |
|  `GLM-5.2-w4a4c8-mxfp4` | **A5**: 950PR&950DT Products | [Modelers](https://modelers.cn/models/Eco-Tech/GLM-5.2-w4a4c8-mxfp4) |

- `GLM-5.2-w8a8c8`: The weights have been verified and are recommended for use on A3. For GLM-5.3, use its own checkpoint and the version-specific commands below; do not apply the GLM-5.2 accuracy or performance results to GLM-5.3.
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.
- `GLM-5.2-w4a4c8-mxfp4` (A5, W4A4 MXFP4 mixed-precision quantization): For how the ultra-low-bit quantization preserves accuracy, see the [W4A4C8 FAQ](#q-how-do-the-w4a4c8-ultra-low-bit-quantized-weights-preserve-accuracy).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

- You can use our official docker image to run GLM-5.2 and GLM-5.3. The hardware-specific container commands below follow the GLM-5.3-Flash guide.

Before launching a container, replace the image tag with a compatible, tested build for your target model and hardware. The documentation-wide image tag is not a model compatibility guarantee. Use the same image digest on all nodes.

For A5 PD deployment, the host must already have the platform-generated HCCL/HIXL configuration in `/etc/hccl_rootinfo.json` and `/etc/hixlep/`, as well as the device and driver paths mounted below. Do not substitute an empty directory for the communication configuration.

=== "950DT Products"

    Start the docker image on each node.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
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
    -v /usr/bin/urma_admin:/usr/bin/urma_admin \
    -v /lib/route.conf:/lib/route.conf \
    -itd $IMAGE bash
    ```

=== "A3 series"

    Start the docker image on each node.

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

    Start the docker image on each node.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    export NAME=vllm-ascend

    docker run --rm \
    --name $NAME \
    --net=host \
    --shm-size=500g \
    --privileged \
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
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
    ```

If you want to deploy multi-node environment, you need to set up environment on each node.

## 5 Online Service Deployment {: #5-online-service-deployment }

For configurations shared by GLM-5.2 and GLM-5.3, use the GLM-5.3 settings
below and select the matching checkpoint. The default model is GLM-5.3.
Changing a checkpoint does not transfer accuracy or performance validation
from one model to the other. Keep thinking enabled for GLM-5.3.

| Scenario | Hardware | Configuration |
| --- | --- | --- |
| W4A4C8 single node | 950DT | Section 5.1.1; GLM-5.2 and GLM-5.3 startup/short requests validated |
| W8A8C8 single node, 32K | A3 | Section 5.1.1; reduced-context layout |
| W8A8C8 colocated | 950DT / A3 / A2 | Section 5.1.2; A3/A2 retain the v0.23.0 reference |
| PP2 PD | 950DT / A3 | Section 5.1.3; A3 uses v0.30.0rc1 |
| MemCache KV pool | A3 | Section 5.1.4; same PP2 PD topology |
| 1M DCP PD | Four A3 nodes | Section 5.2.1; Prefill DP2/TP16/DCP16, Decode DP4/TP8/DCP8 |

For the A3 release PD, pooling and 1M examples, use
`quay.io/ascend/vllm-ascend:v0.30.0rc1-a3` on every node with the mounts
in Section 4. A2 uses `quay.io/ascend/vllm-ascend:v0.30.0rc1`.
The A5 W4A4C8 validation uses the version-pinned image listed in Section 5.1.1.

### 5.1 Context Below 1M

#### 5.1.1 Single-Node Deployment

=== "950DT Products"

    Both GLM-5.3 and GLM-5.2 W4A4C8/MXFP4 checkpoints passed startup and
    short chat/completion requests on one 950DT Product (96GB × 8) in the
    2026-10-10 validation. Use W4A4C8 for this single-node example; the
    GLM-5.3 W8A8C8/MXFP8 checkpoint failed during expert-weight allocation
    on that machine and is covered by the PP2 examples instead.

    The tested image was
    `vllm-ascend:dev-26.2.0.day20261008-A5-py311-openEuler24.03-lts-aarch64`,
    with vLLM `0.30.0+empty` and vLLM-Ascend commit `a50eb57f1`.
    The local GLM-5.3 checkpoint has 78 layers and 102 indexed shards;
    its identity with a public release revision was not verified.
    Use a validated local checkpoint path. See
    [W4A4C8 Validation](#73-w4a4c8-a5-validation-records) for the exact scope.

    For GLM-5.2, replace the checkpoint path in `vllm serve` with
    `/root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a4c8-mxfp4`.
    The GLM-5.3 path below is an example local directory, not a download source.

    ```shell

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=1
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-W4A4C8-mxfp4 \
      --host 0.0.0.0 \
      --port 8000 \
      --api-server-count 1 \
      --data-parallel-size 1 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --max-model-len 135000 \
      --max-num-batched-tokens 16384 \
      --max-num-seqs 20 \
      --served-model-name glm \
      --gpu-memory-utilization 0.9 \
      --enable-prefix-caching \
      --trust-remote-code \
      --enforce-eager \
      --quantization ascend \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --kv-cache-dtype fp8 \
      --attention_config.indexer_kv_dtype fp8 \
      --additional-config '{"enable_flashcomm1": true, "enable_cpu_binding": true, "multistream_overlap_shared_expert": true, "enable_dsa_cp": true}'
    ```

=== "A3 Series"

    For a reduced context of 32768 tokens, the W8A8C8 checkpoint can run on 1 Atlas 800I A3 (128GB × 8) with DP1/TP16. Select the GLM-5.2 or GLM-5.3 checkpoint below. This is not the full-context deployment: use the multi-node deployment for longer contexts.

    ```shell
    # GLM-5.3:
    MODEL_PATH=/root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8
    # GLM-5.2 (use this line instead):
    # MODEL_PATH=/root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8

    export HCCL_BUFFSIZE=200
    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port 8000 \
      --api-server-count 1 \
      --data-parallel-size 1 \
      --tensor-parallel-size 16 \
      --seed 1024 \
      --served-model-name glm \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --enable-auto-tool-choice \
      --max-num-seqs 12 \
      --max-model-len 32768 \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --gpu-memory-utilization 0.92 \
      --quantization ascend \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --kv-cache-dtype int8 \
      --attention_config.indexer_kv_dtype int8 \
      --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "multistream_overlap_shared_expert": true, "enable_flashcomm1": true}' \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

#### 5.1.2 Multi-node Deployment

If you want to deploy a multi-node environment, first verify the data-plane network by following [Verify Multi-node Communication Environment](../../getting_started/installation.md#installation-multi-node-interconnect).

=== "950DT Products"

    - `GLM-5.3-W8A8C8-mxfp8` (or the matching GLM-5.2 checkpoint): can be deployed on 2 950DT Products (96GB × 8) with PP2/TP8 and expert parallelism. Both nodes must mount the same checkpoint path and expose all 8 NPUs to their containers. The default checkpoint below is `GLM-5.3-W8A8C8-mxfp8`; select `GLM-5.2-W8A8C8-mxfp8` on both nodes only when deploying GLM-5.2.

    Run the following scripts on two nodes respectively.

    **node 0**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="xxx"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=1
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
    export VLLM_PP_LAYER_PARTITION="38,40"

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-W8A8C8-mxfp8 \
      --host 0.0.0.0 \
      --port 30001 \
      --api-server-count 1 \
      --nnodes 2 \
      --node-rank 0 \
      --master-addr "$node0_ip" \
      --master-port 13388 \
      --pipeline-parallel-size 2 \
      --distributed-executor-backend mp \
      --tensor-parallel-size 8 \
      --data-parallel-size 1 \
      --data-parallel-size-local 1 \
      --data-parallel-address "$node0_ip" \
      --data-parallel-rpc-port 12980 \
      --enable-expert-parallel \
      --max-model-len 135000 \
      --max-num-batched-tokens 16384 \
      --max-num-seqs 20 \
      --served-model-name glm \
      --gpu-memory-utilization 0.9 \
      --enable-prefix-caching \
      --trust-remote-code \
      --enforce-eager \
      --quantization ascend \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --kv-cache-dtype fp8 \
      --attention_config.indexer_kv_dtype fp8 \
      --additional-config '{"enable_flashcomm1": true, "enable_cpu_binding": true, "multistream_overlap_shared_expert": true, "enable_dsa_cp": true}'
    ```

    **node 1**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="xxx"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=1
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
    export VLLM_PP_LAYER_PARTITION="38,40"

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-W8A8C8-mxfp8 \
      --host 0.0.0.0 \
      --port 30001 \
      --headless \
      --nnodes 2 \
      --node-rank 1 \
      --master-addr "$node0_ip" \
      --master-port 13388 \
      --pipeline-parallel-size 2 \
      --distributed-executor-backend mp \
      --tensor-parallel-size 8 \
      --data-parallel-size 1 \
      --data-parallel-size-local 1 \
      --data-parallel-address "$node0_ip" \
      --data-parallel-rpc-port 12980 \
      --enable-expert-parallel \
      --max-model-len 135000 \
      --max-num-batched-tokens 16384 \
      --max-num-seqs 20 \
      --served-model-name glm \
      --gpu-memory-utilization 0.9 \
      --enable-prefix-caching \
      --trust-remote-code \
      --enforce-eager \
      --quantization ascend \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --kv-cache-dtype fp8 \
      --attention_config.indexer_kv_dtype fp8 \
      --additional-config '{"enable_flashcomm1": true, "enable_cpu_binding": true, "multistream_overlap_shared_expert": true, "enable_dsa_cp": true}'
    ```

=== "A3 series"
    -  `GLM-5.3-w8a8c8`: can be deployed on 2 Atlas 800I A3 (64GB × 16).

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
    export HCCL_TRANSFER_TIMEOUT=600
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=3600
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_MLAPO=1

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --safetensors-load-strategy prefetch \
        --api-server-count 1 \
        --data-parallel-size 8 \
        --data-parallel-start-rank 0 \
        --data-parallel-size-local 4 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm-5 \
        --tool-call-parser glm47 \
        --reasoning-parser glm47 \
        --enable-auto-tool-choice \
        --max-num-seqs 6 \
        --max-model-len 202752 \
        --max-num-batched-tokens 4096 \
        --trust-remote-code \
        --gpu-memory-utilization 0.90 \
        --quantization ascend \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "enable_fused_mc2": 1, "enable_flashcomm1": true}'  \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
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
    export HCCL_TRANSFER_TIMEOUT=600
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=3600
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_MLAPO=1

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --headless \
        --data-parallel-size 8 \
        --data-parallel-start-rank 4 \
        --data-parallel-size-local 4 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm-5 \
        --tool-call-parser glm47 \
        --reasoning-parser glm47 \
        --enable-auto-tool-choice \
        --max-num-seqs 6 \
        --max-model-len 202752 \
        --max-num-batched-tokens 4096 \
        --trust-remote-code \
        --gpu-memory-utilization 0.92 \
        --quantization ascend \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --async-scheduling \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "enable_fused_mc2": 1, "enable_flashcomm1": true}' \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

=== "A2 series"

    - `GLM-5.3-w8a8c8`: can be deployed on 4 Atlas 800I A2 (64GB × 8).

    Run the following scripts on four nodes respectively.

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
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=200
    export HCCL_CONNECT_TIMEOUT=120
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ACL_OP_INIT_MODE=1
    export CPU_AFFINITY_CONF=1
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1200

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --max-model-len 135000 \
        --data-parallel-size 4 \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank 0 \
        --data-parallel-address "${node0_ip}" \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm-5 \
        --reasoning-parser glm47 \
        --safetensors-load-strategy prefetch \
        --max-num-seqs 128 \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --quantization ascend \
        --gpu-memory-utilization 0.92 \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_flashcomm1": true}' \
        --enable-prefix-caching \
        --async-scheduling \
        --api-server-count 1
    ```

    **node 1-3**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="xxx"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxx"

    # node1: dp_start_rank=1, node2: dp_start_rank=2, node3: dp_start_rank=3
    dp_start_rank=1

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export VLLM_RPC_TIMEOUT=360000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=200
    export HCCL_CONNECT_TIMEOUT=120
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ACL_OP_INIT_MODE=1
    export CPU_AFFINITY_CONF=1
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1200

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --headless \
        --max-model-len 135000 \
        --data-parallel-size 4 \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank ${dp_start_rank} \
        --data-parallel-address "${node0_ip}" \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm-5 \
        --reasoning-parser glm47 \
        --safetensors-load-strategy prefetch \
        --max-num-seqs 128 \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --quantization ascend \
        --gpu-memory-utilization 0.92 \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true,"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_flashcomm1": true}' \
        --enable-prefix-caching \
        --async-scheduling
    ```

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. max-model-len and max-num-seqs need to be set according to the actual usage scenario.

**Multi-node network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`: Network interface configuration for multi-node communication. Set `nic_name` to the network interface name (obtained via `ifconfig`) and `local_ip` to the current node's IP address. These must be correctly configured on each node for successful multi-node communication.
- `--data-parallel-size 8`: Total number of data parallel ranks across all nodes (4 ranks per node in this scenario).
- `--data-parallel-size-local 4`: Number of data parallel ranks on the current node.
- `--data-parallel-start-rank`: Starting rank offset for data parallel ranks on this node. Node 0 uses `0`, node 1 uses `4`.
- `--data-parallel-address`: IP address of the data parallel master node (node 0). Must match the `local_ip` of the master node.
- `--data-parallel-rpc-port 12980`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: Indicates a non-master node (used on node 1-3). Do not use on node 0.

**A2-specific environment variables:**

- `CPU_AFFINITY_CONF=1`: Enables CPU core affinity binding for worker processes.
- `ACL_OP_INIT_MODE=1`: ACL operator initialization mode to speed up operator compilation.
- `VLLM_RPC_TIMEOUT=360000` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000` / `HCCL_EXEC_TIMEOUT=200` / `HCCL_CONNECT_TIMEOUT=120` / `VLLM_ENGINE_READY_TIMEOUT_S=1200`: Timeout settings for multi-node startup and model execution on the slower A2 platform. Increase them if the engine fails to become ready in time.

**KV cache configuration:**

- `--kv-cache-dtype turboquant_4bit_nc`: Enables 4-bit TurboQuant quantization for the SFA packed main KV cache, as an alternative to `--kv-cache-dtype int8`. It requires an A2/A3 device, cannot be combined with the C8 cache, and in Prefill-Decode disaggregation the Mooncake option `"use_ascend_direct"` must be `false`.

**Notice:**
This scenario enables `additional_config.enable_fused_mc2=1` (fused `dispatch_ffn_combine`/`mega_moe` operators). Fused MC2 conflicts with `multistream_overlap_shared_expert` — the two optimizations must not be enabled at the same time (the runtime forcibly disables `multistream_overlap_shared_expert` when fused MC2 is on).

#### 5.1.3 Prefill-Decode Disaggregation

Select the hardware tab below for the prefill and decode topology. Prefill and decode run on separate node groups and exchange KV cache through Mooncake.

In the PD disaggregation scenario, Mooncake is used as the KV cache transfer connector between the prefill and decode nodes. Please refer to [KV Cache Pool (Ascend Store) Deployment Guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md) for the Mooncake configuration.

=== "950DT Products"

    The reference topology uses four 950DT Products (96GB × 8): two prefill nodes with PP2/TP8 and two decode nodes with global DP16/TP1. All nodes must mount the same checkpoint at the same path. Use `GLM-5.3-W8A8C8-mxfp8` by default; select `GLM-5.2-W8A8C8-mxfp8` on all nodes when deploying GLM-5.2.

    | Node | Role | Parallelism | Service/rank range |
    | ---- | ---- | ----------- | ------------------ |
    | node 0 | Prefill master | PP rank 0, TP8 | HTTP `9081` |
    | node 1 | Prefill worker | PP rank 1, TP8 | headless |
    | node 2 | Decode | DP ranks 0–7, TP1 | HTTP `9900`–`9907` |
    | node 3 | Decode | DP ranks 8–15, TP1 | HTTP `9900`–`9907` |

    Before you start, please

    **1. Prepare `launch_online_dp.py`.** Save it on node 2 and node 3. It starts one decode process per local NPU and assigns the global DP ranks requested on the command line.

    ```python
    import argparse
    import multiprocessing
    import os
    import subprocess
    import sys


    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dp-size", type=int, required=True)
        parser.add_argument("--tp-size", type=int, default=1)
        parser.add_argument("--pp-size", type=int, default=1)
        parser.add_argument("--dp-size-local", type=int, required=True)
        parser.add_argument("--dp-rank-start", type=int, required=True)
        parser.add_argument("--dp-address", type=str, required=True)
        parser.add_argument("--dp-rpc-port", type=str, default="12321")
        parser.add_argument("--vllm-start-port", type=int, default=9900)
        return parser.parse_args()


    def run_command(args, local_rank):
        devices_per_rank = args.tp_size * args.pp_size
        first_device = local_rank * devices_per_rank
        visible_devices = ",".join(
            str(device)
            for device in range(first_device, first_device + devices_per_rank)
        )
        dp_rank = args.dp_rank_start + local_rank
        engine_port = args.vllm_start_port + local_rank
        command = [
            "bash",
            "./run_dp_template.sh",
            visible_devices,
            str(engine_port),
            str(args.dp_size),
            str(dp_rank),
            args.dp_address,
            args.dp_rpc_port,
            str(args.tp_size),
            str(args.pp_size),
        ]
        subprocess.run(command, check=True)


    if __name__ == "__main__":
        parsed = parse_args()
        if not os.path.exists("./run_dp_template.sh"):
            sys.exit("./run_dp_template.sh does not exist")
        if parsed.dp_size_local * parsed.tp_size * parsed.pp_size > 8:
            sys.exit("The local DP/TP configuration requires more than 8 NPUs")

        processes = []
        for rank in range(parsed.dp_size_local):
            process = multiprocessing.Process(
                target=run_command, args=(parsed, rank)
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
            if process.exitcode != 0:
                sys.exit(process.exitcode)
    ```

    **2. Prepare `run_dp_template.sh`.** Save the role-specific version on each node.

    1. node 0 (Prefill master)

        ```shell
        #!/usr/bin/env bash
        set -e

        # this obtained through ifconfig
        # nic_name is the network interface name corresponding to local_ip of the current node
        nic_name="xxx"
        local_ip="xxx"

        # node 0 is the PP master node
        node0_ip=$local_ip

        unset ftp_proxy FTP_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
        unset HCCL_INTRA_ROCE_ENABLE
        export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
        mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
        export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_ALGO=level0:fullmesh
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export HCCL_BUFFSIZE=1200
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
        export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TASK_QUEUE_ENABLE=1
        export VLLM_PP_LAYER_PARTITION="42,36"
        export VLLM_USE_FASTOKENS=1


        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-W8A8C8-mxfp8 \
          --host 0.0.0.0 \
          --port 9081 \
          --api-server-count 1 \
          --tensor-parallel-size 8 \
          --pipeline-parallel-size 2 \
          --distributed-executor-backend mp \
          --master-addr "$node0_ip" \
          --master-port 7060 \
          --nnodes 2 \
          --node-rank 0 \
          --max-model-len 135000 \
          --max-num-batched-tokens 16384 \
          --max-num-seqs 500 \
          --served-model-name glm \
          --gpu-memory-utilization 0.9 \
          --enable-expert-parallel \
          --trust-remote-code \
          --enforce-eager \
          --quantization ascend \
          --enable-auto-tool-choice \
          --tool-call-parser glm47 \
          --reasoning-parser glm47 \
          --async-scheduling \
          --enable-prefix-caching \
          --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": true}' \
          --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":1,"tp_size":8,"pp_size":2,"pp_layer_partition":"42,36"},"decode":{"dp_size":16,"tp_size":1},"ascend_local_comm_res_path":"/etc/hixlep"}}' \
          --kv-cache-dtype fp8 \
          --attention_config.indexer_kv_dtype fp8 \
          --additional-config '{"enable_cpu_binding": true, "multistream_overlap_shared_expert": false, "recompute_scheduler_enable": true, "enable_dsa_cp": true, "enable_flashcomm1": true, "enable_fused_mc2": 1}'
        ```

    2. node 1 (Prefill worker)

        ```shell
        #!/usr/bin/env bash
        set -e

        # this obtained through ifconfig
        # nic_name is the network interface name corresponding to local_ip of the current node
        nic_name="xxx"
        local_ip="xxx"

        # The value of node0_ip must be consistent with the value of local_ip set in node 0 (PP master node)
        node0_ip="xxxx"

        unset ftp_proxy FTP_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
        unset HCCL_INTRA_ROCE_ENABLE
        export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
        mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
        export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_ALGO=level0:fullmesh
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export HCCL_BUFFSIZE=1200
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
        export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TASK_QUEUE_ENABLE=1
        export VLLM_PP_LAYER_PARTITION="42,36"
        export VLLM_USE_FASTOKENS=1


        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-W8A8C8-mxfp8 \
          --host 0.0.0.0 \
          --port 9081 \
          --headless \
          --tensor-parallel-size 8 \
          --pipeline-parallel-size 2 \
          --distributed-executor-backend mp \
          --master-addr "$node0_ip" \
          --master-port 7060 \
          --nnodes 2 \
          --node-rank 1 \
          --max-model-len 135000 \
          --max-num-batched-tokens 16384 \
          --max-num-seqs 500 \
          --served-model-name glm \
          --gpu-memory-utilization 0.9 \
          --enable-expert-parallel \
          --trust-remote-code \
          --enforce-eager \
          --quantization ascend \
          --enable-auto-tool-choice \
          --tool-call-parser glm47 \
          --reasoning-parser glm47 \
          --async-scheduling \
          --enable-prefix-caching \
          --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": true}' \
          --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":1,"tp_size":8,"pp_size":2,"pp_layer_partition":"42,36"},"decode":{"dp_size":16,"tp_size":1},"ascend_local_comm_res_path":"/etc/hixlep"}}' \
          --kv-cache-dtype fp8 \
          --attention_config.indexer_kv_dtype fp8 \
          --additional-config '{"enable_cpu_binding": true, "multistream_overlap_shared_expert": false, "recompute_scheduler_enable": true, "enable_dsa_cp": true, "enable_flashcomm1": true, "enable_fused_mc2": 1}'
        ```

    3. node 2 (Decode ranks 0–7)

        ```shell
        #!/usr/bin/env bash
        set -e

        VISIBLE_DEVICES=$1
        PORT=$2
        DP_SIZE=$3
        DP_RANK=$4
        DP_ADDRESS=$5
        DP_RPC_PORT=$6
        TP_SIZE=$7
        # this obtained through ifconfig
        # nic_name is the network interface name corresponding to local_ip of the current node
        nic_name="xxx"
        local_ip="xxx"

        unset ftp_proxy FTP_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
        unset HCCL_INTRA_ROCE_ENABLE
        export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
        mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
        export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
        export VLLM_ASCEND_DP_ALLREDUCE_ON_NPU=1
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_ALGO=level0:fullmesh
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export HCCL_BUFFSIZE=900
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES
        export VLLM_USE_FASTOKENS=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-W8A8C8-mxfp8 \
          --host 0.0.0.0 \
          --port "$PORT" \
          --data-parallel-size "$DP_SIZE" \
          --data-parallel-rank "$DP_RANK" \
          --data-parallel-address "$DP_ADDRESS" \
          --data-parallel-rpc-port "$DP_RPC_PORT" \
          --tensor-parallel-size "$TP_SIZE" \
          --max-model-len 135000 \
          --max-num-batched-tokens 60 \
          --max-num-seqs 10 \
          --served-model-name glm \
          --gpu-memory-utilization 0.95 \
          --enable-expert-parallel \
          --trust-remote-code \
          --quantization ascend \
          --enable-auto-tool-choice \
          --tool-call-parser glm47 \
          --reasoning-parser glm47 \
          --async-scheduling \
          --enable-prefix-caching \
          --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
          --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}' \
          --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"30300","engine_id":"3","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":1,"tp_size":8,"pp_size":2,"pp_layer_partition":"42,36"},"decode":{"dp_size":16,"tp_size":1},"ascend_local_comm_res_path":"/etc/hixlep"}}' \
          --kv-cache-dtype fp8 \
          --attention_config.indexer_kv_dtype fp8 \
          --additional-config '{"enable_cpu_binding": true, "ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}, "recompute_scheduler_enable":true, "multistream_overlap_shared_expert":true, "finegrained_tp_config":{"lmhead_tensor_parallel_size":8}, "enable_mlapo":true}'
        ```

    4. node 3 (Decode ranks 8–15)

        ```shell
        #!/usr/bin/env bash
        set -e

        VISIBLE_DEVICES=$1
        PORT=$2
        DP_SIZE=$3
        DP_RANK=$4
        DP_ADDRESS=$5
        DP_RPC_PORT=$6
        TP_SIZE=$7
        # this obtained through ifconfig
        # nic_name is the network interface name corresponding to local_ip of the current node
        nic_name="xxx"
        local_ip="xxx"

        unset ftp_proxy FTP_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
        unset HCCL_INTRA_ROCE_ENABLE
        export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
        mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
        export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
        export VLLM_ASCEND_DP_ALLREDUCE_ON_NPU=1
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_ALGO=level0:fullmesh
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export HCCL_BUFFSIZE=900
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES
        export VLLM_USE_FASTOKENS=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-W8A8C8-mxfp8 \
          --host 0.0.0.0 \
          --port "$PORT" \
          --data-parallel-size "$DP_SIZE" \
          --data-parallel-rank "$DP_RANK" \
          --data-parallel-address "$DP_ADDRESS" \
          --data-parallel-rpc-port "$DP_RPC_PORT" \
          --tensor-parallel-size "$TP_SIZE" \
          --max-model-len 135000 \
          --max-num-batched-tokens 60 \
          --max-num-seqs 10 \
          --served-model-name glm \
          --gpu-memory-utilization 0.95 \
          --enable-expert-parallel \
          --trust-remote-code \
          --quantization ascend \
          --enable-auto-tool-choice \
          --tool-call-parser glm47 \
          --reasoning-parser glm47 \
          --async-scheduling \
          --enable-prefix-caching \
          --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
          --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}' \
          --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"30300","engine_id":"3","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":1,"tp_size":8,"pp_size":2,"pp_layer_partition":"42,36"},"decode":{"dp_size":16,"tp_size":1},"ascend_local_comm_res_path":"/etc/hixlep"}}' \
          --kv-cache-dtype fp8 \
          --attention_config.indexer_kv_dtype fp8 \
          --additional-config '{"enable_cpu_binding": true, "ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}, "recompute_scheduler_enable":true, "multistream_overlap_shared_expert":true, "finegrained_tp_config":{"lmhead_tensor_parallel_size":8}, "enable_mlapo":true}'
        ```

    **3. Start the services.** Start node 0 and node 1 first, then start node 2 and node 3. The decode commands use node 2 as the global DP coordinator.

    **node 0 (Prefill master)**

    ```shell
    bash run_dp_template.sh
    ```

    **node 1 (Prefill worker)**

    ```shell
    bash run_dp_template.sh
    ```

    **node 2 (Decode ranks 0–7)**

    ```shell
    python launch_online_dp.py \
      --dp-size 16 --tp-size 1 --pp-size 1 --dp-size-local 8 --dp-rank-start 0 \
      --dp-address <node-2-data-ip> --dp-rpc-port 12321 --vllm-start-port 9900
    ```

    **node 3 (Decode ranks 8–15)**

    ```shell
    python launch_online_dp.py \
      --dp-size 16 --tp-size 1 --pp-size 1 --dp-size-local 8 --dp-rank-start 8 \
      --dp-address <node-2-data-ip> --dp-rpc-port 12321 --vllm-start-port 9900
    ```

    **4. Start the proxy.** To set up request forwarding, run the following script on any machine. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py). Register all sixteen decode endpoints; node 1 is a headless PP worker and is not a separate HTTP endpoint.

    ```shell
    node_p0_ip="<node-0-data-ip>"
    node_d0_ip="<node-2-data-ip>"
    node_d1_ip="<node-3-data-ip>"

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
          $node_d1_ip \
          $node_d1_ip \
          $node_d1_ip \
          $node_d1_ip \
          $node_d1_ip \
          $node_d1_ip \
          $node_d1_ip \
          $node_d1_ip \
        --decoder-ports \
          9900 9901 9902 9903 9904 9905 9906 9907 \
          9900 9901 9902 9903 9904 9905 9906 9907
    ```

    Send a text request to the proxy:

    ```shell
    curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"glm","messages":[{"role":"user","content":"What is 1 + 1?"}],"max_tokens":128,"stream":false}'
    ```

    Initialize the installed CANN environment on every node before launching the scripts. The source PD configuration uses CANN 9.2.0. The commands use the GLM-5.3 checkpoint path directly. For GLM-5.2, replace it with `/root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-W8A8C8-mxfp8` on all four nodes before starting either script. These adapted commands have been checked for syntax, but have not been revalidated end to end in this documentation update.

    Profiling and compile-cache debugging are optional. To temporarily disable the compile cache, set `VLLM_DISABLE_COMPILE_CACHE=1`; unset it for normal deployment.

=== "A3 Series"

    This A3 reference serves GLM-5.3 W8A8C8 with a **200,000-token** maximum
    sequence length for the 198K workload. It uses one Prefill engine spanning
    P0/P1 with TP16 and PP2, and sixteen TP2 Decode engines on D0/D1.
    Section 5.1.4 adds MemCache KV pooling to the same deployment.

    ##### 5.1.3.1 Topology and Prerequisites

    | Role | Nodes | Global DP | TP | PP | Placement | Devices per role |
    | --- | --- | --- | --- | --- | --- | --- |
    | Prefill | P0, P1 | 1 | 16 | 2 | One PP stage per node | 32 |
    | Decode | D0, D1 | 16 | 2 | 1 | Eight DP ranks per node | 32 |

    Each container exposes devices 0-15. P0 exposes the only Prefill API;
    P1 runs the second PP stage without an API server. Decode ranks 0-7 run
    on D0 and ranks 8-15 on D1, with API ports 8000-8007 on each node.

    Use the A3 `v0.30.0rc1-a3` image from Section 4 on all four nodes.

    1. Prepare the same weights and software on all four nodes. Set `MODEL_PATH`
       and `PYTHON_LIB_DIR` in each role script to the installed paths. Install
       `fastokens` on Prefill nodes for `VLLM_USE_FASTOKENS=1`.
    2. Verify [multi-node communication](../../getting_started/installation.md#installation-multi-node-interconnect).
       Set `LOCAL_IP="<PREFILL_NODE_IP>"` on each Prefill node and
       `LOCAL_IP="<DECODE_NODE_IP>"` on each Decode node to that node's IP.
       Set `NIC_NAME="<NETWORK_INTERFACE>"` to its matching interface and
       `MODEL_PATH="<YOUR_MODEL_PATH>"` to its model weight directory.
    3. Reserve the PP master port 7060, Decode DP RPC port 16600, API ports,
       proxy port 8000, and the topology-derived Mooncake transfer/handshake ports.
       Prefill and Decode run on separate node groups.
    4. Keep the 78-layer PP partition `42,36` consistent in the Prefill
       environment and both roles' connector topology. Both roles explicitly
       use int8 KV cache and int8 indexer KV cache for these W8A8C8 weights.

    ##### 5.1.3.2 Prepare the Scripts

    Before you start, please

    1. prepare the script `launch_online_dp.py` on each Decode node:

        ```python
        import argparse
        import multiprocessing
        import subprocess
        import sys


        def run_command(args, local_rank):
            dp_rank = args.dp_rank_start + local_rank
            devices = ",".join(str(i) for i in range(
                local_rank * args.tp_size, (local_rank + 1) * args.tp_size))
            subprocess.run([
                "bash", args.script, devices, str(args.vllm_start_port + local_rank),
                str(args.dp_size), str(dp_rank), args.dp_address,
                str(args.dp_rpc_port), str(args.tp_size),
            ], check=True)


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--dp-size", type=int, required=True)
            parser.add_argument("--tp-size", type=int, required=True)
            parser.add_argument("--dp-size-local", type=int, required=True)
            parser.add_argument("--dp-rank-start", type=int, required=True)
            parser.add_argument("--dp-address", required=True)
            parser.add_argument("--dp-rpc-port", type=int, default=16600)
            parser.add_argument("--vllm-start-port", type=int, default=8000)
            parser.add_argument("--script", default="./run_dp_template.sh")
            args = parser.parse_args()
            if min(args.dp_size, args.tp_size, args.dp_size_local) <= 0:
                parser.error("Parallel sizes must be positive")
            if args.dp_rank_start < 0 or args.dp_rank_start + args.dp_size_local > args.dp_size:
                parser.error("Local ranks must be within the global DP group")
            if args.dp_size_local * args.tp_size > 16:
                parser.error("This A3 configuration exposes 16 devices per node")
            processes = []
            for local_rank in range(args.dp_size_local):
                process = multiprocessing.Process(target=run_command, args=(args, local_rank))
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
            return 1 if any(process.exitcode != 0 for process in processes) else 0


        if __name__ == "__main__":
            sys.exit(main())
        ```

    2. prepare the script `run_dp_template.sh` on each node.

        1. Prefill node 0

            ```shell
            #!/usr/bin/env bash

            LOCAL_IP="<PREFILL_NODE_IP>"
            NIC_NAME="<NETWORK_INTERFACE>"
            MODEL_PATH="<YOUR_MODEL_PATH>"
            PYTHON_LIB_DIR=/path/to/python/lib

            # Set the current node IP and shared P0 master IP before starting.

            export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
            export HCCL_EXEC_TIMEOUT=1800
            export HCCL_CONNECT_TIMEOUT=1800
            export ASCEND_TRANSFER_TIMEOUT=10000

            NODE_P0_IP="$LOCAL_IP"

            export VLLM_HOST_IP="$LOCAL_IP"
            export HCCL_IF_IP="$LOCAL_IP"
            export GLOO_SOCKET_IFNAME="$NIC_NAME"
            export TP_SOCKET_IFNAME="$NIC_NAME"
            export HCCL_SOCKET_IFNAME="$NIC_NAME"

            export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

            export HCCL_BUFFSIZE=1024
            export HCCL_OP_EXPANSION_MODE="AIV"

            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export TASK_QUEUE_ENABLE=1

            export VLLM_USE_V2_MODEL_RUNNER=0

            export VLLM_USE_FASTOKENS=1
            export VLLM_PP_LAYER_PARTITION="42,36"

            export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

            exec vllm serve "${MODEL_PATH}" \
                --host 0.0.0.0 \
                --port 8000 \
                --tensor-parallel-size 16 \
                --enable-expert-parallel \
                --pipeline-parallel-size 2 \
                --distributed-executor-backend mp \
                --master-addr "$NODE_P0_IP" \
                --master-port 7060 \
                --nnodes 2 \
                --node-rank 0 \
                --enable-chunked-prefill \
                --enable-prefix-caching \
                --seed 1024 \
                --served-model-name glm5 \
                --max-model-len 200000 \
                --max-num-batched-tokens 16384 \
                --trust-remote-code \
                --max-num-seqs 64 \
                --gpu-memory-utilization 0.92 \
                --async-scheduling \
                --quantization ascend \
                --safetensors-load-strategy 'prefetch' \
                --enable-auto-tool-choice \
                --tool-call-parser glm47 \
                --reasoning-parser glm47 \
                --enforce-eager \
                --kv-cache-dtype int8 \
                --attention_config.indexer_kv_dtype int8 \
                --additional-config '{
                    "enable_dsa_cp":true,
                    "enable_fused_mc2": 1,
                    "enable_flashcomm1": true
                }' \
                --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
                --kv-transfer-config '{
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_producer",
                    "kv_port": "30000",
                    "engine_id": "glm53-prefill",
                    "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                        "decode": {"dp_size": 16, "tp_size": 2}
                    }
                }'
            ```

        2. Prefill node 1

            ```shell
            #!/usr/bin/env bash

            LOCAL_IP="<PREFILL_NODE_IP>"
            NIC_NAME="<NETWORK_INTERFACE>"
            MODEL_PATH="<YOUR_MODEL_PATH>"
            PYTHON_LIB_DIR=/path/to/python/lib

            # Set the current node IP and shared P0 master IP before starting.

            export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
            export HCCL_EXEC_TIMEOUT=1800
            export HCCL_CONNECT_TIMEOUT=1800
            export ASCEND_TRANSFER_TIMEOUT=10000

            NODE_P0_IP="<PREFILL_NODE0_IP>"

            export VLLM_HOST_IP="$LOCAL_IP"
            export HCCL_IF_IP="$LOCAL_IP"
            export GLOO_SOCKET_IFNAME="$NIC_NAME"
            export TP_SOCKET_IFNAME="$NIC_NAME"
            export HCCL_SOCKET_IFNAME="$NIC_NAME"

            export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

            export HCCL_BUFFSIZE=1024
            export HCCL_OP_EXPANSION_MODE="AIV"

            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export TASK_QUEUE_ENABLE=1

            export VLLM_USE_V2_MODEL_RUNNER=0

            export VLLM_USE_FASTOKENS=1
            export VLLM_PP_LAYER_PARTITION="42,36"

            export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

            exec vllm serve "${MODEL_PATH}" \
                --host 0.0.0.0 \
                --port 8000 \
                --tensor-parallel-size 16 \
                --enable-expert-parallel \
                --pipeline-parallel-size 2 \
                --distributed-executor-backend mp \
                --master-addr "$NODE_P0_IP" \
                --master-port 7060 \
                --nnodes 2 \
                --node-rank 1 \
                --enable-chunked-prefill \
                --enable-prefix-caching \
                --seed 1024 \
                --served-model-name glm5 \
                --max-model-len 200000 \
                --max-num-batched-tokens 16384 \
                --trust-remote-code \
                --max-num-seqs 64 \
                --gpu-memory-utilization 0.92 \
                --async-scheduling \
                --quantization ascend \
                --safetensors-load-strategy 'prefetch' \
                --enable-auto-tool-choice \
                --tool-call-parser glm47 \
                --reasoning-parser glm47 \
                --enforce-eager \
                --kv-cache-dtype int8 \
                --attention_config.indexer_kv_dtype int8 \
                --additional-config '{
                    "enable_dsa_cp":true,
                    "enable_fused_mc2": 1,
                    "enable_flashcomm1": true
                }' \
                --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
                --kv-transfer-config '{
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_producer",
                    "kv_port": "30000",
                    "engine_id": "glm53-prefill",
                    "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                        "decode": {"dp_size": 16, "tp_size": 2}
                    }
                }' \
                --headless
            ```

        3. Decode node 0

            ```shell
            #!/usr/bin/env bash

            LOCAL_IP="<DECODE_NODE_IP>"
            NIC_NAME="<NETWORK_INTERFACE>"
            MODEL_PATH="<YOUR_MODEL_PATH>"
            PYTHON_LIB_DIR=/path/to/python/lib

            # Arguments $1-$7 are supplied by launch_online_dp.py.

            export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
            export HCCL_EXEC_TIMEOUT=1800
            export HCCL_CONNECT_TIMEOUT=1800
            export ASCEND_TRANSFER_TIMEOUT=10000


            export VLLM_HOST_IP="$LOCAL_IP"
            export HCCL_IF_IP="$LOCAL_IP"
            export GLOO_SOCKET_IFNAME="$NIC_NAME"
            export TP_SOCKET_IFNAME="$NIC_NAME"
            export HCCL_SOCKET_IFNAME="$NIC_NAME"

            export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

            export HCCL_BUFFSIZE=1024
            export HCCL_OP_EXPANSION_MODE="AIV"

            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export TASK_QUEUE_ENABLE=1

            export VLLM_USE_V2_MODEL_RUNNER=0
            export ASCEND_RT_VISIBLE_DEVICES=$1
            export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

            exec vllm serve "${MODEL_PATH}" \
                --host 0.0.0.0 \
                --port $2 \
                --data-parallel-size $3 \
                --data-parallel-rank $4 \
                --data-parallel-address $5 \
                --data-parallel-rpc-port $6 \
                --tensor-parallel-size $7 \
                --enable-expert-parallel \
                --enable-chunked-prefill \
                --enable-prefix-caching \
                --seed 1024 \
                --served-model-name glm5 \
                --max-model-len 200000 \
                --max-num-batched-tokens 256 \
                --trust-remote-code \
                --max-num-seqs 64 \
                --gpu-memory-utilization 0.92 \
                --async-scheduling \
                --quantization ascend \
                --safetensors-load-strategy 'prefetch' \
                --enable-auto-tool-choice \
                --tool-call-parser glm47 \
                --reasoning-parser glm47 \
                --kv-cache-dtype int8 \
                --attention_config.indexer_kv_dtype int8 \
                --additional-config '{
                    "recompute_scheduler_enable": true,
                    "enable_fused_mc2": 1
                }' \
                --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
                --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
                --kv-transfer-config '{
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_consumer",
                    "kv_port": "30100",
                    "engine_id": "glm53-decode-dp'"$4"'",
                    "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                        "decode": {"dp_size": 16, "tp_size": 2}
                    }
                }'
            ```

        4. Decode node 1

            ```shell
            #!/usr/bin/env bash

            LOCAL_IP="<DECODE_NODE_IP>"
            NIC_NAME="<NETWORK_INTERFACE>"
            MODEL_PATH="<YOUR_MODEL_PATH>"
            PYTHON_LIB_DIR=/path/to/python/lib

            # Arguments $1-$7 are supplied by launch_online_dp.py.

            export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
            export HCCL_EXEC_TIMEOUT=1800
            export HCCL_CONNECT_TIMEOUT=1800
            export ASCEND_TRANSFER_TIMEOUT=10000


            export VLLM_HOST_IP="$LOCAL_IP"
            export HCCL_IF_IP="$LOCAL_IP"
            export GLOO_SOCKET_IFNAME="$NIC_NAME"
            export TP_SOCKET_IFNAME="$NIC_NAME"
            export HCCL_SOCKET_IFNAME="$NIC_NAME"

            export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

            export HCCL_BUFFSIZE=1024
            export HCCL_OP_EXPANSION_MODE="AIV"

            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export TASK_QUEUE_ENABLE=1

            export VLLM_USE_V2_MODEL_RUNNER=0
            export ASCEND_RT_VISIBLE_DEVICES=$1
            export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

            exec vllm serve "${MODEL_PATH}" \
                --host 0.0.0.0 \
                --port $2 \
                --data-parallel-size $3 \
                --data-parallel-rank $4 \
                --data-parallel-address $5 \
                --data-parallel-rpc-port $6 \
                --tensor-parallel-size $7 \
                --enable-expert-parallel \
                --enable-chunked-prefill \
                --enable-prefix-caching \
                --seed 1024 \
                --served-model-name glm5 \
                --max-model-len 200000 \
                --max-num-batched-tokens 256 \
                --trust-remote-code \
                --max-num-seqs 64 \
                --gpu-memory-utilization 0.92 \
                --async-scheduling \
                --quantization ascend \
                --safetensors-load-strategy 'prefetch' \
                --enable-auto-tool-choice \
                --tool-call-parser glm47 \
                --reasoning-parser glm47 \
                --kv-cache-dtype int8 \
                --attention_config.indexer_kv_dtype int8 \
                --additional-config '{
                    "recompute_scheduler_enable": true,
                    "enable_fused_mc2": 1
                }' \
                --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
                --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
                --kv-transfer-config '{
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_consumer",
                    "kv_port": "30100",
                    "engine_id": "glm53-decode-dp'"$4"'",
                    "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                        "decode": {"dp_size": 16, "tp_size": 2}
                    }
                }'
            ```

    ##### 5.1.3.3 Start the Engines

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
        D0_IP="<DECODE_NODE0_IP>"
        python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
        ```

    4. Decode node 1

        ```shell
        D0_IP="<DECODE_NODE0_IP>"
        python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 8 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
        ```

    ##### 5.1.3.4 Start the Proxy

    Replace each node's IP and installation-path placeholders before starting.
    Run the four commands in Section 5.1.3.3 in separate terminals. Keep the
    master processes running while the other nodes join; start both nodes of
    each group before waiting for the group to become healthy.

    After the Prefill API on P0:8000 and all sixteen Decode APIs respond
    to `GET /v1/models`, start the
    [PD load-balancing proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)
    from the matching checkout on a separate host that can reach all P/D
    API endpoints. The proxy listens on port 8000; using a separate host
    avoids a port conflict with P0:8000 and the Decode APIs on 8000-8007.
    Register P0 once and all sixteen Decode endpoints. P1 is not a separate
    Prefill API endpoint.

    ```shell
    P0_IP="<P0_IP>"
    D0_IP="<D0_IP>"
    D1_IP="<D1_IP>"

    python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 8000 \
        --prefiller-hosts "$P0_IP" --prefiller-ports 8000 \
        --decoder-hosts "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
            "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
            "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
            "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
        --decoder-ports 8000 8001 8002 8003 8004 8005 8006 8007 \
            8000 8001 8002 8003 8004 8005 8006 8007
    ```

    Key parameters:

    - Prefill uses TP16/PP2 over two nodes with the `mp` executor and
      `VLLM_PP_LAYER_PARTITION="42,36"`. Its batch budget is 16,384 tokens,
      maximum sequence count is 64, and it runs in eager mode with one MTP token.
    - Decode uses DP16/TP2, a 256-token batch budget, 64 sequences, five MTP
      tokens with an eager draft model, and `FULL_DECODE_ONLY` graph mode.
      `recompute_scheduler_enable=true` allows recomputation scheduling.
    - Both roles retain chunked prefill, prefix caching, async scheduling,
      seed 1024, memory utilization 0.92, and fused MC2. Prefill also enables
      DSA CP and FlashComm1. Both use `glm47` tool and reasoning parsers and
      the served model name `glm5`.
    - `MooncakeConnectorV2` declares the same Prefill DP1/PP2/TP16 and
      Decode DP16/TP2 topology on both sides. The non-pooled base KV ports
      are 30000 on Prefill and 30100 on Decode; reserve derived worker ports.
    - The examples give Prefill and each Decode engine distinct `engine_id`
      values so P-to-D peer registration cannot conflict.

#### 5.1.4 MemCache KV Cache Pool Deployment

This extends the same four-node PP/PD topology with `MultiConnector`:
`MooncakeConnectorV2` transfers KV cache from Prefill to Decode, and
`AscendStoreConnector` provides non-layerwise MemCache pooling. Retain
the launcher, PP placement, and proxy routing from Section 5.1.3.

##### 5.1.4.1 Prepare MemCache

Complete [MemCache backend setup](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend),
including the hardware/CANN prerequisites, `memfabric-hybrid` and
`memcache-hybrid` installation, configuration files, and metadata service.
All P/D instances must access the same pool; connector arguments do not
start the MemCache service. Set `MEMCACHE_ROOT` to the installed
`memcache_hybrid` directory and `PYTHON_LIB_DIR` to the library directory
of the Python installation actually used by vLLM on each node.

##### 5.1.4.2 Prepare the Scripts

Before you start, please

1. prepare the script `launch_online_dp.py` on each Decode node, using the
   launcher in Section 5.1.3.2.

2. prepare the pooled script `run_dp_template.sh` on each node.

    1. Prefill node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid

        # Set the current node IP and shared P0 master IP before starting.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000

        NODE_P0_IP="$LOCAL_IP"

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export VLLM_USE_FASTOKENS=1
        export VLLM_PP_LAYER_PARTITION="42,36"

        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr "$NODE_P0_IP" \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 0 \
            --enable-chunked-prefill \
            --async-scheduling \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --enforce-eager \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "enable_dsa_cp":true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "glm53-prefill",
                "kv_connector_extra_config": {
                    "connectors":[
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_producer",
                        "kv_port": "30000",
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                            "decode": { "dp_size": 16, "tp_size": 2}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_producer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port":"0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                    ]
                }
            }'
        ```

    2. Prefill node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid

        # Set the current node IP and shared P0 master IP before starting.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000

        NODE_P0_IP="<PREFILL_NODE0_IP>"

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export VLLM_USE_FASTOKENS=1
        export VLLM_PP_LAYER_PARTITION="42,36"

        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr "$NODE_P0_IP" \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 1 \
            --enable-chunked-prefill \
            --async-scheduling \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --enforce-eager \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "enable_dsa_cp":true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "engine_id": "glm53-prefill",
                "kv_connector_extra_config": {
                    "connectors":[
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_producer",
                        "kv_port": "30000",
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                            "decode": { "dp_size": 16, "tp_size": 2}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_producer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port":"0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                    ]
                }
            }' \
            --headless
        ```

    3. Decode node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid

        # Arguments $1-$7 are supplied by launch_online_dp.py.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000


        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export ASCEND_RT_VISIBLE_DEVICES=$1

        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 256 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_fused_mc2": 1
            }' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_consumer",
                "kv_port": "30200",
                "engine_id": "glm53-decode-dp'"$4"'",
                "kv_connector_extra_config": {
                    "connectors":[
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_consumer",
                        "kv_port": "30200",
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                            "decode": { "dp_size": 16, "tp_size": 2}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_consumer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port":"0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                    ]
                }
            }'
        ```

    4. Decode node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"
        PYTHON_LIB_DIR=/path/to/python/lib
        MEMCACHE_ROOT=/path/to/site-packages/memcache_hybrid

        # Arguments $1-$7 are supplied by launch_online_dp.py.

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=1800
        export HCCL_CONNECT_TIMEOUT=1800
        export ASCEND_TRANSFER_TIMEOUT=10000


        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH}"

        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE="AIV"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        export VLLM_USE_V2_MODEL_RUNNER=0

        export ASCEND_RT_VISIBLE_DEVICES=$1

        export PYTHONHASHSEED=0
        export ACL_OP_INIT_MODE=1
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
        export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

        export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

        exec vllm serve "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --seed 1024 \
            --served-model-name glm5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 256 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --safetensors-load-strategy 'prefetch' \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_fused_mc2": 1
            }' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_consumer",
                "kv_port": "30200",
                "engine_id": "glm53-decode-dp'"$4"'",
                "kv_connector_extra_config": {
                    "connectors":[
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_consumer",
                        "kv_port": "30200",
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                            "decode": { "dp_size": 16, "tp_size": 2}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_consumer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port":"0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                    ]
                }
            }'
        ```

The pooled configuration uses base KV ports 30000 on Prefill and
30200 on Decode. The outer `engine_id` is unique per engine and is
inherited by the child connectors. This differs from the supplied pooled
reference's repeated `"0"` IDs to avoid Mooncake peer identity conflicts.
`lookup_rpc_port="0"` is retained: Ascend Store uses an IPC endpoint that
also includes the DP rank, and P/D run on separate hosts. It is not a TCP
listener on port zero. `backend="memcache"` and `use_layerwise=false`
are set on both sides.

##### 5.1.4.3 Start the Engines

Start the MemCache metadata service and complete its configuration on all
four nodes before starting vLLM.

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
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

4. Decode node 1

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 8 --dp-rank-start 8 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

##### 5.1.4.4 Start the Proxy and Verify Pooling

1. Start the MemCache metadata service and complete its configuration on
   all four nodes before starting vLLM.
2. Start P0, P1, D0, and D1 with the commands in
   Section 5.1.4.3. Both Prefill scripts already specify port 8000.
3. Wait for P0:8000 and all sixteen Decode APIs to become healthy. Start
   the proxy from Section 5.1.3.4 on a separate host, using the same
   Prefill endpoint and Decode ports 8000-8007. Its public port is 8000.
4. Send requests to the proxy using the served model name `glm5`:

```shell
curl "http://<proxy_ip>:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm5",
        "messages": [{"role": "user", "content": "Explain prefill and decode in an LLM."}],
        "max_tokens": 128,
        "temperature": 0
    }'
```

For either mode, check P/D logs for successful KV transfer. For pooling,
also check MemCache initialization, then warm the pool with a prompt longer
than one cache block and repeat requests with the same prefix. Inspect
Ascend Store/MemCache saves, lookups, loads, and hit metrics. A successful
response or a local prefix-cache hit alone does not demonstrate shared pool
reuse; verify a pool load when the prefix is absent from the engine's local
cache.  These references do not
establish A2, 1M-context, or other parallel-layout support, and do not add
accuracy or performance claims for the 198K workload.

### 5.2 1M Context Configuration

#### 5.2.1 1M Prefill-Decode Disaggregation (DCP)

This configuration uses GLM-5.3 W8A8C8 weights on four Atlas 800I A3
nodes, with 16 logical NPU devices per node. Use the A3 image from
[Docker Image Installation](#4-installation) on all nodes.
The maximum context length is 1,048,576 tokens, including input and output.

##### 5.2.1.1 Topology and Prerequisites

| Role | Nodes | Global DP | TP | PP | DCP | Local DP ranks | Devices per role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Prefill | P0, P1 | 2 | 16 | 1 | 16 | One per node | 32 |
| Decode | D0, D1 | 4 | 8 | 1 | 8 | Two per node | 32 |

P0 and P1 run DP ranks 0 and 1. D0 runs ranks 0-1 and D1 runs ranks 2-3.
Prefill context parallel size is 1 on both roles. This DCP layout is
separate from the PP2 layout in Sections 5.1.2 and 5.1.3.

1. Prepare identical weights on all four nodes and verify
   [multi-node communication](../../getting_started/installation.md#installation-multi-node-interconnect).
2. Set `LOCAL_IP`, `NIC_NAME`, and `MODEL_PATH` separately on each node.
   Replace the master address placeholders in the startup commands with
   P0's IP for Prefill and D0's IP for Decode.
3. Reserve DP RPC ports 16591 and 16600, API port 8000 on each Prefill
   node, API ports 8000-8001 on each Decode node, and Mooncake transfer
   ports derived from the configured base ports 30000 and 30200.
   Run the public proxy on a separate host at port 8000.

##### 5.2.1.2 Prepare the Scripts

Before you start, please

1. prepare the script `launch_online_dp.py` on each node:

    ```python
    import argparse
    import multiprocessing
    import subprocess
    import sys


    def run_command(args, local_rank):
        dp_rank = args.dp_rank_start + local_rank
        devices = ",".join(str(i) for i in range(
            local_rank * args.tp_size, (local_rank + 1) * args.tp_size))
        subprocess.run([
            "bash", args.script, devices, str(args.vllm_start_port + local_rank),
            str(args.dp_size), str(dp_rank), args.dp_address,
            str(args.dp_rpc_port), str(args.tp_size),
        ], check=True)


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dp-size", type=int, required=True)
        parser.add_argument("--tp-size", type=int, required=True)
        parser.add_argument("--dp-size-local", type=int, required=True)
        parser.add_argument("--dp-rank-start", type=int, required=True)
        parser.add_argument("--dp-address", required=True)
        parser.add_argument("--dp-rpc-port", type=int, default=16600)
        parser.add_argument("--vllm-start-port", type=int, default=8000)
        parser.add_argument("--script", default="./run_dp_template.sh")
        args = parser.parse_args()
        if min(args.dp_size, args.tp_size, args.dp_size_local) <= 0:
            parser.error("Parallel sizes must be positive")
        if args.dp_rank_start < 0 or args.dp_rank_start + args.dp_size_local > args.dp_size:
            parser.error("Local ranks must be within the global DP group")
        if args.dp_size_local * args.tp_size > 16:
            parser.error("This A3 configuration exposes 16 devices per node")
        processes = []
        for local_rank in range(args.dp_size_local):
            process = multiprocessing.Process(target=run_command, args=(args, local_rank))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        return 1 if any(process.exitcode != 0 for process in processes) else 0


    if __name__ == "__main__":
        sys.exit(main())
    ```

2. prepare the script `run_dp_template.sh` on each node.

    1. Prefill node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"

        export HCCL_OP_EXPANSION_MODE="AIV"
        export VLLM_ENGINE_READY_TIMEOUT_S=100000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=7200
        export ASCEND_CONNECT_TIMEOUT=10000
        export ASCEND_TRANSFER_TIMEOUT=10000
        export MF_GROUP_JOIN_MAX_TIMEOUT=1200

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES="$1"

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        export VLLM_USE_V2_MODEL_RUNNER=0

        exec vllm serve "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --enable-expert-parallel \
            --prefill-context-parallel-size 1 \
            --decode-context-parallel-size 16 \
            --cp-kv-cache-interleave-size 128 \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 1048576 \
            --safetensors-load-strategy 'prefetch' \
            --additional-config '{
                "enable_dsa_cp": true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --max-num-batched-tokens 8192 \
            --trust-remote-code \
            --enable-prefix-caching \
            --max-num-seqs 64 \
            --quantization ascend \
            --gpu-memory-utilization 0.85 \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV1",
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
                        "dp_size": 4,
                        "tp_size": 8
                    }
                }
            }'
        ```

    2. Prefill node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<PREFILL_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"

        export HCCL_OP_EXPANSION_MODE="AIV"
        export VLLM_ENGINE_READY_TIMEOUT_S=100000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=7200
        export ASCEND_CONNECT_TIMEOUT=10000
        export ASCEND_TRANSFER_TIMEOUT=10000
        export MF_GROUP_JOIN_MAX_TIMEOUT=1200

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES="$1"

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        export VLLM_USE_V2_MODEL_RUNNER=0

        exec vllm serve "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --enable-expert-parallel \
            --prefill-context-parallel-size 1 \
            --decode-context-parallel-size 16 \
            --cp-kv-cache-interleave-size 128 \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 1048576 \
            --safetensors-load-strategy 'prefetch' \
            --additional-config '{
                "enable_dsa_cp": true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true
            }' \
            --max-num-batched-tokens 8192 \
            --trust-remote-code \
            --enable-prefix-caching \
            --max-num-seqs 64 \
            --quantization ascend \
            --gpu-memory-utilization 0.85 \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV1",
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
                        "dp_size": 4,
                        "tp_size": 8
                    }
                }
            }'
        ```

    3. Decode node 0

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"

        export HCCL_OP_EXPANSION_MODE="AIV"
        export VLLM_ENGINE_READY_TIMEOUT_S=100000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=7200
        export ASCEND_CONNECT_TIMEOUT=10000
        export ASCEND_TRANSFER_TIMEOUT=10000
        export MF_GROUP_JOIN_MAX_TIMEOUT=1200

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES="$1"

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        export VLLM_USE_V2_MODEL_RUNNER=0

        exec vllm serve "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
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
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_mlapo": true,
                "enable_fused_mc2": 1
            }' \
            --trust-remote-code \
            --max-num-seqs 32 \
            --gpu-memory-utilization 0.90 \
            --async-scheduling \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_consumer",
                "kv_port": "30200",
                "engine_id": "2",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {
                        "dp_size": 2,
                        "tp_size": 16
                    },
                    "decode": {
                        "dp_size": 4,
                        "tp_size": 8
                    }
                }
            }'
        ```

    4. Decode node 1

        ```shell
        #!/usr/bin/env bash

        LOCAL_IP="<DECODE_NODE_IP>"
        NIC_NAME="<NETWORK_INTERFACE>"
        MODEL_PATH="<YOUR_MODEL_PATH>"

        export HCCL_OP_EXPANSION_MODE="AIV"
        export VLLM_ENGINE_READY_TIMEOUT_S=100000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=7200
        export ASCEND_CONNECT_TIMEOUT=10000
        export ASCEND_TRANSFER_TIMEOUT=10000
        export MF_GROUP_JOIN_MAX_TIMEOUT=1200

        export VLLM_HOST_IP="$LOCAL_IP"
        export HCCL_IF_IP="$LOCAL_IP"
        export GLOO_SOCKET_IFNAME="$NIC_NAME"
        export TP_SOCKET_IFNAME="$NIC_NAME"
        export HCCL_SOCKET_IFNAME="$NIC_NAME"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES="$1"

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        export VLLM_USE_V2_MODEL_RUNNER=0

        exec vllm serve "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
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
            --additional-config '{
                "recompute_scheduler_enable": true,
                "enable_mlapo": true,
                "enable_fused_mc2": 1
            }' \
            --trust-remote-code \
            --max-num-seqs 32 \
            --gpu-memory-utilization 0.90 \
            --async-scheduling \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_consumer",
                "kv_port": "30200",
                "engine_id": "2",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {
                        "dp_size": 2,
                        "tp_size": 16
                    },
                    "decode": {
                        "dp_size": 4,
                        "tp_size": 8
                    }
                }
            }'
        ```

##### 5.2.1.3 Start the Engines

Run these commands from the directory containing both scripts on each node.
The two Prefill commands use the same P0 master address, and the two Decode
commands use the same D0 master address.

1. Prefill node 0

    ```shell
    P0_IP="<PREFILL_NODE0_IP>"
    python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 0 --dp-address "$P0_IP" --dp-rpc-port 16591 --vllm-start-port 8000
    ```

2. Prefill node 1

    ```shell
    P0_IP="<PREFILL_NODE0_IP>"
    python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 1 --dp-address "$P0_IP" --dp-rpc-port 16591 --vllm-start-port 8000
    ```

3. Decode node 0

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

4. Decode node 1

    ```shell
    D0_IP="<DECODE_NODE0_IP>"
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 2 --dp-rank-start 2 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
    ```

##### 5.2.1.4 Start the Proxy

After all six APIs become healthy, start the proxy on its separate host:

```shell
P0_IP="<P0_IP>"
P1_IP="<P1_IP>"
D0_IP="<D0_IP>"
D1_IP="<D1_IP>"

python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 --port 8000 \
    --prefiller-hosts "$P0_IP" "$P1_IP" --prefiller-ports 8000 8000 \
    --decoder-hosts "$D0_IP" "$D0_IP" "$D1_IP" "$D1_IP" \
    --decoder-ports 8000 8001 8000 8001
```

Send requests to `http://<PROXY_IP>:8000` using the served model name `glm-5`.

Key Parameter Descriptions:

- The launcher passes `$1` = visible devices, `$2` = API port, `$3` = global
  DP size, `$4` = DP rank, `$5` = DP master IP, `$6` = DP RPC port, and
  `$7` = TP size to `run_dp_template.sh`.
- `--decode-context-parallel-size`: 16 on Prefill and 8 on Decode;
  `--prefill-context-parallel-size` remains 1 and
  `--cp-kv-cache-interleave-size` is 128 on both roles.
- `--max-num-batched-tokens`: 8192 on Prefill and 192 on Decode.
  `--max-num-seqs`: 64 on Prefill and 32 on Decode. These are scheduler
  limits; start the 1M workload with one concurrent request and increase
  concurrency according to available KV cache capacity.
- MTP uses `deepseek_mtp` with an eager draft: 1 speculative token on
  Prefill and 5 on Decode. Decode uses `FULL_DECODE_ONLY` graph mode.
- `MooncakeConnectorV1` uses `kv_producer` on Prefill and `kv_consumer`
  on Decode, with the same `prefill: DP2/TP16` and `decode: DP4/TP8`
  topology on both roles. This example does not enable MemCache pooling.
- Both roles use int8 KV and indexer KV caches. Prefill enables DSA CP,
  fused MC2 and FlashComm1; Decode enables fused MC2, MLAPO and the
  recomputation scheduler. See [Performance Tuning](#9-performance-tuning).
- `--kv-cache-dtype turboquant_4bit_nc`: Enables 4-bit TurboQuant
  quantization for the SFA packed main KV cache, as an alternative to
  `--kv-cache-dtype int8`. It requires an A2/A3 device, cannot be
  combined with the C8 cache, and in Prefill-Decode disaggregation the
  Mooncake option `"use_ascend_direct"` must be `false`.

## 6 Functional Verification

All serving commands on this page use `--served-model-name glm`, including GLM-5.3. For colocated deployment, use the API-facing node and its configured HTTP port. For PD deployment, use the proxy host and port `8000`; do not send client requests to a headless worker.

```shell
# Replace with the API server URL, or http://<proxy_ip>:8000 for PD.
BASE_URL="http://<node0_ip>:<port>"
curl --fail-with-body -sS -i "$BASE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0
    }'
```

Expected result: HTTP 200 with a JSON `choices` array containing generated text and `usage` statistics. The following is an illustrative response, not a captured test result; generated text, IDs, timing, token counts, and optional fields vary.

```json
{
    "id": "cmpl-example",
    "object": "text_completion",
    "created": 0,
    "model": "glm",
    "choices": [
        {
            "index": 0,
            "text": " an open question.",
            "logprobs": null,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 4,
        "total_tokens": 9
    }
}
```

An HTTP error, startup failure, or empty/invalid response is not a successful verification. Retain the startup command, runtime versions, image digest, server log, and actual HTTP response with the result.

## 7 Accuracy Evaluation

### 7.1 Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).
Use the same checkpoint, hardware, deployment topology and generation
settings when comparing results. Scores are specific to the model shown.

### 7.2 GLM-5.3 W8A8C8 PD Reference

| dataset | model | hardware | deployment | metric | mode | vllm-api-general-chat |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| GPQA Diamond | GLM-5.3-w8a8c8 | A3 | PD disaggregation | accuracy | gen | 91.92 |
| GPQA Diamond | GLM-5.3-w8a8c8 | A2 | PD disaggregation | accuracy | gen | 89.39 |

### 7.3 W4A4C8 A5 Validation Records

Both records below use one 950DT Product (96GB × 8), the 2026-10-08 A5
image in Section 5.1.1, TP8/DP1/EP, eager mode, a configured maximum
context of 135000, 16384 batched tokens, 20 sequences and FP8 KV/indexer
caches. On 2026-10-10, `/health`, `/v1/chat/completions` and
`/v1/completions` returned HTTP 200 for both local checkpoints.
This verifies startup and short requests, not actual 135K inputs,
tool use or performance. The historical command used `--reasoning-parser glm45`;
the shared commands now use `glm47` to align with GLM-5.3. That parser
update has not been revalidated on NPU hardware in this documentation update.

| Model / local checkpoint | Startup and short requests | Dataset result |
| --- | --- | --- |
| GLM-5.2 W4A4C8/MXFP4 | Passed | No dataset score from this validation |
| GLM-5.3 W4A4C8/MXFP4, 78 layers, 102 indexed shards | Passed | GSM8K: 1279/1319, 96.97%, including 10 request errors |

The GLM-5.3 GSM8K run used zero-shot chat, numeric exact match on the
content field, `temperature=0`, `seed=42`, `max_tokens=8192`, concurrency
20, a 900-second timeout and no retries. All 1319 samples were attempted;
1309 requests finished normally and 10 failed. The score includes those
10 errors in the denominator. It is a local-checkpoint result, not GPQA
or an AISBench benchmark result, and does not establish a throughput gate.
The local checkpoint was not matched to a public release revision or
verified by per-shard hashes.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#5-online-service-deployment)** chapter.

## 9 Performance Tuning

The following A3 W8A8C8 settings summarize the deployment scripts in
Sections 5.1.2, 5.1.4 and 5.2.1. Global DP includes all nodes of one role;
local DP is the number of ranks on one node. Each role uses 32 logical
NPU devices.

### 9.1 Detailed Node Configuration

| Scenario | Role | TP | Global DP | Local DP | PP | DCP | Max Num Seqs | Max Num Batched Tokens | Max Model Len | MTP Spec Num |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 198K PD, with or without pooling | Prefill | 16 | 1 | 1 | 2 | 1 | 64 | 16384 | 200000 | 1 |
| 198K PD, with or without pooling | Decode | 2 | 16 | 8 | 1 | 1 | 64 | 256 | 200000 | 5 |
| 1M PD (A3) | Prefill | 16 | 2 | 1 | 1 | 16 | 64 | 8192 | 1048576 | 1 |
| 1M PD (A3) | Decode | 8 | 4 | 2 | 1 | 8 | 32 | 192 | 1048576 | 5 |

### 9.2 Performance Workloads

The 0.30 RC performance reference covers the following workloads.
Lengths use `input/output` notation. The 1M A3 case uses one concurrent
request; reserve output tokens within the 1,048,576-token context limit.

| Scenario | Input / Output | Prefix cache hit rate | Deployment reference |
| --- | --- | --- | --- |
| Prefill-only | 16K / 1, 128K / 1 | 0% | Section 5.1.3 |
| Decode-only | 16K / 1K, 128K / 1K | 100% | Section 5.1.3 |
| Complete PD | 16K / 1K | 0% | Section 5.1.3 |
| Complete PD, 1M context (A3) | Approximately 1M / 1K | 0% | Section 5.2.1 |
| Pooling | 128K / 1K | 90%, 32 distinct prefixes | Section 5.1.4 |
| Pooling | 40K-80K / 2.5K | Repeated prefixes | Section 5.1.4 |

### 9.3 Optimizations Requiring Explicit Enablement

Combine enabled options in one `--additional-config` JSON object, as in
the deployment scripts.

| Optimization | Scenario | Enablement | Purpose and constraints |
| --- | --- | --- | --- |
| FlashComm1 | A3 Prefill | `"enable_flashcomm1": true` | Reduces TP communication overhead; the Decode reference keeps it disabled. |
| Fused MC2 | A3 PD | `"enable_fused_mc2": 1` | Fuses MoE dispatch, computation and combine. Do not combine with shared-expert multistream overlap. |
| DSA CP | A3 Prefill | `"enable_dsa_cp": true` | Distributes long-context attention work; use the layout specified in each scenario. |
| DCP | 1M PD (A3) | `--decode-context-parallel-size 16` on Prefill, `8` on Decode | Shards KV cache across the TP group. Keep PCP at 1 and KV interleave at 128 on both roles. |
| Int8 attention KV cache | A3 W8A8C8 | `--kv-cache-dtype int8` | Reduces attention KV cache memory. |
| Int8 indexer KV cache | A3 W8A8C8 | `--attention_config.indexer_kv_dtype int8` | Quantizes the indexer cache separately from the attention cache. |
| MLAPO | A3 1M Decode | `"enable_mlapo": true` | Fuses MLA preprocessing; account for its memory requirements. |
| Recompute scheduler | A3 Decode | `"recompute_scheduler_enable": true` | Enables Prefill-side recomputation when Decode KV cache capacity is insufficient. |
| Decode graph | A3 Decode | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | Captures Decode execution; the reference Prefill uses eager mode. |
| KV pooling | Repeated-prefix PD | Ascend Store/MemCache configuration in Section 5.1.4 | Reuses KV across requests and engines; confirm pool hits in backend metrics. |

### 9.4 TurboQuant KV Cache for GLM-5.2 and GLM-5.3

Use a main-branch build containing TurboQuant support, with
`VLLM_USE_V2_MODEL_RUNNER=1` and `--dtype bfloat16` on both roles.
The v0.30.0rc1 PP2/PD/pooling/DCP reference scripts above explicitly
use model runner V1; changing only their KV cache dtype is insufficient.
Revalidate the topology and connector with runner V2 before using
TurboQuant, and disable xLite graph mode.

On A2/A3, `--kv-cache-dtype turboquant_4bit_nc` enables 4-bit
TurboQuant for the SFA packed main KV cache as an alternative to
`--kv-cache-dtype int8`. It cannot be combined with `enable_sparse_sfa_c8: true`.
For PD, set `"use_ascend_direct": false` in the Mooncake connector
configuration on both roles. Apply the same cache format on Prefill
and Decode. This option does not apply to the A5 FP8 examples or
GLM-5.3-Flash.

## 10 FAQ

### Q: How to enable function calling for GLM-5.2 and GLM-5.3?

A: Please add following configurations in vLLM startup command

```shell
--tool-call-parser glm47 \
--reasoning-parser glm47 \
--enable-auto-tool-choice \
```

### Q: How do the W4A4C8 ultra-low-bit quantized weights preserve accuracy?

A: For 950PR&950DT Products, the W4A4C8 weights of GLM-5.2 are quantized with [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/glm_5_2/glm_5_2_w4a4c8_mxfp4.yaml). The quantization pipeline first applies QuaRot rotation and FlexSmoothQuant smoothing to suppress activation outliers, and then quantizes the model with a mixed-precision strategy: only the modules that dominate the parameter count are quantized to 4 bit, while precision-sensitive modules fall back to higher precision or are not quantized at all.

| Module | Quantization configuration |
| ------ | -------------------------- |
| MoE FFN routed experts (`mlp.experts`) | W4A4 MXFP4 |
| Attention linear layers (except `kv_b_proj`, `wk`, `weights_proj`) | W8A8 MXFP8 |
| Dense FFN linear layers (except router gates) | W8A8 MXFP8 |
| Router gates, `kv_b_proj`, `wk`, `weights_proj` | Not quantized (kept in high precision) |

- **Quantization algorithm**: The weights and activations use the MXFP4/MXFP8 microscaling formats. Each tensor is divided into blocks of 32 elements, and each block shares a single scaling factor obtained with symmetric min-max calibration on a chain-of-thought calibration dataset. Combined with the QuaRot/FlexSmoothQuant outlier suppression, block-wise scaling adapts to the local dynamic range of each tensor, so the quantization error is far lower than that of uniform INT4 quantization.
- **C8 (FP8 KV cache)**: C8 is a dynamic quantization applied to the KV cache during inference. On 950PR&950DT Products, the KV cache is quantized to FP8 (not INT8). It is unrelated to the parameters stored in the weight checkpoint and is controlled only by the serving configuration:
    - **Attention KV cache** (SFA C8, enabled with `--kv-cache-dtype fp8`): the compressed KV latent, which serves as both K and V, is dynamically quantized per 128-element tile to FP8 (E4M3) with a per-tile scale; the RoPE part of the KV cache stays in BF16, and Q is not quantized.
    - **Indexer cache** (LI C8, enabled with `--attention_config.indexer_kv_dtype fp8`): both indexer Q and K are first rotated by a 128-dimensional Hadamard matrix to spread outliers, and then dynamically quantized per token to FP8 (E4M3) with per-token FP32 scales.
    - Precision-sensitive modules (the indexer and some attention layers) are configured as C8 fallback layers in the quantization configuration: they are excluded from C8 quantization during inference and run in high precision.

For W4A4C8 accuracy scores, see [Accuracy Evaluation](#7-accuracy-evaluation).

### Q: Can thinking be disabled for GLM-5.3?

No. Keep thinking enabled for the GLM-5.3 commands on this page.
