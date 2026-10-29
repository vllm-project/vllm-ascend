# GLM-5.2 & GLM-5.3

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) uses a Mixture-of-Experts (MoE) architecture and targets complex systems engineering and long-horizon agentic tasks. [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) shares its base model with GLM-5.2 but has different post-training. This page covers both models; commands and validation results are version-specific unless explicitly stated otherwise.

This document shows the verification steps for both models, including supported features, environment preparation, single-node and multi-node deployment, PP2 Prefill-Decode disaggregation, MemCache KV pooling, 1M DCP, accuracy, and performance evaluation. Shared deployment settings follow the GLM-5.3 configurations. Checkpoint identity and measured results remain model-specific.

GLM-5.2 and GLM-5.3 are officially supported in vLLM-Ascend v0.30.0.
This guide follows the `releases/v0.30.0` deployment configurations;
use a build containing the maintenance updates required by the selected scenario.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

Download the checkpoint from ModelScope for the hardware shown below. A3 and 950PR&950DT Products checkpoints use different quantization formats; do not interchange them. The scripts use `/root/.cache/modelscope/hub/models/vllm-ascend/<checkpoint-name>` as a local mount path, not as a download source. Preserve the exact case of the repository name when naming the local directory.

|  Weight Version          | Hardware Requirements                                             | Download Links |
|--------------------------|-------------------------------------------------------------------|----------------|
|  `GLM-5.2-w8a8c8`        | 2 A3 series (64GB × 16) nodes or 4 A2 series (64GB × 8) nodes | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w8a8c8) |
|  `GLM-5.2-W8A8C8-mxfp8` | 2 950DT products (96GB × 8) nodes for colocated deployment | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-W8A8C8-mxfp8) |
|  `GLM-5.3-w8a8c8`        | 2 A3 series (64GB × 16) nodes or 4 A2 series (64GB × 8) nodes | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-w8a8c8) |
|  `GLM-5.3-W8A8C8-mxfp8` | 2 950DT products (96GB × 8) nodes for colocated deployment | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-W8A8C8-mxfp8) |
|  `GLM-5.3-W4A4C8-mxfp4` | 1 950DT products (96GB × 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-w4a4c8-mxfp4) |
|  `GLM-5.2-w4a4c8-mxfp4` | 1 950DT products (96GB × 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w4a4c8-mxfp4) |

- `GLM-5.2-w8a8c8`: The weights have been verified and are recommended for use on A3. For GLM-5.3, use its own checkpoint and the version-specific commands below; do not apply the GLM-5.2 accuracy or performance results to GLM-5.3.
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.
- `GLM-5.2-w4a4c8-mxfp4` (950PR&950DT Products, W4A4 MXFP4 mixed-precision quantization): For how the ultra-low-bit quantization preserves accuracy, see the [W4A4C8 FAQ](#q-how-do-the-w4a4c8-ultra-low-bit-quantized-weights-preserve-accuracy).

The checkpoints linked above are already quantized and can be served directly. For self-quantization, follow the installation and command examples in the [msModelSlim quantization guide](https://www.hiascend.com/document/detail/en/mindstudio/2610/TITools/msModelSlim/docs/en/user_guide/feature_guide/quick_quantization_v1/usage.md) and select the configuration for the exact model and target hardware from the [model recipes](https://gitcode.com/Ascend/msmodelslim/tree/master/lab_practice). Do not reuse a different model's YAML file.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

Before using any script with `VLLM_USE_FASTOKENS=1`, install the [fastokens tokenizer](https://github.com/crusoecloud/fastokens) in the same Python environment as vLLM with `python -m pip install fastokens`. Run `python -c "import fastokens"` in that environment to check that the package is available.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

Select an image for your hardware and start a container on each node.
Refer to [Using Docker](../../getting_started/installation.md#installation-prebuilt-image) for details.

For 950DT products PD deployment, prepare the platform-generated HCCL/HIXL configuration at `/etc/hccl_rootinfo.json` and `/etc/hixlep/` before mounting it into the containers.

=== "950DT products"

    Start the Docker container on each node.

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

    Start the Docker container on each node.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    export NAME=vllm-ascend

    # Run the container using the defined variables
    # If you use Docker bridge networking, expose the required ports for multi-node communication in advance.
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

    Start the Docker container on each node.

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

For the A3 and A2 PD, pooling and 1M examples, select the hardware-specific
image and container mounts from Section 4. Use the same image on every
node in a deployment.

<a id="glm-common-parameters"></a>

**Common Parameter Descriptions**

These definitions apply to the deployment scenarios in this chapter. Use the
values in the selected hardware and role scripts; hardware-specific values
are not interchangeable.

| Parameter | Meaning and constraints |
| --- | --- |
| `HCCL_BUFFSIZE` | CANN/HCCL communication buffer size in MB; a positive integer, with a documented default of 200. Larger buffers consume more device memory, multiplied by the number of communication domains. Keep the example's hardware/role-specific value and account for this memory when sizing the KV cache. See the [CANN HCCL reference](https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/hcclug/hcclenvref_07_0005.html) for allocation details and version-specific constraints. |
| `HCCL_IF_IP` and socket interface variables | Set the current node's communication IP and matching network interface. Peers must be reachable through the selected network. |
| `--data-parallel-size`, `--tensor-parallel-size`, `--pipeline-parallel-size` | DP is the number of data-parallel ranks; TP and PP partition each model engine across devices and pipeline stages. Follow the topology and placement in the selected tab. |
| `--max-model-len` | Maximum total sequence length in tokens, including input and generated output. Requests must fit this limit and the available KV cache capacity. |
| `--max-num-seqs`, `--max-num-batched-tokens` | Scheduler limits for sequences and tokens processed in an iteration, respectively. These are per engine, not aggregate proxy limits; tune them together with context length and memory capacity. |
| `--gpu-memory-utilization` | Fraction of device memory budgeted to the model executor. Despite the flag name, it also applies to the NPU backend. Leave room for communication buffers and other device allocations. |
| `--quantization ascend` | Selects the Ascend quantization loader for ModelSlim checkpoints. Use weights matching the selected hardware and quantization format. |
| `--kv-cache-dtype`, `--attention_config.indexer_kv_dtype` | Select the attention and indexer KV cache formats. Follow the selected checkpoint and hardware example; the supported formats differ across platforms. |
| `--tool-call-parser glm47`, `--enable-auto-tool-choice` | Parse tool calls and allow automatic tool selection. Configure them on both API-serving roles in a PD deployment. |
| `--reasoning-parser` | Separates reasoning from the final answer. Use the parser in the selected example (`glm47` or `glm45`); do not infer it from the tool-call parser name. |

### 5.1 Context Below 1M

<a id="glm-950-preparation"></a>

**950DT products Preparation**

Each node exposes eight devices. Install `fastokens` before enabling
`VLLM_USE_FASTOKENS=1`.

Before running a role script, export `MODEL_PATH`, `LOCAL_IP`, and
`NIC_NAME`. PP ranks within each Prefill instance must use that instance's
master address; non-master PP ranks run headless. Follow the master-address
variable in the selected script. Use the same checkpoint path on all nodes.

#### 5.1.1 Single-Node Deployment {: #511-single-node-deployment }

=== "950DT products"

    ##### 950DT products {: #glm52-53-5-1-1-950dt }

    The GLM-5.2 and GLM-5.3 W4A4C8/MXFP4 checkpoints can run on one
    950DT products (96GB × 8) node with DP1/TP8. The command below uses
    GLM-5.3; for GLM-5.2, replace the checkpoint path with
    `/root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a4c8-mxfp4`.
    Startup and short requests were validated; see
    [W4A4C8 Validation](#73-w4a4c8-950dt-products-validation-records) for details.

    ###### Start the Engine

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

    **Key Parameter Descriptions — 950DT products**

    - DP1/TP8 runs one engine across eight devices; `--enable-expert-parallel`
      distributes MoE experts within the configured parallel groups.
    - `--max-model-len 135000` caps input plus output tokens. The scheduler
      limits are 20 sequences and 16384 batched tokens per iteration; these
      settings do not establish full-length capacity or accuracy validation.
    - `--quantization ascend` loads the W4A4C8/MXFP4 checkpoint. The example
      uses FP8 attention and indexer KV caches; retain the matching checkpoint.
    - `--enforce-eager` disables graph execution. FlashComm1, CPU binding,
      shared-expert overlap and DSA CP are selected by `--additional-config`.
    - `HCCL_BUFFSIZE=400` reserves the communication buffer size described in
      [Common Parameter Descriptions](#glm-common-parameters).

=== "A3 Series"

    ##### A3 Series {: #glm52-53-5-1-1-a3 }

    For a reduced context of 32768 tokens, the W8A8C8 checkpoint can run on 1 A3 series (64GB × 16) with DP1/TP16. Select the GLM-5.2 or GLM-5.3 checkpoint below. This is not the full-context deployment: use the multi-node deployment for longer contexts.

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
      --enable-expert-parallel \
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

    **Key Parameter Descriptions — A3 series**

    - DP1/TP16 uses all 16 logical devices for one engine. This is the
      reduced-context W8A8C8 layout, with a maximum of 32768 total tokens.
    - `--max-num-seqs 12` and `--max-num-batched-tokens 8192` cap concurrent
      sequences and scheduled tokens per iteration; adjust them to KV capacity.
    - `--quantization ascend` loads ModelSlim W8A8C8 weights. Both attention
      and indexer KV caches use `int8` in this hardware tab.
    - `FULL_DECODE_ONLY` captures only Decode execution; `deepseek_mtp` uses
      three speculative tokens with an eager draft model.
    - `--enable-expert-parallel` is required for the requested DSA context
      parallelism and FlashComm1 with this TP16/DP1 layout.
    - `HCCL_BUFFSIZE=200` follows the common HCCL buffer definition above.

#### 5.1.2 Multi-node Deployment {: #512-multi-node-deployment }

If you want to deploy a multi-node environment, first verify the data-plane network by following [Verify Multi-node Communication Environment](../../getting_started/installation.md#installation-multi-node-interconnect).

=== "950DT products"

    ##### 950DT products {: #glm52-53-5-1-2-950dt }

    ###### Topology and Prerequisites

    See [950DT preparation](#glm-950-preparation) for checkpoint, environment and PP-node settings.

    Run the following scripts on two nodes respectively. Set `MODEL_PATH`,
    `LOCAL_IP`, `NIC_NAME`, and the same `P0_IP` on both nodes. Only node 0
    exposes the API at port 8000.

    ###### Prepare the Scripts

    ###### node 0

    Prepare `run_dp_template.sh` on node 0:

    ```shell
    #!/usr/bin/env bash

    : "${MODEL_PATH:?Set the checkpoint path on every node}"
    : "${LOCAL_IP:?Set this node address}"
    : "${NIC_NAME:?Set the matching network interface}"
    local_ip="$LOCAL_IP"
    nic_name="$NIC_NAME"
    : "${P0_IP:?Set the shared PP master address}"

    unset ftp_proxy FTP_PROXY
    unset https_proxy HTTPS_PROXY
    unset http_proxy HTTP_PROXY
    export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
    export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
    unset ftp_proxy
    unset https_proxy
    unset http_proxy
    unset HCCL_INTRA_ROCE_ENABLE

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

    LOG_DIR="${LOG_DIR:-$PWD/logs}"
    mkdir -p "$LOG_DIR"
    export ASCEND_PROCESS_LOG_PATH="$LOG_DIR"

    export VLLM_PP_LAYER_PARTITION="42,36"
    export VLLM_USE_FASTOKENS=1

    export VLLM_DISABLE_COMPILE_CACHE=1
    export VLLM_USE_V2_MODEL_RUNNER=0

    vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port 8000 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --pipeline-parallel-size 2 \
      --distributed-executor-backend mp \
      --master-addr "$P0_IP" \
      --master-port 7060 \
      --nnodes 2 \
      --node-rank 0 \
      --max-model-len 135000 \
      --max-num-batched-tokens 16384 \
      --served-model-name glm \
      --gpu-memory-utilization 0.9 \
      --max-num-seqs 500 \
      --trust-remote-code \
      --enforce-eager \
      --quantization ascend \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --async-scheduling \
      --enable-prefix-caching \
      --speculative-config '{"num_speculative_tokens": 5,"method": "deepseek_mtp", "enforce_eager":true}' \
      --kv-cache-dtype fp8 \
      --attention_config.indexer_kv_dtype fp8 \
      --additional-config '{"enable_cpu_binding": true, "enable_dsa_cp": true, "enable_flashcomm1": true, "enable_fused_mc2": 1}'
    ```

    ###### node 1

    Prepare `run_dp_template.sh` on node 1:

    ```shell
    #!/usr/bin/env bash

    : "${MODEL_PATH:?Set the checkpoint path on every node}"
    : "${LOCAL_IP:?Set this node address}"
    : "${NIC_NAME:?Set the matching network interface}"
    local_ip="$LOCAL_IP"
    nic_name="$NIC_NAME"
    : "${P0_IP:?Set the shared PP master address}"

    unset ftp_proxy FTP_PROXY
    unset https_proxy HTTPS_PROXY
    unset http_proxy HTTP_PROXY
    export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
    export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
    unset ftp_proxy
    unset https_proxy
    unset http_proxy
    unset HCCL_INTRA_ROCE_ENABLE

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

    LOG_DIR="${LOG_DIR:-$PWD/logs}"
    mkdir -p "$LOG_DIR"
    export ASCEND_PROCESS_LOG_PATH="$LOG_DIR"

    export VLLM_PP_LAYER_PARTITION="42,36"
    export VLLM_USE_FASTOKENS=1

    export VLLM_DISABLE_COMPILE_CACHE=1
    export VLLM_USE_V2_MODEL_RUNNER=0

    vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port 8000 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --pipeline-parallel-size 2 \
      --distributed-executor-backend mp \
      --master-addr "$P0_IP" \
      --master-port 7060 \
      --nnodes 2 \
      --node-rank 1 \
      --max-model-len 135000 \
      --max-num-batched-tokens 16384 \
      --served-model-name glm \
      --gpu-memory-utilization 0.9 \
      --max-num-seqs 500 \
      --trust-remote-code \
      --enforce-eager \
      --quantization ascend \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --async-scheduling \
      --enable-prefix-caching \
      --speculative-config '{"num_speculative_tokens": 5,"method": "deepseek_mtp", "enforce_eager":true}' \
      --kv-cache-dtype fp8 \
      --attention_config.indexer_kv_dtype fp8 \
      --additional-config '{"enable_cpu_binding": true, "enable_dsa_cp": true, "enable_flashcomm1": true, "enable_fused_mc2": 1}' \
      --headless
    ```

    ###### Start the Engines

    Run on each node after setting its environment variables:

    ```shell
    bash run_dp_template.sh
    ```

    **Key Parameter Descriptions — 950DT products**

    The following settings apply only to this 950DT products tab; use the
    topology and scripts in the other tabs for A3 or A2 series.

    - `--pipeline-parallel-size 2` and `--tensor-parallel-size 8` split one
      engine across two nodes. `VLLM_PP_LAYER_PARTITION="42,36"` assigns
      the layers to the two PP stages.
    - `--master-addr` selects node 0. Node 0 exposes API port `8000`;
      node 1 uses `--headless` and joins the same PP engine.
    - `--max-model-len 135000`, `--max-num-seqs 500`, and
      `--max-num-batched-tokens 16384` set the context and scheduler limits.
      Adjust the scheduler limits for the target workload.
    - `--kv-cache-dtype fp8` and `--attention_config.indexer_kv_dtype fp8`
      enable FP8 attention and indexer KV caches.
    - `--speculative-config` enables `deepseek_mtp` with five speculative
      tokens and an eager draft. `--enforce-eager` also keeps the main
      engine in eager mode.
    - `--additional-config` enables DSA CP, FlashComm1, CPU binding and
      `enable_fused_mc2: 1`. Both PP nodes use the same configuration.

=== "A3 series"

    ##### A3 series {: #glm52-53-5-1-2-a3 }
    -  `GLM-5.3-w8a8c8`: can be deployed on 2 A3 series (64GB × 16).

    Run the following scripts on two nodes respectively.

    ###### node 0

    ```shell
    # Obtain the current node IP and its network interface using ifconfig.
    # Set nic_name to the network interface corresponding to local_ip on this node.
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
        --data-parallel-size-local 4 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm \
        --tool-call-parser glm47 \
        --reasoning-parser glm47 \
        --enable-auto-tool-choice \
        --max-num-seqs 6 \
        --max-model-len 202752 \
        --max-num-batched-tokens 4096 \
        --trust-remote-code \
        --gpu-memory-utilization 0.90 \
        --quantization ascend \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --async-scheduling \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-cache-dtype int8 \
        --attention_config.indexer_kv_dtype int8 \
        --additional-config '{"enable_dsa_cp": true, "enable_balance_scheduling": true, "enable_fused_mc2": 1, "enable_flashcomm1": true}'  \
        --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

    ###### node 1

    ```shell
    # Obtain the current node IP and its network interface using ifconfig.
    # Set nic_name to the network interface corresponding to local_ip on this node.
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
        --safetensors-load-strategy prefetch \
        --data-parallel-size 8 \
        --data-parallel-start-rank 4 \
        --data-parallel-size-local 4 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm \
        --tool-call-parser glm47 \
        --reasoning-parser glm47 \
        --enable-auto-tool-choice \
        --max-num-seqs 6 \
        --max-model-len 202752 \
        --max-num-batched-tokens 4096 \
        --trust-remote-code \
        --gpu-memory-utilization 0.90 \
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

    ##### A2 series {: #glm52-53-5-1-2-a2 }

    - `GLM-5.3-w8a8c8`: can be deployed on 4 A2 series (64GB × 8).

    Run the following scripts on four nodes respectively.

    ###### node 0

    ```shell
    # Obtain the current node IP and its network interface using ifconfig.
    # Set nic_name to the network interface corresponding to local_ip on this node.
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
    export VLLM_USE_V2_MODEL_RUNNER=0

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-w8a8c8 \
        --host 0.0.0.0 \
        --port 8000 \
        --max-model-len 135000 \
        --data-parallel-size 4 \
        --data-parallel-size-local 1 \
        --data-parallel-address "${node0_ip}" \
        --data-parallel-rpc-port 12980 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm \
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
        --additional-config '{
            "enable_dsa_cp": true,
            "scheduler_config": {"enable_balance_scheduling": true},
            "ascend_compilation_config": {"fuse_muls_add": true},
            "multistream_overlap_shared_expert": true,
            "enable_flashcomm1": true
        }' \
        --enable-prefix-caching \
        --async-scheduling \
        --api-server-count 1
    ```

    ###### node 1-3

    ```shell
    # Obtain the current node IP and its network interface using ifconfig.
    # Set nic_name to the network interface corresponding to local_ip on this node.
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
    export VLLM_USE_V2_MODEL_RUNNER=0

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
        --served-model-name glm \
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
        --additional-config '{
            "enable_dsa_cp": true,
            "scheduler_config": {"enable_balance_scheduling": true},
            "ascend_compilation_config": {"fuse_muls_add": true},
            "multistream_overlap_shared_expert": true,
            "enable_flashcomm1": true
        }' \
        --enable-prefix-caching \
        --async-scheduling
    ```

**Key Parameter Descriptions**

The following descriptions cover the hardware tabs in this section; each
platform-specific group is labeled. Shared parameter meanings are listed in
[Common Parameter Descriptions](#glm-common-parameters).

**A3 network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`: Network interface configuration for multi-node communication. Set `nic_name` to the network interface name (obtained via `ifconfig`) and `local_ip` to the current node's IP address. These must be correctly configured on each node for successful multi-node communication.
- `--data-parallel-size 8`: Total number of data parallel ranks across all nodes (4 ranks per node in this scenario).
- `--data-parallel-size-local 4`: Number of data parallel ranks on the current node.
- `--data-parallel-start-rank`: Omit this option on API node 0 so rank `0` is inferred and internal DP load balancing is retained. Only headless node 1 explicitly sets `4`.
- `--data-parallel-address`: IP address of the data parallel master node (node 0). Must match the `local_ip` of the master node.
- `--data-parallel-rpc-port 12980`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: Used on node 1. Node 0 hosts the API server.

**A2 four-node network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and
  `HCCL_SOCKET_IFNAME`: Set each node's local IP and matching network
  interface for multi-node communication.
- `--data-parallel-size 4`: Four global DP ranks, one per A2 node.
- `--data-parallel-size-local 1`: One local DP rank per node, using TP8.
- Node 0 hosts the API and omits `--data-parallel-start-rank`, so rank 0
  is inferred automatically. Nodes 1-3 use `--headless` and explicitly
  set `--data-parallel-start-rank` to 1, 2, and 3, respectively.
- `--data-parallel-address`: Node 0's IP, shared by all four nodes.
  `--data-parallel-rpc-port 12980` must also match on every node.

In this internal-DP layout, explicitly setting `--data-parallel-start-rank 0` on the
API node triggers external/hybrid DP load-balancing inference. The remote
headless engines are then rejected with
`Remote engine must not use --headless in external or hybrid dp lb mode`.
Keep the API node's rank implicit so all four EngineCores join the same
DP group.

Use `scheduler_config.enable_balance_scheduling` and
`ascend_compilation_config.fuse_muls_add` inside `--additional-config`.
The scheduler setting belongs under `scheduler_config` (the top-level form
is deprecated); `fuse_muls_add` belongs under `ascend_compilation_config`.

**A2-specific environment variables:**

- `CPU_AFFINITY_CONF=1`: Enables CPU core affinity binding for worker processes.
- `ACL_OP_INIT_MODE=1`: ACL operator initialization mode to speed up operator compilation.
- `VLLM_RPC_TIMEOUT=360000` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000` / `HCCL_EXEC_TIMEOUT=200` / `HCCL_CONNECT_TIMEOUT=120` / `VLLM_ENGINE_READY_TIMEOUT_S=1200`: Timeout settings for multi-node startup and model execution on the slower A2 platform. Increase them if the engine fails to become ready in time.
- `VLLM_USE_V2_MODEL_RUNNER=0`: Selects the V1 model runner for the A2 examples.

**KV cache configuration:**

- `--kv-cache-dtype turboquant_4bit_nc`: Enables 4-bit TurboQuant quantization for the SFA packed main KV cache, as an alternative to `--kv-cache-dtype int8`. It requires an A2/A3 device, does not support context parallelism (DCP/PCP), cannot be combined with the C8 cache, and in Prefill-Decode disaggregation the Mooncake option `"use_ascend_direct"` must be `false`.

**A3-specific fused MC2 notice:**
This scenario enables `additional_config.enable_fused_mc2=1` (fused `dispatch_ffn_combine`/`mega_moe` operators). Fused MC2 conflicts with `multistream_overlap_shared_expert` — the two optimizations must not be enabled at the same time (the runtime forcibly disables `multistream_overlap_shared_expert` when fused MC2 is on).
The A2 four-node commands in this section do not enable fused MC2.

#### 5.1.3 Prefill-Decode Disaggregation {: #513-prefill-decode-disaggregation }

Shared parameter meanings are listed in [Common Parameter Descriptions](#glm-common-parameters).
Topology, role-specific limits and connector settings are explained in each
hardware tab; apply only the settings for the selected platform.

The proxy commands below use the source tree bundled in the official
vLLM-Ascend image at `/vllm-workspace/vllm-ascend`. On a separate proxy
host, run them inside a vLLM-Ascend container. For a source installation,
replace that prefix with the actual checkout directory.

Select the hardware tab below for the prefill and decode topology. Prefill and decode run on separate node groups and exchange KV cache through Mooncake.

The ordinary PD examples use Mooncake for KV transfer between Prefill and
Decode. KV pooling is configured separately in [Section 5.1.4](#514-memcache-kv-cache-pool-deployment): all three
hardware tabs combine Mooncake P-to-D transfer with an Ascend Store
connector backed by MemCache. Follow each tab's connector version,
layerwise setting and topology.

=== "950DT products"

    ##### 950DT products {: #glm52-53-5-1-3-950dt }

    ###### Topology and Prerequisites

    See [950DT preparation](#glm-950-preparation) for checkpoint, environment and PP-node settings.

    This 2P1D service layout has two independent Prefill instances and one Decode group. Each Prefill instance uses PP2/TP8 across two eight-device nodes: P0/P1 form the first instance, and P2/P3 form the second. P0 and P2 expose the two Prefill API endpoints; P1 and P3 run headless. Decode uses DP16/TP1 across D0/D1, with eight local ranks per node (global ranks 0–15). Both roles declare DP16/TP1 in MooncakeConnectorV1. Prefill uses one MTP token; Decode uses five, a 60-token batch budget, and at most ten sequences per rank.

    ###### Prepare the Scripts

    1. prepare the script `launch_online_dp.py` on each Decode node:

        <a id="glm-950-dp-launcher"></a>

        The launcher passes these positional arguments to `run_dp_template.sh`: `$1` = visible devices, `$2` = API port, `$3` = global DP size, `$4` = global DP rank, `$5` = DP master IP, `$6` = DP RPC port, and `$7` = TP size. Start the launcher from the directory containing both scripts; do not invoke the Decode template without its arguments.

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
            if args.dp_size_local * args.tp_size > 8:
                parser.error("This 950DT configuration exposes eight devices per node")
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

        1. Prefill PP rank 0 (P0 and P2)

            ```shell
            #!/usr/bin/env bash

            : "${MODEL_PATH:?Set the checkpoint path on every node}"
            : "${LOCAL_IP:?Set this node address}"
            : "${NIC_NAME:?Set the matching network interface}"
            local_ip="$LOCAL_IP"
            nic_name="$NIC_NAME"
            # P0/P1 use <P0_IP>; P2/P3 use <P2_IP>.
            PP_MASTER_IP="<P0_IP>"

            unset ftp_proxy FTP_PROXY
            unset https_proxy HTTPS_PROXY
            unset http_proxy HTTP_PROXY
            export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
            export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
            unset ftp_proxy
            unset https_proxy
            unset http_proxy
            unset HCCL_INTRA_ROCE_ENABLE

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

            LOG_DIR="${LOG_DIR:-$PWD/logs}"
            mkdir -p "$LOG_DIR"
            export ASCEND_PROCESS_LOG_PATH="$LOG_DIR"

            export VLLM_PP_LAYER_PARTITION="42,36"
            export VLLM_USE_FASTOKENS=1

            export VLLM_DISABLE_COMPILE_CACHE=1

            export VLLM_USE_V2_MODEL_RUNNER=0
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

            vllm serve "$MODEL_PATH" \
              --host 0.0.0.0 \
              --port 8000 \
              --tensor-parallel-size 8 \
              --enable-expert-parallel \
              --pipeline-parallel-size 2 \
              --distributed-executor-backend mp \
              --master-addr "$PP_MASTER_IP" \
              --master-port 7060 \
              --nnodes 2 \
              --node-rank 0 \
              --max-model-len 135000 \
              --max-num-batched-tokens 16384 \
              --served-model-name glm \
              --gpu-memory-utilization 0.9 \
              --max-num-seqs 500 \
              --trust-remote-code \
              --enforce-eager \
              --quantization ascend \
              --enable-auto-tool-choice \
              --tool-call-parser glm47 \
              --reasoning-parser glm47 \
              --async-scheduling \
              --enable-prefix-caching \
              --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager":true}' \
              --kv-transfer-config  \
                '{"kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "engine_id": "0",
                "kv_port": "30000",
                "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {
                                    "dp_size": 1,
                                    "tp_size": 8,
                                    "pp_size": 2,
                                    "pp_layer_partition": "42,36"
                            },
                            "decode": {
                                    "dp_size": 16,
                                    "tp_size": 1
                            }
                    }
                }'  \
              --kv-cache-dtype fp8 \
              --attention_config.indexer_kv_dtype fp8 \
              --additional-config '{"enable_cpu_binding": true, "recompute_scheduler_enable": true, "enable_dsa_cp": true, "enable_flashcomm1": true, "enable_fused_mc2": 2, "multistream_overlap_shared_expert": true}'
            ```

        2. Prefill PP rank 1 (P1 and P3)

            ```shell
            #!/usr/bin/env bash

            : "${MODEL_PATH:?Set the checkpoint path on every node}"
            : "${LOCAL_IP:?Set this node address}"
            : "${NIC_NAME:?Set the matching network interface}"
            local_ip="$LOCAL_IP"
            nic_name="$NIC_NAME"
            # P0/P1 use <P0_IP>; P2/P3 use <P2_IP>.
            PP_MASTER_IP="<P0_IP>"

            unset ftp_proxy FTP_PROXY
            unset https_proxy HTTPS_PROXY
            unset http_proxy HTTP_PROXY
            export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
            export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
            unset ftp_proxy
            unset https_proxy
            unset http_proxy
            unset HCCL_INTRA_ROCE_ENABLE

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

            LOG_DIR="${LOG_DIR:-$PWD/logs}"
            mkdir -p "$LOG_DIR"
            export ASCEND_PROCESS_LOG_PATH="$LOG_DIR"

            export VLLM_PP_LAYER_PARTITION="42,36"
            export VLLM_USE_FASTOKENS=1

            export VLLM_DISABLE_COMPILE_CACHE=1

            export VLLM_USE_V2_MODEL_RUNNER=0
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

            vllm serve "$MODEL_PATH" \
              --host 0.0.0.0 \
              --port 8000 \
              --tensor-parallel-size 8 \
              --enable-expert-parallel \
              --pipeline-parallel-size 2 \
              --distributed-executor-backend mp \
              --master-addr "$PP_MASTER_IP" \
              --master-port 7060 \
              --nnodes 2 \
              --node-rank 1 \
              --max-model-len 135000 \
              --max-num-batched-tokens 16384 \
              --served-model-name glm \
              --gpu-memory-utilization 0.9 \
              --max-num-seqs 500 \
              --trust-remote-code \
              --enforce-eager \
              --quantization ascend \
              --enable-auto-tool-choice \
              --tool-call-parser glm47 \
              --reasoning-parser glm47 \
              --async-scheduling \
              --enable-prefix-caching \
              --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager":true}' \
              --kv-transfer-config  \
                '{"kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "engine_id": "1",
                "kv_port": "30000",
                "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {
                                    "dp_size": 1,
                                    "tp_size": 8,
                                    "pp_size": 2,
                                    "pp_layer_partition": "42,36"
                            },
                            "decode": {
                                    "dp_size": 16,
                                    "tp_size": 1
                            }
                    }
                }'  \
              --kv-cache-dtype fp8 \
              --attention_config.indexer_kv_dtype fp8 \
              --additional-config '{"enable_cpu_binding": true, "recompute_scheduler_enable": true, "enable_dsa_cp": true, "enable_flashcomm1": true, "enable_fused_mc2": 2, "multistream_overlap_shared_expert": true}' \
              --headless
            ```

        3. Decode nodes 0 and 1

            ```shell
            #!/usr/bin/env bash

            : "${MODEL_PATH:?Set the checkpoint path on every node}"
            : "${LOCAL_IP:?Set this node address}"
            : "${NIC_NAME:?Set the matching network interface}"
            local_ip="$LOCAL_IP"
            nic_name="$NIC_NAME"

            export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
            export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
            export VLLM_ASCEND_DP_ALLREDUCE_ON_NPU=1

            unset ftp_proxy
            unset https_proxy
            unset http_proxy
            unset HCCL_INTRA_ROCE_ENABLE

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

            LOG_DIR="${LOG_DIR:-$PWD/logs}"
            mkdir -p "$LOG_DIR"
            export ASCEND_PROCESS_LOG_PATH="$LOG_DIR"

            export ASCEND_RT_VISIBLE_DEVICES=$1
            export VLLM_USE_FASTOKENS=1

            export VLLM_DISABLE_COMPILE_CACHE=1

            export VLLM_USE_V2_MODEL_RUNNER=0
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

            vllm serve "$MODEL_PATH" \
              --host 0.0.0.0 \
              --port "$2" \
              --data-parallel-size $3 \
              --data-parallel-rank $4 \
              --data-parallel-address $5 \
              --data-parallel-rpc-port $6 \
              --tensor-parallel-size $7 \
              --max-model-len 135000  \
              --max-num-batched-tokens 60 \
              --served-model-name glm \
              --gpu-memory-utilization 0.95 \
              --enable-expert-parallel \
              --max-num-seqs 10 \
              --trust-remote-code \
              --quantization ascend \
              --enable-auto-tool-choice \
              --tool-call-parser glm47 \
              --reasoning-parser glm47 \
              --async-scheduling \
              --enable-prefix-caching \
              --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
              --speculative-config '{"num_speculative_tokens": 5,"method": "deepseek_mtp", "enforce_eager":true}' \
              --kv-transfer-config \
                '{"kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_consumer",
                "engine_id": "2",
                "kv_port": "30300",
                "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {
                                    "dp_size": 1,
                                    "tp_size": 8,
                                    "pp_size": 2,
                                    "pp_layer_partition": "42,36"
                            },
                            "decode": {
                                    "dp_size": 16,
                                    "tp_size": 1
                            }
                    }
                }' \
              --kv-cache-dtype fp8 \
              --attention_config.indexer_kv_dtype fp8 \
              --additional-config '{"enable_cpu_binding": true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false, "enable_super_kernel": false}, "recompute_scheduler_enable": true, "multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":8}, "enable_mlapo": true}'
            ```

    ###### Start the Engines

    Export the node-specific `MODEL_PATH`, `LOCAL_IP` and `NIC_NAME`, and the shared
    `D0_IP` address before starting the corresponding node script. Set
    `PP_MASTER_IP` inside each Prefill script: `<P0_IP>` on P0/P1 and `<P2_IP>`
    on P2/P3. Save the appropriate PP-rank script above as `run_dp_template.sh`
    on each node. Run each command on its named node in a separate terminal.
    Start both PP ranks of each Prefill instance and both Decode nodes before
    waiting for the APIs to become ready.

    1. Prefill node 0 (PP rank 0 of the first instance)

        ```shell
        bash run_dp_template.sh
        ```

    2. Prefill node 1 (PP rank 1 of the first instance)

        ```shell
        bash run_dp_template.sh
        ```

    3. Prefill node 2 (PP rank 0 of the second instance)

        ```shell
        bash run_dp_template.sh
        ```

    4. Prefill node 3 (PP rank 1 of the second instance)

        ```shell
        bash run_dp_template.sh
        ```

    5. Decode node 0

        ```shell
        python launch_online_dp.py --dp-size 16 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
        ```

    6. Decode node 1

        ```shell
        python launch_online_dp.py --dp-size 16 --tp-size 1 --dp-size-local 8 --dp-rank-start 8 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
        ```

    <a id="glm-950-proxy"></a>

    ###### Start the Proxy

    Reserve API ports, PP master port 7060 where applicable, DP RPC ports,
    and connector-derived transfer/handshake ports. Base KV ports are P:30000
    and D:30300 for ordinary PD/pooling, or D:30200 for 1M. Do not run multiple
    scenarios simultaneously on the same devices or ports.

    Run the proxy on a separate host after all API endpoints are healthy.
    Register P0 and P2 as the two Prefill endpoints. P1 and P3 are headless
    and must not be registered as API endpoints. Replace the host placeholders below.

    ```shell
    P0_IP="<P0_IP>"
    P2_IP="<P2_IP>"
    D0_IP="<D0_IP>"
    D1_IP="<D1_IP>"

    python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 8000 \
        --prefiller-hosts "$P0_IP" "$P2_IP" --prefiller-ports 8000 8000 \
        --decoder-hosts "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
            "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
            "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
            "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
        --decoder-ports 8000 8001 8002 8003 8004 8005 8006 8007 \
            8000 8001 8002 8003 8004 8005 8006 8007
    ```

    Use `glm` in requests to the proxy and follow [Section 6](#6-functional-verification) for smoke tests.
    For pooling, additionally verify MemCache saves, remote loads and hit metrics
    with repeated prefixes; a successful request alone is not evidence of pool
    reuse. Validate long-context capacity and accuracy on the exact model/build
    before increasing concurrency. No new measured results accompany these scripts.

    **Key Parameter Descriptions — 950DT products**

    The following settings apply only to this 950DT products tab; use the
    topology and scripts in the other tabs for A3 or A2 series.

    - `--pipeline-parallel-size 2` and `--tensor-parallel-size 8` define
      each Prefill instance, with `VLLM_PP_LAYER_PARTITION="42,36"` across
      two nodes. Two instances use four Prefill nodes in total.
      `PP_MASTER_IP` is P0 for the first instance and P2 for the second;
      only PP rank 0 in each instance exposes its Prefill API.
    - The Decode launcher uses `--dp-size 16`, `--tp-size 1`, and
      `--dp-size-local 8`: eight local ranks on each of two nodes.
      `--dp-address` and `--dp-rpc-port` must match across Decode nodes;
      `--dp-rank-start` assigns their disjoint global rank ranges.
    - `--vllm-start-port 8000` assigns Decode API ports `8000-8007` on
      each node. Register all 16 endpoints with the proxy.
    - Both roles use `--max-model-len 135000`. Prefill sets
      `--max-num-seqs 500` and `--max-num-batched-tokens 16384`;
      Decode sets `10` and `60`, respectively, per engine.
    - Both roles use FP8 attention and indexer KV caches. MTP uses
      `deepseek_mtp` with an eager draft: one speculative token on Prefill
      and five on Decode. Prefill uses `--enforce-eager`; Decode uses
      `FULL_DECODE_ONLY` graph mode.
    - `MooncakeConnectorV1` uses `kv_producer` on Prefill and `kv_consumer`
      on Decode, with base KV ports `30000` and `30300`, respectively.
      The connector describes each Prefill instance as DP1/PP2/TP8 and
      the shared Decode group as DP16/TP1.
    - Prefill enables DSA CP, FlashComm1 and `enable_fused_mc2: 2`.
      Decode enables MLAPO. Both roles enable shared-expert multistream
      overlap, prefix caching and recomputation scheduling.
    - `enable_fused_mc2: 2` selects MegaMoE when `cann_ops_transformer`
      is installed. In the selected maintenance branch, MTP draft MoE
      layers with an unsupported quantization layout use the non-fused
      communication path. Keep the draft model's `enforce_eager: true`.

=== "A3 series"

    ##### A3 series {: #glm52-53-5-1-3-a3 }

    This A3 reference serves GLM-5.3 W8A8C8 with a **200,000-token** maximum
    sequence length for the 198k workload. It uses one Prefill engine spanning
    P0/P1 with TP16 and PP2, and sixteen TP2 Decode engines on D0/D1.
    [Section 5.1.4](#514-memcache-kv-cache-pool-deployment) adds MemCache KV pooling to the same deployment.

    These scripts incorporate the supplied 198k PD+PP configuration; no new
    runtime, accuracy, or performance validation is implied. The 200,000-token
    limit covers the entire sequence, including generated tokens. Use the
    GLM-5.3 W8A8C8 A3 checkpoint; these settings are not a GLM-5.2 validation.
    Local weight/Python paths are placeholders, and API port 8000 and model
    name `glm` are retained for consistency with the launcher and proxy.

    The HCCS-only `ASCEND_GLOBAL_RESOURCE_CONFIG` and communication resource
    version `1.2` are retained from the source. Confirm that the target CANN
    stack and inter-node fabric support them before launch; do not apply this
    transport restriction to an unsupported network. Decode retains
    `LOCAL_WORLD_SIZE=16` and OpenMP settings from the source. Check static-kernel
    compatibility with the selected build and TP2 worker layout. Fused MC2 mode
    selection is version-dependent; confirm the effective mode in startup logs.

    Here, “198k” is the workload label, not an additional sequence-length setting. The `--max-model-len 200000` limit applies to prompt and generated tokens together; leave room for the requested output. The colocated example's separate 202752-token limit does not apply to this PD deployment.

    ###### Topology and Prerequisites

    | Role | Nodes | Global DP | TP | PP | Placement | Devices per role |
    | --- | --- | --- | --- | --- | --- | --- |
    | Prefill | P0, P1 | 1 | 16 | 2 | One PP stage per node | 32 |
    | Decode | D0, D1 | 16 | 2 | 1 | Eight DP ranks per node | 32 |

    Each container exposes devices 0-15. P0 exposes the only Prefill API.
    P1 runs the second PP stage without an API server. Decode ranks 0-7 run
    on D0 and ranks 8-15 on D1, with API ports 8000-8007 on each node.

    Use the A3 image selected in Section 4 on all four nodes.

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

    For `MooncakeConnectorV2`, let `B` be `kv_port` and `N = DP × PP × PCP × TP`. Worker handshake ports occupy `B` through `B + N - 1`; scheduler control ports occupy `B + N` through `B + N + DP - 1`. In this A3 PD example, allow Prefill ports `30000-30032` and Decode ports `30100-30147` between the participating nodes. The transfer engine also advertises its own RPC port; check the startup handshake metadata for the selected build rather than assuming `kv_port` is its only network port.

    <a id="glm53-pd-a3-scripts"></a>

    The launcher passes these positional arguments to `run_dp_template.sh`: `$1` = visible devices, `$2` = API port, `$3` = global DP size, `$4` = global DP rank, `$5` = DP master IP, `$6` = DP RPC port, and `$7` = TP size. Start the launcher from the directory containing both scripts; do not invoke the Decode template without its arguments.

    ###### Prepare the Scripts

    Before you start, please do the following:

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

            export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
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
                --served-model-name glm \
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
                    "enable_fused_mc2": 2,
                    "enable_flashcomm1": true,
                    "enable_shared_expert_dp": true
                }' \
                --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
                --kv-transfer-config '{
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_producer",
                    "engine_id": "0",
                    "kv_port": "30000",
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

            export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
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
                --served-model-name glm \
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
                    "enable_fused_mc2": 2,
                    "enable_flashcomm1": true,
                    "enable_shared_expert_dp": true
                }' \
                --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
                --kv-transfer-config '{
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_producer",
                    "engine_id": "1",
                    "kv_port": "30000",
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
            export OMP_PROC_BIND=false
            export OMP_NUM_THREADS=10
            export LOCAL_WORLD_SIZE=16
            export ASCEND_RT_VISIBLE_DEVICES=$1
            export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
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
                --no-enable-prefix-caching \
                --seed 1024 \
                --served-model-name glm \
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
                    "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": true},
                    "multistream_overlap_shared_expert": true
                }' \
                --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
                --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
                --kv-transfer-config '{
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_consumer",
                    "engine_id": "2",
                    "kv_port": "30100",
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
            export OMP_PROC_BIND=false
            export OMP_NUM_THREADS=10
            export LOCAL_WORLD_SIZE=16
            export ASCEND_RT_VISIBLE_DEVICES=$1
            export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
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
                --no-enable-prefix-caching \
                --seed 1024 \
                --served-model-name glm \
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
                    "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": true},
                    "multistream_overlap_shared_expert": true
                }' \
                --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
                --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
                --kv-transfer-config '{
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_consumer",
                    "engine_id": "3",
                    "kv_port": "30100",
                    "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "42,36"},
                        "decode": {"dp_size": 16, "tp_size": 2}
                    }
                }'
            ```

    <a id="glm53-pd-a3-start"></a>

    ###### Start the Engines

    Once preparation is complete, run the following commands on each node from the directory containing the scripts:

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

    <a id="glm53-pd-a3-proxy"></a>

    ###### Start the Proxy

    Replace each node's IP and installation-path placeholders before starting.
    Run the four commands in [Start the Engines](#glm53-pd-a3-start) in
    separate terminals. Keep the master processes running while the other
    nodes join; start both nodes of each group before waiting for the group
    to become healthy.

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

    python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
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
    - Both roles retain chunked prefill, async scheduling, seed 1024, and
      memory utilization 0.92. Prefill enables prefix caching, DSA CP,
      FlashComm1, `enable_fused_mc2=2`, and `enable_shared_expert_dp=true`.
      Non-pooled Decode disables prefix caching and NPU graph/static-kernel
      compilation with shared-expert multistream overlap, without fused MC2.
      Both use `glm47` tool and reasoning parsers and the served model name `glm`.
    - `MooncakeConnectorV2` declares the same Prefill DP1/PP2/TP16 and
      Decode DP16/TP2 topology on both sides. The non-pooled base KV ports
      are 30000 on Prefill and 30100 on Decode; reserve derived worker ports.

=== "A2 series"

    ##### A2 series {: #glm52-53-5-1-3-a2 }

    <a id="glm53-pd-a2"></a>

    ###### Eight-Node Deployment

    This reference deploys GLM-5.3 W8A8C8 on eight A2 series (64GB × 8)
    nodes. Use the A2 image and container command from Section 4. Both roles enable expert
    parallelism to distribute the MoE expert weights across their DP × TP ranks.

    | Role | Nodes | Global DP | TP | PP | Placement | Devices per role |
    | --- | --- | --- | --- | --- | --- | --- |
    | Prefill | P0-P3 | 1 | 8 | 4 | Four nodes | 32 |
    | Decode | D0-D3 | 16 | 2 | 1 | Four DP ranks per node, starting at 0/4/8/12 | 32 |

    P0 exposes the only Prefill API on port 8077; P1-P3 are headless.
    Decode exposes ports 10010-10013 on each node. Prefill uses the `mp`
    executor with `--nnodes 4`; Decode uses a launcher to start one TP2
    engine per DP rank. Both roles pin the V1 model runner and use int8
    attention and indexer KV caches. This deployment uses pipeline parallelism
    (PP4); the [A2 pooling topology in Section 5.1.4](#glm53-pool-a2) is different.

    Prepare identical weights and software on all eight nodes, verify
    [multi-node communication](../../getting_started/installation.md#installation-multi-node-interconnect),
    and set the model path, node IP, and network interface in each script.
    Install `fastokens` on every Prefill node for `VLLM_USE_FASTOKENS=1`.
    Reserve Prefill master port 29501, Prefill DP RPC port 12980, Decode
    DP RPC port 12990, the API and proxy ports, and Mooncake ports derived
    from base ports 30000 on Prefill and 30100 on Decode.

    <a id="glm53-pd-a2-launcher"></a>

    ###### Prepare the Decode DP Launcher

    Save the following A2 launcher as `launch_online_dp.py` on all Decode
    nodes. It allocates disjoint local device lists within the eight devices
    exposed by each A2 container. Arguments `$1`-`$7` passed to the engine
    script are visible devices, API port, global DP size, DP rank, DP master
    IP, DP RPC port, and TP size. Prefill is started directly with `vllm serve`.

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
        parser.add_argument("--dp-rpc-port", type=int, default=12990)
        parser.add_argument("--vllm-start-port", type=int, default=10010)
        parser.add_argument("--script", default="./start_d_engine.sh")
        args = parser.parse_args()
        if min(args.dp_size, args.tp_size, args.dp_size_local) <= 0:
            parser.error("Parallel sizes must be positive")
        if args.dp_rank_start < 0 or args.dp_rank_start + args.dp_size_local > args.dp_size:
            parser.error("Local ranks must be within the global DP group")
        if args.dp_size_local * args.tp_size > 8:
            parser.error("This A2 configuration exposes 8 devices per node")
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

    ###### Prefill Engine Script

    Save this script as `start_p_engine.sh` on P0-P3. Set `P_MASTER_IP` to
    P0's IP in every copy and `NIC_NAME` to the current node's interface.
    The command-line arguments select the node rank, API port, local IP,
    and visible devices.

    If the installed Mooncake worker rejects a missing local DP rank, see
    [Local DP Rank Compatibility and Troubleshooting](#glm53-a2-rc1-compatibility)
    and check the active vLLM-Ascend installation on every Prefill node
    before starting the group.

    ```shell
    #!/usr/bin/env bash
    set -eo pipefail

    NODE_RANK=${1:?need node_rank}
    PORT=${2:-8077}
    LOCAL_IP=${3:?need local_ip}
    VISIBLE_DEVICES=${4:-0,1,2,3,4,5,6,7}
    P_MASTER_IP="<PREFILL_NODE0_IP>"
    NIC_NAME="<NETWORK_INTERFACE>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    case "$NODE_RANK" in
      0) SERVER_ROLE_ARGS=(--api-server-count 1);;
      1|2|3) SERVER_ROLE_ARGS=(--headless);;
      *) echo "node_rank must be 0-3" >&2; exit 1;;
    esac

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY ALL_PROXY all_proxy
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

    # Cross-node PP4 uses the NPU RoCE interfaces for device communication.
    # Leave HCCL_IF_IP unset and pin the host-side sockets to the business NIC.
    unset HCCL_IF_IP
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_HOST_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=7200
    export HCCL_CONNECT_TIMEOUT=7200
    export VLLM_RPC_TIMEOUT=360000
    export VLLM_ENGINE_READY_TIMEOUT_S=1200
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE=AIV
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=1
    export VLLM_USE_V2_MODEL_RUNNER=0
    export CPU_AFFINITY_CONF=1
    export ACL_OP_INIT_MODE=1
    export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
    export VLLM_USE_FASTOKENS=1
    export VLLM_PP_LAYER_PARTITION="20,20,20,18"

    exec vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port "$PORT" \
      --pipeline-parallel-size 4 \
      --tensor-parallel-size 8 \
      --distributed-executor-backend mp \
      --nnodes 4 \
      --node-rank "$NODE_RANK" \
      --master-addr "$P_MASTER_IP" \
      --master-port 29501 \
      --data-parallel-address "$P_MASTER_IP" \
      --data-parallel-rpc-port 12980 \
      --enable-expert-parallel \
      --quantization ascend \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --seed 1024 \
      --served-model-name glm \
      --max-model-len 200000 \
      --max-num-batched-tokens 8192 \
      --max-num-seqs 64 \
      --gpu-memory-utilization 0.92 \
      --trust-remote-code \
      --async-scheduling \
      --enforce-eager \
      --safetensors-load-strategy prefetch \
      --kv-cache-dtype int8 \
      --attention_config.indexer_kv_dtype int8 \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --speculative-config '{"num_speculative_tokens":1,"method":"deepseek_mtp","enforce_eager":true}' \
      --additional-config '{"enable_dsa_cp":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true,"ascend_compilation_config":{"fuse_muls_add":true}}' \
      --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV2",
        "kv_role": "kv_producer",
        "engine_id": "0",
        "kv_port": "30000",
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {"dp_size": 1, "pp_size": 4, "tp_size": 8, "pp_layer_partition": "20,20,20,18"},
          "decode": {"dp_size": 16, "tp_size": 2}
        }
      }' \
      "${SERVER_ROLE_ARGS[@]}"
    ```

    ###### Decode Engine Script

    Save this script as `start_d_engine.sh` on D0-D3. Configure the current
    node's `LOCAL_IP`, `NIC_NAME`, and `MODEL_PATH` in each copy. The launcher
    supplies `$1`-`$7`, including the device list for each TP2 engine.

    ```shell
    #!/usr/bin/env bash
    set -eo pipefail

    VISIBLE_DEVICES=${1:?need visible_devices}
    PORT=${2:?need port}
    DP_SIZE=${3:?need dp_size}
    DP_RANK=${4:?need dp_rank}
    DP_ADDRESS=${5:?need dp_address}
    DP_RPC_PORT=${6:?need dp_rpc_port}
    TP_SIZE=${7:?need tp_size}
    LOCAL_IP="<CURRENT_DECODE_NODE_IP>"
    NIC_NAME="<NETWORK_INTERFACE>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY ALL_PROXY all_proxy
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

    export VLLM_HOST_IP="$LOCAL_IP"
    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=7200
    export HCCL_CONNECT_TIMEOUT=7200
    export VLLM_RPC_TIMEOUT=360000
    export VLLM_ENGINE_READY_TIMEOUT_S=1200
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE=AIV
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=1
    export VLLM_USE_V2_MODEL_RUNNER=0
    export CPU_AFFINITY_CONF=1
    export ACL_OP_INIT_MODE=1
    export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"

    exec vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port "$PORT" \
      --data-parallel-size "$DP_SIZE" \
      --data-parallel-rank "$DP_RANK" \
      --data-parallel-address "$DP_ADDRESS" \
      --data-parallel-rpc-port "$DP_RPC_PORT" \
      --tensor-parallel-size "$TP_SIZE" \
      --enable-expert-parallel \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --seed 1024 \
      --served-model-name glm \
      --max-model-len 200000 \
      --max-num-batched-tokens 256 \
      --max-num-seqs 32 \
      --gpu-memory-utilization 0.92 \
      --trust-remote-code \
      --async-scheduling \
      --quantization ascend \
      --safetensors-load-strategy prefetch \
      --kv-cache-dtype int8 \
      --attention_config.indexer_kv_dtype int8 \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --additional-config '{"recompute_scheduler_enable":true,"enable_sparse_sfa_c8":true,"enable_sparse_li_c8":true,"enable_shared_expert_dp":true}' \
      --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV2",
        "kv_role": "kv_consumer",
        "engine_id": "1",
        "kv_port": "30100",
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {"dp_size": 1, "pp_size": 4, "tp_size": 8, "pp_layer_partition": "20,20,20,18"},
          "decode": {"dp_size": 16, "tp_size": 2}
        }
      }'
    ```

    Both sides declare the same topology and use `MooncakeConnectorV2` for
    this non-uniform DP × TP layout.

    ###### Start the Engines

    Run the Decode commands on their respective nodes in separate terminals.
    Start all four nodes before waiting for the DP group to become ready.

    ```shell
    # D0: DP ranks 0-3
    python launch_online_dp.py --script ./start_d_engine.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 0 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 10010

    # D1: DP ranks 4-7
    python launch_online_dp.py --script ./start_d_engine.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 4 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 10010

    # D2: DP ranks 8-11
    python launch_online_dp.py --script ./start_d_engine.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 8 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 10010

    # D3: DP ranks 12-15
    python launch_online_dp.py --script ./start_d_engine.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 12 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 10010
    ```

    Start Prefill on P0, then P2, P1, and P3. Run each command on its target
    node, keeping the master process running while the other nodes join.

    ```shell
    # P0: API server and TP rank-group leader
    bash start_p_engine.sh 0 8077 "<PREFILL_NODE0_IP>" 0,1,2,3,4,5,6,7

    # P2: second TP rank-group leader
    bash start_p_engine.sh 2 8077 "<PREFILL_NODE2_IP>" 0,1,2,3,4,5,6,7

    # P1: headless partner of P0
    bash start_p_engine.sh 1 8077 "<PREFILL_NODE1_IP>" 0,1,2,3,4,5,6,7

    # P3: headless partner of P2
    bash start_p_engine.sh 3 8077 "<PREFILL_NODE3_IP>" 0,1,2,3,4,5,6,7
    ```

    ###### Start the Proxy

    Wait for P0:8077 and all sixteen Decode APIs on ports 10010-10013 to
    answer `GET /v1/models`. Confirm in the logs that P1-P3 have joined;
    headless nodes do not expose HTTP APIs. When HTTP proxy variables are
    set, use `curl --noproxy '*'` for business-IP checks to avoid proxy
    interception and HTTP 403 responses. For example, check P0 as follows,
    then check each Decode endpoint in the same way:

    ```shell
    curl --noproxy '*' "http://<PREFILL_NODE0_IP>:8077/v1/models"
    ```

    Start the load-balancing proxy on P0 or a host that can reach all
    endpoints. Clear inherited proxy settings before starting it:

    ```shell
    P0_IP="<PREFILL_NODE0_IP>"
    D0_IP="<DECODE_NODE0_IP>"
    D1_IP="<DECODE_NODE1_IP>"
    D2_IP="<DECODE_NODE2_IP>"
    D3_IP="<DECODE_NODE3_IP>"

    unset ftp_proxy FTP_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY ALL_PROXY all_proxy

    python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
      --host 0.0.0.0 --port 8000 --workers 8 \
      --prefiller-hosts "$P0_IP" --prefiller-ports 8077 \
      --decoder-hosts "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
        "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
        "$D2_IP" "$D2_IP" "$D2_IP" "$D2_IP" \
        "$D3_IP" "$D3_IP" "$D3_IP" "$D3_IP" \
      --decoder-ports 10010 10011 10012 10013 10010 10011 10012 10013 \
        10010 10011 10012 10013 10010 10011 10012 10013
    ```

    Check the proxy and send requests with model name `glm`, explicitly
    bypassing any client-side HTTP proxy:

    ```shell
    PROXY_IP="<PROXY_IP>"
    curl --noproxy '*' "http://${PROXY_IP}:8000/v1/models"
    curl --noproxy '*' "http://${PROXY_IP}:8000/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "glm",
        "messages": [{"role": "user", "content": "What is 12 times 7?"}],
        "max_tokens": 128,
        "temperature": 0
      }'
    ```

    Both roles use a 200,000-token maximum length, memory utilization 0.92,
    and three MTP tokens. Prefill uses eager mode, an 8,192-token batch budget,
    64 sequences, DSA CP, and FlashComm1. Decode uses a 256-token
    batch budget, 32 sequences, `FULL_DECODE_ONLY` graphs, sparse C8 options,
    and recomputation scheduling.

    <a id="glm53-a2-rc1-compatibility"></a>

    ###### Local DP Rank Compatibility and Troubleshooting

    If Prefill fails during Mooncake KV cache registration with
    `ValueError: Mooncake worker requires a local DP rank`, first verify that
    the conditional single-local-rank fix below is present in the installed
    worker. This topology uses DP1 with PP4 across P0-P3; it has no separate
    global DP ranks, and a PP headless worker may still receive
    `data_parallel_rank_local=None`.

    If the installed worker still contains the unconditional `None` check
    shown below, apply the following change to
    `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py`
    in the vLLM-Ascend installation used by **every Prefill node** before
    launching the scripts. It infers local rank 0 only when
    `data_parallel_size_local == 1`; multiple-local-rank configurations
    still require an explicit local rank.

    ```diff
    --- a/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py
    +++ b/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py
    @@ -93,9 +93,14 @@
             self.pp_rank = get_pp_group().rank_in_group
             self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
    -        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
    -        if self.dp_rank is None:
    -            raise ValueError("Mooncake worker requires a local DP rank")
             self.dp_size = vllm_config.parallel_config.data_parallel_size_local
    +        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
    +        if self.dp_rank is None:
    +            if self.dp_size != 1:
    +                raise ValueError("Mooncake worker requires a local DP rank")
    +            self.dp_rank = 0
    +            logger.warning(
    +                "Inferring Mooncake local DP rank 0 for a single local DP rank"
    +            )
             pcp_group = get_pcp_group()
             self.pcp_rank = pcp_group.rank_in_group
             self.pcp_size = pcp_group.world_size
    ```

    The same check and conditional fix apply to the pooled Prefill PP
    headless workers in the [A2 pooling tab of Section 5.1.4](#glm53-pool-a2).
    Confirm that the active Python package contains the change on every
    Prefill node, then start all members of the Prefill group before waiting
    for it to become ready.

    The supplied A2 deployment validation record dated 2026-10-15 reports
    the following results with the rc1 fix:

    | Check | Recorded result |
    | --- | --- |
    | Smoke request | HTTP 200; answer `84` |
    | GSM8K requests / successful / failed | 200 / 200 / 0 |
    | Elapsed time for the 200-request run | 105.324 seconds |

    The request counts report successful completion, not an accuracy score.

#### 5.1.4 MemCache KV Cache Pool Deployment {: #514-memcache-kv-cache-pool-deployment }

Shared parameter meanings are listed in [Common Parameter Descriptions](#glm-common-parameters).
Topology, role-specific limits and connector settings are explained in each
hardware tab; apply only the settings for the selected platform.

Select the tab that matches the deployment hardware and its Prefill topology.

=== "950DT products"

    ##### 950DT products {: #glm52-53-5-1-4-950dt }

    ###### Topology and Prerequisites

    See [950DT preparation](#glm-950-preparation) for checkpoint, environment and PP-node settings.

    Complete [MemCache backend setup](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend) first, including the metadata service, dependencies and per-node configuration. Export `MMC_LOCAL_CONFIG_PATH` to the actual configuration file on every node and make the installed MemCache libraries available to the vLLM Python runtime.

    This source uses **W4A4C8 MXFP4**, a 1,048,576-token limit, and layerwise pooling (`use_layerwise=true`) through MultiConnector, MooncakeConnectorV2 and AscendStoreConnector. It is not the A3 198k non-layerwise configuration. Both roles enable prefix caching. The source does not explicitly set `use_ascend_direct` for pooling; verify the selected build's default and backend transport rather than assuming the non-pooled setting. Decode needs four nodes with four TP2 ranks per node. Do not infer 1M capacity from the earlier single-node short-request W4A4C8 test.

    ###### Prepare the Scripts

    1. prepare the script `launch_online_dp.py` on each Decode node:

        Use the [950DT DP launcher](#glm-950-dp-launcher) from [Section 5.1.3](#513-prefill-decode-disaggregation).

    2. prepare the script `run_dp_template.sh` on each node.

        1. Prefill node 0

            ```shell
            #!/usr/bin/env bash

            : "${MODEL_PATH:?Set the checkpoint path on every node}"
            : "${LOCAL_IP:?Set this node address}"
            : "${NIC_NAME:?Set the matching network interface}"
            local_ip="$LOCAL_IP"
            nic_name="$NIC_NAME"
            : "${P0_IP:?Set the shared PP master address}"

            unset ftp_proxy FTP_PROXY
            unset https_proxy HTTPS_PROXY
            unset http_proxy HTTP_PROXY
            export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
            export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
            unset ftp_proxy
            unset https_proxy
            unset http_proxy
            unset HCCL_INTRA_ROCE_ENABLE

            export PYTHONHASHSEED=0
            : "${MMC_LOCAL_CONFIG_PATH:?Set the MemCache configuration path}"
            export MMC_LOCAL_CONFIG_PATH
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

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

            export VLLM_USE_FASTOKENS=1
            export VLLM_USE_V2_MODEL_RUNNER=0
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
            export VLLM_PP_LAYER_PARTITION="42,36"

            vllm serve "$MODEL_PATH" \
              --host 0.0.0.0 \
              --port 8000 \
              --distributed-executor-backend mp \
              --pipeline-parallel-size 2 \
              --tensor-parallel-size 8 \
              --enable-expert-parallel \
              --master-addr "$P0_IP" \
              --master-port 7060 \
              --nnodes 2 \
              --node-rank 0 \
              --max-model-len 1048576 \
              --max-num-batched-tokens 4096 \
              --served-model-name glm \
              --gpu-memory-utilization 0.95 \
              --max-num-seqs 8 \
              --trust-remote-code \
              --enforce-eager \
              --quantization ascend \
              --enable-auto-tool-choice \
              --tool-call-parser glm47 \
              --reasoning-parser glm47 \
              --enable-prefix-caching \
              --async-scheduling \
              --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager":true}' \
                --kv-transfer-config \
                '{
                    "kv_connector": "MultiConnector",
                    "kv_role": "kv_producer",
                    "engine_id": "0",
                    "kv_port": "30000",
                    "kv_connector_extra_config": {
                        "connectors":[
                        {
                            "kv_connector": "MooncakeConnectorV2",
                            "kv_role": "kv_producer",
                            "kv_port": "30000",
                            "kv_connector_extra_config": {
                            "prefill": {
                                    "dp_size": 1,
               "tp_size": 8,
                                    "pp_size": 2,
                                    "pp_layer_partition": "42,36"
                            },
                            "decode": {
                                    "dp_size": 16,
                                    "tp_size": 2
                            }
                            }
                        },
                        {
                            "kv_connector": "AscendStoreConnector",
                            "kv_role": "kv_producer",
                            "kv_connector_extra_config": {
                                "lookup_rpc_port":"0",
                                "backend": "memcache",
                                "use_layerwise": true
                            }
                        }
                        ]
                    }
                }' \
              --kv-cache-dtype fp8 \
              --attention_config.indexer_kv_dtype fp8 \
              --additional-config '{"enable_cpu_binding": true, "recompute_scheduler_enable": true, "enable_dsa_cp": true, "enable_flashcomm1": true, "enable_fused_mc2": 1}'
            ```

        2. Prefill node 1

            ```shell
            #!/usr/bin/env bash

            : "${MODEL_PATH:?Set the checkpoint path on every node}"
            : "${LOCAL_IP:?Set this node address}"
            : "${NIC_NAME:?Set the matching network interface}"
            local_ip="$LOCAL_IP"
            nic_name="$NIC_NAME"
            : "${P0_IP:?Set the shared PP master address}"

            unset ftp_proxy FTP_PROXY
            unset https_proxy HTTPS_PROXY
            unset http_proxy HTTP_PROXY
            export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
            export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
            unset ftp_proxy
            unset https_proxy
            unset http_proxy
            unset HCCL_INTRA_ROCE_ENABLE

            export PYTHONHASHSEED=0
            : "${MMC_LOCAL_CONFIG_PATH:?Set the MemCache configuration path}"
            export MMC_LOCAL_CONFIG_PATH
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

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

            export VLLM_USE_FASTOKENS=1
            export VLLM_USE_V2_MODEL_RUNNER=0
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
            export VLLM_PP_LAYER_PARTITION="42,36"

            vllm serve "$MODEL_PATH" \
              --host 0.0.0.0 \
              --port 8000 \
              --distributed-executor-backend mp \
              --pipeline-parallel-size 2 \
              --tensor-parallel-size 8 \
              --enable-expert-parallel \
              --master-addr "$P0_IP" \
              --master-port 7060 \
              --nnodes 2 \
              --node-rank 1 \
              --max-model-len 1048576 \
              --max-num-batched-tokens 4096 \
              --served-model-name glm \
              --gpu-memory-utilization 0.95 \
              --max-num-seqs 8 \
              --trust-remote-code \
              --enforce-eager \
              --quantization ascend \
              --enable-auto-tool-choice \
              --tool-call-parser glm47 \
              --reasoning-parser glm47 \
              --enable-prefix-caching \
              --async-scheduling \
              --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager":true}' \
                --kv-transfer-config \
                '{
                    "kv_connector": "MultiConnector",
                    "kv_role": "kv_producer",
                    "engine_id": "1",
                    "kv_port": "30000",
                    "kv_connector_extra_config": {
                        "connectors":[
                        {
                            "kv_connector": "MooncakeConnectorV2",
                            "kv_role": "kv_producer",
                            "kv_port": "30000",
                            "kv_connector_extra_config": {
                            "prefill": {
                                    "dp_size": 1,
               "tp_size": 8,
                                    "pp_size": 2,
                                    "pp_layer_partition": "42,36"
                            },
                            "decode": {
                                    "dp_size": 16,
                                    "tp_size": 2
                            }
                            }
                        },
                        {
                            "kv_connector": "AscendStoreConnector",
                            "kv_role": "kv_producer",
                            "kv_connector_extra_config": {
                                "lookup_rpc_port":"0",
                                "backend": "memcache",
                                "use_layerwise": true
                            }
                        }
                        ]
                    }
                }' \
              --kv-cache-dtype fp8 \
              --attention_config.indexer_kv_dtype fp8 \
              --additional-config '{"enable_cpu_binding": true, "recompute_scheduler_enable": true, "enable_dsa_cp": true, "enable_flashcomm1": true, "enable_fused_mc2": 1}' \
              --headless
            ```

        3. Decode nodes 0–3

            ```shell
            #!/usr/bin/env bash

            : "${MODEL_PATH:?Set the checkpoint path on every node}"
            : "${LOCAL_IP:?Set this node address}"
            : "${NIC_NAME:?Set the matching network interface}"
            local_ip="$LOCAL_IP"
            nic_name="$NIC_NAME"

            export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
            export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
            export VLLM_ASCEND_DP_ALLREDUCE_ON_NPU=1

            unset ftp_proxy
            unset https_proxy
            unset http_proxy
            unset HCCL_INTRA_ROCE_ENABLE

            export PYTHONHASHSEED=0
            : "${MMC_LOCAL_CONFIG_PATH:?Set the MemCache configuration path}"
            export MMC_LOCAL_CONFIG_PATH
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

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

            export ASCEND_RT_VISIBLE_DEVICES=$1
            export VLLM_USE_FASTOKENS=1

            export VLLM_USE_V2_MODEL_RUNNER=0
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

            vllm serve "$MODEL_PATH" \
              --host 0.0.0.0 \
              --port "$2" \
              --data-parallel-size $3 \
              --data-parallel-rank $4 \
              --data-parallel-address $5 \
              --data-parallel-rpc-port $6 \
              --tensor-parallel-size $7 \
              --max-model-len 1048576 \
              --max-num-batched-tokens 60 \
              --served-model-name glm \
              --gpu-memory-utilization 0.95 \
              --enable-expert-parallel \
              --max-num-seqs 10 \
              --trust-remote-code \
              --quantization ascend \
              --enable-auto-tool-choice \
              --tool-call-parser glm47 \
              --reasoning-parser glm47 \
              --async-scheduling \
              --enable-prefix-caching \
              --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
              --speculative-config '{"num_speculative_tokens": 5,"method": "deepseek_mtp", "enforce_eager":true}' \
                --kv-transfer-config \
                '{
                    "kv_connector": "MultiConnector",
                    "kv_role": "kv_consumer",
                    "engine_id": "2",
                    "kv_port": "30300",
                    "kv_connector_extra_config": {
                        "connectors":[
                        {
                            "kv_connector": "MooncakeConnectorV2",
                            "kv_role": "kv_consumer",
                            "kv_port": "30300",
                            "kv_connector_extra_config": {
                            "prefill": {
                                    "dp_size": 1,
               "tp_size": 8,
                                    "pp_size": 2,
                                    "pp_layer_partition": "42,36"
                            },
                            "decode": {
                                    "dp_size": 16,
                                    "tp_size": 2
                            }
                            }
                        },
                        {
                            "kv_connector": "AscendStoreConnector",
                            "kv_role": "kv_consumer",
                            "kv_connector_extra_config": {
                                "lookup_rpc_port":"0",
                                "backend": "memcache",
                                "use_layerwise": true
                            }
                        }
                        ]
                    }
                }' \
              --kv-cache-dtype fp8 \
              --attention_config.indexer_kv_dtype fp8 \
              --additional-config '{"enable_cpu_binding": true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false, "enable_super_kernel": false}, "recompute_scheduler_enable": true, "multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":8}, "enable_mlapo": true}'
            ```

    ###### Start the Engines

    Start MemCache and complete its configuration on all nodes first.

    Export the node-specific `MODEL_PATH`, `LOCAL_IP` and `NIC_NAME`, and the shared
    `P0_IP` / `D0_IP` addresses before starting the corresponding node script.

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
        python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
        ```

    4. Decode node 1

        ```shell
        python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 4 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
        ```

    5. Decode node 2

        ```shell
        python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 8 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
        ```

    6. Decode node 3

        ```shell
        python launch_online_dp.py --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 12 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
        ```

    ###### Start the Proxy

    Run the proxy on a separate host after all API endpoints are healthy.

    ```shell
    P0_IP="<P0_IP>"
    D0_IP="<D0_IP>"
    D1_IP="<D1_IP>"
    D2_IP="<D2_IP>"
    D3_IP="<D3_IP>"

    python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 8000 \
        --prefiller-hosts "$P0_IP" --prefiller-ports 8000 \
        --decoder-hosts "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
            "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
            "$D2_IP" "$D2_IP" "$D2_IP" "$D2_IP" \
            "$D3_IP" "$D3_IP" "$D3_IP" "$D3_IP" \
        --decoder-ports 8000 8001 8002 8003 8000 8001 8002 8003 \
            8000 8001 8002 8003 8000 8001 8002 8003
    ```


    **Key Parameter Descriptions — 950DT products**

    The following settings apply only to this 950DT products tab; use the
    topology and scripts in the other tabs for A3 or A2 series.

    - `--pipeline-parallel-size 2` and `--tensor-parallel-size 8` define
      Prefill across two nodes, with `VLLM_PP_LAYER_PARTITION="42,36"`.
      Only PP rank 0 exposes the Prefill API.
    - The Decode launcher uses `--dp-size 16`, `--tp-size 2`, and
      `--dp-size-local 4` across four nodes. Each node exposes API ports
      `8000-8003`; register all 16 endpoints with the proxy.
    - Both roles use `--max-model-len 1048576`. Prefill sets
      `--max-num-seqs 8` and `--max-num-batched-tokens 4096`;
      Decode sets `10` and `60`, respectively, per engine.
    - Both roles enable prefix caching and FP8 attention and indexer KV
      caches. MTP uses an eager draft with one speculative token on
      Prefill and five on Decode. Prefill runs in eager mode; Decode uses
      `FULL_DECODE_ONLY` graph mode.
    - `MultiConnector` combines `MooncakeConnectorV2` for P-to-D transfer
      with `AscendStoreConnector` for KV pooling. Both roles declare
      Prefill DP1/PP2/TP8 and Decode DP16/TP2, and set `use_layerwise: true`.
      Mooncake base KV ports are `30000` on Prefill and `30300` on Decode.
    - `MMC_LOCAL_CONFIG_PATH` selects each node's MemCache configuration.
      Prepare the backend before starting the engines and verify shared
      pool reuse through its hit/load metrics.
    - Prefill enables DSA CP, FlashComm1 and `enable_fused_mc2: 1`.
      Decode enables MLAPO and shared-expert multistream overlap.
      Both roles enable recomputation scheduling.

=== "A3 series"

    ##### A3 series {: #glm52-53-5-1-4-a3 }

    This extends the same four-node PP/PD topology with `MultiConnector`:
    `MooncakeConnectorV2` transfers KV cache from Prefill to Decode, and
    `AscendStoreConnector` provides non-layerwise MemCache pooling. Retain
    the launcher, PP placement, and proxy routing from the A3 tab of [Section 5.1.3](#513-prefill-decode-disaggregation).
    This is the supplied 198k pooled variant, with `--max-model-len 200000`.
    Both roles enable prefix caching; unlike the non-pooled variant, Decode
    must not use `--no-enable-prefix-caching`. Prefill retains fused MC2 mode 2
    and shared-expert DP, while Decode uses NPU graph/static-kernel compilation
    and shared-expert multistream overlap without fused MC2. The same build,
    HCCS fabric, and static-kernel compatibility checks apply. Pool-specific
    `PYTHONHASHSEED=0`, `ACL_OP_INIT_MODE=1`, and MemCache paths are set below.

    ###### Prepare MemCache

    Complete [MemCache backend setup](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend),
    including the hardware/CANN prerequisites, `memfabric-hybrid` and
    `memcache-hybrid` installation, configuration files, and metadata service.
    All P/D instances must access the same pool; connector arguments do not
    start the MemCache service. Set `MEMCACHE_ROOT` to the installed
    `memcache_hybrid` directory and `PYTHON_LIB_DIR` to the library directory
    of the Python installation actually used by vLLM on each node.

    ###### Prepare the Scripts

    Before you start, please do the following:

    1. prepare the script `launch_online_dp.py` on each Decode node, using the
       launcher in [Prepare the Scripts](#glm53-pd-a3-scripts).

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

            export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
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
                --served-model-name glm \
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
                    "enable_fused_mc2": 2,
                    "enable_flashcomm1": true,
                    "enable_shared_expert_dp": true
                }' \
                --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
                --kv-transfer-config '{
                    "kv_connector": "MultiConnector",
                    "kv_role": "kv_producer",
                    "engine_id": "0",
                    "kv_port": "30000",
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

            export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
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
                --served-model-name glm \
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
                    "enable_fused_mc2": 2,
                    "enable_flashcomm1": true,
                    "enable_shared_expert_dp": true
                }' \
                --speculative-config '{"num_speculative_tokens": 1,  "method":"deepseek_mtp","enforce_eager":true}' \
                --kv-transfer-config '{
                    "kv_connector": "MultiConnector",
                    "kv_role": "kv_producer",
                    "engine_id": "1",
                    "kv_port": "30000",
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

            export OMP_PROC_BIND=false
            export OMP_NUM_THREADS=10
            export LOCAL_WORLD_SIZE=16
            export ASCEND_RT_VISIBLE_DEVICES=$1

            export PYTHONHASHSEED=0
            export ACL_OP_INIT_MODE=1
            export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
            export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

            export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
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
                --served-model-name glm \
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
                    "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": true},
                    "multistream_overlap_shared_expert": true
                }' \
                --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
                --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
                --kv-transfer-config '{
                    "kv_connector": "MultiConnector",
                    "kv_role": "kv_consumer",
                    "engine_id": "2",
                    "kv_port": "30200",
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

            export OMP_PROC_BIND=false
            export OMP_NUM_THREADS=10
            export LOCAL_WORLD_SIZE=16
            export ASCEND_RT_VISIBLE_DEVICES=$1

            export PYTHONHASHSEED=0
            export ACL_OP_INIT_MODE=1
            export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-local.conf"
            export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"

            export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
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
                --served-model-name glm \
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
                    "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": true},
                    "multistream_overlap_shared_expert": true
                }' \
                --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
                --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
                --kv-transfer-config '{
                    "kv_connector": "MultiConnector",
                    "kv_role": "kv_consumer",
                    "engine_id": "3",
                    "kv_port": "30200",
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
    30200 on Decode.
    `lookup_rpc_port="0"` is retained: Ascend Store uses an IPC endpoint that
    also includes the DP rank, and P/D run on separate hosts. It is not a TCP
    listener on port zero. `backend="memcache"` and `use_layerwise=false`
    are set on both sides.

    <a id="glm53-pool-a3-start"></a>

    ###### Start the Engines

    Start the MemCache metadata service and complete its configuration on all
    four nodes before starting vLLM.

    Once preparation is complete, run the following commands on each node from the directory containing the scripts:

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

    ###### Start the Proxy and Verify Pooling

    1. Start the MemCache metadata service and complete its configuration on
       all four nodes before starting vLLM.
    2. Start P0, P1, D0, and D1 with the commands in
       [Start the Engines](#glm53-pool-a3-start). Both Prefill scripts
       already specify port 8000.
    3. Wait for P0:8000 and all sixteen Decode APIs to become healthy. Start
       the [A3 PD proxy](#glm53-pd-a3-proxy) on a separate host, using the same
       Prefill endpoint and Decode ports 8000-8007. Its public port is 8000.
    4. Send requests to the proxy using the served model name `glm`:

    ```shell
    curl "http://<proxy_ip>:8000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "glm",
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
    cache. These A3 references do not
    establish A2, 1M-context, or other parallel-layout support, and do not add
    accuracy or performance claims for the 198k workload.

=== "A2 series"

    ##### A2 series {: #glm52-53-5-1-4-a2 }

    <a id="glm53-pool-a2"></a>

    ###### Eight-Node Deployment with MemCache KV Cache Pool

    This A2 pooling reference uses four Prefill nodes and four Decode nodes,
    with eight devices per node. One Prefill instance spans four nodes with
    TP8 × PP4 and has DP size 1 with one API server. Decode uses one DP16 × TP2
    group across four nodes.

    | Role | Nodes | Parallelism | Placement | Devices per role |
    | --- | --- | --- | --- | --- |
    | Prefill | P0-P3 (1 instance) | TP8 × PP4 | One instance on P0-P3 | 32 |
    | Decode | D0-D3 | DP16 × TP2 | Four local DP ranks per node, starting at 0/4/8/12 | 32 |

    P0 exposes the API on port 8000; P1-P3 are headless PP partners.
    Decode exposes ports 8000-8003 on each node, and the proxy listens on
    1999. This pooled PP4 layout uses a wider Prefill partition than the
    non-pooled A2 PD deployment and adds MemCache KV pooling.
    Use the A2 image from Section 4 and the
    [A2 Decode launcher](#glm53-pd-a2-launcher).

    The pooled Prefill and Decode engines both use `--max-model-len 200000`,
    with MTP token counts 1 and 5, respectively.
    The end-to-end limit is **200,000 tokens**, including input and output.
    Both roles declare the actual Prefill DP1/PP4/TP8 partition `20,20,20,18` and
    Decode DP16/TP2 topology.

    <a id="glm53-pool-a2-prepare"></a>

    ###### Prepare MemCache

    Complete [MemCache backend setup](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend)
    on every node, including the hardware/CANN prerequisites and installation
    of `memfabric-hybrid` and `memcache-hybrid`. Install `fastokens` on all
    four Prefill nodes. All eight nodes must reach the same MetaService.

    On every Prefill and Decode **host**, configure hugepages before
    starting MemCache services:

    ```shell
    # Run on every Prefill and Decode host before starting MemCache services.
    sudo sysctl -w vm.nr_hugepages=200000
    sysctl -n vm.nr_hugepages
    ```

    The check must print:

    ```text
    200000
    ```

    This is a host prerequisite for the `device_rdma` configuration with
    `40GB` DRAM per LocalService; check all eight hosts before proceeding.

    Prepare `mmc-meta.conf` for the MetaService on `<MEMCACHE_META_IP>`:

    ```ini
    ock.mmc.meta_service_url = tcp://<MEMCACHE_META_IP>:5000
    ock.mmc.meta_service.config_store_url = tcp://<MEMCACHE_META_IP>:6000
    ock.mmc.meta_service.metrics_url = http://<MEMCACHE_META_IP>:8009
    ock.mmc.log_level = info
    ```

    Prepare `mmc-local.conf` in shared storage, or an identical copy on every
    node. Use the A2/RoCE protocol and allocate `40GB` of pooled DRAM per
    LocalService:

    ```ini
    ock.mmc.meta_service_url = tcp://<MEMCACHE_META_IP>:5000
    ock.mmc.local_service.config_store_url = tcp://<MEMCACHE_META_IP>:6000
    ock.mmc.log_level = error
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_rdma
    ock.mmc.local_service.dram.size = 40GB
    ```

    `world_size=256` is the maximum number of LocalServices supported, with
    headroom; it is not the number of NPUs in this deployment. `dram.size`
    is allocated per LocalService: eight LocalServices contribute `320GB`
    per fully populated A2 node. MetaService ports 5000/6000, metrics port
    8009, Prefill master port 7060 on the Prefill instance leader, Decode DP RPC
    port 12990, API/proxy ports, and the Mooncake ports derived from base
    ports 30000/30200 must be reachable as appropriate.

    On **each** Prefill and Decode node, export the following settings using
    that node's installed paths and interface:

    ```shell
    export MODEL_PATH="<YOUR_MODEL_PATH>"
    export NIC_NAME="<NETWORK_INTERFACE>"
    export PYTHON_LIB_DIR="<PYTHON_LIB_DIR>"
    export MEMCACHE_ROOT="<MEMCACHE_ROOT>"
    export MMC_LOCAL_CONFIG_PATH="<PATH_TO_MMC_LOCAL_CONF>"
    ```

    The scripts use `MMC_LOCAL_CONFIG_PATH` to locate the same local
    configuration. On the MetaService host, set `MEMCACHE_ROOT` to its
    installation directory and place `mmc-meta.conf` under its `config`
    directory, or set `MMC_META_CONFIG_PATH` to the actual file path.

    ###### Prefill Engine Script (Pooled)

    Save this script as `start_p_pool.sh` on P0-P3. The arguments select the
    PP node rank, API port, local IP, instance leader IP, and visible devices.
    All four PP stages of one instance use the same master address and partition.
    The Prefill instance uses its own master address.

    ```shell
    #!/usr/bin/env bash
    set -eo pipefail

    NODE_RANK=${1:?need node_rank}
    PORT=${2:-8000}
    LOCAL_IP=${3:?need local_ip}
    INSTANCE_LEADER_IP=${4:?need instance_leader_ip}
    VISIBLE_DEVICES=${5:-0,1,2,3,4,5,6,7}
    : "${MODEL_PATH:?export MODEL_PATH}"
    : "${NIC_NAME:?export NIC_NAME}"
    : "${MEMCACHE_ROOT:?export MEMCACHE_ROOT}"
    : "${PYTHON_LIB_DIR:?export PYTHON_LIB_DIR}"
    : "${MMC_LOCAL_CONFIG_PATH:?export MMC_LOCAL_CONFIG_PATH}"

    case "$NODE_RANK" in
      0) SERVER_ROLE_ARGS=(--api-server-count 1);;
      1|2|3) SERVER_ROLE_ARGS=(--headless);;
      *) echo "node_rank must be 0-3" >&2; exit 1;;
    esac

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY ALL_PROXY all_proxy
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

    export VLLM_SERVER_DEV_MODE=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=1800
    export HCCL_CONNECT_TIMEOUT=1800
    export VLLM_HOST_IP="$LOCAL_IP"
    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH:-}"
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE=AIV
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=1
    export VLLM_USE_V2_MODEL_RUNNER=0
    export VLLM_USE_FASTOKENS=1
    export VLLM_PP_LAYER_PARTITION="20,20,20,18"
    export PYTHONHASHSEED=0
    export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"
    export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'
    export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"

    exec vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port "$PORT" \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --pipeline-parallel-size 4 \
      --distributed-executor-backend mp \
      --master-addr "$INSTANCE_LEADER_IP" \
      --master-port 7060 \
      --nnodes 4 \
      --node-rank "$NODE_RANK" \
      --enable-chunked-prefill \
      --no-async-scheduling \
      --enable-prefix-caching \
      --seed 1024 \
      --served-model-name glm \
      --max-model-len 200000 \
      --max-num-batched-tokens 8192 \
      --max-num-seqs 64 \
      --gpu-memory-utilization 0.92 \
      --trust-remote-code \
      --quantization ascend \
      --safetensors-load-strategy prefetch \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --enforce-eager \
      --kv-cache-dtype int8 \
      --attention_config.indexer_kv_dtype int8 \
      --additional-config '{"enable_dsa_cp":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true,"scheduler_config":{"profiling_chunk_config":{"enabled":true}}}' \
      --speculative-config '{"num_speculative_tokens":1,"method":"deepseek_mtp","enforce_eager":true}' \
      --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "engine_id": "0",
        "kv_port": "30000",
        "kv_connector_extra_config": {
          "connectors": [
            {
              "kv_connector": "MooncakeConnectorV2",
              "kv_role": "kv_producer",
              "kv_port": "30000",
              "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 1, "pp_size": 4, "tp_size": 8, "pp_layer_partition": "20,20,20,18"},
                "decode": {"dp_size": 16, "tp_size": 2}
              }
            },
            {
              "kv_connector": "AscendStoreConnector",
              "kv_role": "kv_producer",
              "kv_connector_extra_config": {
                "lookup_rpc_port": "0",
                "backend": "memcache",
                "use_layerwise": false
              }
            }
          ]
        }
      }' \
      "${SERVER_ROLE_ARGS[@]}"
    ```

    The Prefill script keeps `--no-async-scheduling` together with the V1
    runner and `profiling_chunk_config.enabled=true`. Enabling async
    scheduling with this combination is rejected by the runtime.

    PP headless workers may fail with
    `Mooncake worker requires a local DP rank`. The pooled Prefill instance
    has DP size 1. Check whether the conditional fix in
    [Local DP Rank Compatibility and Troubleshooting](#glm53-a2-rc1-compatibility)
    applies to the installed worker on all Prefill nodes before starting
    it.

    ###### Decode Engine Script (Pooled)

    Save this script as `start_d_engine_pool.sh` on D0-D3. Configure the
    current node's `LOCAL_IP` in each copy and export the paths and NIC
    settings from [Prepare MemCache](#glm53-pool-a2-prepare).
    The A2 launcher supplies `$1`-`$7`.

    ```shell
    #!/usr/bin/env bash
    set -eo pipefail

    VISIBLE_DEVICES=${1:?need visible_devices}
    PORT=${2:?need port}
    DP_SIZE=${3:?need dp_size}
    DP_RANK=${4:?need dp_rank}
    DP_ADDRESS=${5:?need dp_address}
    DP_RPC_PORT=${6:?need dp_rpc_port}
    TP_SIZE=${7:?need tp_size}
    LOCAL_IP="<CURRENT_DECODE_NODE_IP>"
    : "${MODEL_PATH:?export MODEL_PATH}"
    : "${NIC_NAME:?export NIC_NAME}"
    : "${MEMCACHE_ROOT:?export MEMCACHE_ROOT}"
    : "${PYTHON_LIB_DIR:?export PYTHON_LIB_DIR}"
    : "${MMC_LOCAL_CONFIG_PATH:?export MMC_LOCAL_CONFIG_PATH}"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY ALL_PROXY all_proxy
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

    export VLLM_SERVER_DEV_MODE=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=1800
    export HCCL_CONNECT_TIMEOUT=1800
    export ASCEND_TRANSFER_TIMEOUT=10000
    export VLLM_HOST_IP="$LOCAL_IP"
    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:/usr/local/lib:${LD_LIBRARY_PATH:-}"
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE=AIV
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=1
    export VLLM_USE_V2_MODEL_RUNNER=0
    export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
    export PYTHONHASHSEED=0
    export ACL_OP_INIT_MODE=1
    export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}"
    export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

    exec vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port "$PORT" \
      --data-parallel-size "$DP_SIZE" \
      --data-parallel-rank "$DP_RANK" \
      --data-parallel-address "$DP_ADDRESS" \
      --data-parallel-rpc-port "$DP_RPC_PORT" \
      --tensor-parallel-size "$TP_SIZE" \
      --enable-expert-parallel \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --seed 1024 \
      --served-model-name glm \
      --max-model-len 200000 \
      --max-num-batched-tokens 256 \
      --max-num-seqs 64 \
      --gpu-memory-utilization 0.92 \
      --async-scheduling \
      --trust-remote-code \
      --quantization ascend \
      --safetensors-load-strategy prefetch \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --kv-cache-dtype int8 \
      --attention_config.indexer_kv_dtype int8 \
      --additional-config '{"recompute_scheduler_enable":true,"enable_shared_expert_dp":true}' \
      --speculative-config '{"num_speculative_tokens":5,"method":"deepseek_mtp","enforce_eager":true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "engine_id": "1",
        "kv_port": "30200",
        "kv_connector_extra_config": {
          "connectors": [
            {
              "kv_connector": "MooncakeConnectorV2",
              "kv_role": "kv_consumer",
              "kv_port": "30200",
              "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 1, "pp_size": 4, "tp_size": 8, "pp_layer_partition": "20,20,20,18"},
                "decode": {"dp_size": 16, "tp_size": 2}
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

    Both scripts use `MultiConnector` with `MooncakeConnectorV2` for KV
    transfer and `AscendStoreConnector` for non-layerwise MemCache pooling.
    `lookup_rpc_port="0"` identifies an IPC endpoint that also includes the
    DP rank; it is not a TCP listener on port zero.

    ###### Start Services

    1. Start the MetaService on `<MEMCACHE_META_IP>` before any vLLM process:

        ```shell
        export MMC_META_CONFIG_PATH="${MEMCACHE_ROOT}/config/mmc-meta.conf"
        python3 -c "from memcache_hybrid import MetaService; MetaService.main()"
        ```

    2. Export the settings from [Prepare MemCache](#glm53-pool-a2-prepare)
       on **every** Prefill node. Start both PP stages of each instance in
       separate terminals before waiting for the instance to become ready:

        ```shell
        # P0-P3: one Prefill instance, with P0 as API leader
        bash start_p_pool.sh 0 8000 "<PREFILL_NODE0_IP>" "<PREFILL_NODE0_IP>" 0,1,2,3,4,5,6,7
        bash start_p_pool.sh 1 8000 "<PREFILL_NODE1_IP>" "<PREFILL_NODE0_IP>" 0,1,2,3,4,5,6,7
        bash start_p_pool.sh 2 8000 "<PREFILL_NODE2_IP>" "<PREFILL_NODE0_IP>" 0,1,2,3,4,5,6,7
        bash start_p_pool.sh 3 8000 "<PREFILL_NODE3_IP>" "<PREFILL_NODE0_IP>" 0,1,2,3,4,5,6,7
        ```

    3. Export the settings from [Prepare MemCache](#glm53-pool-a2-prepare)
       on every Decode node and start all four nodes before waiting for the
       DP group to become ready:

        ```shell
        # D0: DP ranks 0-3
        python launch_online_dp.py --script ./start_d_engine_pool.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 0 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 8000

        # D1: DP ranks 4-7
        python launch_online_dp.py --script ./start_d_engine_pool.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 4 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 8000

        # D2: DP ranks 8-11
        python launch_online_dp.py --script ./start_d_engine_pool.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 8 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 8000

        # D3: DP ranks 12-15
        python launch_online_dp.py --script ./start_d_engine_pool.sh --dp-size 16 --tp-size 2 --dp-size-local 4 --dp-rank-start 12 --dp-address "<DECODE_NODE0_IP>" --dp-rpc-port 12990 --vllm-start-port 8000
        ```

    4. Confirm that the Decode worker logs report completion of the MemCache
       DP initialization barrier before treating the group as ready:

        ```text
        All DP MemCache initializations completed
        ```

        The Prefill instance has DP size 1 and does not enter this DP
        barrier; check their logs for successful MemCache initialization and
        verify that P1-P3 joined the Prefill PP instance. They do not expose
        HTTP APIs. On every node, confirm that the active `mmc-local.conf`
        uses `device_rdma` and `dram.size = 40GB`.

        Wait for the API on P0:8000 and all sixteen Decode APIs on
        ports 8000-8003 to answer `GET /v1/models`. Use `curl --noproxy '*'`
        for these checks when HTTP proxies are configured. Start the proxy
        on P0 or another reachable host:

        ```shell
        P0_IP="<PREFILL_NODE0_IP>"
        D0_IP="<DECODE_NODE0_IP>"
        D1_IP="<DECODE_NODE1_IP>"
        D2_IP="<DECODE_NODE2_IP>"
        D3_IP="<DECODE_NODE3_IP>"

        unset ftp_proxy FTP_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY ALL_PROXY all_proxy

        python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
          --host 0.0.0.0 --port 1999 --workers 8 \
          --prefiller-hosts "$P0_IP" --prefiller-ports 8000 \
          --decoder-hosts "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
            "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
            "$D2_IP" "$D2_IP" "$D2_IP" "$D2_IP" \
            "$D3_IP" "$D3_IP" "$D3_IP" "$D3_IP" \
          --decoder-ports 8000 8001 8002 8003 8000 8001 8002 8003 \
            8000 8001 8002 8003 8000 8001 8002 8003
        ```

### 5.2 1M Context Configuration {: #52-1m-context-configuration }

Shared parameter meanings are listed in [Common Parameter Descriptions](#glm-common-parameters).
Topology, role-specific limits and connector settings are explained in each
hardware tab; apply only the settings for the selected platform.

Select the tab that matches your deployment hardware.

=== "950DT products"

    #### 950DT products {: #glm52-53-5-2-950dt }

    ##### Topology and Prerequisites

    See [950DT preparation](#glm-950-preparation) for checkpoint, environment and PP-node settings.

    This is a separate FP8 reference from the A3 int8 1M example. Both roles use DP2/TP8 with MooncakeConnectorV2; Prefill enables KVPP and PCP1, while Decode explicitly sets DCP8. Do not add PP2 or copy A3 DCP settings into this scenario.

    The source sets Prefill to **1,046,576** tokens and Decode to **1,048,576**. These values are preserved, not silently rounded. Until validated and reconciled, keep each request's total input plus requested output within **1,046,576** tokens, the smaller engine limit. Prefill has a 32,768-token batch budget and Decode 192 tokens.

    ##### Prepare the Scripts

    1. prepare the script `launch_online_dp.py` on each node:

        Use the [950DT DP launcher](#glm-950-dp-launcher) from [Section 5.1.3](#513-prefill-decode-disaggregation).

    2. prepare the script `run_dp_template.sh` on each node.

        1. Prefill nodes 0 and 1

            ```shell
            #!/usr/bin/env bash

            : "${MODEL_PATH:?Set the checkpoint path on every node}"
            : "${LOCAL_IP:?Set this node address}"
            : "${NIC_NAME:?Set the matching network interface}"
            local_ip="$LOCAL_IP"
            nic_name="$NIC_NAME"

            unset ftp_proxy FTP_PROXY
            unset https_proxy HTTPS_PROXY
            unset http_proxy HTTP_PROXY
            export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
            export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
            unset ftp_proxy
            unset https_proxy
            unset http_proxy
            unset HCCL_INTRA_ROCE_ENABLE

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

            LOG_DIR="${LOG_DIR:-$PWD/logs}"
            mkdir -p "$LOG_DIR"
            export ASCEND_PROCESS_LOG_PATH="$LOG_DIR"

            export VLLM_USE_FASTOKENS=1

            export VLLM_DISABLE_COMPILE_CACHE=1

            export VLLM_USE_V2_MODEL_RUNNER=0
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

            vllm serve "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --prefill-context-parallel-size 1 \
            --cp-kv-cache-interleave-size 128 \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm \
            --max-model-len 1046576 \
            --safetensors-load-strategy 'prefetch' \
            --additional-config '{
                "enable_dsa_cp":true,
                "enable_fused_mc2": 1,
                "enable_flashcomm1": true,
                "enable_kvpp": true
            }' \
            --max-num-batched-tokens 32768 \
            --trust-remote-code \
            --enable-prefix-caching \
            --max-num-seqs 64 \
            --quantization ascend \
            --gpu-memory-utilization 0.8 \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm47 \
            --kv-cache-dtype fp8 \
            --attention_config.indexer_kv_dtype fp8 \
            --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_producer",
                "engine_id": "0",
                "kv_port": "30000",
                "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 2, "tp_size": 8},
                "decode": { "dp_size": 2, "tp_size": 8}
                }
            }'
            ```

        2. Decode nodes 0 and 1

            ```shell
            #!/usr/bin/env bash

            : "${MODEL_PATH:?Set the checkpoint path on every node}"
            : "${LOCAL_IP:?Set this node address}"
            : "${NIC_NAME:?Set the matching network interface}"
            local_ip="$LOCAL_IP"
            nic_name="$NIC_NAME"

            export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
            export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
            export VLLM_ASCEND_DP_ALLREDUCE_ON_NPU=1

            unset ftp_proxy
            unset https_proxy
            unset http_proxy
            unset HCCL_INTRA_ROCE_ENABLE

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

            LOG_DIR="${LOG_DIR:-$PWD/logs}"
            mkdir -p "$LOG_DIR"
            export ASCEND_PROCESS_LOG_PATH="$LOG_DIR"

            export ASCEND_RT_VISIBLE_DEVICES=$1
            export VLLM_USE_FASTOKENS=1

            export VLLM_DISABLE_COMPILE_CACHE=1

            export VLLM_USE_V2_MODEL_RUNNER=0
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

            vllm serve "$MODEL_PATH" \
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
            --served-model-name glm \
            --max-model-len 1048576 \
            --safetensors-load-strategy 'prefetch' \
            --max-num-batched-tokens 192 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
            --additional-config '{
                "recompute_scheduler_enable":true,
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
            --kv-cache-dtype fp8 \
            --attention_config.indexer_kv_dtype fp8 \
            --kv-transfer-config '{
            "kv_connector": "MooncakeConnectorV2",
            "kv_role": "kv_consumer",
            "engine_id": "1",
            "kv_port": "30200",
            "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {"dp_size": 2, "tp_size": 8},
            "decode": { "dp_size": 2, "tp_size":8}
            }
            }'
            ```

    ##### Start the Engines

    Export the node-specific `MODEL_PATH`, `LOCAL_IP` and `NIC_NAME`, and the shared
    `P0_IP` / `D0_IP` addresses before starting the corresponding node script.

    1. Prefill node 0

        ```shell
        python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address "$P0_IP" --dp-rpc-port 16500 --vllm-start-port 8000
        ```

    2. Prefill node 1

        ```shell
        python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address "$P0_IP" --dp-rpc-port 16500 --vllm-start-port 8000
        ```

    3. Decode node 0

        ```shell
        python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
        ```

    4. Decode node 1

        ```shell
        python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
        ```

    ##### Start the Proxy

    Run the proxy on a separate host after all API endpoints are healthy.

    ```shell
    P0_IP="<P0_IP>"
    P1_IP="<P1_IP>"
    D0_IP="<D0_IP>"
    D1_IP="<D1_IP>"

    python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 8000 \
        --prefiller-hosts "$P0_IP" "$P1_IP" --prefiller-ports 8000 8000 \
        --decoder-hosts "$D0_IP" "$D1_IP" \
        --decoder-ports 8000 8000
    ```

    **Key Parameter Descriptions — 950DT products**

    - `--data-parallel-size 2` and `--tensor-parallel-size 8` define both
      roles. Each role spans two nodes, with one TP8 engine per node and
      a separate DP group. Each node exposes API port `8000`.
    - Prefill sets `enable_kvpp: true` in `--additional-config`;
      Decode sets `--decode-context-parallel-size 8` to shard KV cache
      across its TP group.
    - `--max-model-len` is `1046576` on Prefill and `1048576` on Decode.
      Prefill sets `--max-num-seqs 64` and
      `--max-num-batched-tokens 32768`; Decode sets `32` and `192`,
      respectively. Tune concurrency for the available KV cache capacity.
    - MTP uses `deepseek_mtp` with an eager draft: one speculative token
      on Prefill and five on Decode. Prefill uses `--enforce-eager`;
      Decode uses `FULL_DECODE_ONLY` graph mode.
    - `MooncakeConnectorV2` uses `kv_producer` on Prefill and `kv_consumer`
      on Decode, with base KV ports `30000` and `30200`, respectively.
      Both roles declare Prefill DP2/TP8 and Decode DP2/TP8.
    - Both roles use FP8 attention and indexer KV caches and
      `enable_fused_mc2: 1`. Prefill also enables DSA CP, FlashComm1 and
      prefix caching.

=== "A3 series"

    #### A3 series {: #glm52-53-5-2-a3 }

    This configuration uses GLM-5.3 W8A8C8 weights on four A3 series
    nodes, with 16 logical NPU devices per node. Use the A3 image from
    [Docker Image Installation](#4-installation) on all nodes.
    The maximum context length is 1,048,576 tokens, including input and output.

    ##### Topology and Prerequisites

    | Role | Nodes | Global DP | TP | PP | DCP | Local DP ranks | Devices per role |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | Prefill | P0, P1 | 1 | 16 | 2 | 16 | One per node | 32 |
    | Decode | D0, D1 | 8 | 4 | 1 | 4 | Four per node | 32 |

    P0 and P1 are the two stages of one PP engine, using
    `VLLM_PP_LAYER_PARTITION="42,36"`. Only P0 exposes the Prefill API at
    port 8077; P1 is headless. D0 runs DP ranks 0-3 and D1 runs ranks 4-7,
    with API ports 8000-8003 on each node. PCP is 1 and KV interleave is
    128 on both roles.

    1. Prepare identical weights and software on all four nodes and verify
       [multi-node communication](../../getting_started/installation.md#installation-multi-node-interconnect).
       This configuration selects `hccs:device` and communication resource
       version `1.2`; use a compatible HCCS network and software build.
    2. Set `LOCAL_IP`, `NIC_NAME`, `MODEL_PATH`, and `PYTHON_LIB_DIR` on each
       node. `PYTHON_LIB_DIR` is the library directory of the Python installation
       used by vLLM. Install `fastokens` on both Prefill nodes.
       Set `P0_IP` to P0's address on both Prefill nodes.
    3. Reserve PP master port 7060, Decode DP RPC port 16600, P0 API port 8077,
       Decode API ports 8000-8003, and Mooncake ports derived from the base
       ports 30000 and 30100. Run the proxy on a separate host at port 8000.

    ##### Prepare the Scripts

    1. Save `launch_online_dp.py` on each Decode node:

        Use the [A3 DP launcher](#glm53-pd-a3-scripts) from [Section 5.1.3](#513-prefill-decode-disaggregation) unchanged. Save it beside the Decode template on each Decode node.


    2. prepare the script `run_dp_template.sh` on each node.

        1. Prefill node 0

            ```shell
            #!/usr/bin/env bash

            LOCAL_IP="<PREFILL_NODE_IP>"
            NIC_NAME="<NETWORK_INTERFACE>"
            MODEL_PATH="<YOUR_MODEL_PATH>"
            PYTHON_LIB_DIR="<PYTHON_LIBRARY_DIRECTORY>"
            P0_IP="<PREFILL_NODE0_IP>"

            export HCCL_OP_EXPANSION_MODE="AIV"
            export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
            export HCCL_EXEC_TIMEOUT=1800
            export HCCL_CONNECT_TIMEOUT=1800
            export ASCEND_TRANSFER_TIMEOUT=10000

            export VLLM_HOST_IP="$LOCAL_IP"
            export HCCL_IF_IP="$LOCAL_IP"
            export GLOO_SOCKET_IFNAME="$NIC_NAME"
            export TP_SOCKET_IFNAME="$NIC_NAME"
            export HCCL_SOCKET_IFNAME="$NIC_NAME"

            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export HCCL_BUFFSIZE=1024

            export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

            export LD_LIBRARY_PATH="$PYTHON_LIB_DIR:/usr/local/lib:${LD_LIBRARY_PATH:-}"

            export VLLM_USE_V2_MODEL_RUNNER=0

            export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
            export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'
            export TASK_QUEUE_ENABLE=1
            export VLLM_USE_FASTOKENS=1
            export VLLM_PP_LAYER_PARTITION="42,36"

            exec vllm serve "$MODEL_PATH" \
                --host 0.0.0.0 \
                --port 8077 \
                --tensor-parallel-size 16 \
                --enable-expert-parallel \
                --prefill-context-parallel-size 1 \
                --decode-context-parallel-size 16 \
                --cp-kv-cache-interleave-size 128 \
                --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
                --seed 1024 \
                --served-model-name glm \
                --max-model-len 1048576 \
                --safetensors-load-strategy 'prefetch' \
                --additional-config '{
                    "enable_dsa_cp": true,
                    "enable_fused_mc2": 2,
                    "enable_flashcomm1": true,
                    "enable_shared_expert_dp": true
                }' \
                --max-num-batched-tokens 16384 \
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
                --pipeline-parallel-size 2 \
                --distributed-executor-backend mp \
                --master-addr "$P0_IP" \
                --master-port 7060 \
                --nnodes 2 \
                --node-rank 0 \
                --enable-chunked-prefill \
                --async-scheduling \
                --kv-transfer-config '{
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_producer",
                    "engine_id": "0",
                    "kv_port": "30000",
                    "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {
                            "dp_size": 1,
                            "tp_size": 16,
                            "pp_size": 2,
                            "pp_layer_partition": "42,36"
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
            #!/usr/bin/env bash

            LOCAL_IP="<PREFILL_NODE_IP>"
            NIC_NAME="<NETWORK_INTERFACE>"
            MODEL_PATH="<YOUR_MODEL_PATH>"
            PYTHON_LIB_DIR="<PYTHON_LIBRARY_DIRECTORY>"
            P0_IP="<PREFILL_NODE0_IP>"

            export HCCL_OP_EXPANSION_MODE="AIV"
            export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
            export HCCL_EXEC_TIMEOUT=1800
            export HCCL_CONNECT_TIMEOUT=1800
            export ASCEND_TRANSFER_TIMEOUT=10000

            export VLLM_HOST_IP="$LOCAL_IP"
            export HCCL_IF_IP="$LOCAL_IP"
            export GLOO_SOCKET_IFNAME="$NIC_NAME"
            export TP_SOCKET_IFNAME="$NIC_NAME"
            export HCCL_SOCKET_IFNAME="$NIC_NAME"

            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export HCCL_BUFFSIZE=1024

            export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

            export LD_LIBRARY_PATH="$PYTHON_LIB_DIR:/usr/local/lib:${LD_LIBRARY_PATH:-}"

            export VLLM_USE_V2_MODEL_RUNNER=0

            export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
            export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'
            export TASK_QUEUE_ENABLE=1
            export VLLM_USE_FASTOKENS=1
            export VLLM_PP_LAYER_PARTITION="42,36"

            exec vllm serve "$MODEL_PATH" \
                --host 0.0.0.0 \
                --port 8077 \
                --tensor-parallel-size 16 \
                --enable-expert-parallel \
                --prefill-context-parallel-size 1 \
                --decode-context-parallel-size 16 \
                --cp-kv-cache-interleave-size 128 \
                --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
                --seed 1024 \
                --served-model-name glm \
                --max-model-len 1048576 \
                --safetensors-load-strategy 'prefetch' \
                --additional-config '{
                    "enable_dsa_cp": true,
                    "enable_fused_mc2": 2,
                    "enable_flashcomm1": true,
                    "enable_shared_expert_dp": true
                }' \
                --max-num-batched-tokens 16384 \
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
                --pipeline-parallel-size 2 \
                --distributed-executor-backend mp \
                --master-addr "$P0_IP" \
                --master-port 7060 \
                --nnodes 2 \
                --node-rank 1 \
                --headless \
                --enable-chunked-prefill \
                --async-scheduling \
                --kv-transfer-config '{
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_producer",
                    "engine_id": "1",
                    "kv_port": "30000",
                    "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {
                            "dp_size": 1,
                            "tp_size": 16,
                            "pp_size": 2,
                            "pp_layer_partition": "42,36"
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
            #!/usr/bin/env bash

            LOCAL_IP="<DECODE_NODE_IP>"
            NIC_NAME="<NETWORK_INTERFACE>"
            MODEL_PATH="<YOUR_MODEL_PATH>"
            PYTHON_LIB_DIR="<PYTHON_LIBRARY_DIRECTORY>"

            export HCCL_OP_EXPANSION_MODE="AIV"
            export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
            export HCCL_EXEC_TIMEOUT=1800
            export HCCL_CONNECT_TIMEOUT=1800
            export ASCEND_TRANSFER_TIMEOUT=10000

            export VLLM_HOST_IP="$LOCAL_IP"
            export HCCL_IF_IP="$LOCAL_IP"
            export GLOO_SOCKET_IFNAME="$NIC_NAME"
            export TP_SOCKET_IFNAME="$NIC_NAME"
            export HCCL_SOCKET_IFNAME="$NIC_NAME"

            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export HCCL_BUFFSIZE=1024

            export TASK_QUEUE_ENABLE=1
            export ASCEND_RT_VISIBLE_DEVICES="$1"

            export LD_LIBRARY_PATH="$PYTHON_LIB_DIR:/usr/local/lib:${LD_LIBRARY_PATH:-}"
            export VLLM_USE_V2_MODEL_RUNNER=0

            export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
            export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'
            export LOCAL_WORLD_SIZE=16

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
                --decode-context-parallel-size 4 \
                --cp-kv-cache-interleave-size 128 \
                --seed 1024 \
                --served-model-name glm \
                --max-model-len 1048576 \
                --safetensors-load-strategy 'prefetch' \
                --max-num-batched-tokens 48 \
                --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
                --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
                --additional-config '{
                    "recompute_scheduler_enable": true,
                    "ascend_compilation_config": {
                        "enable_npugraph_ex": true,
                        "enable_static_kernel": true
                    },
                    "multistream_overlap_shared_expert": true
                }' \
                --trust-remote-code \
                --max-num-seqs 8 \
                --gpu-memory-utilization 0.90 \
                --async-scheduling \
                --quantization ascend \
                --enable-auto-tool-choice \
                --tool-call-parser glm47 \
                --reasoning-parser glm47 \
                --kv-cache-dtype int8 \
                --attention_config.indexer_kv_dtype int8 \
                --enable-chunked-prefill \
                --no-enable-prefix-caching \
                --kv-transfer-config '{
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_consumer",
                    "engine_id": "2",
                    "kv_port": "30100",
                    "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {
                            "dp_size": 1,
                            "tp_size": 16,
                            "pp_size": 2,
                            "pp_layer_partition": "42,36"
                        },
                        "decode": {
                            "dp_size": 8,
                            "tp_size": 4
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
            PYTHON_LIB_DIR="<PYTHON_LIBRARY_DIRECTORY>"

            export HCCL_OP_EXPANSION_MODE="AIV"
            export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
            export HCCL_EXEC_TIMEOUT=1800
            export HCCL_CONNECT_TIMEOUT=1800
            export ASCEND_TRANSFER_TIMEOUT=10000

            export VLLM_HOST_IP="$LOCAL_IP"
            export HCCL_IF_IP="$LOCAL_IP"
            export GLOO_SOCKET_IFNAME="$NIC_NAME"
            export TP_SOCKET_IFNAME="$NIC_NAME"
            export HCCL_SOCKET_IFNAME="$NIC_NAME"

            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export HCCL_BUFFSIZE=1024

            export TASK_QUEUE_ENABLE=1
            export ASCEND_RT_VISIBLE_DEVICES="$1"

            export LD_LIBRARY_PATH="$PYTHON_LIB_DIR:/usr/local/lib:${LD_LIBRARY_PATH:-}"
            export VLLM_USE_V2_MODEL_RUNNER=0

            export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
            export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'
            export LOCAL_WORLD_SIZE=16

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
                --decode-context-parallel-size 4 \
                --cp-kv-cache-interleave-size 128 \
                --seed 1024 \
                --served-model-name glm \
                --max-model-len 1048576 \
                --safetensors-load-strategy 'prefetch' \
                --max-num-batched-tokens 48 \
                --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
                --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
                --additional-config '{
                    "recompute_scheduler_enable": true,
                    "ascend_compilation_config": {
                        "enable_npugraph_ex": true,
                        "enable_static_kernel": true
                    },
                    "multistream_overlap_shared_expert": true
                }' \
                --trust-remote-code \
                --max-num-seqs 8 \
                --gpu-memory-utilization 0.90 \
                --async-scheduling \
                --quantization ascend \
                --enable-auto-tool-choice \
                --tool-call-parser glm47 \
                --reasoning-parser glm47 \
                --kv-cache-dtype int8 \
                --attention_config.indexer_kv_dtype int8 \
                --enable-chunked-prefill \
                --no-enable-prefix-caching \
                --kv-transfer-config '{
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_consumer",
                    "engine_id": "3",
                    "kv_port": "30100",
                    "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {
                            "dp_size": 1,
                            "tp_size": 16,
                            "pp_size": 2,
                            "pp_layer_partition": "42,36"
                        },
                        "decode": {
                            "dp_size": 8,
                            "tp_size": 4
                        }
                    }
                }'
            ```

    ##### Start the Engines

    Run the commands on their respective nodes in separate terminals.
    Start both PP stages and all Decode ranks before waiting for readiness.

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
        python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
        ```

    4. Decode node 1

        ```shell
        D0_IP="<DECODE_NODE0_IP>"
        python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 4 --dp-rank-start 4 --dp-address "$D0_IP" --dp-rpc-port 16600 --vllm-start-port 8000
        ```

    ##### Start the Proxy

    Wait for P0:8077 and all eight Decode APIs to answer `GET /v1/models`,
    and confirm P1 has joined its PP engine. Start the proxy on its separate host:

    ```shell
    P0_IP="<P0_IP>"
    D0_IP="<D0_IP>"
    D1_IP="<D1_IP>"

    python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 8000 \
        --prefiller-hosts "$P0_IP" --prefiller-ports 8077 \
        --decoder-hosts "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
            "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
        --decoder-ports 8000 8001 8002 8003 8000 8001 8002 8003
    ```

    Send requests to `<PROXY_IP>:8000` using the served model name `glm`.

    **Key Parameter Descriptions**

    - Prefill uses DP1/PP2/TP16/DCP16 with the layer partition `42,36`.
      P0 hosts the API; P1 joins through `--headless --node-rank 1`.
      Decode uses DP8/TP4/DCP4, four local DP ranks per node.
    - The launcher passes `$1` = visible devices, `$2` = API port, `$3` = global
      DP size, `$4` = DP rank, `$5` = DP master IP, `$6` = DP RPC port, and
      `$7` = TP size to `run_dp_template.sh`.
    - `--max-model-len` is `1048576` on both roles. Prefill uses
      `--max-num-seqs 64` and `--max-num-batched-tokens 16384`; Decode uses
      `8` and `48`, respectively, per engine. Tune concurrency to available
      KV cache capacity and reserve output tokens within the context limit.
    - PCP remains 1, while DCP is 16 on Prefill and 4 on Decode.
      `--cp-kv-cache-interleave-size 128` and int8 attention/indexer KV caches
      are configured on both roles.
    - Prefill enables chunked prefill, prefix caching, async scheduling, DSA CP,
      FlashComm1, fused MC2 mode 2 and shared-expert DP. Decode explicitly disables
      prefix caching, enables recomputation scheduling and shared-expert
      multistream overlap, and selects NpuGraphEx/StaticKernel through
      `ascend_compilation_config`.
    - MTP uses an eager draft with one speculative token on Prefill and five
      on Decode. Prefill uses `--enforce-eager`; Decode uses `FULL_DECODE_ONLY`.
      Both roles set `HCCL_BUFFSIZE=1024`, HCCL execution/connection timeouts
      to 1800 seconds, and `ASCEND_TRANSFER_TIMEOUT=10000`.
    - `MooncakeConnectorV2` uses `kv_producer` on Prefill and `kv_consumer`
      on Decode, with base KV ports 30000 and 30100. Both declare the same
      Prefill DP1/PP2/TP16 partition `42,36` and Decode DP8/TP4 topology.
      This example does not enable MemCache pooling.
    - TurboQuant cannot be combined with DCP or the int8 C8 cache and therefore
      does not apply to this 1M example. See [Performance Tuning](#9-performance-tuning).

## 6 Functional Verification {: #6-functional-verification }

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

### 7.3 W4A4C8 950DT products Validation Records

Both records below use one 950DT products (96GB × 8), a historical local
950DT products image, TP8/DP1/EP, eager mode, a configured maximum
context of 135000, 16384 batched tokens, 20 sequences and FP8 KV/indexer
caches. On 2026-10-10, `/health`, `/v1/chat/completions` and
`/v1/completions` returned HTTP 200 for both local checkpoints.
This verifies startup and short requests, not actual 135k inputs,
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

Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#5-online-service-deployment)** chapter.

## 9 Performance Tuning

The following settings summarize the hardware-specific deployment scripts in
Sections 5.1 and 5.2. Global DP includes all nodes within one DP group;
independent instances are listed per instance. Local DP is the number of
ranks on one node. Scheduler limits are per engine.

For 950DT ordinary PD, the two independent PP2/TP8 Prefill instances use
four Prefill nodes in total (P0/P1 and P2/P3). The Prefill row below gives
DP1 and scheduler limits per instance; it does not combine them into a DP2
group. The single DP16/TP1 Decode group spans D0/D1. Pooling and 1M rows
refer to their own deployment topologies in Sections 5.1.4 and 5.2.

### 9.1 Recommended Configurations {: #91-detailed-node-configuration }

The Colocated rows summarize the multi-node examples in
[Section 5.1.2](#512-multi-node-deployment). The single-node examples in
[Section 5.1.1](#511-single-node-deployment) use different parallelism and
scheduler limits; do not apply the multi-node rows to those examples.

| Scenario | Hardware | Role | TP | Global DP | Local DP | PP | DCP | Max Num Seqs | Max Num Batched Tokens | Max Model Len | MTP Spec Num |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Colocated | 950DT | Shared | 8 | 1 | 1 | 2 | 1 | 500 | 16384 | 135000 | 5 |
| Ordinary PD | 950DT | Prefill (2 instances) | 8 | 1 | 1 | 2 | 1 | 500 | 16384 | 135000 | 1 |
| Ordinary PD | 950DT | Decode | 1 | 16 | 8 | 1 | 1 | 10 | 60 | 135000 | 5 |
| Pooled PD | 950DT | Prefill | 8 | 1 | 1 | 2 | 1 | 8 | 4096 | 1048576 | 1 |
| Pooled PD | 950DT | Decode | 2 | 16 | 4 | 1 | 1 | 10 | 60 | 1048576 | 5 |
| 1M PD | 950DT | Prefill | 8 | 2 | 1 | 1 | 1 | 64 | 32768 | 1046576 | 1 |
| 1M PD | 950DT | Decode | 8 | 2 | 1 | 1 | 8 | 32 | 192 | 1048576 | 5 |
| Colocated | A3 | Shared | 4 | 8 | 4 | 1 | 1 | 6 | 4096 | 202752 | 3 |
| 198k PD, with or without pooling | A3 | Prefill | 16 | 1 | 1 | 2 | 1 | 64 | 16384 | 200000 | 1 |
| 198k PD, with or without pooling | A3 | Decode | 2 | 16 | 8 | 1 | 1 | 64 | 256 | 200000 | 5 |
| 1M PD | A3 | Prefill | 16 | 1 | 1 | 2 | 16 | 64 | 16384 | 1048576 | 1 |
| 1M PD | A3 | Decode | 4 | 8 | 4 | 1 | 4 | 8 | 48 | 1048576 | 5 |
| Colocated | A2 | Shared | 8 | 4 | 1 | 1 | 1 | 128 | 8192 | 135000 | 3 |
| Ordinary PD | A2 | Prefill | 8 | 1 | 1 | 4 | 1 | 64 | 8192 | 200000 | 1 |
| Ordinary PD | A2 | Decode | 2 | 16 | 4 | 1 | 1 | 32 | 256 | 200000 | 3 |
| Pooled PD | A2 | Prefill | 8 | 1 | 1 | 4 | 1 | 64 | 8192 | 200000 | 1 |
| Pooled PD | A2 | Decode | 2 | 16 | 4 | 1 | 1 | 64 | 256 | 200000 | 5 |

### 9.2 Tuning Guidelines {: #92-performance-workloads }

The A3 performance reference covers the following workloads.
Lengths use `input/output` notation. The 1M A3 case uses one concurrent
request; reserve output tokens within the 1,048,576-token context limit.

| Scenario | Hardware | Input / Output | Prefix cache hit rate | Deployment reference |
| --- | --- | --- | --- | --- |
| Prefill-only | A3 | 16k / 1, 128k / 1 | 0% | [Section 5.1.3](#513-prefill-decode-disaggregation) |
| Decode-only | A3 | 16k / 1k, 128k / 1k | 100% | [Section 5.1.3](#513-prefill-decode-disaggregation) |
| Complete PD | A3 | 16k / 1k | 0% | [Section 5.1.3](#513-prefill-decode-disaggregation) |
| Complete PD, 1M context | A3 | Approximately 1M / 1k | 0% | [Section 5.2](#52-1m-context-configuration) |
| Pooling | A3 | 128k / 1k | 90%, 32 distinct prefixes | [Section 5.1.4](#514-memcache-kv-cache-pool-deployment) |
| Pooling | A3 | 40k-80k / 2.5k | Repeated prefixes | [Section 5.1.4](#514-memcache-kv-cache-pool-deployment) |

### 9.3 TurboQuant KV Cache for GLM-5.2 and GLM-5.3

Use a build of the selected 0.30 maintenance branch containing the packed
SFA TurboQuant support. This SFA path supports model runner V1; it does
not require switching the V1 reference deployments to runner V2.
Use `--dtype bfloat16` on A2/A3 and disable xLite graph mode.

`--kv-cache-dtype turboquant_4bit_nc` replaces the int8 attention KV cache
and cannot be combined with `enable_sparse_sfa_c8: true`.
It does not support context parallelism (DCP/PCP), so it must not be
enabled in the DCP-enabled 1M reference deployment in [Section 5.2](#52-1m-context-configuration).
In PD deployments, set `"use_ascend_direct": false` on the Mooncake
connector on both roles, and keep the cache format consistent across
Prefill and Decode. Revalidate capacity and accuracy after changing it.
This option does not apply to the 950DT products FP8 examples or GLM-5.3-Flash.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md); this chapter only covers model-specific issues.

### Q: How to enable function calling for GLM-5.2 and GLM-5.3?

A: Please add following configurations in vLLM startup command.

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

No. Keep thinking enabled for the GLM-5.3 commands on this page. If thinking is disabled for GLM-5.3,
the model still generates reasoning text. vLLM returns this text in the
`content` field, while vLLM-Ascend returns it in the `reasoning` field.
