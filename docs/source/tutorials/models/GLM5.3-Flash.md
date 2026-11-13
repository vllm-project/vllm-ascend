{% set vllm_ascend_version = "nightly-releases-v0.32.0rc" %}

# GLM-5.3-Flash

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is the first natively multimodal model in the GLM-5 series. Built on a hybrid architecture that combines sparse and linear attention for the first time in the GLM series, it adopts Manifold-Constrained Hyper-Connections (mHC) and is trained on a 30T-token multimodal pre-training corpus. With 320B total parameters and only 18B active parameters, it outperforms GLM-5.2 across benchmarks and real-world workloads at one-tenth the price, while approaching Claude Opus 4.8 on coding and agentic benchmarks. GLM-5.3-Flash also supports controlling the thinking budget through the `reasoning_effort` parameter (`low`, `high`, `max`).

This document shows the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, Prefill-Decode (PD) disaggregated deployment, multi-node deployment, and accuracy and performance evaluation.

GLM-5.3-Flash is supported in **vLLM-Ascend v0.30.0 and later**.
This guide follows the deployment configurations on the vLLM-Ascend
`releases/v0.32.0rc` branch.
To use the latest features, use the latest supported release candidate or
official version of vLLM-Ascend.

Use the matching upstream vLLM revision for the selected vLLM-Ascend version.
For source builds, use the upstream vLLM tag recorded in
`.github/vllm-release-tag.commit` at the selected vLLM-Ascend release tag.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

All deployment examples on this page use Model Runner V2 with
`VLLM_USE_V2_MODEL_RUNNER=1`. Every PD Prefill role uses PCP8 with TP1;
Decode keeps PCP1.

The [A3 ordinary PD examples](#glm53-flash-5-3-a3) use Prefill Context
Parallelism (PCP) and BF16 SSM state (`--mamba-ssm-cache-dtype bfloat16`).

## 3 Prerequisites

### 3.1 Model Weight

Download the checkpoint from ModelScope for the hardware shown below. A3 and 950DT products checkpoints use different quantization formats; do not interchange them. The scripts use `/root/.cache/modelscope/hub/models/vllm-ascend/<checkpoint-name>` as a local mount path, not as a download source. Preserve the exact case of the repository name when naming the local directory.

| Checkpoint | Hardware | Deployment | Download | Notes |
| --- | --- | --- | --- | --- |
| `GLM-5.3-Flash-w8a8-mxfp8` | 950DT products (96GB × 8 per node) | 1 node for single-node deployment; 2 nodes for PD | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8) | Supports a W8A8C8 configuration: C8 is dynamic KV-cache quantization. Set `--kv-cache-dtype fp8` in the serving command to enable FP8 KV cache; no separate C8 checkpoint is required. |
| `GLM-5.3-Flash-w8a8` | A3 series (64GB × 16) | 1 node for single-node deployment; 2 nodes for PD | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8) | Supports a W8A8C8 configuration: C8 is dynamic KV-cache quantization. Set `--kv-cache-dtype int8` in the serving command to enable INT8 KV cache; no separate C8 checkpoint is required. |
| `GLM-5.3-Flash-w8a8` | A2 series (64GB × 8 per node) | 2 nodes for colocated deployment; 4 nodes for PD | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8) | / |

- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

The checkpoints linked above are already quantized and can be served directly. For self-quantization, follow the installation and command examples in the [msModelSlim quantization guide](https://www.hiascend.com/document/detail/en/mindstudio/2610/TITools/msModelSlim/docs/en/user_guide/feature_guide/quick_quantization_v1/usage.md) and select the configuration for the exact model and target hardware from the [model recipes](https://gitcode.com/Ascend/msmodelslim/tree/master/lab_practice). Do not reuse a different model's YAML file.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

Select an image for your hardware and start a container on each node.
Refer to [Using Docker](../../getting_started/installation.md#installation-prebuilt-image) for details.

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
    --shm-size=100g \
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
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
    ```

=== "A2 series"

    Start the Docker container on each node.

    If you use the A2 KV Cache Pool deployment with SSD offload, mount a
    fast, writable host SSD directory into the container on **all four nodes**.
    The mount is required: creating `/mnt/mooncake_offload` inside the
    container without it writes to the container's overlay filesystem and can
    exhaust the container root filesystem. Replace the host path below with a
    local SSD filesystem that has sufficient capacity and permissions for the
    container user.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    export NAME=vllm-ascend
    export MOONCAKE_SSD_PATH=/path/to/host/ssd

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
    -v "$MOONCAKE_SSD_PATH:/mnt/mooncake_offload" \
    -it $IMAGE bash
    ```

    **Check the software stack inside each A2 container**

    ```shell
    command -v python3 vllm
    python3 -m pip show vllm vllm-ascend transformers
    ```

    If startup reports that Transformers does not recognize `glm5_next`, the failure occurs while
    loading the model configuration, before DP engine initialization. Check
    the active software versions and recreate the containers with a compatible
    image if they still use the legacy `glm-5.3-flash` image or a vLLM
    `0.23.0` environment. For source builds, follow the
    [installation guide](../../getting_started/installation.md) and use the
    matching vLLM, vLLM-Ascend, and Transformers dependencies for the checked-out
    revision.

    Before starting the pool, verify the mount inside the container with
    `df -h /mnt/mooncake_offload` and ensure that the path is writable. The
    `enable_ssd_offload` configuration in the A2 KV Pool section assumes this
    host-to-container mount is present.

## 5 Online Service Deployment

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
| `--reasoning-parser` | Separates reasoning from the final answer. Use `glm47` for all examples on this page. |
| `--safetensors-load-strategy prefetch` | Reads checkpoint files into the operating system page cache before workers load them. It can help with network or high-latency storage and needs sufficient host memory; it does not preload weights into NPU memory. See [vLLM weight-loading options](https://docs.vllm.ai/en/latest/configuration/engine_args/#--safetensors-load-strategy). |

### 5.1 Single-Node Online Deployment

=== "950DT products"

    #### 950DT products {: #glm53-flash-5-1-950dt }

    - Quantized model `GLM-5.3-Flash-w8a8-mxfp8` can be deployed on 1 950DT products (96GB × 8).

    Run the following script to execute online inference.

    ```shell
    # Replace with the checkpoint path recorded in Section 3.1.
    MODEL_PATH="<YOUR_MODEL_PATH>"
    export VLLM_USE_V2_MODEL_RUNNER=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024

    vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port 8000 \
      --data-parallel-size 1 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --kv-cache-dtype fp8 \
      --attention-config.indexer_kv_dtype fp8 \
      --served-model-name glm \
      --max-num-seqs 32 \
      --max-model-len 133120 \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --gpu-memory-utilization 0.9 \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --enable-auto-tool-choice \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp"}'
    ```

=== "A3 series"

    #### A3 series {: #glm53-flash-5-1-a3 }

    - Quantized model `GLM-5.3-Flash-w8a8` can be deployed on 1 A3 series (64GB × 16) .

    Run the following script to execute online inference.

    ```shell
    #!/bin/sh

    export VLLM_USE_V2_MODEL_RUNNER=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_BUFFSIZE=1024

    vllm serve Eco-Tech/GLM-5.3-Flash-w8a8   \
      --host 0.0.0.0 \
      --port 8000 \
      --max-model-len 1048576 \
      --data-parallel-size 1 \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name glm \
      --safetensors-load-strategy prefetch \
      --max-num-seqs 32 \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --quantization ascend \
      --kv-cache-dtype int8 \
      --mamba-ssm-cache-dtype bfloat16 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --gpu-memory-utilization 0.92 \
      --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": false}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --additional-config '{"multistream_overlap_shared_expert": true}' \
      --api-server-count 1
    ```

**Key Parameter Descriptions**

Only the key parameters specific to this model/scenario are described below. `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario.

**Model-specific parameters:**

- `--data-parallel-size 1`: Runs a single DP rank. `--tensor-parallel-size` is 8 on 950DT products and 16 on A3 series. This layout is recommended to balance memory capacity and compute efficiency for the w8a8 weights.
- `--enable-expert-parallel`: Must be enabled for the MoE architecture of GLM-5.3-Flash.
- `--quantization ascend`: Enables Ascend quantization for the w8a8 quantized weights.
- `--kv-cache-dtype`: Enables 8-bit KV cache quantization (C8) for Sparse Flash Attention (SFA).
  Use C8 when long contexts or concurrent requests are limited by KV cache memory.
  The complete single-node commands above use `fp8` on 950DT products and `int8` on A3 series.
  KV cache quantization is independent of weight quantization: enabling C8 does not require W8A8 weights or a separate C8 checkpoint.
  The examples use W8A8-MXFP8 weights on 950DT products and W8A8 weights on A3 series, as listed in Section 3.1.
- `--attention-config.indexer_kv_dtype`: Uses FP8 KPool indexer cache on 950DT products, independently of `--kv-cache-dtype`. The A3 series example leaves the indexer cache dtype at its default.
- `--mamba-ssm-cache-dtype bfloat16`: Uses BF16 SSM state in the A3 series single-node example.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables graph capture for the decode phase only, improving decode performance by reducing kernel launch overhead.
- `--limit-mm-per-prompt '{"image":1,"video":0}'`: Allows at most one image per prompt and disables video input where specified. A3 ordinary PD uses `'{"image":1}'` to limit image input only.
- `--speculative-config`: Enables Multi-Token Prediction (MTP) speculative decoding with the DeepSeek-style MTP draft head of GLM-5.3-Flash. The single-node examples use five speculative tokens. GLM-5.3-Flash supports MTP draft graph execution.

### 5.2 Multi-Node Colocated Deployment

=== "A2 series"

    #### A2 series {: #glm53-flash-5-2-a2 }

    - Quantized model `GLM-5.3-Flash-w8a8` can be deployed on 2 A2 series (64GB × 8) nodes with DP2 across the two nodes (one DP rank per node) and TP8 inside each node.

    Run the following scripts on two nodes respectively.

    ##### node 0

    ```shell
    # Obtain the current node IP and its network interface using ifconfig.
    # Set nic_name to the network interface corresponding to local_ip on this node.
    nic_name="xxxx"
    local_ip="xx.xx.xx.1"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xx.xx.xx.1"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=1200
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_IF_IP=$local_ip
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    vllm serve /path/to/GLM-5.3-Flash-w8a8 \
        --host 0.0.0.0 \
        --port 8000 \
        --max-model-len 133120 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 12321 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm \
        --safetensors-load-strategy prefetch \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --quantization ascend \
        --limit-mm-per-prompt '{"image":1,"video":0}' \
        --gpu-memory-utilization 0.85 \
        --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16,32,64,96,128]}' \
        --api-server-count 1
    ```

    ##### node 1

    ```shell
    # Obtain the current node IP and its network interface using ifconfig.
    # Set nic_name to the network interface corresponding to local_ip on this node.
    nic_name="xxxx"
    local_ip="xx.xx.xx.2"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xx.xx.xx.1"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=1200
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_IF_IP=$local_ip
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    vllm serve /path/to/GLM-5.3-Flash-w8a8 \
        --host 0.0.0.0 \
        --port 8000 \
        --headless \
        --max-model-len 133120 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank 1 \
        --data-parallel-address $node0_ip \
        --data-parallel-rpc-port 12321 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm \
        --safetensors-load-strategy prefetch \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --quantization ascend \
        --limit-mm-per-prompt '{"image":1,"video":0}' \
        --gpu-memory-utilization 0.85 \
        --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16,32,64,96,128]}'
    ```

**Key Parameter Descriptions**

**Multi-node network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`: Network interface configuration for multi-node communication. Set `nic_name` to the network interface name (obtained via `ifconfig`) and `local_ip` to the current node's IP address. These must be correctly configured on each node for successful multi-node communication.
- `--data-parallel-size 2 --data-parallel-size-local 1`: Runs two DP ranks across the two nodes, one rank per node; each rank uses TP8 within its node.
- `--data-parallel-start-rank`: Starting DP rank offset of the current node. Node 0 omits this option and automatically uses rank `0`; headless node 1 explicitly uses `1`.
- `--data-parallel-address`: IP address of the data parallel master node (node 0). Must match the `local_ip` of the master node.
- `--data-parallel-rpc-port 12321`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: Indicates a non-master node (used on node 1). Do not use on node 0.

After starting all nodes of the selected deployment, open another terminal and run `curl --fail <API_HOST>:<API_PORT>/health` against every API endpoint. For single-node deployment use that node's API; for colocated A2 use the API-serving master, since the headless node has no API. Wait for successful responses before sending the [functional verification requests](#6-functional-verification). Check the launch terminals on all nodes if initialization fails.

### 5.3 PD Disaggregated Deployment {: #53-1p1d-pd-disaggregated-deployment }

Before starting any role script, verify [multi-node communication](../../getting_started/installation.md#installation-multi-node-interconnect), select the matching host IP and interface on every node, and allow the DP RPC, API, HCCL socket, Mooncake handshake/control, and transfer-engine ports through the firewall. Use the port settings in the selected hardware tab; allowing only the proxy port is insufficient. Physical node count and the number of DP engines are different: follow each tab's topology and endpoint mapping.

The 950DT products, A3 series and A2 series deployments use the same `launch_online_dp.py` script to
start one vLLM process per local DP rank. Save it on every Prefill and Decode
node before following the platform-specific instructions below.

```python
import argparse
import multiprocessing
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="./run_p.sh")
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pcp-size", type=int, default=1)
    parser.add_argument("--dp-size-local", type=int, default=-1)
    parser.add_argument("--dp-rank-start", type=int, default=0)
    parser.add_argument("--dp-address", required=True)
    parser.add_argument("--dp-rpc-port", default="12325")
    parser.add_argument("--vllm-start-port", type=int, default=8000)
    args = parser.parse_args()
    if args.dp_size_local == -1:
        args.dp_size_local = args.dp_size
    return args


def run(args, devices, port, dp_rank):
    script_args = [
        devices,
        str(port),
        str(args.dp_size),
        str(dp_rank),
        args.dp_address,
        args.dp_rpc_port,
        str(args.tp_size),
        str(args.pcp_size),
    ]
    if args.dp_size == 1:
        with open(args.template, encoding="utf-8") as f:
            script = f.read()
        script = "\n".join(
            line for line in script.splitlines()
            if "--data-parallel-rank" not in line
            and "--data-parallel-external-lb" not in line
        )
        command = ["bash", "-o", "pipefail", "-c", script, args.template, *script_args]
    else:
        command = ["bash", "-o", "pipefail", args.template, *script_args]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    args = parse_args()
    processes = []
    devices_per_dp_rank = args.tp_size * args.pcp_size
    for i in range(args.dp_size_local):
        devices = ",".join(
            str(device)
            for device in range(i * devices_per_dp_rank, (i + 1) * devices_per_dp_rank)
        )
        process = multiprocessing.Process(
            target=run,
            args=(
                args,
                devices,
                args.vllm_start_port + i,
                args.dp_rank_start + i,
            ),
        )
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
```

The launcher passes visible devices, engine port, global DP size, DP rank, DP
address, DP RPC port, TP size, and PCP size to each role script as `$1` through
`$8`. Each local DP rank uses TP×PCP devices. `--pcp-size` defaults to `1`.

When `--dp-size 1` is selected, the launcher removes lines containing
`--data-parallel-rank` or `--data-parallel-external-lb` before running the role
script. These options enable external DP load balancing, which requires DP>1.
Keep each option on its own line so other arguments are preserved when filtering.
For DP>1, the launcher executes the role script without filtering.

=== "950DT products"

    #### 950DT products {: #glm53-flash-5-3-950dt }

    ##### Topology and Prerequisites

    Use two nodes with eight devices each: Prefill TP1/PCP8/DP1 and Decode
    TP1/PCP1/DP8. Both roles use EP8.
    Both nodes use the same GLM-5.3-Flash W8A8-MXFP8 checkpoint and software build.
    Set the local IP, network interface and checkpoint path in each script.

    ##### Prepare the Scripts

    1. prepare the script `launch_online_dp.py` on each node, using the
       [shared DP launcher](#53-1p1d-pd-disaggregated-deployment) above.

    2. prepare the script `run_dp_template.sh` on each node.

        1. Prefill node

            ```shell
            #!/usr/bin/env bash

            local_ip="<P0_IP>"
            nic_name="<NETWORK_INTERFACE>"
            # Replace with the checkpoint path recorded in Section 3.1.
            MODEL_PATH="<YOUR_MODEL_PATH>"

            export VLLM_USE_V2_MODEL_RUNNER=1
            export HCCL_IF_IP=$local_ip
            export GLOO_SOCKET_IFNAME=$nic_name
            export TP_SOCKET_IFNAME=$nic_name
            export HCCL_SOCKET_IFNAME=$nic_name
            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export HCCL_BUFFSIZE=2048
            export HCCL_OP_EXPANSION_MODE="AIV"
            export ASCEND_RT_VISIBLE_DEVICES="$1"
            export PYTHONHASHSEED=0
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

            vllm serve "$MODEL_PATH" \
              --allowed-local-media-path / \
              --host 0.0.0.0 \
              --port $2 \
              --data-parallel-size $3 \
              --data-parallel-rank $4 \
              --data-parallel-address $5 \
              --data-parallel-rpc-port $6 \
              --tensor-parallel-size $7 \
              --prefill-context-parallel-size $8 \
              --enable-expert-parallel \
              --seed 1024 \
              --served-model-name glm \
              --safetensors-load-strategy prefetch \
              --kv-cache-dtype fp8 \
              --attention-config.indexer_kv_dtype fp8 \
              --max-num-seqs 128 \
              --max-model-len 1048576 \
              --max-num-batched-tokens 32768 \
              --trust-remote-code \
              --quantization ascend \
              --tool-call-parser glm47 \
              --enable-auto-tool-choice \
              --reasoning-parser glm47 \
              --enable-prefix-caching \
              --limit-mm-per-prompt '{"video": 0}' \
              --gpu-memory-utilization 0.95 \
              --enforce-eager \
              --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp"}' \
              --additional-config '{
                "enable_cpu_binding": true,
                "enable_flashcomm1": false,
                "enable_fused_mc2": 0,
                "enable_shared_expert_dp": false,
                "multistream_overlap_shared_expert": false
              }' \
              --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_producer",
                "engine_id": "0",
                "kv_port": "30000",
                "kv_connector_extra_config": {
                  "prefill": {"dp_size": 1, "tp_size": 1, "pcp_size": 8},
                  "decode": {"dp_size": 8, "tp_size": 1, "pcp_size": 1}
                }
              }'
            ```

        2. Decode node

            ```shell
            #!/usr/bin/env bash

            local_ip="<D0_IP>"
            nic_name="<NETWORK_INTERFACE>"
            # Replace with the checkpoint path recorded in Section 3.1.
            MODEL_PATH="<YOUR_MODEL_PATH>"

            export VLLM_USE_V2_MODEL_RUNNER=1
            export HCCL_IF_IP=$local_ip
            export GLOO_SOCKET_IFNAME=$nic_name
            export TP_SOCKET_IFNAME=$nic_name
            export HCCL_SOCKET_IFNAME=$nic_name
            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export HCCL_BUFFSIZE=1024
            export HCCL_OP_EXPANSION_MODE="AIV"
            export ASCEND_RT_VISIBLE_DEVICES="$1"
            export PYTHONHASHSEED=0
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

            vllm serve "$MODEL_PATH" \
              --allowed-local-media-path / \
              --host 0.0.0.0 \
              --port $2 \
              --data-parallel-size $3 \
              --data-parallel-rank $4 \
              --data-parallel-address $5 \
              --data-parallel-rpc-port $6 \
              --tensor-parallel-size $7 \
              --prefill-context-parallel-size $8 \
              --enable-expert-parallel \
              --seed 1024 \
              --served-model-name glm \
              --safetensors-load-strategy prefetch \
              --kv-cache-dtype fp8 \
              --attention-config.indexer_kv_dtype fp8 \
              --max-num-seqs 60 \
              --max-model-len 1048576 \
              --max-num-batched-tokens 360 \
              --trust-remote-code \
              --quantization ascend \
              --tool-call-parser glm47 \
              --enable-auto-tool-choice \
              --reasoning-parser glm47 \
              --enable-prefix-caching \
              --limit-mm-per-prompt '{"video": 0}' \
              --gpu-memory-utilization 0.95 \
              --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
              --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp"}' \
              --additional-config '{
                "enable_fused_mc2": 2,
                "multistream_overlap_shared_expert": true,
                "scheduler_config": {"recompute_scheduler_enable": true},
                "ascend_compilation_config": {
                  "enable_npugraph_ex": true,
                  "enable_static_kernel": false,
                  "enable_super_kernel": false
                }
              }' \
              --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_consumer",
                "engine_id": "1",
                "kv_port": "30800",
                "kv_connector_extra_config": {
                  "prefill": {"dp_size": 1, "tp_size": 1, "pcp_size": 8},
                  "decode": {"dp_size": 8, "tp_size": 1, "pcp_size": 1},
                  "use_ascend_direct": true
                }
              }'
            ```

    ##### Start the Engines

    1. Prefill node

        ```shell
        P0_IP="<P0_IP>"
        python launch_online_dp.py --template ./run_dp_template.sh \
            --dp-size 1 --tp-size 1 --pcp-size 8 --dp-size-local 1 \
            --dp-rank-start 0 --dp-address "$P0_IP" --dp-rpc-port 12325 \
            --vllm-start-port 8000
        ```

    2. Decode node

        ```shell
        D0_IP="<D0_IP>"
        python launch_online_dp.py --template ./run_dp_template.sh \
            --dp-size 8 --tp-size 1 --dp-size-local 8 \
            --dp-rank-start 0 --dp-address "$D0_IP" --dp-rpc-port 12325 \
            --vllm-start-port 8000
        ```

    ##### Start the Proxy

    Run on a separate host after the Prefill and Decode endpoints are healthy.

    ```shell
    P0_IP="<P0_IP>"
    D0_IP="<D0_IP>"

    python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 8000 \
        --prefiller-hosts "$P0_IP" --prefiller-ports 8000 \
        --decoder-hosts "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
            "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
        --decoder-ports 8000 8001 8002 8003 8004 8005 8006 8007
    ```

    **Key Parameter Descriptions**

    - `--data-parallel-size`, `--tensor-parallel-size`, and
      `--prefill-context-parallel-size` define TP1/PCP8/DP1 on Prefill
      and TP1/PCP1/DP8 on Decode. Each role uses eight devices on one node.
      Prefill runs one eight-device engine. `ASCEND_RT_VISIBLE_DEVICES="$1"` binds each
      engine to the device list passed by the launcher: devices `0-7` for
      Prefill and one device per DP rank for Decode.
    - `--data-parallel-address` and `--data-parallel-rpc-port` coordinate
      ranks within each role. Prefill and Decode use their respective node
      IPs; RPC port `12325` is reused on the separate hosts.
    - `--vllm-start-port 8000` assigns Prefill API port `8000` and Decode
      API ports `8000-8007`. Register all eight Decode endpoints with the
      proxy.
    - Both roles use `--max-model-len 1048576`, `--kv-cache-dtype fp8`, and
      `deepseek_mtp` with five speculative tokens.
    - Prefill sets `--max-num-seqs 128` and
      `--max-num-batched-tokens 32768`; Decode sets `60` and `360`,
      respectively. Tune these per-engine scheduler limits for the workload.
    - Prefill enables CPU binding and disables FlashComm1, fused MC2,
      shared-expert DP and multistream overlap. Its target and MTP draft
      run eagerly. Decode uses `enable_fused_mc2: 2`, shared-expert
      multistream overlap and recomputation scheduling through `scheduler_config`.
      Decode uses `FULL_DECODE_ONLY` with `enable_npugraph_ex: true`,
      while static and super kernels are disabled.
    - Prefill uses PCP8, while Decode keeps the default PCP1. Both roles
      keep the default DCP1. For PCP configuration, refer to
      [Context Parallel](../../user_guide/feature_guide/context_parallel.md).
    - Both roles enable prefix caching and disable video input through
      `--limit-mm-per-prompt`. Images retain the runtime default limit.
      `--allowed-local-media-path /` permits local media under the container root.
    - `MooncakeConnectorV2` transfers KV cache directly between the roles. It uses
      `kv_producer` on Prefill and `kv_consumer` on Decode, with base KV ports
      `30000` and `30800`. Both roles declare Prefill TP1/PCP8/DP1 and
      Decode TP1/PCP1/DP8; Decode enables `use_ascend_direct`.
    - `HCCL_IF_IP` and the socket interface variables select each node's
      communication network. `HCCL_BUFFSIZE` is `2048` on Prefill and
      `1024` on Decode. The DP RPC, API, Mooncake and proxy ports must be
      reachable between the corresponding hosts.

=== "A3 series"

    #### A3 series {: #glm53-flash-5-3-a3 }

    This example uses two A3 series servers. The Prefill node runs two
    DP ranks with TP1/PCP8 (DP2/TP1/PCP8), and the Decode node runs eight DP ranks
    with TP2 (DP8/TP2). Both layouts consume all 16 logical devices on their
    respective servers. `MooncakeConnectorV2` transfers KV cache from the
    Prefill engines to the Decode engines.

    Before starting the services, replace `LOCAL_IP`, `NIC_NAME`, and
    `MODEL_PATH` in the following scripts with values for the deployment
    environment. These scripts target an A3 series supernode using HCCS with its
    device RoCE ports down. They explicitly select `hccs:device` for device
    communication; the socket interface variables still select the host
    network. Adjust the resource configuration for other network topologies.

    Start Decode first, then Prefill. In another terminal, check `<DECODE_NODE_IP>:9900/health` through port `9907` and `<PREFILL_NODE_IP>:9081/health` through port `9082` with `curl --fail`. Start the proxy only after all ten endpoints respond successfully; inspect each launcher terminal if a rank fails to initialize.

    ##### A3 series Serving Scripts and Startup Commands

    ##### Start the Prefill node

    On the Prefill node, save the following script as `run_p.sh`. The launcher
    passes the visible devices, API port, DP configuration, TP size, and PCP size as
    positional arguments.

    ```shell
    #!/usr/bin/env bash
    # Usage: bash run_p.sh <visible_devices> <http_port> <dp_size> \
    #   <dp_rank> <dp_address> <rpc_port> <tp_size> <pcp_size>

    # This A3 series supernode uses HCCS; its device RoCE ports are down.
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'

    LOCAL_IP="<PREFILL_NODE_IP>"
    NIC_NAME="<NETWORK_INTERFACE>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export ASCEND_RT_VISIBLE_DEVICES="$1"

    exec vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port "$2" \
      --data-parallel-size "$3" \
      --data-parallel-rank "$4" \
      --data-parallel-address "$5" \
      --data-parallel-rpc-port "$6" \
      --tensor-parallel-size "$7" \
      --prefill-context-parallel-size "$8" \
      --enable-expert-parallel \
      --enable-eplb \
      --eplb-config '{"window_size":3000,"step_interval":3000,"num_redundant_experts":0,"use_async":true,"policy":"stair","communicator":"hixl"}' \
      --seed 1024 \
      --served-model-name glm \
      --safetensors-load-strategy prefetch \
      --max-num-seqs 32 \
      --max-model-len 1048576 \
      --max-num-batched-tokens 16384 \
      --trust-remote-code \
      --quantization ascend \
      --limit-mm-per-prompt '{"image":1}' \
      --gpu-memory-utilization 0.92 \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --enable-auto-tool-choice \
      --kv-cache-dtype int8 \
      --prefix-cache-retention-interval None \
      --enforce-eager \
      --mamba-ssm-cache-dtype bfloat16 \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": false}' \
      --additional-config '{"enable_shared_expert_dp": true,"enable_fused_mc2":2}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_producer", "engine_id": "0", "kv_port": "36680"}'
    ```

    <a id="glm53-flash-a3-prefill-command"></a>

    Start two TP1/PCP8 Prefill engines (total DP2). `--dp-address` uses the Prefill node IP.
    The launcher assigns API ports `9081-9082`.

    ```shell
    python launch_online_dp.py \
      --template ./run_p.sh \
      --dp-size 2 \
      --tp-size 1 \
      --pcp-size 8 \
      --dp-address "<PREFILL_NODE_IP>" \
      --vllm-start-port 9081
    ```

    ##### Start the Decode Node

    On the Decode node, save the following script as `run_d.sh`. Decode uses
    `FULL_DECODE_ONLY` graph mode for the target model, enables graph mode
    for the MTP draft model, and enables FlashComm1.

    ```shell
    #!/usr/bin/env bash
    # Usage is the same as run_p.sh.

    # This A3 series supernode uses HCCS; its device RoCE ports are down.
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'

    LOCAL_IP="<DECODE_NODE_IP>"
    NIC_NAME="<NETWORK_INTERFACE>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export ASCEND_RT_VISIBLE_DEVICES="$1"

    exec vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port "$2" \
      --data-parallel-size "$3" \
      --data-parallel-rank "$4" \
      --data-parallel-address "$5" \
      --data-parallel-rpc-port "$6" \
      --tensor-parallel-size "$7" \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name glm \
      --safetensors-load-strategy prefetch \
      --max-num-seqs 40 \
      --max-model-len 1048576 \
      --max-num-batched-tokens 240 \
      --trust-remote-code \
      --quantization ascend \
      --limit-mm-per-prompt '{"image":1}' \
      --gpu-memory-utilization 0.92 \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --enable-auto-tool-choice \
      --kv-cache-dtype int8 \
      --mamba-ssm-cache-dtype bfloat16 \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": false}' \
      --additional-config '{
        "multistream_overlap_shared_expert": true,
        "scheduler_config": {"recompute_scheduler_enable": true},
        "enable_shared_expert_dp": true,
        "enable_flashcomm1": true
      }' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_consumer", "engine_id": "1", "kv_port": "36580"}'
    ```

    <a id="glm53-flash-a3-decode-command"></a>

    Start eight TP2 Decode engines (total DP8). `--dp-address` uses the Decode node
    IP. The launcher assigns API ports `9900-9907`.

    ```shell
    python launch_online_dp.py \
      --template ./run_d.sh \
      --dp-size 8 \
      --tp-size 2 \
      --dp-address "<DECODE_NODE_IP>" \
      --vllm-start-port 9900
    ```

    <a id="glm53-flash-a3-proxy"></a>

    ##### Deploy the PD Proxy

    After all Prefill and Decode engines are ready, open another terminal in
    the Prefill container and start the
    [load-balancing proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).
    The proxy listens on port `8081`, distributes requests across Prefill
    endpoints `9081-9082`, and then forwards Decode work to endpoints
    `9900-9907`.

    ```shell
    #!/usr/bin/env bash

    P_IP="${P_IP:-<PREFILL_NODE_IP>}"
    P_N=${P_N:-2}
    P_PORT0=${P_PORT0:-9081}
    D_IP="${D_IP:-<DECODE_NODE_IP>}"
    D_N=${D_N:-8}
    D_PORT0=${D_PORT0:-9900}

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
      --host 0.0.0.0 \
      --port 8081 \
      --prefiller-hosts $(printf "$P_IP %.0s" $(seq 1 "$P_N")) \
      --prefiller-ports $(seq "$P_PORT0" $((P_PORT0 + P_N - 1))) \
      --decoder-hosts $(printf "$D_IP %.0s" $(seq 1 "$D_N")) \
      --decoder-ports $(seq "$D_PORT0" $((D_PORT0 + D_N - 1)))
    ```

    **Key Parameter Descriptions**

    - Prefill uses DP2/TP1/PCP8: each of its two engines receives eight
      devices from `launch_online_dp.py`. Decode uses DP8/TP2: each of its
      eight engines receives two devices. Both roles use all 16 logical devices
      on their respective A3 series nodes.
    - `--data-parallel-address` and `--data-parallel-rpc-port` coordinate DP
      ranks within one role. Prefill and Decode use their respective node IPs.
      The default RPC port `12325` can be reused on the separate hosts.
    - `ASCEND_RT_VISIBLE_DEVICES="$1"` selects the launcher-assigned devices
      for each engine.
    - Prefill API ports are `9081-9082`; Decode API ports are `9900-9907`.
      All ten endpoints must be reachable from the proxy.
    - Both roles use Model Runner V2, INT8 main KV cache,
      `--max-model-len 1048576`, MTP3, and
      `--mamba-ssm-cache-dtype bfloat16`. Prefix caching uses the model default.
    - Prefill runs the target eagerly and enables draft graph execution.
      It enables asynchronous EPLB with the `stair` policy and `hixl`
      communicator. The Prefill configuration sets `enable_fused_mc2` to `2`
      and `enable_shared_expert_dp` to `true`.
    - Prefill uses `--max-num-seqs 32` and
      `--max-num-batched-tokens 16384`. Decode uses `40` and `240`, respectively.
    - Decode uses `FULL_DECODE_ONLY` with default capture sizes, FlashComm1,
      shared-expert DP and multistream
      overlap. Recompute scheduling is enabled through `scheduler_config`.
    - Both roles explicitly limit image input to one image per prompt.
    - `MooncakeConnectorV2` uses `kv_producer` and engine ID `0` on Prefill,
      and `kv_consumer` and engine ID `1` on Decode. These IDs identify the
      roles in this deployment. Base KV ports are `36680` and `36580`.
    - The HCCS resource settings select device communication; `HCCL_IF_IP`
      and the socket interface variables select the host network. The DP RPC,
      API, Mooncake and proxy ports must be available on the respective hosts.

=== "A2 series"

    #### A2 series {: #glm53-flash-5-3-a2 }

    This example uses four A2 series (64GB × 8) servers: two Prefill
    nodes running one DP rank with TP1/PCP8 each (DP2/TP1/PCP8 in total), and two
    Decode nodes running four DP ranks with TP2 each (DP8/TP2 in total).
    Each Prefill rank uses eight devices, and each Decode rank uses two.
    Together, the local ranks use all eight devices on each server.
    `MooncakeConnectorV2` transfers KV cache from the Prefill engines to the
    Decode engines. The topology names the two Prefill nodes P0/P1 and the
    two Decode nodes D0/D1. P0 serves the Prefill API endpoint; P1 is
    headless. The proxy addresses the API endpoints on both Decode nodes.

    The shared `launch_online_dp.py` launcher from [Section 5.3](#53-1p1d-pd-disaggregated-deployment) starts the
    engines on all four nodes. Save the common serving templates below and the
    node-specific wrapper scripts on their respective nodes. Replace all IP,
    NIC, and model-path placeholders.

    ##### Common Prefill template

    Save the following script as `run_p.sh` on P0 and P1. The node wrappers
    below provide `LOCAL_IP` and the API/headless role.

    Prefill uses internal DP load balancing: P0 infers global DP rank 0
    and exposes the only Prefill API; P1 uses
    `--headless --data-parallel-start-rank 1`. Keep
    `--data-parallel-size-local 1` in this template. Although the launcher
    passes the DP rank as `$4`, do not forward it as
    `--data-parallel-rank "$4"` on Prefill: an explicit rank selects external DP
    load balancing from that option and rejects the remote headless
    engine with `Remote engine must not use --headless in external or hybrid dp lb mode`.
    Decode uses explicit DP ranks and API-serving engines, so its
    `--data-parallel-rank "$4"` configuration is intentional.

    ```shell
    #!/usr/bin/env bash
    # Usage: bash run_p.sh <visible_devices> <http_port> <dp_size> \
    #   <dp_rank> <dp_address> <rpc_port> <tp_size> <pcp_size>

    : "${LOCAL_IP:?Set LOCAL_IP in the P0 or P1 wrapper}"
    : "${NIC_NAME:?Set NIC_NAME in the P0 or P1 wrapper}"
    : "${MODEL_PATH:?Set MODEL_PATH in the P0 or P1 wrapper}"

    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_HOST_IP="$LOCAL_IP"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,pin_memory_expandable_segments:True
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=1200
    export VLLM_RPC_TIMEOUT=3600000
    export HCCL_INTRA_ROCE_ENABLE=1
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTHONHASHSEED=0

    exec vllm serve "$MODEL_PATH" \
      --device-ids "$1" \
      --host 0.0.0.0 \
      --port "$2" \
      --data-parallel-size "$3" \
      --data-parallel-size-local 1 \
      --data-parallel-address "$5" \
      --data-parallel-rpc-port "$6" \
      --tensor-parallel-size "$7" \
      --prefill-context-parallel-size "$8" \
      --enable-expert-parallel \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --seed 1024 \
      --served-model-name glm \
      --safetensors-load-strategy prefetch \
      --max-model-len 1048576 \
      --max-num-seqs 64 \
      --max-num-batched-tokens 4096 \
      --trust-remote-code \
      --quantization ascend \
      --kv-cache-dtype int8 \
      --attention_config.indexer_kv_dtype int8 \
      --gpu-memory-utilization 0.9 \
      --async-scheduling \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --enable-auto-tool-choice \
      --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
      --additional-config '{
        "enable_flashcomm1": true,
        "enable_shared_expert_dp": true
      }' \
      ${SERVER_ROLE_ARGS:-} \
      --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV2",
        "kv_role": "kv_producer",
        "engine_id": "0",
        "kv_port": "36000",
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {"dp_size": 2, "tp_size": 1, "pcp_size": 8},
          "decode": {"dp_size": 8, "tp_size": 2, "pcp_size": 1}
        }
      }' \
      --prefix-cache-retention-interval None
    ```

    ##### P0 and P1 wrappers

    Save the following `run_p0.sh` on P0. P0 hosts the Prefill API endpoint.

    ```shell
    #!/usr/bin/env bash
    export LOCAL_IP="<PREFILL_NODE0_IP>"
    export NIC_NAME="<PREFILL_NODE0_NIC>"
    export MODEL_PATH="<YOUR_MODEL_PATH>"
    export SERVER_ROLE_ARGS="--api-server-count 1"
    exec bash ./run_p.sh "$@"
    ```

    Save the following `run_p1.sh` on P1. P1 is the headless Prefill rank.

    ```shell
    #!/usr/bin/env bash
    export LOCAL_IP="<PREFILL_NODE1_IP>"
    export NIC_NAME="<PREFILL_NODE1_NIC>"
    export MODEL_PATH="<YOUR_MODEL_PATH>"
    export SERVER_ROLE_ARGS="--headless --data-parallel-start-rank 1"
    exec bash ./run_p.sh "$@"
    ```

    <a id="glm53-flash-a2-prefill-command"></a>

    Start the two Prefill ranks with the shared launcher:

    ```shell
    # P0
    python launch_online_dp.py \
      --template ./run_p0.sh \
      --dp-size 2 \
      --tp-size 1 \
      --pcp-size 8 \
      --dp-size-local 1 \
      --dp-rank-start 0 \
      --dp-address "<PREFILL_NODE0_IP>" \
      --dp-rpc-port 12321 \
      --vllm-start-port 9081

    # P1
    python launch_online_dp.py \
      --template ./run_p1.sh \
      --dp-size 2 \
      --tp-size 1 \
      --pcp-size 8 \
      --dp-size-local 1 \
      --dp-rank-start 1 \
      --dp-address "<PREFILL_NODE0_IP>" \
      --dp-rpc-port 12321 \
      --vllm-start-port 9082
    ```

    ##### Common Decode template

    Save the following `run_d.sh` on D0 and D1. Decode uses
    `FULL_DECODE_ONLY` graph mode for the target model and keeps FlashComm1
    disabled. The node wrappers below provide `LOCAL_IP`.

    ```shell
    #!/usr/bin/env bash
    # Usage: bash run_d.sh <visible_devices> <http_port> <dp_size> \
    #   <dp_rank> <dp_address> <rpc_port> <tp_size> <pcp_size>

    : "${LOCAL_IP:?Set LOCAL_IP in the D0 or D1 wrapper}"
    : "${NIC_NAME:?Set NIC_NAME in the D0 or D1 wrapper}"
    : "${MODEL_PATH:?Set MODEL_PATH in the D0 or D1 wrapper}"

    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_HOST_IP="$LOCAL_IP"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,pin_memory_expandable_segments:True
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=1200
    export VLLM_RPC_TIMEOUT=3600000
    export HCCL_INTRA_ROCE_ENABLE=1
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    # Reserve disjoint HCCL port ranges per DP rank so multiple engines on
    # one node never share host/NPU socket ports.
    export HCCL_HOST_SOCKET_PORT_RANGE="$((60000 + $4 * 100))-$((60099 + $4 * 100))"
    export HCCL_NPU_SOCKET_PORT_RANGE="$((60000 + $4 * 100))-$((60099 + $4 * 100))"
    export PYTHONHASHSEED=0
    export PYTHONPATH="/usr/local/python3.12.13/lib/python3.12/site-packages:${PYTHONPATH:-}"

    exec vllm serve "$MODEL_PATH" \
      --device-ids "$1" \
      --host 0.0.0.0 \
      --port "$2" \
      --data-parallel-size "$3" \
      --data-parallel-rank "$4" \
      --data-parallel-address "$5" \
      --data-parallel-rpc-port "$6" \
      --tensor-parallel-size "$7" \
      --enable-expert-parallel \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --seed 1024 \
      --served-model-name glm \
      --safetensors-load-strategy prefetch \
      --max-model-len 1048576 \
      --max-num-seqs 40 \
      --max-num-batched-tokens 240 \
      --trust-remote-code \
      --quantization ascend \
      --kv-cache-dtype int8 \
      --attention_config.indexer_kv_dtype int8 \
      --gpu-memory-utilization 0.92 \
      --async-scheduling \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --enable-auto-tool-choice \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": false}' \
      --additional-config '{
        "enable_shared_expert_dp": true,
        "scheduler_config": {
          "recompute_scheduler_enable": true
        },
        "enable_flashcomm1": false,
        "ascend_compilation_config": {
          "enable_npugraph_ex": true,
          "enable_static_kernel": false,
          "enable_super_kernel": false
        }
      }' \
      --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV2",
        "kv_role": "kv_consumer",
        "engine_id": "1",
        "kv_port": "30100",
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {"dp_size": 2, "tp_size": 1, "pcp_size": 8},
          "decode": {"dp_size": 8, "tp_size": 2, "pcp_size": 1}
        }
      }' \
      --prefix-cache-retention-interval None
    ```

    ##### D0 and D1 wrappers

    Save the following `run_d0.sh` on D0:

    ```shell
    #!/usr/bin/env bash
    export LOCAL_IP="<DECODE_NODE0_IP>"
    export NIC_NAME="<DECODE_NODE0_NIC>"
    export MODEL_PATH="<YOUR_MODEL_PATH>"
    exec bash ./run_d.sh "$@"
    ```

    Save the following `run_d1.sh` on D1:

    ```shell
    #!/usr/bin/env bash
    export LOCAL_IP="<DECODE_NODE1_IP>"
    export NIC_NAME="<DECODE_NODE1_NIC>"
    export MODEL_PATH="<YOUR_MODEL_PATH>"
    exec bash ./run_d.sh "$@"
    ```

    <a id="glm53-flash-a2-decode-command"></a>

    Start four TP2 engines on each Decode node(total DP8). `--dp-rank-start` is
    `0` on D0 and `4` on D1. `--dp-address` is D0's IP on both nodes, and
    the launcher assigns API ports `9900-9903` on each node.

    ```shell
    # D0
    python launch_online_dp.py \
      --template ./run_d0.sh \
      --dp-size 8 \
      --tp-size 2 \
      --dp-size-local 4 \
      --dp-rank-start 0 \
      --dp-address "<DECODE_NODE0_IP>" \
      --vllm-start-port 9900

    # D1
    python launch_online_dp.py \
      --template ./run_d1.sh \
      --dp-size 8 \
      --tp-size 2 \
      --dp-size-local 4 \
      --dp-rank-start 4 \
      --dp-address "<DECODE_NODE0_IP>" \
      --vllm-start-port 9900
    ```

    ##### Startup order

    Start the services in this order: the first node of each group, the
    remaining nodes, then the proxy.

    1. Complete the container and multi-node communication checks on all
       four nodes.
    2. Start P0 with the launcher command above (`--dp-rank-start 0`). P0
       hosts the Prefill API endpoint and the Prefill DP master.
    3. Start D0 with its launcher command (`--dp-rank-start 0`). D0 hosts
       the Decode DP master and four API endpoints.
    4. Immediately after launching P0 and D0, start P1 and D1 with their
       commands (`--dp-rank-start 1` and `--dp-rank-start 4`). Each DP group
       completes initialization after all its ranks have started.
    5. After all ten engines are ready, verify the API endpoints on P0, D0,
       and D1 with `curl http://127.0.0.1:<port>/v1/models` and check the
       headless P1 initialization logs. Start the proxy, then send requests
       to `<PREFILL_NODE0_IP>:8081`.

    <a id="glm53-flash-a2-proxy"></a>

    ##### Deploy the PD proxy

    Open another terminal in one Prefill container and start the
    [load-balancing proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).
    The proxy listens on port `8081`, sends Prefill requests to the P0 API
    endpoint, and forwards Decode work to the eight Decode endpoints.

    ```shell
    #!/usr/bin/env bash

    P0_IP="${P0_IP:-<PREFILL_NODE0_IP>}"
    D0_IP="${D0_IP:-<DECODE_NODE0_IP>}"
    D1_IP="${D1_IP:-<DECODE_NODE1_IP>}"

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
      --host 0.0.0.0 \
      --port 8081 \
      --prefiller-hosts "$P0_IP" \
      --prefiller-ports 9081 \
      --decoder-hosts "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" "$D1_IP" "$D1_IP" "$D1_IP" "$D1_IP" \
      --decoder-ports 9900 9901 9902 9903 9900 9901 9902 9903
    ```

    ##### Key parameter differences from the A3 series layout

    - A2 series uses global DP2/TP1/PCP8 Prefill and DP8/TP2 Decode, with each
      role spread across two nodes with eight devices per node. A3 ordinary PD uses
      DP2/TP1/PCP8 Prefill and DP8/TP2 Decode, each on one node with 16 logical devices.
    - P0 and P1 use the shared launcher with one local DP rank each. P0
      exposes the Prefill API with `--api-server-count 1`; P1 is headless.
      D0 and D1 each run four API-serving TP2 engines(total DP8) through their
      node-specific wrapper scripts.
    - `kv_connector_extra_config` enables `use_ascend_direct` for direct
      NPU-to-NPU KV transfer and declares the actual Prefill and Decode
      `{dp_size, tp_size, pcp_size}` topology. The declared values must match the
      real deployment.
    - Prefill uses `--max-num-seqs 64` and `--max-num-batched-tokens 4096`
      without `--enforce-eager`; Decode uses `--max-num-seqs 40` and
      `--max-num-batched-tokens 240` with `gpu-memory-utilization=0.92` and
      `FULL_DECODE_ONLY` graph mode.
      Both roles run with `--async-scheduling`.
    - Both roles use Model Runner V2 via `VLLM_USE_V2_MODEL_RUNNER=1`.
      Prefill uses PCP8 and TP1; Decode keeps PCP1.
      `--device-ids "$1"` binds each engine to the launcher-assigned devices.
      `VLLM_HOST_IP` advertises the IP that peers use to reach each
      engine for KV transfer. Prefill enables FlashComm1 and shared-expert DP;
      Decode enables the recompute scheduler and NPU graph extension while
      keeping FlashComm1 disabled and static/super kernels disabled.
    - MTP speculative decoding uses three speculative tokens on A2 series.
      Prefill uses `enforce_eager: true`; Decode uses `enforce_eager: false`
      with `FULL_DECODE_ONLY` graph mode.
    - `HCCL_HOST_SOCKET_PORT_RANGE` and `HCCL_NPU_SOCKET_PORT_RANGE` on
      Decode reserve a disjoint 100-port window per DP rank (starting at
      `60000 + rank * 100`). When several engines share one node, these
      ranges prevent the HCCL host-side sockets from silently sharing
      ports across ranks, which can block engine startup.
    - The remaining timeout and runtime variables
      (`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, `HCCL_EXEC_TIMEOUT`,
      `HCCL_CONNECT_TIMEOUT`, `VLLM_RPC_TIMEOUT`, `OMP_PROC_BIND`,
      `OMP_NUM_THREADS`) are the validated A2 series startup defaults for this
      model size; keep them unless your environment requires different
      values.
    - `HCCL_IF_IP` and all socket interface variables must select the
      service network used by the configured node IPs. The DP RPC,
      engine, Mooncake, and proxy ports must be allowed by the host
      firewall.

### 5.4 Prefill-Decode Disaggregation with KV Cache Pool

This section uses MemCache as the AscendStoreConnector backend for KV Cache Pooling.
On A2 series, MooncakeConnectorV2 remains the Prefill-to-Decode KV transfer connector;
MemCache provides the shared pool. Select the platform subsection that matches the
PD topology in [Section 5.3](#53-1p1d-pd-disaggregated-deployment).
`MultiConnector` combines two connectors:

- `MooncakeConnectorV2` continues to transfer KV cache from Prefill to Decode.
- `AscendStoreConnector` lets Prefill look up, load, and save cached prefixes
  in the KV Cache Pool, reducing repeated Prefill computation.

For backend installation, hardware dependencies, memory sizing, eviction,
and tenant options, refer to the [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md).

=== "950DT products"

    #### 950DT products {: #glm53-flash-5-4-950dt }

    Reuse the Prefill TP1/PCP8/DP1 and Decode TP1/PCP1/DP8 topology,
    launcher and proxy from [Section 5.3](#glm53-flash-5-3-950dt).
    Both roles use EP8, FP8 main KV cache and an FP8 indexer.
    For this pool example, keep `--max-model-len 133120`,
    `--limit-mm-per-prompt '{"image": 1, "video": 0}'` and explicit
    `--enable-chunked-prefill` on both roles. Prefill uses
    `HCCL_BUFFSIZE=3072`, `--max-num-seqs 256`,
    `--max-num-batched-tokens 24576`, `--gpu-memory-utilization 0.9` and
    `multistream_overlap_shared_expert: true`. Decode uses
    `--max-num-seqs 30`, `--max-num-batched-tokens 120`,
    `enable_fused_mc2: 0` and `enable_npugraph_ex: false`.

    Follow [Scenario 2: Memcache Backend](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend)
    for the Memcache backend configuration.

    Keep prefix caching, chunked prefill and the hybrid KV cache manager
    enabled. With prefix caching enabled, the model configuration selects
    `mamba_cache_mode=align`, which is required by `AscendStoreConnector`.
    Keep `use_layerwise=false` and PP=DCP=1. Do not enable
    [RecoverSSM](../../user_guide/feature_guide/speculative_decoding.md)
    for this deployment: its KV-transfer path does not accept `MultiConnector`.

    1. On the Prefill node, replace the final `--kv-transfer-config`
       argument in `run_dp_template.sh` with the following fragment.
       Keep the remaining Prefill flags from Section 5.3, including
       `ASCEND_RT_VISIBLE_DEVICES="$1"`, PCP8, FP8 indexer and eager execution.

        ```shell
          --kv-transfer-config \
          '{
            "kv_connector": "MultiConnector",
            "kv_role": "kv_producer",
            "engine_id": "0",
            "kv_connector_extra_config": {
              "connectors": [
                {
                  "kv_connector": "MooncakeConnectorV2",
                  "kv_role": "kv_producer",
                  "kv_port": "30100",
                  "kv_connector_extra_config": {
                    "prefill": {"dp_size": 1, "tp_size": 1, "pcp_size": 8},
                    "decode": {"dp_size": 8, "tp_size": 1, "pcp_size": 1}
                  }
                },
                {
                  "kv_connector": "AscendStoreConnector",
                  "kv_role": "kv_producer",
                  "kv_connector_extra_config": {
                    "backend": "memcache",
                    "use_layerwise": false,
                    "lookup_rpc_port": '"$((37000 + $4))"'
                  }
                }
              ]
            }
          }'
        ```

    2. On the Decode node, replace the final `--kv-transfer-config`
       argument in `run_dp_template.sh` with the following fragment.
       Keep the remaining Decode flags from Section 5.3, including
       `--device-ids "$1"`, the FP8 indexer, MTP5 and `FULL_DECODE_ONLY`.

        ```shell
          --kv-transfer-config \
          '{
            "kv_connector": "MultiConnector",
            "kv_role": "kv_consumer",
            "engine_id": "1",
            "kv_connector_extra_config": {
              "connectors": [
                {
                  "kv_connector": "MooncakeConnectorV2",
                  "kv_role": "kv_consumer",
                  "kv_port": "30800",
                  "kv_connector_extra_config": {
                    "prefill": {"dp_size": 1, "tp_size": 1, "pcp_size": 8},
                    "decode": {"dp_size": 8, "tp_size": 1, "pcp_size": 1}
                  }
                },
                {
                  "kv_connector": "AscendStoreConnector",
                  "kv_role": "kv_consumer",
                  "kv_connector_extra_config": {
                    "backend": "memcache",
                    "use_layerwise": false,
                    "consumer_is_to_load": false,
                    "consumer_is_to_put": false,
                    "lookup_rpc_port": '"$((37100 + $4))"'
                  }
                }
              ]
            }
          }'
        ```

    `$4` is the DP rank passed by the launcher. The Prefill lookup value
    is `37000`; the Decode values are `37100-37107`. `lookup_rpc_port`
    identifies the local IPC endpoint between the pooling scheduler and
    worker. Prefill loads and saves pooled prefixes; Decode receives KV
    cache through `MooncakeConnectorV2` and contributes DDR pool memory.

=== "A3 series"

    #### A3 series {: #glm53-flash-5-4-a3 }

    This KV Cache Pool example uses DP2/TP1/PCP8 Prefill and DP8/TP2 Decode,
    using Model Runner V2 and native KDA. Prepare separate `run_p.sh` and
    `run_d.sh` files for this scenario from the [A3 PD scripts](#glm53-flash-5-3-a3).
    Keep the ordinary PD serving parameters on both roles, including
    `ASCEND_RT_VISIBLE_DEVICES="$1"`, INT8 main KV cache,
    BF16 SSM state, MTP3, and the role-specific scheduling and tuning options.
    Prefill uses 32 sequences and 16384 batched tokens; Decode uses 40
    sequences and 240 batched tokens. Both roles limit image input to one image.
    Add `--enable-prefix-caching` for this pooling example.
    Add the Memcache environment settings and replace the connectors as
    described below. Reuse the ordinary PD launch commands and proxy mapping.

    ##### Prepare the Memcache Backend

    Follow [Scenario 2: Memcache Backend](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend)
    for prerequisite checks and memory sizing. The A3 series HCCS
    configuration uses `device_sdma`, which requires LingQu Computing Network
    1.5 or later. Also check the HDK and CANN requirements in the guide's
    [HDK and CANN Version Compatibility Reference](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#hdk-and-cann-version-compatibility-reference).

    LingQu Computing Network is a host-side platform prerequisite for `device_sdma`, not the version of the Memcache Python package. Before starting the containers, have the platform administrator check the installed LingQu release against the server's vendor installation records and confirm version 1.5 or later together with the HDK/CANN compatibility matrix linked above. Installing or upgrading `memcache-hybrid` alone does not establish this prerequisite.

    **When starting the A3 series containers on both Prefill and Decode nodes, you must include the following read-only mount in the Docker command from Section 4:**

    ```shell
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    ```

    The Docker images in Section 4 already include `memfabric-hybrid` and
    `memcache-hybrid`. Locate the installed Memcache package in each container:

    ```shell
    pip show memcache_hybrid
    ```

    The `Location` field from `pip show` is `<PYTHON_SITE_PACKAGES>` in the
    commands below. Replace `<CONFIG_DIRECTORY>` with the absolute directory
    containing the configuration files on each node.

    This example stores pooled KV cache in DDR memory on both the Prefill
    and Decode nodes. The Memcache LocalService initializes within each vLLM
    worker, and each worker contributes `20GB` of DDR memory to the pool.
    Decode receives KV cache through `MooncakeConnectorV2`.

    On the Prefill node, create `mmc-meta.conf`:

    ```ini
    ock.mmc.meta_service_url = tcp://<PREFILL_NODE_IP>:5000
    ock.mmc.meta_service.config_store_url = tcp://<PREFILL_NODE_IP>:6000
    ock.mmc.meta_service.metrics_url = http://<PREFILL_NODE_IP>:8000
    ock.mmc.log_level = info
    ```

    On both the Prefill and Decode nodes, create the following `mmc-local.conf`:

    ```ini
    ock.mmc.meta_service_url = tcp://<PREFILL_NODE_IP>:5000
    ock.mmc.local_service.config_store_url = tcp://<PREFILL_NODE_IP>:6000
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_sdma
    ock.mmc.local_service.dram.size = 20GB
    ock.mmc.local_service.max.dram.size = 1024GB
    ock.mmc.local_service.storage.enabled = false
    ```

    `dram.size` specifies the DDR capacity allocated per worker. Prefill
    (DP2/TP1/PCP8) and Decode (DP8/TP2) each start 16 workers. At `20GB` per worker,
    each node contributes `320GB`, for a total DDR pool capacity of `640GB`.
    Reserve sufficient DDR memory for the operating system and vLLM. For
    `device_sdma`, keep each non-zero size aligned to `1GB`.

    `world_size` is the maximum number of LocalServices supported by Memcache.
    Keep it consistent across all workers and large enough for the 32 workers
    in this deployment.

    ##### Add the Pool Environment Variables

    In both `run_p.sh` and `run_d.sh`, keep the [Section 5.3](#glm53-flash-5-3-a3) environment variables
    and add the following exports before `exec vllm serve`:

    ```shell
    MEMCACHE_PACKAGE_DIR="<PYTHON_SITE_PACKAGES>/memcache_hybrid"
    PYTHON_LIB_DIR="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"

    export PYTHONHASHSEED=0
    export MMC_LOCAL_CONFIG_PATH="<CONFIG_DIRECTORY>/mmc-local.conf"
    export LD_LIBRARY_PATH="${MEMCACHE_PACKAGE_DIR}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH:-}"

    # A3 series HCCS fabric-memory pooling.
    export ACL_OP_INIT_MODE=1
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    ```

    Use the same `PYTHONHASHSEED` and Memcache service endpoints on both nodes.
    Keep the HCCS communication settings from [Section 5.3](#glm53-flash-5-3-a3), including
    `ASCEND_GLOBAL_RESOURCE_CONFIG`.

    ##### Update the Prefill and Decode Connectors

    For A3 series KV Cache Pool deployment, add `--enable-prefix-caching` to both
    `run_p.sh` and `run_d.sh` prepared above, and keep
    chunked prefill enabled (the model default). GLM-5.3-Flash has
    linear-attention state, and `AscendStoreConnector`
    requires `mamba_cache_mode=align`; the model configuration selects this mode
    when prefix caching is enabled. Keep the hybrid KV cache manager enabled
    and `use_layerwise` disabled for this configuration.

    Replace the final `--kv-transfer-config` argument in `run_p.sh` with the
    following fragment. Apart from the prefix-caching change above, keep
    the remaining A3 series Prefill flags from ordinary PD, including
    `--max-num-seqs 32`.

    ```shell
      --kv-transfer-config \
      '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "engine_id": "0",
        "kv_connector_extra_config": {
          "connectors": [
            {
              "kv_connector": "MooncakeConnectorV2",
              "kv_role": "kv_producer",
              "kv_port": "36680"
            },
            {
              "kv_connector": "AscendStoreConnector",
              "kv_role": "kv_producer",
              "kv_connector_extra_config": {
                "backend": "memcache",
                "use_layerwise": false,
                "lookup_rpc_port": '"$((37000 + $4))"'
              }
            }
          ]
        }
      }'
    ```

    Replace the final `--kv-transfer-config` argument in `run_d.sh` with the
    following fragment. Apart from the prefix-caching change above, keep
    the remaining A3 series Decode flags from ordinary PD, including
    `FULL_DECODE_ONLY` graph mode and the MTP configuration.

    ```shell
      --kv-transfer-config \
      '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "engine_id": "1",
        "kv_connector_extra_config": {
          "connectors": [
            {
              "kv_connector": "MooncakeConnectorV2",
              "kv_role": "kv_consumer",
              "kv_port": "36580"
            },
            {
              "kv_connector": "AscendStoreConnector",
              "kv_role": "kv_consumer",
              "kv_connector_extra_config": {
                "backend": "memcache",
                "use_layerwise": false,
                "consumer_is_to_load": false,
                "consumer_is_to_put": false,
                "lookup_rpc_port": '"$((37100 + $4))"'
              }
            }
          ]
        }
      }'
    ```

    `$4` is the DP rank passed by `launch_online_dp.py`. Each engine must have a
    unique `lookup_rpc_port`. The Prefill lookup values are `37000-37001`, and the
    Decode lookup values are `37100-37107`. Keep the role-specific Mooncake KV
    base ports from [Section 5.3](#glm53-flash-5-3-a3); its workers and schedulers derive their own
    ports from the topology.

    ##### Start the Services

    Start the services in this order: Memcache MetaService → Decode → Prefill → Proxy. 
    Each vLLM worker initializes its Memcache LocalService during startup.

    1. In a separate terminal in the Prefill container, start MetaService.
       Ensure ports `5000` and `6000` are reachable from both nodes and port
       `8000` is available for metrics.

       ```shell
       MEMCACHE_PACKAGE_DIR="<PYTHON_SITE_PACKAGES>/memcache_hybrid"
       PYTHON_LIB_DIR="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
       export LD_LIBRARY_PATH="${MEMCACHE_PACKAGE_DIR}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH:-}"
       export MMC_META_CONFIG_PATH="<CONFIG_DIRECTORY>/mmc-meta.conf"
       python -c "from memcache_hybrid import MetaService; MetaService.main()"
       ```

    2. Start Decode with DP8/TP2. Wait until all eight engines on ports
       `9900-9907` are ready and their Memcache LocalServices have reserved
       the configured DDR pool memory.

       ```shell
       python launch_online_dp.py --template ./run_d.sh \
         --dp-size 8 --tp-size 2 --dp-address "<DECODE_NODE_IP>" \
         --vllm-start-port 9900
       ```

    3. Start Prefill with DP2/TP1/PCP8. Wait until both engines on ports
       `9081-9082` are ready and their Memcache LocalServices have reserved
       the configured DDR pool memory.

       ```shell
       python launch_online_dp.py --template ./run_p.sh \
         --dp-size 2 --tp-size 1 --pcp-size 8 --dp-address "<PREFILL_NODE_IP>" \
         --vllm-start-port 9081
       ```

    4. In a separate terminal, set `export D_N=8`, then run the
       [A3 series proxy script](#glm53-flash-a3-proxy). It uses Prefill ports
       `9081-9082` and Decode ports `9900-9907` for this pooling scenario.
       Send requests to `<PREFILL_NODE_IP>:8081` using
       [Section 6](#6-functional-verification).

    <a id="glm53-flash-pool-reuse"></a>

    ##### Verify KV Cache Reuse

    Warm up the pool through the proxy with a prompt longer than one cache
    block, then repeat requests with the same prefix. Check Prefill output for
    successful pool saves, lookups, loads, and cache hits, and check Decode
    output for successful P→D transfers.

    Capture the Prefill launcher's standard output and standard error when starting the service. Compare `vllm:ascend_store_load_get_keys_total` at each Prefill API's `/metrics` endpoint before and after repeating the prefix on the other Prefill rank: an increase records pool GET keys, unlike a local prefix-cache hit alone. Also verify that requests complete successfully; a GET counter alone does not prove a successful transfer or an accuracy result.

    The proxy distributes requests across two Prefill engines. A repeated
    request can also hit an engine's local prefix cache; confirm pool load/hit
    information on the other Prefill rank to verify shared pool reuse. Only
    complete, cacheable blocks are reused, so a partial trailing block can
    still require computation.

=== "A2 series"

    #### A2 series {: #glm53-flash-5-4-a2 }

    Use the common A2 Prefill/Decode templates and PCP-aware launcher from
    [Section 5.3](#glm53-flash-5-3-a2) as the baseline. This subsection gives
    only the pool-specific configuration; it does not repeat the serving or
    proxy scripts. Keep the Section 5.3 model parameters, including prefix
    caching and MTP with three speculative tokens.

    The pool topology is Prefill DP2/TP1/PCP8 and Decode DP8/TP2/PCP1.
    P0 and P1 both expose Prefill APIs in this pool deployment. In the common
    Prefill template from Section 5.3, keep the launcher-supplied DP arguments
    and add the external DP rank argument:
    `--data-parallel-rank "$4"`. Keep P0's `--api-server-count 1`.

    In P1's `run_p1.sh` wrapper from Section 5.3, replace
    `export SERVER_ROLE_ARGS="--headless --data-parallel-start-rank 1"` with
    `export SERVER_ROLE_ARGS="--api-server-count 1"`. The common `run_p.sh`
    expands `${SERVER_ROLE_ARGS:-}` into the `vllm serve` command; the launcher
    supplies P1's DP rank, so do not keep the headless/start-rank flags. This
    makes both Prefill ranks API-serving. Keep Decode DP settings from Section 5.3.

    Configure the pool proxy for both Prefill APIs and all Decode APIs:

    | Role | Engines | API endpoints |
    | --- | --- | --- |
    | Prefill | P0 rank 0; P1 rank 1 | P0:8000; P1:8000 |
    | Decode | D0 ranks 0-3; D1 ranks 4-7 | D0:8000-8003; D1:8000-8003 |
    | Pool proxy | one proxy instance | listen on 28000; route to both Prefill endpoints and all eight Decode endpoints |

    Keep `MooncakeConnectorV2` for Prefill-to-Decode KV transfer and configure
    `AscendStoreConnector` with the MemCache backend for the shared pool.

    ##### Prepare MemCache

    Follow the [MemCache backend setup](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend) for prerequisites and installation. For A2, check the guide's [HDK/CANN compatibility reference](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#hdk-and-cann-version-compatibility-reference): HDK 25.5 or later is recommended and CANN 9.1.0 or later is required.
    Confirm `/etc/hccn.conf` is visible inside every engine container (mount the host file when needed), and that the Section 5.3 `LOCAL_IP` and `NIC_NAME` match the A2 RoCE interface. This A2 setup uses `device_rdma`; A3 HCCS-specific `device_sdma` and FabricMem settings belong in the A3 instructions. Install `memfabric-hybrid>=1.3.0` and `memcache-hybrid>=1.3.0`. All nodes must reach the same MetaService.

    Prepare `mmc-meta.conf` on the MetaService host:

    ```ini
    ock.mmc.meta_service_url = tcp://<MEMCACHE_META_IP>:5000
    ock.mmc.meta_service.config_store_url = tcp://<MEMCACHE_META_IP>:6000
    ock.mmc.meta_service.metrics_url = http://<MEMCACHE_META_IP>:8009
    ock.mmc.log_level = info
    ```

    Prepare `mmc-local.conf` in shared storage or place an identical copy on
    every Prefill and Decode node. Keep its MetaService endpoints identical to
    `mmc-meta.conf`. `device_rdma` is the A2 RoCE protocol. `dram.size` is the
    allocation per LocalService; non-zero `device_*` allocations are aligned
    to 1GB. `40GB` is an example only. Check available host memory with
    `free -h`, reserve headroom for the OS and vLLM, and size the pool for the
    workload.

    In this topology, each host runs eight model-parallel workers when all
    configured devices are used: one Prefill rank with PCP8, or four Decode
    ranks with TP2. Each worker initializes a LocalService, so estimate each
    host's DRAM contribution as eight times `dram.size`; use the actual worker
    count if it differs. `world_size` is the maximum supported number of
    LocalServices, not a DP-rank count. Keep it consistent and large enough
    across the deployment; changing it after ranks connect requires restarting
    MetaService.

    ```ini
    ock.mmc.meta_service_url = tcp://<MEMCACHE_META_IP>:5000
    ock.mmc.local_service.config_store_url = tcp://<MEMCACHE_META_IP>:6000
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_rdma
    ock.mmc.local_service.dram.size = 40GB
    ock.mmc.local_service.max.dram.size = 1024GB
    ock.mmc.local_service.storage.enabled = false
    ```

    This configuration pools DRAM only; SSD device parameters are not needed.

    On every engine node, use the same MemCache installation/configuration
    paths and add these pool-related settings to the existing Section 5.3
    wrappers. Use `pip show memcache_hybrid` to find its `Location`; set
    `MEMCACHE_ROOT` to `<Location>/memcache_hybrid`. Set `PYTHON_LIB_DIR` to
    Python's `LIBDIR` (from `sysconfig`). Keep `PYTHONHASHSEED` and the
    `MMC_LOCAL_CONFIG_PATH` consistent across nodes. Keep Section 5.3's A2
    RoCE exports, including `HCCL_INTRA_ROCE_ENABLE=1` and the HCCL/GLOO/TP
    socket interface settings, as well as the existing `LOCAL_IP`, `NIC_NAME`,
    and `MODEL_PATH` wrapper settings.

    ```shell
    export PYTHONHASHSEED=0
    export ACL_OP_INIT_MODE=1
    export MEMCACHE_ROOT="<MEMCACHE_ROOT>"
    export PYTHON_LIB_DIR="<PYTHON_LIB_DIR>"
    export MMC_LOCAL_CONFIG_PATH="<PATH_TO_MMC_LOCAL_CONF>"
    export LD_LIBRARY_PATH="${MEMCACHE_ROOT}/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${PYTHON_LIB_DIR}:${PYTHONPATH:-}"
    ```

    `ACL_OP_INIT_MODE=1` selects ACL operator initialization mode to speed up operator compilation.

    Start MetaService before the vLLM engines using Step 4 in the
    [MemCache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend).
    Set `MMC_META_CONFIG_PATH` on the MetaService host to the configured
    `mmc-meta.conf` file.

    ##### Configure the KV connectors

    In the existing Prefill script from Section 5.3, replace only its
    `--kv-transfer-config` value with this fragment. `$4` is the DP rank
    supplied by `launch_online_dp.py`; it keeps each Prefill engine ID unique.

    ```shell
    --kv-transfer-config '{
      "kv_connector": "MultiConnector",
      "kv_role": "kv_producer",
      "kv_port": "36000",
      "engine_id": "'$4'",
      "kv_connector_extra_config": {
        "connectors": [
          {
            "kv_connector": "MooncakeConnectorV2",
            "kv_role": "kv_producer",
            "kv_port": "36000",
            "kv_connector_extra_config": {
              "use_ascend_direct": true,
              "prefill": {
                "dp_size": 2,
                "tp_size": 1,
                "pcp_size": 8
              },
              "decode": {
                "dp_size": 8,
                "tp_size": 2,
                "pcp_size": 1
              }
            }
          },
          {
            "kv_connector": "AscendStoreConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
              "backend": "memcache",
              "use_layerwise": false,
              "lookup_rpc_port": '"$((37000 + $4))"'
            }
          }
        ]
      }
    }'
    ```

    In the existing Decode script from Section 5.3, replace only its
    `--kv-transfer-config` value. Decode engine IDs use `100 +` the launcher
    DP rank to remain unique across the pool.

    ```shell
    --kv-transfer-config '{
      "kv_connector": "MultiConnector",
      "kv_role": "kv_consumer",
      "kv_port": "30100",
      "engine_id": "'$((100 + $4))'",
      "kv_connector_extra_config": {
        "connectors": [
          {
            "kv_connector": "MooncakeConnectorV2",
            "kv_role": "kv_consumer",
            "kv_port": "30100",
            "kv_connector_extra_config": {
              "use_ascend_direct": true,
              "prefill": {
                "dp_size": 2,
                "tp_size": 1,
                "pcp_size": 8
              },
              "decode": {
                "dp_size": 8,
                "tp_size": 2,
                "pcp_size": 1
              }
            }
          },
          {
            "kv_connector": "AscendStoreConnector",
            "kv_role": "kv_consumer",
            "kv_connector_extra_config": {
              "backend": "memcache",
              "use_layerwise": false,
              "lookup_rpc_port": '"$((37100 + $4))"'
            }
          }
        ]
      }
    }'
    ```

    `launch_online_dp.py` starts each DP rank in a separate engine process, so set
    `lookup_rpc_port` uniquely for every P/D engine: `37000 + $4` on Prefill and
    `37100 + $4` on Decode, where `$4` is the launcher-supplied DP rank. This
    value identifies a local IPC endpoint; it is not a TCP listening port. Also
    check Mooncake worker and scheduler TCP port ranges for conflicts with other
    listeners.

    ##### Start and verify the pool

    Start the services in this order: MetaService, Decode, Prefill, then the
    pool proxy. Start MetaService using Step 4 in the
    [MemCache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend),
    with `MMC_META_CONFIG_PATH` pointing to `mmc-meta.conf`. Ensure the configured
    MetaService ports 5000/6000 and metrics port 8009 are reachable from all
    engine nodes.

    Use the Section 5.3 launcher commands for D0/D1, then P0/P1 with the pool
    role settings above. Wait for all eight Decode APIs on ports 8000-8003 and
    both Prefill APIs on port 8000 to become ready, and confirm the MemCache
    LocalServices initialize before starting the existing proxy on port 28000.
    Route it to both Prefill endpoints and all eight Decode endpoints, then
    send the functional-verification requests from Section 6.

    Warm the pool with a prompt longer than one cache block, then repeat the
    same prefix on a different Prefill rank. Check both Prefill launch logs for
    pool saves, lookups, loads, and cache hits, and Decode logs for successful
    Prefill-to-Decode transfers. Compare
    `vllm:ascend_store_load_get_keys_total` at the Prefill API metrics endpoints
    before and after the cross-rank repeat; a local prefix-cache hit alone does
    not demonstrate shared-pool reuse. Only complete cache blocks are reused.

    Keep the Section 5.3 serving settings except for the pool-specific P1 API
    role, API endpoint mapping above, and the connector configuration here.

## 6 Functional Verification {: #6-functional-verification }

Once the selected deployment is ready, query its service endpoint. For the
950DT products PD deployment, use the separate proxy host and port `8000`.
For the A3 series/A2 PD deployment, use the Prefill node IP and proxy
port `8081`. For the A2 KV Cache Pool deployment, use the configured
proxy host and port `28000`. For a
single-node or A2 series colocated deployment, use the API endpoint exposed by its
vLLM service.

```shell
curl http://<service_ip>:<service_port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm",
        "prompt": "The future of AI is",
        "max_tokens": 50
    }'
```

Expected Result:
The response is a JSON payload containing the model's generated text in a text_completion format.

```json
{
  "id": "cmpl-123abc",
  "object": "text_completion",
  "created": 1725444000,
  "model": "glm",
  "choices": [
    {
      "text": " incredibly promising, with rapid advancements in machine learning and autonomous systems.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 15,
    "total_tokens": 20
  }
}
```

## 7 Accuracy Evaluation

### 7.1 Using AISBench {: #glm53-flash-accuracy-aisbench }

For detailed instructions, refer to [Using AISBench for accuracy evaluation](../../developer_guide/evaluation/using_ais_bench.md).

### 7.2 Accuracy Results with PD Disaggregation

The A3 series results below were obtained using an earlier PD
configuration with DP2/TP8 on the Prefill node and DP8/TP2 on the Decode node.
Evaluation requests were sent through the PD proxy. The table below lists the
generation parameters used for each dataset. See [Section 5.3](#53-1p1d-pd-disaggregated-deployment)
for the current serving examples.

| Dataset | Hardware | Score | generation_kwargs |
| --- | --- | --- | --- |
| GPQA | A3 series | 89.9 | `temperature=1.0, top_p=0.95, max_tokens=131072` |
| GPQA | 950DT products | 91.2 | `temperature=1.0, top_p=0.95, max_tokens=131072` |
| MMMU-PRO | A3 series | 78.4 | `temperature=1.0, top_p=0.95, max_tokens=98304` |

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations {: #91-atlas-800i-a3-pd-optimizations }

The following optimizations are already enabled in the A3 series reference
configuration in [Section 5.3](#glm53-flash-5-3-a3). Use the same W8A8 weights
and role-specific settings when reproducing the deployment.

| Optimization | Role | Configuration | Purpose and constraints |
| --- | --- | --- | --- |
| FlashComm1 | Decode | `enable_flashcomm1: true` | Optimizes tensor-parallel communication. |
| Prefill context parallelism | Prefill | `--prefill-context-parallel-size 8` | Each TP1 Prefill engine uses eight devices. |
| Prefill MoE communication | Prefill | `enable_fused_mc2: 2`, `--enable-eplb` | Uses fused MoE communication and asynchronous EPLB with the `stair` policy and `hixl` communicator. |
| Shared-expert overlap | Decode | `multistream_overlap_shared_expert: true` | Overlaps shared-expert execution with other work; the Decode configuration does not enable fused MC2. |
| Decode execution | Decode | `FULL_DECODE_ONLY` | Reduces decode launch overhead. |
| MTP | Prefill and Decode | `num_speculative_tokens: 3` | Uses draft graph execution on both roles; the Prefill target is eager. |
| KV capacity | Prefill and Decode | `--kv-cache-dtype int8` | Reduces KV cache memory consumption for long contexts. |

### 9.2 Tuning Guidelines {: #92-workload-specific-settings }

The settings below refer to the [A3 series PD example](#glm53-flash-5-3-a3).
Use the role-specific values in the 950DT products example when deploying on that hardware.

- Tune `--max-num-seqs` and `--max-num-batched-tokens` independently on each
  role. The reference uses 32 sequences and 16384 batched tokens on Prefill,
  and 40 sequences and 240 batched tokens on Decode. Measure time to first
  token, time per output token, and throughput before changing these limits.
- Set `--max-model-len` to the required context length and adjust
  `--gpu-memory-utilization` to the available NPU memory. The reference uses
  1048576 tokens and a utilization of 0.92 on both roles.
- Decode enables `recompute_scheduler_enable: true` to allow recomputation
  on Prefill when Decode KV cache is insufficient. Keep it disabled on Prefill.
- For repeated prefixes, use the DDR KV Cache Pool configuration in
  [Section 5.4](#54-prefill-decode-disaggregation-with-kv-cache-pool). Both
  Prefill and Decode contribute `20GB` per worker. Confirm pool hits when
  evaluating reuse benefits.

### 9.3 Tool-Assisted Tuning with msmodeling

The msmodeling tool (MindStudio-Modeling) tunes serving and benchmark parameters
using measured results and knowledge-base guidance. For the complete walkthrough,
refer to [Optix-for-PD-disaggregation](https://gitcode.com/Ascend/msmodeling/blob/master/docs/en/quick_start/optix_agent_pd_disaggregation.md).

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md); this chapter only covers model-specific issues.

- **Q: How to enable function calling for GLM-5.3-Flash?**

  A: Please add following configurations in vLLM startup command.

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --enable-auto-tool-choice \
  ```

- **Q: Does GLM-5.3-Flash support `enable_thinking: false`?**

  A: No, GLM-5.3-Flash does not support `enable_thinking`.
