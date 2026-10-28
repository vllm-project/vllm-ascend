# GLM-5.3-Flash

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is the first natively multimodal model in the GLM-5 series. Built on a hybrid architecture that combines sparse and linear attention for the first time in the GLM series, it adopts Manifold-Constrained Hyper-Connections (mHC) and is trained on a 30T-token multimodal pre-training corpus. With 320B total parameters and only 18B active parameters, it outperforms GLM-5.2 across benchmarks and real-world workloads at one-tenth the price, while approaching Claude Opus 4.8 on coding and agentic benchmarks. GLM-5.3-Flash also supports controlling the thinking budget through the `reasoning_effort` parameter (`low`, `high`, `max`).

This document shows the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, 1P1D Prefill-Decode (PD) disaggregated deployment, multi-node deployment, and accuracy and performance evaluation.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

GLM-5.3-Flash model currently supports only model runner V1 on Ascend, so
all A3 series scripts set `VLLM_USE_V2_MODEL_RUNNER=0` explicitly.

## 3 Prerequisites

### 3.1 Model Weight

Download the checkpoint from ModelScope for the hardware shown below. A3 and 950DT products checkpoints use different quantization formats; do not interchange them. The scripts use `/root/.cache/modelscope/hub/models/vllm-ascend/<checkpoint-name>` as a local mount path, not as a download source. Preserve the exact case of the repository name when naming the local directory.

| Checkpoint | Hardware | Deployment | Download | Notes |
| --- | --- | --- | --- | --- |
| `GLM-5.3-Flash-w8a8-mxfp8` | 950DT products (96GB × 8 per node) | 1 node for single-node deployment; 2 nodes for 1P1D PD | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8) | Supports a W8A8C8 configuration: C8 is dynamic KV-cache quantization. Set `--kv-cache-dtype fp8` in the serving command to enable FP8 KV cache; no separate C8 checkpoint is required. |
| `GLM-5.3-Flash-w8a8` | A3 series (16 logical devices × 64GB per node) | 1 node for single-node deployment; 2 nodes for 1P1D PD | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8) | Supports a W8A8C8 configuration: C8 is dynamic KV-cache quantization. Set `--kv-cache-dtype int8` in the serving command to enable INT8 KV cache; no separate C8 checkpoint is required. |
| `GLM-5.3-Flash-w8a8` | A2 series (64GB × 8 per node) | 2 nodes for colocated deployment; 4 nodes for PD | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8) | / |

- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

You can use the official all-in-one Docker image. For the available image tags and published versions, refer to [Using Docker](../../getting_started/installation.md#installation-prebuilt-image).

**Step 1:** Download the Docker image

```bash
docker pull quay.io/ascend/vllm-ascend:{tag}
```

**Step 2:** Start Docker container

Select the `docker run` command for your hardware platform:

=== "950DT products"

    Start the Docker container on each node.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:{tag}
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

    export IMAGE=quay.io/ascend/vllm-ascend:{tag}
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

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:{tag}
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

    **Check the software stack inside each A2 container**

    ```shell
    command -v python3 vllm
    python3 -m pip show vllm vllm-ascend transformers
    ```

    If startup reports that Transformers does not recognize `glm5_next`, the failure occurs while
    loading the model configuration, before DP engine initialization. Check
    the active software versions and recreate the containers with a compatible
    image tag if they still use the legacy `glm-5.3-flash` image or a vLLM
    `0.23.0` environment. For source builds, follow the
    [installation guide](../../getting_started/installation.md) and use the
    matching vLLM, vLLM-Ascend, and Transformers dependencies for the checked-out
    revision.

## 5 Online Service Deployment

!!! note

    Do not set `enable_thinking: false` / `thinking: false` for GLM-5.3-Flash, otherwise the output quality may degrade.

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

### 5.1 Single-Node Online Deployment

=== "950DT products"

    #### 950DT products {: #glm53-flash-5-1-950dt }

    - Quantized model `GLM-5.3-Flash-w8a8-mxfp8` can be deployed on 1 950DT products (96GB × 8).

    Run the following script to execute online inference.

    ```shell

    export VLLM_USE_V2_MODEL_RUNNER=0
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024

    vllm serve Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8 \
      --host 0.0.0.0 \
      --port 8000 \
      --data-parallel-size 1 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --kv-cache-dtype fp8 \
      --served-model-name glm \
      --max-num-seqs 32 \
      --max-model-len 132096 \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --gpu-memory-utilization 0.9 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32,64,96,128]}' \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

=== "A3 series"

    #### A3 series {: #glm53-flash-5-1-a3 }

    - Quantized model `GLM-5.3-Flash-w8a8` can be deployed on 1 A3 series (64GB × 16) .

    Run the following script to execute online inference.

    ```shell
    #!/bin/sh

    export VLLM_USE_V2_MODEL_RUNNER=0
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
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables graph capture for the decode phase only, improving decode performance by reducing kernel launch overhead.
- `--limit-mm-per-prompt '{"image":1,"video":0}'`: Allows at most one image per prompt and disables video input in all deployment examples.
- `--speculative-config`: Enables Multi-Token Prediction (MTP) speculative decoding with the DeepSeek-style MTP draft head of GLM-5.3-Flash. The single-node examples use three speculative tokens on 950DT products and five on A3 series. `enforce_eager: true` keeps the MTP draft model in eager mode on 950DT products, while `enforce_eager: false` enables graph mode for the draft model on A3 series. GLM-5.3-Flash supports MTP draft graph execution; the forced-eager restriction for other GLM models does not apply to this model.

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

    export VLLM_USE_V2_MODEL_RUNNER=0
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

    export VLLM_USE_V2_MODEL_RUNNER=0
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

### 5.3 1P1D PD Disaggregated Deployment

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
    subprocess.run(
        [
            "bash",
            args.template,
            devices,
            str(port),
            str(args.dp_size),
            str(dp_rank),
            args.dp_address,
            args.dp_rpc_port,
            str(args.tp_size),
        ],
        check=True,
    )


if __name__ == "__main__":
    args = parse_args()
    processes = []
    for i in range(args.dp_size_local):
        devices = ",".join(
            str(device)
            for device in range(i * args.tp_size, (i + 1) * args.tp_size)
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
address, DP RPC port, and TP size to each role script as `$1` through `$7`.

=== "950DT products"

    #### 950DT products {: #glm53-flash-5-3-950dt }

    ##### Topology and Prerequisites

    Use two nodes with eight devices each: Prefill DP1/TP8 and Decode DP8/TP1.
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
            MODEL_PATH="<MODEL_PATH>/GLM-5.3-Flash-W8A8-MXFP8-0918"

            export HCCL_IF_IP=$local_ip
            export GLOO_SOCKET_IFNAME=$nic_name
            export TP_SOCKET_IFNAME=$nic_name
            export HCCL_SOCKET_IFNAME=$nic_name
            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export HCCL_BUFFSIZE=3072
            export HCCL_OP_EXPANSION_MODE="AIV"
            export ASCEND_RT_VISIBLE_DEVICES=$1

            vllm serve "$MODEL_PATH" \
              --host 0.0.0.0 \
              --port $2 \
              --data-parallel-size $3 \
              --data-parallel-rank $4 \
              --data-parallel-address $5 \
              --data-parallel-rpc-port $6 \
              --tensor-parallel-size $7 \
              --enable-expert-parallel \
              --seed 1024 \
              --served-model-name glm \
              --safetensors-load-strategy prefetch \
              --kv-cache-dtype fp8 \
              --max-num-seqs 256 \
              --max-model-len 133120 \
              --max-num-batched-tokens 24576 \
              --trust-remote-code \
              --quantization ascend \
              --tool-call-parser glm47 \
              --enable-auto-tool-choice \
              --reasoning-parser glm47 \
              --enable-prefix-caching \
              --limit-mm-per-prompt '{"image": 1, "video": 0}' \
              --gpu-memory-utilization 0.9 \
              --compilation-config '{"cudagraph_capture_sizes":[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]}' \
              --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp"}' \
              --additional-config '{
                "enable_cpu_binding": true,
                "enable_flashcomm1": true,
                "enable_dsa_cp": true,
                "enable_fused_mc2": 0,
                "enable_shared_expert_dp": true,
                "multistream_overlap_shared_expert": true
              }' \
              --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_producer",
                "engine_id": "0",
                "kv_port": "30100",
                "kv_connector_extra_config": {
                  "prefill": {"dp_size": 1, "tp_size": 8},
                  "decode": {"dp_size": 8, "tp_size": 1}
                }
              }'
            ```

        2. Decode node

            ```shell
            #!/usr/bin/env bash

            local_ip="<D0_IP>"
            nic_name="<NETWORK_INTERFACE>"
            MODEL_PATH="<MODEL_PATH>/GLM-5.3-Flash-W8A8-MXFP8-0918"

            export HCCL_IF_IP=$local_ip
            export GLOO_SOCKET_IFNAME=$nic_name
            export TP_SOCKET_IFNAME=$nic_name
            export HCCL_SOCKET_IFNAME=$nic_name
            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export HCCL_BUFFSIZE=1024
            export HCCL_OP_EXPANSION_MODE="AIV"
            export ASCEND_RT_VISIBLE_DEVICES=$1

            vllm serve "$MODEL_PATH" \
              --host 0.0.0.0 \
              --port $2 \
              --data-parallel-size $3 \
              --data-parallel-rank $4 \
              --data-parallel-address $5 \
              --data-parallel-rpc-port $6 \
              --tensor-parallel-size $7 \
              --enable-expert-parallel \
              --seed 1024 \
              --served-model-name glm \
              --safetensors-load-strategy prefetch \
              --kv-cache-dtype fp8 \
              --max-num-seqs 30 \
              --max-model-len 133120 \
              --max-num-batched-tokens 120 \
              --trust-remote-code \
              --quantization ascend \
              --tool-call-parser glm47 \
              --enable-auto-tool-choice \
              --reasoning-parser glm47 \
              --enable-prefix-caching \
              --limit-mm-per-prompt '{"image": 1, "video": 0}' \
              --gpu-memory-utilization 0.95 \
              --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
              --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp"}' \
              --additional-config '{
                "enable_shared_expert_dp": true,
                "multistream_overlap_shared_expert": true,
                "recompute_scheduler_enable": true
              }' \
              --kv-transfer-config '{
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_consumer",
                "engine_id": "1",
                "kv_port": "30800",
                "kv_connector_extra_config": {
                  "prefill": {"dp_size": 1, "tp_size": 8},
                  "decode": {"dp_size": 8, "tp_size": 1}
                }
              }'
            ```

    ##### Start the Engines

    1. Prefill node

        ```shell
        P0_IP="<P0_IP>"
        python launch_online_dp.py --template ./run_dp_template.sh \
            --dp-size 1 --tp-size 8 --dp-size-local 1 \
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

    python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 8000 \
        --prefiller-hosts "$P0_IP" --prefiller-ports 8000 \
        --decoder-hosts "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
            "$D0_IP" "$D0_IP" "$D0_IP" "$D0_IP" \
        --decoder-ports 8000 8001 8002 8003 8004 8005 8006 8007
    ```

    **Key Parameter Descriptions**

    - `--data-parallel-size` and `--tensor-parallel-size` define DP1/TP8
      on Prefill and DP8/TP1 on Decode. Each role uses eight devices on
      one node.
    - `--data-parallel-address` and `--data-parallel-rpc-port` coordinate
      ranks within each role. Prefill and Decode use their respective node
      IPs; RPC port `12325` is reused on the separate hosts.
    - `--vllm-start-port 8000` assigns Prefill API port `8000` and Decode
      API ports `8000-8007`. Register all eight Decode endpoints with the
      proxy.
    - Both roles use `--max-model-len 133120`, `--kv-cache-dtype fp8`, and
      `deepseek_mtp` with five speculative tokens.
    - Prefill sets `--max-num-seqs 256` and
      `--max-num-batched-tokens 24576`; Decode sets `30` and `120`,
      respectively. Tune these per-engine scheduler limits for the workload.
    - Prefill enables FlashComm1, DSA CP, shared-expert DP and
      shared-expert multistream overlap, with `enable_fused_mc2: 0`.
      Decode enables shared-expert DP, shared-expert multistream overlap,
      recomputation scheduling and `FULL_DECODE_ONLY` graph mode.
    - Both roles enable prefix caching and allow one image per prompt,
      with video input disabled through `--limit-mm-per-prompt`.
    - `MooncakeConnectorV2` uses `kv_producer` on Prefill and `kv_consumer`
      on Decode, with base KV ports `30100` and `30800`, respectively.
      Both roles declare Prefill DP1/TP8 and Decode DP8/TP1.
    - `HCCL_IF_IP` and the socket interface variables select each node's
      communication network. `HCCL_BUFFSIZE` is `3072` on Prefill and
      `1024` on Decode. The DP RPC, API, Mooncake and proxy ports must be
      reachable between the corresponding hosts.

=== "A3 series"

    #### A3 series {: #glm53-flash-5-3-a3 }

    This example uses two A3 series servers. The Prefill node runs two
    DP ranks with TP8 (DP2/TP8), and the Decode node runs four DP ranks
    with TP4 (DP4/TP4). Both layouts consume all 16 logical devices on their
    respective servers. `MooncakeConnectorV2` transfers KV cache from the
    Prefill engines to the Decode engines.

    Before starting the services, replace `LOCAL_IP`, `NIC_NAME`, and
    `MODEL_PATH` in the following scripts with values for the deployment
    environment. These scripts target an A3 series supernode using HCCS with its
    device RoCE ports down. They explicitly select `hccs:device` for device
    communication; the socket interface variables still select the host
    network. Adjust the resource configuration for other network topologies.

    ##### A3 series Serving Scripts and Startup Commands

    ##### Start the Prefill node

    On the Prefill node, save the following script as `run_p.sh`. The launcher
    passes the visible devices, API port, DP configuration, and TP size as
    positional arguments.

    ```shell
    #!/usr/bin/env bash
    # Usage: bash run_p.sh <visible_devices> <http_port> <dp_size> \
    #   <dp_rank> <dp_address> <rpc_port> <tp_size>

    # This A3 series supernode uses HCCS; its device RoCE ports are down.
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
    export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

    LOCAL_IP="<PREFILL_NODE_IP>"
    NIC_NAME="<NETWORK_INTERFACE>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_USE_V2_MODEL_RUNNER=0
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
      --max-num-seqs 64 \
      --max-model-len 1048576 \
      --max-num-batched-tokens 4096 \
      --trust-remote-code \
      --quantization ascend \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --gpu-memory-utilization 0.92 \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --enable-auto-tool-choice \
      --kv-cache-dtype int8 \
      --prefix-cache-retention-interval None \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
      --additional-config '{
        "enable_flashcomm1": true,
        "enable_dsa_cp": true,
        "enable_prefill_mc2": true,
        "enable_fused_mc2": 1,
        "enable_shared_expert_dp": true,
        "multistream_overlap_shared_expert": false
      }' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_producer", "engine_id": "0", "kv_port": "36680"}'
    ```

    Start two DP2/TP8 Prefill engines. `--dp-address` uses the Prefill node IP.
    The launcher assigns API ports `9081-9082`.

    ```shell
    python launch_online_dp.py \
      --template ./run_p.sh \
      --dp-size 2 \
      --tp-size 8 \
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
    export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

    LOCAL_IP="<DECODE_NODE_IP>"
    NIC_NAME="<NETWORK_INTERFACE>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_USE_V2_MODEL_RUNNER=0
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
      --max-num-seqs 32 \
      --max-model-len 1048576 \
      --max-num-batched-tokens 192 \
      --trust-remote-code \
      --quantization ascend \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --gpu-memory-utilization 0.92 \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --enable-auto-tool-choice \
      --kv-cache-dtype int8 \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": false}' \
      --additional-config '{
        "multistream_overlap_shared_expert": true,
        "recompute_scheduler_enable": true,
        "enable_shared_expert_dp": true,
        "enable_flashcomm1": true
      }' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_consumer", "engine_id": "1", "kv_port": "36580"}'
    ```

    Start four TP4 Decode engines(total DP4). `--dp-address` uses the Decode node
    IP. The launcher assigns API ports `9900-9903`.

    ```shell
    python launch_online_dp.py \
      --template ./run_d.sh \
      --dp-size 4 \
      --tp-size 4 \
      --dp-address "<DECODE_NODE_IP>" \
      --vllm-start-port 9900
    ```

    ##### Deploy the PD Proxy

    After all Prefill and Decode engines are ready, open another terminal in
    the Prefill container and start the
    [load-balancing proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).
    The proxy listens on port `8081`, distributes requests across Prefill
    endpoints `9081-9082`, and then forwards Decode work to endpoints
    `9900-9903`.

    ```shell
    #!/usr/bin/env bash

    P_IP="${P_IP:-<PREFILL_NODE_IP>}"
    P_N=${P_N:-2}
    P_PORT0=${P_PORT0:-9081}
    D_IP="${D_IP:-<DECODE_NODE_IP>}"
    D_N=${D_N:-4}
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

    - `--data-parallel-size` and `--tensor-parallel-size` define DP2/TP8 on
      Prefill and DP4/TP4 on Decode. Both layouts use all 16 logical devices
      on their respective A3 series nodes.
    - `--data-parallel-address` and `--data-parallel-rpc-port` coordinate DP
      ranks within one role. Prefill and Decode use their respective node IPs.
      The default RPC port `12325` can be reused because the roles run on
      different hosts.
    - `--vllm-start-port 9081` assigns API ports `9081-9082` on Prefill, while
      `--vllm-start-port 9900` assigns `9900-9903` on Decode. All endpoints
      must be reachable from the proxy.
    - Both roles use `--max-model-len 1048576`, `--kv-cache-dtype int8`, and
      MTP speculative decoding with three speculative tokens. The MTP draft
      model uses `enforce_eager: true` on Prefill and `enforce_eager: false`
      on Decode.
    - `VLLM_USE_V2_MODEL_RUNNER=0` explicitly selects model runner V1 on both
      roles for this configuration.
    - Prefill uses `--max-num-seqs 64` and
      `--max-num-batched-tokens 4096`. Decode uses `--max-num-seqs 32` and
      `--max-num-batched-tokens 192`. Tune these role-specific scheduler limits
      independently for the target workload.
    - Prefill sets `--prefix-cache-retention-interval None`. For KV Cache
      Pool deployment, apply the prefix-caching changes in Section 5.4.
    - Decode uses `FULL_DECODE_ONLY` graph mode. Both roles explicitly
      enable FlashComm1.
    - Prefill sets `enable_dsa_cp: true`, `enable_prefill_mc2: true`,
      `enable_fused_mc2: 1`, and `enable_shared_expert_dp: true`, with
      `multistream_overlap_shared_expert: false`. Decode sets
      `multistream_overlap_shared_expert: true`,
      `enable_shared_expert_dp: true`, and `recompute_scheduler_enable: true`.
    - Both Prefill and Decode use `--limit-mm-per-prompt '{"image":1,"video":0}'`, allowing
      at most one image per prompt and disabling video input.
    - `MooncakeConnectorV2` transfers KV cache between the two roles.
      `kv_role` must be `kv_producer` on Prefill and `kv_consumer` on Decode.
      The role-specific KV ports must be available on their respective hosts.
    - `ASCEND_GLOBAL_RESOURCE_CONFIG` selects `hccs:device`, and
      `ASCEND_LOCAL_COMM_RES` sets resource configuration version `1.2` for
      this HCCS supernode. `HCCL_IF_IP` and all socket interface variables must
      select the host network used by the configured node IPs. The DP RPC,
      engine, Mooncake, and proxy ports must be allowed by the host firewall.

=== "A2 series"

    #### A2 series {: #glm53-flash-5-3-a2 }

    This example uses four A2 series (64GB × 8) servers: two Prefill
    nodes running one DP rank with TP8 each (DP2/TP8 in total), and two
    Decode nodes running four DP ranks with TP2 each (DP8/TP2 in total).
    Each Prefill rank uses eight devices, and each Decode rank uses two.
    Together, the local ranks use all eight devices on each server.
    `MooncakeConnectorV2` transfers KV cache from the Prefill engines to the
    Decode engines. The topology names the two Prefill nodes P0/P1 and the
    two Decode nodes D0/D1. P0 serves the Prefill API endpoint; P1 is
    headless. The proxy addresses the API endpoints on both Decode nodes.

    The shared `launch_online_dp.py` launcher from Section 5.3 starts the
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
    #   <dp_rank> <dp_address> <rpc_port> <tp_size>

    : "${LOCAL_IP:?Set LOCAL_IP in the P0 or P1 wrapper}"
    : "${NIC_NAME:?Set NIC_NAME in the P0 or P1 wrapper}"
    : "${MODEL_PATH:?Set MODEL_PATH in the P0 or P1 wrapper}"

    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_HOST_IP="$LOCAL_IP"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_USE_V2_MODEL_RUNNER=0
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=1200
    export VLLM_RPC_TIMEOUT=3600000
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export ASCEND_RT_VISIBLE_DEVICES="$1"
    export PYTHONHASHSEED=0
    export VLLM_TORCH_PROFILER_DIR="./vllm_profile"
    export VLLM_TORCH_PROFILER_WITH_STACK=1
    export PYTHONPATH="/usr/local/python3.12.13/lib/python3.12/site-packages:${PYTHONPATH:-}"

    exec vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port "$2" \
      --data-parallel-size "$3" \
      --data-parallel-size-local 1 \
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
      --max-num-seqs 64 \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --quantization ascend \
      --kv-cache-dtype int8 \
      --attention_config.indexer_kv_dtype int8 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --gpu-memory-utilization 0.9 \
      --async-scheduling \
      --enforce-eager \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --enable-auto-tool-choice \
      --speculative-config '{"num_speculative_tokens":5,"method":"deepseek_mtp","enforce_eager":true}' \
      --profiler-config '{"profiler":"torch","torch_profiler_dir":"./profiling","torch_profiler_with_stack":false}' \
      --additional-config '{"mc2_comm_alg":"hierarchy","multistream_overlap_shared_expert":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true}' \
      ${SERVER_ROLE_ARGS:-} \
      --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV2",
        "kv_role": "kv_producer",
        "engine_id": "0",
        "kv_port": "36000",
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {"dp_size": 2, "tp_size": 8},
          "decode": {"dp_size": 8, "tp_size": 2}
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

    Start the two Prefill ranks with the shared launcher:

    ```shell
    # P0
    python launch_online_dp.py \
      --template ./run_p0.sh \
      --dp-size 2 \
      --tp-size 8 \
      --dp-size-local 1 \
      --dp-rank-start 0 \
      --dp-address "<PREFILL_NODE0_IP>" \
      --dp-rpc-port 12321 \
      --vllm-start-port 9081

    # P1
    python launch_online_dp.py \
      --template ./run_p1.sh \
      --dp-size 2 \
      --tp-size 8 \
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
    #   <dp_rank> <dp_address> <rpc_port> <tp_size>

    : "${LOCAL_IP:?Set LOCAL_IP in the D0 or D1 wrapper}"
    : "${NIC_NAME:?Set NIC_NAME in the D0 or D1 wrapper}"
    : "${MODEL_PATH:?Set MODEL_PATH in the D0 or D1 wrapper}"

    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_HOST_IP="$LOCAL_IP"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,pin_memory_expandable_segments:True
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_USE_V2_MODEL_RUNNER=0
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=1200
    export VLLM_RPC_TIMEOUT=3600000
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    # Reserve disjoint HCCL port ranges per DP rank so multiple engines on
    # one node never share host/NPU socket ports.
    export HCCL_HOST_SOCKET_PORT_RANGE="$((60000 + $4 * 100))-$((60099 + $4 * 100))"
    export HCCL_NPU_SOCKET_PORT_RANGE="$((60000 + $4 * 100))-$((60099 + $4 * 100))"
    export ASCEND_RT_VISIBLE_DEVICES="$1"
    export PYTHONHASHSEED=0
    export PYTHONPATH="/usr/local/python3.12.13/lib/python3.12/site-packages:${PYTHONPATH:-}"

    exec vllm serve "$MODEL_PATH" \
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
      --max-num-seqs 60 \
      --max-num-batched-tokens 192 \
      --trust-remote-code \
      --quantization ascend \
      --kv-cache-dtype int8 \
      --attention_config.indexer_kv_dtype int8 \
      --gpu-memory-utilization 0.85 \
      --skip-mm-profiling \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --async-scheduling \
      --tool-call-parser glm47 \
      --reasoning-parser glm47 \
      --enable-auto-tool-choice \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": false}' \
      --profiler-config '{"profiler":"torch","torch_profiler_dir":"./profile","torch_profiler_with_stack":false}' \
      --additional-config '{"mc2_comm_alg": "hierarchy","multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_static_kernel": true}}' \
      --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV2",
        "kv_role": "kv_consumer",
        "engine_id": "1",
        "kv_port": "30100",
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {"dp_size": 2, "tp_size": 8},
          "decode": {"dp_size": 8, "tp_size": 2}
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

    - A2 series uses global DP2/TP8 Prefill and DP8/TP2 Decode, with each
      role spread across two nodes with eight devices per node. A3 series uses
      DP2/TP8 Prefill and DP4/TP4 Decode, each on one node with 16 logical devices.
    - P0 and P1 use the shared launcher with one local DP rank each. P0
      exposes the Prefill API with `--api-server-count 1`; P1 is headless.
      D0 and D1 each run four API-serving TP2 engines(total DP8) through their
      node-specific wrapper scripts.
    - `kv_connector_extra_config` enables `use_ascend_direct` for direct
      NPU-to-NPU KV transfer and declares the actual Prefill and Decode
      `{dp_size, tp_size}` topology. The declared values must match the
      real deployment.
    - Prefill uses `--max-num-seqs 64` and `--max-num-batched-tokens 8192`
      with `--enforce-eager`; Decode uses `--max-num-seqs 32` and
      `--max-num-batched-tokens 1024` with `FULL_DECODE_ONLY` graph mode.
      Both roles run with `--async-scheduling`.
    - Both roles rely on model runner V1 via
      `VLLM_USE_V2_MODEL_RUNNER=0`, consistent with the other tabs in
      this guide. `VLLM_HOST_IP` advertises the IP that peers use to
      reach each engine for KV transfer.
    - MTP speculative decoding uses three speculative tokens on A2 series.
      `enforce_eager: true` keeps the MTP draft model in eager mode.
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

This section uses the Memcache backend for A3 series and the Mooncake
backend for A2 series. Select the platform subsection that matches the PD
topology in [Section 5.3](#53-1p1d-pd-disaggregated-deployment). Both
platforms use `MultiConnector` to combine two connectors:

- `MooncakeConnectorV2` continues to transfer KV cache from Prefill to Decode.
- `AscendStoreConnector` lets Prefill look up, load, and save cached prefixes
  in the KV Cache Pool, reducing repeated Prefill computation.

For backend installation, hardware dependencies, memory sizing, eviction,
and tenant options, refer to the [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md).

=== "A3 series"

    #### A3 series {: #glm53-flash-5-4-a3 }

    Reuse the A3 series DP2/TP8 Prefill and DP4/TP4 Decode topology, launcher, and
    proxy endpoint mapping from Section 5.3.

    ##### Prepare the Memcache Backend

    Follow [Scenario 2: Memcache Backend](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend)
    for prerequisite checks and memory sizing. The A3 series HCCS
    configuration uses `device_sdma`, which requires LingQu Computing Network
    1.5 or later. Also check the HDK and CANN requirements in the guide's
    [HDK and CANN Version Compatibility Reference](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#hdk-and-cann-version-compatibility-reference).
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
    (DP2/TP8) and Decode (DP4/TP4) each start 16 workers. At `20GB` per worker,
    each node contributes `320GB`, for a total DDR pool capacity of `640GB`.
    Reserve sufficient DDR memory for the operating system and vLLM. For
    `device_sdma`, keep each non-zero size aligned to `1GB`.

    `world_size` is the maximum number of LocalServices supported by Memcache.
    Keep it consistent across all workers and large enough for the 32 workers
    in this deployment.

    ##### Add the Pool Environment Variables

    In both `run_p.sh` and `run_d.sh`, keep the Section 5.3 environment variables
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
    Keep the HCCS communication settings from Section 5.3, including
    `ASCEND_GLOBAL_RESOURCE_CONFIG` and `ASCEND_LOCAL_COMM_RES`.

    ##### Update the Prefill and Decode Connectors

    For A3 series KV Cache Pool deployment, add `--enable-prefix-caching` to both
    `run_p.sh` and `run_d.sh` in Section 5.3, and keep
    chunked prefill enabled (the model default). GLM-5.3-Flash has
    linear-attention state, and `AscendStoreConnector`
    requires `mamba_cache_mode=align`; the model configuration selects this mode
    when prefix caching is enabled. Keep the hybrid KV cache manager enabled
    and `use_layerwise` disabled for this configuration.

    Replace the final `--kv-transfer-config` argument in `run_p.sh` with the
    following fragment. Apart from the prefix-caching change above, keep
    the remaining A3 series Prefill flags from Section 5.3, including
    `--max-num-seqs 64`.

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
                "memcache_dp_init_barrier": true,
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
    the remaining A3 series Decode flags from Section 5.3, including
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
                "memcache_dp_init_barrier": true,
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
    Decode lookup values are `37100-37103`. Keep the role-specific Mooncake KV
    base ports from Section 5.3; its workers and schedulers derive their own
    ports from the topology.

    Keep `memcache_dp_init_barrier` set to `true` on every rank in each DP
    group. It synchronizes workers after Memcache initialization and before
    NPU graph capture. Set it in the `AscendStoreConnector` child, as shown
    above.

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

    2. Start Decode with the A3 series launcher command in Section 5.3. Wait until
       all four Decode engines on ports `9900-9903` are ready and their
       Memcache LocalServices have reserved the configured DDR pool memory.
    3. Start Prefill with the A3 series launcher command in Section 5.3. Wait until
       both Prefill engines on ports `9081-9082` are ready and their Memcache
       LocalServices have reserved the configured DDR pool memory.
    4. Start the A3 series proxy from Section 5.3. Send inference requests to
       `<PREFILL_NODE_IP>:8081` using the examples in Section 6.

    ##### Verify KV Cache Reuse

    Warm up the pool through the proxy with a prompt longer than one cache
    block, then repeat requests with the same prefix. Check Prefill output for
    successful pool saves, lookups, loads, and cache hits, and check Decode
    output for successful P→D transfers.

    The proxy distributes requests across two Prefill engines. A repeated
    request can also hit an engine's local prefix cache; confirm pool load/hit
    information on the other Prefill rank to verify shared pool reuse. Only
    complete, cacheable blocks are reused, so a partial trailing block can
    still require computation.

=== "A2 series"

    #### A2 series {: #glm53-flash-5-4-a2 }

    Reuse the `launch_online_dp.py` launcher, the P0/P1 and D0/D1 scripts, the
    DP2/TP8 Prefill and DP8/TP2 Decode topology, and the proxy configuration from
    Section 5.3. A2 series nodes do not have the A3 series HCCS fabric: the
    pool runs over the A2 series RoCE network with the `P2PHANDSHAKE` metadata server
    and the `ascend` protocol, and the A3 series fabric-memory exports do not apply.

    Retain the Prefill internal-LB settings from Section 5.3, including
    P1's `--headless --data-parallel-start-rank 1` wrapper and the P0-only
    proxy endpoint. The pool fragments derive distinct lookup ports from `$4`;
    this does not require adding `--data-parallel-rank` to the Prefill serving
    command.

    For backend installation, memory sizing, eviction, and tenant options, refer
    to the [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md).

    ##### Prepare the Mooncake Configuration

    The A2 series Docker command in Section 4 already mounts `/etc/hccn.conf`.
    Install the Mooncake backend according to the KV Cache Pool Deployment
    Guide, then create the following `mooncake.json` on both Prefill nodes.
    Mooncake Master runs in the first Prefill container at
    `<PREFILL_NODE0_IP>:50088`.

    ```json
    {
      "metadata_server": "P2PHANDSHAKE",
      "protocol": "ascend",
      "device_name": "",
      "master_server_address": "<PREFILL_NODE0_IP>:50088",
      "global_segment_size": "8GB",
      "preferred_segment": true,
      "prefer_alloc_in_same_node": true,
      "enable_ssd_offload": false,
      "tenant_id": "default"
    }
    ```

    On both Decode nodes, use the same JSON configuration with
    `global_segment_size` set to `0`. Use the same `tenant_id` on all four
    nodes. `global_segment_size` is registered per worker: with DP2/TP8
    spread across two Prefill nodes, each node starts 8 workers. The validated
    example uses `8GB` per worker, a `128GB` pool in total; adjust the
    per-worker size to the available host memory, keeping each non-zero segment
    size aligned to `1GB`. Keep `global_segment_size` at `0` in the Decode
    `mooncake.json`.

    ##### Add the Pool Environment Variables

    In both `run_p.sh` and `run_d.sh`, keep the Section 5.3 environment
    variables and add the following exports before `exec vllm serve`. Replace
    `<CONFIG_DIRECTORY>` with the absolute directory containing that node's
    `mooncake.json`.

    ```shell
    export PYTHONHASHSEED=0
    export MOONCAKE_CONFIG_PATH="<CONFIG_DIRECTORY>/mooncake.json"
    ```

    All nodes must use the same `PYTHONHASHSEED`. Do not add the A3 series
    `ASCEND_ENABLE_USE_FABRIC_MEM` export; it applies to the A3 series HCCS fabric
    only. The general A2 series branch of the KV Cache Pool Deployment Guide also
    suggests huge pages and `HCCL_INTRA_ROCE_ENABLE`; with the `P2PHANDSHAKE`
    metadata server and the `ascend` protocol these are not required, and the
    validated configuration does not set them.

    ##### Update the Prefill and Decode Connectors

    Keep prefix caching and chunked prefill enabled. `use_layerwise` must
    remain `false` for this model: the GLM-5.3-Flash hybrid KV cache contains a
    non-prefix-cacheable group, and layer-wise pool transfer requires every
    group to be cacheable, so the pool stores and loads whole segments. No
    manual `--mamba-cache-mode` flag is needed either: when the connector list
    contains `AscendStoreConnector`, the engine switches the mamba cache mode to
    `align` automatically.

    Replace the final `--kv-transfer-config` argument in `run_p.sh` with the
    following fragment. Keep the remaining Prefill flags from Section 5.3,
    including `--max-num-seqs 64`.

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
              "kv_port": "30000",
              "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 2, "tp_size": 8},
                "decode": {"dp_size": 8, "tp_size": 2}
              }
            },
            {
              "kv_connector": "AscendStoreConnector",
              "kv_role": "kv_producer",
              "kv_connector_extra_config": {
                "backend": "mooncake",
                "use_layerwise": false,
                "lookup_rpc_port": '"$((37000 + $4))"'
              }
            }
          ]
        }
      }'
    ```

    Replace the final `--kv-transfer-config` argument in `run_d.sh` with the
    following fragment. Keep the remaining Decode flags from Section 5.3,
    including `FULL_DECODE_ONLY` graph mode and the MTP configuration.

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
              "kv_port": "30100",
              "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 2, "tp_size": 8},
                "decode": {"dp_size": 8, "tp_size": 2}
              }
            },
            {
              "kv_connector": "AscendStoreConnector",
              "kv_role": "kv_consumer",
              "kv_connector_extra_config": {
                "backend": "mooncake",
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

    Both role templates use the launcher DP-rank argument `$4` for the
    lookup port. Each engine must have a unique `lookup_rpc_port`. The Prefill
    lookup ports are `37000-37001`, and the Decode lookup ports are
    `37100-37107`. The `prefill` and `decode` topology declared in the
    `MooncakeConnectorV2` fragment must match the real deployment.

    ##### Start the Services

    Start the services in this order: Mooncake Master → Decode → Prefill →
    Proxy.

    1. In a separate terminal in the first Prefill container, start Mooncake
       Master. Ensure port `50088` is reachable from all four nodes.

       ```shell
       mooncake_master \
         --port 50088 \
         --eviction_high_watermark_ratio 0.9 \
         --eviction_ratio 0.1 \
         --default_kv_lease_ttl 11000 \
         --enable_offload=false \
         --client_ttl=120
       ```

    2. Start D0 and D1 with the launcher commands from Section 5.3
       (`--dp-rank-start 0` and `4`; four engines per node on ports
       `9900-9903`). Wait until all eight Decode engines answer
       `curl /v1/models`.
    3. Start P0 and P1 with the launcher commands from Section 5.3
       (`--dp-rank-start 0` and `1`; P0 serves the API endpoint on port
       `9081`, while P1 is headless). Wait until the P0 API endpoint is
       ready and P1 logs confirm successful initialization.
    4. Start the A2 series proxy from Section 5.3. Send inference requests to
       `<PREFILL_NODE0_IP>:8081` using the examples in Section 6.

    ##### Verify KV Cache Reuse

    Follow the KV Cache reuse verification steps in the A3 series subsection. The proxy
    sends requests to the P0 API while the DP group uses both Prefill ranks. A
    repeated request can hit a local prefix cache, so confirm pool load/hit
    information on the other Prefill rank to verify shared pool reuse.

## 6 Functional Verification

Once the selected deployment is ready, query its service endpoint. For the
950DT products PD deployment, use the separate proxy host and port `8000`.
For the A3 series/A2 PD deployment, use the Prefill node IP and proxy
port `8081`. For a
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

### 7.1 Using AISBench

For detailed instructions, refer to [Using AISBench for accuracy evaluation](../../developer_guide/evaluation/using_ais_bench.md).

### 7.2 Accuracy Results with PD Disaggregation

The following accuracy results were obtained using an earlier A3 series PD
configuration with DP2/TP8 on the Prefill node and DP8/TP2 on the Decode node.
Evaluation requests were sent through the PD proxy. The table below lists the
generation parameters used for each dataset. See [Section 5.3](#53-1p1d-pd-disaggregated-deployment)
for the current serving examples.

| Dataset | Hardware | Score | generation_kwargs |
| --- | --- | --- | --- |
| GPQA | A3 series | 89.9 | `temperature=1.0, top_p=0.95, max_tokens=131072` |
| MMMU-PRO | A3 series | 78.4 | `temperature=1.0, top_p=0.95, max_tokens=98304` |

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations {: #91-atlas-800i-a3-pd-optimizations }

The following optimizations are already enabled in the reference configuration
in [Section 5.3](#53-1p1d-pd-disaggregated-deployment). Use the same W8A8 weights
and role-specific settings when reproducing the deployment.

| Optimization | Role | Configuration | Purpose and constraints |
| --- | --- | --- | --- |
| FlashComm1 | Prefill and Decode | `enable_flashcomm1: true` | Optimizes tensor-parallel communication. |
| DSA context parallelism | Prefill | `enable_dsa_cp: true` | Distributes sparse-attention work for long prompts; use model runner V1. |
| Prefill MoE communication | Prefill | `enable_prefill_mc2: true`, `enable_fused_mc2: 1`, `enable_shared_expert_dp: true` | Uses fused MoE communication with the reference W8A8 weights and EP16 layout; keep `multistream_overlap_shared_expert: false` on this role. |
| Shared-expert overlap | Decode | `multistream_overlap_shared_expert: true` | Overlaps shared-expert execution with other work; the Decode configuration does not enable fused MC2. |
| Decode execution | Decode | `FULL_DECODE_ONLY` | Reduces decode launch overhead. |
| MTP | Prefill and Decode | `num_speculative_tokens: 5` | Keeps the Prefill draft in eager mode and enables draft graph execution on Decode. |
| KV capacity | Prefill and Decode | `--kv-cache-dtype int8` | Reduces KV cache memory consumption for long contexts. |

### 9.2 Tuning Guidelines {: #92-workload-specific-settings }

- Tune `--max-num-seqs` and `--max-num-batched-tokens` independently on each
  role. The reference uses 64 sequences and 4096 batched tokens on Prefill,
  and 32 sequences and 192 batched tokens on Decode. Measure time to first
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
