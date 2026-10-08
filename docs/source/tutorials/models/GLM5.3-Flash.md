# GLM-5.3-Flash (Experimental)

## 1 Introduction

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is the first natively multimodal model in the GLM-5 series. Built on a hybrid architecture that combines sparse and linear attention for the first time in the GLM series, it adopts Manifold-Constrained Hyper-Connections (mHC) and is trained on a 30T-token multimodal pre-training corpus. With 320B total parameters and only 18B active parameters, it outperforms GLM-5.2 across benchmarks and real-world workloads at one-tenth the price, while approaching Claude Opus 4.8 on coding and agentic benchmarks. GLM-5.3-Flash also supports controlling the thinking budget through the `reasoning_effort` parameter (`low`, `high`, `max`).

This document shows the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, 1P1D Prefill-Decode (PD) disaggregated deployment, multi-node deployment, and accuracy and performance evaluation.

This document is written based on the vLLM-Ascend v0.30.0RC. This model is supported in this release.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

The A3 PD configuration in this guide uses DP2/TP8 on the Prefill node and
DP16/TP1 on the Decode node. Prefill runs in eager mode with FlashComm1, while
Decode uses `FULL_DECODE_ONLY` graph mode and keeps FlashComm1 disabled.

The A2 PD configuration in this guide uses DP2/TP8 across two Prefill
nodes and DP8/TP2 across two Decode nodes. Prefill runs in eager mode,
while Decode uses `FULL_DECODE_ONLY` graph mode and keeps FlashComm1
disabled.

GLM-5.3-Flash model currently supports only model runner V1 on Ascend, so
all A3 scripts set `VLLM_USE_V2_MODEL_RUNNER=0` explicitly.

## 3 Prerequisites

### 3.1 Model Weight

- `GLM-5.3-Flash-w8a8-mxfp8 (950DT Products mxfp8 Quantized)`: requires 1 950DT Products (96GB × 8) node.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8).
- `GLM-5.3-Flash-w8a8`: requires 1 Atlas 800 A3 (128GB × 8) node for
  single-node deployment, or 2 nodes for 1P1D PD disaggregated deployment.
  [Download model weight](https://modelers.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8).
- `GLM-5.3-Flash-w8a8`: requires 2 Atlas 800 A2 (64GB × 8) nodes.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8).

- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

- You can use our official docker image to run GLM-5.3-Flash directly.

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

## 5 Online Service Deployment

!!! note

    Do not set `enable_thinking: false` / `thinking: false` for GLM-5.3-Flash, otherwise the output quality may degrade.

### 5.1 Single-Node Online Deployment

=== "950DT Products"

    - Quantized model `GLM-5.3-Flash-w8a8-mxfp8` can be deployed on 1 950DT Products (96GB × 8) .

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
      --served-model-name glm \
      --max-num-seqs 32 \
      --max-model-len 132096 \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --gpu-memory-utilization 0.9 \
      --limit-mm-per-prompt '{"image": 1, "video": 0}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32,64,96,128]}' \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

=== "Atlas 800 A3 series"

    - Quantized model `GLM-5.3-Flash-w8a8` can be deployed on 1 A3 (64GB × 16) .

    Run the following script to execute online inference.

    ```shell
    #!/bin/sh

    export VLLM_USE_V2_MODEL_RUNNER=0
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_BUFFSIZE=400

    vllm serve Eco-Tech/GLM-5.3-Flash-w8a8   \
      --host 0.0.0.0 \
      --port 8000 \
      --max-model-len 133120  \
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
      --limit-mm-per-prompt '{"image": 1, "video": 0}' \
      --gpu-memory-utilization 0.92 \
      --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --additional_config '{"multistream_overlap_shared_expert": true}' \
      --api-server-count 1
    ```

#### Key Parameter Descriptions

Only the key parameters specific to this model/scenario are described below. `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario.

**Model-specific parameters:**

- `--data-parallel-size 1`: Runs a single DP rank. `--tensor-parallel-size` is 8 on 950DT Products and 16 on Atlas 800 A3. This layout is recommended to balance memory capacity and compute efficiency for the w8a8 weights.
- `--enable-expert-parallel`: Must be enabled for the MoE architecture of GLM-5.3-Flash.
- `--quantization ascend`: Enables Ascend quantization for the w8a8 quantized weights.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables graph capture for the decode phase only, improving decode performance by reducing kernel launch overhead.
- `--limit-mm-per-prompt '{"image": 1, "video": 0}'`: For text-only deployment, --limit-mm-per-prompt can be omitted. For multimodal deployment, configure this parameter according to the actual request shape. For example, use --limit-mm-per-prompt '{"image":2,"video":0}' for two-image requests, and use --limit-mm-per-prompt '{"image":0,"video":1}' for one-video requests.
- `--speculative-config`: Enables Multi-Token Prediction (MTP) speculative decoding with the DeepSeek-style MTP draft head of GLM-5.3-Flash. The single-node examples use three speculative tokens on 950DT Products and five on Atlas 800 A3. `enforce_eager: true` keeps the MTP draft model in eager mode because GLM-5.3-Flash does not support graph-mode speculative decoding.

### 5.2 1P1D PD Disaggregated Deployment

Both A3 and A2 deployments use the same `launch_online_dp.py` script to
start one vLLM process per local DP rank. Save it on every Prefill and Decode
node before following the platform-specific instructions below.

#### 5.2.1 Prepare the DP Launcher

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

=== "Atlas 800 A3 series"

    This example uses two Atlas 800 A3 servers. The Prefill node runs two
    DP ranks with TP8 (DP2/TP8), and the Decode node runs sixteen DP ranks
    with TP1 (DP16/TP1). Both layouts consume all 16 logical devices on their
    respective servers. `MooncakeConnectorV2` transfers KV cache from the
    Prefill engines to the Decode engines.

    Before starting the services, replace `LOCAL_IP`, `NIC_NAME`, and
    `MODEL_PATH` in the following scripts with values for the deployment
    environment.

    **A3 Serving Scripts and Startup Commands**

    **Start the Prefill node**

    On the Prefill node, save the following script as `run_p.sh`. The launcher
    passes the visible devices, API port, DP configuration, and TP size as
    positional arguments.

    ```shell
    #!/usr/bin/env bash
    # Usage: bash run_p.sh <visible_devices> <http_port> <dp_size> \
    #   <dp_rank> <dp_address> <rpc_port> <tp_size>

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
      --max-num-seqs 32 \
      --max-model-len 133120 \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --enforce-eager \
      --quantization ascend \
      --limit-mm-per-prompt '{"image": 1, "video": 0}' \
      --gpu-memory-utilization 0.92 \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}' \
      --additional_config '{"multistream_overlap_shared_expert":true,"enable_flashcomm1":true}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_producer", "kv_port": "36680"}'
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

    **Start the Decode Node**

    On the Decode node, save the following script as `run_d.sh`. Decode uses
    `FULL_DECODE_ONLY` graph mode for the target model. FlashComm1 remains
    disabled by default.

    ```shell
    #!/usr/bin/env bash
    # Usage is the same as run_p.sh.

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
      --max-num-seqs 10 \
      --max-model-len 133120 \
      --max-num-batched-tokens 60 \
      --trust-remote-code \
      --quantization ascend \
      --limit-mm-per-prompt '{"image": 1, "video": 0}' \
      --gpu-memory-utilization 0.92 \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": true}' \
      --additional_config '{"multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_static_kernel": true}}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_consumer", "kv_port": "36580"}'
    ```

    Start sixteen DP16/TP1 Decode engines. `--dp-address` uses the Decode node
    IP. The launcher assigns API ports `9900-9915`.

    ```shell
    python launch_online_dp.py \
      --template ./run_d.sh \
      --dp-size 16 \
      --tp-size 1 \
      --dp-address "<DECODE_NODE_IP>" \
      --vllm-start-port 9900
    ```

    **Deploy the PD Proxy**

    After all Prefill and Decode engines are ready, open another terminal in
    the Prefill container and start the
    [load-balancing proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).
    The proxy listens on port `8081`, distributes requests across Prefill
    endpoints `9081-9082`, and then forwards Decode work to endpoints
    `9900-9915`.

    ```shell
    #!/usr/bin/env bash

    P_IP="${P_IP:-<PREFILL_NODE_IP>}"
    P_N=${P_N:-2}
    P_PORT0=${P_PORT0:-9081}
    D_IP="${D_IP:-<DECODE_NODE_IP>}"
    D_N=${D_N:-16}
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
      Prefill and DP16/TP1 on Decode. Their product must be 16 on each A3 node.
    - `--data-parallel-address` and `--data-parallel-rpc-port` coordinate DP
      ranks within one role. Prefill and Decode use their respective node IPs.
      The default RPC port `12325` can be reused because the roles run on
      different hosts.
    - `--vllm-start-port 9081` assigns API ports `9081-9082` on Prefill, while
      `--vllm-start-port 9900` assigns `9900-9915` on Decode. All endpoints
      must be reachable from the proxy.
    - Both roles use a maximum model length of `133120` and MTP speculative
      decoding with five speculative tokens. `enforce_eager` in the
      speculative configuration keeps the MTP draft model in eager mode.
    - `VLLM_USE_V2_MODEL_RUNNER=0` explicitly selects model runner V1 on both
      roles for this validated configuration.
    - Prefill uses `--max-num-seqs 32` and
      `--max-num-batched-tokens 8192`. Decode uses `--max-num-seqs 10` and
      `--max-num-batched-tokens 60`. Tune these role-specific scheduler limits
      independently for the target workload.
    - `MooncakeConnectorV2` transfers KV cache between the two roles.
      `kv_role` must be `kv_producer` on Prefill and `kv_consumer` on Decode.
      The role-specific KV ports must be available on their respective hosts.
    - Prefill uses `--enforce-eager` for the target model and explicitly
      enables FlashComm1 with `enable_flashcomm1: true`. Decode leaves
      FlashComm1 disabled and uses `FULL_DECODE_ONLY` for the target model.
    - `multistream_overlap_shared_expert: true` overlaps shared-expert and
      routed-expert work. CPU binding remains enabled by default on both roles;
      Decode additionally enables the static kernel in
      `ascend_compilation_config`.
    - `HCCL_IF_IP` and all socket interface variables must select the service
      network used by the configured node IPs. The DP RPC, engine, Mooncake,
      and proxy ports must be allowed by the host firewall.

=== "Atlas 800 A2 series"

    This example uses four Atlas 800 A2 (64GB × 8) servers: two Prefill
    nodes running one DP rank with TP8 each (DP2/TP8 in total), and two
    Decode nodes running four DP ranks with TP2 each (DP8/TP2 in total).
    Each rank consumes all eight devices on its server.
    `MooncakeConnectorV2` transfers KV cache from the Prefill engines to the
    Decode engines. The topology names the two Prefill nodes P0/P1 and the
    two Decode nodes D0/D1. P0 serves the Prefill API endpoint; P1 is
    headless. The proxy addresses the API endpoints on both Decode nodes.

    The shared `launch_online_dp.py` launcher from Section 5.2.1 starts the
    engines on all four nodes. Save the common serving templates below and the
    node-specific wrapper scripts on their respective nodes. Replace all IP,
    NIC, and model-path placeholders.

    **Common Prefill template**

    Save the following script as `run_p.sh` on P0 and P1. The node wrappers
    below provide `LOCAL_IP` and the API/headless role.

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
    export VLLM_USE_V2_MODEL_RUNNER=0
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=1200
    export VLLM_RPC_TIMEOUT=3600000
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
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
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --seed 1024 \
      --served-model-name glm \
      --safetensors-load-strategy prefetch \
      --max-model-len 200000 \
      --max-num-seqs 64 \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --quantization ascend \
      --gpu-memory-utilization 0.92 \
      --async-scheduling \
      --enforce-eager \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
      ${SERVER_ROLE_ARGS:-} \
      --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV2",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {"dp_size": 2, "tp_size": 8},
          "decode": {"dp_size": 8, "tp_size": 2}
        }
      }'
    ```

    **P0 and P1 wrappers**

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
    export SERVER_ROLE_ARGS="--headless"
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

    **Common Decode template**

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
    export VLLM_USE_V2_MODEL_RUNNER=0
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
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
      --max-model-len 200000 \
      --max-num-seqs 32 \
      --max-num-batched-tokens 1024 \
      --trust-remote-code \
      --quantization ascend \
      --gpu-memory-utilization 0.85 \
      --skip-mm-profiling \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --async-scheduling \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
      --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV2",
        "kv_role": "kv_consumer",
        "kv_port": "30100",
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {"dp_size": 2, "tp_size": 8},
          "decode": {"dp_size": 8, "tp_size": 2}
        }
      }'
    ```

    **D0 and D1 wrappers**

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

    Start four DP8/TP2 engines on each Decode node. `--dp-rank-start` is
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

    **Startup order**

    Start the services in this order: the first node of each group, the
    remaining nodes, then the proxy.

    1. Complete the container and multi-node communication checks on all
       four nodes.
    2. Start P0 with the launcher command below (`--dp-rank-start 0`). P0
       hosts the Prefill API endpoint and the Prefill DP master.
    3. Start D0 with its launcher command (`--dp-rank-start 0`). D0 hosts
       the Decode DP master and four API endpoints.
    4. After P0 and D0 are ready, start P1 and D1 with their commands
       (`--dp-rank-start 1` and `--dp-rank-start 4`).
    5. After all ten engines are ready, start the proxy. Verify each
       engine first with
       `curl http://127.0.0.1:<port>/v1/models`, then send requests to
       `<PREFILL_NODE0_IP>:8081`.

    **Deploy the PD proxy**

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

    **Key parameter differences from the A3 layout**

    - DP2/TP8 on Prefill and DP8/TP2 on Decode are the validated A2
      layouts. Their products must be 8 on each A2 node instead of 16 on
      each A3 node.
    - P0 and P1 use the shared launcher with one local DP rank each. P0
      exposes the Prefill API with `--api-server-count 1`; P1 is headless.
      D0 and D1 each run four API-serving DP8/TP2 engines through their
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
    - MTP speculative decoding uses three speculative tokens on A2.
      `enforce_eager: true` keeps the MTP draft model in eager mode, as in
      the other tabs of this guide.
    - `HCCL_HOST_SOCKET_PORT_RANGE` and `HCCL_NPU_SOCKET_PORT_RANGE` on
      Decode reserve a disjoint 100-port window per DP rank (starting at
      `60000 + rank * 100`). When several engines share one node, these
      ranges prevent the HCCL host-side sockets from silently sharing
      ports across ranks, which can block engine startup.
    - The remaining timeout and runtime variables
      (`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, `HCCL_EXEC_TIMEOUT`,
      `HCCL_CONNECT_TIMEOUT`, `VLLM_RPC_TIMEOUT`, `OMP_PROC_BIND`,
      `OMP_NUM_THREADS`) are the validated A2 startup defaults for this
      model size; keep them unless your environment requires different
      values.
    - `HCCL_IF_IP` and all socket interface variables must select the
      service network used by the configured node IPs. The DP RPC,
      engine, Mooncake, and proxy ports must be allowed by the host
      firewall.

### 5.3 Multi-Node Colocated Deployment

=== "A2 series"

    - Quantized model `GLM-5.3-Flash-w8a8` can be deployed on 2 Atlas 800 A2 (64GB × 8) nodes with DP2 across the two nodes (one DP rank per node) and TP8 inside each node.

    Run the following scripts on two nodes respectively.

    **node 0**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
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
        --data-parallel-start-rank 0 \
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

    **node 1**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
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

#### Key Parameter Descriptions

**Multi-node network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`: Network interface configuration for multi-node communication. Set `nic_name` to the network interface name (obtained via `ifconfig`) and `local_ip` to the current node's IP address. These must be correctly configured on each node for successful multi-node communication.
- `--data-parallel-size 2 --data-parallel-size-local 1`: Runs two DP ranks across the two nodes, one rank per node; each rank uses TP8 within its node.
- `--data-parallel-start-rank`: Starting DP rank offset of the current node. Node 0 uses `0`, node 1 uses `1`.
- `--data-parallel-address`: IP address of the data parallel master node (node 0). Must match the `local_ip` of the master node.
- `--data-parallel-rpc-port 12321`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: Indicates a non-master node (used on node 1). Do not use on node 0.

### 5.4 Prefill-Decode Disaggregation with KV Cache Pool

This section documents Mooncake KV Cache Pool deployment for both Atlas 800
A3 and Atlas 800 A2. Select the platform subsection that matches the PD
topology in [Section 5.2](#52-1p1d-pd-disaggregated-deployment). Both
platforms use `MultiConnector` to combine two connectors:

- `MooncakeConnectorV2` continues to transfer KV cache from Prefill to Decode.
- `AscendStoreConnector` lets Prefill look up, load, and save cached prefixes
  in the Mooncake KV Cache Pool, reducing repeated Prefill computation.

Decode receives KV through the P→D connector. Its pool connector keeps
`consumer_is_to_load` and `consumer_is_to_put` disabled, and contributes no
pool memory. This configuration reuses pooled prefixes among Prefill engines
with the same TP8 layout; it does not load TP8 pool entries directly into TP1
Decode engines.

For backend installation, hardware dependencies, memory sizing, eviction,
and tenant options, refer to the [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_pool.md).

=== "Atlas 800 A3 series"

    Reuse the A3 DP2/TP8 Prefill and DP16/TP1 Decode topology, launcher, and
    proxy endpoint mapping from Section 5.2.

    **Prepare the Containers and Mooncake Configuration**

    Install the Mooncake backend according to the KV Cache Pool Deployment Guide.
    For A3 HCCS pooling, check the HDK, CANN, and LingQu Computing Network
    requirements in its [Hardware Dependency Quick Reference](../../user_guide/feature_guide/kv_pool.md#ascend_global_resource_config).
    Add the following mount to the A3 Docker command in Section 4.1 on both nodes:

    ```shell
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    ```

    Create a separate `mooncake.json` on each node. Replace `<PREFILL_NODE_IP>`
    with the Prefill node IP used in Section 5.2; Mooncake Master runs on that
    node at port `50088`. Use the same `tenant_id` on both nodes.

    Prefill `mooncake.json`:

    ```json
    {
      "metadata_server": "P2PHANDSHAKE",
      "protocol": "ascend",
      "device_name": "",
      "master_server_address": "<PREFILL_NODE_IP>:50088",
      "global_segment_size": "64GB",
      "preferred_segment": true,
      "prefer_alloc_in_same_node": true,
      "enable_ssd_offload": false,
      "tenant_id": "default"
    }
    ```

    `global_segment_size` is registered per worker, not per node. With DP2/TP8,
    the Prefill node starts 16 workers, so `64GB` per worker reserves `1TB` in
    total. Adjust this example to the available fabric memory, keeping each
    non-zero segment size aligned to `1GB`.

    Decode `mooncake.json`:

    ```json
    {
      "metadata_server": "P2PHANDSHAKE",
      "protocol": "ascend",
      "device_name": "",
      "master_server_address": "<PREFILL_NODE_IP>:50088",
      "global_segment_size": 0,
      "preferred_segment": true,
      "prefer_alloc_in_same_node": true,
      "enable_ssd_offload": false,
      "tenant_id": "default"
    }
    ```

    **Add the Pool Environment Variables**

    In both `run_p.sh` and `run_d.sh`, keep the Section 5.2 environment variables
    and add the following exports before `exec vllm serve`. Replace
    `<CONFIG_DIRECTORY>` with the absolute directory containing that node's
    `mooncake.json`.

    ```shell
    export PYTHONHASHSEED=0
    export MOONCAKE_CONFIG_PATH="<CONFIG_DIRECTORY>/mooncake.json"

    # A3 HCCS fabric-memory pooling.
    export ACL_OP_INIT_MODE=1
    export ASCEND_ENABLE_USE_FABRIC_MEM=1

    # Optional: set MOONCAKE_LIB_DIRS if Mooncake uses a custom library path.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    ```

    The two nodes must use the same `PYTHONHASHSEED`. For A3 RoCE pooling, use the
    communication and huge-page settings in the KV Cache Pool Deployment Guide
    instead of the HCCS fabric-memory exports above.

    **Update the Prefill and Decode Connectors**

    Prefix caching and chunked prefill are enabled by default for this model,
    so the serving scripts omit their explicit flags. Keep both features
    enabled. GLM-5.3-Flash has linear-attention state, and `AscendStoreConnector`
    requires `mamba_cache_mode=align`; the model configuration selects this mode
    when prefix caching is enabled. Keep the hybrid KV cache manager enabled
    and `use_layerwise` disabled for this configuration.

    Replace the final `--kv-transfer-config` argument in `run_p.sh` with the
    following fragment. Keep the remaining
    Prefill flags from Section 5.2.2, including `--max-num-seqs 32`.

    ```shell
      --kv-transfer-config \
      '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "engine_id": "glm53-flash-prefill-dp'"$4"'",
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
    following fragment. Keep the remaining Decode flags from Section 5.2.3,
    including `FULL_DECODE_ONLY` graph mode and the MTP configuration.

    ```shell
      --kv-transfer-config \
      '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "engine_id": "glm53-flash-decode-dp'"$4"'",
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

    `$4` is the DP rank passed by `launch_online_dp.py`. Each engine must have a
    unique `engine_id` and `lookup_rpc_port`; the child connectors inherit the
    outer `engine_id`. The Prefill lookup values are `37000-37001`, and the
    Decode lookup values are `37100-37115`. Keep the role-specific Mooncake KV
    base ports from Section 5.2; its workers and schedulers derive their own
    ports from the topology.

    **Start the Services**

    Start the services in this order: Mooncake Master → Decode → Prefill → Proxy.

    1. In a separate terminal in the Prefill container, start Mooncake Master.
       Ensure port `50088` is reachable from both nodes.

       ```shell
       mooncake_master \
         --port 50088 \
         --eviction_high_watermark_ratio 0.9 \
         --eviction_ratio 0.1 \
         --default_kv_lease_ttl 11000 \
         --enable_offload=false \
         --client_ttl=120
       ```

    2. Start Decode with the launcher command in Section 5.2.3. Wait until all
       sixteen Decode engines on ports `9900-9915` are ready.
    3. Start Prefill with the launcher command in Section 5.2.2. Wait until both
       Prefill engines on ports `9081-9082` are ready.
    4. Start the proxy from Section 5.2.4. Send inference requests to
       `<PREFILL_NODE_IP>:8081` using the examples in Section 6.

    **Verify KV Cache Reuse**

    Warm up the pool through the proxy with a prompt longer than one cache
    block, then repeat requests with the same prefix. Check Prefill output for
    successful pool saves, lookups, loads, and cache hits, and check Decode
    output for successful P→D transfers.

    The proxy distributes requests across two Prefill engines. A repeated
    request can also hit an engine's local prefix cache; confirm pool load/hit
    information on the other Prefill rank to verify shared pool reuse. Only
    complete, cacheable blocks are reused, so a partial trailing block can
    still require computation.

=== "Atlas 800 A2 series"

    Reuse the `launch_online_dp.py` launcher, the P0/P1 and D0/D1 scripts, the
    DP2/TP8 Prefill and DP8/TP2 Decode topology, and the proxy configuration from
    Section 5.2. A2 nodes do not have the A3 HCCS fabric: the
    pool runs over the A2 RoCE network with the `P2PHANDSHAKE` metadata server
    and the `ascend` protocol, and the A3 fabric-memory exports do not apply.

    For backend installation, memory sizing, eviction, and tenant options, refer
    to the [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_pool.md).

    **Prepare the Mooncake Configuration**

    The A2 Docker command in Section 4.1 already mounts `/etc/hccn.conf`.
    Install the Mooncake backend according to the KV Cache Pool Deployment
    Guide, then create the Prefill and Decode `mooncake.json` files using the
    JSON shape shown in the A3 subsection. Set `master_server_address` to
    `<PREFILL_NODE0_IP>:50088`; Mooncake Master runs in the first Prefill
    container. `global_segment_size` is registered per worker: with DP2/TP8
    spread across two Prefill nodes, each node starts 8 workers. The validated
    example uses `8GB` per worker, a `128GB` pool in total; adjust the
    per-worker size to the available host memory, keeping each non-zero segment
    size aligned to `1GB`. Keep `global_segment_size` at `0` in the Decode
    `mooncake.json`.

    **Add the Pool Environment Variables**

    In both `run_p.sh` and `run_d.sh`, keep the Section 5.2 environment
    variables and add the following exports before `exec vllm serve`. Replace
    `<CONFIG_DIRECTORY>` with the absolute directory containing that node's
    `mooncake.json`.

    ```shell
    export PYTHONHASHSEED=0
    export MOONCAKE_CONFIG_PATH="<CONFIG_DIRECTORY>/mooncake.json"
    ```

    All nodes must use the same `PYTHONHASHSEED`. Do not add the A3
    `ASCEND_ENABLE_USE_FABRIC_MEM` export; it applies to the A3 HCCS fabric
    only. The general A2 branch of the KV Cache Pool Deployment Guide also
    suggests huge pages and `HCCL_INTRA_ROCE_ENABLE`; with the `P2PHANDSHAKE`
    metadata server and the `ascend` protocol these are not required, and the
    validated configuration does not set them.

    **Update the Prefill and Decode Connectors**

    Keep prefix caching and chunked prefill enabled. `use_layerwise` must
    remain `false` for this model: the GLM-5.3-Flash hybrid KV cache contains a
    non-prefix-cacheable group, and layer-wise pool transfer requires every
    group to be cacheable, so the pool stores and loads whole segments. No
    manual `--mamba-cache-mode` flag is needed either: when the connector list
    contains `AscendStoreConnector`, the engine switches the mamba cache mode to
    `align` automatically.

    Replace the final `--kv-transfer-config` argument in `run_p.sh` with the
    following fragment. Keep the remaining Prefill flags from Section 5.2,
    including `--max-num-seqs 64`.

    ```shell
      --kv-transfer-config \
      '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "engine_id": "glm53-flash-prefill-dp'"$4"'",
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
    following fragment. Keep the remaining Decode flags from Section 5.2,
    including `FULL_DECODE_ONLY` graph mode and the MTP configuration.

    ```shell
      --kv-transfer-config \
      '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "engine_id": "glm53-flash-decode-dp'"$4"'",
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
    `engine_id` suffix and lookup port. Each engine must have a unique
    `engine_id` and `lookup_rpc_port`; the child connectors inherit the outer
    `engine_id`. The Prefill lookup ports are `37000-37001`, and the Decode lookup
    ports are `37100-37107`. The `prefill` and `decode` topology declared in the
    `MooncakeConnectorV2` fragment must match the real deployment.

    **Start the Services**

    Start the services in this order: Mooncake Master → Decode → Prefill →
    Proxy.

    1. In a separate terminal in the first Prefill container, start Mooncake
       Master with the command from Section 5.4.1.4. Ensure port `50088` is
       reachable from all four nodes.
    2. Start D0 and D1 with the launcher commands from Section 5.2
       (`--dp-rank-start 0` and `4`; four engines per node on ports
       `9900-9903`). Wait until all eight Decode engines answer
       `curl /v1/models`.
    3. Start P0 and P1 with the launcher commands from Section 5.2
       (`--dp-rank-start 0` and `1`; P0 serves the API endpoint on port
       `9081`, while P1 is headless). Wait until both engines answer.
    4. Start the A2 proxy from Section 5.2. Send inference requests to
       `<PREFILL_NODE0_IP>:8081` using the examples in Section 6.

    **Verify KV Cache Reuse**

    Follow the verification steps in Section 5.4.1.5 through the proxy. The proxy
    sends requests to the P0 API while the DP group uses both Prefill ranks. A
    repeated request can hit a local prefix cache, so confirm pool load/hit
    information on the other Prefill rank to verify shared pool reuse.

## 6 Functional Verification

Once the selected deployment is ready, query its service endpoint. For the
1P1D PD deployment, use the Prefill node IP and proxy port `8081`. For a
single-node or A2 colocated deployment, use the API endpoint exposed by its
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
The expected result of this request is a JSON payload containing the model’s generated text in a text_completion format.

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

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 FAQ

- **Q: How to enable function calling for GLM-5.3-Flash?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```

- **Q: Does GLM-5.3-Flash support `enable_thinking: false`?**

  A: No, GLM-5.3-Flash does not support `enable_thinking`.
