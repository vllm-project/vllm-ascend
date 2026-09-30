# GLM-5.3-Flash (Experimental)

## 1 Introduction

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is the first natively multimodal model in the GLM-5 series. Built on a hybrid architecture that combines sparse and linear attention for the first time in the GLM series, it adopts Manifold-Constrained Hyper-Connections (mHC) and is trained on a 30T-token multimodal pre-training corpus. With 320B total parameters and only 18B active parameters, it outperforms GLM-5.2 across benchmarks and real-world workloads at one-tenth the price, while approaching Claude Opus 4.8 on coding and agentic benchmarks. GLM-5.3-Flash also supports controlling the thinking budget through the `reasoning_effort` parameter (`low`, `high`, `max`).

This document shows the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, 1P1D Prefill-Decode (PD) disaggregated deployment, multi-node deployment, and accuracy and performance evaluation.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

The A3 PD configuration in this guide uses DP2/TP8 on the Prefill node and
DP16/TP1 on the Decode node. Prefill runs in eager mode with FlashComm1, while
Decode uses `FULL_DECODE_ONLY` graph mode and keeps FlashComm1 disabled.
GLM-5.3-Flash model currently supports only model runner V1 on Ascend, so
all A3 scripts set `VLLM_USE_V2_MODEL_RUNNER=0` explicitly.

## 3 Prerequisites

### 3.1 Model Weight

- `GLM-5.3-Flash-w8a8-mxfp8 (950DT Products mxfp8 Quantized)`: requires 1 950DT Products (96GB × 8) node.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8).
- `GLM-5.3-Flash-w8a8`: requires 1 Atlas 800 A3 (128GB × 8) node for
  single-node deployment, or 2 nodes for 1P1D PD disaggregated deployment.
  [Download model weight](https://modelers.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8).
- `GLM-5.3-Flash-w8a8`: requires 2 Atlas 800 A2 (64GB × 16) nodes.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8).

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

    source /usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer/bin/set_env.bash
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
      --gpu-memory-utilization 0.85 \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32,64,96,128]}' \
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
- `--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'`: Enables Multi-Token Prediction (MTP) speculative decoding with the DeepSeek-style MTP draft head of GLM-5.3-Flash. `num_speculative_tokens` (3-5) controls how many tokens are speculated per step; `enforce_eager: true` is required because GLM-5.3-Flash does not support graph-mode speculative decoding.

### 5.2 1P1D PD Disaggregated Deployment

=== "Atlas 800 A3 series"

    This example uses two Atlas 800 A3 servers. The Prefill node runs two
    DP ranks with TP8 (DP2/TP8), and the Decode node runs sixteen DP ranks
    with TP1 (DP16/TP1). Both layouts consume all 16 logical devices on their
    respective servers. `MooncakeConnectorV2` transfers KV cache from the
    Prefill engines to the Decode engines.

    Before starting the services, replace `LOCAL_IP`, `NIC_NAME`, and
    `MODEL_PATH` in the following scripts with values for the deployment
    environment.

    #### 5.2.1 Prepare the DP Launcher

    Save the following script as `launch_online_dp.py` on both nodes. It
    divides the node's visible devices among local DP ranks and starts one
    vLLM process per rank. The same launcher is used with the role-specific
    Prefill and Decode templates.

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
                for device in range(
                    i * args.tp_size,
                    (i + 1) * args.tp_size,
                )
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

    The launcher arguments are:

    | Parameter | Description |
    |-----------|-------------|
    | `--template` | Role-specific serving script. Use `run_p.sh` on Prefill and `run_d.sh` on Decode. |
    | `--dp-size` | Global DP size within the Prefill or Decode node group. |
    | `--tp-size` | Number of logical devices used by each DP rank. |
    | `--dp-size-local` | Number of DP ranks started on the current node. The default is the global DP size. |
    | `--dp-rank-start` | First DP rank assigned to this node. The default is `0`. |
    | `--dp-address` | IP address that coordinates the corresponding DP group. Use the Prefill IP on Prefill and the Decode IP on Decode. |
    | `--dp-rpc-port` | DP coordination port. It must be unused and reachable within the node group. |
    | `--vllm-start-port` | First API port; the launcher increments it for each local DP rank. |

    #### 5.2.2 Start the Prefill Node

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

    #### 5.2.3 Start the Decode Node

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

    #### 5.2.4 Deploy the PD Proxy

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

    #### 5.2.5 Key Parameter Descriptions

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
        "max_completion_tokens": 50
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

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

### 7.2 Using Language Model Evaluation Harness

Not tested yet.

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
