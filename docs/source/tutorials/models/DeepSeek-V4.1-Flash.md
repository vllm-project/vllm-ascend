# DeepSeek-V4.1-Flash

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM Ascend is recommended.**

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
is a native multimodal Mixture-of-Experts (MoE) model with 552B backbone
parameters and a context length of up to one million tokens. It uses a
40-layer Causal Encoder-Decoder (CED) architecture with 20 causal-encoder
layers and 20 decoder layers, activating 8B parameters per token during
Prefill and 16B during Decode.

The model introduces Compressed Sparse Attention 2 (CSA2), FP4 main KV cache,
SWA Bounded Replay, Single-Pass mHC, Engram conditional memory, and DSpark
speculative decoding. These designs reduce the global KV cache footprint to
890 bytes per token and the persistent KV cache footprint to approximately
one eighth of DeepSeek-V4-Flash. The model accepts text and images and supports
a continuously adjustable reasoning effort from 1 to 100.

vLLM Ascend supports W4A8 and W8A8 deployment on A2 products, A3 products,
and 950DT products. This guide provides a single-node colocated A3
configuration and a two-node A3 Prefill-Decode (PD) disaggregated
configuration. It also supports a two-node hybrid A2 configuration and a
four-node Prefill-Decode (PD) disaggregated configuration.

DeepSeek-V4.1-Flash is supported in **vLLM Ascend v0.30.0rc1 and later**.
This guide uses the deployment configurations for **vLLM Ascend v0.32.0**
and recommends W4A8 for performance.
To use the latest features, use the latest supported release candidate or
official version of vLLM Ascend.

Use the matching upstream vLLM revision for the selected vLLM Ascend version.
For source builds, use the upstream vLLM tag recorded in
`.github/vllm-release-tag.commit` at the selected vLLM Ascend release tag.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md)
for the complete support matrix and the
[Feature Guide](../../user_guide/feature_guide/index.md) for feature
configuration.

The A3 configurations in this guide use W4A8 weights, INT8 Engram storage,
and model runner V2. The single-node colocated configuration uses DP4/TP4.
The PD configuration uses DP4/PCP4/TP1 on the Prefill node, DP16/TP1 on the
Decode node, SWA bounded replay and MegaMoE on Prefill, DSpark speculative
decoding, and `FULL_DECODE_ONLY` ACL Graph on Decode.

## 3 Prerequisites

### 3.1 Model Weight

The official DeepSeek-V4.1-Flash-1017 checkpoint is available from
[Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) and
[ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017).

Download the quantized checkpoint from ModelScope for the hardware shown below.

| Checkpoint | Hardware | Deployment | Download | Notes |
| --- | --- | --- | --- | --- |
| `DeepSeek-V4.1-Flash-1017-w4a8` | A3 series (64GB × 16) | 1 node for single-node deployment; 2 nodes for PD | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8) | Recommended for performance. Includes DSpark draft parameters and INT8 Engram tables. |
| `DeepSeek-V4.1-Flash-1017-w8a8` | A3 series (64GB × 16) | 1 node for single-node deployment; 2 nodes for PD | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8) | Also supported. For PD, use Prefill DP4/TP4 and Decode DP8/TP2. |
| `DeepSeek-V4.1-Flash-1017-w4a8` | A2 series (64GB × 8 per node) | 2 nodes for colocated deployment; 4 nodes for PD | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8) | / |
| `DeepSeek-V4.1-Flash-1017-w8a8` | A2 series (64GB × 8 per node) | 2 nodes for colocated deployment; 4 nodes for PD | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8) | / |

Alternatively, use [ModelSlim](https://github.com/Ascend/msmodelslim) to
prepare a ModelSlim-compatible W8A8 checkpoint from the official weights.

Download the selected checkpoint to the same absolute path on every server,
for example `/data/weights/DeepSeek-V4.1-Flash-1017-w4a8` or
`/data/weights/DeepSeek-V4.1-Flash-1017-w8a8`. Set `MODEL_PATH` in each serving
command to that checkpoint directory. The examples below use W4A8; to use
W8A8, set `MODEL_PATH` to the W8A8 directory instead. For A3 PD deployment
with W8A8 weights, use Prefill DP4/TP4 and Decode DP8/TP2, as described in
Section 5.2.

### 3.2 Verify Multi-node Communication

Before deployment, follow
[Verify Multi-node Communication](../../getting_started/installation.md#installation-multi-node-interconnect).
All servers must be able to communicate through the selected network
interfaces, and the service ports must not be blocked.

## 4 Installation

### 4.1 Docker Image Installation

Select the tab for the target hardware. A2 products and A3 products use
separate images.

=== "A3 series"

    An A3 server exposes 16 logical devices. Run this command on every A3
    server.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    export MODEL_ROOT="/data/weights"

    docker pull "$IMAGE"

    docker run --rm -it \
      --name deepseek-v41 \
      --net=host \
      --ipc=host \
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
      -v "$MODEL_ROOT:$MODEL_ROOT" \
      "$IMAGE" bash
    ```

=== "A2 series"

    An A2 server exposes 8 devices. Run this command on all four A2 servers.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    export MODEL_ROOT="/data/weights"

    docker pull "$IMAGE"

    docker run --rm -it \
      --name deepseek-v41 \
      --net=host \
      --shm-size=512g \
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
      -v "$MODEL_ROOT:$MODEL_ROOT" \
      "$IMAGE" bash
    ```

Change `MODEL_ROOT` if the checkpoint is stored elsewhere. Keep the same
absolute path inside and outside every container.

### 4.2 Source Code Installation

To build from source, follow the
[software environment installation guide](../../getting_started/installation.md#installation-software-environment)
and check out the **v0.32.0rc** tag of vLLM Ascend. Use the matching upstream
vLLM tag recorded in `.github/vllm-release-tag.commit` at that tag.

## 5 Online Service Deployment

### 5.1 Single-Node Colocated Deployment

=== "A3 series"

    This configuration runs Prefill and Decode on one Atlas 800 A3 server. It
    uses DP4/TP4 across all 16 logical devices, expert parallelism, Engram host
    offload, asynchronous scheduling, and `FULL_DECODE_ONLY` ACL Graph.

    Set `MODEL_PATH` to the local `DeepSeek-V4.1-Flash-1017-w4a8` checkpoint
    directory described in Section 3.1. The `--quantization ascend` option
    loads the W4A8 quantization configuration from this checkpoint.

    ```shell
    #!/usr/bin/env bash

    MODEL_PATH="<YOUR_MODEL_PATH>"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=36000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE="AIV"

    # Ensure the model path matches the directory recorded during download
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8900 \
        --max-model-len 150000 \
        --max-num-batched-tokens 8192 \
        --served-model-name dsv41 \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 32 \
        --data-parallel-size 4 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --tokenizer-mode deepseek_v41 \
        --reasoning-parser deepseek_v41 \
        --tool-call-parser deepseek_v41 \
        --enable-auto-tool-choice \
        --safetensors-load-strategy lazy \
        --quantization ascend \
        --block-size 128 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --async-scheduling \
        --swa-bounded-replay \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --additional-config '{
            "ascend_compilation_config":{
                "enable_npugraph_ex":true,
                "enable_static_kernel":false
            },
            "enable_cpu_binding":true,
            "enable_fused_mc2":1,
            "enable_flashcomm1":false,
            "enable_shared_expert_dp":true
        }'
    ```

    Key parameters:

    - `--max-model-len 150000` limits the total input and output length of one
        request. `--max-num-batched-tokens 8192` limits the tokens scheduled in one
        iteration, while `--max-num-seqs 32` limits the sequences scheduled by each
        DP engine. Increasing either scheduler limit can improve throughput but also
        increases memory usage.
    - `VLLM_ENGINE_READY_TIMEOUT_S=36000` allows up to 36,000 seconds for engine
        processes to finish initialization, including weight loading and graph
        preparation.
    - `VLLM_USE_V2_MODEL_RUNNER=1` explicitly selects model runner V2.
    - `--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'` keeps the
        Engram table in host memory and lets local DP ranks share the host-memory
        allocation. This is required on A3 because the model's 24 Engram heads
        cannot be evenly divided across 16 logical devices. It also reduces
        duplicate host-memory copies. The A3 container example uses `--ipc=host` so
        all four local DP ranks share the host IPC namespace.
    - `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` captures the
        Decode path while Prefill remains outside the captured graph.
    - `--safetensors-load-strategy lazy` avoids eagerly materializing the whole
        checkpoint during loading.
    - `--swa-bounded-replay` enables SWA bounded replay.
    - `enable_flashcomm1: false` disables FlashComm for compatibility with SWA bounded replay.
    - `enable_fused_mc2` and `enable_shared_expert_dp` enable the A3 MoE and
        communication optimizations
        used by this configuration. `enable_cpu_binding` pins worker processes to
        CPUs, while `enable_npugraph_ex` enables the enhanced ACL Graph path.

### 5.2 1P1D PD Disaggregated Deployment

=== "A3 series"

    This example uses W4A8 weights and model runner V2 on two Atlas 800 A3
    servers. The Prefill node runs DP4/PCP4/TP1, and the Decode node runs
    DP16/TP1. Both layouts consume all 16 logical devices on their respective
    servers. Mooncake transfers KV cache from the Prefill engines to the Decode
    engines.

    For W8A8 weights, use Prefill DP4/TP4 and Decode DP8/TP2. Adjust the
    launcher, connector layouts, and proxy endpoints to match that topology.

    #### 5.2.1.1 Prepare the DP Launcher

    Save the following script as `launch_online_dp.py` on both nodes. It divides
    the node's visible devices among local DP ranks and starts one vLLM process per
    rank.

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
        parser.add_argument("--pcp-size", type=int, default=1)
        parser.add_argument("--dp-size-local", type=int, default=-1)
        parser.add_argument("--dp-rank-start", type=int, default=0)
        parser.add_argument("--dp-address", type=str, required=True)
        parser.add_argument("--dp-rpc-port", type=str, default="12345")
        parser.add_argument("--vllm-start-port", type=int, default=9000)
        return parser.parse_args()


    args = parse_args()
    dp_size = args.dp_size
    tp_size = args.tp_size
    pcp_size = args.pcp_size
    dp_size_local = args.dp_size if args.dp_size_local == -1 else args.dp_size_local


    def run_command(visible_devices, dp_rank, vllm_engine_port):
        command = [
            "bash",
            "./run_dp_template.sh",
            visible_devices,
            str(vllm_engine_port),
            str(dp_size),
            str(dp_rank),
            args.dp_address,
            args.dp_rpc_port,
            str(tp_size),
            str(pcp_size),
        ]
        subprocess.run(command, check=True)


    if __name__ == "__main__":
        if not os.path.exists("./run_dp_template.sh"):
            print("Template file ./run_dp_template.sh does not exist.")
            sys.exit(1)

        devices_per_rank = tp_size * pcp_size
        processes = []
        for i in range(dp_size_local):
            dp_rank = args.dp_rank_start + i
            vllm_engine_port = args.vllm_start_port + i
            visible_devices = ",".join(
                str(device)
                for device in range(i * devices_per_rank, (i + 1) * devices_per_rank)
            )
            process = multiprocessing.Process(
                target=run_command,
                args=(visible_devices, dp_rank, vllm_engine_port),
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
    ```

    The launcher arguments are:

    | Parameter | Description |
    |-----------|-------------|
    | `--dp-size` | Global DP size within the Prefill or Decode node group. |
    | `--tp-size` | Tensor-parallel size within each DP rank. |
    | `--pcp-size` | Prefill context-parallel size; each DP rank uses `tp-size * pcp-size` logical devices. Defaults to `1`. |
    | `--dp-size-local` | Number of DP ranks started on the current node. |
    | `--dp-rank-start` | First DP rank assigned to this node. It is `0` for both single-node groups in this 1P1D example. |
    | `--dp-address` | IP address of the node that coordinates the corresponding DP group. Use the Prefill IP on the Prefill node and the Decode IP on the Decode node. |
    | `--dp-rpc-port` | DP coordination port. It must be unused and reachable within the node group. |
    | `--vllm-start-port` | First API port; the launcher increments it for each local DP rank. |

    The launcher passes eight positional arguments to `run_dp_template.sh`:

    | Position | Value | Used by the template |
    |----------|-------|----------------------|
    | `$1` | `visible_devices` | `ASCEND_RT_VISIBLE_DEVICES` |
    | `$2` | `vllm_engine_port` | `--port` |
    | `$3` | `dp_size` | `--data-parallel-size` |
    | `$4` | `dp_rank` | `--data-parallel-rank` |
    | `$5` | `args.dp_address` | `--data-parallel-address` |
    | `$6` | `args.dp_rpc_port` | `--data-parallel-rpc-port` |
    | `$7` | `tp_size` | `--tensor-parallel-size` |
    | `$8` | `pcp_size` | `--prefill-context-parallel-size` |

    #### 5.2.1.2 Start the Prefill Node

    On the Prefill node, save the following script as `run_dp_template.sh`. Set
    `LOCAL_IP`, `NIC_NAME`, and `MODEL_PATH` to the Prefill node's service IP,
    network interface, and local checkpoint path.

    ```shell
    #!/usr/bin/env bash

    unset https_proxy
    unset http_proxy
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"

    NIC_NAME="<NETWORK_INTERFACE>"
    LOCAL_IP="<PREFILL_NODE_IP>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=36000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE="AIV"
    export ASCEND_RT_VISIBLE_DEVICES=$1

    # Ensure the model path matches the directory recorded during download
    vllm serve "$MODEL_PATH" \
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
        --served-model-name dsv41 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 16 \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
        --trust-remote-code \
        --block-size 128 \
        --tokenizer-mode deepseek_v41 \
        --reasoning-parser deepseek_v41 \
        --tool-call-parser deepseek_v41 \
        --enable-auto-tool-choice \
        --safetensors-load-strategy lazy \
        --gpu-memory-utilization 0.9 \
        --quantization ascend \
        --enforce-eager \
        --swa-bounded-replay \
        --enable-prefix-caching \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --additional-config '{
            "enable_cpu_binding":true,
            "enable_fused_mc2":2,
            "enable_flashcomm1":false,
            "multistream_overlap_shared_expert":false
        }' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeHybridConnector",
            "kv_role":"kv_producer",
            "kv_port":"30000",
            "kv_connector_extra_config":{
                "prefill":{"dp_size":4,"tp_size":1},
                "decode":{"dp_size":16,"tp_size":1}
            }
        }'
    ```

    Start four DP4/PCP4/TP1 Prefill engines. `--dp-address` uses the Prefill node IP.

    ```shell
    PREFILL_NODE_IP="<PREFILL_NODE_IP>"

    python launch_online_dp.py \
        --dp-size 4 \
        --tp-size 1 \
        --pcp-size 4 \
        --dp-size-local 4 \
        --dp-rank-start 0 \
        --dp-address "$PREFILL_NODE_IP" \
        --dp-rpc-port 12321 \
        --vllm-start-port 7100
    ```

    #### 5.2.1.3 Start the Decode Node

    On the Decode node, save the following script as `run_dp_template.sh`. Set
    `LOCAL_IP`, `NIC_NAME`, and `MODEL_PATH` to the Decode node's service IP,
    network interface, and the same local checkpoint path used on the Prefill
    node.

    ```shell
    #!/usr/bin/env bash

    unset https_proxy
    unset http_proxy
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"

    NIC_NAME="<NETWORK_INTERFACE>"
    LOCAL_IP="<DECODE_NODE_IP>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=36000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_BUFFSIZE=4096
    export ASCEND_RT_VISIBLE_DEVICES=$1

    # Ensure the model path matches the directory recorded during download
    vllm serve "$MODEL_PATH" \
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
        --served-model-name dsv41 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 512 \
        --max-num-seqs 72 \
        --async-scheduling \
        --block-size 128 \
        --no-enable-prefix-caching \
        --no-swa-bounded-replay \
        --trust-remote-code \
        --tokenizer-mode deepseek_v41 \
        --reasoning-parser deepseek_v41 \
        --tool-call-parser deepseek_v41 \
        --enable-auto-tool-choice \
        --safetensors-load-strategy lazy \
        --gpu-memory-utilization 0.95 \
        --quantization ascend \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeHybridConnector",
            "kv_role":"kv_consumer",
            "kv_port":"30100",
            "kv_connector_extra_config":{
                "prefill":{"dp_size":4,"tp_size":1},
                "decode":{"dp_size":16,"tp_size":1}
            }
        }' \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --additional-config '{
            "ascend_compilation_config":{
                "enable_npugraph_ex":true,
                "enable_static_kernel":false
            },
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "recompute_scheduler_enable":true
        }'
    ```

    Start sixteen DP16/TP1 Decode engines. `--dp-address` uses the Decode node IP.

    ```shell
    DECODE_NODE_IP="<DECODE_NODE_IP>"

    python launch_online_dp.py \
        --dp-size 16 \
        --tp-size 1 \
        --pcp-size 1 \
        --dp-size-local 16 \
        --dp-rank-start 0 \
        --dp-address "$DECODE_NODE_IP" \
        --dp-rpc-port 12321 \
        --vllm-start-port 7100
    ```

    #### 5.2.1.4 Deploy the PD Proxy

    After all Prefill and Decode engines are ready, deploy the proxy as described
    in [Prefill-Decode Disaggregation (DeepSeek)](../features/pd_disaggregation_mooncake_multi_node.md).
    Configure the proxy with Prefill endpoints `<PREFILL_NODE_IP>:7100` through
    `<PREFILL_NODE_IP>:7103` and Decode endpoints `<DECODE_NODE_IP>:7100` through
    `<DECODE_NODE_IP>:7115`.

    #### 5.2.1.5 Key Parameter Descriptions

    - `VLLM_ENGINE_READY_TIMEOUT_S=36000` gives every Prefill and Decode engine up
      to 36,000 seconds to finish startup. This includes weight loading and Decode
      graph preparation.
    - `VLLM_USE_V2_MODEL_RUNNER=1` selects model runner V2 on both roles.
    - `--data-parallel-size`, `--prefill-context-parallel-size`, and
      `--tensor-parallel-size` define DP4/PCP4/TP1 on Prefill and DP16/PCP1/TP1
      on Decode. Their product is 16 on each node. The launcher assigns four
      devices to each Prefill rank and one device to each Decode rank.
    - `--swa-bounded-replay` enables SWA bounded replay on Prefill together with
      PCP4; Prefill still uses `--enforce-eager`.
    - `--data-parallel-address` and `--data-parallel-rpc-port` coordinate DP ranks
      within one role. Prefill and Decode use their respective node IPs; port
      `12321` can be reused because the roles run on different hosts.
    - `--vllm-start-port 7100` assigns API ports `7100-7103` on Prefill and
      `7100-7115` on Decode. These endpoints must be reachable by the PD proxy.
    - Both roles use `--max-model-len 1048576`. Prefill uses
      `--max-num-batched-tokens 8192` and `--max-num-seqs 16` to favor prompt
      throughput, while Decode uses `512` and `72` respectively to favor decode
      concurrency. Tune these role-specific scheduler limits independently.
    - `MooncakeHybridConnector` transfers KV cache between the two roles.
      `kv_role` must be `kv_producer` on Prefill and `kv_consumer` on Decode.
      `kv_port` must be reachable and must not conflict with another service.
    - `kv_connector_extra_config` must match the actual global layouts on both
      sides: Prefill DP4/TP1 and Decode DP16/TP1. PCP4 is configured separately
      with `--prefill-context-parallel-size`; it does not change `tp_size` in the
      connector configuration. Keep both connector layouts identical in the
      Prefill and Decode commands.
    - Both roles use DSpark with five speculative tokens. Prefill runs the target
      model in eager mode. Decode uses `FULL_DECODE_ONLY` for the target model
      and allows graph execution for the DSpark draft model as well.
    - `--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'` keeps the
      Engram table in host memory and lets local DP ranks share the host-memory
      allocation. `dp_shared_memory` is required on A3 because the model's 24
      Engram heads cannot be evenly divided across 16 logical devices. Both A3
      containers use `--ipc=host` so the local DP ranks can share the allocation.
    - `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` captures the
      Decode path while Prefill remains outside the captured graph.
    - Prefix caching is enabled only on Prefill. Decode disables it and enables
      `recompute_scheduler_enable` so Prefill can recompute unavailable KV cache.
    - Prefill uses the following `--additional-config` options:

        - `enable_cpu_binding: true` enables Ascend-native CPU affinity for worker
        processes and runtime threads.
        - `enable_fused_mc2: 2` selects the MegaMoE path on Prefill.
        - `enable_flashcomm1: false` disables FlashComm for compatibility with SWA bounded replay.
        - `multistream_overlap_shared_expert: false` disables shared-expert
        multistream overlap on Prefill, where MegaMoE is enabled.

    - Decode uses the following `--additional-config` options:

        - `ascend_compilation_config.enable_npugraph_ex: true` enables the enhanced
        ACL Graph compilation and execution path used by `FULL_DECODE_ONLY`.
        - `ascend_compilation_config.enable_static_kernel: false` disables static
        kernel generation while retaining the enhanced ACL Graph path.
        - `enable_cpu_binding: true` enables Ascend-native CPU affinity for worker
        processes and runtime threads.
        - `multistream_overlap_shared_expert: true` enables shared-expert multistream
        overlap on Decode, where MegaMoE is not enabled. Shared-expert execution
        overlaps routed-expert computation on separate streams.
        - `recompute_scheduler_enable: true` enables the PD Decode recomputation
        scheduler. When the Decode-side KV cache is unavailable, the request can be
        sent back to Prefill to recompute it. Enable this option only on Decode.

      The Prefill and Decode configurations are role-specific. Do not copy the
      Prefill MegaMoE and PCP options to Decode, or the Decode graph, multistream,
      and recomputation options to Prefill, without validating the resulting
      topology and performance.

      `HCCL_IF_IP` and the socket interface variables must select the same high-speed
      service network used by the configured node IPs. `HCCL_BUFFSIZE` is `1024` on
      Prefill and `4096` on Decode; retain these role-specific values unless another
      setting has been validated. The DP RPC, engine, Mooncake, and proxy ports must
      be allowed by the host firewall.
      Wait until every engine finishes loading weights and Decode finishes graph
      capture. A successful startup includes output similar to:

      ```output
      INFO:     Started server process
      INFO:     Waiting for application startup.
      INFO:     Application startup complete.
      ```

=== "A2 series"

    This example uses four Atlas 800 A2 servers. The Prefill role runs four DP
    ranks with TP4 (DP4/TP4) across two servers: the first Prefill server hosts
    DP ranks 0-1 and the second Prefill server hosts DP ranks 2-3. The Decode
    role runs eight DP ranks with TP2 (DP8/TP2) across the other two servers:
    the first Decode server hosts DP ranks 0-3 and the second Decode server
    hosts DP ranks 4-7. Each layout consumes all 8 logical devices on its
    server. Mooncake transfers KV cache from the Prefill engines to the Decode
    engines.

    Example IP mapping used below:

    | Role | Server | Local DP ranks | Devices per rank | Engine ports |
    |------|--------|----------------|------------------|--------------|
    | Prefill | `<PREFILL_NODE_1_IP>` | 0, 1 | 4 | 7100, 7101 |
    | Prefill | `<PREFILL_NODE_2_IP>` | 2, 3 | 4 | 7100, 7101 |
    | Decode | `<DECODE_NODE_1_IP>` | 0-3 | 2 | 7100-7103 |
    | Decode | `<DECODE_NODE_2_IP>` | 4-7 | 2 | 7100-7103 |

    #### 5.2.2.1 Prepare the DP Launcher

    Save the following script as `launch_online_dp.py` on all four nodes. It
    divides the node's visible devices among local DP ranks and starts one
    vLLM process per rank. The script is identical to the A3 example.

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
        parser.add_argument("--dp-size-local", type=int, default=-1)
        parser.add_argument("--dp-rank-start", type=int, default=0)
        parser.add_argument("--dp-address", type=str, required=True)
        parser.add_argument("--dp-rpc-port", type=str, default="12345")
        parser.add_argument("--vllm-start-port", type=int, default=9000)
        return parser.parse_args()

    args = parse_args()
    dp_size = args.dp_size
    tp_size = args.tp_size
    dp_size_local = args.dp_size if args.dp_size_local == -1 else args.dp_size_local

    def run_command(visible_devices, dp_rank, vllm_engine_port):
        command = [
            "bash",
            "./run_dp_template.sh",
            visible_devices,
            str(vllm_engine_port),
            str(dp_size),
            str(dp_rank),
            args.dp_address,
            args.dp_rpc_port,
            str(tp_size),
        ]
        subprocess.run(command, check=True)


    if __name__ == "__main__":
        if not os.path.exists("./run_dp_template.sh"):
            print("Template file ./run_dp_template.sh does not exist.")
            sys.exit(1)

        processes = []
        for i in range(dp_size_local):
            dp_rank = args.dp_rank_start + i
            vllm_engine_port = args.vllm_start_port + i
            visible_devices = ",".join(
                str(device) for device in range(i * tp_size, (i + 1) * tp_size)
            )
            process = multiprocessing.Process(
                target=run_command,
                args=(visible_devices, dp_rank, vllm_engine_port),
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
    ```

    The launcher arguments are:

    | Parameter | Description |
    |-----------|-------------|
    | `--dp-size` | Global DP size within the Prefill or Decode node group. |
    | `--tp-size` | Number of logical devices used by each DP rank. |
    | `--dp-size-local` | Number of DP ranks started on the current node. |
    | `--dp-rank-start` | First DP rank assigned to this node. Prefill servers use `0` and `2`; Decode servers use `0` and `4`. |
    | `--dp-address` | IP address of the node that coordinates the corresponding DP group. Use the first Prefill IP on both Prefill nodes and the first Decode IP on both Decode nodes. |
    | `--dp-rpc-port` | DP coordination port. It must be unused and reachable within the node group. |
    | `--vllm-start-port` | First API port; the launcher increments it for each local DP rank. |

    #### 5.2.2.2 Start the Prefill Nodes

    On each Prefill node, save the following script as `run_dp_template.sh`.
    Set `LOCAL_IP`, `NIC_NAME`, and `MODEL_PATH` to the local server's service
    IP, network interface, and checkpoint path.

    ```shell
    #!/usr/bin/env bash
    set -euo pipefail
    unset https_proxy
    unset http_proxy
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"

    NIC_NAME="<NETWORK_INTERFACE>"
    LOCAL_IP="<LOCAL_PREFILL_NODE_IP>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    export no_proxy="127.0.0.1,localhost,<PREFILL_NODE_1_IP>,<PREFILL_NODE_2_IP>,<DECODE_NODE_1_IP>,<DECODE_NODE_2_IP>"
    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_USE_V2_MODEL_RUNNER=0
    export VLLM_ENGINE_READY_TIMEOUT_S=360000
    export HCCL_BUFFSIZE=1024
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
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
        --served-model-name dsv41 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 4096 \
        --max-num-seqs 16 \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
        --trust-remote-code \
        --block-size 128 \
        --tokenizer-mode deepseek_v41 \
        --reasoning-parser deepseek_v41 \
        --tool-call-parser deepseek_v41 \
        --enable-auto-tool-choice \
        --safetensors-load-strategy lazy \
        --gpu-memory-utilization 0.9 \
        --quantization ascend \
        --enforce-eager \
        --enable-prefix-caching \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --additional-config '{
            "enable_cpu_binding":true,
            "enable_dsa_cp":true,
            "enable_flashcomm1":true,
            "enable_shared_expert_dp":true
        }' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeHybridConnector",
            "kv_role":"kv_producer",
            "kv_port":"30000",
            "engine_id":"0",
            "kv_connector_extra_config":{
                "prefill":{"dp_size":4,"tp_size":4},
                "decode":{"dp_size":8,"tp_size":2}
            }
        }'
    ```

    Start two DP4/TP4 Prefill engines on each Prefill node. On the first
    Prefill node (DP ranks 0-1), `--dp-address` uses that node's IP:

    ```shell
    PREFILL_NODE_1_IP="<PREFILL_NODE_1_IP>"

    python launch_online_dp.py \
        --dp-size 4 \
        --tp-size 4 \
        --dp-size-local 2 \
        --dp-rank-start 0 \
        --dp-address "$PREFILL_NODE_1_IP" \
        --dp-rpc-port 12321 \
        --vllm-start-port 7100
    ```

    On the second Prefill node (DP ranks 2-3), `--dp-address` still points to
    the first Prefill node:

    ```shell
    PREFILL_NODE_1_IP="<PREFILL_NODE_1_IP>"

    python launch_online_dp.py \
        --dp-size 4 \
        --tp-size 4 \
        --dp-size-local 2 \
        --dp-rank-start 2 \
        --dp-address "$PREFILL_NODE_1_IP" \
        --dp-rpc-port 12321 \
        --vllm-start-port 7100
    ```

    #### 5.2.2.3 Start the Decode Nodes

    On each Decode node, save the following script as `run_dp_template.sh`.
    Set `LOCAL_IP`, `NIC_NAME`, and `MODEL_PATH` to the local server's service
    IP, network interface, and the same checkpoint path used on the Prefill
    nodes.

    ```shell
    #!/usr/bin/env bash
    set -euo pipefail
    unset https_proxy
    unset http_proxy
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"

    NIC_NAME="<NETWORK_INTERFACE>"
    LOCAL_IP="<LOCAL_DECODE_NODE_IP>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    export no_proxy="127.0.0.1,localhost,<PREFILL_NODE_1_IP>,<PREFILL_NODE_2_IP>,<DECODE_NODE_1_IP>,<DECODE_NODE_2_IP>"
    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_USE_V2_MODEL_RUNNER=0
    export VLLM_ENGINE_READY_TIMEOUT_S=360000
    export HCCL_BUFFSIZE=1800
    export HCCL_CONNECT_TIMEOUT=1200
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
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
        --served-model-name dsv41 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 400 \
        --max-num-seqs 32 \
        --async-scheduling \
        --block-size 128 \
        --no-enable-prefix-caching \
        --trust-remote-code \
        --tokenizer-mode deepseek_v41 \
        --reasoning-parser deepseek_v41 \
        --tool-call-parser deepseek_v41 \
        --enable-auto-tool-choice \
        --safetensors-load-strategy lazy \
        --gpu-memory-utilization 0.95 \
        --quantization ascend \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeHybridConnector",
            "kv_role":"kv_consumer",
            "kv_port":"30100",
            "engine_id":"1",
            "kv_connector_extra_config":{
                "prefill":{"dp_size":4,"tp_size":4},
                "decode":{"dp_size":8,"tp_size":2}
            }
        }' \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --additional-config '{
            "ascend_compilation_config":{
                "enable_npugraph_ex":true,
                "enable_static_kernel":false
            },
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "recompute_scheduler_enable":true
        }'
    ```

    Start four DP8/TP2 Decode engines on each Decode node. On the first Decode
    node (DP ranks 0-3), `--dp-address` uses that node's IP:

    ```shell
    DECODE_NODE_1_IP="<DECODE_NODE_1_IP>"

    python launch_online_dp.py \
        --dp-size 8 \
        --tp-size 2 \
        --dp-size-local 4 \
        --dp-rank-start 0 \
        --dp-address "$DECODE_NODE_1_IP" \
        --dp-rpc-port 12321 \
        --vllm-start-port 7100
    ```

    On the second Decode node (DP ranks 4-7), `--dp-address` still points to
    the first Decode node:

    ```shell
    DECODE_NODE_1_IP="<DECODE_NODE_1_IP>"

    python launch_online_dp.py \
        --dp-size 8 \
        --tp-size 2 \
        --dp-size-local 4 \
        --dp-rank-start 4 \
        --dp-address "$DECODE_NODE_1_IP" \
        --dp-rpc-port 12321 \
        --vllm-start-port 7100
    ```

    #### 5.2.2.4 Deploy the PD Proxy

    After all twelve engines are ready (four Prefill and eight Decode), deploy
    the proxy as described in
    [Prefill-Decode Disaggregation (DeepSeek)](../features/pd_disaggregation_mooncake_multi_node.md).
    Configure the proxy with Prefill endpoints `<PREFILL_NODE_1_IP>:7100` and
    `<PREFILL_NODE_1_IP>:7101`, `<PREFILL_NODE_2_IP>:7100` and
    `<PREFILL_NODE_2_IP>:7101`, and Decode endpoints
    `<DECODE_NODE_1_IP>:7100` through `<DECODE_NODE_1_IP>:7103`,
    `<DECODE_NODE_2_IP>:7100` through `<DECODE_NODE_2_IP>:7103`. For example,
    run the load-balancing proxy on the first Prefill node:

    ```shell
    cd <vllm_ascend_repo>/examples/disaggregated_prefill_v1

    python3 load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 2999 \
        --prefiller-hosts <PREFILL_NODE_1_IP> <PREFILL_NODE_1_IP> <PREFILL_NODE_2_IP> <PREFILL_NODE_2_IP> \
        --prefiller-ports 7100 7101 7100 7101 \
        --decoder-hosts <DECODE_NODE_1_IP> <DECODE_NODE_1_IP> <DECODE_NODE_1_IP> <DECODE_NODE_1_IP> <DECODE_NODE_2_IP> <DECODE_NODE_2_IP> <DECODE_NODE_2_IP> <DECODE_NODE_2_IP> \
        --decoder-ports 7100 7101 7102 7103 7100 7101 7102 7103
    ```

    #### 5.2.2.5 Key Parameter Descriptions

    - The A2 layout is 1P1D: the Prefill role spans two servers (DP4/TP4
      ranks each) and the Decode role spans two servers (DP8/TP2 ranks
      each). On every A2 server the local DP ranks multiplied by TP must
      consume all 8 devices: 2 × 4 on each Prefill server and 4 × 2 on each
      Decode server.
    - `VLLM_ENGINE_READY_TIMEOUT_S=360000` gives every Prefill and Decode
      engine up to 100 hours to finish startup. Weights are loaded from shared
      storage, which is slower on A2; this validated value is intentionally
      larger than the A3 value.
    - Both roles use `--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'`.
      The Engram table is kept in host memory and shared by the local DP
      ranks of each server. With TP4 and TP2 the model's 24 Engram heads are
      divided evenly (6 and 12 heads per rank), but the host-memory table must
      still be shared across the node's local DP ranks, which requires the
      container to run with `--ipc=host` or a `/dev/shm` of at least 320g.
    - Prefix caching is enabled only on Prefill. Decode disables it and
      enables `recompute_scheduler_enable` so requests can be recomputed when
      the Prefill-side KV cache is unavailable.
    - Prefill and Decode use the same --additional-config options as A3.
    - `HCCL_IF_IP` and the socket interface variables must select the same
      high-speed service network used by the configured node IPs.
      `HCCL_INTRA_ROCE_ENABLE=1` enables RoCE transport for intra-node HCCL
      on A2. `HCCL_BUFFSIZE` is `1024` on Prefill and `1800` on Decode;
      retain these role-specific values unless another setting has been
      validated. `HCCL_OP_EXPANSION_MODE="AIV"` is set on Prefill only.
    Wait until every engine finishes loading weights and Decode finishes graph
    capture. A successful startup includes output similar to:

    ```output
    INFO:     Started server process
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    ```

### 5.3 1P1D Deployment with Memcache KV Cache Pool

This configuration extends the A3 1P1D deployment in [Section 5.2](#52-1p1d-pd-disaggregated-deployment) with a
Memcache KV Cache Pool. `MooncakeHybridConnector` continues to transfer KV
cache from Prefill to Decode, while `AscendStoreConnector` stores reusable KV
cache in Memcache. Both connectors are configured as children of
`MultiConnector`.

DeepSeek-V4.1-Flash does not support layerwise KV Cache Pool access because its
CED hybrid-attention model structure is incompatible with layer-by-layer KV
save and load. Keep `use_layerwise` set to `false` on both Prefill and Decode.

Before starting vLLM, follow the
[Memcache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend)
to install `memfabric-hybrid` and `memcache-hybrid`, configure
`meta.conf`, `mmc-local.conf`, and `mmc-local-standalone.conf`, and start the
Memcache MetaService. The
Prefill and Decode nodes must use the same MetaService endpoint. Keep the
`/etc/hccn.conf` container mount from [Section 4.1](#41-docker-image-installation).

This example uses separated Memcache deployment so that the pool reserves host
memory before vLLM loads the model. Prepare one MetaService configuration and
two LocalService configurations as described below.

#### 5.3.1 Prepare the Memcache Services

Create `meta.conf` on the node that runs MetaService:

```ini
ock.mmc.log_level = error
ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
ock.mmc.meta_service.config_store_url = tcp://<META_SERVICE_IP>:6000
ock.mmc.meta_service.metrics_url = http://<META_SERVICE_IP>:7000
```

Create `mmc-local.conf` on both Prefill and Decode nodes for the LocalService
embedded in vLLM:

```ini
ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:6000
ock.mmc.log_level = error
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 0GB
ock.mmc.local_service.max.dram.size = 1024GB
ock.mmc.local_service.storage.enabled = false
```

Create `mmc-local-standalone.conf` on both nodes for the standalone Memcache
process. It uses the same endpoints, world size, and protocol, but contributes
the host DRAM used by the pool:

```ini
ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:6000
ock.mmc.log_level = error
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 200GB
ock.mmc.local_service.max.dram.size = 1024GB
ock.mmc.local_service.storage.enabled = false
```

`world_size` is the maximum number of LocalServices that can join the pool, not
the DP size. Changing it after LocalServices connect requires restarting
MetaService. `device_sdma` is recommended for A3 HCCS and requires LingQu
Computing Network 1.5 or later. The `200GB` standalone DRAM value is an example;
set it according to the host memory available after reserving capacity for the
operating system and vLLM. `max.dram.size` must be at least the largest
`dram.size` contributed by any LocalService.

Start MetaService with `meta.conf`:

```shell
export MMC_META_CONFIG_PATH="<MEMCACHE_CONFIG_DIR>/meta.conf"
python3 -c 'from memcache_hybrid import MetaService; MetaService.main()'
```

Save the following script as `memcache_standalone.py` on both Prefill and
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

After MetaService is ready, start the standalone Memcache process on both
Prefill and Decode nodes and wait for successful initialization:

```shell
export MMC_LOCAL_CONFIG_PATH="<MEMCACHE_CONFIG_DIR>/mmc-local-standalone.conf"
python3 memcache_standalone.py
```

For additional parameters and capacity constraints, refer to
[Memcache and vLLM separated deployment](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#Memcache-and-vLLM-Separated-Deployment).

Reuse `launch_online_dp.py` from [Section 5.2.1.1](#5211-prepare-the-dp-launcher). Set `MEMCACHE_INSTALL_PATH` to
the `Location` reported by `pip show memcache_hybrid`, and set
`MEMCACHE_CONFIG_DIR` to the absolute directory containing the configuration
files above on both nodes.

#### 5.3.2 Start the Prefill Node with Pooling

On the Prefill node, save the following script as `run_dp_template.sh`. Set
`LOCAL_IP`, `NIC_NAME`, `MODEL_PATH`, `MEMCACHE_INSTALL_PATH`, and
`MEMCACHE_CONFIG_DIR` for the local environment.

```shell
#!/usr/bin/env bash

unset https_proxy
unset http_proxy

NIC_NAME="<NETWORK_INTERFACE>"
LOCAL_IP="<PREFILL_NODE_IP>"
MODEL_PATH="<YOUR_MODEL_PATH>"
MEMCACHE_INSTALL_PATH="<MEMCACHE_INSTALL_PATH>"
MEMCACHE_CONFIG_DIR="<MEMCACHE_CONFIG_DIR>"

export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=$1

# Parameters required by the Memcache KV Cache Pool.
export PYTHONHASHSEED=0
export MMC_LOCAL_CONFIG_PATH="$MEMCACHE_CONFIG_DIR/mmc-local.conf"
export LD_LIBRARY_PATH="$MEMCACHE_INSTALL_PATH/memcache_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:/usr/local/lib/"

vllm serve "$MODEL_PATH" \
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
    --served-model-name dsv41 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
    --trust-remote-code \
    --block-size 128 \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --enforce-eager \
    --swa-bounded-replay \
    --enable-prefix-caching \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --additional-config '{
        "enable_cpu_binding":true,
        "enable_fused_mc2":2,
        "enable_flashcomm1":false,
        "multistream_overlap_shared_expert":false
    }' \
    --kv-transfer-config '{
        "kv_connector":"MultiConnector",
        "kv_role":"kv_producer",
        "kv_port":"30000",
        "engine_id":"0",
        "kv_connector_extra_config":{
            "connectors":[
                {
                    "kv_connector":"MooncakeHybridConnector",
                    "kv_role":"kv_producer",
                    "kv_port":"30000",
                    "kv_connector_extra_config":{
                        "prefill":{"dp_size":4,"tp_size":1},
                        "decode":{"dp_size":16,"tp_size":1}
                    }
                },
                {
                    "kv_connector":"AscendStoreConnector",
                    "kv_role":"kv_producer",
                    "kv_connector_extra_config":{
                        "backend":"memcache",
                        "lookup_rpc_port":"0",
                        "use_layerwise":false
                    }
                }
            ]
        }
    }'
```

Use the PCP-aware launcher from [Section 5.2.1.1](#5211-prepare-the-dp-launcher).
Start the four DP4/PCP4/TP1 Prefill engines with the launcher command from
[Section 5.2.1.2](#5212-start-the-prefill-node).

#### 5.3.3 Start the Decode Node with Pooling

On the Decode node, save the following script as `run_dp_template.sh`. Set
`LOCAL_IP`, `NIC_NAME`, `MODEL_PATH`, `MEMCACHE_INSTALL_PATH`, and
`MEMCACHE_CONFIG_DIR` for the local environment.

```shell
#!/usr/bin/env bash

unset https_proxy
unset http_proxy

NIC_NAME="<NETWORK_INTERFACE>"
LOCAL_IP="<DECODE_NODE_IP>"
MODEL_PATH="<YOUR_MODEL_PATH>"
MEMCACHE_INSTALL_PATH="<MEMCACHE_INSTALL_PATH>"
MEMCACHE_CONFIG_DIR="<MEMCACHE_CONFIG_DIR>"

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export HCCL_BUFFSIZE=4096
export ASCEND_RT_VISIBLE_DEVICES=$1

# Parameters required by the Memcache KV Cache Pool.
export PYTHONHASHSEED=0
export MMC_LOCAL_CONFIG_PATH="$MEMCACHE_CONFIG_DIR/mmc-local.conf"
export LD_LIBRARY_PATH="$MEMCACHE_INSTALL_PATH/memcache_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:/usr/local/lib/"

vllm serve "$MODEL_PATH" \
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
    --served-model-name dsv41 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 72 \
    --async-scheduling \
    --block-size 128 \
    --no-enable-prefix-caching \
    --no-swa-bounded-replay \
    --trust-remote-code \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config '{
        "kv_connector":"MultiConnector",
        "kv_role":"kv_consumer",
        "kv_port":"30100",
        "engine_id":"1",
        "kv_connector_extra_config":{
            "connectors":[
                {
                    "kv_connector":"MooncakeHybridConnector",
                    "kv_role":"kv_consumer",
                    "kv_port":"30100",
                    "kv_connector_extra_config":{
                        "prefill":{"dp_size":4,"tp_size":1},
                        "decode":{"dp_size":16,"tp_size":1}
                    }
                },
                {
                    "kv_connector":"AscendStoreConnector",
                    "kv_role":"kv_consumer",
                    "kv_connector_extra_config":{
                        "backend":"memcache",
                        "lookup_rpc_port":"0",
                        "use_layerwise":false
                    }
                }
            ]
        }
    }' \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --additional-config '{
        "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
        },
        "enable_cpu_binding":true,
        "multistream_overlap_shared_expert":true,
        "recompute_scheduler_enable":true
    }'
```

Start the sixteen DP16/TP1 Decode engines with the launcher command from
[Section 5.2.1.3](#5213-start-the-decode-node).

#### 5.3.4 Startup and Verification

Start the components in this order: Memcache MetaService, standalone Memcache
on both Prefill and Decode nodes, Decode engines, Prefill engines, and then the
PD proxy.

Wait until both standalone Memcache processes report successful initialization
before starting the Decode engines. Then wait until every Decode engine reports
`Application startup complete` before starting Prefill. After every Prefill engine is ready, start the proxy with the
same endpoint mapping described in [Section 5.2.1.4](#5214-deploy-the-pd-proxy).

Send the same request twice with a reusable prefix. Confirm that Prefill logs
show Memcache lookup, get, put, and hit activity, and that Decode logs still
show normal Prefill-to-Decode KV transfer.

Key pooling parameters:

- `MultiConnector` combines live P-to-D transfer through
  `MooncakeHybridConnector` with persistent prefix reuse through
  `AscendStoreConnector`.
- `backend: "memcache"` selects Memcache as the KV Cache Pool backend.
- `use_layerwise: false` is required for DeepSeek-V4.1-Flash. Do not enable
  layerwise pooling for this model because its CED hybrid-attention structure
  is not supported by the layerwise path.

For Memcache configuration, capacity tuning, and troubleshooting, refer to the
[KV Cache Pool guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend).

### 5.4 1P1D Deployment with Mooncake KV Cache Pool

This configuration extends the A2 1P1D deployment in [Section 5.2](#52-1p1d-pd-disaggregated-deployment) with a
Mooncake KV Cache Pool. `MooncakeHybridConnector` continues to transfer KV
cache from Prefill to Decode, while `AscendStoreConnector` stores reusable KV
cache in Mooncake. Both connectors are configured as children of
`MultiConnector`.

DeepSeek-V4.1-Flash does not support layerwise KV Cache Pool access because its
CED hybrid-attention model structure is incompatible with layer-by-layer KV
save and load. Keep `use_layerwise` set to `false` on both Prefill and Decode.

Before starting vLLM, configure `mooncake.json`, and start the
mooncake_master Service. The Prefill and Decode nodes must export `MOONCAKE_CONFIG_PATH`.

#### 5.4.1 Prepare the Mooncake Services

Configure `mooncake.json`:

```json
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "xx.xx.xx.xx:50088",
    "global_segment_size": "24GB",
    "preferred_segment": false,
    "prefer_alloc_in_same_node": true
}
```

Start mooncake_master:

After `mooncake.json` is ready, start the mooncake_master process on one node and wait for successful initialization:

```bash
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 11000
```

eviction_high_watermark_ratio determines the watermark where Mooncake Store will perform eviction.
eviction_ratio determines the portion of stored objects that would be evicted.
default_kv_lease_ttl controls the default lease TTL for KV objects (milliseconds).
configure it via --default_kv_lease_ttl and keep it larger than ASCEND_CONNECT_TIMEOUT and ASCEND_TRANSFER_TIMEOUT.

#### 5.4.2 Start the Prefill Node with Pooling

On the Prefill node, save the following script as `run_dp_template.sh`. Set
`LOCAL_IP`, `NIC_NAME`, `MODEL_PATH`, and
`MOONCAKE_CONFIG_DIR` for the local environment.

```shell
#!/usr/bin/env bash
set -euo pipefail
unset https_proxy
unset http_proxy
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"
export no_proxy="127.0.0.1,localhost,<PREFILL_NODE_1_IP>,<PREFILL_NODE_2_IP>,<DECODE_NODE_1_IP>,<DECODE_NODE_2_IP>"
NIC_NAME="<NETWORK_INTERFACE>"
LOCAL_IP="<PREFILL_NODE_IP>"
MODEL_PATH="<YOUR_MODEL_PATH>"
MOONCAKE_CONFIG_DIR="<MOONCAKE_CONFIG_DIR>"

export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export VLLM_USE_V2_MODEL_RUNNER=0
export HCCL_INTRA_ROCE_ENABLE=1
export VLLM_ENGINE_READY_TIMEOUT_S=360000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_BUFFSIZE=1024
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=$1

# Parameters required by the Mooncake KV Cache Pool.
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH="$MOONCAKE_CONFIG_DIR/mooncake.json"

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
    --served-model-name dsv41 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 16 \
    --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
    --trust-remote-code \
    --block-size 128 \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --enforce-eager \
    --enable-prefix-caching \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --additional-config '{
        "enable_cpu_binding":true,
        "enable_dsa_cp":true,
        "enable_flashcomm1":true,
        "enable_shared_expert_dp":true
    }' \
    --kv-transfer-config '{
        "kv_connector":"MultiConnector",
        "kv_role":"kv_producer",
        "kv_port":"30000",
        "engine_id":"0",
        "kv_connector_extra_config":{
            "connectors":[
                {
                    "kv_connector":"MooncakeHybridConnector",
                    "kv_role":"kv_producer",
                    "kv_port":"30000",
                    "kv_connector_extra_config":{
                        "prefill":{"dp_size":4,"tp_size":4},
                        "decode":{"dp_size":8,"tp_size":2}
                    }
                },
                {
                    "kv_connector":"AscendStoreConnector",
                    "kv_role":"kv_producer",
                    "kv_connector_extra_config":{
                        "lookup_rpc_port":"0",
                        "backend":"mooncake",
                        "use_layerwise":false
                    }
                }
            ]
        }
    }'
```

Start the four DP4/TP4 Prefill engines with the launcher command from [Section 5.2.2.2](#5222-start-the-prefill-nodes) after all Decode engines are ready.

#### 5.4.3 Start the Decode Node with Pooling

On the Decode node, save the following script as `run_dp_template.sh`. Set
`LOCAL_IP`, `NIC_NAME`, `MODEL_PATH`, and
`MOONCAKE_CONFIG_DIR` for the local environment.

```shell
#!/usr/bin/env bash
set -euo pipefail
unset https_proxy
unset http_proxy
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"
export no_proxy="127.0.0.1,localhost,<PREFILL_NODE_1_IP>,<PREFILL_NODE_2_IP>,<DECODE_NODE_1_IP>,<DECODE_NODE_2_IP>"
NIC_NAME="<NETWORK_INTERFACE>"
LOCAL_IP="<DECODE_NODE_IP>"
MODEL_PATH="<YOUR_MODEL_PATH>"
MOONCAKE_CONFIG_DIR="<MOONCAKE_CONFIG_DIR>"

export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=360000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_BUFFSIZE=1800
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=$1

# Parameters required by the Mooncake KV Cache Pool.
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH="$MOONCAKE_CONFIG_DIR/mooncake.json"

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
    --served-model-name dsv41 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 400 \
    --max-num-seqs 32 \
    --async-scheduling \
    --block-size 128 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config '{
        "kv_connector":"MultiConnector",
        "kv_role":"kv_consumer",
        "kv_port":"30100",
        "engine_id":"1",
        "kv_connector_extra_config":{
            "connectors":[
                {
                    "kv_connector":"MooncakeHybridConnector",
                    "kv_role":"kv_consumer",
                    "kv_port":"30100",
                    "kv_connector_extra_config":{
                        "prefill":{"dp_size":4,"tp_size":4},
                        "decode":{"dp_size":8,"tp_size":2}
                    }
                },
                {
                    "kv_connector":"AscendStoreConnector",
                    "kv_role":"kv_consumer",
                    "kv_connector_extra_config":{
                        "lookup_rpc_port":"0",
                        "backend":"mooncake",
                        "use_layerwise":false
                    }
                }
            ]
        }
    }' \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --additional-config '{
        "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
        },
        "enable_cpu_binding":true,
        "multistream_overlap_shared_expert":true,
        "recompute_scheduler_enable":true
    }'
```

Start the eight DP8/TP2 Decode engines with the launcher command from [Section 5.2.2.3](#5223-start-the-decode-nodes).

#### 5.4.4 Startup and Verification

Start the components in this order: Mooncake Master Service, Decode engines, Prefill engines, and then the PD proxy. The Mooncake Store clients are embedded in the vLLM processes; no standalone Store process is required for this configuration.
Wait until the Mooncake master process reports successful initialization before starting the Decode engines and the Prefill engines. After every Prefill engine and every Decode engine is ready, start the proxy with the same endpoint mapping described in [Section 5.2.2.4](#5224-deploy-the-pd-proxy).

Send the same request twice with a reusable prefix. Confirm that Prefill logs show Mooncake lookup, get, put, and hit activity, and that Decode logs still show normal Prefill-to-Decode KV transfer.

Key pooling parameters:

- `MultiConnector` combines live P-to-D transfer through `MooncakeHybridConnector` with persistent prefix reuse through `AscendStoreConnector`.
- `backend: "mooncake"` selects Mooncake as the KV Cache Pool backend.
- `use_layerwise: false` is required for DeepSeek-V4.1-Flash. Do not enable layerwise pooling for this model because its CED hybrid-attention structure is not supported by the layerwise path.

For Mooncake configuration, and troubleshooting, refer to the [KV Cache Pool guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-1-mooncake-backend).

### 5.5 Multi-Node Colocated Deployment

=== "A2 series"

    This two-node colocated configuration uses two Atlas 800 A2 servers with one
    local DP rank per server and a global DP2/TP8/EP topology.

    Run this script on both A2 servers. Set `NODE_RANK` to `0` or
    `1` on the corresponding node. Set `NODE0_IP` to the IP address of Node 0 and
    set `LOCAL_IP`, `NIC_NAME`, and `MODEL_PATH` for each node.

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail
    NODE_RANK=0

    NODE0_IP="<NODE0_IP>"
    LOCAL_IP="<LOCAL_IP>"
    NIC_NAME="<NETWORK_INTERFACE>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_BUFFSIZE=1024
    export HCCL_CONNECT_TIMEOUT=1200
    export VLLM_ENGINE_READY_TIMEOUT_S=360000
    export VLLM_USE_V2_MODEL_RUNNER=0
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    HEADLESS_ARGS=()
    if [[ "$NODE_RANK" != "0" ]]; then
      HEADLESS_ARGS+=(--headless --data-parallel-start-rank "$NODE_RANK")
    fi

    vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port 8000 \
      "${HEADLESS_ARGS[@]}" \
      --data-parallel-address "$NODE0_IP" \
      --data-parallel-rpc-port 13399 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name dsv41 \
      --max-model-len 1048576 \
      --max-num-batched-tokens 8192 \
      --max-num-seqs 32 \
      --async-scheduling \
      --block-size 128 \
      --enable-prefix-caching \
      --trust-remote-code \
      --tokenizer-mode deepseek_v41 \
      --reasoning-parser deepseek_v41 \
      --tool-call-parser deepseek_v41 \
      --enable-auto-tool-choice \
      --safetensors-load-strategy lazy \
      --gpu-memory-utilization 0.9 \
      --quantization ascend \
      --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
      --additional-config '{
          "ascend_compilation_config":{
              "enable_npugraph_ex":true,
              "enable_static_kernel":false
          },
          "enable_cpu_binding":true,
          "enable_dsa_cp":true,
          "enable_flashcomm1":true,
          "enable_shared_expert_dp":true
      }'
    ```

    Omit `--data-parallel-start-rank` on Node 0. Start Node 0 first, then start Node 1. Only Node 0 exposes the API endpoint. Each server contributes one TP8 rank to the global DP2 topology.

### 5.6 Service Verification

Set the endpoint for the selected deployment, then verify the health endpoint.
Use `<A3_IP>` with port `8900` for A3 single-node colocated deployment,
`<PROXY_IP>` with `<PROXY_PORT>` for A3 PD deployment, or `<A2_NODE0_IP>` with
port `8000` for the retained A2 deployment.

```shell
export SERVICE_URL="http://<SERVICE_IP>:<SERVICE_PORT>"

curl -sS -o /dev/null -w 'HTTP %{http_code}\n' \
  "$SERVICE_URL/health"
```

Expected output:

```output
HTTP 200
```

Then verify that the configured model is available:

```shell
curl -sS "$SERVICE_URL/v1/models" | \
  jq '{object, models: [.data[] | {id, object}]}'
```

The response must contain a model entry whose `id` matches the configured
`--served-model-name`: `dsv41` for the A3 examples and `deepseek-v41` for the
retained A2 example.

## 6 Functional Verification

### 6.1 Text Request

```shell
export SERVICE_URL="http://<SERVICE_IP>:<SERVICE_PORT>"

curl -sS "$SERVICE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv41",
    "messages": [{"role": "user", "content": "Who are you?"}],
    "temperature": 0,
    "max_completion_tokens": 256
  }' | jq -e '.choices[0].message.content | length > 0'
```

Expected output:

```output
true
```

### 6.2 Image Request

Set `IMAGE_URL` to an HTTP(S) image URL reachable from the node that executes
Prefill, and send a multimodal request:

```shell
export IMAGE_URL="<YOUR_IMAGE_URL>"
export SERVICE_URL="http://<SERVICE_IP>:<SERVICE_PORT>"

curl -sS "$SERVICE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"dsv41\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"${IMAGE_URL}\"}},
        {\"type\": \"text\", \"text\": \"Describe this image.\"}
      ]
    }],
    \"temperature\": 0,
    \"max_completion_tokens\": 256
  }" | jq -e '.choices[0].message.content | length > 0'
```

Expected output:

```output
true
```

## 7 Accuracy Evaluation

### 7.1 Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md)
to evaluate the deployed service.

### 7.2 Accuracy Results

The following table lists accuracy results for the
`DeepSeek-V4.1-Flash-1017-w4a8` checkpoint on A3 products and the generation
parameters used for each dataset.

| Dataset | Hardware | Score | generation_kwargs |
|---------|----------|-------|-------------------|
| GPQA | A3 series | 89.9 | `temperature=1.0, top_p=1.0, max_tokens=131072` |
| MMMU-PRO | A3 series | 81.3 | `temperature=1.0, top_p=0.95, max_tokens=131072` |

## 8 Performance Evaluation

Refer to the
[AISBench performance evaluation guide](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation)
or the [vLLM benchmark guide](https://github.com/vllm-project/vllm/blob/84030bbe3d74d99bad477a3d2e37a973ccd8865c/docs/benchmarking/README.md).
No production performance baseline is published for this configuration.

## 9 Performance Tuning

The values in [Section 5.1](#51-single-node-colocated-deployment) and [Section 5.2](#52-1p1d-pd-disaggregated-deployment) are starting points rather than globally
optimal settings. Tune `--max-num-seqs`, `--max-num-batched-tokens`, and
`--gpu-memory-utilization` together for the target input length, image sizes,
output length, and concurrency. Keep DP4/TP4 for A3 single-node deployment,
or Prefill DP4/PCP4/TP1 and Decode DP16/TP1 for A3 PD deployment, until an
alternative configuration has been validated.

### 9.1 Tool-Assisted Tuning with msmodeling

The msmodeling tool (MindStudio-Modeling) optimizes serving and benchmark
parameters together, combining measured results with knowledge-base experience
to determine a better parameter set in one run and push performance toward the
hardware limit.
For the complete, step-by-step usage guide please refer to
**[Optix-for-PD-disaggregation](https://gitcode.com/Ascend/msmodeling/blob/master/docs/en/quick_start/optix_agent_pd_disaggregation.md)**.

## 10 FAQ

### 10.1 How do I enable tool calling and reasoning parsing?

Keep the following options in the serving command:

```shell
--tokenizer-mode deepseek_v41 \
--tool-call-parser deepseek_v41 \
--reasoning-parser deepseek_v41 \
--enable-auto-tool-choice
```

For common environment, installation, and parameter issues, refer to the
[Public FAQs](../../faqs.md).

## 11 Limitations

- The documented A3 deployments use either one server in colocated mode or two
  servers in 1P1D mode, with an Ascend W4A8 checkpoint and INT8 Engram storage.
- The A2 configuration is retained unchanged and is not revalidated by this
  update. Its revised configuration will be documented separately.
- In the A3 PD example, Prefill runs in eager mode. Decode uses
  `FULL_DECODE_ONLY` ACL Graph with DSpark draft graph execution enabled.
- Production performance qualification and task-level accuracy evaluation are
  not complete.
