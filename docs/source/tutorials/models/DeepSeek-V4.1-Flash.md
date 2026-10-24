# DeepSeek-V4.1-Flash

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
is a native multimodal Mixture-of-Experts (MoE) model with 552B backbone
parameters and a context length of up to one million tokens. It uses a
40-layer Causal Encoder-Decoder (CED) architecture with 20 causal-encoder
layers and 20 decoder layers, activating 8B parameters per token during
Prefill and 16B during Decode.

The model introduces Compressed Sparse Attention 2 (CSA2), FP4 main KV cache,
Single-Pass mHC, Engram conditional memory, and DSpark
speculative decoding. These designs reduce the global KV cache footprint to
890 bytes per token and the persistent KV cache footprint to approximately
one eighth of DeepSeek-V4-Flash. The model accepts text and images and supports
a continuously adjustable reasoning effort from 1 to 100.

vLLM Ascend supports W8A8 deployment on Atlas 800 A3 and A2 servers, and
deployment with the original checkpoint on Ascend 950DT servers. This guide
covers colocated serving, distributed data parallelism, Prefill-Decode
(PD) disaggregation, and KV cache pooling. The 950DT examples use text-only
serving with Engram host offloading.

Use vLLM Ascend `releases/v0.30.0` with vLLM v0.30.0, or a development
build with the matching upstream revision recorded in
`.github/vllm-main-verified.commit` on the selected vLLM Ascend branch.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md)
for the complete support matrix and the
[Feature Guide](../../user_guide/feature_guide/index.md) for feature
configuration.

The A3 configurations in this guide use W8A8 weights and INT8 Engram storage.
The single-node colocated configuration uses DP4/TP4. The PD configuration
uses DP4/TP4 on the Prefill node, DP8/TP2 on the Decode node, DSpark
speculative decoding, and `FULL_DECODE_ONLY` ACL Graph on Decode. The
A3 scripts set `VLLM_USE_V2_MODEL_RUNNER=0` explicitly.

The 950DT configurations require model runner V2
(`VLLM_USE_V2_MODEL_RUNNER=1`). They use the original MXFP checkpoint,
TP1, expert parallelism, Engram host offloading, and DSpark with five
speculative tokens. The 1P1D example uses PCP8 on Prefill and DP8 on Decode;
the distributed DP example uses DP32 across four servers. Memcache pooling
uses bulk access to preserve the hybrid attention and state caches.

## 3 Prerequisites

### 3.1 Model Weights and Hardware

The official DeepSeek-V4.1-Flash checkpoint is available from
[Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and
[ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash).

For A2 and A3, the Ascend W8A8 checkpoint will be published as
[Eco-Tech/DeepSeek-V4.1-Flash-w8a8](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8)
on ModelScope. It includes the DSpark draft parameters and INT8 Engram tables.
After the checkpoint is available, download it to the same absolute path on
every server and replace the checkpoint-path placeholder in each serving
command.

Alternatively, use [ModelSlim](https://github.com/Ascend/msmodelslim) to
prepare a ModelSlim-compatible W8A8 checkpoint from the official weights.

For 950DT, download the original DeepSeek-V4.1-Flash checkpoint, including
its tokenizer, DSpark draft parameters, Engram tables, and safetensors
index. The native MXFP8 linear and MXFP4 MoE weights are loaded with
`--quantization deepseek_v4_fp8`; Ascend W8A8 conversion and
`--quantization ascend` are not used in the 950DT examples.

Use one of the following hardware configurations:

- **A3 single-node colocated**: one Atlas 800I A3 server with 8 NPUs and 128GB
  memory per NPU. The server exposes 16 logical devices to the container.
- **A3 1P1D**: two Atlas 800I A3 servers with the same device configuration.
- **A2 series**: four Atlas 800I A2 servers. Each server has 8 NPUs with 64GB
  memory per NPU and exposes 8 devices to the container. Its deployment
  configuration is retained unchanged in this update.
- **A2 1P1D**: four Atlas 800I A2 servers with the same device configuration.
- **950DT single-node colocated**: one Ascend 950DT server with 8 NPUs and
  96GB memory per NPU, using DP8/TP1/EP8.
- **950DT 1P1D, with or without pooling**: two such servers. Prefill uses
  DP1/TP1/PCP8/EP8; Decode uses DP8/TP1/EP8.
- **950DT distributed DP**: four such servers using global DP32/TP1/EP32,
  with eight local DP ranks per server.

Store the checkpoint in a shared directory or copy it to the same absolute
path on every server. Reserve host memory for Engram, model loading, and
runtime buffers in addition to any DRAM contributed to the KV cache pool.
Engram host allocations shared by local DP ranks must not be counted by
simply summing worker RSS.

### 3.2 Verify Multi-node Communication

Before deployment, follow
[Verify Multi-node Communication](../../getting_started/installation.md#installation-multi-node-interconnect).
All servers must be able to communicate through the selected network
interfaces, and the service ports must not be blocked.

## 4 Installation

### 4.1 Docker Image Installation

Select the tab for the target hardware. A2 and A3 use separate validation
images. For 950DT, use the matching A5 image and ensure the host HDK is
compatible with its CANN version, as described in the
[installation guide](../../getting_started/installation.md).

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

=== "950DT series"

    Run this command on every participating 950DT server. Set `MODEL_ROOT` to
    the directory containing the original checkpoint. Confirm that
    `/lib/route.conf`, `/etc/hccl_rootinfo.json`, and `/etc/hixlep` are configured
    for the server's UB topology before starting the container.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
    export MODEL_ROOT="/data/weights"
    export NPU_SMI="$(command -v npu-smi)"

    docker pull "$IMAGE"
    docker run --rm -it \
      --name deepseek-v41-950dt \
      --net=host \
      --shm-size=512g \
      --privileged=true \
      --ulimit memlock=-1:-1 \
      --ulimit nofile=65536:65536 \
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
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v "$NPU_SMI:/usr/local/bin/npu-smi:ro" \
      -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
      -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json:ro \
      -v /etc/hixlep:/etc/hixlep:ro \
      -v /lib/route.conf:/lib/route.conf:ro \
      -v /usr/bin/urma_admin:/usr/bin/urma_admin:ro \
      -v "$MODEL_ROOT:$MODEL_ROOT" \
      "$IMAGE" bash
    ```

    For openEuler, select the matching `-a5-openeuler` image. The container must
    have enough host memory and shared memory for Engram offloading and local
    DP sharing. Pooling also requires `memfabric-hybrid` and `memcache-hybrid`
    from a compatible runtime; see Section 5.3.5.

Change `MODEL_ROOT` if the checkpoint is stored elsewhere. Keep the same
absolute path inside and outside every container.

### 4.2 Source Code Installation

To build from source, follow the
[software environment installation guide](../../getting_started/installation.md#installation-software-environment)
and select `releases/v0.30.0` with vLLM v0.30.0. For development builds,
use the matching vLLM revision recorded in
`.github/vllm-main-verified.commit` on the selected branch.

## 5 Online Service Deployment

### 5.1 Single-Node Colocated Deployment

=== "A3 series"

    This configuration runs Prefill and Decode on one Atlas 800 A3 server. It
    uses DP4/TP4 across all 16 logical devices, expert parallelism, Engram host
    offload, asynchronous scheduling, and `FULL_DECODE_ONLY` ACL Graph.

    Set `MODEL_PATH` to the local checkpoint path.

    ```shell
    #!/usr/bin/env bash

    MODEL_PATH="<YOUR_MODEL_PATH>"

    export VLLM_USE_V2_MODEL_RUNNER=0
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
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --additional-config '{
            "ascend_compilation_config":{
                "enable_npugraph_ex":true,
                "enable_static_kernel":false
            },
            "enable_cpu_binding":true,
            "enable_fused_mc2":1,
            "enable_dsa_cp":true,
            "enable_flashcomm1":true,
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
    - `VLLM_USE_V2_MODEL_RUNNER=0` explicitly selects model runner V1. Keep this
        value at `0` because DeepSeek-V4.1-Flash currently does not support model
        runner V2 on Ascend.
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
    - `enable_fused_mc2`, `enable_dsa_cp`, `enable_flashcomm1`, and
        `enable_shared_expert_dp` enable the A3 MoE and communication optimizations
        used by this configuration. `enable_cpu_binding` pins worker processes to
        CPUs, while `enable_npugraph_ex` enables the enhanced ACL Graph path.

=== "950DT series"

    This configuration uses all eight NPUs as DP8/TP1/EP8. Save the following
    common environment as `env_950dt.sh` in the container. Set `LOCAL_IP` to
    this server's communication IP, `NIC_NAME` to its matching interface, and
    `MODEL_PATH` to the original checkpoint directory. Reuse this file for the
    950DT examples below, with the local values on each server.

    ```bash
    #!/usr/bin/env bash
    export LOCAL_IP="<LOCAL_IP>"
    export NIC_NAME="<NETWORK_INTERFACE>"
    export MODEL_PATH="/data/weights/DeepSeek-V4.1-Flash"
    export VLLM_ASCEND_PATH="/vllm-workspace/vllm-ascend"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_HOST_IP="$LOCAL_IP"
    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
    export HCCL_OP_EXPANSION_MODE=CCU_SCHED
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_EXEC_TIMEOUT=2040
    export VLLM_ENGINE_READY_TIMEOUT_S=3600
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    ulimit -n 65536
    ```

    Change `VLLM_ASCEND_PATH` if the repository is installed elsewhere. Save the
    following script as `run_colocated_950dt.sh` alongside `env_950dt.sh`:

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail
    source ./env_950dt.sh
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_BUFFSIZE=1600

    exec vllm serve "$MODEL_PATH" \
      --host "$LOCAL_IP" --port 8000 \
      --served-model-name dsv41 \
      --data-parallel-size 8 --data-parallel-size-local 8 \
      --tensor-parallel-size 1 --enable-expert-parallel \
      --max-model-len 1048576 --block-size 128 \
      --max-num-batched-tokens 4096 --max-num-seqs 64 \
      --gpu-memory-utilization 0.90 \
      --quantization deepseek_v4_fp8 \
      --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
      --tokenizer-mode deepseek_v41 \
      --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
      --reasoning-parser deepseek_v41 --trust-remote-code \
      --language-model-only --enable-prefix-caching --async-scheduling \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":false}' \
      --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":0,"enable_force_eplb":false,"multistream_engram_overlap":true,"multistream_overlap_shared_expert":false,"multistream_dsv4_dsa_overlap":true,"combine_quant_mode":4}'
    ```

    Run `bash run_colocated_950dt.sh` from this directory. Wait for model
    loading, graph capture, and the API on `http://<LOCAL_IP>:8000/health`.
    `--language-model-only` disables the vision encoder in all 950DT examples
    in this guide. `--max-model-len` is the combined input/output limit; reduce
    it or the batch limits if required by the workload and KV cache capacity.

### 5.2 1P1D PD Disaggregated Deployment

=== "A3 series"

    This example uses two Atlas 800 A3 servers. The Prefill node runs four DP
    ranks with TP4 (DP4/TP4), and the Decode node runs eight DP ranks with TP2
    (DP8/TP2). Both layouts consume all 16 logical devices on their respective
    servers. Mooncake transfers KV cache from the Prefill engines to the Decode
    engines.

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
    | `--dp-rank-start` | First DP rank assigned to this node. It is `0` for both single-node groups in this 1P1D example. |
    | `--dp-address` | IP address of the node that coordinates the corresponding DP group. Use the Prefill IP on the Prefill node and the Decode IP on the Decode node. |
    | `--dp-rpc-port` | DP coordination port. It must be unused and reachable within the node group. |
    | `--vllm-start-port` | First API port; the launcher increments it for each local DP rank. |

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
    export VLLM_USE_V2_MODEL_RUNNER=0
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
        --enable-prefix-caching \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --additional-config '{
            "enable_cpu_binding":true,
            "enable_fused_mc2":1,
            "enable_dsa_cp":true,
            "enable_flashcomm1":true,
            "enable_shared_expert_dp":true
        }' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeHybridConnector",
            "kv_role":"kv_producer",
            "kv_port":"30000",
            "kv_connector_extra_config":{
                "prefill":{"dp_size":4,"tp_size":4},
                "decode":{"dp_size":8,"tp_size":2}
            }
        }'
    ```

    Start four DP4/TP4 Prefill engines. `--dp-address` uses the Prefill node IP.

    ```shell
    PREFILL_NODE_IP="<PREFILL_NODE_IP>"

    python launch_online_dp.py \
        --dp-size 4 \
        --tp-size 4 \
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

    export VLLM_USE_V2_MODEL_RUNNER=0
    export VLLM_ENGINE_READY_TIMEOUT_S=36000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_BUFFSIZE=1800
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
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name dsv41 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 512 \
        --max-num-seqs 72 \
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

    Start eight DP8/TP2 Decode engines. `--dp-address` uses the Decode node IP.

    ```shell
    DECODE_NODE_IP="<DECODE_NODE_IP>"

    python launch_online_dp.py \
        --dp-size 8 \
        --tp-size 2 \
        --dp-size-local 8 \
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
    `<DECODE_NODE_IP>:7107`.

    #### 5.2.1.5 Key Parameter Descriptions

    - `VLLM_ENGINE_READY_TIMEOUT_S=36000` gives every Prefill and Decode engine up
      to 36,000 seconds to finish startup. This includes weight loading and Decode
      graph preparation.
    - `VLLM_USE_V2_MODEL_RUNNER=0` explicitly selects model runner V1 on both
      Prefill and Decode. Keep this value at `0` because DeepSeek-V4.1-Flash
      currently does not support model runner V2 on Ascend.
    - `--data-parallel-size` and `--tensor-parallel-size` define DP4/TP4 on
      Prefill and DP8/TP2 on Decode. Their product must be 16 on each A3 node.
    - `--data-parallel-address` and `--data-parallel-rpc-port` coordinate DP ranks
      within one role. Prefill and Decode use their respective node IPs; port
      `12321` can be reused because the roles run on different hosts.
    - `--vllm-start-port 7100` assigns API ports `7100-7103` on Prefill and
      `7100-7107` on Decode. These endpoints must be reachable by the PD proxy.
    - Both roles use `--max-model-len 1048576`. Prefill uses
      `--max-num-batched-tokens 8192` and `--max-num-seqs 16` to favor prompt
      throughput, while Decode uses `512` and `72` respectively to favor decode
      concurrency. Tune these role-specific scheduler limits independently.
    - `MooncakeHybridConnector` transfers KV cache between the two roles.
      `kv_role` must be `kv_producer` on Prefill and `kv_consumer` on Decode;
      `kv_port` must be reachable and must not conflict with another service.
    - `kv_connector_extra_config` must match the actual global layouts on both
      sides: Prefill DP4/TP4 and Decode DP8/TP2. Keep the values identical in the
      Prefill and Decode commands.
    - Both roles use DSpark with five speculative tokens. Prefill runs the target
      model in eager mode. Decode uses `FULL_DECODE_ONLY` for the target model,
      while `enforce_eager` inside `--speculative-config` applies to the Decode
      draft model.
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
        - `enable_fused_mc2: 1` enables the fused MoE communication and computation
        path to reduce expert dispatch, FFN, and combine overhead.
        - `enable_dsa_cp: true` enables context parallelism for DeepSeek Sparse
        Attention. This option applies to models with an indexer and requires the
        sequence-parallel MoE path used by this topology.
        - `enable_flashcomm1: true` enables the FlashComm communication backend used
        by the sequence-parallel MoE and DSA context-parallel paths.
        - `enable_shared_expert_dp: true` uses data parallelism for shared experts.
        It takes effect only when expert parallelism is enabled and TP is greater
        than one; both conditions are met by Prefill DP4/TP4.

    - Decode uses the following `--additional-config` options:

        - `ascend_compilation_config.enable_npugraph_ex: true` enables the enhanced
        ACL Graph compilation and execution path used by `FULL_DECODE_ONLY`.
        - `ascend_compilation_config.enable_static_kernel: false` disables static
        kernel generation while retaining the enhanced ACL Graph path.
        - `enable_cpu_binding: true` enables Ascend-native CPU affinity for worker
        processes and runtime threads.
        - `multistream_overlap_shared_expert: true` overlaps shared-expert execution
        with routed-expert computation on separate streams to improve Decode MoE
        throughput.
        - `recompute_scheduler_enable: true` enables the PD Decode recomputation
        scheduler. When the Decode-side KV cache is unavailable, the request can be
        sent back to Prefill to recompute it. Enable this option only on Decode.

      The Prefill and Decode configurations are role-specific. Do not copy the
      Prefill fused-MC2 and DSA options to Decode, or the Decode graph, multistream,
      and recomputation options to Prefill, without validating the resulting
      topology and performance.

      `HCCL_IF_IP` and the socket interface variables must select the same high-speed
      service network used by the configured node IPs. `HCCL_BUFFSIZE` is `1024` on
      Prefill and `1800` on Decode; retain these role-specific values unless another
      setting has been validated. The DP RPC, engine, Mooncake, and proxy ports must
      be allowed by the host firewall.
      Wait until every engine finishes loading weights and Decode finishes graph
      capture. A successful startup includes output similar to:

      ```text
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

    NIC_NAME="<NETWORK_INTERFACE>"
    LOCAL_IP="<LOCAL_PREFILL_NODE_IP>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    export no_proxy="127.0.0.1,localhost,<PREFILL_NODE_1_IP>,<PREFILL_NODE_2_IP>,<DECODE_NODE_1_IP>,<DECODE_NODE_2_IP>"
    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_HOST_IP="$LOCAL_IP"
    export VLLM_USE_V2_MODEL_RUNNER=0
    export VLLM_ENGINE_READY_TIMEOUT_S=360000
    export HCCL_BUFFSIZE=1024
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_INTRA_ROCE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
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
    export VLLM_HOST_IP="$LOCAL_IP"
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_INTRA_ROCE_ENABLE=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
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

    ```text
    INFO:     Started server process
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    ```

=== "950DT series"

    #### 5.2.3 950DT PCP8 Prefill and DP8 Decode

    Use one complete eight-NPU server per role. Prefill is one logical engine
    with DP1/TP1/PCP8/EP8 and one HTTP endpoint. Decode runs eight external DP
    ranks with TP1/EP8 and one HTTP endpoint per rank. PCP partitions the prompt
    across Prefill's eight NPUs; it does not create eight Prefill API servers.

    | Role | Devices | HTTP ports | DP RPC port | Mooncake handshake ports | Engine IDs |
    | ---- | ------- | ---------- | ----------- | ------------------------ | ---------- |
    | Prefill | 0–7, one PCP8 engine | 8000 | Not required for DP1 | 26010–26017 | 0 |
    | Decode | One device per DP rank | 8000–8007 | 13399 | 27010–27017 | 100–107 |
    | Proxy | No NPU required | 1999 | — | — | — |

    Prepare `env_950dt.sh` from Section 5.1 on both servers, using each
    server's own `LOCAL_IP` and `NIC_NAME`. The original checkpoint, model
    name, tokenizer, block size, and DSpark settings must match on both roles.
    Save this script as `run_dp_template.sh` alongside the environment file
    on both servers. It also supports the optional pool in Section 5.3.5.

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail
    source ./env_950dt.sh

    export ASCEND_RT_VISIBLE_DEVICES=${1:?visible devices required}
    API_PORT=${2:?API port required}
    DP_SIZE=${3:?DP size required}
    DP_RANK=${4:?DP rank required}
    DP_ADDRESS=${5:?DP address required}
    DP_RPC_PORT=${6:?DP RPC port required}
    TP_SIZE=${7:?TP size required}
    ROLE=${ROLE:?set ROLE to prefill or decode}
    POOL=${POOL:-0}

    DP_ARGS=(--data-parallel-size "$DP_SIZE" --tensor-parallel-size "$TP_SIZE")
    if (( DP_SIZE > 1 )); then
      DP_ARGS+=(--data-parallel-size-local 1 --data-parallel-rank "$DP_RANK"
        --data-parallel-address "$DP_ADDRESS" --data-parallel-rpc-port "$DP_RPC_PORT")
    fi

    case "$ROLE" in
      prefill)
        export HCCL_BUFFSIZE=1024
        ROLE_ARGS=(--prefill-context-parallel-size 8 --enforce-eager
          --max-num-batched-tokens 16384 --max-num-seqs 16
          --gpu-memory-utilization 0.90
          --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}')
        ADDITIONAL_CONFIG='{"enable_cpu_binding":true,"enable_fused_mc2":0,"enable_force_eplb":false,"enable_flashcomm1":true,"enable_shared_expert_dp":true,"enable_pcp_o_proj_weight_sharding":true,"multistream_engram_overlap":false,"multistream_overlap_shared_expert":false,"multistream_dsv4_dsa_overlap":true}'
        ;;
      decode)
        export HCCL_BUFFSIZE=1600
        ROLE_ARGS=(--max-num-batched-tokens 1024 --max-num-seqs 64
          --gpu-memory-utilization 0.92 --async-scheduling
          --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
          --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":false}')
        ADDITIONAL_CONFIG='{"enable_cpu_binding":true,"enable_fused_mc2":0,"enable_force_eplb":false,"multistream_engram_overlap":true,"multistream_overlap_shared_expert":false,"multistream_dsv4_dsa_overlap":true,"combine_quant_mode":4,"scheduler_config":{"recompute_scheduler_enable":true}}'
        ;;
      *) echo "ROLE must be prefill or decode" >&2; exit 2 ;;
    esac

    if [[ "$POOL" == 1 ]]; then
      source ./env_memcache_950dt.sh
    fi

    KV_CONFIG=$(python3 - "$ROLE" "$DP_RANK" "$POOL" <<'PY'
    import json
    import sys

    role, rank, pool = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
    producer = role == "prefill"
    kv_role = "kv_producer" if producer else "kv_consumer"
    engine_id = str(rank if producer else 100 + rank)
    transfer = {
        "kv_connector": "MooncakeHybridConnector",
        "kv_role": kv_role,
        "kv_port": 26010 if producer else 27010,
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 1},
            "decode": {"dp_size": 8, "tp_size": 1},
        },
    }
    if pool:
        store = {
            "kv_connector": "AscendStoreConnector",
            "kv_role": kv_role,
            "kv_connector_extra_config": {
                "backend": "memcache",
                "lookup_rpc_port": "0",
                "use_layerwise": False,
            },
        }
        config = {
            "kv_connector": "MultiConnector",
            "kv_role": kv_role,
            "engine_id": engine_id,
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {"connectors": [transfer, store]},
        }
    else:
        config = {**transfer, "engine_id": engine_id}
    print(json.dumps(config))
    PY
    )

    exec vllm serve "$MODEL_PATH" \
      --host "$LOCAL_IP" --port "$API_PORT" --served-model-name dsv41 \
      "${DP_ARGS[@]}" --enable-expert-parallel \
      --max-model-len 1048576 --block-size 128 \
      --quantization deepseek_v4_fp8 \
      --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
      --tokenizer-mode deepseek_v41 \
      --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
      --reasoning-parser deepseek_v41 --trust-remote-code \
      --language-model-only --enable-prefix-caching \
      "${ROLE_ARGS[@]}" \
      --kv-transfer-config "$KV_CONFIG" \
      --additional-config "$ADDITIONAL_CONFIG"
    ```

    `kv_port` is a base port. The connector adds DP/TP/PCP rank offsets, so keep
    the same base on all Decode ranks. Set `engine_id` only on the outer
    `MultiConnector` when pooling is enabled. `prefill.dp_size=1` and
    `decode.dp_size=8` describe logical DP ranks, not the total number of NPUs.
    Keep `enable_dsa_cp` disabled when using PCP with model runner V2.

    On Decode, copy the repository's external DP launcher into this directory
    and start all eight ranks in one terminal:

    ```bash
    source ./env_950dt.sh
    cp "$VLLM_ASCEND_PATH/examples/external_online_dp/launch_online_dp.py" .
    ROLE=decode POOL=0 python3 launch_online_dp.py \
      --dp-size 8 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 \
      --dp-address "$LOCAL_IP" --dp-rpc-port 13399 --vllm-start-port 8000
    ```

    Each Decode rank receives device `rank` and API port `8000 + rank` from
    the launcher. In another Decode terminal, wait for all eight endpoints:

    ```bash
    source ./env_950dt.sh
    for port in {8000..8007}; do
      until curl --noproxy '*' --fail --silent "http://$LOCAL_IP:$port/health"; do
        sleep 5
      done
    done
    ```

    On Prefill, start one PCP8 engine. All eight visible devices belong to
    this engine; do not use the DP launcher to split them into TP1 workers:

    ```bash
    source ./env_950dt.sh
    ROLE=prefill POOL=0 bash run_dp_template.sh \
      0,1,2,3,4,5,6,7 8000 1 0 "$LOCAL_IP" 13398 1
    ```

    Wait for Prefill's `/health` endpoint as well. Then start the repository's
    PD proxy on a host that can reach both nodes. Set `P_IP` and `D_IP` to the
    same communication addresses used by the serving scripts:

    ```bash
    source ./env_950dt.sh
    P_IP="<PREFILL_IP>"
    D_IP="<DECODE_IP>"
    python3 "$VLLM_ASCEND_PATH/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py" \
      --host 0.0.0.0 --port 1999 \
      --prefiller-hosts "$P_IP" --prefiller-ports 8000 \
      --decoder-hosts "$D_IP" "$D_IP" "$D_IP" "$D_IP" "$D_IP" "$D_IP" "$D_IP" "$D_IP" \
      --decoder-ports 8000 8001 8002 8003 8004 8005 8006 8007
    ```

    Check `http://<PROXY_IP>:1999/healthcheck`, then send the text request in
    Section 6.1 to `http://<PROXY_IP>:1999`. Confirm successful P-to-D KV
    transfer in both role logs as well as a nonempty response. HTTP success
    alone does not establish that KV transfer avoided recomputation.
    Reserve the listed service ports and allow the transport's communication
    ports between nodes. The handshake ranges above are outside the usual
    Linux ephemeral port range; check the actual host range with
    `cat /proc/sys/net/ipv4/ip_local_port_range` before deployment.

### 5.3 1P1D Deployment with Memcache KV Cache Pool

Sections 5.3.1 through 5.3.4 extend the A3 1P1D deployment in Section 5.2 with a
Memcache KV Cache Pool. For 950DT embedded LocalServices, use Section 5.3.5.
`MooncakeHybridConnector` continues to transfer KV
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
`/etc/hccn.conf` container mount from Section 4.1.

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

Reuse `launch_online_dp.py` from Section 5.2.1.1. Set `MEMCACHE_INSTALL_PATH` to
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
export VLLM_USE_V2_MODEL_RUNNER=0
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
    --enable-prefix-caching \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --additional-config '{
        "enable_cpu_binding":true,
        "enable_fused_mc2":1,
        "enable_dsa_cp":true,
        "enable_flashcomm1":true,
        "enable_shared_expert_dp":true
    }' \
    --kv-transfer-config '{
        "kv_connector":"MultiConnector",
        "kv_role":"kv_producer",
        "kv_port":"30000",
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
                        "backend":"memcache",
                        "lookup_rpc_port":"0",
                        "use_layerwise":false
                    }
                }
            ]
        }
    }'
```

Start the four DP4/TP4 Prefill engines with the launcher command from Section 5.2.2.1 after all Decode engines are ready.

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

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export HCCL_BUFFSIZE=1800
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
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name dsv41 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 72 \
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

Start the eight DP8/TP2 Decode engines with the launcher command from Section
5.2.3.

#### 5.3.4 Startup and Verification

Start the components in this order: Memcache MetaService, standalone Memcache
on both Prefill and Decode nodes, Decode engines, Prefill engines, and then the
PD proxy.

Wait until both standalone Memcache processes report successful initialization
before starting the Decode engines. Then wait until every Decode engine reports
`Application startup complete` before starting Prefill. After every Prefill engine is ready, start the proxy with the
same endpoint mapping described in Section 5.2.1.4.

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

#### 5.3.5 950DT Deployment with Embedded Memcache LocalServices

This example extends the 950DT PCP8 Prefill / DP8 Decode topology in Section
5.2.3. Both roles use `MultiConnector` with `MooncakeHybridConnector` for live
P-to-D transfer and `AscendStoreConnector` for reusable prefixes in Memcache.
Both roles retain vLLM HBM prefix caching. Use a single MetaService shared by
both servers and embedded LocalServices with DRAM storage. SSD storage and
additional Memcache HBM allocation are disabled in this example.

Install compatible `memfabric-hybrid` and `memcache-hybrid` packages according
to the [Memcache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend).
Use the same versions on both nodes. 950DT uses embedded LocalServices with
the `device_urma` protocol for Device UB. For a UBOE deployment, select the
compatible `device_uboe` protocol. The standalone A3 LocalService process in
Section 5.3.1 is not part of this topology.

Create a configuration directory on each node. Save this as `meta.conf` on
the MetaService node, replacing `<META_SERVICE_IP>` with its communication IP:

```ini
ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
ock.mmc.meta_service.config_store_url = tcp://<META_SERVICE_IP>:6000
ock.mmc.meta_service.metrics_url = http://<META_SERVICE_IP>:18000
ock.mmc.log_level = info
```

Save this as `mmc-local.conf` on both nodes with the same MetaService IP:

```ini
ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:6000
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_urma
ock.mmc.local_service.dram.size = 25GB
ock.mmc.local_service.max.dram.size = 25GB
ock.mmc.local_service.hbm.size = 0GB
ock.mmc.local_service.max.hbm.size = 0GB
ock.mmc.local_service.storage.enabled = false
ock.mmc.log_level = info
```

`dram.size` is allocated **per LocalService**. PCP8 Prefill has eight worker
LocalServices even though DP is one; DP8 Decode also has eight. At 25 GiB per
LocalService, budget approximately 200 GiB per server for the pool alone.
`world_size=256` is the maximum number of LocalServices, not an instruction
to allocate 256 DRAM segments. `max.dram.size` is an upper bound per service;
it does not allocate a second pool. Memcache `hbm.size=0GB` does not disable
vLLM HBM prefix caching.

Size the pool after accounting for Engram offloading, loading peaks, runtime
buffers, NUMA placement, and the container's memory limit. Check memory after
the model is ready; available memory before loading is not the pool budget.
Larger or asymmetric contributions, such as Prefill 64 GiB and Decode 32 GiB
per LocalService, require sufficient host memory and a common `max.dram.size`
at least as large as the largest contribution. Restart the affected services
after changing capacity. Use fresh pool data when changing model or KV layout.

Save this as `env_memcache_950dt.sh` alongside `run_dp_template.sh` on both
nodes. Set `MEMCACHE_CONFIG_DIR` to the local absolute configuration path:

```bash
#!/usr/bin/env bash
export MEMCACHE_CONFIG_DIR="<MEMCACHE_CONFIG_DIR>"
export MMC_LOCAL_CONFIG_PATH="$MEMCACHE_CONFIG_DIR/mmc-local.conf"
export PYTHONHASHSEED=0
MEMCACHE_PACKAGE_DIR=$(python3 -c 'import importlib.util; print(next(iter(importlib.util.find_spec("memcache_hybrid").submodule_search_locations)))')
export LD_LIBRARY_PATH="$MEMCACHE_PACKAGE_DIR/lib:${LD_LIBRARY_PATH:-}"
```

Start the services in separate terminals, keeping each terminal open:

1. On the MetaService node, load the Memcache environment and start MetaService:

    ```bash
    source ./env_memcache_950dt.sh
    export MMC_META_CONFIG_PATH="$MEMCACHE_CONFIG_DIR/meta.conf"
    python3 -u -c 'from memcache_hybrid import MetaService; MetaService.main()'
    ```

2. On Decode, run the same external DP launcher as in Section 5.2.3 with
   `POOL=1`. The environment file is loaded by each rank's template:

    ```bash
    source ./env_950dt.sh
    ROLE=decode POOL=1 python3 launch_online_dp.py \
      --dp-size 8 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 \
      --dp-address "$LOCAL_IP" --dp-rpc-port 13399 --vllm-start-port 8000
    ```

3. Wait for all eight Decode health endpoints, then start Prefill with `POOL=1`:

    ```bash
    source ./env_950dt.sh
    ROLE=prefill POOL=1 bash run_dp_template.sh \
      0,1,2,3,4,5,6,7 8000 1 0 "$LOCAL_IP" 13398 1
    ```

4. Wait for Prefill health, then start the same proxy and endpoint mapping as
   Section 5.2.3. The pool does not add any HTTP engines.

Keep `use_layerwise=false` for this model's hybrid attention/state layout.
The pool script sets `kv_load_failure_policy=fail` because hybrid-cache load
failures must not silently enter an unsupported recomputation path. This is
separate from the Decode recomputation scheduler setting.

To verify reuse, record counters before and after requests on the Prefill
server and MetaService:

```bash
P_IP="<PREFILL_IP>"
META_IP="<META_SERVICE_IP>"
curl --noproxy '*' -fsS "http://$P_IP:8000/metrics" | \
  grep -E 'prefix_cache_(hits|queries)_total|external_prefix_cache_(hits|queries)_total'
curl --noproxy '*' -fsS "http://$META_IP:18000/metrics" | \
  grep -E 'memcache_get_hits_dram_total|memcache_.*dram'
```

First send and complete a long text request, then reuse its prefix with a
different suffix. A local HBM hit can satisfy the second request without a
DRAM read. To exercise DRAM, use enough distinct prefixes to exceed the
Prefill HBM KV capacity while fitting in the DRAM pool, then repeat an older
prefix. Alternatively, on an idle test deployment, reset only Prefill's local
prefix cache with `POST http://<PREFILL_IP>:8000/reset_prefix_cache` before
repeating the request; verify that the reset succeeded.

Require both **Prefill external prefix-hit counters** and **MetaService DRAM
get-hit counters** to increase, and check the Prefill log for a successful
external load. Decode external hits include live P-to-D transfer and cannot
prove DRAM reuse by themselves. Confirm zero failed requests, valid outputs,
and sufficient host-memory headroom. A 90% hit workload or historical hit
ratio is not a guaranteed pool hit rate for a different prompt set.

Stop the proxy, then Prefill and Decode, then MetaService for this deployment.
Finish shutting down all worker processes before restarting the pool or
reusing its ports.

### 5.4 1P1D Deployment with Mooncake KV Cache Pool

This configuration extends the A2 1P1D deployment in Section 5.2 with a
Mooncake KV Cache Pool. `MooncakeHybridConnector` continues to transfer KV
cache from Prefill to Decode, while `AscendStoreConnector` stores reusable KV
cache in Mooncake. Both connectors are configured as children of
`MultiConnector`.

DeepSeek-V4.1-Flash does not support layerwise KV Cache Pool access because its
CED hybrid-attention model structure is incompatible with layer-by-layer KV
save and load. Keep `use_layerwise` set to `false` on both Prefill and Decode.

Before starting vLLM, configure `mooncake.json`, and start the
mooncake_master Service. The Prefill and Decode nodes must export `MOONCAKE_CONFIG_PATH` .

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

unset https_proxy
unset http_proxy

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
export TASK_QUEUE_ENABLE=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
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

Start the four DP4/TP4 Prefill engines with the launcher command from Section 5.2.2.1 after all Decode engines are ready.

#### 5.4.3 Start the Decode Node with Pooling

On the Decode node, save the following script as `run_dp_template.sh`. Set
`LOCAL_IP`, `NIC_NAME`, `MODEL_PATH`, and
`MOONCAKE_CONFIG_DIR` for the local environment.

```shell
#!/usr/bin/env bash

unset https_proxy
unset http_proxy

NIC_NAME="<NETWORK_INTERFACE>"
LOCAL_IP="<DECODE_NODE_IP>"
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
export HCCL_BUFFSIZE=1800
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
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

Start the eight DP8/TP2 Decode engines with the launcher command from Section 5.2.2.1.

#### 5.4.4 Startup and Verification

Start the components in this order: Mooncake Master Service, standalone Mooncake Store on both Prefill and Decode nodes, Decode engines, Prefill engines, and then the PD proxy.
Wait until the Mooncake master process reports successful initialization before starting the Decode engines and the Prefill engines. After every Prefill engine and every Decode engine is ready, start the proxy with the same endpoint mapping described in Section 5.2.2.4.

Send the same request twice with a reusable prefix. Confirm that Prefill logs show Mooncake lookup, get, put, and hit activity, and that Decode logs still show normal Prefill-to-Decode KV transfer.

Key pooling parameters:

- `MultiConnector` combines live P-to-D transfer through `MooncakeHybridConnector` with persistent prefix reuse through `AscendStoreConnector`.
- `backend: "mooncake"` selects Mooncake as the KV Cache Pool backend.
- `use_layerwise: false` is required for DeepSeek-V4.1-Flash. Do not enable layerwise pooling for this model because its CED hybrid-attention structure is not supported by the layerwise path.

### 5.5 Multi-Node Colocated Deployment

=== "A2 series"

    The existing A2 configuration is retained unchanged in this update. It uses
    four Atlas 800 A2 servers with one local DP rank per server and a global
    DP4/TP8/EP32 topology.

    Run this script on all four A2 servers. Set `NODE_RANK` to `0`, `1`, `2`, or
    `3` on the corresponding node. Set `NODE0_IP` to the IP address of Node 0 and
    set `LOCAL_IP`, `NIC_NAME`, and `MODEL_PATH` for each node.

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail

    NODE_RANK=0
    NODE0_IP="<NODE0_IP>"
    LOCAL_IP="<LOCAL_IP>"
    NIC_NAME="<NETWORK_INTERFACE>"
    MODEL_PATH="<YOUR_MODEL_PATH>"

    # Allow time for weight loading and graph capture on large models.
    export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_IF_IP="$LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
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
      --data-parallel-size 4 \
      --data-parallel-size-local 1 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --served-model-name deepseek-v41 \
      --max-model-len 1048576 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 32 \
      --gpu-memory-utilization 0.90 \
      --block-size 128 \
      --tokenizer-mode deepseek_v41 \
      --reasoning-parser deepseek_v41 \
      --tool-call-parser deepseek_v41 \
      --enable-auto-tool-choice \
      --trust-remote-code \
      --safetensors-load-strategy lazy \
      --quantization ascend \
      --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
      --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    ```

    Omit `--data-parallel-start-rank` on Node 0. Start Node 0 first, followed by
    Nodes 1 through 3. Only Node 0 exposes the API endpoint. Each server contributes
    one TP8 rank to the global DP4 topology.

=== "950DT series"

    Use four eight-NPU servers for global DP32/TP1/EP32 colocated serving.
    Each server contributes eight local DP ranks. This example uses vLLM's
    internal DP load balancing: Node 0 exposes a single API, and Nodes 1–3
    run headless. It is separate from the one-process-per-rank PD launcher.

    Prepare `env_950dt.sh` from Section 5.1 on each server. Save this as
    `run_distributed_950dt.sh`, set `NODE_RANK` to 0, 1, 2, or 3, and use the
    same `NODE0_IP` on all nodes:

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail
    source ./env_950dt.sh
    NODE_RANK=0
    NODE0_IP="<NODE0_IP>"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_BUFFSIZE=1600

    NODE_ARGS=()
    if (( NODE_RANK > 0 )); then
      NODE_ARGS+=(--headless --data-parallel-start-rank "$((NODE_RANK * 8))")
    fi

    exec vllm serve "$MODEL_PATH" \
      --host "$LOCAL_IP" --port 8000 --served-model-name dsv41 \
      --data-parallel-size 32 --data-parallel-size-local 8 \
      --data-parallel-address "$NODE0_IP" --data-parallel-rpc-port 13399 \
      "${NODE_ARGS[@]}" \
      --tensor-parallel-size 1 --enable-expert-parallel \
      --max-model-len 1048576 --block-size 128 \
      --max-num-batched-tokens 4096 --max-num-seqs 64 \
      --gpu-memory-utilization 0.90 \
      --quantization deepseek_v4_fp8 \
      --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
      --tokenizer-mode deepseek_v41 \
      --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
      --reasoning-parser deepseek_v41 --trust-remote-code \
      --language-model-only --enable-prefix-caching --async-scheduling \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":false}' \
      --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":0,"enable_force_eplb":false,"multistream_engram_overlap":true,"multistream_overlap_shared_expert":false,"multistream_dsv4_dsa_overlap":true,"combine_quant_mode":4}'
    ```

    Start Node 0, then Nodes 1–3. Their global DP rank ranges are 0–7, 8–15,
    16–23, and 24–31. All four nodes must join before the service is ready.
    Check `http://<NODE0_IP>:8000/health` and send requests to Node 0.
    Keep host-memory capacity for the shared Engram allocation on each server.
    Changing global DP or local DP requires matching rank ranges and enough
    devices on every participating server.

### 5.6 Service Verification

Set the endpoint for the selected deployment, then verify the health endpoint.
Use `<A3_IP>` with port `8900` for A3 single-node colocated deployment,
`<PROXY_IP>` with `<PROXY_PORT>` for A3 PD deployment, or `<A2_NODE0_IP>` with
port `8000` for the retained A2 deployment. For 950DT, use port `8000` on
the colocated server or distributed DP Node 0, or port `1999` on the PD
proxy. The example PD proxy exposes `/healthcheck`; check `/health` on
each vLLM backend directly as described in Section 5.2.3.

```shell
export SERVICE_URL="http://<SERVICE_IP>:<SERVICE_PORT>"
export HEALTH_PATH=health  # Use healthcheck for the PD proxy.

curl -sS -o /dev/null -w 'HTTP %{http_code}\n' \
  "$SERVICE_URL/${HEALTH_PATH:-health}"
```

Expected output:

```text
HTTP 200
```

Then verify that the configured model is available:

```shell
curl -sS "$SERVICE_URL/v1/models" | \
  jq '{object, models: [.data[] | {id, object}]}'
```

The response must contain a model entry whose `id` matches the configured
`--served-model-name`: `dsv41` for the A3 and 950DT examples and `deepseek-v41` for the
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

```text
true
```

### 6.2 Image Request

The 950DT scripts in this guide use `--language-model-only`; use the A3
configuration for this image example.

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

```text
true
```

## 7 Accuracy Evaluation

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md)
to evaluate the deployed service. Accuracy results will be documented in the
[release notes](../../user_guide/release_notes.md). When reporting results, record the
checkpoint, prompt encoder, reasoning effort, sampling parameters, dataset
version, and whether DSpark is enabled.

## 8 Performance Evaluation

Refer to the
[AISBench performance evaluation guide](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation)
or the [vLLM benchmark guide](https://github.com/vllm-project/vllm/blob/84030bbe3d74d99bad477a3d2e37a973ccd8865c/docs/benchmarking/README.md).
No production performance baseline is published for this configuration.

## 9 Performance Tuning

The values in Sections 5.1 and 5.2 are starting points rather than globally
optimal settings. Tune `--max-num-seqs`, `--max-num-batched-tokens`, and
`--gpu-memory-utilization` together for the target input length, image sizes,
output length, and concurrency. Keep DP4/TP4 for A3 single-node deployment,
or Prefill DP4/TP4 and Decode DP8/TP2 for A3 PD deployment, until an
alternative configuration has been validated.

The 950DT starting configurations are summarized below. `max-num-seqs` and
`max-num-batched-tokens` apply per DP engine. PCP8 Prefill is one engine spread
across eight devices.

| Deployment / role | NPUs | DP / TP / PCP | Max sequences | Batch tokens | Memory utilization | Execution |
| ----------------- | ---- | ------------- | ------------- | ------------ | ------------------ | --------- |
| Single-node colocated | 8 | 8 / 1 / 1 | 64 | 4096 | 0.90 | FULL_DECODE_ONLY |
| Four-node colocated | 32 | 32 / 1 / 1 | 64 | 4096 | 0.90 | FULL_DECODE_ONLY |
| 1P1D Prefill | 8 | 1 / 1 / 8 | 16 | 16384 | 0.90 | Eager |
| 1P1D Decode | 8 | 8 / 1 / 1 | 64 | 1024 | 0.92 | FULL_DECODE_ONLY |

All four use DSpark with five speculative tokens and Engram host offloading.
Keep TP1 and expert parallelism for these examples. PCP8 needs all eight
devices visible to the same Prefill engine. The Memcache example keeps the
same model topology and batch settings; size its DRAM pool independently of
the vLLM HBM utilization. Increase batch limits or memory utilization only
after measuring startup, graph capture, KV capacity, and the target workload.

## 10 FAQ

### How do I enable tool calling and reasoning parsing?

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
  servers in 1P1D mode, with an Ascend W8A8 checkpoint and INT8 Engram storage.
- The A2 configuration is retained unchanged and is not revalidated by this
  update. Its revised configuration will be documented separately.
- A2/A3 scripts use model runner V1. Keep `VLLM_USE_V2_MODEL_RUNNER=0` in
  every A3 serving script. 950DT requires model runner V2 and
  `VLLM_USE_V2_MODEL_RUNNER=1`. Pipeline parallelism is not covered by this
  guide.
- The 950DT examples use the original checkpoint and text-only serving.
  Their runtime configuration is distinct from the A2/A3 W8A8 examples.
- KV pool access for this hybrid model uses bulk transfers. Keep
  `use_layerwise=false`; the A3 standalone Memcache process is not used
  by the 950DT embedded LocalService example.
- In the A3 PD example, Prefill runs in eager mode. Decode uses
  `FULL_DECODE_ONLY` ACL Graph for the target model and eager execution for the
  DSpark draft model.
- Production performance qualification and task-level accuracy evaluation are
  not complete.
