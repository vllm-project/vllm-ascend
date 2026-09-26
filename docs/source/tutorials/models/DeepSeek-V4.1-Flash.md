# DeepSeek-V4.1-Flash

## 1 Introduction

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

vLLM Ascend supports W8A8 deployment on Atlas 800 A3 and A2 servers. This
guide provides a single-node colocated A3 configuration and a two-node A3
Prefill-Decode (PD) disaggregated configuration.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md)
for the complete support matrix and the
[Feature Guide](../../user_guide/feature_guide/index.md) for feature
configuration.

The A3 configurations in this guide use W8A8 weights and INT8 Engram storage.
The single-node colocated configuration uses DP4/TP4. The PD configuration
uses DP4/TP4 on the Prefill node, DP8/TP2 on the Decode node, DSpark
speculative decoding, and `FULL_DECODE_ONLY` ACL Graph on Decode. Both use
model runner V1.

## 3 Prerequisites

### 3.1 Model Weights and Hardware

The official DeepSeek-V4.1-Flash checkpoint is available from
[Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and
[ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash).

The Ascend W8A8 checkpoint used by this guide will be published as
[Eco-Tech/DeepSeek-V4.1-Flash-w8a8](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8)
on ModelScope. It includes the DSpark draft parameters and INT8 Engram tables.
After the checkpoint is available, download it to the same absolute path on
every server and replace the checkpoint-path placeholder in each serving
command.

Alternatively, use [ModelSlim](https://github.com/Ascend/msmodelslim) to
prepare a ModelSlim-compatible W8A8 checkpoint from the official weights.

Use one of the following hardware configurations:

- **A3 single-node colocated**: one Atlas 800 A3 server with 8 NPUs and 128GB
  memory per NPU. The server exposes 16 logical devices to the container.
- **A3 1P1D**: two Atlas 800 A3 servers with the same device configuration.
- **A2 series**: four Atlas 800 A2 servers. Each server has 8 NPUs with 64GB
  memory per NPU and exposes 8 devices to the container. Its deployment
  configuration is retained unchanged in this update.

Store the checkpoint in a shared directory or copy it to the same absolute
path on every server.

### 3.2 Verify Multi-node Communication

Before deployment, follow
[Verify Multi-node Communication](../../getting_started/installation.md#installation-multi-node-interconnect).
All servers must be able to communicate through the selected network
interfaces, and the service ports must not be blocked.

## 4 Installation

### 4.1 Docker Image Installation

#### A3 series

An A3 server exposes 16 logical devices. Run this command on every A3 server.

```shell
export IMAGE=quay.io/ascend/vllm-ascend:deepseek-v4.1-flash-a3
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

#### A2 series

An A2 server exposes 8 devices. Run this command on all four A2 servers.

```shell
export IMAGE=quay.io/ascend/vllm-ascend:deepseek-v4.1-flash
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
and use the `main` branch with the matching vLLM revision recorded in
`.github/vllm-main-verified.commit`.

## 5 Online Service Deployment

### 5.1 A3 Single-Node Colocated Deployment

This configuration runs Prefill and Decode on one Atlas 800 A3 server. It
uses DP4/TP4 across all 16 logical devices, expert parallelism, Engram host
offload, asynchronous scheduling, and `FULL_DECODE_ONLY` ACL Graph.

Replace `/your/path/DeepSeek-V4.1-Flash-W8A8` in the command below with the
local checkpoint path.

```shell
#!/usr/bin/env bash

export VLLM_ENGINE_READY_TIMEOUT_S=36000

export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"

# Optional: use jemalloc when it is installed in the container
if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# Ensure the model path matches the directory recorded during download
vllm serve /your/path/DeepSeek-V4.1-Flash-W8A8 \
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
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
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
- `--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'` keeps the
  Engram table in host memory and lets local DP ranks share the host-memory
  allocation. This reduces duplicate host-memory copies, but requires enough
  `/dev/shm` capacity and a shared IPC namespace. The A3 container example uses
  `--shm-size=512g` and starts all four DP ranks in the same container.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` captures the
  Decode path while Prefill remains outside the captured graph.
- `--safetensors-load-strategy lazy` avoids eagerly materializing the whole
  checkpoint during loading.
- `enable_fused_mc2`, `enable_dsa_cp`, `enable_flashcomm1`, and
  `enable_shared_expert_dp` enable the A3 MoE and communication optimizations
  used by this configuration. `enable_cpu_binding` pins worker processes to
  CPUs, while `enable_npugraph_ex` enables the enhanced ACL Graph path.

### 5.2 A3 1P1D PD Separation Deployment

This example uses two Atlas 800 A3 servers. The Prefill node runs four DP
ranks with TP4 (DP4/TP4), and the Decode node runs eight DP ranks with TP2
(DP8/TP2). Both layouts consume all 16 logical devices on their respective
servers. Mooncake transfers KV cache from the Prefill engines to the Decode
engines.

#### 5.2.1 Prepare the DP Launcher

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

#### 5.2.2 Start the Prefill Node

On the Prefill node, save the following script as `run_dp_template.sh`. Replace
`xx.xx.xx.1` and `xxxx` with the Prefill node's service IP and network
interface. Replace `/your/path/DeepSeek-V4.1-Flash-W8A8` with the local
checkpoint path.

```shell
#!/usr/bin/env bash

unset https_proxy
unset http_proxy
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"

nic_name="xxxx"       # Prefill node service network interface
local_ip=xx.xx.xx.1    # Prefill node service IP

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=$1

# Ensure the model path matches the directory recorded during download
vllm serve /your/path/DeepSeek-V4.1-Flash-W8A8 \
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
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
    --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --enforce-eager \
    --enable-prefix-caching \
    --engram-config '{"cpu_offload":true}' \
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
        "engine_id":"0",
        "kv_connector_extra_config":{
            "prefill":{"dp_size":4,"tp_size":4},
            "decode":{"dp_size":8,"tp_size":2}
        }
    }'
```

Start four DP4/TP4 Prefill engines. `--dp-address` uses the Prefill node IP.

```shell
python launch_online_dp.py \
    --dp-size 4 \
    --tp-size 4 \
    --dp-size-local 4 \
    --dp-rank-start 0 \
    --dp-address xx.xx.xx.1 \
    --dp-rpc-port 12321 \
    --vllm-start-port 7100
```

#### 5.2.3 Start the Decode Node

On the Decode node, save the following script as `run_dp_template.sh`. Replace
`xx.xx.xx.2` and `xxxx` with the Decode node's service IP and network
interface. Replace `/your/path/DeepSeek-V4.1-Flash-W8A8` with the same local
checkpoint path used on the Prefill node.

```shell
#!/usr/bin/env bash

unset https_proxy
unset http_proxy
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"

nic_name="xxxx"       # Decode node service network interface
local_ip=xx.xx.xx.2    # Decode node service IP

export VLLM_ENGINE_READY_TIMEOUT_S=36000
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_BUFFSIZE=1800
export ASCEND_RT_VISIBLE_DEVICES=$1

# Ensure the model path matches the directory recorded during download
vllm serve /your/path/DeepSeek-V4.1-Flash-W8A8 \
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
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
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
    --engram-config '{"cpu_offload":true}' \
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
python launch_online_dp.py \
    --dp-size 8 \
    --tp-size 2 \
    --dp-size-local 8 \
    --dp-rank-start 0 \
    --dp-address xx.xx.xx.2 \
    --dp-rpc-port 12321 \
    --vllm-start-port 7100
```

#### 5.2.4 Deploy the PD Proxy

After all Prefill and Decode engines are ready, deploy the proxy as described
in [Prefill-Decode Disaggregation (DeepSeek)](../features/pd_disaggregation_mooncake_multi_node.md).
Configure the proxy with Prefill endpoints `xx.xx.xx.1:7100` through
`xx.xx.xx.1:7103` and Decode endpoints `xx.xx.xx.2:7100` through
`xx.xx.xx.2:7107`.

#### 5.2.5 Key Parameter Descriptions

- `VLLM_ENGINE_READY_TIMEOUT_S=36000` gives every Prefill and Decode engine up
  to 36,000 seconds to finish startup. This includes weight loading and Decode
  graph preparation.
- `--data-parallel-size` and `--tensor-parallel-size` define DP4/TP4 on
  Prefill and DP8/TP2 on Decode. Their product must be 16 on each A3 node.
- `--data-parallel-address` and `--data-parallel-rpc-port` coordinate DP ranks
  within one role. Prefill and Decode use their respective node IPs; port
  `12321` can be reused because the roles run on different hosts.
- `--vllm-start-port 7100` assigns API ports `7100-7103` on Prefill and
  `7100-7107` on Decode. These endpoints must be reachable by the PD proxy.
- Both roles use `--max-model-len 1048576`. Prefill uses
  `--max-num-batched-tokens 8192` and `--max-num-seqs 16` to favor prompt
  throughput, while Decode uses `400` and `32` respectively to favor decode
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
- `--engram-config '{"cpu_offload":true}'` keeps the Engram table in host
  memory. Because `dp_shared_memory` is not enabled in the P/D commands, each
  local DP rank has its own host-memory allocation; reserve enough host memory
  on both nodes.
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

### 5.3 A2 Multi-Node Colocated Deployment

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
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
  export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
fi

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
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy \
  --quantization ascend \
  --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Omit `--data-parallel-start-rank` on Node 0. Start Node 0 first, followed by
Nodes 1 through 3. Only Node 0 exposes the API endpoint. Each server contributes
one TP8 rank to the global DP4 topology.

### 5.4 Service Verification

Set the endpoint for the selected deployment, then verify the health endpoint.
Use `http://<A3_IP>:8900` for A3 single-node colocated deployment,
`http://<PROXY_IP>:<PROXY_PORT>` for A3 PD deployment, or
`http://<A2_NODE0_IP>:8000` for the retained A2 deployment.

```shell
export SERVICE_URL="http://<SERVICE_IP>:<SERVICE_PORT>"

curl -sS -o /dev/null -w 'HTTP %{http_code}\n' \
  "$SERVICE_URL/health"
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

```text
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

```text
true
```

## 7 Accuracy Evaluation

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md)
to evaluate the deployed service. No vLLM Ascend task-level accuracy result is
published for this configuration yet. When reporting results, record the
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
- Pipeline parallelism and model runner V2 are not covered by this guide.
- In the A3 PD example, Prefill runs in eager mode. Decode uses
  `FULL_DECODE_ONLY` ACL Graph for the target model and eager execution for the
  DSpark draft model.
- Production performance qualification and task-level accuracy evaluation are
  not complete.
