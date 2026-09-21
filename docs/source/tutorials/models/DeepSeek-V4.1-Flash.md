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

vLLM Ascend supports W8A8 Prefill-Decode (PD) disaggregated deployment on two
Atlas 800 A3 servers. One server runs the Prefill engines and the other runs
the Decode engines. A single A3 server can use Engram host offload as described
below.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md)
for the complete support matrix and the
[Feature Guide](../../user_guide/feature_guide/index.md) for feature
configuration.

The PD configuration in this guide uses W8A8 weights, INT8 Engram storage,
DP4/TP4 on the Prefill node, DP8/TP2 on the Decode node, DSpark speculative
decoding, and `FULL_DECODE_ONLY` ACL Graph on Decode. It uses model runner V1.

## 3 Prerequisites

### 3.1 Model Weights and Hardware

The official DeepSeek-V4.1-Flash checkpoint is available from
[Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and
[ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash).

The Ascend W8A8 checkpoint used by this guide will be published as
[Eco-Tech/DeepSeek-V4.1-Flash-w8a8](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8)
on ModelScope. It includes the DSpark draft parameters and INT8 Engram tables.
After the checkpoint is available, download it to the same absolute path on
every server; the examples use `<YOUR_MODEL_PATH>`.

Alternatively, use [ModelSlim](https://github.com/Ascend/msmodelslim) to
prepare a ModelSlim-compatible W8A8 checkpoint from the official weights.

The 1P1D deployment requires two Atlas 800 A3 servers. Each server has 8 NPUs
with 128GB memory per NPU and exposes 16 logical devices to the container.

Store the checkpoint in a shared directory or copy it to the same absolute
path on every server.

### 3.2 Verify Multi-node Communication

Before deployment, follow
[Verify Multi-node Communication](../../getting_started/installation.md#installation-multi-node-interconnect).
All servers must be able to communicate through the selected network
interfaces, and the service ports must not be blocked.

## 4 Installation

### 4.1 Docker Image Installation

An A3 server exposes 16 logical devices. Run this command on both A3 servers.

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

Change `MODEL_ROOT` if the checkpoint is stored elsewhere. Keep the same
absolute path inside and outside every container.

### 4.2 Source Code Installation

To build from source, follow the
[software environment installation guide](../../getting_started/installation.md#installation-software-environment)
and use the `main` branch with the matching vLLM revision recorded in
`.github/vllm-main-verified.commit`.

## 5 Online Service Deployment

### 5.1 A3 1P1D PD Separation Deployment

This example uses two Atlas 800 A3 servers. The Prefill node runs four DP
ranks with TP4 (DP4/TP4), and the Decode node runs eight DP ranks with TP2
(DP8/TP2). Both layouts consume all 16 logical devices on their respective
servers. Mooncake transfers KV cache from the Prefill engines to the Decode
engines.

#### 5.1.1 Prepare the DP Launcher

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
    parser.add_argument("--dp-rpc-port", type=str, default=12345)
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

#### 5.1.2 Start the Prefill Node

On the Prefill node, save the following script as `run_dp_template.sh`. Replace
`xx.xx.xx.1` and `xxxx` with the Prefill node's service IP and network
interface. Change `MODEL_PATH` if the checkpoint is stored elsewhere.

```shell
#!/usr/bin/env bash

unset https_proxy
unset http_proxy
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"

nic_name="xxxx"       # for example, enp67s0f0np0
local_ip=xx.xx.xx.1    # Prefill node service IP
MODEL_PATH="/mnt/share/DeepSeek-V4.1-Flash-W8A8-no-wq-wkv"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
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
    --served-model-name dsv41 \
    --max-model-len 150000 \
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

#### 5.1.3 Start the Decode Node

On the Decode node, save the following script as `run_dp_template.sh`. Replace
`xx.xx.xx.2` and `xxxx` with the Decode node's service IP and network
interface. Use the same `MODEL_PATH` as the Prefill node.

```shell
#!/usr/bin/env bash

unset https_proxy
unset http_proxy
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"

nic_name="xxxx"       # for example, enp67s0f0np0
local_ip=xx.xx.xx.2    # Decode node service IP
MODEL_PATH="/mnt/share/DeepSeek-V4.1-Flash-W8A8-no-wq-wkv"

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1800
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
    --max-model-len 150000 \
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

#### 5.1.4 Deploy the PD Proxy

After all Prefill and Decode engines are ready, deploy the proxy as described
in [Prefill-Decode Disaggregation (DeepSeek)](../features/pd_disaggregation_mooncake_multi_node.md).
Configure the proxy with Prefill endpoints `xx.xx.xx.1:7100` through
`xx.xx.xx.1:7103` and Decode endpoints `xx.xx.xx.2:7100` through
`xx.xx.xx.2:7107`.

#### 5.1.5 Key Parameter Descriptions

- `--data-parallel-size` and `--tensor-parallel-size` define DP4/TP4 on
  Prefill and DP8/TP2 on Decode. Their product must be 16 on each A3 node.
- `--data-parallel-address` and `--data-parallel-rpc-port` coordinate DP ranks
  within one node group. The Prefill and Decode groups use their own node IPs;
  port `12321` can be reused because the groups run on different hosts.
- `--vllm-start-port 7100` assigns API ports `7100-7103` on Prefill and
  `7100-7107` on Decode. These endpoints must be reachable by the PD proxy.
- `MooncakeHybridConnector` transfers KV cache between the two node groups.
  `kv_role` must be `kv_producer` on Prefill and `kv_consumer` on Decode;
  `kv_port` must be reachable and must not conflict with another service.
- `kv_connector_extra_config` must match the actual global layouts on both
  sides: Prefill DP4/TP4 and Decode DP8/TP2. Keep the values identical in the
  Prefill and Decode commands.
- `--enforce-eager` keeps Prefill in eager mode. Decode uses
  `FULL_DECODE_ONLY` graph mode, while `enforce_eager` inside the DSpark config
  applies only to speculative draft execution.
- `--engram-config '{"cpu_offload":true}'` stores Engram tables in host memory.
  Ensure that both nodes have enough host memory and use lazy safetensors
  loading to avoid materializing the complete table on every rank.
- `--enable-prefix-caching` is enabled only on Prefill. Decode disables prefix
  caching and enables `recompute_scheduler_enable` so missing KV cache can be
  recomputed by Prefill.
- `HCCL_IF_IP` and the socket interface variables must select the same
  high-speed service network used by the configured IP addresses. The HCCL,
  RPC, engine, Mooncake, and proxy ports must be allowed by the host firewall.

Wait until every engine finishes loading weights and Decode finishes graph
capture. A successful startup includes output similar to:

```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 5.2 Service Verification

Set the proxy address, then verify the health endpoint:

```shell
export SERVICE_URL="http://<PROXY_IP>:<PROXY_PORT>"

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

The response must contain a model entry whose `id` is `dsv41`.

### 5.3 Single A3 with Engram Host Offload

Keep the INT8 Engram weights and their scale tensors available in the
checkpoint. `--safetensors-load-strategy lazy`
is required to avoid eagerly materializing the entire table on each rank.

For a single A3, use TP8/DP2/EP16 across all 16 logical devices with both
DP replicas local (`--data-parallel-size 2 --data-parallel-size-local 2`).
Keep model runner V1, `FULL_DECODE_ONLY`, and DSpark with eager draft execution.
Use INT8 Engram tables and turn the offload on through vLLM's Engram config.
This needs a vLLM that provides `--engram-config`; without it the tables stay
on the device:

```bash
--engram-config '{"cpu_offload": true, "dp_shared_memory": true}'
```

With `cpu_offload` the shard stays in host memory, is registered with
`aclrtHostRegisterV2`, and the NPU gather kernel reads it through the device
address `aclrtHostGetDevicePointer` returns, so the offloaded table needs
neither an H2D copy nor a host-side gather.

Start with 4 sequences per DP replica, 512 batched tokens, 131072 model
length, and 1 GiB of KV cache per rank, with prefix caching disabled.
Ensure enough host RAM for all compressed Engram shards and runtime memory;
CPU/NUMA page migration can add several minutes to startup.

This configuration passed model loading, decode graph capture, natural-text
requests, and mixed-length concurrent request smoke tests. These checks do
not establish dataset accuracy or performance. Other Engram storage formats
and model runner V2 are not covered by this smoke validation.

## 6 Functional Verification

### 6.1 Text Request

```shell
export SERVICE_URL="http://<PROXY_IP>:<PROXY_PORT>"

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

Set `IMAGE_URL` to an HTTP(S) image URL reachable from the Prefill node, and send a
multimodal request:

```shell
export IMAGE_URL="<YOUR_IMAGE_URL>"
export SERVICE_URL="http://<PROXY_IP>:<PROXY_PORT>"

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

The values in Section 5.1 are a validated starting point rather than globally
optimal settings. Tune `--max-num-seqs`, `--max-num-batched-tokens`, and
`--gpu-memory-utilization` together for the target input length, image sizes,
output length, and concurrency. Keep the documented Prefill DP4/TP4 and Decode
DP8/TP2 layouts, `--block-size 128`, and Decode `FULL_DECODE_ONLY` mode until
an alternative configuration has been validated.

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

- The documented PD deployment uses two Atlas 800 A3 servers and an Ascend
  W8A8 checkpoint with INT8 Engram storage.
- Colocated multi-node deployment, A2 deployment, pipeline parallelism, and
  model runner V2 are not covered by this guide.
- DSpark draft execution runs in eager mode while the target model uses
  `FULL_DECODE_ONLY` ACL Graph.
- Production performance qualification and task-level accuracy evaluation are
  not complete.
