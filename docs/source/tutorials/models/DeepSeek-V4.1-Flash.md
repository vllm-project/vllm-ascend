# DeepSeek-V4.1-Flash (Experimental)

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

Support on vLLM Ascend is experimental. This guide documents the validated
W8A8 colocated deployment on two Atlas 800 A3 servers. Prefill-Decode
disaggregation and the full one-million-token context are not covered by this
guide.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md)
for the complete support matrix and the
[Feature Guide](../../user_guide/feature_guide/index.md) for feature
configuration.

The configuration in this guide has been validated with W8A8 weights, INT8
Engram storage, TP8/DP4/EP32, DSpark speculative decoding, and
`FULL_DECODE_ONLY` ACL Graph. It uses model runner V1 and disables automatic
prefix caching.

## 3 Prerequisites

### 3.1 Model Weights and Hardware

The official DeepSeek-V4.1-Flash checkpoint is available from
[Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and
[ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash).

The deployment command below requires an Ascend W8A8 checkpoint derived from
the official weights, including the DSpark draft parameters and INT8 Engram
tables. Use [ModelSlim](https://gitcode.com/Ascend/msmodelslim) to prepare a
ModelSlim-compatible checkpoint. Record its absolute path on both servers;
the examples use `<YOUR_MODEL_PATH>`.

The validated deployment requires two Atlas 800 A3 servers (128GB × 8 NPUs
per server). Store the checkpoint in a shared directory or copy it to the same
absolute path on both servers.

### 3.2 Verify Multi-node Communication

Before deployment, follow
[Verify Multi-node Communication](../../getting_started/installation.md#installation-multi-node-interconnect).
The two servers must be able to communicate through the selected network
interfaces, and the service ports must not be blocked.

## 4 Installation

### 4.1 Docker Image Installation

Use the A3 image built from the `main` branch after DeepSeek-V4.1 support is
merged:

```shell
export IMAGE=quay.io/ascend/vllm-ascend:nightly-main-a3
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

Run this command on both servers. Change `MODEL_ROOT` if the checkpoint is
stored elsewhere, and keep the same absolute path inside and outside the
container.

### 4.2 Source Code Installation

To build from source, follow the
[software environment installation guide](../../getting_started/installation.md#installation-software-environment)
and use the `main` branch with the matching vLLM revision recorded in
`.github/vllm-main-verified.commit`.

## 5 Online Service Deployment

### 5.1 Two-Node Colocated Deployment

Run the following script on both servers. Change only `NODE_RANK`, `NODE0_IP`,
`LOCAL_IP`, `NIC_NAME`, and `MODEL_PATH`. Set `NODE_RANK=0` on the server that
exposes the API and `NODE_RANK=1` on the headless worker.

```bash
#!/usr/bin/env bash
set -euo pipefail

# Node 0 uses NODE_RANK=0; Node 1 uses NODE_RANK=1.
NODE_RANK=0
NODE0_IP="<NODE0_IP>"
LOCAL_IP="<LOCAL_IP>"
NIC_NAME="<NETWORK_INTERFACE>"
MODEL_PATH="<YOUR_MODEL_PATH>"

export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=7200
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_RPC_TIMEOUT=1800000
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1

if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
  export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
fi

DP_START_RANK=$((NODE_RANK * 2))
HEADLESS_ARGS=()
if [[ "$NODE_RANK" == "1" ]]; then
  HEADLESS_ARGS+=(--headless)
fi

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  "${HEADLESS_ARGS[@]}" \
  --data-parallel-address "$NODE0_IP" \
  --data-parallel-rpc-port 13399 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank "$DP_START_RANK" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --served-model-name deepseek-v41 \
  --max-model-len 131072 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy \
  --quantization ascend \
  --additional-config '{"enable_engram":true,"engram_storage":"int8","enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"${MODEL_PATH}\",\"num_speculative_tokens\":5,\"enforce_eager\":true}" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Start Node 0 first and then Node 1. The global topology is DP4/TP8/EP32: each
server hosts two local DP ranks, and each rank uses eight logical devices.
Only Node 0 exposes the API endpoint.

Wait until every DP engine finishes loading weights and graph capture. A
successful startup includes output similar to:

```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 5.2 Service Verification

On Node 0, verify the health endpoint:

```shell
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' \
  http://127.0.0.1:8000/health
```

Expected output:

```text
HTTP 200
```

Then verify that the configured model is available:

```shell
curl -sS http://127.0.0.1:8000/v1/models | \
  jq '{object, models: [.data[] | {id, object}]}'
```

The response must contain a model entry whose `id` is `deepseek-v41`.

## 6 Functional Verification

### 6.1 Text Request

```shell
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v41",
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

Set `IMAGE_URL` to an HTTP(S) image URL reachable from Node 0, and send a
multimodal request:

```shell
export IMAGE_URL="<YOUR_IMAGE_URL>"

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"deepseek-v41\",
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
published for this experimental configuration yet. When reporting results,
record the checkpoint, prompt encoder, reasoning effort, sampling parameters,
dataset version, and whether DSpark is enabled.

## 8 Performance Evaluation

Refer to the
[AISBench performance evaluation guide](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation)
or the [vLLM benchmark guide](https://docs.vllm.ai/en/latest/benchmarking/).
No production performance baseline is published for this experimental
configuration.

## 9 Performance Tuning

The values in Section 5.1 are a validated starting point rather than globally
optimal settings. Tune `--max-num-seqs`, `--max-num-batched-tokens`, and
`--gpu-memory-utilization` together for the target input length, image sizes,
output length, and concurrency. Keep the documented DP4/TP8/EP32 topology,
`--block-size 128`, and `FULL_DECODE_ONLY` mode until an alternative
configuration has been validated.

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

- The documented deployment uses two Atlas 800 A3 servers and an Ascend W8A8
  checkpoint with INT8 Engram storage.
- The validated maximum model length is 131072 tokens; the official model's
  one-million-token context is not validated in this configuration.
- Prefill-Decode disaggregation, pipeline parallelism, and model runner V2 are
  not supported by this guide.
- DSpark draft execution runs in eager mode while the target model uses
  `FULL_DECODE_ONLY` ACL Graph.
- Production performance qualification and task-level accuracy evaluation are
  not complete.
