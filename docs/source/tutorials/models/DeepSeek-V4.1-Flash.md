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

vLLM Ascend supports W8A8 colocated deployment on either two Atlas 800 A3
servers or four Atlas 800 A2 servers. Prefill-Decode disaggregation and Engram
host offloading are not covered by this guide.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md)
for the complete support matrix and the
[Feature Guide](../../user_guide/feature_guide/index.md) for feature
configuration.

The configuration in this guide has been validated with W8A8 weights, INT8
Engram storage, TP8/DP4/EP32, DSpark speculative decoding, and
`FULL_DECODE_ONLY` ACL Graph. It uses model runner V1 and supports automatic
prefix caching.

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

Alternatively, use [ModelSlim](https://gitcode.com/Ascend/msmodelslim) to
prepare a ModelSlim-compatible W8A8 checkpoint from the official weights.

Use one of the following hardware configurations:

- **A3 series**: two Atlas 800 A3 servers. Each server has 8 NPUs with 128GB
  memory per NPU and exposes 16 logical devices to the container.
- **A2 series**: four Atlas 800 A2 servers. Each server has 8 NPUs with 64GB
  memory per NPU and exposes 8 devices to the container.

Store the checkpoint in a shared directory or copy it to the same absolute
path on every server.

### 3.2 Verify Multi-node Communication

Before deployment, follow
[Verify Multi-node Communication](../../getting_started/installation.md#installation-multi-node-interconnect).
All servers must be able to communicate through the selected network
interfaces, and the service ports must not be blocked.

## 4 Installation

### 4.1 Docker Image Installation

Select the tab for the target hardware. A2 and A3 use separate validation
images.

=== "A3 series"

    An A3 server exposes 16 logical devices. Run this command on both A3
    servers.

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

=== "A2 series"

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

### 5.1 Multi-Node Colocated Deployment

The A3 and A2 configurations use the same global DP4/TP8/EP32 topology. A3
places two local DP ranks on each of two servers; A2 places one local DP rank
on each of four servers.

Select the tab for the target hardware. In each script, change `NODE_RANK`,
`NODE0_IP`, `LOCAL_IP`, `NIC_NAME`, and `MODEL_PATH`. Node 0 exposes the API;
every other node is a headless worker.

=== "A3 series"

    Run this script on both A3 servers. Set `NODE_RANK=0` on Node 0 and
    `NODE_RANK=1` on Node 1.

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
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

    if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
      export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
    fi

    DP_START_RANK=$((NODE_RANK * 2))
    HEADLESS_ARGS=()
    if [[ "$NODE_RANK" != "0" ]]; then
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
      --additional-config '{"enable_engram":true,"engram_storage":"int8","enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
      --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    ```

=== "A2 series"

    Run this script on all four A2 servers. Set `NODE_RANK` to `0`, `1`, `2`,
    or `3` on the corresponding node.

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
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
      export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
    fi

    HEADLESS_ARGS=()
    if [[ "$NODE_RANK" != "0" ]]; then
      HEADLESS_ARGS+=(--headless)
    fi

    vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port 8000 \
      "${HEADLESS_ARGS[@]}" \
      --data-parallel-address "$NODE0_IP" \
      --data-parallel-rpc-port 13399 \
      --data-parallel-size 4 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank "$NODE_RANK" \
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
      --additional-config '{"enable_engram":true,"engram_storage":"int8","enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
      --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    ```

Start Node 0 first and then the remaining nodes. The global topology is
DP4/TP8/EP32 in both configurations:

- **A3 series**: two servers, local DP2 per server, and 16 visible logical
  devices per server.
- **A2 series**: four servers, local DP1 per server, and 8 visible devices per
  server.

Each DP rank uses eight devices through TP8. Only Node 0 exposes the API
endpoint.

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
published for this configuration yet. When reporting results, record the
checkpoint, prompt encoder, reasoning effort, sampling parameters, dataset
version, and whether DSpark is enabled.

## 8 Performance Evaluation

Refer to the
[AISBench performance evaluation guide](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation)
or the [vLLM benchmark guide](https://docs.vllm.ai/en/latest/benchmarking/).
No production performance baseline is published for this configuration.

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

- The documented deployment uses either two Atlas 800 A3 servers or four
  Atlas 800 A2 servers and an Ascend W8A8 checkpoint with INT8 Engram storage.
- Prefill-Decode disaggregation, pipeline parallelism, and model runner V2 are
  not supported by this guide.
- DSpark draft execution runs in eager mode while the target model uses
  `FULL_DECODE_ONLY` ACL Graph.
- Production performance qualification and task-level accuracy evaluation are
  not complete.
