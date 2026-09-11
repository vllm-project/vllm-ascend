# Qwen3.8-Flash-Next

## 1 Introduction

Qwen3.8-Flash-Next is a multimodal Mixture-of-Experts (MoE) model and an experimental preview of the architecture that will underpin Qwen4. Its language model combines Gated DeltaNet and Qwen Sparse Attention (QSA), gated residual connections, Position Learning Enhancement (PLE), and a native Multi-Token Prediction (MTP) head.

This tutorial describes the W8A8 deployment on Atlas 800 A3 and Ascend 950DT. Text and multimodal input have been validated on both hardware platforms.

!!! warning

    Automatic Prefix Caching is experimental for Qwen3.8-Flash-Next. Enable it
    with `--enable-prefix-caching --mamba-cache-mode align` and revalidate NPU
    memory capacity for the target context length and concurrency.

## 2 Supported Features

The following table summarizes the features covered by this tutorial.

| Feature | Status | Notes |
| --- | --- | --- |
| Hardware | Supported | Atlas 800 A3 and Ascend 950DT |
| Text generation | Supported | Validated with an Ascend-compatible W8A8 checkpoint on A3 and 950DT |
| Tensor parallelism | Supported | A3 uses TP8; 950DT uses TP4 in the examples below |
| Expert parallelism | Supported | Enabled with `--enable-expert-parallel` |
| MTP speculative decoding | Supported | Uses `qwen3_5_mtp` with three speculative tokens |
| Full Decode ACLGraph | Supported | Uses `FULL_DECODE_ONLY`; the MTP proposer remains eager |
| Function calling | Supported | Uses `qwen3_xml` |
| Reasoning parsing | Supported | Uses `qwen3` |
| Automatic Prefix Caching | Experimental | Uses aligned GDN and PLE state checkpoints and requires additional NPU memory |
| Multimodal input | Supported | Image-and-text input has been validated on A3 and 950DT |

Refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the general model support matrix and the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

## 3 Prerequisites

### 3.1 Hardware

The A3 deployment uses eight NPUs on one node with DP1 × TP8.

The 950DT deployment uses four NPUs on one node with DP1 × TP4.

### 3.2 Model Weight

Download the Ascend-compatible quantized checkpoint for the target hardware. Both checkpoints require `--quantization ascend`.

| Hardware | Quantization | ModelScope checkpoint |
| --- | --- | --- |
| Atlas 800 A3 | W8A8 | [Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp) |
| Ascend 950DT | W8A8 MXFP8 | [Eco-Tech/Qwen3.8-Flash-Next-w8a8-mxfp8-mtp](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w8a8-mxfp8-mtp) |

Mount or copy the selected checkpoint into the container. For example, set the A3 checkpoint path as follows:

```bash
export MODEL_PATH=/models/Qwen3.8-Flash-Next-w8a8
```

## 4 Environment Preparation

Only pre-built images are supported by this tutorial. Source installation is not provided.

### 4.1 A3 Image and Container

A pre-built Qwen3.8 A3 image is available in the [vllm-atlas-temp repository](https://quay.io/repository/atlas-ci/vllm-atlas-temp?tab=tags&tag=latest). Select the image that matches the host CPU architecture:

- AArch64: `quay.io/atlas-ci/vllm-atlas-temp:qwen3.8-next-a3-ubuntu-34178549844-2-arm64-temp`
- x86_64: `quay.io/atlas-ci/vllm-atlas-temp:qwen3.8-next-a3-ubuntu-34178549844-2-amd64-temp`

The following example uses the AArch64 image and exposes all 16 devices on an Atlas 800 A3 node. The service command later uses the first eight devices by default. Replace `IMAGE` with the x86_64 image on an x86_64 host.

```bash
export IMAGE=quay.io/atlas-ci/vllm-atlas-temp:qwen3.8-next-a3-ubuntu-34178549844-2-arm64-temp

docker pull "$IMAGE"

docker run --rm \
    --name vllm-ascend-qwen38-flash-next \
    --shm-size=1g \
    --net=host \
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
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /path/to/models:/models \
    -it "$IMAGE" bash
```

After entering the container, verify the installation:

```bash
pip show vllm vllm-ascend
```

### 4.2 950DT Image and Container

The following example uses the ARM64 image for Ascend 950DT:

```bash
export IMAGE=quay.io/atlas-ci/vllm-atlas-temp:qwen3.8-next-a5-ubuntu-34178549844-2-arm64-temp

docker pull "$IMAGE"

docker run --runtime=runc \
    -u root \
    -it -d \
    --name vllm-ascend-qwen38-flash-next-950dt \
    --net=host \
    --privileged=true \
    --shm-size=500g \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/ummu \
    --device=/dev/uburma \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /root/host:/root/host \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /var/log/npu:/usr/slog \
    -v /mnt:/mnt \
    -v /data:/data \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
    -v /usr/lib64:/usr/lib64 \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /home:/home \
    -v /etc/hixlep:/etc/hixlep \
    "$IMAGE" bash
```

The command removes the duplicate `npu-smi` mount from the original example and fixes the `/home` and `/etc/hixlep` mount syntax.

## 5 Online Service Deployment

### 5.1 A3 Single-Node Online Deployment

The following DP1 × TP8 command was validated with an Ascend-compatible W8A8 checkpoint on Atlas 800 A3. It enables QSA Lightning Indexer and QSA Expand E3, MTP speculative decoding, Function Calling, and reasoning parsing.

```bash
unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export MODEL_PATH=/models/Qwen3.8-Flash-Next-w8a8
export VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER=1
export VLLM_ASCEND_ENABLE_QSA_E3V=1
unset VLLM_ASCEND_FORCE_QSA_REFERENCE

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8088 \
    --served-model-name qwen3.8-flash-next \
    --trust-remote-code \
    --quantization ascend \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --enable-expert-parallel \
    --max-model-len 135168 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.93 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --compilation-config '{"cudagraph_capture_sizes":[4,8,12,16,20,24,28,32],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false}}'
```

Key parameter descriptions:

- `VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER=1` enables the fused QSA Lightning Indexer path.
- `VLLM_ASCEND_ENABLE_QSA_E3V=1` enables the fused QSA Expand E3 path.
- `--quantization ascend` loads the Ascend-compatible W8A8 checkpoint.
- `--tensor-parallel-size 8` and `--data-parallel-size 1` configure the A3 single-DP TP8 topology.
- `--enable-expert-parallel` enables expert parallelism for the MoE layers.
- `--enable-prefix-caching --mamba-cache-mode align` enables Prefix Caching with aligned GDN and PLE state checkpoints.
- The validated command does not set `--language-model-only`, so the vision encoder remains enabled and the service accepts both text and multimodal requests. For a text-only deployment, you may add `--language-model-only` to skip loading the vision encoder.
- `--enable-auto-tool-choice --tool-call-parser qwen3_xml` enables automatic Function Calling with the Qwen3 XML parser.
- `--reasoning-parser qwen3` separates reasoning from the final answer in the OpenAI-compatible response.
- `--speculative-config` uses the model's MTP head to draft three tokens. `enforce_eager=true` keeps the MTP proposer in eager mode.
- `--compilation-config` enables `FULL_DECODE_ONLY` ACLGraph for decode.
- `fuse_norm_quant=false` selects the validated norm and quantization path for this W8A8 deployment.

When the service is ready, the log contains `Application startup complete`. You can also check the model endpoint:

```bash
curl -sf http://127.0.0.1:8088/v1/models
```

### 5.2 950DT Single-Node Online Deployment

The following command starts a DP1 × TP4 service on Ascend 950DT. It uses the 950DT MXFP8 checkpoint and supports both text and multimodal input.

```bash
unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export VLLM_SERVER_DEV_MODE=1
export SOC_VERSION=ascend950dt_9582
export MODEL_PATH=/models/Qwen3.8-Flash-Next-w8a8-mxfp8

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8089 \
    --served-model-name qwen38-flash-next-950dt \
    --trust-remote-code \
    --quantization ascend \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --enable-expert-parallel \
    --max-model-len 135168 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --compilation-config '{"cudagraph_capture_sizes":[4,8,12,16,20,24,28,32],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false}}'
```

- `VLLM_SERVER_DEV_MODE=1` enables the server development mode required by this 950DT setup, including cache-clearing support.
- `SOC_VERSION=ascend950dt_9582` selects the 950DT SoC target used by the image.
- `--language-model-only` is optional. Add it only when a text-only service is desired and the vision encoder should not be loaded.
- `--enable-prefix-caching --mamba-cache-mode align` enables Prefix Caching.
  Revalidate memory headroom on the target 950DT deployment rather than reusing
  the A3 memory-utilization observation as a fixed value.

Prefix Caching stores aligned GDN and PLE state checkpoints in addition to the
reusable attention cache, so it consumes additional NPU memory. The exact
overhead depends on the maximum context length, concurrency, and cache layout.
If memory is tight, reduce `--gpu-memory-utilization`, `--max-model-len`, or
`--max-num-seqs`, and then repeat the real-weight capacity validation. In one A3
DP1 × TP8 correctness validation, `--gpu-memory-utilization 0.95` left
insufficient QSA workspace headroom and caused an out-of-memory error, while
`0.93` passed. This is a workload-specific A3 observation, not a universal
recommended value or a 950DT validation result.

The minimal 950DT command above does not enable automatic Function Calling or reasoning parsing. To use the verification requests in Sections 6.3 and 6.4, add:

```text
--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
```

When the service is ready, verify the 950DT endpoint with:

```bash
curl -sf http://127.0.0.1:8089/v1/models
```

## 6 Functional Verification

The requests below target the A3 example at port `8088` with served model name `qwen3.8-flash-next`. For 950DT, use port `8089` and model name `qwen38-flash-next-950dt`. The multimodal request in Section 6.2 has been validated on both hardware platforms.

### 6.1 Basic Chat Completion

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {"role": "user", "content": "Who are you?"}
        ],
        "max_tokens": 1024,
        "temperature": 0
    }'
```

Expected result: the service returns HTTP 200 and a non-empty `choices` field.

### 6.2 Multimodal Chat Completion

The validated deployment accepts image-and-text input. Replace `<IMAGE_URL>` with an image URL accessible from the serving environment:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "<IMAGE_URL>"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0
    }'
```

Expected result: the service returns HTTP 200 and describes the supplied image in `choices[0].message.content`.

### 6.3 Function Calling

Function Calling requires the following startup options, which are already included in Section 5:

```text
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

Send a request containing the available tools and set `tool_choice` to `auto`:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Beijing?"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0
    }'
```

When the model chooses the function, the parsed result appears in `choices[0].message.tool_calls`. The caller must execute the function and send its result back to the model in a follow-up message.

### 6.4 Reasoning Parser and Thinking Control

Reasoning parsing requires the following startup option, which is already included in Section 5:

```text
--reasoning-parser qwen3
```

Qwen3.8-Flash-Next uses thinking mode by default. To request reasoning explicitly, set `enable_thinking` to `true` (or omit it). The parser returns the reasoning separately from the final answer:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {
            "enable_thinking": true
        }
    }'
```

The final answer is returned in `choices[0].message.content`, while the parsed reasoning is returned separately in the reasoning field of the response.

To disable thinking and request a direct answer, use the validated non-thinking request:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {
            "enable_thinking": false
        }
    }'
```

## 7 Accuracy Evaluation

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for evaluation setup and usage.

The A3 W8A8 deployment described in Section 5.1 was validated on GPQA Diamond with the following result. This score is an A3 result; no 950DT accuracy result is reported here.

| Hardware | Dataset | Metric | Score |
| --- | --- | --- | --- |
| Atlas 800 A3 | GPQA Diamond | Accuracy | 91.4 |

## 8 Limitations

- Atlas 800 A3 and Ascend 950DT are currently supported.
- Automatic Prefix Caching is experimental and consumes additional NPU memory.
  Use `--enable-prefix-caching --mamba-cache-mode align` and validate memory
  capacity for the target workload.
- Text and multimodal input have been validated on A3 and 950DT. Add `--language-model-only` only for an optional text-only deployment.
- The GPQA Diamond score in this tutorial was measured on A3 and must not be treated as a 950DT accuracy result.
