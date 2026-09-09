# Qwen3.8-Flash-Next

## 1 Introduction

Qwen3.8-Flash-Next is a multimodal Mixture-of-Experts (MoE) model and an experimental preview of the architecture that will underpin Qwen4. Its language model combines Gated DeltaNet and Qwen Sparse Attention (QSA), gated residual connections, Position Learning Enhancement (PLE), and a native Multi-Token Prediction (MTP) head.

This tutorial describes the validated W8A8, language-model-only deployment on Ascend NPUs. The current vLLM Ascend implementation supports **Atlas 800 A3 only**. Atlas 800 A2 and Atlas inference products are not supported by this implementation.

!!! warning

    Automatic Prefix Caching is currently unavailable for Qwen3.8-Flash-Next. Keep `--no-enable-prefix-caching` in the startup command. Do not enable Prefix Caching for this model.

## 2 Supported Features

The following table summarizes the features covered by this tutorial.

| Feature | Status | Notes |
| --- | --- | --- |
| Hardware | Supported | Atlas 800 A3 only |
| Text generation | Supported | Validated with an Ascend-compatible W8A8 checkpoint |
| Tensor parallelism | Supported | TP8 is used in the validated configuration |
| Expert parallelism | Supported | Enabled with `--enable-expert-parallel` |
| MTP speculative decoding | Supported | Uses `qwen3_5_mtp` with three speculative tokens |
| Full Decode ACLGraph | Supported | Uses `FULL_DECODE_ONLY`; the MTP proposer remains eager |
| Function calling | Supported | Uses `qwen3_xml` |
| Reasoning parsing | Supported | Uses `qwen3` |
| Automatic Prefix Caching | **Unsupported** | `--no-enable-prefix-caching` is required |
| Multimodal input | Not covered | The validated command uses `--language-model-only` |

Refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the general model support matrix and the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

## 3 Prerequisites

### 3.1 Hardware

The validated deployment uses eight NPUs on one Atlas 800 A3 node with DP1 × TP8. The example selects devices 8 through 15; replace `ASCEND_RT_VISIBLE_DEVICES` with any eight available A3 device IDs when necessary.

### 3.2 Model Weight

Download the original [Qwen3.8-Flash-Next model weight](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next), then prepare an Ascend-compatible W8A8 checkpoint. The deployment below was validated with the W8A8 checkpoint and therefore requires `--quantization ascend`.

Mount or copy the checkpoint into the container and set `MODEL_PATH` to its local path, for example:

```bash
export MODEL_PATH=/models/Qwen3.8-Flash-Next-w8a8
```

## 4 Environment Preparation

### 4.1 Pre-built Qwen3.8 Image

A pre-built Qwen3.8 image is available in the [vllm-atlas-temp repository](https://quay.io/repository/atlas-ci/vllm-atlas-temp?tab=tags&tag=latest). Select the image that matches the host CPU architecture:

- AArch64: `quay.io/atlas-ci/vllm-atlas-temp:qwen3.8-next-a3-ubuntu-34178549844-2-arm64-temp`
- x86_64: `quay.io/atlas-ci/vllm-atlas-temp:qwen3.8-next-a3-ubuntu-34178549844-2-amd64-temp`

The following example uses the AArch64 image and exposes all 16 devices on an Atlas 800 A3 node so that a group of eight can be selected inside the container. Replace `IMAGE` with the x86_64 image on an x86_64 host.

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

### 4.2 Source Code Installation

Alternatively, build the validated source stack from the following exact components:

- vLLM `v0.26.0`;
- vLLM Ascend v0.26.0 release commit `3047bd476c77c8676231b39e5a1e6cffc9f29b5a`;
- the complete change from [vLLM Ascend PR #15162](https://github.com/vllm-project/vllm-ascend/pull/15162) applied on top of that release commit.

Install vLLM first:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.26.0
VLLM_TARGET_DEVICE=empty pip install --no-deps -v -e .
cd ..
```

Then install vLLM Ascend from the release commit with PR #15162 applied. The pull-request head is a descendant of the specified release commit, so `--ff-only` preserves the exact PR history and fails instead of silently creating an unintended merge if that relationship changes.

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout -b qwen38-flash-next 3047bd476c77c8676231b39e5a1e6cffc9f29b5a
git fetch origin pull/15162/head:pr-15162
git merge --ff-only pr-15162
pip install -v -e .
pip show vllm vllm-ascend
```

For complete environment preparation instructions, refer to [Installation](../../installation.md).

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

The following DP1 × TP8 command was validated with an Ascend-compatible W8A8 checkpoint on Atlas 800 A3. It enables QSA Lightning Indexer and QSA Expand E3, MTP speculative decoding, Function Calling, and reasoning parsing.

```bash
unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export MODEL_PATH=/models/Qwen3.8-Flash-Next-w8a8
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
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
    --gpu-memory-utilization 0.95 \
    --no-enable-prefix-caching \
    --language-model-only \
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
- `--tensor-parallel-size 8` and `--data-parallel-size 1` configure the validated single-DP TP8 topology.
- `--enable-expert-parallel` enables expert parallelism for the MoE layers.
- `--no-enable-prefix-caching` is mandatory because Prefix Caching is currently unavailable for this model.
- `--language-model-only` skips the vision encoder; this tutorial covers text serving only.
- `--enable-auto-tool-choice --tool-call-parser qwen3_xml` enables automatic Function Calling with the Qwen3 XML parser.
- `--reasoning-parser qwen3` separates reasoning from the final answer in the OpenAI-compatible response.
- `--speculative-config` uses the model's MTP head to draft three tokens. `enforce_eager=true` keeps the MTP proposer in eager mode.
- `--compilation-config` enables `FULL_DECODE_ONLY` ACLGraph for decode.
- `fuse_norm_quant=false` selects the validated norm and quantization path for this W8A8 deployment.

When the service is ready, the log contains `Application startup complete`. You can also check the model endpoint:

```bash
curl -sf http://127.0.0.1:8088/v1/models
```

## 6 Functional Verification

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

### 6.2 Function Calling

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

### 6.3 Reasoning Parser and Thinking Control

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

The W8A8 deployment described in Section 5 was validated on GPQA Diamond with the following result:

| Dataset | Metric | Score |
| --- | --- | --- |
| GPQA Diamond | Accuracy | 91.4 |

## 8 Limitations

- Only Atlas 800 A3 is currently supported.
- Automatic Prefix Caching is currently unavailable. Always use `--no-enable-prefix-caching`.
- The validated deployment is language-model-only and uses an Ascend-compatible W8A8 checkpoint.
- Performance data is intentionally not included in this tutorial.
