# Qwen3.8-Flash-Next (Experimental)

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.
    
    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

Qwen3.8-Flash-Next is a multimodal Mixture-of-Experts (MoE) model and an experimental preview of the architecture that will underpin Qwen4. Its language model combines Gated DeltaNet and Qwen Sparse Attention (QSA), gated residual connections, Position Learning Enhancement (PLE), and a native Multi-Token Prediction (MTP) head.

The current version supports only Atlas A3 series hardware. Atlas A2 series hardware and Ascend 950DT and 950PR products are not yet supported and will be enabled progressively in future releases. This tutorial describes the W8A8 deployment on Atlas 800 A3. Text and multimodal input have been validated on Atlas 800 A3.

This document is validated and written based on **vLLM-Ascend 0.26.0rc release**. The current model (Qwen3.8-Flash-Next) is first supported in this version.

!!! warning

    Automatic Prefix Caching is experimental for Qwen3.8-Flash-Next. Enable it
    with `--enable-prefix-caching --mamba-cache-mode align` and revalidate NPU
    memory capacity for the target context length and concurrency.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_features.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

## 3 Prerequisites

### 3.1 Model Weight

The following model weights are available:

- `Qwen3.8-Flash-Next-w8a8-mtp` (Quantized version): requires 1 Atlas 800 A3 (64GB × 8) node. [https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp)

It is recommended to download the model weight to the shared directory of multiple nodes, such as /root/.cache/.

## 4 Installation

### 4.1 Docker Image Installation

The pre-built images listed below have been validated for this tutorial.

A pre-built Qwen3.8 A3 image is available in the [vllm-ascend repository](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=latest). Select the image that matches the host operating system:

- openEuler: `quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler`
- Ubuntu: `quay.io/ascend/vllm-ascend:qwen3.8-next-a3`

The following example uses the AArch64 image and exposes 8 devices on an Atlas 800 A3 node. The service command in Section 5.1 uses the first eight devices.

=== "A3 series"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:qwen3.8-next-a3
    export NAME=vllm-ascend-qwen38-flash-next

    docker pull "$IMAGE"

    docker run --rm \
        --name "$NAME" \
        --shm-size=16g \
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

After entering the container, verify that vLLM and vLLM-Ascend can be imported:

```shell
python -c "import vllm, vllm_ascend; print('vllm and vllm-ascend are ready')"
```

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md#installation-existing-cann-install).

If you want to deploy a multi-node environment, you need to set up the environment on each node.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

For single-node online deployment on Atlas 800 A3, the following DP1 × TP8 command was validated with an Ascend-compatible W8A8 checkpoint. It enables QSA Lightning Indexer, QSA Expand E3, MTP speculative decoding, Function Calling, and reasoning parsing.

Before starting the service:

- Replace `MODEL_PATH` with the directory recorded when downloading the model weight.
- Revalidate memory capacity before changing the context length, concurrency, or memory utilization.

=== "A3 series"

    The following example is for Atlas 800 A3.

    ```bash
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export MODEL_PATH=/models/Qwen3.8-Flash-Next-w8a8-mtp
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

### 5.2 Model Runner V2 with DFlash

Model Runner V2 can use a separate DFlash draft instead of the built-in MTP
head. For the [Flash-Next DFlash2 checkpoint](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2),
keep `method` set to `dflash`; the checkpoint architecture selects DFlash2
automatically. The target supplies the completed decoder-layer features listed
in the draft configuration, averaged over the HyperConnection streams.

The following A3 TP8/EP8 configuration is a validation recipe. This combination
still requires NPU accuracy and performance validation; the checkpoint author's
SGLang results do not establish Ascend support or expected acceptance rates.
Install this branch together with the vLLM revision required by the parent PR
(`a98247ab4db686ee03c66d5feb3c761e52a2f8ab`).

```shell
export VLLM_USE_V2_MODEL_RUNNER=1
export MODEL_PATH=/models/Qwen3.8-Flash-Next-w8a8-mtp
export DRAFT_MODEL_PATH=/models/Qwen3.8-Flash-Next-DFlash2

vllm serve "$MODEL_PATH" \
    --served-model-name qwen3.8-flash-next \
    --trust-remote-code \
    --quantization ascend \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --block-size 128 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]}' \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_MODEL_PATH\",\"num_speculative_tokens\":3}" \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false}}'
```

Download the draft checkpoint separately into `DRAFT_MODEL_PATH`. It must match
the target's vocabulary, embeddings, and LM head. DFlash2 requires an
unquantized target LM head. With `FULL_DECODE_ONLY`, omitting `enforce_eager`
lets both the target and draft use ACL graphs. The draft backbone and candidate
selector skip Dynamo compilation; their NPU operations are captured by the
draft's full ACL graph, including candidate TopK, selection and sampling.
Set `enforce_eager=true` to run the draft eagerly when comparing results.
This graph path still requires validation on the intended NPU workload.
Capture sizes must be multiples of
`num_speculative_tokens + 1`. Pipeline parallelism with auxiliary features is
not supported by this adaptation. This recipe uses a single serving instance
without a KV-transfer connector; disaggregated prefill/decode is outside its
validation scope.

If an older revision fails while registering `QKNormRopeFusionPass` with
`assert rope_dim <= head_dim`, update this branch. The fusion pass now skips
incompatible target/draft dimension combinations while retaining ACL graph
capture. As a workaround for older revisions, merge `"fuse_qknorm_rope":false`
into `ascend_compilation_config` in the existing `--additional-config` value;
this only disables that fusion and does not require eager execution.

Before deployment, compare greedy token IDs against a target-only baseline
under eager execution, then repeat with target graph replay and asynchronous
scheduling. Include chunked prefill, mixed request lengths, padded decode
batches, and request reuse. Measure draft acceptance and throughput on the same
workload; do not assume that a working draft improves performance. Validate
image inputs separately when using the multimodal target.

The opt-in eight-card regression compares greedy token IDs with a target-only
baseline with eager execution, target graphs plus an eager draft, and graphs
for both models. It exercises chunked prefill and request reuse, and checks
that every TP rank actually captured and replayed the draft graph:

```shell
QWEN38_DFLASH_TARGET_MODEL="$MODEL_PATH" \
QWEN38_DFLASH_DRAFT_MODEL="$DRAFT_MODEL_PATH" \
pytest -sv tests/e2e/pull_request/eight_card/model_runner_v2/test_qwen38_dflash.py
```

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt. The `chat/completions` API is supported. Use the `--served-model-name` you configured (`qwen3.8-flash-next` for `Qwen3.8-Flash-Next`).

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

Expected result: the service returns HTTP 200 OK.

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

Expected result: the service returns HTTP 200 OK and describes the supplied image in `choices[0].message.content`.

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

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for evaluation setup and usage.

2. After execution, you can get the result. Here are the results of `Qwen3.8-Flash-Next` in `vllm-ascend:v0.26.0rc` for reference only.

The A3 W8A8 deployment described in Section 5.1 was validated on GPQA Diamond with the following result.

| Hardware | Dataset | Metric | Score |
| --- | --- | --- | --- |
| Atlas 800 A3 | GPQA Diamond | Accuracy | 90.4 |

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Chapter 5](#5-online-service-deployment).

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and parameter issues, refer to the [Public FAQs](../../faqs.md).
