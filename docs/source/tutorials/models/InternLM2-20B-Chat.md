# InternLM2-20B-Chat

## 1 Introduction

InternLM2-20B-Chat is the chat-aligned 20B model of the InternLM2 series released by the Shanghai AI Laboratory. It is a text-only dense decoder with 48 layers, grouped-query attention (48 query heads over 8 key/value heads, head size 128), SwiGLU feed-forward blocks, RMSNorm, and BF16 weights. The checkpoint declares a 32,768-token context with dynamic NTK RoPE scaling on top of a RoPE theta of 1,000,000.

This document covers deployment of InternLM2-20B-Chat on Ascend NPUs with vLLM Ascend: weight preparation, service start-up, chat-template behaviour, parallelism choices, and the accuracy and performance figures measured for this tutorial.

The model architecture `InternLM2ForCausalLM` is implemented upstream in vLLM, so no model definition is added in vLLM Ascend. The dynamic NTK rotary embedding that the checkpoint selects is dispatched to the Ascend fused rotary kernel.

## 2 Supported Features

Please refer to the [Supported Models List](../../user_guide/support_matrix/supported_models.md) for the model support matrix, and to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

The following notes are specific to this model:

| Item | Note |
|------|------|
| Weight format | BF16 only; no quantized checkpoint is published upstream |
| Context length | 32,768 tokens, from `max_position_embeddings` |
| RoPE | `rope_scaling.type=dynamic` with `factor=2.0`; see section 9.1 |
| Expert parallelism | Not applicable, the model is dense |
| Multimodal input | Not applicable, the model is text-only |
| Speculative decoding (MTP) | Not available, the checkpoint ships no draft weights |

## 3 Prerequisites

### 3.1 Model Weight

| Model | Weight size | Hardware requirement | Download |
|-------|------------:|----------------------|----------|
| InternLM2-20B-Chat | about 37 GiB in 21 safetensors shards | 2 Atlas A2 inference products (64GB) | [ModelScope](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b), [Hugging Face](https://huggingface.co/internlm/internlm2-chat-20b) |

The weights alone exceed half of a single 64 GB device, so tensor parallelism of 2 is the recommended baseline: it leaves room for the KV cache, the graph capture buffers, and the activation peak. The accuracy workflow in this repository sets `VLLM_USE_MODELSCOPE=True`, so the end-to-end configuration refers to the ModelScope namespace `Shanghai_AI_Laboratory/internlm2-chat-20b`.

Download the weights to a directory of your choice and record it:

```bash
pip install modelscope
modelscope download \
    --model Shanghai_AI_Laboratory/internlm2-chat-20b \
    --local_dir your_model_path
```

>**Path description**: Ensure the model path in the subsequent deployment command matches this directory.

## 4 Installation

### 4.1 Docker Image Installation

**Docker Pull:**

```bash
docker pull quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
```

**Docker Run:**

=== "Atlas A2 inference products"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run --rm \
      --name vllm-ascend-env \
      --shm-size=1g \
      --net=host \
      --device /dev/davinci0 \
      --device /dev/davinci1 \
      --device /dev/davinci_manager \
      --device /dev/devmm_svm \
      --device /dev/hisi_hdc \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /root/.cache:/root/.cache \
      -it $IMAGE bash
    ```

=== "Atlas A3 inference products"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3

    docker run --rm \
      --name vllm-ascend-env \
      --shm-size=1g \
      --net=host \
      --device /dev/davinci0 \
      --device /dev/davinci1 \
      --device /dev/davinci_manager \
      --device /dev/devmm_svm \
      --device /dev/hisi_hdc \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /root/.cache:/root/.cache \
      -it $IMAGE bash
    ```

### 4.2 Source Code Installation

If you prefer to build from source, follow the [Installation Guide](../../getting_started/installation.md).

To verify the installation:

```bash
pip show vllm-ascend
```

Expected result: the version information is displayed.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Starting the Server

> The following command is an example configuration. Adjust the parameters based on your actual scenario.

=== "Atlas A2 inference products / Atlas A3 inference products"

    ```bash
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    # Ensure the model path matches the directory recorded during download
    vllm serve your_model_path \
        --served-model-name internlm2-20b-chat \
        --trust-remote-code \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.9 \
        --port <port>
    ```

### 5.2 Parameter Description

| Parameter | Note |
|-----------|------|
| `--trust-remote-code` | Required. The repository ships `configuration_internlm2.py` and `tokenization_internlm2.py` through `auto_map`, and the tokenizer cannot be built without them. The model itself still runs the native vLLM implementation, not the remote `modeling_internlm2.py`. |
| `--tensor-parallel-size` | 2 is the recommended baseline on 64 GB devices. Both 48 query heads and 8 key/value heads divide by 2 and by 4, so TP=4 is also valid; see section 9.2 for the trade-off. |
| `--dtype bfloat16` | Matches the published checkpoint. |
| `--max-model-len` | Up to 32768. Raising it increases the per-sequence KV cache, so reduce `--max-num-seqs` accordingly. |

### 5.3 Chat Template

The checkpoint ships a ChatML chat template, so `/v1/chat/completions` works without passing `--chat-template`. Each turn is wrapped as:

```text
<|im_start|>{role}
{content}<|im_end|>
```

Two properties are worth noting:

- The template emits `bos_token` once at the start of the conversation, not per turn.
- The assistant turn ends with `<|im_end|>` (id 92542). `generation_config.json` lists `eos_token_id: [2, 92542]`, so both `</s>` and `<|im_end|>` terminate generation. Keep `generation_config.json` next to the weights; without it, only `</s>` is treated as a stop token and completions run on past the end of the turn.

The system role is supported. If no system message is supplied, the template simply omits it.

### 5.4 Service Verification

The service is ready once the model list is returned:

```bash
curl http://localhost:<port>/v1/models
```

Expected result: HTTP 200 with `internlm2-20b-chat` in the `data` list.

## 6 Functional Verification

**Chat Completions API:**

```bash
curl http://localhost:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "internlm2-20b-chat",
        "messages": [
            {"role": "user", "content": "Give me a short introduction to large language models."}
        ],
        "temperature": 0,
        "max_tokens": 128
    }'
```

Expected result: HTTP 200 with a JSON response whose `choices[0].message.content` is non-empty and whose `finish_reason` is `stop` or `length`.

**Completions API:**

```bash
curl http://localhost:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "internlm2-20b-chat",
        "prompt": "Shanghai is",
        "temperature": 0,
        "max_tokens": 32
    }'
```

**Multi-turn chat:** pass the previous assistant reply back in `messages`. The response must stay on topic and must not contain the literal markers `<|im_start|>` or `<|im_end|>`; their appearance in the output means the stop tokens were not applied.

## 7 Accuracy Evaluation

This repository ships an end-to-end configuration at `tests/e2e/models/configs/InternLM2-20B-Chat.yaml`, evaluated with lm-eval on the C-Eval validation split.

```bash
export VLLM_USE_MODELSCOPE=True

pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
    --config tests/e2e/models/configs/InternLM2-20B-Chat.yaml \
    --report-dir ./benchmarks/accuracy
```

The configuration pins the parallelism, so no `--tp-size` is required. The test compares the measured metric against the checked-in baseline with the repository-wide relative tolerance and writes a Markdown report to the report directory.

| Dataset | Shots | Metric | Value |
|---------|------:|--------|------:|
| ceval-valid | 5 | acc,none | 0.6293 |

## 8 Performance Evaluation

Refer to the [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) documentation for details. Take `serve` as an example, against a running server:

```shell
vllm bench serve \
    --model your_model_path \
    --served-model-name internlm2-20b-chat \
    --port <port> \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 64 \
    --max-concurrency 8 \
    --ignore-eos
```

Keep input length, output length, request count, and concurrency fixed when comparing configurations; results obtained under different settings are not comparable.

The command above, run against a TP=2 server started with `--max-model-len 8192 --gpu-memory-utilization 0.85` on two Atlas A2 devices, gives:

| TP | Max model len | Requests | Concurrency | Request throughput | Output throughput | Mean TTFT | Mean TPOT |
|---:|--------------:|---------:|------------:|-------------------:|------------------:|----------:|----------:|
| 2 | 8192 | 64 | 8 | 1.96 req/s | 250.42 tok/s | 401.7 ms | 28.9 ms |

## 9 Performance Tuning

Please refer to the [vLLM Features](https://docs.vllm.ai/en/stable/features), [vLLM Ascend Additional Configuration](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/configuration/additional_config.html) and [vLLM Ascend Feature Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide) for key parameter descriptions.

### 9.1 Dynamic NTK RoPE

The checkpoint declares `rope_scaling = {"type": "dynamic", "factor": 2.0}`. vLLM builds a `DynamicNTKScalingRotaryEmbedding` for it. Because the checkpoint provides no `max_trained_positions`, vLLM defaults it to `max_position_embeddings`, the NTK correction term collapses to one, and the resulting cos/sin table is identical to a plain RoPE table at theta 1,000,000. Serving at the full 32,768-token context therefore needs no additional scaling configuration.

This rotary variant is dispatched to the Ascend fused rotary kernel. On releases where that dispatch is absent, the layer silently falls back to the elementwise reference implementation; nothing in the log reports the fallback, so confirm the class in use when debugging rotary behaviour on this model. Measured on two Atlas A2 devices with the workload above, the fused kernel was worth 15.4% in output throughput and 13.9% in mean time per output token, and its bf16 output was closer to an fp32 reference than the fallback at every sequence length tested.

### 9.2 Parallelism

The attention shape divides evenly at both TP=2 and TP=4. Higher tensor parallelism lowers the per-device weight and KV-cache footprint, but the upstream InternLM2 implementation stores q, k, and v interleaved in a single `wqkv` projection and reconstructs them with an all-gather in every decoder layer, so collective traffic grows with the degree of tensor parallelism. On 64 GB devices, TP=2 is the recommended starting point; move to TP=4 only when the memory headroom at the target context length and batch size requires it.

### 9.3 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods, and to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md). This section only covers issues specific to InternLM2-20B-Chat.

### 10.1 The tokenizer fails to load

The error names `tokenization_internlm2` or `InternLM2Tokenizer`. Pass `--trust-remote-code` and make sure the Python files from the model repository were downloaded alongside the weights.

### 10.2 Generation does not stop at the end of a turn

The reply continues past the answer or contains `<|im_start|>`. Check that `generation_config.json` is present in the model directory: it carries `eos_token_id: [2, 92542]`, and `tokenizer_config.json` alone declares only `</s>`.

### 10.3 Out of memory at start-up or during graph capture

Confirm that `--tensor-parallel-size` is at least 2, then lower `--max-num-seqs`, `--max-model-len`, or `--gpu-memory-utilization`. Use `--enforce-eager` to distinguish a graph-capture overhead from a steady-state shortfall.
