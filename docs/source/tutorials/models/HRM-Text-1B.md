# HRM-Text-1B

## Introduction

`HRM-Text-1B` is a 1-billion-parameter base language model from Sapient Intelligence. It uses two recurrent Transformer stacks at different timescales and was pretrained with a PrefixLM objective. The checkpoint is a pre-alignment base model rather than a chat or instruction-following assistant.

This guide describes single-NPU deployment, PrefixLM prompting, functional verification, and accuracy evaluation with vLLM Ascend. See the [model card](https://huggingface.co/sapientinc/HRM-Text-1B) for architecture and training details.

## Supported Features

| Feature | Status | Notes |
| --- | --- | --- |
| BF16 | ✅ | Verified with the released checkpoint. |
| Tensor parallelism | ✅ | TP=1 was verified on one Atlas A2 64 GB NPU. |
| Piecewise ACL Graph | ✅ | Verified with real weights. |
| Chunked prefill | ❌ | Disabled by vLLM because PrefixLM uses non-causal attention during prefill. |
| Automatic prefix caching | ❌ | Disabled for the same non-causal attention constraint. |
| Expert parallelism / FlashComm | N/A | The checkpoint is dense, not MoE. |
| MTP | N/A | The checkpoint does not contain speculative decoding weights. |
| Multimodal input | N/A | This is a text-only model. |

The configured maximum sequence length is 4096 tokens. Because chunked prefill is unavailable, `--max-num-batched-tokens` must be at least `--max-model-len`; otherwise prompts larger than the scheduling token budget cannot be processed.

## Environment Preparation

### Model Weight

Download the BF16 checkpoint from [Hugging Face](https://huggingface.co/sapientinc/HRM-Text-1B). One Atlas A2 64 GB NPU is sufficient for the configuration in this guide.

```bash
huggingface-cli download sapientinc/HRM-Text-1B \
  --local-dir /data/models/HRM-Text-1B
```

### Docker

Use a vLLM Ascend image that matches the target vLLM version. Replace `/data/models` with the host directory containing the checkpoint.

=== "Atlas A2 inference products"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run --rm -it \
      --name vllm-ascend-hrm \
      --net host \
      --shm-size=1g \
      --device /dev/davinci0 \
      --device /dev/davinci_manager \
      --device /dev/devmm_svm \
      --device /dev/hisi_hdc \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /data/models:/data/models \
      $IMAGE bash
    ```

=== "Atlas A3 inference products"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run --rm -it \
      --name vllm-ascend-hrm \
      --net host \
      --shm-size=1g \
      --device /dev/davinci0 \
      --device /dev/davinci_manager \
      --device /dev/devmm_svm \
      --device /dev/hisi_hdc \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /data/models:/data/models \
      $IMAGE bash
    ```

For source installation, follow the [Installation Guide](../../getting_started/installation.md).

## Deployment

Start the server from the container workspace. ACL Graph is enabled by default.

```bash
cd /workspace

vllm serve /data/models/HRM-Text-1B \
  --served-model-name hrm-text-1b \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --port 8000
```

Use `--enforce-eager` as an isolation fallback if graph capture fails in a development environment. Eager mode is also used by the accuracy regression configuration to minimize graph-capture overhead in CI.

## Functional Verification

Use the Completions API because this checkpoint is not chat-tuned. Condition tokens control the desired generation style: `<|object_ref_start|>` requests a direct answer, while `<|quad_end|><|object_ref_end|>` requests synthetic chain-of-thought style generation.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hrm-text-1b",
    "prompt": "<|im_start|><|object_ref_start|>What is the capital of France?<|im_end|>",
    "max_tokens": 16,
    "temperature": 0
  }'
```

Expected result: HTTP 200 with `Paris` in the generated completion.

## Accuracy Evaluation

The regression test uses the GSM8K test split with 5-shot prompting, no chat template, and 100 samples. The model is evaluated as a base model; these numbers are intended as a reproducible vLLM Ascend regression baseline, not as a reproduction of the model card's full benchmark recipe.

| Dataset | Samples | Few-shot | Metric | Result |
| --- | ---: | ---: | --- | ---: |
| GSM8K | 100 | 5 | exact_match,strict-match | 0.01 |
| GSM8K | 100 | 5 | exact_match,flexible-extract | 0.66 |

Run the registered evaluation with:

```bash
pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/HRM-Text-1B.yaml \
  --tp-size 1
```

## Performance

On one Atlas A2 64 GB NPU, real-weight startup and inference were verified with a 4096-token maximum context and up to 16 concurrent requests. The practical maximum context matched the checkpoint's theoretical 4096-token limit when `--max-num-batched-tokens 4096` was set.

In a stability run, 320 short requests completed successfully without sustained HBM growth. Actual latency and throughput depend on prompt length, generation length, concurrency, graph cache state, and runtime versions; benchmark with production traffic before choosing serving parameters.
