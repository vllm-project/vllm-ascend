# InternLM2-20B-Chat

## Introduction

InternLM2-20B-Chat is a 20-billion-parameter bilingual chat model from the
InternLM team. It uses a dense decoder-only architecture with grouped-query
attention (GQA), BF16 weights, and dynamic RoPE scaling for a context length
of up to 32,768 tokens.

This tutorial shows how to deploy `internlm/internlm2-chat-20b` with
vLLM Ascend, verify the OpenAI-compatible API, and run the repository's
end-to-end accuracy test. The instructions target the current vLLM Ascend
development version.

## Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| BF16 inference | ✅ | Native model weight format |
| Tensor parallelism | ✅ | TP=2 is recommended for the 20B BF16 model |
| ACL Graph | ✅ | Verified with TP=2 and real weights at a 2,048-token model length |
| Eager mode | ✅ | Available with `--enforce-eager` for troubleshooting |
| 32K context | ✅ | Verified in eager mode with TP=2, `max-num-seqs=16`, and a 32,002-token prompt |
| Pipeline parallelism | ✅ | Supported by the upstream vLLM model implementation |
| LoRA | ✅ | Supported by the upstream vLLM model implementation |
| Expert parallelism | N/A | InternLM2-20B-Chat is a dense model |
| Multimodal input | N/A | InternLM2-20B-Chat is a text-only model |
| MTP | ⚠️ | Checkpoint missing: the model does not provide MTP draft weights |

For the project-wide model and feature matrices, see
[Supported Models](../../user_guide/support_matrix/supported_models.md) and
[Supported Features](../../user_guide/support_matrix/supported_features.md).

## Environment Preparation

### Model Weight

The BF16 checkpoint contains approximately 39.7 GB (37.0 GiB) of safetensors
weights. Inference also requires memory for the runtime and KV cache, so TP=2
is recommended on 64 GB NPUs, especially for long-context workloads and ACL
Graph capture.

- [Hugging Face: internlm/internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b)
- [ModelScope: Shanghai_AI_Laboratory/internlm2-chat-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b)

The repository's accuracy workflow uses ModelScope, so its E2E configuration
references the `Shanghai_AI_Laboratory/internlm2-chat-20b` mirror.

Download the checkpoint to a local or shared directory before starting the
service. The following example uses ModelScope:

```bash
python -m pip install modelscope
modelscope download \
  --model Shanghai_AI_Laboratory/internlm2-chat-20b \
  --local_dir /data/models/internlm2-chat-20b
```

### Installation

Use a vLLM Ascend image whose vLLM, PyTorch, torch-npu, CANN, and driver
versions follow the compatibility matrix in the
[installation guide](../../installation.md).

=== "Atlas A2 inference products"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run --rm -it \
      --name internlm2-20b-chat \
      --shm-size=16g \
      --net host \
      --device /dev/davinci0 \
      --device /dev/davinci1 \
      --device /dev/davinci_manager \
      --device /dev/devmm_svm \
      --device /dev/hisi_hdc \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /data/models:/data/models \
      "$IMAGE" bash
    ```

=== "Atlas A3 inference products"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3

    docker run --rm -it \
      --name internlm2-20b-chat \
      --shm-size=16g \
      --net host \
      --device /dev/davinci0 \
      --device /dev/davinci1 \
      --device /dev/davinci_manager \
      --device /dev/devmm_svm \
      --device /dev/hisi_hdc \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /data/models:/data/models \
      "$IMAGE" bash
    ```

Inside the container, verify that both NPUs are visible:

```bash
npu-smi info
```

## Online Service Deployment

Start the OpenAI-compatible server with two NPUs:

```bash
vllm serve /data/models/internlm2-chat-20b \
  --served-model-name internlm2-20b-chat \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --port 8000
```

Key parameters:

- `--tensor-parallel-size 2` distributes the 20B BF16 checkpoint across two
  NPUs.
- `--max-model-len 2048` and `--max-num-seqs 4` match the real-weight ACL
  Graph validation. Increase them only after validating the target workload
  and available KV-cache memory.
- `--trust-remote-code` is required for the model repository's tokenizer and
  configuration code. vLLM still uses its native `InternLM2ForCausalLM`
  implementation for inference.

With a complete Ascend custom-op package, the default command captures both
PIECEWISE mixed prefill/decode graphs and FULL decode graphs. The real-weight
TP=2 validation captured batch sizes 1, 2, and 4, then returned HTTP 200 from
the models, completions, and chat-completions endpoints. This validation used
a 2,048-token model length; it does not establish ACL Graph support at 32K.

If graph compilation or a custom-op installation fails, add
`--enforce-eager` as a troubleshooting fallback. The 32K capacity result and
the C-Eval and performance numbers below were measured in eager mode and
should not be presented as ACL Graph results.

The service is ready when `GET /v1/models` returns HTTP 200 and lists
`internlm2-20b-chat`:

```bash
curl -f http://127.0.0.1:8000/v1/models
```

## Functional Verification

Send a deterministic chat request:

```bash
curl -f http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm2-20b-chat",
    "messages": [
      {"role": "user", "content": "Explain why the sky appears blue in one sentence."}
    ],
    "temperature": 0,
    "max_tokens": 64
  }'
```

A successful response has HTTP status 200, contains a non-empty
`choices[0].message.content` field, and does not contain NaN or Inf values in
the server log.

The completions endpoint can be checked independently:

```bash
curl -f http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm2-20b-chat",
    "prompt": "Shanghai is",
    "temperature": 0,
    "max_tokens": 32
  }'
```

## Accuracy Evaluation

The repository provides the end-to-end configuration at
`tests/e2e/models/configs/InternLM2-20B-Chat.yaml`. It evaluates the model on
the C-Eval validation split with the same harness used by vLLM Ascend CI.

Install the evaluation dependencies, then run:

```bash
export VLLM_USE_MODELSCOPE=True

pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/InternLM2-20B-Chat.yaml \
  --tp-size 2 \
  --report-dir benchmarks/accuracy
```

The real-weight TP=2 eager accuracy baseline is:

| Dataset | Shots | Metric | Value |
|---------|------:|--------|------:|
| ceval-valid | 5 | acc,none | 0.6226 |

The test compares measured metrics with the checked-in baseline using the
repository's relative tolerance and writes a Markdown report under
`benchmarks/accuracy/`.

## Performance Evaluation

Use `vllm bench serve` against the running server. Keep input length, output
length, request count, and concurrency fixed when comparing configurations:

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model internlm2-20b-chat \
  --tokenizer /data/models/internlm2-chat-20b \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 100 \
  --max-concurrency 8 \
  --ignore-eos
```

Report request throughput, output-token throughput, time to first token, and
inter-token latency together with the hardware and software versions. Results
from different context lengths or concurrency settings are not directly
comparable.

The following TP=2 results were measured on two 64 GB Ascend 910 NPUs:

| Mode | Max model length | Max sequences | Requests | Success | Duration | Request throughput | Output throughput | Mean TTFT | Mean TPOT | Mean ITL |
|------|-----------------:|--------------:|---------:|--------:|---------:|-------------------:|------------------:|----------:|----------:|---------:|
| ACL Graph | 2,048 | 4 | 100 | 100% | 101.45 s | 0.9857 req/s | 126.17 tok/s | 4145.54 ms | 29.97 ms | 29.97 ms |
| Eager | 32,768 | 16 | 100 | 100% | 166.51 s | 0.6005 req/s | 76.87 tok/s | 1295.02 ms | 91.43 ms | 91.43 ms |

Both runs use the request workload shown above. They are operational baselines,
not a strict graph-versus-eager A/B comparison, because the server-side maximum
model length and sequence capacity differ. With `max-num-seqs=4`, the ACL Graph
run queues part of the concurrency-8 burst, which increases TTFT.

For the ACL Graph run, P99 TTFT, TPOT, and ITL were 5551.16 ms, 30.72 ms,
and 125.66 ms, respectively. Graph capture completed in 12 seconds and used
approximately 0.46 GiB of graph memory per NPU.

## Troubleshooting

### The model cannot be loaded without remote code

Keep `--trust-remote-code` enabled. The checkpoint references custom
configuration and tokenizer modules even though the model architecture is
natively registered in vLLM.

### Out of memory during graph capture or long-context startup

First confirm that TP=2 is active. Then reduce `--max-num-seqs`, lower
`--max-model-len`, or reduce `--gpu-memory-utilization` to leave headroom for
non-vLLM processes. Use `--enforce-eager` only to determine whether the extra
memory is specific to ACL Graph capture.

### The server starts but chat requests fail

Check `/v1/models`, confirm that the request's `model` matches
`--served-model-name`, and verify that all tokenizer files were downloaded.
Test `/v1/completions` to separate tokenizer/chat-template problems from the
model execution path.
