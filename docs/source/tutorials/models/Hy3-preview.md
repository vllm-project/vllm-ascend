# Hy3-preview

## Introduction

Hy3-preview is a Mixture-of-Experts model, with 295B total parameters, 21B active parameters and 3.8B MTP layer parameters, developed by the Tencent Hy Team. It is the first model trained on Tencent rebuilt infrastructure. It improves significantly on complex reasoning, instruction following, context learning, coding, and agent tasks.

This guide records the verified vLLM Ascend serving path for Hy3-preview on one Atlas A3 16-NPU node. The verified default path is TP16 + EP + MTP + ACLGraph.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- Hugging Face: [tencent/Hy3-preview](https://huggingface.co/tencent/Hy3-preview)
- ModelScope: [Tencent-Hunyuan/Hy3-preview](https://modelscope.cn/models/Tencent-Hunyuan/Hy3-preview)
- GitCode: [tencent_hunyuan/Hy3-preview](https://ai.gitcode.com/tencent_hunyuan/Hy3-preview)

Download or mount the checkpoint to a path shared by the runtime container, for example `/models/Hy3-preview`.

### Hardware

The verified configuration uses one Atlas A3 node with 16 NPUs and 64 GB HBM per NPU. The real-weight run used about 58 GB process memory per NPU after startup.

### Installation

You can use our official docker image to run Hy3-preview directly. For Atlas A3 machines, select the image variant with the `-a3` suffix. The official image already includes the vLLM and vLLM Ascend runtime needed for the verified serving path.

```bash
  export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
  export NAME=vllm-ascend

  docker run --rm \
    --name $NAME \
    --net=host \
    --shm-size=1g \
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
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /models:/models \
    -it $IMAGE bash
```

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/installation.md).

## Deployment

### Single-node Deployment

Run `vllm serve` from `/workspace`.

```bash
cd /workspace
export MODEL_PATH=/models/Hy3-preview

HCCL_OP_EXPANSION_MODE=AIV \
vllm serve ${MODEL_PATH} \
  --served-model-name hy3-preview \
  --tensor-parallel-size 16 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 1 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --host 0.0.0.0 \
  --port 8000
```

- `--enable-ep-weight-filter` is optional. It skips expert weights that do not belong to the local EP rank during loading, reducing disk and host-memory pressure for very large MoE checkpoints. We recommend keeping it enabled.
- Tool calling and reasoning are service interfaces declared in the Hy3 README, so it is recommended to pass the corresponding `hy_v3` parsers by default.

## Functional Verification

Check model readiness first:

```bash
curl -sf http://127.0.0.1:8000/v1/models
```

Run a text smoke request:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "hy3-preview",
    "messages": [{"role": "user", "content": "Say hi in one word."}],
    "max_tokens": 16,
    "temperature": 0,
    "top_p": 1,
    "chat_template_kwargs": {"reasoning_effort": "no_think"}
  }'
```

Expected result:

```json
{
  "model": "hy3-preview",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Hi"
      },
      "finish_reason": "stop"
    }
  ]
}
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result for reference only.

| dataset | version | metric | mode | vllm-api-general-chat | note |
|----- | ----- | ----- | ----- | -----| ----- |
| GSM8K | - | accuracy | gen | 93.07 | 1 Atlas 800 A3 (64GB × 16) |
| C-Eval | - | accuracy | gen | 87.64 | 1 Atlas 800 A3 (64GB × 16) |

### Using Language Model Evaluation Harness

Not tested yet.

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

### Lightweight Online Benchmark

The following numbers are from a real-weight smoke benchmark on one Atlas A3 16-NPU node, using TP16 + EP + MTP + ACLGraph. The benchmark used `vllm bench serve`, random prompts, output length 128, `--max-concurrency 1`, `--temperature 0`, and 4 requests per input length. These numbers are functional performance evidence, not tuned throughput limits.

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model /models/Hy3-preview \
  --served-model-name hy3-preview \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 4 \
  --request-rate inf \
  --max-concurrency 1 \
  --temperature 0 \
  --top-p 1
```

| Random input length | Success / total | Mean TTFT (ms) | Mean TPOT (ms) | Output throughput (tok/s) | Total token throughput (tok/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1,024 | 4 / 4 | 484.64 | 30.10 | 29.71 | 270.90 |
| 4,096 | 4 / 4 | 1379.41 | 30.24 | 24.52 | 811.99 |
| 16,384 | 4 / 4 | 2604.58 | 30.43 | 19.79 | 2554.65 |

## Known Limitations

- The model config supports 262,144 tokens, but this guide only verifies 32,768-token serving. Larger contexts require a separate capacity validation.
- Formal AISBench accuracy results are pending and should be added only after a real benchmark run.
