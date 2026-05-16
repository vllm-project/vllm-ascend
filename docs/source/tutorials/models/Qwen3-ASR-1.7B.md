# Qwen3-ASR-1.7B

## Introduction

The released Qwen3-ASR-1.7B is a lightweight, high-performance automatic speech recognition (ASR) model developed by the Qwen Team. It delivers industry-leading recognition accuracy across Chinese/English multi-scene speech, Chinese dialects, multilingual and singing voice scenarios, with native support for long audio and streaming inference, and deep optimization for Ascend NPU hardware.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, accuracy and performance evaluation.

## Environment Preparation

### Model Weight

`Qwen3-ASR-1.7B`(BF16 version): requires 1 Ascend 910B (with 1 x 64G NPUs). [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

`Qwen3-ASR-1.7B` is supported in `vllm-ascend`.

You can use our official docker image to run `Qwen3-ASR-1.7B` directly.

```{code-block} bash
   :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
  --name vllm-ascend \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -v /data/vllm-workspace/models:/data/vllm-workspace/models \
  -p 8000:8000 \
  -it $IMAGE bash
```

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## Deployment

```{test} bash
:sync-yaml: tests/e2e/models/configs/Qwen3-ASR-1.7B.yaml
:sync-target: test_cases[0].model test_cases[0].server_cmd
:sync-class: cmd

vllm serve "Qwen/Qwen3-ASR-1.7B" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --port 8000
```

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://localhost:8000/v1/chat/completions
    -H "Content-Type: application/json"
    -d '{
    "messages": [
    {"role": "user", "content": [
        {"type": "audio_url",
        "audio_url":
        {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}}
    ]}
    ]
}'
```

## Accuracy Evaluation

After all samples were processed, transcription quality was measured using:

- WER (Word Error Rate) for word-level recognition accuracy
- CER (Character Error Rate) for character-level recognition accuracy

The current evaluation results are:

| Category | Dataset | Metric | Result |
|----------|---------|--------|--------|
| Accuracy | librispeech_asr / clean / test | Total Samples | 500 |
| Accuracy | librispeech_asr / clean / test | Success | 500 |
| Accuracy | librispeech_asr / clean / test | Failure | 0 |
| Accuracy | librispeech_asr / clean / test | WER | 0.035 |

## Performance

### Baseline Result

In the current evaluation, **Qwen3-ASR-1.7B** processed **100 samples** in approximately **57 seconds**, achieving an average throughput of **1.73 samples/s** under the current online serving setup.

| Category | Dataset | Metric | Result |
|----------|---------|--------|--------|
| Performance | LibriSpeech test/clean (100 samples) | Total Samples | 100 |
| Performance | LibriSpeech test/clean (100 samples) | Total Runtime | 57 s |
| Performance | LibriSpeech test/clean (100 samples) | Average Throughput | 1.73 samples/s |

### Remarks

This result reflects end-to-end serving performance, including audio preprocessing, request construction, API communication, inference, and response parsing. Actual performance may vary depending on hardware, concurrency, audio length, and deployment configuration.

Further benchmarking is recommended for latency distribution, concurrent throughput, long-audio scenarios, and system resource utilization.
