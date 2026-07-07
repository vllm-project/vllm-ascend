# Qwen3-ASR-1.7B

## Introduction

The released Qwen3-ASR-1.7B is a lightweight, high-performance automatic speech recognition (ASR) model developed by the Qwen Team. It delivers industry-leading recognition accuracy across Chinese/English multi-scene speech, Chinese dialects, multilingual and singing voice scenarios, with native support for long audio and streaming inference, and deep optimization for Ascend NPU hardware.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, accuracy and performance evaluation.

## Environment Preparation

### Model Weight

`Qwen3-ASR-1.7B`(BF16 version): requires 1 Ascend 910B (with 1 x 64G NPUs).

`Qwen3-ASR-1.7B`(BF16 version): requires 1 Ascend 310P (with 1 x 48G NPUs).
[Download model weight](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### Installation

`Qwen3-ASR-1.7B` is supported in `vllm-ascend`.

You can use our official docker image to run `Qwen3-ASR-1.7B` directly.

```bash
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
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

### Atlas 300I A2 2UP

```shell
vllm serve "Qwen/Qwen3-ASR-1.7B" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --port 8000
```

### Atlas 300I DUO

```shell
vllm serve "Qwen/Qwen3-ASR-1.7B" \
  --tensor-parallel-size 1 \
  --gpu_memory_utilization 0.9 \
  --dtype float16 \
  --max_model_len 4096 \
  --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,4]}' \
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

### Remarks

This result reflects end-to-end serving performance, including audio preprocessing, request construction, API communication, inference, and response parsing. Actual performance may vary depending on hardware, concurrency, audio length, and deployment configuration.

Further benchmarking is recommended for latency distribution, concurrent throughput, long-audio scenarios, and system resource utilization.
