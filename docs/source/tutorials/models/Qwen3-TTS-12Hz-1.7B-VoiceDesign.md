# Qwen3-TTS-12Hz-1.7B-VoiceDesign

## Introduction

Qwen3-TTS-12Hz-1.7B-VoiceDesign is a text-to-speech (TTS) model developed by the Qwen Team. It supports high-quality 12Hz audio synthesis, natural-language voice design, and multilingual speech generation (including Chinese and English), with optimization for Ascend NPU hardware.

This document will show the main verification steps of the model, including environment preparation, single-node deployment, functional verification, and accuracy evaluation.

## Environment Preparation

### Model Weight

`Qwen3-TTS-12Hz-1.7B-VoiceDesign` (FP16 version): requires 1 Ascend 910B (with 1 x 64G NPUs). [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

`Qwen3-TTS-12Hz-1.7B-VoiceDesign` is supported in `vllm-ascend`.

You can use our official docker image to run `Qwen3-TTS-12Hz-1.7B-VoiceDesign` directly.

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

```bash
vllm serve "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --enforce-eager \
  --port 8000
```

## Functional Verification

Once your server is started, you can query the model with input prompts:

### Test Case 1: Chinese Speech Synthesis

```shell
curl http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
    "input": "人工智能正在深刻改变我们的生活方式。",
    "task_type": "VoiceDesign",
    "language": "Chinese",
    "instructions": "温暖友好的女声，正常语速"
}' --output chinese_tts.wav
```

### Test Case 2: English Speech Synthesis

```shell
curl http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
    "input": "Artificial intelligence is transforming our daily lives.",
    "task_type": "VoiceDesign",
    "language": "English",
    "instructions": "Professional male voice"
}' --output english_tts.wav
```

## Accuracy Evaluation

Run the E2E correctness test script:

```bash
pytest -sv tests/e2e/models/test_tts_eval_correctness.py \
    --config tests/e2e/models/configs/Qwen3-TTS-12Hz-1.7B-VoiceDesign.yaml
```

After all samples were processed, synthesis quality was measured using:

- Sample rate validation (expected 24000 Hz)
- RTF (Real-Time Factor) for inference latency

Example script output:

```bash
Loading TTS model: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
  device_map=npu:0, dtype=torch.float16, attn_implementation=eager

Task: tts_synthesis (2 case(s))
  Running case: chinese_synthesis
  chinese_synthesis | sr=24000 (expect 24000) | rtf=1.300 (limit 1.5) | ✅
  Running case: english_synthesis
  english_synthesis | sr=24000 (expect 24000) | rtf=0.800 (limit 1.5) | ✅
tts_synthesis | rtf_average: limit=1.5 | measured=1.05 | ✅
```

The current evaluation results are:

| Category | Test Case | Metric | Result |
|----------|-----------|--------|--------|
| Accuracy | tts_synthesis/chinese_synthesis | sample_rate | 24000 |
| Accuracy | tts_synthesis/chinese_synthesis | rtf | 1.3 |
| Accuracy | tts_synthesis/english_synthesis | sample_rate | 24000 |
| Accuracy | tts_synthesis/english_synthesis | rtf | 0.8 |
| Accuracy | tts_synthesis | rtf_average | 1.05 |
