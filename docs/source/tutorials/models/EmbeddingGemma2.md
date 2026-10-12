# EmbeddingGemma2

## 1 Introduction

EmbeddingGemma2 is Google DeepMind's open omni-modal embedding model, built on the Gemma 4 architecture. It maps text, images, video and audio into a single 768-dimensional embedding space, with 740M total parameters (270M text backbone, 170M vision encoder, 300M audio encoder) and native Matryoshka Representation Learning (MRL) supporting 768/512/256/128-dimensional outputs. This guide describes how to run the model on Ascend with vLLM Ascend.

The model uses bidirectional (encoder-only) attention with a sliding window, mean pooling, and L2 normalization. On Atlas 300I Duo (310P) the encoder-only attention path runs with native torch ops.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model's feature matrix.

- Text, image, video and audio inputs (single-modal or interleaved)
- Mean pooling with L2 normalization; MRL truncation via the `dimensions` request parameter
- FP16 on Atlas 300I Duo (BF16 is not supported by 310P; convert `config.json` dtype or pass `--dtype float16`)

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version | Download Links |
|----------------|----------------|
| `embeddinggemma-2` | [ModelScope](https://www.modelscope.cn/models/google/embeddinggemma-2) / [Hugging Face](https://huggingface.co/google/embeddinggemma-2) |

### 3.2 Software Requirements

- vLLM with the `EmbeddingGemma2Model` architecture (vLLM main after 2026-10-06, PR [#60254](https://github.com/vllm-project/vllm/pull/60254))
- `transformers >= 5.19` (provides `transformers.models.embedding_gemma2`)

## 4 Deployment

```bash
vllm serve /path/to/embeddinggemma-2 \
    --runner pooling \
    --dtype float16 \
    --max-model-len 8192 \
    --skip-mm-profiling \
    --enforce-eager \
    --host 0.0.0.0 --port 8000
```

Notes:

- Mean pooling is the model default; no `--convert embed` is needed.
- On 310P, convert the `dtype` fields in `config.json` from `bfloat16` to `float16` and add top-level `"is_matryoshka": true` plus `"matryoshka_dimensions": [128, 256, 512, 768]` to enable MRL truncation.

## 5 Verification

```bash
# Text
curl http://127.0.0.1:8000/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "/path/to/embeddinggemma-2",
  "input": ["What causes the northern lights?"]
}'

# Image (chat-style input; video uses video_url, audio uses input_audio)
curl http://127.0.0.1:8000/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "/path/to/embeddinggemma-2",
  "input": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}]
}'

# MRL truncation
curl http://127.0.0.1:8000/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "/path/to/embeddinggemma-2",
  "input": ["hello"],
  "dimensions": 256
}'
```

Expected: 768-dimensional (or MRL-truncated) unit-normalized embeddings; batch and single requests return consistent vectors.

## 6 Validation Results on Atlas 300I Duo (310P3, FP16)

| Benchmark | Result | HF eager fp16 reference |
|-----------|--------|--------------------------|
| EN STSBenchmark Spearman (1379 pairs) | 0.8623 | 0.8749 |
| ZH STSB Spearman (C-MTEB STSB, 1361 pairs) | 0.7848 | 0.7999 |
| COCO caption retrieval Hit@1 / Hit@5 (200 candidates) | 67% / 86% | - |
| ESC50 audio-to-query Hit@1 (6 classes, random 16.7%) | 67% | - |
| Cross-modal retrieval (image/video/text corpus) | correct directionality | - |

Throughput (single Atlas 300I Duo NPU): 32 short-text batch = 102 req/s; 4x600-token batch = 2.2 req/s. TP=2 is supported with identical outputs.

## 7 Notes and Limitations

- Audio longer than 11.2 s (`audio_seq_length` 280 x 40 ms/token) is truncated to its first 11.2 s.
- Video is uniformly sampled at 1 fps up to 32 frames (140 soft tokens per video); longer clips only increase decode time.
- Images are resized by the processor to 280 soft tokens; any resolution works.
- Context length is 8192 tokens for the combined multi-modal prompt.
