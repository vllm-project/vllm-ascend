# EmbeddingGemma2 (Experimental)

## 1 Introduction

EmbeddingGemma2 is Google DeepMind's open omni-modal embedding model, built on the Gemma 4 architecture. It maps text, images, video and audio into a single 768-dimensional embedding space, with 740M total parameters (270M text backbone, 170M vision encoder, 300M audio encoder) and native Matryoshka Representation Learning (MRL) supporting 768/512/256/128-dimensional outputs. The model uses bidirectional (encoder-only) attention with a sliding window, mean pooling, and L2 normalization.

This guide describes how to deploy and evaluate EmbeddingGemma2 with vLLM Ascend on **Atlas 300I Duo**, including environment preparation, single-node deployment, functional verification, and accuracy evaluation.

> **Version and adaptation status:** The EmbeddingGemma2 support in vLLM Ascend is an initial experimental (0day/POC) release. The model implementation is provided by vLLM (main after 2026-10-06, PR [#60254](https://github.com/vllm-project/vllm/pull/60254)) and requires `transformers >= 5.19`. It is intended solely for evaluation and validation following the fixed deployment configuration outlined in this guide. Supported scenarios, performance, and configuration interfaces are subject to change in subsequent releases; please do not interpret this guide as a production support commitment.

## 2 Supported Features

Refer to [Supported Models](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

- Text, image, video and audio inputs (single-modal or interleaved)
- Mean pooling with L2 normalization; MRL truncation via the `dimensions` request parameter
- FP16 on Atlas 300I Duo (BF16 is not supported by Atlas 300I Duo; pass `--dtype float16`)
- Tensor Parallelism (TP=2), with outputs identical to TP=1

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version | Hardware Requirements | Download Links |
|----------------|-----------------------|----------------|
| `embeddinggemma-2` (FP16) | 1 Atlas 300I Duo node | [ModelScope](https://www.modelscope.cn/models/google/embeddinggemma-2) \| [Hugging Face](https://huggingface.co/google/embeddinggemma-2) |

> **Path description:** Please download the model weights to a directory of your choice and record this path. For example: `/root/.cache/modelscope/hub/models/google/embeddinggemma-2`. In subsequent deployment commands, the placeholder `<YOUR_MODEL_PATH>` will be used; please replace it with the path you have recorded here.

### 3.2 Software Requirements

- vLLM with the `EmbeddingGemma2Model` architecture (vLLM main after 2026-10-06, PR [#60254](https://github.com/vllm-project/vllm/pull/60254))
- `transformers >= 5.19` (provides `transformers.models.embedding_gemma2`)

## 4 Installation

> **Hardware support:** EmbeddingGemma2 in this tutorial is supported and validated on **Atlas 300I Duo** only. Use the corresponding Docker image and the installation steps below.

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image. For the available image tags and published versions, refer to [Using Docker](../../getting_started/installation.md#installation-prebuilt-image).

- Step 1: Download the latest Docker image

  ```bash
  docker pull quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p
  ```

- Step 2: Start Docker container

  ```bash
  # Set the vLLM Ascend image name.
  export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p
  export NAME=embeddinggemma2-dev

  # Start the container with the variables defined above.
  # Atlas 300I Duo: select its single NPU (/dev/davinci0).
  docker run --rm \
  -e ASCEND_VISIBLE_DEVICES=0 \
  --name $NAME \
  --net=host \
  --shm-size=1g \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /root/.cache:/root/.cache \
  -it $IMAGE bash
  ```

- Step 3: Installation Verification

  After starting the container, run the following command to verify the installation:

  ```bash
  docker ps | grep $NAME
  ```

  Expected Result: The container is listed with status `Up`. You can also verify the vllm-ascend version inside the container:

  ```bash
  pip show vllm-ascend
  ```

  Expected Result: The version information is displayed, matching the pulled image version.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md).

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment completes pooling inference within the same node, suitable for development, testing, and online `/v1/embeddings` serving of text, image, video and audio embeddings.

Startup command:

```bash
# Replace <YOUR_MODEL_PATH> with the actual path recorded in Section 3.1
export MODEL_PATH=<YOUR_MODEL_PATH>

vllm serve $MODEL_PATH \
    --served-model-name $MODEL_PATH \
    --runner pooling \
    --dtype float16 \
    --max-model-len 8192 \
    --skip-mm-profiling \
    --enforce-eager \
    --host 0.0.0.0 --port 8000
```

The output upon successful startup is:

```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Key Parameter Descriptions:

- `--runner pooling` is required. EmbeddingGemma2 is an embedding model, not a generative LLM.
- `--served-model-name` must match the `"model"` field in `/v1/embeddings` requests; use the same value as `MODEL_PATH`.
- `--dtype float16` is required: Atlas 300I Duo does not support BF16, while the checkpoint ships in bfloat16.
- `--max-model-len 8192` matches the model's context length for the combined multi-modal prompt.
- `--skip-mm-profiling` skips multimodal profiling, which is not needed for pooling.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQs](../../faqs.md) for troubleshooting.

### 5.2 Multi-Node Deployment

EmbeddingGemma2 is a pooling embedding model with a single-NPU memory footprint; multi-node and PD (Prefill-Decode) separation deployment are not applicable. Use [§5.1 Single-Node Online Deployment](#5-online-service-deployment) instead. Tensor Parallelism (TP=2) within one node is supported and produces outputs identical to TP=1.

## 6 Functional Verification

Once your server is started, you can verify with the following commands.

### 6.1 Text Embedding

```bash
export MODEL_PATH=<YOUR_MODEL_PATH>  # match --served-model-name from Section 5.1

curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL_PATH}\",
    \"input\": [\"What causes the northern lights?\"]
  }"
```

Expected Result: The service returns HTTP 200 OK with a JSON response containing the `embedding` field. The embedding is a 768-dimensional unit-normalized vector. Example output (embedding truncated for brevity):

```json
{
  "id": "embd-xxxxxxxx",
  "object": "list",
  "created": 1770000000,
  "model": "<YOUR_MODEL_PATH>",
  "data": [
    {
      "index": 0,
      "object": "embedding",
      "embedding": [
        0.0123,
        -0.0456
      ]
    }
  ]
}
```

### 6.2 Multimodal Embedding

Image, video and audio inputs use chat-style content. Encode the image as base64 and send via `messages` (video uses `video_url`, audio uses `input_audio`):

```bash
export MODEL_PATH=<YOUR_MODEL_PATH>  # match --served-model-name from Section 5.1
IMAGE_PATH=/path/to/image.jpg  # Replace with your local image path
IMG_B64=$(base64 -w 0 "${IMAGE_PATH}")

curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL_PATH}\",
    \"input\": [{\"role\": \"user\", \"content\": [
      {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,${IMG_B64}\"}}
    ]}]
  }"
```

Expected Result: HTTP 200 OK with a 768-dimensional unit-normalized embedding. Batched and single requests produce identical vectors.

### 6.3 MRL Truncation

Request a truncated embedding with the `dimensions` parameter:

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL_PATH}\",
    \"input\": [\"hello\"],
    \"dimensions\": 256
  }"
```

Expected Result: HTTP 200 OK with a 256-dimensional unit-normalized embedding. Supported values are 768/512/256/128.

## 7 Accuracy Evaluation

The following benchmarks are recommended for evaluating EmbeddingGemma2 embedding quality. All results below were measured on Atlas 300I Duo (FP16) with the configuration from Section 5.1.

### 7.1 STS Benchmark (Semantic Textual Similarity)

Embed the sentence pairs of the [STSBenchmark](https://huggingface.co/datasets/mteb/stsbenchmark-sts) dataset, compute cosine similarity per pair, and report Spearman correlation against the ground-truth scores. The Chinese `STSB` dataset from [C-MTEB](https://huggingface.co/datasets/C-MTEB/STSB) is evaluated the same way.

Reference results:

| Benchmark | Result | HF eager fp16 reference |
|-----------|--------|--------------------------|
| EN STSBenchmark Spearman (1379 pairs) | 0.8623 | 0.8749 |
| ZH STSB Spearman (C-MTEB STSB, 1361 pairs) | 0.7848 | 0.7999 |

### 7.2 Cross-Modal Retrieval

| Benchmark | Result |
|-----------|--------|
| COCO caption retrieval Hit@1 / Hit@5 (200 candidates) | 67% / 86% |
| ESC50 audio-to-query Hit@1 (6 classes, random 16.7%) | 67% |
| Cross-modal retrieval (image/video/text corpus) | correct directionality |

## 8 Performance Evaluation

Throughput measured on a single Atlas 300I Duo NPU with the configuration from Section 5.1:

| Workload | Throughput |
|----------|------------|
| 32 short-text batch | 102 req/s |
| 4 x 600-token batch | 2.2 req/s |

TP=2 throughput can be measured with the same workload by adding `--tensor-parallel-size 2` to the startup command in Section 5.1.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in a specific test environment and are for reference only. The optimal configuration depends on factors such as maximum input length, batch size, and precision requirements. It is recommended to tune based on actual conditions.

| Scenario | Configuration | NPUs | TP | Max Model Len | Enforce Eager |
|----------|--------------|------|----|---------------|---------------|
| Short-text high throughput | Server / Single Machine | 1 | 1 | 8192 | On |
| Long-text batch (600-token) | Server / Single Machine | 1 | 1 | 8192 | On |
| Large model memory headroom | Server / Single Machine | 2 | 2 | 8192 | On |

### 9.2 Tuning Guidelines

#### 9.2.1 Model-Specific Optimizations

| Optimization Technique | Applicable Scenarios | Enablement Method | Precautions |
| ---------------------- | -------------------- | ----------------- | ----------- |
| MRL truncation | Downstream vector DBs with dim < 768 | Pass `dimensions` in the `/v1/embeddings` request | Lower dimensions trade recall for storage |
| Tensor Parallelism | Serving with tight single-NPU memory | `--tensor-parallel-size 2` | Outputs are identical to TP=1 |

#### 9.2.2 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Matrix](../user_guide/feature_guide/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md); this chapter only covers model-specific issues.

### 10.1 Startup fails with dtype errors on Atlas 300I Duo

**Problem phenomenon**: The server fails to start or produces NaN embeddings when loading the checkpoint with its default dtype.

**Cause analysis**: The checkpoint ships in bfloat16, which Atlas 300I Duo does not support.

**Solution**: Pass `--dtype float16` as shown in Section 5.1.

### 10.2 Audio input is truncated

**Problem phenomenon**: Audio longer than ~11 seconds produces embeddings that ignore the tail of the clip.

**Cause analysis**: The audio encoder limits input to `audio_seq_length` 280 x 40 ms/token = 11.2 s.

**Solution**: Truncate the audio to its first 11.2 s before sending; the behavior is expected and consistent with the upstream model.

### 10.3 Video embedding is slow for long clips

**Problem phenomenon**: Long video clips take much longer to embed, with unchanged embedding quality.

**Cause analysis**: Video is uniformly sampled at 1 fps up to 32 frames (140 soft tokens per video); longer clips only increase decode time.

**Solution**: Pre-trim clips to at most 32 seconds where possible.

## 11 Notes and Limitations

- Images are resized by the processor to 280 soft tokens; any resolution works.
- Context length is 8192 tokens for the combined multi-modal prompt.
