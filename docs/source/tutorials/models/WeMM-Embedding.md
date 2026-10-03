# WeMM-Embedding

## 1 Introduction

[WeMM-Embedding](https://github.com/Tencent/WeMM-Embedding) is a universal multimodal
embedding model family released by Tencent, built on the Qwen3.5 hybrid
(linear attention / full attention) backbone. It accepts text, images, videos,
visual documents and interleaved multimodal inputs, and returns a single
L2-normalized embedding per input (2048-dim for 2B, 2560-dim for 4B,
4096-dim for 9B) with Matryoshka truncation support. This guide describes how
to run the 2B / 4B / 9B models on Atlas 300I Duo with vLLM Ascend.

The model weights are a stock `Qwen3_5ForConditionalGeneration` checkpoint,
so no custom model code is required: vLLM's native Qwen3.5 implementation is
wrapped into an embedding model with `--runner pooling`, whose default pooler
(last-token pooling + L2 normalization) matches the reference
`WeMMEmbedding.embedding()` implementation.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md)
for the model's support status. The results below were verified on Atlas 300I
Duo (310P3) in FP16: text, image and video embeddings, Matryoshka dimension
truncation, and the sentence-transformers reference example from the model card.

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version | Download Links |
|----------------|----------------|
| `WeMM-Embedding-2B` | [ModelScope](https://modelscope.cn/models/tencent-community/WeMM-Embedding-2B) / [HuggingFace](https://huggingface.co/tencent/WeMM-Embedding-2B) |
| `WeMM-Embedding-4B` | [ModelScope](https://modelscope.cn/models/tencent-community/WeMM-Embedding-4B) / [HuggingFace](https://huggingface.co/tencent/WeMM-Embedding-4B) |
| `WeMM-Embedding-9B` | [ModelScope](https://modelscope.cn/models/tencent-community/WeMM-Embedding-9B) / [HuggingFace](https://huggingface.co/tencent/WeMM-Embedding-9B) |

The checkpoints ship with a `bfloat16` config. Atlas 300I Duo does not support
BF16, so pass `--dtype float16` when serving (this overrides the checkpoint
dtype; no config edit is needed).

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

### 3.2 Docker Image

Use the standard 310P image, e.g. `quay.io/ascend/vllm-ascend:{vllm_ascend_version}-310p`.

>**Known issue**: the `v0.26.0rc2-310p` release image is missing the compiled
>`vllm_ascend_C` extension, which the 310P GDN (gated delta net) path requires —
>every Qwen3.5-based model then fails at the first request with
>`AttributeError: '_C_ascend' object has no attribute 'npu_causal_conv1d_310'`.
>See [#17713](https://github.com/vllm-project/vllm-ascend/issues/17713) for
>details and an in-container rebuild workaround until the image is fixed.

## 4 Installation

### 4.1 Docker Image Installation

=== "Atlas 300I DUO"

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net=host \
        --privileged=true \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

After a successful docker run, you can verify the running container service by executing the `docker ps` command.

## 5 Serving the Model

All three sizes use the same recipe (each fits on one 310P NPU in FP16);
only the model path changes.

```shell
#!/bin/sh
# Ensure the model path matches the directory recorded during download
vllm serve /path/to/WeMM-Embedding-2B \
  --runner pooling \
  --convert embed \
  --chat-template /path/to/WeMM-Embedding-2B/embedding_chat_template.jinja \
  --dtype float16 \
  --max-model-len 8192 \
  --skip-mm-profiling \
  --enforce-eager \
  --port 8000
```

Key Parameter Descriptions:

- `--runner pooling --convert embed` wraps the generative Qwen3.5 checkpoint
  into an embedding model. The default pooler (last-token pooling with L2
  normalization) matches the reference implementation; the chat template that
  ships with the checkpoint (`embedding_chat_template.jinja`) appends the
  `<embedding>` token that is pooled.
- `--dtype float16` is required because Atlas 300I Duo does not support BF16.
- `--max-model-len` should stay conservative. The multimodal profiling run
  would otherwise pre-run the vision tower on a maximum-size video (the
  default processor budget allows up to 768 frames), which does not complete
  on 310P. `--skip-mm-profiling` disables that pre-run; large videos should
  still be capped or preprocessed client-side (e.g. with
  `qwen_vl_utils.process_vision_info`).
- `--enforce-eager` avoids ACL graph capture stream exhaustion on 310P
  (`Stream resources are insufficient`, error 207008). For higher throughput
  a reduced capture-size list can be tried instead, e.g.
  `--compilation-config '{"cudagraph_capture_sizes": [1024, 512]}'`.

## 6 Verifying the Service

Text embeddings (`/v1/embeddings`):

```shell
curl http://127.0.0.1:8000/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "/path/to/WeMM-Embedding-2B",
  "input": ["A man is eating food", "红烧肉是一道经典的中式菜肴。"]
}'
```

Multimodal inputs use the chat form (any mix of `image_url` / `video_url` /
`text`):

```shell
curl http://127.0.0.1:8000/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "/path/to/WeMM-Embedding-2B",
  "input": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "What is shown in this picture?"}
  ]}]
}'
```

Add `"dimensions": 256` to any request for Matryoshka truncation (the server
re-normalizes the truncated vector).

Verified results on Atlas 300I Duo (310P3), FP16, single NPU:

| Check | 2B | 4B | 9B |
| ----- | -- | -- | -- |
| Embedding dim / L2 norm | 2048 / 1.0000 | 2560 / 1.0000 | 4096 / 1.0000 |
| Text semantic similarity (related vs unrelated) | 0.760 vs 0.192 | 0.765 vs 0.189 | 0.812 vs 0.299 |
| Text→image retrieval direction | 0.541 vs 0.174 | 0.497 vs 0.119 | 0.556 vs 0.230 |
| Matryoshka `dimensions=256` | ✅ | ✅ | ✅ |

Reproducing the model-card reference example (queries/documents from the
`sentence-transformers/example-documents` dataset) gives the same ranking as
the published BF16 GPU numbers, with absolute cosine deviations up to ~0.09
on the matching pairs (expected for FP16 vs BF16 plus template surface-form
differences).

## 7 Performance Reference

Atlas 300I Duo, single NPU, FP16, 2B model, eager mode: batch of 32 short
texts ≈ 473 tokens/s (≈ 22 requests/s); single-request latency ≈ 355 ms.
