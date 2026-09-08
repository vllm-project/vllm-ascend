# SigLIP2

## 1 Introduction

SigLIP2 is a family of vision-language embedding models from Google. Each checkpoint provides **separate** text and image encoders for contrastive embedding (not text generation). vLLM runs SigLIP2 as a **pooling** model via `llm.embed()` (offline) or `/v1/embeddings` (online).

Supported use cases include image-text similarity, zero-shot ImageNet classification, and multimodal retrieval. **Text and image must be embedded in separate requests**; do not pass both in one call.

This guide describes how to deploy and evaluate SigLIP2 with vLLM Ascend on **Atlas 300I DUO**.

This document is validated and written based on the **vLLM-Ascend main branch** as of **2026-09-03**. The current model (`siglip2-base-patch16-224`) is fully supported for text and image embedding on this branch. As a pooling model, SigLIP2 is used for offline `llm.embed()` and online `/v1/embeddings` serving; features such as PD separation and MTP are not applicable. Use the **main branch** snapshot from that date or a later official release that includes SigLIP2 support.

## 2 Supported Features

Refer to [Supported Models](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version | Hardware Requirements | Download Links |
|----------------|-----------------------|----------------|
| `siglip2-base-patch16-224` (FP16) | 1 Atlas 300I DUO node | [Modelscope](https://modelscope.cn/models/google/siglip2-base-patch16-224) \| [HuggingFace](https://huggingface.co/google/siglip2-base-patch16-224) |

>**Path description:** Please download the model weights to a directory of your choice and record this path. For example: `/root/.cache/modelscope/hub/models/google/siglip2-base-patch16-224`. In subsequent commands, replace `<YOUR_MODEL_PATH>` with the path you recorded here (a local directory or a Hugging Face / ModelScope model id such as `google/siglip2-base-patch16-224`).

### 3.2 ImageNet Labels (Optional, for Accuracy Evaluation)

For ImageNet val zero-shot Top-1 evaluation, prepare:

- [ImageNet ILSVRC 2012 val images](https://www.image-net.org/download.php) (login and agree to terms)
- `val_label.txt` in PyTorch index format (0–999), for example from [simple-imagenet-test](https://github.com/rentainhe/simple-imagenet-test/blob/master/val_label.txt)
- `imagenet1000_clsidx_to_labels.txt` from [yrevar gist](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)

## 4 Installation

> **Hardware support:** SigLIP2 in this tutorial is supported and validated on **Atlas 300I DUO** only. Use the corresponding Docker image and the installation steps below.

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
  export NAME=siglip2-dev

  # Start the container with the variables defined above.
  # Atlas 300I DUO uses a single NPU device (/dev/davinci0).
  docker run --rm \
  --name $NAME \
  --net=host \
  --shm-size=1g \
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

- Step 3: Installation Verification

  After starting the container, run the following command to verify the installation:

  ```bash
  docker ps | grep $NAME
  ```

  Expected result: The container is listed with status `Up`. You can also verify the vllm-ascend version inside the container:

  ```bash
  pip show vllm-ascend
  ```

  Expected result: The version information is displayed, matching the pulled image version.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment runs text and image embedding on one Atlas 300I DUO node, suitable for development, testing, and online `/v1/embeddings` serving.

Startup command:

```shell
#!/bin/sh
# Replace <YOUR_MODEL_PATH> with the path recorded in Section 3.1.
export MODEL_PATH=<YOUR_MODEL_PATH>

vllm serve $MODEL_PATH \
    --served-model-name $MODEL_PATH \
    --runner pooling \
    --chat-template template_basic.jinja \
    --limit-mm-per-prompt '{"image": 1}' \
    --compilation-config '{"cudagraph_capture_sizes": [64,32]}' \
    --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}' \
    --dtype float16 \
    --port 8000 \
    --max-model-len 64
```

Required Parameter Descriptions:

- `--compilation-config` For Atlas 300I DUO, due to limited hardware streams, the size of cudagraph_capture_sizes is restricted.

Key Parameter Descriptions:

- **Tensor parallelism (TP) is not supported** for SigLIP2 online serving. Do not set `--tensor-parallel-size`; deploy on a single Atlas 300I DUO NPU as shown above.
- `--served-model-name` must match the `"model"` field in `/v1/embeddings` requests; use the same value as `MODEL_PATH`.
- `--runner pooling` is required. SigLIP2 is an embedding model, not a generative LLM.
- `--max-model-len 64` matches SigLIP2 text tokenization (`padding=max_length`, `max_length=64`).
- `--chat-template template_basic.jinja` is required when sending images via `messages` on `/v1/embeddings`.
- `--limit-mm-per-prompt '{"image": 1}'` allows one image per request.
- For image-only embedding over HTTP, use an **empty** text prompt in `messages` or offline `prompt=""` with `multi_modal_data`.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQs](../../faqs.md) for troubleshooting.

### 5.2 Multi-Node PD Separation Deployment

SigLIP2 is a pooling embedding model and **does not support** multi-node PD (Prefill-Decode) separation deployment. Use [§5.1 Single-Node Online Deployment](#51-single-node-online-deployment) instead.

## 6 Functional Verification

Once your server is started, you can verify with the following commands.

### 6.1 Text Embedding

```bash
export MODEL_PATH=<YOUR_MODEL_PATH>  # match --served-model-name from Section 5.1

curl -X POST http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_PATH}\",
    \"input\": [\"This is a photo of a dog.\"]
  }"
```

Use the template `"This is a photo of {}."` for zero-shot classification prompts. SigLIP2 was trained with `padding=max_length` and `max_length=64` for text; vLLM applies this when using offline `tokenization_kwargs`.

### 6.2 Image Embedding

Encode the image as base64 and send via `messages`:

```bash
export MODEL_PATH=<YOUR_MODEL_PATH>  # match --served-model-name from Section 5.1
IMG_B64=$(base64 -w 0 /path/to/image.jpg)

curl -X POST http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_PATH}\",
    \"encoding_format\": \"float\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [{
        \"type\": \"image_url\",
        \"image_url\": {\"url\": \"data:image/jpeg;base64,${IMG_B64}\"}
      }]
    }]
  }"
```

Expected Result:

The service returns HTTP 200 OK with a JSON response containing the `embedding` field for each request.

For more usage examples, please reference the [vLLM pooling embed examples](https://github.com/vllm-project/vllm/tree/main/examples/pooling/embed).

### 6.3 Offline Embedding

```python
from vllm import LLM

MODEL_PATH = "<YOUR_MODEL_PATH>"  # Replace with the path recorded in Section 3.1

llm = LLM(
    model=MODEL_PATH,
    runner="pooling",
    limit_mm_per_prompt={"image": 1},
    max_model_len=64,
)

# Text
text_out = llm.embed(
    ["This is a photo of a dog."],
    tokenization_kwargs={"padding": "max_length", "max_length": 64},
)
print(len(text_out[0].outputs.embedding))

# Image (empty prompt; field name must be multi_modal_data)
from PIL import Image

img = Image.open("/path/to/image.jpg").convert("RGB")
img_out = llm.embed(
    {"prompt": "", "multi_modal_data": {"image": img}},
)
print(len(img_out[0].outputs.embedding))
```

## 7 Accuracy Evaluation

ImageNet val zero-shot Top-1 is a common accuracy benchmark for SigLIP2.

### 7.1 Dataset and Labels

1. Download [ImageNet ILSVRC 2012 val images](https://www.image-net.org/download.php) (login required).
2. Download `val_label.txt` ([example](https://github.com/rentainhe/simple-imagenet-test/blob/master/val_label.txt)). Each line: `ILSVRC2012_val_00000001.JPEG 65` (PyTorch class id 0–999).
3. Download `imagenet1000_clsidx_to_labels.txt` for the 1000 class text templates.

### 7.2 Offline Evaluation

Embed 1000 class texts and val images separately, then compute cosine similarity (L2-normalized dot product). Example workflow:

```python
import ast
import numpy as np
from PIL import Image
from vllm import LLM

TEXT_TEMPLATE = "This is a photo of {}."
TOKEN_KWARGS = {"padding": "max_length", "max_length": 64}

def load_classnames(path):
    with open(path, encoding="utf-8") as f:
        d = ast.literal_eval(f.read())
    return [d[i] for i in range(1000)]

def load_val_label(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                gt[parts[0].split(".")[0]] = int(parts[1])
    return gt

MODEL_PATH = "<YOUR_MODEL_PATH>"  # Replace with the path recorded in Section 3.1

llm = LLM(
    model=MODEL_PATH,
    runner="pooling",
    limit_mm_per_prompt={"image": 1},
    max_model_len=64,
)

classnames = load_classnames("imagenet1000_clsidx_to_labels.txt")
prompts = [TEXT_TEMPLATE.format(c) for c in classnames]
text_feats = np.asarray(
    [o.outputs.embedding for o in llm.embed(prompts, tokenization_kwargs=TOKEN_KWARGS)],
    dtype=np.float32,
)
text_feats /= np.linalg.norm(text_feats, axis=1, keepdims=True)

gt = load_val_label("val_label.txt")
correct = 0
total = 0
for stem, label in gt.items():
    path = f"ImageNet/val/{stem}.JPEG"  # or .jpeg
    img = Image.open(path).convert("RGB")
    feat = np.asarray(
        llm.embed({"prompt": "", "multi_modal_data": {"image": img}})[0]
        .outputs.embedding,
        dtype=np.float32,
    )
    feat /= np.linalg.norm(feat)
    if np.argmax(feat @ text_feats.T) == label:
        correct += 1
    total += 1

print(f"Top-1 accuracy: {100.0 * correct / total:.2f}%")
```

Reference Top-1 on ImageNet val (approximate):

| Model | Top-1 |
|-------|-------|
| `siglip2-base-patch16-224` | ~69% |

## 8 Performance Evaluation

Benchmark `/v1/embeddings` over HTTP with the script below.

### 8.1 HTTP Serving Benchmark

Start the server from [§5 Online Service Deployment](#5-online-service-deployment), then run:

```python
"""Benchmark SigLIP2 /v1/embeddings serving over HTTP."""

import base64
import io
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"
MODEL = "<YOUR_MODEL_PATH>"  # match --served-model-name from Section 5.1
NUM_REQUESTS = 200
CONCURRENCY = 8
WARMUP = 10
MODE = "both"  # "text", "image", or "both"
TEXT = "This is a photo of a dog."
IMAGE_SIZE = (224, 224)
TIMEOUT_S = 120.0


def post_json(url: str, payload: dict) -> tuple[bool, float, str]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            resp.read()
            ok = resp.status == 200
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:200]
        return False, time.perf_counter() - t0, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return False, time.perf_counter() - t0, str(e)
    return ok, time.perf_counter() - t0, ""


def random_jpeg_b64(width: int, height: int, seed: int) -> str:
    arr = np.random.default_rng(seed).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8
    )
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_payload(mode: str, seed: int) -> dict:
    if mode == "text":
        return {
            "model": MODEL,
            "input": [TEXT],
            "encoding_format": "float",
        }
    w, h = IMAGE_SIZE
    b64 = random_jpeg_b64(w, h, seed)
    return {
        "model": MODEL,
        "encoding_format": "float",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                        },
                    }
                ],
            }
        ],
    }


def percentile(values: list[float], percent: float) -> float:
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (percent / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (
        rank - lower
    )


def run_benchmark(mode: str) -> None:
    url = f"{BASE_URL.rstrip('/')}/v1/embeddings"
    total = NUM_REQUESTS + WARMUP
    payloads = [build_payload(mode, i) for i in range(total)]

    def send_one(index: int) -> tuple[bool, float]:
        ok, latency_s, _ = post_json(url, payloads[index])
        return ok, latency_s

    if WARMUP:
        with ThreadPoolExecutor(max_workers=min(CONCURRENCY, WARMUP)) as pool:
            list(pool.map(send_one, range(WARMUP)))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(send_one, WARMUP + i) for i in range(NUM_REQUESTS)]
        results = [fut.result() for fut in as_completed(futures)]
    wall_s = time.perf_counter() - t0

    ok_lat_ms = [lat * 1000 for ok, lat in results if ok]
    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n=== {mode} ===")
    print(f"  successful: {len(ok_lat_ms)}/{NUM_REQUESTS}")
    print(f"  failed:     {failed}")
    print(f"  duration:   {wall_s:.2f}s")
    print(f"  throughput: {len(ok_lat_ms) / wall_s:.2f} req/s")
    if ok_lat_ms:
        print(f"  mean E2EL:  {statistics.mean(ok_lat_ms):.2f} ms")
        print(f"  median E2EL:{statistics.median(ok_lat_ms):.2f} ms")
        print(f"  p99 E2EL:   {percentile(ok_lat_ms, 99):.2f} ms")


if __name__ == "__main__":
    if MODE in ("text", "both"):
        run_benchmark("text")
    if MODE in ("image", "both"):
        run_benchmark("image")
```

### 8.2 Metrics

The script reports:

- **Request throughput (req/s)**
- **Mean / median / p99 E2EL** (end-to-end latency in ms)

Adjust `NUM_REQUESTS`, `CONCURRENCY`, and `WARMUP` at the top of the script. Set `MODE` to `"text"`, `"image"`, or `"both"`.

After about several minutes, you can get the performance evaluation result.

## 9 Performance Tuning

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as text vs. image workload, request concurrency, batch size, and image resolution. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

### 9.1 Recommended Configurations

The following configurations are validated on Atlas 300I DUO and are categorized by use case. Start from the [§5.1 Single-Node Online Deployment](#51-single-node-online-deployment) command, then adjust the serve flags below.

| Scenario | Workload | Deployment | NPUs | Max Num Seqs | Max Num Batched Tokens | Max Model Len | Client Concurrency (ref.) |
|----------|----------|------------|------|--------------|------------------------|---------------|---------------------------|
| Text high throughput | Text `/v1/embeddings` | Single node | 1 (300I DUO) | 32 | 512 | 64 | 16–32 |
| Image high throughput | Image `/v1/embeddings` (224×224) | Single node | 1 (300I DUO) | 16 | 256 | 64 | 8–16 |
| Low latency | Text or image | Single node | 1 (300I DUO) | 8 | 128 | 64 | 4–8 |

> **Note**: `--max-num-seqs` and `--max-num-batched-tokens` are set at `vllm serve` startup. Client concurrency in [§8.1 HTTP Serving Benchmark](#81-http-serving-benchmark) controls how many HTTP requests are sent in parallel; keep it close to `--max-num-seqs` for stable batching. SigLIP2 does not support TP or PD separation.

Example serve flags for the text high-throughput row:

```shell
vllm serve $MODEL_PATH \
    ... \
    --max-num-seqs 32 \
    --max-num-batched-tokens 512 \
    --max-model-len 64
```

### 9.2 Tuning Guidelines

#### 9.2.1 Model-Specific Optimizations

##### Optimizations Enabled by Default

The following optimizations are enabled in the recommended [§5.1](#51-single-node-online-deployment) configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
| ---------------------- | ------------------- | ------------------- |
| ACL graph capture | Uses `--compilation-config '{"cudagraph_capture_sizes": [64,32]}'` to capture fixed small batch shapes on Atlas 300I DUO | Reduces per-request scheduling overhead for short text and image embed paths |
| Pooling runner | Uses `--runner pooling` for embedding-only forward passes | Required for SigLIP2; avoids generative decode paths |
| FP16 inference | Uses `--dtype float16` on Atlas 300I DUO | Matches 300I DUO supported precision and model weights |

##### Optimizations That Require Explicit Enabling

| Optimization Technique | Applicable Scenarios | Enablement Method | Technical Principle | Precautions |
| ---------------------- | -------------------- | ----------------- | ------------------- | ----------- |
| Server batch tuning | Online text/image serving | Set `--max-num-seqs` and `--max-num-batched-tokens` at serve startup (see Section 9.1) | Controls how many embed requests are batched on the server | Reduce both flags if OOM occurs; image embed usually needs smaller batches than text |
| Client concurrency tuning | HTTP benchmark or production clients | Increase parallel `/v1/embeddings` requests (see Section 8.1) | Raises offered load to the server batcher | Throughput gains plateau once client concurrency exceeds `--max-num-seqs`; watch p99 latency |

#### 9.2.2 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md); this chapter only covers model-specific issues.

**Q: Top-1 accuracy is near 0% but embeddings look valid.**

A: Check that ground-truth labels use PyTorch index (`val_label.txt`), not devkit `ILSVRC2012_validation_ground_truth.txt` with yrevar class names.

**Q: Can I embed text and image in one request?**

A: No. SigLIP2 accepts text-only or image-only inputs per request.
