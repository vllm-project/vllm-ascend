# LocateAnything-3B

## 1 Introduction

LocateAnything-3B is a visual grounding model that outputs bounding boxes as text tokens (`<box><x1><y1><x2><y2></box>`). Its architecture consists of a MoonViT vision tower, a projector (`mlp1`), and a Qwen2-2B language model, plus an MTP (multi-token prediction) head that decodes a box in a few forward steps using a custom "6-token window + `handle_pattern`" rule-based decoder.

That MTP decoder is model-specific and is **not** implemented in vLLM. This tutorial therefore deploys the model in a **decoupled (encoder-disaggregated) topology**:

- The vision tower, projector and `embed_tokens` run in a client process on an Ascend NPU.
- The Qwen2 language model runs in vLLM Ascend and receives pre-computed embeddings through `--enable-prompt-embeds`.

```text
+-----------------------------+          +------------------------------------+
| Client process (NPU A)      |          | vLLM Ascend server (NPU B)         |
|                             |          |                                    |
| image -> MoonViT -> mlp1    | prompt_  | Qwen2ForCausalLM (36 layers,       |
|       -> embed_tokens       | embeds   | hidden 2048, vocab 152681)         |
|       -> prompt_embeds  ----+--------->+ -> plain autoregressive decoding   |
+-----------------------------+          +------------------------------------+
```

Because MTP is bypassed, decoding falls back to plain autoregressive generation. Accuracy is preserved (see [Chapter 7](#7-accuracy-evaluation)); single-request latency is higher than the native implementation, while throughput scales through continuous batching (see [Chapter 8](#8-performance-evaluation)).

The measurements in this tutorial were taken on Atlas 800 A2 (64GB x 8) with vLLM Ascend v0.18.0rc1. The commands below use the version placeholder configured by the documentation build system; re-verify the numbers after upgrading to a later release.

## 2 Supported Features

The table below describes which parts of the model are served by vLLM Ascend in this topology.

| Feature | Status | Note |
| ------- | ------ | ---- |
| Text-only Qwen2 LM path | ✅ | Runs natively through `Qwen2ForCausalLM` |
| `--enable-prompt-embeds` | ✅ | Verified on Ascend; embeddings are accepted per request |
| BF16 weights | ✅ | `torch_dtype=bfloat16` |
| Automatic prefix cache | ✅ | About +10% throughput when requests repeat |
| Chunked prefill | 🟡 | Not specifically validated for `prompt_embeds`; keep `--max-num-batched-tokens` below `--max-model-len` |
| Vision tower inside vLLM | ❌ | Runs in the client process; not a vLLM multimodal model |
| MTP / multi-token prediction | ❌ | Custom rule-based decoder, not implemented upstream |
| Speculative decoding | ❌ | No draft model or MTP support for this architecture |

Refer to [Supported Features List](../../user_guide/support_matrix/supported_features.md) for the generic feature matrix and to [Feature Guide](../../user_guide/feature_guide/index.md) for configuration details.

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version | Hardware Requirements | Download Links |
| -------------- | --------------------- | -------------- |
| `LocateAnything-3B` (BF16) | 2 NPUs: 1 for the vision client, 1 for the LM server | [AtomGit](https://atomgit.com/Ascend-SACT/LocateAnything-3B) |

The original weights ship as a single checkpoint that contains both the vision tower and the language model. They must be split before serving; see [Section 5.1](#51-split-the-checkpoint).

### 3.2 Environment

- The client process needs `transformers`, `torch_npu`, `pillow`, `safetensors` and `openai` (all present in the vLLM Ascend image).
- The client and the server can run in the same container on different NPUs, or in two containers on different nodes. In both cases the client talks to the server over the OpenAI-compatible HTTP API.

## 4 Installation

### 4.1 Docker Image Installation

=== "A2/A3"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    export NAME=vllm-ascend

    docker run --rm \
      --name $NAME \
      --net=host \
      --shm-size=16g \
      --device /dev/davinci0 \
      --device /dev/davinci1 \
      --device /dev/davinci_manager \
      --device /dev/hisi_hdc \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /etc/hccn.conf:/etc/hccn.conf \
      -v /path/to/LocateAnything-3B:/models/LocateAnything-3B:ro \
      -it $IMAGE bash
    ```

    Install the client-side dependency if the image does not ship it:

    ```bash
    pip install -q peft==0.19.1
    ```

## 5 Decoupled Service Deployment

### 5.1 Split the checkpoint

Run the split once. It writes a standalone Qwen2 model directory plus the embedding table, because `embed_tokens` is needed by the client while the LM runs in vLLM.

```python
import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import save_file

src = "/models/LocateAnything-3B"
out = "/models/split"
lm_dir = os.path.join(out, "locate_qwen2_model")
os.makedirs(lm_dir, exist_ok=True)

model = AutoModel.from_pretrained(
    src, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="sdpa",
).to("npu").eval()

# The client needs embed_tokens, so keep it as a standalone table.
save_file(
    {"weight": model.language_model.model.embed_tokens.weight.detach().cpu()},
    os.path.join(out, "qwen2_embed_tokens.safetensors"),
)

# The LM half becomes a plain Qwen2ForCausalLM directory.
model.language_model.save_pretrained(lm_dir)
AutoTokenizer.from_pretrained(src, trust_remote_code=True).save_pretrained(lm_dir)
```

The client keeps using `AutoProcessor`/`AutoTokenizer` from the original checkpoint path, because the image processor is model-specific.

After splitting, `/models/split/locate_qwen2_model` contains a plain `Qwen2ForCausalLM` (36 layers, hidden size 2048, vocab size 152681) that vLLM can load without `--trust-remote-code`.

### 5.2 Start the language model server

The LM server only hosts the text model. `--enable-prompt-embeds` is required; `--max-model-len` only needs to cover visual tokens (about 1.5k at 512 px) plus the output (about 16 tokens).

```bash
export ASCEND_RT_VISIBLE_DEVICES=1

vllm serve /models/split/locate_qwen2_model \
  --served-model-name locate_qwen2_model \
  --enable-prompt-embeds \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.5 \
  --host 0.0.0.0 \
  --port 33333
```

Startup takes about 90–100 s on A2 because ACLGraph compilation is included. When it is ready, `curl -s http://localhost:33333/health` returns `200`, and the engine logs report the KV cache size, for example `24.5 GiB / 713,600 tokens` with `--gpu-memory-utilization 0.5`.

### 5.3 Run the vision tower client

The client builds the prompt, replaces the image placeholder tokens with the projected vision features, and sends the resulting embeddings through `prompt_embeds`. Each request carries about 1.3 MB of BF16 embeddings at 512 px.

```python
import io, pybase64, torch, torch.nn as nn
from PIL import Image
from openai import OpenAI
from safetensors.torch import load_file
from transformers import AutoModel, AutoProcessor

MODEL = "/models/LocateAnything-3B"
SPLIT = "/models/split"
client = OpenAI(api_key="EMPTY", base_url="http://localhost:33333/v1")

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="sdpa",
).to("npu").eval()
# Keep only the vision tower + projector on the client.
model.language_model.to("cpu")
del model.language_model
torch.npu.empty_cache()

# Standalone embedding table, extracted in Section 5.1.
embed_tokens = nn.Embedding(152681, 2048, padding_idx=None)
embed_tokens.load_state_dict(load_file(f"{SPLIT}/qwen2_embed_tokens.safetensors"))
embed_tokens = embed_tokens.to("npu").eval()

def build_embeds(image: Image.Image, phrase: str) -> str:
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    prompt = f"Locate a single instance that matches the following description: {phrase}."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("npu")

    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    input_ids = inputs["input_ids"]
    grid = inputs.get("image_grid_hws")
    if grid is not None and not torch.is_tensor(grid):
        grid = torch.from_numpy(grid).to("npu", dtype=torch.int32)

    with torch.no_grad():
        vit = model.extract_feature(pixel_values, grid)
        if isinstance(vit, list):
            vit = torch.cat(vit, dim=0)
        vit = model.mlp1(vit)

        emb = embed_tokens(input_ids).to(torch.bfloat16)
        b, n, c = emb.shape
        emb = emb.reshape(b * n, c)
        sel = input_ids.reshape(b * n) == model.image_token_index
        emb[sel] = vit.reshape(-1, c).to(emb.device)
        emb = emb.reshape(b, n, c).squeeze(0).clone().detach()

    buf = io.BytesIO()
    torch.save(emb, buf)
    return pybase64.b64encode(buf.getvalue()).decode()

image = Image.open("demo.jpg").convert("RGB")
resp = client.completions.create(
    model="locate_qwen2_model",
    prompt=None,
    max_tokens=64,
    temperature=0.0,
    extra_body={
        "prompt_embeds": build_embeds(image, "the bed"),
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    },
)
print(resp.choices[0].text)   # <box><x1><y1><x2><y2></box>
```

Coordinates are normalized to 1000, so the pixel box is `x * W / 1000`, `y * H / 1000`.

## 6 Functional Verification

Send the 10 images from the model repository demo and check that each response contains exactly one `<box>` element:

```bash
# text output: 10/10 responses contain a parseable box
grep -c "<box>" outputs.txt
```

Expected output for a 512 px input on A2:

```text
<box><123><456><789><901></box>
```

## 7 Accuracy Evaluation

Because a bounding-box prompt has a single reference answer, IoU against the ground-truth box is used instead of a text benchmark. On the 10-image demo set (`ground_single` prompt, `temperature=0`):

| Configuration | IoU mean | IoU min |
| ------------- | -------- | ------- |
| `max-size 512` | **96.3%** | 92.7% |
| `max-size 768` | 95.9% | 92.3% |
| `max-size 384` | 87.2% | 8.6% |

The pure autoregressive path is at least as accurate as the native MTP path (94.4% on the same set), because the rule-based MTP decoder occasionally stops early and produces a worse box on some samples.

!!! note

    Greedy decoding on this model is not bit-wise reproducible: high-concurrency batching changes reduction order, so near-tie tokens can flip. Expect about ±1.5 percentage points of IoU variation between runs on small evaluation sets. Disabling the prefix cache does not change this.

## 8 Performance Evaluation

Measured on one Atlas 800 A2 node (910B3, 64GB), `max-size 512`, 10 demo images, 100 requests, 0 errors.

Single request, with the two stages broken down:

| Stage | Latency |
| ----- | ------- |
| Vision tower + projector (client) | 73.9 ms |
| Prefill / TTFT (server) | 121.2 ms |
| Decode, about 15.6 output tokens | 470 ms |
| **End to end** | **686 ms** |
| TPOT | 32.2 ms |

Under load (single client process, one NPU for the vision tower, one for the server):

| Concurrency | Throughput | TTFT p95 | TPOT p95 | E2E p95 |
| ----------- | ---------- | -------- | -------- | ------- |
| 8 | 8.08 img/s | 200.7 ms | 53.5 ms | 1376 ms |
| 32 | 9.64 img/s | 215.3 ms | 64.8 ms | 3842 ms |

The server alone, with pre-computed embeddings, reaches 21–22 req/s at concurrency 32–64. The end-to-end ceiling of a single client process comes from the client, not from vLLM; see [Section 9.3](#93-model-specific-optimizations).

Scaling the vision stage horizontally (4 client processes, one NPU each, sharing one server):

| Client processes | NPUs used | Throughput |
| ---------------- | --------- | ---------- |
| 1 | 2 | 8.08–9.64 img/s |
| 4 | 5 | **22.9 img/s** |

## 9 Performance Tuning

### 9.1 Recommended Configurations

| Scenario | NPUs | Key settings | Main consideration |
| -------- | ---- | ------------ | ------------------ |
| Low concurrency, lowest latency | 2 | `--max-num-seqs 8`, `max-size 512` | Single-stream latency dominates; MTP is unavailable, so expect about 690 ms per image |
| Online serving, medium concurrency | 2–3 | `--max-num-seqs 32`, `max-size 512` | 8–9.6 img/s per client process |
| Offline high throughput | 5+ | `--max-num-seqs 32`, 4 client processes | Near-linear scaling with client processes |

### 9.2 Tuning Guidelines

Tune in this order; each step was measured on the configuration above.

1. **Input resolution (`max-size` in the client).** This is the dominant knob because it changes the number of vision tokens. Going from 768 to 512 gives +9% throughput at concurrency 8 and +38% at concurrency 32, with slightly better IoU. Do not go below 512 on this model: at 384 the IoU collapses to 87.2%.
2. **Client concurrency.** End-to-end throughput saturates around concurrency 8–32 per client process. Beyond that, queueing grows faster than throughput.
3. **Number of client processes.** The vision stage is serialized per process, and the image processor holds the GIL, so a single process caps at about 7–9 img/s. Adding one client process per NPU scales throughput nearly linearly.
4. **Server knobs.** `--max-num-seqs` beyond 32 does not help (21.06 req/s at 32 vs 22.62 at 64). Keep the default prefix cache: disabling it costs about 10%.
5. **Graph mode.** ACLGraph is enabled by default (`enable_npugraph_ex`); no extra flags were needed for this workload.

### 9.3 Model-Specific Optimizations

- **The vision tower must be serialized inside a client process.** Running MoonViT concurrently on one NPU produces silently wrong boxes (IoU drops from 96% to 82.7%, with some samples at 0.0) rather than raising an error. Guard `extract_feature` with a lock, or use one process per NPU.
- **Do not batch images through the vision tower on Ascend.** Batching 8 images into one `extract_feature` call is about 5x slower per image than running them one by one (516 ms/img vs 104 ms/img), and processor-level batching with variable shapes returns the wrong number of feature tokens. Feed one image per call.
- **The image processor has no fast implementation.** `AutoProcessor(..., use_fast=True)` falls back to the slow path, so preprocessing costs about 35 ms per image at concurrency 1 and about 140 ms under GIL contention at concurrency 32. That is the practical ceiling of a single client process.
- **Pre-compute embeddings when the same image is queried repeatedly** (for example many prompts on one image). The vision stage then disappears from the critical path, and the server-only ceiling of 21–22 req/s applies.

## 10 FAQ

### Q1: Is `--enable-prompt-embeds` supported on Ascend?

Yes. It is validated with this model on A2/A3: the engine accepts per-request `prompt_embeds` and the scheduler handles them like ordinary prefill tokens. Comments in older example scripts that claim the feature needs confirmation are outdated.

### Q2: Why is one request slower than the native implementation?

Because MTP is not available. The native decoder needs about 3.9 forward steps for 15.6 tokens, while plain autoregressive decoding needs about 14.8 steps. Each step is faster in vLLM (32.2 ms vs 79.7 ms), but not enough to compensate (686 ms vs 538 ms end to end).

### Q3: Throughput stays at about 8–9 img/s no matter how high I set the concurrency. Why?

The limit is the client, not the server. Measure the server alone by sending cached `prompt_embeds`; it reaches 21–22 req/s. To exceed that, add client processes (one NPU each).

### Q4: Can I batch several images in one `extract_feature` call?

No. On Ascend this is a 5x regression per image, and processor batching returns a wrong number of feature tokens. Always pass one image per call.

### Q5: Why do the boxes differ slightly between runs with `temperature=0`?

Greedy decoding is not bit-wise reproducible under batching; near-tie tokens flip when the reduction order changes. This affects about ±1.5 percentage points of IoU on small evaluation sets. It is unrelated to the prefix cache.

### Q6: The server reports OOM during startup. What should I check?

This LM is small (about 3 GB in BF16), so OOM usually comes from `--gpu-memory-utilization` being shared with the client process on the same NPU. Give the vision client and the server separate NPUs, or lower `--gpu-memory-utilization`.
