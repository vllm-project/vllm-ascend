# JEV-27B-VL

## 1 Introduction

JEV-27B-VL is a ~27-billion-parameter native vision-language model published on ModelScope by `autotrust`. It belongs to the Qwen3.8 family and uses the same `Qwen3_5ForConditionalGeneration` architecture as Qwen3.8-27B: a hybrid Mamba-Transformer backbone in which only 16 of the 64 layers run full (gated) attention (`full_attention_interval: 4`) and the other 48 run linear attention (Gated DeltaNet, GDN) with a constant recurrent state. The built-in 27-layer vision encoder makes it a true multimodal model — it processes text and images in the same forward pass — and it ships with a native 262,144-token context window.

What makes JEV-27B-VL distinctive is its **System 1 decision head**. The model card ships with a LoRA adapter (`adapter_vllm/`, rank 32 / alpha 64) that targets both the backbone projections and the `lm_head`. Once the adapter is loaded, a single engine exposes two capabilities:

- **System 2** — the unmodified base model through the standard OpenAI endpoints (`/v1/chat/completions`, `/v1/completions`). Use it for generation, explanation, rewriting and free-form reasoning.
- **System 1** — a classification/decision path through a thin `POST /v1/decide` route. It performs one forward pass and reads the `lm_head` log-probabilities of a small set of *verbalizer* tokens, producing a structured answer (`noul` yes/no, `score` 0–5, or `choice` 2–256 options) together with calibrated probabilities in roughly one decoded token. An optional adaptive-thinking mode can route hard cases to System 2 reasoning.

This document describes the main validation steps for serving JEV-27B-VL on Ascend NPUs, including supported features, prerequisites, deployment, functional verification, the System 1 decision head, performance characteristics and FAQs.

!!! note

    **No source-code patch is required on vLLM-Ascend v0.30.0rc1.** Because JEV-27B-VL uses the `Qwen3_5ForConditionalGeneration` architecture, which is natively supported by vLLM v0.30.0rc1, and its GDN linear-attention layers are handled by vllm-ascend's built-in GDN patch, the model runs out of the box on Ascend NPUs. The only addition is an application-level HTTP router (`serve_decide_030.py`) that adds the `/v1/decide` endpoint; it does not modify any vLLM or vllm-ascend source. Earlier v0.20.x releases needed three source patches; from v0.30.0rc1 they are no longer necessary.

This document is validated and written based on **vLLM v0.30.0rc1 + vllm-ascend v0.30.0rc1**.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) for the model's support matrix, and to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration details.

The following features were verified for JEV-27B-VL on v0.30.0rc1:

- Tensor parallelism (TP=2) on 2× Ascend 910C (64 GB) with BF16 weights.
- Native vision-language input (text + images) with `--limit-mm-per-prompt`.
- Hybrid GDN linear attention (built-in Triton/FLA kernels on Ascend).
- Automatic prefix caching (`--enable-prefix-caching`).
- LoRA, including **runtime adapter loading** through `POST /v1/load_lora_adapter`.
- `logprobs` readout (`--logprobs-mode processed_logprobs --max-logprobs 24`) for the decision head.

## 3 Prerequisites

### 3.1 Model Weight

| Weight / Artifact | Description | Download Link |
|---|---|---|
| `JEV-27B-VL` (BF16) | ~27B native VLM, ~51.75 GiB, 18 shards | [ModelScope](https://modelscope.cn/models/autotrust/JEV-27B-VL) |
| `adapter_vllm/` | Decision-head LoRA (rank 32, alpha 64), includes `lm_head` | bundled inside the model repository |

Hardware requirements: the BF16 model with a TP=2 deployment fits on **2× Ascend 910C (64 GB)** with a comfortable margin at `--gpu-memory-utilization 0.90` (verified). A larger `--max-model-len` (up to the native 262,144) increases KV-cache pressure proportionally.

> **Path description**: Download the model repository to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory. The adapter is served from `<model_dir>/adapter_vllm`.

### 3.2 Software

| Item | Version (validated) |
|---|---|
| vLLM | v0.30.0rc1 (V1 engine) |
| vllm-ascend | v0.30.0rc1 |
| CANN | 9.1.0 |
| torch / torch_npu | 2.10.x / 2.10.0.post4 |
| Python | 3.12.x |

## 4 Installation

### 4.1 vLLM Ascend

Install vLLM and vllm-ascend, either through the prebuilt Docker image (refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image)) or from source. Use v0.30.0rc1 or later for both components.

After installing, verify the imports work inside the container/environment:

```shell
python -c "import vllm, vllm_ascend; print('vllm and vllm_ascend are ready')"
```

Expected output:

```shell
vllm and vllm_ascend are ready
```

### 4.2 The decision-head server (application level)

To expose the System 1 `POST /v1/decide` endpoint, add a small HTTP router to the OpenAI server. The recommended way is a thin wrapper that patches `entry.build_app` before `vllm.entrypoints.launchers.api_server.entry` starts (for example, `serve_decide_030.py`, run exactly like `python -m vllm.entrypoints.openai.api_server`):

```python
import vllm.entrypoints.launchers.api_server.entry as entry

_orig_build_app = entry.build_app

def build_app(args, *a, **kw):
    app = _orig_build_app(args, *a, **kw)
    add_decide_router(app)          # registers POST /v1/decide, GET /v1/decide/info
    return app

entry.build_app = build_app
```

The router receives `{kind, state, question, options}` where `kind` is `noul` | `score` | `choice` and `state` may be free text, a JSON object, or a list mixing text and images (`{"image": "https://... | data:..."}`). It runs the LoRA-equipped backbone once and reads the verbalizer `lm_head` log-probabilities, returning calibrated probabilities, the chosen option, latency, and usage.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 System 2 (base model)

JEV-27B-VL runs out of the box. Start the standard server with the flags recommended for its hybrid backbone:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1   # logical device ids
export ASCEND_VISIBLE_DEVICES=6,7      # physical device ids (example)

python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/JEV-27B-VL \
  --served-model-name autotrust/JEV-27B-VL \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image": 8}' \
  --max-num-seqs 8 \
  --enforce-eager \
  --port 8000
```

### 5.2 System 1 + System 2 (with decision head)

Start the same server through the decision-head wrapper, enable LoRA and log-probability readout, and allow image decisions to carry their own chat template:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export ASCEND_VISIBLE_DEVICES=6,7
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1   # required for the runtime LoRA API
export JEV_DECIDE_ADAPTER=/path/to/JEV-27B-VL/adapter_vllm

python3 /path/to/serve_decide_030.py \
  --model /path/to/JEV-27B-VL \
  --served-model-name autotrust/JEV-27B-VL \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image": 8}' \
  --max-num-seqs 8 \
  --enforce-eager \
  --enable-lora \
  --max-lora-rank 32 \
  --max-logprobs 24 \
  --logprobs-mode processed_logprobs \
  --trust-request-chat-template \
  --port 8000
```

> `--max-num-seqs 8` is recommended for the decision head: with more than 8 sequences batched together, the LoRA path for this multimodal model class can return incorrect System 1 probabilities. `--trust-request-chat-template` is mandatory for vision decisions, which send the raw decision template with the request.

### 5.3 Loading the adapter at runtime

On vLLM v0.30.0rc1 there is no `--lora-modules` engine flag; load the adapter with the runtime LoRA API:

```bash
curl -X POST http://127.0.0.1:8000/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name": "jev-decision", "lora_path": "/path/to/JEV-27B-VL/adapter_vllm"}'
# -> Success: LoRA adapter 'jev-decision' added successfully.
```

Verify the adapter is registered:

```bash
curl http://127.0.0.1:8000/v1/models
# -> autotrust/JEV-27B-VL (parent: None); jev-decision (parent: autotrust/JEV-27B-VL)
```

From now on, address the decision head with `model: "jev-decision"` and the base model with `model: "autotrust/JEV-27B-VL"` on the same engine.

## 6 Functional Verification

After the service starts, invoke the model. Send a prompt to chat:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "autotrust/JEV-27B-VL",
    "messages": [{"role": "user", "content": "你好，你是谁？"}]
  }'
```

Expected result: HTTP 200 with generated text (the model self-identifies correctly as a Qwen-family assistant).

**Vision chat** (image understood natively):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "autotrust/JEV-27B-VL",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "What is shown in this photo?"},
      {"type": "image_url", "image_url": {"url": "https://example.com/concert.jpg"}}
    ]}]
  }'
```

### 6.1 Decision head — text

| kind | Request (state / question) | Result | Probabilities | Latency |
|---|---|---|---|---|
| `noul` | "三角形的内角和是 180 度。" — is it true? | `true` | `[0.0, 1.0]` | ~1.07 s |
| `choice` | 60 km/h × 2 h, how many km? (4 options) | `120公里` | `[0,1,0,0]` | ~0.26 s |
| `score` | normal pickup text, violation severity 0-5 | `0` (p=0.80) | — | ~0.25 s |
| `score` | offensive escalation text, severity 0-5 | `4~5` (p=0.67/0.19) | — | ~0.21 s |

### 6.2 Decision head — vision (multimodal)

| kind | Request | Result | Probabilities | Latency |
|---|---|---|---|---|
| `noul` | concert photo: is this a concert? | `true` | `[0,1]` | ~0.55 s |
| `noul` (negative) | same photo: is this a cat? | `false` | `[1,0]` | ~0.22 s |
| `score` | concert photo: how lively 0-5? | `5` (p=0.73) | — | ~0.21 s |

Vision decisions require `--trust-request-chat-template`.

### 6.3 Adaptive thinking (`POST /v1/decide`, `thinking` field)

| mode | Request | Result |
|---|---|---|
| `auto` | 50 m pool, 25 m swum — finished? (threshold 0.99) | System 1 confidence 1.0 ≥ threshold → no thinking, returns `false` |
| `on` | 5 apples +3 −2, how many left? (4 options) | triggers thinking (38 tokens, ~5.4 s) → `6个` (p≈1.0) |

When `thinking` is on, the base model reasons in its thinking mode over the same state and question; when the thinking channel closes, the answer-letter distribution is read in one step, and the final result is a weighted mix of the fast System 1 distribution and the System 2 distribution (`mix` default 0.5).

```
GET /v1/decide/info
{"protocol":"jev27-bare-v1","model_type":"qwen3_5","max_options":256,"native_choice_options":16, ...}
```

## 7 Accuracy

The decision head returns calibrated probabilities: decision logits are computed from the `lm_head`-LoRA log-probabilities of the verbalizer ids plus a per-kind bias, then renormalized with a per-kind temperature. On the tasks in its training domain (content safety, severity rating, classification, routing) the results are consistent and the probabilities are meaningful.

Two important caveats:

- The `score` head is trained for **task/severity semantics** (0 = none, 5 = severe), not as a generic quality/liking score. Asking it out-of-domain questions yields less intuitive numbers.
- Calibrated probabilities reflect the model's self-report; treat them as a decision aid, and spot-check in safety-critical scenarios.

## 8 Performance Evaluation

System 1 performs a single forward pass and reads one token, so its latency is nearly constant regardless of task verbosity; System 2 latency grows with the length of the generated reasoning. Representative values on 2× Ascend 910C (TP=2, BF16, v0.30.0rc1):

| Path | Latency (representative) | Tokens |
|---|---|---|
| System 1 decision (1-token readout) | ≈ 0.2 – 1.1 s (differs with first-run / prefix-cache hit) | 1 |
| System 2 chat / thinking | ≈ 5.4 s for the 38-token reasoning example | tens – hundreds |

Prefix-cache hits significantly speed up repeated decision calls (vision decisions measured ≈ 2.2 s on first run → ≈ 0.80 s on a cache hit). Figures are representative demo values; measure with your own workload and batch size.

## 9 Performance Tuning

Recommended flags for this model (do not remove):

- `--mamba-cache-mode align` — required alignment of the hybrid (Mamba/GDN + attention) cache.
- `--enforce-eager` — avoids graph-mode capture issues with the LoRA + multimodal path (verified).
- `--logprobs-mode processed_logprobs --max-logprobs 24` — needed to read the verbalizer log-probabilities for the decision head.
- `--trust-request-chat-template` — required for vision decisions.
- `--max-num-seqs 8` — keeps batched LoRA decoding within the exact System 1 probability regime.

Tuning knobs:

- `--max-model-len`: native context is 262,144; the validation used 32,768. A full 256K prompt needs roughly 17 GB of KV cache.
- `--gpu-memory-utilization`: standard vLLM knob controlling the KV-cache allocation share (validated at 0.90 on 2× 910C).
- `--enable-prefix-caching` (with optional `--prefix-match-unit`) helps decision-heavy workloads with long shared prefixes.
- Use **logical device ids** (`ASCEND_RT_VISIBLE_DEVICES`) rather than physical ids to avoid `aclInit` error 107001 in multi-device setups.

## 10 FAQ

- **The decision head is not active.** The LoRA module name must appear in the `model` field (`jev-decision`); the completions path ignores a separate `lora_request`. Checking it via `/v1/models` confirms the adapter is registered.
- **`POST /v1/load_lora_adapter` returns 404.** `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` must be set before the server starts.
- **`aclInit` reports 107001.** Use logical device numbers (`ASCEND_RT_VISIBLE_DEVICES=0,1`) instead of physical ids.
- **The service crashes when a flag is removed.** The flags listed in [Performance Tuning](#9-performance-tuning) are required for this model + decision head combination.
- **First call is slow, later calls are fast.** The decision endpoint is a 1-token decode; prefix-cache hits make repeat calls much faster.
- **How do I stop the server?** `kill -TERM $(pgrep -f serve_decide_030.py)`. Avoid `pkill -f`, which matches the command line of the shell that launched it.

For general environment, installation and parameter questions, refer to the [Public FAQs](../../faqs.md).
