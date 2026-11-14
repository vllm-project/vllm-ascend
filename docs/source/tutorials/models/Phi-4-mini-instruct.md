# Phi-4-mini-instruct

## 1 Introduction

Phi-4-mini-instruct is a 3.8B-parameter, text-only, instruction-tuned large language model open-sourced by Microsoft. The config declares `Phi3ForCausalLM`, so it runs on vLLM-Ascend **with zero adaptation code** (natively registered in vLLM's model registry).

Key architecture facts (verified from the model `config.json`, cross-checked on the raw file and the local snapshot):

| Item | Value | Note |
|---|---|---|
| Architecture | `Phi3ForCausalLM` | natively registered, no remote code needed for the model |
| hidden_size / layers | 3072 / 32 | |
| Attention heads | 24 Q / 8 KV | |
| head_dim | 128 (3072 / 24) | within the Ascend FlashAttention support set {64, 128, 192} |
| vocab_size / tied embeddings | 200064 / true | |
| RoPE | partial rotary 0.75 + LongRoPE (4096 -> 131072) | >4k contexts use the long factor domain |
| sliding_window | 262144 | effectively full attention |
| Tokenizer | tiktoken-style `tokenizer.json` bundled | loads without `--trust-remote-code` |
| Chat template | user-only single turn, no system role | embedded in tokenizer config |

This document is validated on **vLLM-Ascend v0.26.0rc1**.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model's supported feature matrix.

## 3 Prerequisites

### 3.1 Model Weight

The model is about 7.7 GB in bf16 and runs on a single Atlas A2 card (64 GB).

Download via ModelScope:

```shell
pip install modelscope
modelscope download --model LLM-Research/Phi-4-mini-instruct --local_dir ./Phi-4-mini-instruct
```

Or from Hugging Face:

```shell
HF_ENDPOINT=https://hf-mirror.com hf download microsoft/Phi-4-mini-instruct --exclude '*.pth'
```

### 3.2 Environment

| Component | Version |
|---|---|
| vllm-ascend | v0.26.0rc1 (commit f2f74a16c) |
| vllm | 0.26.0 |
| torch_npu | 2.10.0.post4 |
| CANN / driver | 9.1.0 / 26.1.1 |

## 4 Deployment

Phi-4-mini is not on the ACL-graph whitelist in this version, so start with `--enforce-eager`:

```shell
export ASCEND_RT_VISIBLE_DEVICES=0
vllm serve LLM-Research/Phi-4-mini-instruct \
    --dtype bfloat16 --max-model-len 4096 --enforce-eager \
    --gpu-memory-utilization 0.8 --port 8000
```

Notes:

- `--trust-remote-code` is **not** required: the config carries an `auto_map` entry for the tokenizer, but `tokenizer.json` ships with the model and loads directly. If a future environment fails to load the tokenizer, retry with `--trust-remote-code`.
- Verified context length in this document: **4096**. Contexts beyond 4k switch into the LongRoPE long-factor domain; validate before relying on them.

## 5 Functional Verification

### 5.1 Offline Inference

```python
from vllm import LLM, SamplingParams

llm = LLM(model="LLM-Research/Phi-4-mini-instruct", dtype="bfloat16",
          max_model_len=4096, enforce_eager=True)
messages = ["<|user|>\nWrite a Python function to check palindromes.<|end|>\n<|assistant|>"]
out = llm.generate(messages, SamplingParams(temperature=0, max_tokens=256))
print(out[0].outputs[0].text)
```

A full offline comparison suite (greedy vs a CPU bf16 reference, sampling self-consistency, boundary cases) is available at `tests/e2e/pull_request/one_card/test_phi4_mini_offline.py`.

### 5.2 Online Inference

With the server from section 4 running:

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LLM-Research/Phi-4-mini-instruct",
    "messages": [{"role": "user", "content": "Write a Python function to check palindromes."}],
    "max_tokens": 256
  }'
```

## 6 Accuracy Evaluation

### 6.1 lm_eval (nightly harness)

The model is registered in the accuracy nightly under `pr-accuracy-group-1` (single card). Run locally with:

```shell
pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/Phi-4-mini-instruct.yaml --tp-size 1 \
  --report-dir ./benchmarks/accuracy
```

### 6.2 Results

| Benchmark | Metric | Value |
|---|---|---|
| gsm8k (limit 200, 5-shot) | strict-match | 0.81 (3-run median) |
| gsm8k (limit 200, 5-shot) | flexible-extract | 0.81 (3-run median) |
| aisbench gsm8klite (100 samples) | accuracy | 0.86 |
| serve smoke (zh/en chat) | manual check | pass |
| offline greedy vs HF bf16 (deterministic prompts) | prefix token equality | up to 32/32 |

Thresholds are calibrated on a single Atlas A2 card with 3 runs (median, RTOL 0.05).

ceval-valid was evaluated during calibration (acc 0.476 at limit 200) and dropped from the nightly config: as an English-centric model this score reflects real capability rather than an engine issue, and a bonus threshold within RTOL 0.05 could not be sustained. Chinese coverage is provided by the one-card e2e cases instead.

## 7 FAQ

**Q: The engine fails to start with ACL-graph errors.**
The model is outside the graph-mode whitelist in this version. Use `--enforce-eager`.

**Q: How was head_dim determined?**
The config has no explicit `head_dim`. It is `hidden_size / num_attention_heads` = 3072 / 24 = 128, which is in the Ascend FlashAttention support set {64, 128, 192}. Always verify head counts from the raw config before computing; second-hand summaries can be wrong (e.g. assuming 32 heads yields the incorrect 96).

**Q: Long-context outputs look different from the CPU reference beyond 4k.**
Contexts >4096 activate LongRoPE long factors, a known precision-divergence area. Cross-framework token equality is not a meaningful assertion there; validate long-context quality with accuracy benchmarks instead and only claim lengths you have measured.

**Q: Why do greedy outputs on open-ended prompts differ between CPU bf16 and NPU bf16?**
near-tie tokens at an early position can flip due to different matmul accumulation orders; both continuations are valid. Use deterministic prompts (code, canonical answers) for token-level comparisons and accuracy benchmarks for open-ended quality.
