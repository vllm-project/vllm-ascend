# vllm-ascend-model-adapter — Usage Guide

## What this skill does

`vllm-ascend-model-adapter` is an AI agent skill that adapts a Hugging Face or local model checkpoint to run on Ascend NPU via `vllm-ascend`. It covers the full lifecycle: pre-flight analysis, Ascend compatibility assessment, code implementation, two-stage hardware validation, and delivery of a single signed commit with documentation.

It handles both cases:

- **Already-supported architectures** that fail to start or produce errors on Ascend.
- **New architectures** not yet registered in vLLM at all.

---

## When to use it

| Situation | Use this skill? |
| --- | --- |
| A new model needs to run on Ascend NPU for the first time | Yes |
| An existing vLLM-supported model fails on Ascend (op error, shape error, boot failure) | Yes |
| A model runs on GPU but needs Ascend-specific feature validation (ACLGraph, EP) | Yes |
| You only want to upgrade a dependency or do unrelated refactoring | No |

---

## Prerequisites

### Hardware

- Ascend A2 server (8-NPU, TP=8 typical) or Ascend A3 server (16-NPU, TP=16 typical).

### Software

- The official `vllm-ascend` Docker image (see `./Dockerfile`).
- All implementation is done inside the container at fixed paths:

```text
/vllm-workspace/vllm          ← vLLM source
/vllm-workspace/vllm-ascend   ← vllm-ascend source
/workspace                    ← working directory for vllm serve
/models/<model-name>          ← model checkpoint root
```

### Before invoking the skill

Verify that the runtime imports from the correct path:

```bash
python -c "import vllm; print(vllm.__file__)"
# Expected: /vllm-workspace/vllm/...
```

---

## How to invoke

In Claude Code, use the skill by name:

```text
/vllm-ascend-model-adapter
```

Then tell the agent:

- The model name or path (e.g. `Qwen3-30B-A3B`, `/models/Qwen3-30B-A3B`).
- Any additional requirements (non-default TP size, specific feature to verify, etc.).

The agent will confirm the model path, implementation roots, and delivery repo before starting.

---

## 10-Step Workflow

### Step 1 — Collect context

The agent confirms all paths and the default feature set to validate:

| Feature | Applies to |
| --- | --- |
| ACLGraph | All models |
| MTP | Only if checkpoint explicitly supports it (determined in Step 2) |
| EP (expert parallel) | MoE models only |
| flashcomm1 | MoE models only |
| Multimodal (VL) | Vision-language models only |

User requirements extend this baseline — they do not replace it.

---

### Step 2 — Analyze model

The agent inspects `config.json`, modeling files, processor files, tokenizer, and the safetensors weight index to determine:

**Model type classification:**

```text
LLM ─┬─ Standard full attention
     ├─ Sliding window attention
     ├─ Mamba (SSM)
     ├─ Multi-latent attention (MLA)
     └─ Hybrid (combination of the above)

VLM  ─── Vision-language model (with vision encoder / multimodal processor)

Whisper ─ Encoder-decoder ASR model
```

The model type drives which features are applicable and which validation steps apply.

---

### Step 3 — Operator compatibility gate (early-exit decision point)

The agent scans all new operators in the model code and classifies them:

| Operator type | Ascend support | Action |
| --- | --- | --- |
| **Torch** (native PyTorch) | ✅ Functional | Note performance uncertainty in report |
| **Triton** kernel | ⚠️ Uncertain | Explicit functional + accuracy verification required on Ascend |
| **CUDA** kernel with fallback | ❌ CUDA unsupported | Use fallback; document the path |
| **CUDA** kernel, **no fallback** | ❌ Blocked | **Early exit — file GitHub issue immediately** |

> **CUDA early-exit rule**: If any operator is a pure CUDA kernel with no Torch/Triton alternative, the agent stops, skips all validation, and files a GitHub issue documenting the blocking operator, why no fallback exists, and the recommended path forward.
>
> **Triton early-exit rule**: If a Triton kernel is verified to be non-functional on Ascend (correctness failure or unacceptable accuracy degradation), the agent stops and files a GitHub issue documenting which kernel fails, the observed failure mode, and the recommended path forward (e.g., replace with a Torch-native fallback or implement a custom Ascend op).

---

### Step 4 — Framework-side code analysis

The agent identifies vLLM framework modules changed alongside the model (scheduler, attention backend, sampler, weight loader, worker, etc.) and checks whether `vllm-ascend` already covers them:

```text
Changed vLLM module
        │
        ├─ Already patched/overridden by vllm-ascend?
        │       └─ YES → Check if the existing patch still applies correctly.
        │                If it needs updating, update it; otherwise no action needed.
        │
        └─ NOT covered + contains Ascend-incompatible logic?
                └─ YES → Add minimal override under /vllm-workspace/vllm-ascend/
```

Patches are kept minimal and scoped to the incompatible paths only.

---

### Step 5 — Choose adaptation strategy

| Situation | Strategy |
| --- | --- |
| Architecture exists in `registry.py` and is compatible | Reuse; patch only what's broken |
| Architecture missing or incompatible | Implement new adapter in `vllm/model_executor/models/`, register in `registry.py` |
| Remote code needs newer `transformers` symbols | Copy required files from source — **never upgrade `transformers`** |
| Failure requires model-specific code in `vllm-ascend` | **Do not proceed** — raise a GitHub issue to analyze the root cause |

New adapter implementation checklist:

1. `vllm/model_executor/models/<new_model>.py` — model adapter
2. `vllm/transformers_utils/processors/<new_model>.py` — processor (VL models only)
3. `vllm/model_executor/models/registry.py` — architecture registration
4. Explicit weight loader/remap rules for: qkv sharding, QK/KV norms, RoPE variants, fp8 scale pairing

---

### Step 6 — Implement minimal code changes (vLLM source only)

All model adaptation code is implemented in `/vllm-workspace/vllm` only. Do not introduce model-specific files or patches in `/vllm-workspace/vllm-ascend` — if a model cannot function on Ascend without that, raise a GitHub issue instead. Weight mapping is kept explicit and auditable. No unrelated refactors.

Syntax check after implementation:

```bash
python -m py_compile /vllm-workspace/vllm/vllm/model_executor/models/<new_model>.py
```

---

### Step 7 — Two-stage validation on Ascend

Both stages must be completed before sign-off.

#### Stage A: Dummy fast gate

```bash
cd /workspace
vllm serve /models/<model-name> \
  --served-model-name <served-name> \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --tensor-parallel-size <TP> \
  --max-num-seqs 16 \
  --load-format dummy \
  --port 8000
```

Validates: architecture path, operator path, API path, ACLGraph evidence.

> **`Application startup complete` alone is NOT a pass. A smoke inference request is mandatory.**

#### Stage B: Real-weight mandatory gate

Remove `--load-format dummy`. Same command otherwise.

Validates: weight key mapping, fp8/fp4 dequantization, KV/QK norm sharding with real tensor shapes, load-time/runtime stability.

> **Stage B is mandatory. You cannot sign off adaptation on dummy-only evidence.**

---

### Step 8 — Validate inference and features

Minimum smoke sequence:

```bash
# 1. Readiness
curl -sf http://127.0.0.1:8000/v1/models

# 2. Text request (all models)
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served-name>","messages":[{"role":"user","content":"say hi"}],"temperature":0,"max_tokens":16}'

# 3. Text+image request (VL models only)
```

Feature status is reported using four categories:

| Symbol | Meaning |
| --- | --- |
| ✅ | Supported and verified |
| ❌ | Framework-level unsupported |
| ⚠️ | Checkpoint missing (weights/config don't provide the feature) |
| N/A | Not applicable (e.g. EP/flashcomm1 on non-MoE models) |

Capacity baseline: `max-model-len=128k` + `max-num-seqs=16`. Expand to 32/64 seqs if requested.

> **Note**: Accuracy evaluation and performance benchmarking are out of scope for this skill. They are handled by a dedicated separate skill. Invoke that skill after completing this step if needed.

---

### Step 9 — Backport, generate artifacts, commit

1. Backport minimal diff from `/vllm-workspace/*` to the delivery repo.
2. Generate `tests/e2e/models/configs/<ModelName>.yaml` (model_name, hardware, tasks with accuracy metrics, num_fewshot).
3. Generate `docs/source/tutorials/models/<ModelName>.md` (Introduction, Supported Features, Environment Preparation, Deployment, Functional Verification, Accuracy Evaluation, Performance).
4. Update `docs/source/tutorials/models/index.md`.
5. One single signed commit: `git commit -sm "..."`.

---

### Step 10 — Handoff artifacts

The final response includes (all in Chinese, compact):

- **Analysis report**: architecture summary, root causes, code changes, feature status matrix, dummy-vs-real validation matrix, theoretical vs practical max-model-len, fallback ladder evidence.
- **Runbook**: server startup command, validation curl commands, eager/TorchDynamo fallbacks.
- SKILL.md summary posted as a comment on the originating GitHub issue.

---

## Fallback ladder

When startup or inference fails, the agent follows this ordered ladder:

```text
1. Reproduce to get a deterministic failure signature
        ↓
2. Add --enforce-eager  (isolate graph-capture issues)
        ↓
3. [VL] Add TORCHDYNAMO_DISABLE=1  (dynamo + interpolate + NPU contiguous errors)
        ↓
4. [VL] Text-only isolation:  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'
   (separate processor issues from model core issues)
        ↓
5. Apply targeted code fix → loop back to Stage A
```

---

## Special cases

| Scenario | Action |
| --- | --- |
| **FP8 checkpoint on Ascend A2/A3** | Dequantize fp8→bf16 at load time; never force fp8 execution kernels |
| **QK norm mismatch under TP** (e.g., `128 vs 64`) | Detect KV-head replication; use local norm-shard path |
| **ACLGraph capture error 507903** | Use `HCCL_OP_EXPANSION_MODE=AIV`; reduce `--max-model-len` |
| **HCCL port bind error** | Kill stale `vllm serve` processes; free port 8000 |
| **Architecture not recognized** | Add mapping in `registry.py`; match class name exactly |
| **MLA runtime failure** (`AtbRingMLA`) | Reproduce with minimal request; eager isolation; fix model/backend code |

---

## Quality gates — cannot sign off without these

- [ ] Service starts from `/workspace` on port 8000
- [ ] At least one text inference request returns HTTP 200 + non-empty output
- [ ] VL models: at least one text+image request returns HTTP 200
- [ ] ACLGraph / EP / flashcomm1 / multimodal all reported (with status); MTP reported if checkpoint supports it
- [ ] `128k + bs16` capacity baseline reported (or explicit reason if not feasible)
- [ ] **Real-weight Stage B evidence present** (dummy-only is never sufficient)
- [ ] `tests/e2e/models/configs/<ModelName>.yaml` exists with correct schema
- [ ] `docs/source/tutorials/models/<ModelName>.md` exists with all sections
- [ ] `docs/source/tutorials/models/index.md` updated
- [ ] Exactly one signed commit in the delivery repo
