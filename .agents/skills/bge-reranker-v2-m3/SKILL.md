---
name: bge-reranker-v2-m3-ascend-adapter
description: "AI-assisted adaptation of BGE-Reranker-V2-M3 for vLLM Ascend NPU — test config, tutorial doc, and SKILL.md workflow documentation"
---

# BGE-Reranker-V2-M3 Ascend Adaptation

## Overview

Adapt BAAI/bge-reranker-v2-m3 cross-encoder reranker for vLLM on Ascend NPU (Atlas A2/A3 series). This is a lightweight (278M params) encoder-only model based on RoBERTa architecture that computes relevance scores for (query, document) pairs. Unlike LLM-based rerankers, it does NOT require `hf_overrides`, custom chat templates, or architecture patches — works with vLLM's `score()` API out of the box via `runner="pooling"`.

## AI Collaboration Workflow

### Phase 1: Repository Exploration (Claude-led)

**Prompt**: "Explore the vllm-ascend repo structure — find existing test configs and tutorial docs as references"

**What worked**: Claude automatically discovered:
- `tests/e2e/models/configs/` with 20+ reference YAML configs
- `docs/source/tutorials/models/` with 30+ reference tutorial docs
- `.agents/skills/vllm-ascend-model-adapter/SKILL.md` as the canonical skill template
- Existing `bge-reranker-v2-m3.yaml` and `bge-reranker-v2-m3.md` already present (partial work from community)

**Key insight**: Always check what already exists before starting. Repo exploration cuts redundant work by 50%+.

### Phase 2: Gap Analysis (Claude-led)

**Prompt**: "Compare the existing bge-reranker config and doc against reference examples (Qwen3-8B config, Qwen3-Reranker doc). What's missing?"

**What worked**: Side-by-side comparison revealed:
- Config is structurally complete (model_name, model_type, hardware, serve, tasks, metrics, limit, batch_size)
- Doc has all required sections (Introduction, Environment Prep, Online/Offline Inference, Performance)
- Performance section has TBD placeholders — needs real benchmark data from A2/A3 hardware
- Doc lacks docker tabs for A2/A3 (present in reference docs but optional for encoder-only model)

### Phase 3: Benchmark Execution (User-led on Ascend hardware)

```bash
# Online benchmark
vllm bench serve \
  --model BAAI/bge-reranker-v2-m3 \
  --backend vllm-rerank \
  --dataset-name random-rerank \
  --host 127.0.0.1 \
  --port 8888 \
  --endpoint /v1/rerank \
  --tokenizer /root/.cache/bge-reranker-v2-m3 \
  --random-input 200 \
  --save-result \
  --result-dir ./

# Offline validation
python -c "
from vllm import LLM
model = LLM(model='BAAI/bge-reranker-v2-m3', runner='pooling')
outputs = model.score('What is the capital of China?', 
    ['The capital of China is Beijing.', 
     'Gravity is a force that attracts two bodies.'])
print([o.outputs.score for o in outputs])
# Expected: [~0.99, ~1e-6]
"
```

## Effective Prompts & Best Practices

### 1. Repo Exploration Prompt Pattern

```
"Find all existing [test configs | tutorial docs | reference files] for [model type] in [repo path]. 
Show me the directory listing and pick 2-3 representative examples to read."
```

**Why it works**: Gives Claude a bounded search scope with clear output expectations. "2-3 representative examples" prevents context overflow while ensuring coverage.

### 2. Gap Analysis Prompt Pattern

```
"Compare [file A] against [file B]. List: (1) what A has that B doesn't, (2) what B has that A doesn't, 
(3) what looks like a placeholder/TODO in either."
```

**Why it works**: Structured comparison forces explicit enumeration of differences. The "placeholder/TODO" check catches incomplete work that looks finished at first glance.

### 3. Parallelization Strategy

Always split independent work between human and AI:
- AI: code exploration, reference finding, documentation drafting, format validation
- Human: hardware-dependent work (benchmark runs, model downloads, NPU-specific debugging)

**Anti-pattern**: Having Claude write code while human waits. Instead: Claude drafts → human tests on hardware → Claude updates based on results.

### 4. SKILL.md Documentation

Document the AI collaboration as you go, not after. Include:
- Exact prompts that worked (copy-paste ready)
- Exact prompts that failed (and why)
- Files discovered vs files created
- Time saved vs doing it manually

## Deliverables Checklist

- [x] `tests/e2e/models/configs/bge-reranker-v2-m3.yaml` — verified structurally complete
- [x] `docs/source/tutorials/models/bge-reranker-v2-m3.md` — content complete, pending benchmark data
- [x] SKILL.md (this file) — AI-assisted workflow documentation
- [ ] Performance benchmark results (running on A2/A3)
- [ ] `docs/source/tutorials/models/index.md` — verify bge-reranker entry exists

## Model-Specific Notes

- **Architecture**: XLMRobertaForSequenceClassification (no special handling needed)
- **Runner**: `pooling` (cross-encoder, not LLM-based)
- **hf_overrides**: NOT needed (unlike Qwen3-Reranker which requires classifier_from_token)
- **Chat template**: NOT needed (unlike Qwen3-VL-Reranker which requires jinja template)
- **EP/flashcomm1/MTP**: Not applicable (non-MoE, non-Generative model)
- **ACLGraph**: Expected to work out-of-box for cross-encoder path
- **Max model len**: 8192 (config.json theoretical = model default)

## Lessons Learned

1. **Check before writing**: Both config and doc already existed. The task shifted from "create" to "verify + complete".
2. **Encoder-only models are simpler**: No hf_overrides, no chat templates, no MTP — the adaptation surface is much smaller than LLM-based rerankers.
3. **Benchmark data is the bottleneck**: The only blocking item is running actual hardware benchmarks. Documentation and configs can be prepared offline.
4. **Reference quality matters**: Qwen3-Reranker and Qwen3-VL-Reranker docs are excellent references — study them before writing your own.
