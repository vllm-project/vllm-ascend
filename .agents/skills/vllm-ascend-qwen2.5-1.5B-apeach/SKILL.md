---
name: vllm-ascend-qwen2.5-1.5B-apeach-adapter
description: "AI-assisted adaptation of jason9693/Qwen2.5-1.5B-apeach for vLLM-Ascend — test config, tutorial doc, and validation workflow"
---

# Qwen2.5-1.5B-apeach Adaptation SKILL.md

## Task Overview

- **Task ID**: #33 from [外部开发者任务池](https://github.com/vllm-project/vllm-ascend/issues/9079)
- **Model**: `jason9693/Qwen2.5-1.5B-apeach` (Korean-optimized Qwen2.5-1.5B fine-tune)
- **Type**: Community model adaptation / offline inference functional testing
- **Status**: Test config and documentation created; accuracy values require NPU hardware verification

## Key Findings

### Model Analysis

- **Architecture**: Standard Qwen2.5 (Transformer + RoPE + SwiGLU + RMSNorm + GQA)
- **Parameters**: 1.5B (28 layers, 12 Q heads, 2 KV heads)
- **Context length**: 32K (tested at 4096)
- **Quantization**: BF16
- **trust_remote_code**: Required (Qwen2.5 custom modeling code)

### Adaptation Strategy

Qwen2.5 is already listed as an "Extended Compatible Model" in vllm-ascend's supported models. It works out-of-the-box via upstream vLLM's built-in `Qwen2ForCausalLM` support. **No patches, model registry changes, or custom operators were needed.**

### New Architecture Support
Not needed — Qwen2.5 architecture is already supported by upstream vLLM.

### Patching
Not needed — no Ascend-specific operator overrides required.

### Model Registry
Not needed — vLLM upstream handles Qwen2.5 via `Qwen2ForCausalLM`.

## Effective Prompts Used

### 1. Research Phase
```
List all files in tests/e2e/models/configs/ and read example configs to understand the test config pattern for vllm-ascend.
```

### 2. Model Analysis
```
Search for Qwen2.5 in the vllm-ascend codebase to check existing support, patches, and test configs.
```

### 3. Documentation Generation
```
Read the model tutorial template at docs/source/_templates/Model-Deployment-Tutorial-Template.md
and follow its structure to create a tutorial for the new model.
```

## Deliverables

| File | Purpose |
|------|---------|
| `tests/e2e/models/configs/Qwen2.5-1.5B-apeach.yaml` | Accuracy test config (gsm8k, 5-shot) |
| `docs/source/tutorials/models/Qwen2.5-1.5B-apeach.md` | Model deployment tutorial (Chinese) |
| `docs/source/tutorials/models/index.md` | Updated toctree with new entry |
| `tests/e2e/models/configs/accuracy.txt` | Updated CI model list |

## Accuracy Notes

The accuracy thresholds in the test config (strict-match: 0.55, flexible-extract: 0.60) are estimated baselines based on Qwen2.5-1.5B-Instruct performance. **These values must be verified on actual NPU hardware** (Atlas A2/A3) before merging. The apeach variant is a Korean-focused fine-tune, which may affect English gsm8k performance.

## Best Practices

1. **Check existing support first**: Qwen2.5 was already supported via upstream vLLM — no need for custom code
2. **Follow established patterns**: Mirrored Qwen3-8B.yaml and Llama-3.2-3B-Instruct.yaml config structure
3. **Use the doc template**: `docs/source/_templates/Model-Deployment-Tutorial-Template.md` provides the standard structure
4. **Dummy-first testing recommended**: For actual hardware validation, test with `--load-format dummy` first, then real weights
5. **Keep docs in Chinese**: vllm-ascend documentation convention for model tutorials
