# References Roadmap

This document tracks planned additions to `vllm-ascend-model-adapter/references`.

## Current deep-analysis coverage

The main deep-analysis documents currently are:

- `attention-v1-analysis.md`
- `moe-fused-analysis.md`

Many other files are still baselines, checklists, lessons learned, or fixed-output templates.

## Planning principles

1. Complete the layer-by-layer backbone for `dense llm` and `moe llm` first.
2. Keep reference documents focused on current capability analysis instead of model-specific adaptation plans.
3. Prioritize layers where structural differences are most likely to be misclassified.
4. Bind each document to concrete code paths instead of abstract model theory.

## Main missing analysis topics

From the model-structure point of view:

- `embedding`
- `positional / rope`
- `mlp / ffn`
- `norm`
- `lm_head / output`
- `moe router / gate`
- `moe experts`
- `shared expert / residual mlp`

From the cross-layer point of view:

- `weight loading / remap`
- `quantization`
- `operator compatibility`
- `framework integration`
