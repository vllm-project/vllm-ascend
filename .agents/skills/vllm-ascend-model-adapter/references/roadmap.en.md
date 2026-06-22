# References Roadmap

This document tracks planned additions to `vllm-ascend-model-adapter/references`.

## Current deep-analysis coverage

The main deep-analysis documents currently are:

- `attention-v1-analysis.md`
- `moe-fused-analysis.md`

Many other files are still baselines, checklists, lessons learned, or fixed-output templates.

## Planning principles

1. Complete the layer-by-layer backbone for `dense llm` and `moe llm` first.
2. Keep reference documents focused on current capability analysis rather than model-specific adaptation plans.
3. Prioritize layers where structural differences are most likely to be misclassified.
4. Bind each document to concrete code paths instead of writing abstract model theory.

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

## Recommended priority

1. `embedding-analysis.md`
2. `positional-rope-analysis.md`
3. `mlp-ffn-analysis.md`
4. `norm-analysis.md`
5. `lm-head-output-analysis.md`
6. `moe-router-analysis.md`
7. `moe-experts-analysis.md`
8. `shared-expert-residual-mlp-analysis.md`
9. `weight-loading-remap-analysis.md`
10. `quantization-analysis.md`
11. `operator-analysis.md`
12. `framework-integration-analysis.md`
