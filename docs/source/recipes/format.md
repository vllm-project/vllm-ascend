# Recipe Format

This page defines the proposed vLLM Ascend recipe format.

Recipes should be short, operational, and linked back to the authoritative
vLLM Ascend documentation. They should not replace the existing model
tutorials, feature guides, support matrix, installation guide, or evaluation
docs.

## Metadata

Each recipe starts with a metadata table.

| Field | Description |
|-------|-------------|
| Model | Model or model family covered by the recipe. |
| Hardware | Ascend hardware target, for example Atlas 800 A3 or Atlas 800I A2. |
| Precision | BF16, W8A8, W4A8, or other precision mode. |
| Parallelism | Recommended DP, TP, PP, EP, or single-card shape. |
| Image | Recommended vLLM Ascend image tag or installation path. |
| Source tutorial | Existing vLLM Ascend tutorial used as the source of truth. |
| Support matrix | Link to the model or feature support matrix. |
| vLLM recipes target | Proposed path or model page if exported to `vllm-project/recipes`. |
| Validation status | Smoke-tested, benchmarked, hardware-required, or draft. |

## Required Sections

Use these sections unless a recipe has a clear reason to omit one.

1. `When to use this recipe`
2. `Prerequisites`
3. `Launch container`
4. `Serve the model`
5. `Smoke test`
6. `Benchmark or evaluation hooks`
7. `Tuning notes`
8. `Promotion checklist`

## Authoring Rules

- Keep commands copy-pasteable.
- Keep hardware assumptions explicit.
- Prefer links over duplicating long explanations from model tutorials.
- Keep environment variables close to the command that needs them.
- Mark optional tuning knobs as optional.
- Keep model paths, image tags, ports, and device lists easy to replace.
- Preserve vLLM CLI conventions so stable recipes can later be adapted for
  `vllm-project/recipes`.

## Promotion Checklist

Before proposing a recipe for `vllm-project/recipes`, confirm:

- The command is runnable on the stated hardware.
- The model is listed in the vLLM Ascend support matrix.
- The recipe states image or installation requirements.
- The recipe has a smoke test.
- The recipe has at least one benchmark or evaluation hook.
- The recipe links back to the vLLM Ascend source tutorial for detailed tuning.
- Ascend-specific settings are isolated so the upstream recipe can keep a
  clean platform section.
