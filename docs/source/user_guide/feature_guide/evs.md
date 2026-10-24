# Efficient Video Sampling (EVS)

EVS prunes redundant video frames and visual tokens before the language model
forward. vLLM-Ascend enables the upstream EVS pipeline for Qwen3.5 dense and
MoE models on Ascend NPU.

## Enable EVS

Pass a video pruning rate greater than zero. The rate is the fraction of
visual content retained by the upstream EVS processor.

```bash
vllm serve /path/to/Qwen3.5-2B \
  --max-model-len 4096 \
  --limit-mm-per-prompt.video 1 \
  --mm-processor-kwargs '{"fps": 1}' \
  --video-pruning-rate 0.75
```

Without `--video-pruning-rate` (or with value `0`), Qwen3.5 keeps the existing
multimodal path. EVS requires a model whose processor supports multimodal
pruning; unsupported models ignore the setting.

## Validation scope

- Dense and MoE Qwen3.5 model classes share the same EVS position handling.
- Video inputs must be supplied through the standard vLLM multimodal API.
- The image-only and text-only paths are unchanged.
- Validate output quality for the target model and video distribution before
  changing the pruning rate in production. Higher pruning rates reduce visual
  tokens and transfer cost but can remove task-specific detail.

## Troubleshooting

If enabling EVS produces a position or embedding device mismatch, confirm that
the installed vLLM version is the version recorded in
`.github/vllm-main-verified.commit` or a later compatible release. EVS relies on
the upstream `recompute_mrope_positions` behavior.
