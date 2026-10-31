# XPress adaptation plan

Status: design placeholder; Ascend XPress execution is not implemented by this change.

RFC: [vllm-ascend #19641](https://github.com/vllm-project/vllm-ascend/issues/19641).

Upstream [vLLM #54448](https://github.com/vllm-project/vllm/pull/54448)
adds a causal refiner to the DFlash drafter. It serves under `method="dflash"`
and resolves the draft block with a fixed number of Jacobi refinement passes.

## Integration

1. Recognize `Qwen3XPressModel` and `DFlashQwen3XPressModel` before the generic
   DFlash branch in the Ascend V2 speculator factory. Reuse the Ascend DFlash
   attention, context-KV and draft-input paths.
2. Adapt the XPress model and refiner head, including checkpoint weight mapping
   and the folded causal mixer. Preserve the anchor at slot 0, full-vocabulary
   draft requirement and rejection of `sample_from_anchor=True`.
3. Provide NPU implementations for the latent pass and the full-vocabulary and
   top-C add/argmax paths. Preserve upstream dtype rounding and tie behavior.
   Keep `xpress_topc` configuration semantics, including 0 for full-vocabulary
   refinement. Establish an eager reference before adapting ACL graph replay.

## Validation before enabling support

Compare refiner outputs with the upstream reference for multiple Jacobi passes,
anchor preservation, top-C and full-vocabulary refinement. Check model routing
and checkpoint loading, then validate Qwen3-8B with
`UIUC-SSAIL/Qwen3-8B-XPress-b16` on NPU. Record GSM8K accuracy, acceptance length,
latency and throughput against the same DFlash baseline. Graph and distributed
execution require their own validation before being advertised as supported.
