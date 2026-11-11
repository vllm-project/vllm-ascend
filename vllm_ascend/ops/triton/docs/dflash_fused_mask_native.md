# dflash_fused_mask_native

## Description

- **Function**: Fused native candidate-pool construction and suffix masking for Qwen DFlash and DeepSeek DSpark router paths.
- **Formula**: Qwen uses BF16 Top-8 candidates; DSpark computes sqrt(softplus(logits)) + bias, builds a Top-6/Top-2 candidate union, masks suffix scores outside the union, and renormalizes the selected raw scores.
- **Algorithm flow**: Kernel 1 computes per-row candidate flags and, for DSpark, raw scores. Kernel 2 reduces the flags across the padded graph batch and applies the suffix mask. The wrapper then performs the final model-specific top-k and weight normalization.
- **Supported modes**: Atlas A2, Atlas A3, and Atlas A5 for the validated Qwen/DSpark contracts. Unsupported shapes use the reference or baseline route.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| logits | Input | Router logits; non-contiguous inputs are copied once. | BF16 or FP32 | 2D [num_tokens, num_experts] |
| roles | Input | 0 padding/prefill, 1 anchor, 2 suffix. | int32/int64 | 1D [num_tokens] |
| e_score_correction_bias | Input | DSpark expert correction bias. | FP32 | 1D [256] |
| weights | Output | Final routing weights after model-specific normalization. | BF16/FP32 | 2D [num_tokens, top_k] |
| ids | Output | Selected expert IDs. | int32 | 2D [num_tokens, top_k] |

## Constraints

- Qwen requires BF16 logits, 128 or 256 experts, Top-8 routing, and no correction bias.
- DSpark requires FP32 logits with 256 experts, Top-6 routing, suffix_pool_top_k=2, renormalization enabled, and a 256-element bias.
- Fused graph rows are restricted to the configured fixed row set; other shapes use the reference path.
- roles must be contiguous and match the logits row count and device.
- `e_score_correction_bias` must be a contiguous FP32 tensor of shape `[256]` on the logits device.
- A supplied workspace must contain contiguous tensors with the documented shape, dtype, and logits device; callers must not reuse it across incompatible shapes.
- The fixed-row fused path specializes on the configured token-row set and expert count; unsupported row counts use the reference route instead of creating an unbounded variant set.

## Origin and Differences

- **Origin**: Native Triton-Ascend implementation of the DFlash/DSpark activation-routing candidate union.
- **Differences**: Uses Ascend vector-core sorting and a two-kernel graph-safe workspace. Stable lower-expert-ID tie handling is preserved.

## Test Cases

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dflash_anchor_union.py
```

The test compares fused output against the graph-safe reference on Qwen and DSpark model shapes, including a strided input case.
