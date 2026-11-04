# Logit bias

## Description

- **Function**: Apply allowed-token filtering, additive logit bias, and minimum-length stop-token masking in place for Model Runner V2.
- **Formula**: Keep allowed logits, set disallowed logits to `-inf`, then add the configured per-token bias. Before the minimum length, mask stop tokens. If enabled and the entire row becomes `-inf`, restore only finite saved stop-token logits.
- **Algorithm flow**: One Triton program processes one logit row. Request metadata is selected through the expanded request mapping. The all-masked fallback scans the vocabulary after masking stop tokens.
- **Supported modes**: Targets Atlas A2, Atlas A3, and 950PR/950DT products. Device validation is required for the deployed Triton-Ascend version.

## Parameters

All parameters are required. Names ending in `_ptr` are device pointers.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits_ptr` | Input/Output | Logits updated in place | float32 | Rows by vocabulary |
| `logits_stride`, `vocab_size` | Attribute | Row stride and vocabulary size | Integer | Scalar |
| `expanded_idx_mapping_ptr` | Input | Map logit rows to request state | int64 | One entry per row |
| `num_allowed_token_ids_ptr`, `allowed_token_ids_ptr` | Input | Allowed-token counts and IDs | int32 | Per-request counts and ID table |
| `allowed_token_ids_stride` | Attribute | Allowed-token table row stride | Integer | Scalar |
| `num_logit_bias_ptr`, `bias_token_ids_ptr` | Input | Bias counts and token IDs | int32 | Per-request counts and ID table |
| `bias_ptr` | Input | Additive logit bias values | float32 | Per-request table |
| `bias_token_ids_stride`, `bias_stride` | Attribute | Bias ID and value table row strides | Integer | Scalar |
| `pos_ptr` | Input | Absolute position of each logit row | int64 | One entry per row |
| `min_lens_ptr` | Input | Absolute minimum sequence lengths | int32 | One entry per request |
| `num_stop_token_ids_ptr`, `stop_token_ids_ptr` | Input | Stop-token counts and IDs | int32 | Per-request counts and ID table |
| `restore_when_all_masked_ptr` | Input | Enable finite stop-token restoration for each request | int32 | One entry per request |
| `stop_token_ids_stride` | Attribute | Stop-token table row stride | Integer | Scalar |
| `BLOCK_SIZE`, `LOGITS_BLOCK_SIZE` | Attribute | Metadata tile and vocabulary scan tile sizes | Compile-time integer | Scalar |
| `CHECK_ALL_MASKED_ROWS` | Attribute | Compile the all-masked fallback | Compile-time boolean | Scalar |

## Constraints

- Logits must be contiguous along the vocabulary dimension. Request mappings and active token IDs must be valid indices.
- Counts must fit their tables and `BLOCK_SIZE`. Tile sizes must satisfy Triton's power-of-two requirements.
- The wrapper selects tile sizes and passes request metadata; the kernel preserves its signature and behavior.
- The implementation adds no host synchronization. Graph capture and replay require separate device validation.

## Implementation

The worker patch binds `vllm.v1.worker.gpu.sample.logit_bias._bias_kernel` to this implementation. The finite-value condition in stop-token restoration is expressed with `tl.where`, leaving only the count bounds on the store mask. Non-finite saved values remain `-inf`.

This avoids the data-dependent masked-store pattern that can generate unordered lock workspaces on Triton-Ascend. Together with the equivalent grammar-mask rewrite, it avoids the affected workspace reuse path without serializing the host and device. The patch can be removed when the supported toolchain safely handles those stores or the common implementation uses an equivalent pattern.

## Test Cases

The tests cover finite and non-finite stop logits, the restoration flag, non-stop valid tokens, the minimum-length boundary, and repeated grammar-to-bias launches without intermediate host synchronization.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_logit_bias.py
```

## Example

For a row with only one finite stop-token logit before the minimum length, ordinary minimum-length filtering masks the entire row. When restoration is enabled, the finite stop-token logit is restored. Saved `NaN`, `+inf`, and `-inf` values stay masked.
