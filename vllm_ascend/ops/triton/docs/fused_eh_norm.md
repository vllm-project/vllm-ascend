# fused_eh_norm

## Description

- **Location**: `vllm_ascend/ops/triton/linearnorm/fused_eh_norm.py`.
- **Function**: Fuse position-zero embedding masking, two weighted RMS normalizations, and concatenation for the GLM MTP input projection.
- **Formula** (per token `i`, with reductions and arithmetic in fp32):

  ```text
  e[i] = 0 if positions[i] == 0 else inputs_embeds[i]
  e_norm[i] = e[i] * rsqrt(mean(e[i]^2) + eps) * enorm_w
  h_norm[i] = previous_hidden[i] * rsqrt(mean(previous_hidden[i]^2) + eps) * hnorm_w
  output[i] = concat(e_norm[i], h_norm[i])
  ```

- **Algorithm flow** (processed row by row, independently):
  1. Launch one Triton program per token, with a block padded to the next power of two of the hidden size.
  2. Mask out padded channels and zero the embedding at position zero. Previous hidden states are not position-masked.
  3. Normalize the two inputs independently, using their respective weights and the same epsilon.
  4. Store the normalized embedding and hidden state side by side, casting to the embedding dtype.
- **Supported modes**: Ascend NPU inference with Triton-Ascend. Device-specific validation is required; this change does not establish additional hardware support.

## Parameters

All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `positions` | Input | Position of each token, shape `[N]`; zero masks the embedding | int32 / int64 | Contiguous 1D |
| `inputs_embeds` | Input | Token embeddings, shape `[N, H]` | fp16 / bf16 / fp32 | ND, contiguous last dimension |
| `previous_hidden` | Input | Previous hidden states, shape `[N, H]` | Same as embeddings in the tests | ND, contiguous last dimension |
| `enorm_w` | Input | Embedding RMSNorm weights, shape `[H]` | Same as embeddings in the tests | Contiguous 1D |
| `hnorm_w` | Input | Hidden-state RMSNorm weights, shape `[H]` | Same as embeddings in the tests | Contiguous 1D |
| `eps` | Attribute | Positive finite normalization epsilon | float | Scalar |
| Return value | Output | Concatenated normalized values, shape `[N, 2H]` | Same as `inputs_embeds` | Contiguous ND |

## Constraints

- All tensor inputs must be on the same NPU. Inputs must have compatible shapes; the wrapper does not validate them.
- The test contract uses matching floating-point dtypes and finite values. Mixed floating-point dtypes are not covered.
- The two activation tensors may have different row strides, but their channel stride must be one. Position and weight strides must be one.
- Test coverage uses positive `N` and `H`; empty batches are not covered.
- Hidden-size padding is masked and the RMS denominator is `H`, not the padded block width. The block must fit the target device's compilation/resource limits.
- Forward inference only; no autograd backward implementation is provided.
- Graph-mode support: not validated by the added tests, which exercise eager execution.

## Origin and Differences

- **Origin**: Existing GLM MTP implementation in `vllm_ascend/models/glm5next/ops/fused_eh_norm.py`.
- **Differences**:
    - Relocated into the shared Triton normalization directory and updated the model import.
    - No changes to the kernel, numerical computation, launch configuration, or wrapper signature.

## Test Cases

The test imports the operator from its new location and compares NPU results with a CPU fp32 PyTorch `rms_norm` reference, cast to the output dtype.

- Model-shaped cases use the GLM text config defaults `H=4096` and `eps=1e-5`, with single-token decode, mixed zero/nonzero positions, and 17-token prefill batches.
- A separate boundary case uses `H=513` and `eps=1e-6` to exercise masked channels beyond a non-power-of-two hidden size.
- All cases run with fp16, bf16, and fp32, both contiguous rows and separately padded embedding/hidden-state row strides (24 parameter combinations).
- Checks include output shape/dtype/device, zero embedding output at position zero, zero input rows, and preservation of input activations.
- Tolerances follow the existing normalization tests: `(rtol, atol)` is `(2e-3, 2e-2)` for fp16, `(2e-2, 5e-2)` for bf16, and `(1e-4, 1e-4)` for fp32.
- These are NPU tests. Syntax/lint validation on a CPU host does not establish kernel correctness; run the following command in a configured Ascend environment.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_eh_norm.py
```
