# turboquant_finalize

## Description

- **Location**: `vllm_ascend/ops/triton/turboquant_finalize.py`.
- **Function**: Fuse TurboQuant scale correction and compact cache-row packing. The CANN `cann_ops_nn.turbo_quant` operator produces packed 4-bit codes and the original vector norm. This kernel computes the corrected FP16 scale and appends its two bytes to the unchanged codes. It returns rows consumed by the DeepSeek V4 TurboQuant cache path; it does not quantize inputs, rotate vectors, compute attention, or scatter rows into the paged cache.
- **Formula** (independently for each row `r`):

  ```text
  norm_lut[b] = centroids[b & 15]^2 + centroids[b >> 4]^2
  selected_norm[r] = sqrt(sum(norm_lut[packed[r, j]], j=0..255))
  scale[r] = fp16(fp32(norm[r]) / selected_norm[r])
  bits[r] = bitcast_uint16(scale[r])
  output[r, 0, 0:256] = packed[r, :]
  output[r, 0, 256] = uint8(bits[r] & 255)
  output[r, 0, 257] = uint8(bits[r] >> 8)
  ```

  The caller constructs `norm_lut` once from the same 16 centroids used for quantization. Each packed byte contains two codes: the low nibble selects the first centroid and the high nibble selects the second. Index conversion to int32 inside the kernel does not remap the codes.

- **Algorithm flow** (processed row by row, independently):
    1. Allocate a contiguous uint8 output of shape `[R, 1, 258]`. For `R=0`, return without launching a kernel.
    2. Launch `min(R, get_vectorcore_num())` programs. Each program processes rows in a grid-stride loop, loading 256 packed bytes per row.
    3. Look up the squared centroid norms, reduce in FP32, take the square root, and divide the original norm by the result.
    4. Round the scale to FP16, bitcast it to uint16, and store its low and high bytes after the 256 unchanged code bytes.
- **Supported modes**: Ascend NPU inference with Triton-Ascend, including eager execution and ACL graph capture/replay. Hardware validation for this change was on Atlas A3. Atlas A2 and 950PR&950DT validation is N/A; this document does not establish additional hardware support.

## Parameters

All three inputs are required. The Python entry point is `turboquant_finalize(packed, norm, norm_lut)`; `_finalize` is the private kernel.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `packed` | Input | Quantized codes for `R` vectors with head dimension 512, shape `[R, 256]` | uint8 | Contiguous ND |
| `norm` | Input | Original vector norm from CANN TurboQuant, shape `[R]` | fp16 in the current caller and tests | Contiguous 1D |
| `norm_lut` | Input | Squared-norm contribution for every packed byte, shape `[256]` | fp32 | Contiguous 1D |
| Return value | Output | Codes followed by one FP16 scale per row, shape `[R, 1, 258]` | uint8 | Contiguous ND |
| `rows` | Internal attribute | `R = packed.shape[0]`; excluded from value specialization | Integer, runtime scalar | Scalar |
| `CODE_BYTES` | Internal attribute | `HEAD_DIM // 2` code bytes per row; currently 256 | Integer, constexpr | Scalar |
| `ROW_BYTES` | Internal attribute | Shared cache constant `SLOT_BYTES`; currently 258. Scale bytes use `SCALE_LOW_OFFSET = CODE_BYTES` and `SCALE_HIGH_OFFSET = CODE_BYTES + 1` | Integer, constexpr | Scalar |

## Constraints

- All inputs must reside on the same NPU and satisfy the shapes, dtypes, and contiguous layouts above. The wrapper relies on its caller and does not validate these preconditions. Arbitrary strides and other head dimensions are not supported.
- Every uint8 code value from 0 through 255 is a valid LUT index. The LUT must correspond to the quantizer's codebook and nibble order. The kernel does not retrain, reorder, or convert the codebook.
- For finite corrected scales, `norm` must be finite and nonnegative, each row's LUT sum must be positive and finite, and the quotient must fit FP16. The fixed codebook used by the caller supplies positive squared-norm contributions. Zero original norms produce zero scales. No epsilon, clamping, or special NaN/Inf handling is added.
- The output stores the FP16 bit pattern as low byte then high byte, matching the existing Ascend cache representation. Inputs are read-only; output is newly allocated. Paginated cache writes are handled separately by `write_dsa_cache` in `vllm_ascend/attention/dsa_attn_kv_plan.py`.
- This path is selected for NPU tensors when `TurboQuantLatent(legacy_hadamard=True)` is used. The default SFA path retains its existing PyTorch postprocessing. Hadamard rotation itself is outside this kernel.
- `rows` is not value-specialized, allowing different row counts to reuse the compiled kernel. This does not make a captured graph's shape dynamic: replay retains captured buffers, shapes, and launch arguments. Input values may be updated in the captured buffers before replay.
- Forward inference only; autograd backward support is N/A.

## Origin and Differences

- **Origin**: Developed from the PyTorch postprocessing in `TurboQuantLatent.compress`, in `vllm_ascend/quantization/methods/kv_cache/turboquant/latent.py`. It is not a port of an upstream vLLM Triton operator.
- **Differences**:
    - Fuse index conversion, LUT lookup, reduction, square root, scale division, FP16 conversion, and concatenation into one kernel launch. Avoid materializing a full int64 index tensor and separate intermediate tensors.
    - Preserve code bytes and the historical norm-correction formula. Reduction rounding may differ from separate PyTorch kernels; numerical tolerances are covered by the tests below.
    - Bound the program count by the vector-core count and avoid specializing on prefill row counts.
    - Keep the standard CANN TurboQuant and MixedQuantSparseFlashMla interfaces unchanged. Quantization and attention still execute through the standard operator packages; this kernel supplies framework-side postprocessing between them.

Moving the same postprocessing into a separate CANN operator changes packaging and invocation but does not by itself remove its launch or intermediate reads. Fusing equivalent work inside a CANN producer could reduce those costs, but requires preserving the original public output semantics or defining an explicit compatible capability. The current `turbo_quant` norm output must not silently become a corrected scale for existing callers. CANN integration must preserve the codebook, rounding contract, and cache layout; a performance advantage requires measurement.

## Test Cases

The existing operator tests use the production width of 256 code bytes and a 258-byte output row. They cover:

- `R=0, 1, 8, 257`, every possible packed byte, and zero/nonzero original norms. These deterministic cases compare the complete output with the PyTorch reference using `rtol=0, atol=0`.
- ACL graph capture/replay for nonempty batches, changing norms in place before replay and comparing with eager output using `rtol=0, atol=0`.
- Reuse of the compiled kernel when row counts change to `8, 17, 256, 257, 513` after warming up one row.
- Random code bytes and finite norms for `R=1, 17, 513`: code bytes compare exactly; decoded FP16 scales compare with `rtol=0.001, atol=0` to account for reduction rounding.
- CPU unit coverage for transform compatibility, packed layout, norm correction, and zero input rows. CPU tests do not execute the Triton kernel.

The existing NPU suite is in the one-card PR directory. This documentation change does not relocate or add tests.

```bash
pytest -sv tests/e2e/pull_request/one_card/test_turboquant_finalize.py
pytest -sv tests/ut/quantization/methods/kv_cache/turboquant/test_latent.py
```
