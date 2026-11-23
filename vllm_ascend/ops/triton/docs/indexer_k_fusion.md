# indexer_k_norm_rope_int8_store

## Description

- **Function**: publish DeepSeek V4.1 pre-normalization `wk(latent)` output into
  ordinary paged INT8 key and FP16 dequantization-scale caches. `wk` remains a
  separate GEMM. Production caller: `DeepseekV41Indexer.update_keys`.
- **Formula**: `n = bf16(x * rsqrt(mean(x*x) + eps) * weight)`; apply interleaved
  partial RoPE to the final R columns and round its result to BF16; let
  `scale = max(abs(key)) / 127`; quantize with FP32 scale and nearest-even
  rounding into [-127,127], then store scale in FP16. For zero rows the draft
  produces zero key and zero scale; acceptance must confirm the deployed
  native dynamic-quantization operator has the same behavior.
- **Algorithm flow**: one program per token; validate physical slot on device;
  normalize, rotate, quantize and scatter to that unique slot. Invalid slots
  perform no writes. No intermediate global buffers are allocated.
- **Supported modes**: ordinary A2/A3 caches are the intended fused path.
  Hardware acceptance is pending. A5 packed/folded publication and loaded
  normalization bias use the caller's existing implementation; this is a
  compatibility requirement, not a claim of completed A5 validation.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| k_pre | Input | Pre-normalization wk output, [T,D] | BF16 | last-axis stride 1 |
| weight | Input | RMSNorm weights, [D] | BF16/FP32 | strided vector |
| eps | Attribute | Finite positive RMSNorm epsilon | Python float | scalar |
| cos, sin | Input | Source-selected paired RoPE angles, [T,R] or [T,1,1,R] | FP32 | physical strides |
| slots | Input | Unique valid [block,offset] coordinates, [T,2] | INT32/INT64 | physical strides |
| k_cache | Input/Output | Caller-owned paged keys, [B,P,1,D] | INT8 | contiguous row, page gaps allowed |
| scale_cache | Input/Output | Caller-owned scales, [B,P,1,1] | FP16 | nonoverlapping rows/page gaps allowed |

## Constraints

- All tensors are on one NPU. D is an even power of two; R is positive/even
  and no greater than D. Adjacent cos/sin columns repeat the same angle as in
  `rope_dsv4.py`. Arbitrary, independently varying adjacent columns are unsupported.
- All numerical inputs are finite. Valid physical slots must be unique;
  negative or out-of-bounds coordinates are skipped. Caller guarantees value
  constraints and stream ordering without host reads in the hot path.
- Inputs do not alias writable outputs. Only valid cache rows are modified;
  k_pre, weight, angles and slot metadata remain unchanged. T=0 is a no-op
  after shape/dtype/device validation. No implicit layout or dtype conversion.
- Positive row/page gaps are supported subject to wrapper checks. Unsupported
  formats are rejected; packed/folded formats are dispatched outside this wrapper.
- Graph path uses stable caller-owned output storage and reads no tensor contents
  on the host. Actual ACL graph capture/replay validation remains pending.
- Specialization preserves vector-layout optimization while keeping row and
  cache addressing dynamic:

  | Arguments | Policy | Reason |
  | --- | --- | --- |
  | `HEAD_DIM` | constexpr | Defines vector and reduction sizes. |
  | `ROTARY_DIM` | constexpr | Fixed per model; exposes the partial-RoPE mask to constant folding. |
  | `WEIGHT_STRIDE` | constexpr | Exposes the per-lane weight access pattern. |
  | `COS_COL_STRIDE`, `SIN_COL_STRIDE` | constexpr | Expose the per-lane RoPE-table access patterns. |
  | `QUANT_MAX` | fixed constexpr | Symmetric INT8 algorithm constant, always 127. |
  | `input_row_stride`, `cos_row_stride`, `sin_row_stride` | runtime, do_not_specialize | Only calculate the selected row base. |
  | `slots_row_stride`, `slots_col_stride` | runtime, do_not_specialize | Address the two scalar slot coordinates. |
  | `key_block_stride`, `key_row_stride`, `scale_block_stride`, `scale_row_stride` | runtime, do_not_specialize | Reuse artifacts across cache row layouts and allocations. |
  | `num_blocks`, `page_size` | runtime, do_not_specialize | Only check physical-slot bounds in this kernel. |
  | `rms_eps` | runtime float | Arithmetic input; float values alone do not specialize. |

  The constexpr domain is the model dimensions, three vector access strides and
  the fixed quantization constant. Changing row strides, slot-record strides,
  cache bounds or grid does not change those constants. Different vector-column
  strides or rotary widths intentionally select different specializations.
  Tensor dtype/alignment, scalar types, compiler options, backend and source/cache
  state can still affect compilation. Supported strided views remain supported;
  the wrapper performs no contiguous copies. This policy is a tuning candidate,
  not evidence of lower latency or fewer variants than the all-runtime
  implementation. Validate both on the deployed Triton version.
- Warmup: no new warmup registration is included in this proposal. First measure
  remaining runtime variants, then, if required, use the shared Ascend wrapper
  and active registration window described in `triton_jit_warmup.md`. Preserve
  a common launch specification between compile-only warmup and runtime.

## Origin and Differences

- **Origin**: developed for the existing DeepSeek V4.1
  `RMSNorm -> partial rotary -> npu_dynamic_quant -> two cache scatters` path.
- **Differences**: fuses these operations; retains BF16 boundaries and the FP32
  quantization scale. No model projection GEMM is fused. Packed/folded caches
  and normalization bias retain the baseline implementation.
- **Accuracy reference**: native NPU RMSNorm, interleaved RoPE and dynamic
  quantization on identical inputs. INT8 outputs must match exactly under the
  declared contract. FP16 scales use rtol=2^-9, atol=2^-9, max error<=0.1,
  with at least 99% of elements within combined tolerance. Unwritten cache
  storage and input tensors require exact equality.

## Test Cases

```bash
python -m pytest tests/ut/ops/test_indexer_k_fusion.py tests/ut/models/test_deepseek_v41_indexer_k_fusion.py -q
python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_indexer_k_fusion.py
```

The test module's MODEL_DIMENSIONS table contains explicit tuples of
index_head_dim, qk_rope_head_dim, head_dim and rms_norm_eps. Replace or extend
the interface-example case using the deployed model config before reporting
model acceptance. No custom pytest option is required. Test A2/A3 separately;
also validate the A5 and loaded-bias
production fallback. Add real workload shapes observed during prefill, decode,
speculative decoding and PCP to the parameter table when collecting results.
Required additional acceptance includes graph replay, rounding ties/native
zero-scale behavior, quantization boundaries and cross-device model evaluation.

## Example

```python
from vllm_ascend.ops.triton.indexer_k_fusion import indexer_k_norm_rope_int8_store

indexer_k_norm_rope_int8_store(k_pre, weight, eps, cos, sin, slots, k_cache, scale_cache)
```
