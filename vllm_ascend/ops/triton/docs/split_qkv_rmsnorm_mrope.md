# Split QKV, RMSNorm, and MRoPE

## Description

- **Function**. Split packed projections into Q, K, and V. Apply RMSNorm and multimodal rotary position embedding (MRoPE) to Q and K. Copy V and the optional attention gate. The Qwen3-VL and Qwen3.5 attention patches call the Python wrapper through `torch.ops.vllm.triton_split_qkv_rmsnorm_mrope`.
- **Formula**. For each Q or K head, compute the RMS reduction in FP32. Let `s(j)` select the temporal, height, or width axis for frequency lane `j`. Each rotating channel pair uses the same phase.

  ```text
  w_eff = cast_to_weight_dtype(weight + rms_weight_offset)
  u = x * rsqrt(mean(x²) + eps) * w_eff + bias
  phase[t, j] = positions[s(j), t] * inv_freq[j]
  cos, sin = cos(phase), sin(phase)
  y_first  = u_first * cos - u_second * sin
  y_second = u_second * cos + u_first * sin
  ```

  An omitted bias contributes zero. Only the first `rope_dim` channels rotate. The remaining channels retain their normalized values. The online path rounds cos and sin to the QKV dtype before the rotation.

- **Algorithm flow**. Distribute contiguous token ranges across at most as many programs as there are vector cores. Each program builds the axis masks once.
    1. Select the MRoPE axis for each frequency. The non-interleaved layout uses contiguous temporal, height, and width sections. The interleaved layout uses the height and width section lengths to select their lanes, with temporal values in the remaining lanes.
    2. Compute cos and sin from FP32 inverse frequencies over a token tile, or load the supplied cache rows and select the same lanes.
    3. Process Q and K in groups of heads. Use FP32 arithmetic for normalization and rotation. Write Q and K in the QKV dtype and copy V and the gate.
- **Supported modes**. Eager execution and ACLGraph capture and replay with Triton on Ascend. The nightly operator tests target Atlas A2, Atlas A3, and Atlas A5. The long-position replay case has been verified on Ascend 910B3.

## Parameters

Let `T` be the token count, `Hq` and `Hkv` the local head counts, `D` the head size, and `R` the rotary dimension.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qkv` | Input | Packed projections. The row layout is `[Q, K, V]` without a gate and `[Q0, gate0, Q1, gate1, ..., K, V]` with a gate. | BF16 or FP16 tested | Two-dimensional ND tensor |
| `q_weight`, `k_weight` | Input | RMSNorm weights shared by the corresponding heads. | Floating point | Contiguous, `D` elements each |
| `cos_sin` | Input | Prefetched cos and sin rows for the cached path. Each row stores cosine values followed by sine values. | QKV dtype at model call sites | `[3, T, R]`, axis order T/H/W |
| `num_q_heads`, `num_kv_heads` | Attribute | Local Q and KV head counts. | Integer | Scalar |
| `head_size` | Attribute | Channels per head. | Integer | Scalar |
| `eps` | Attribute | RMSNorm epsilon. Default `1e-6`. | Float | Scalar |
| `mrope_section` | Attribute | Frequency section lengths in temporal, height, and width order. | Integer list | Three elements |
| `is_interleaved` | Attribute | Selects interleaved MRoPE when true. Defaults to false. | Boolean | Scalar |
| `rope_dim` | Attribute | Number of channels to rotate. Defaults to `head_size`. | Integer or `None` | Scalar |
| `q_bias`, `k_bias` | Input | Optional RMSNorm biases. Supply both or omit both. | Floating point | Contiguous, `D` elements each |
| `has_gate` | Attribute | Indicates that the Q projection packs a gate for each head. Defaults to false. | Boolean | Scalar |
| `positions` | Input | Temporal, height, and width position IDs for online calculation. | Int32 or Int64 | `[3, T]`, strided layouts supported |
| `inv_freq` | Input | Inverse frequencies for online calculation. | FP32 | Contiguous, `R / 2` elements |
| `rms_weight_offset` | Attribute | Value added to Q and K weights. Defaults to `0.0`. Qwen3.5 passes `1.0`. | Float | Scalar |
| `q` | Output | Normalized and rotated query. | QKV dtype | Contiguous `[T, Hq * D]` |
| `k` | Output | Normalized and rotated key. | QKV dtype | Contiguous `[T, Hkv * D]` |
| `v` | Output | Copied value. | QKV dtype | Contiguous `[T, Hkv * D]` |
| `gate` | Output | Copied gate, or an empty-width tensor when `has_gate=False`. | QKV dtype | Contiguous `[T, Hq * D]` or `[T, 0]` |

## Constraints

- Pass `cos_sin` or pass both `positions` and `inv_freq`. The operator expects the selected cache rows in cached mode. The caller gathers those rows from the model cache.
- `Hq`, `Hkv`, and `D` must be positive. `Hkv`, `D`, and `R` must be powers of two, with `2 <= R <= D`. The three nonnegative section lengths must sum to `R / 2`. The fused path uses NeoX half-pair rotation.
- Pack `qkv` to width `(Hq + 2 * Hkv) * D` without a gate or `(2 * Hq + 2 * Hkv) * D` with a gate. Keep tensor inputs on the same NPU. The wrapper makes `qkv` and cached rows contiguous when needed. For `T = 0`, it returns empty outputs without launching the kernel.
- The model patches select online calculation for eligible three-axis positions with unscaled FP32 inverse frequencies on the QKV device. One-dimensional text positions share the same cache index on all three axes. Scaled MRoPE such as YaRN uses cached rows. The model patches use separate QKV splitting, normalization, and rotation if the fused MRoPE capability check fails.
- Use position IDs in the range supported by the model configuration. In ACLGraph replay, retain the captured tensor storage and update position values in place for the same shape, strides, and dtype. The model registers inverse frequencies before capture.

## Origin and Differences

- **Origin**. Extends the existing vLLM Ascend [Triton split QKV, RMSNorm, and MRoPE operator](../linearnorm/split_qkv_rmsnorm_mrope.py).
- **Differences**. The [Qwen3.5](../../../patch/worker/patch_qwen3_5.py) and [Qwen3-VL](../../../patch/worker/patch_qwen3vl.py) call sites can pass positions and inverse frequencies to compute cos and sin in the fused kernel. This avoids gathering cache rows and writing an intermediate cos/sin tensor on the online path. Both modes use the same Q/K normalization and rotation code and allocate new outputs.

## Test Cases

- [Nightly operator test](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_mrope.py). The cases cover Qwen3.5 and Qwen3-VL head layouts, full and partial RoPE, BF16 and FP16, strided positions, optional bias and gate, zero tokens, and core allocation boundaries.
- **Reference and accuracy**. The general tests compare against a PyTorch split, RMSNorm, and MRoPE reference. The short-position inline cases also require exact equality with the cached path. The BF16 long-position case compares Q and K with model-generated cached results through position ID `262143`, including two ACLGraph replays. These comparisons use the test's `atol=rtol=1e-2` for Q and K, with exact V and gate checks in the inline cases. See the [repository precision criteria](../../../../docs/source/developer_guide/contribution/triton_operators.md#reference-accuracy-criteria) for dtype-specific acceptance limits.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_mrope.py
```
