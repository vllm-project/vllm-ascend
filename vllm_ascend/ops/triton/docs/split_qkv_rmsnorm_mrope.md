# split_qkv_rmsnorm_mrope

## Description

- **Function**: Fuses QKV splitting, per-head Q/K RMSNorm, multimodal rotary position embedding, and optional attention-gate extraction. It supports both chunked and T/H/W-interleaved MRoPE frequency layouts through `torch.ops.vllm.triton_split_qkv_rmsnorm_mrope`.
- **Formula**: Let `T` be the token count, `D` the head size, `Q = Hq * D`, `K = Hkv * D`, and `R` the rotary dimension. Without a gate, split the input as `[Q | K | V]`. With a gate, each Q head is stored as `[q_head(D) | gate_head(D)]`, followed by K and V. Q and K use

  $$
  z = x \cdot \operatorname{rsqrt}\left(\operatorname{mean}(x^2) + \epsilon\right) \cdot w + b.
  $$

  For every half-frequency index, select cos/sin from the temporal, height, or width plane according to `mrope_section` and `is_interleaved`. Repeat the selected half-vector across both halves and apply NeoX rotation:

  $$
  y_{0:R/2} = z_{0:R/2}c - z_{R/2:R}s,
  \qquad
  y_{R/2:R} = z_{R/2:R}c + z_{0:R/2}s.
  $$

  Dimensions `R:D` remain unchanged for partial RoPE. V and the optional gate are copied unchanged.
- **Algorithm flow** (processed row by row, independently):
  1. Derive Q/K widths, optional gate width, token distribution, and rotary mode in the wrapper.
  2. Partition tokens contiguously across at most the available vector cores. Front cores process `ceil(T / core_count)` rows and tail cores process `floor(T / core_count)` rows.
  3. For each token, load Q, K, V, and the optional per-head-interleaved gate; compute per-head Q/K RMSNorm and affine transforms in fp32.
  4. Select T/H/W cos and sin values using either contiguous sections or the interleaved pattern, rotate Q and K, and store Q, K, V, and the optional gate.
- **Supported modes**: Atlas A2 and Atlas A3 for Qwen3.5 and Qwen3-VL multimodal inference, in eager and graph-captured execution. The current direct single-operator test has been verified on Atlas A2; Ascend 950 verification is not claimed by the current test evidence.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qkv` | Input | Fused QKV input; Q, K, V without a gate, or per-Q-head q/gate pairs followed by K and V when gated | fp16 / bf16 | Contiguous 2-D, `[T, Q + 2 * K]` or `[T, 2 * Q + 2 * K]` |
| `q_weight` | Input | RMSNorm weight shared by all Q heads | fp16 / bf16 | Contiguous 1-D, `[D]` |
| `k_weight` | Input | RMSNorm weight shared by all K heads | fp16 / bf16 | Contiguous 1-D, `[D]` |
| `cos_sin` | Input | T/H/W cache planes, each row containing a cos half followed by a sin half | same as `qkv` | Contiguous 3-D, `[3, T, R]` |
| `num_q_heads` | Input (attribute) | Number of local Q heads `Hq` | int32 | Scalar |
| `num_kv_heads` | Input (attribute) | Number of local K/V heads `Hkv` | int32 | Scalar |
| `head_size` | Input (attribute) | Per-head feature size `D` | int32 | Scalar |
| `eps` | Input (attribute) | Positive RMSNorm epsilon | fp32 | Scalar |
| `mrope_section` | Input (attribute) | Half-dimension counts `[temporal, height, width]` | list of three int32 values | 1-D attribute |
| `is_interleaved` | Input (attribute) | Selects T/H/W-interleaved frequency-source assignment instead of contiguous sections | bool | Scalar |
| `rope_dim` | Optional input (attribute) | Rotary dimension `R`; defaults to `head_size` | int32 | Scalar |
| `q_bias` | Optional input | Bias shared by all normalized Q heads | fp16 / bf16 | Contiguous 1-D, `[D]` |
| `k_bias` | Optional input | Bias shared by all normalized K heads | fp16 / bf16 | Contiguous 1-D, `[D]` |
| `has_gate` | Input (attribute) | Indicates whether each Q head contains an adjacent gate vector | bool | Scalar |
| `q_output` | Output | Normalized and MRoPE-rotated Q | same as `qkv` | Contiguous 2-D, `[T, Q]` |
| `k_output` | Output | Normalized and MRoPE-rotated K | same as `qkv` | Contiguous 2-D, `[T, K]` |
| `v_output` | Output | V copied from the fused input | same as `qkv` | Contiguous 2-D, `[T, K]` |
| `gate_output` | Output | Extracted gate, or a zero-width tensor when gating is disabled | same as `qkv` | Contiguous 2-D, `[T, Q]` or `[T, 0]` |

## Constraints

- `Q = num_q_heads * head_size` and `K = num_kv_heads * head_size`; the input width must exactly match the selected gated or non-gated layout.
- `rope_dim` must be positive, even, and no greater than `head_size`. `sum(mrope_section)` must equal `rope_dim / 2`, and all three section lengths must be non-negative.
- `cos_sin` must have the exact contiguous shape `[3, T, rope_dim]`; plane 0 is temporal, plane 1 is height, and plane 2 is width.
- `q_weight` and `k_weight` must each contain `head_size` elements. `q_bias` and `k_bias` must either both be present or both be absent.
- All tensors must be contiguous because the kernel uses flat pointer arithmetic and does not accept strides.
- `T` must be positive. The wrapper does not provide a zero-token fast path before calculating the launch size.
- `is_interleaved` changes which modality supplies each frequency; the rotary pairing itself remains NeoX half-style.
- The registered fake implementation supports Dynamo/AOT tracing. The optional bias branch is implemented but not covered by the current single-operator test.

## Origin and Differences

- **Origin**: Developed as a vLLM-Ascend-native fused Triton operator in PR #6730 for the Qwen3.5 and Qwen3-VL multimodal attention paths.
- **Differences**:
    - NPU adaptation for performance: combines QKV/gate extraction, Q/K RMSNorm, modality-frequency selection, partial MRoPE, and output stores in one vector-core launch;
    - Modified for a specific vllm-ascend logic or different input parameters: accepts cache rows already indexed by three-dimensional multimodal positions, supports both chunked and interleaved `mrope_section` layouts, and extracts the Qwen3.5 per-head attention gate when present.

## Test Cases

The direct custom-op test covers `T={1,4096}`, `(Hq,Hkv)={(2,1),(16,2)}`, `D={128,256}`, `mrope_section={[11,11,10],[24,20,20]}`, interleaved and non-interleaved layouts, gate enabled and disabled, and both fp16 and bf16. It produces 128 parameter combinations and compares Q, K, V, and the enabled gate with a PyTorch reference using `rtol=1e-2, atol=1e-2`. The optional bias path remains uncovered.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_mrope.py
```
