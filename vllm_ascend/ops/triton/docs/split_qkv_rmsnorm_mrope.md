# Split-QKV RMSNorm MRoPE

## Description

- **Function**: Split a packed Q/K/V projection, apply per-head RMSNorm and multimodal RoPE to Q and K, and copy V and an optional Q gate to separate outputs. The public entry point is `triton_split_qkv_rmsnorm_mrope`, used by the Qwen3-VL and Qwen3.5 attention paths.
- **Formula**: For each Q or K head `x` of width `D`, compute `rstd = rsqrt(mean(x^2) + eps)` and `y = x * rstd * weight [+ bias]` in FP32. Select temporal, height and width coefficients according to `mrope_section`, then rotate the first `R = rope_dim` channels as `[y_1 * cos - y_2 * sin, y_2 * cos + y_1 * sin]`. The normalized suffix beyond `R` is unchanged. Weights are the effective scales supplied by the caller; this operator does not add one to them.
- **Algorithm flow**:
    1. Select a tile from the input geometry, token workload and UB capacity, then allocate Q, K, V and gate outputs.
    2. Partition tokens into contiguous per-core ranges, or distribute independent head jobs for eligible single-token inputs.
    3. Load norm weights once per core, normalize Q/K, compute the RoPE halves directly, and store the outputs. V and gate are copied without normalization or rotation.
- **Supported modes**: NPU inference with optional gate, paired Q/K biases, partial RoPE, and interleaved or contiguous MRoPE sections. Adaptive multi-row selection supports BF16 and FP16. Device validation is bounded to Ascend910B3/P40; it does not establish all-device or model/graph coverage.

### Automatic tiling

`M` is the number of token rows processed per iteration, not the number of heads or the total rows assigned to a core. `_select_tiling` checks eligible single-token head grouping first, then independently considers M4 and M2, with ordinary M1 as the fallback.

| Route | Processing granularity | Selection conditions |
| --- | --- | --- |
| Head-grouped M1 | One Q/Gate and K/V head job, with independent validity masks | `T=1`, shared BF16 conditions below, and `2 <= max(num_q_heads, num_kv_heads) <= 40` |
| M4 | Four token rows per iteration | Shared BF16 conditions below, at most two KV heads, both head counts at most 40, and `working_set(4) < min(UB capacity, 192 KiB)` |
| M2 | Two token rows per iteration | BF16/FP16, `head_size <= 256`, and `working_set(2) < UB capacity` |
| Ordinary M1 | One complete token row | Diagnostic override or inputs outside the optimized routes |

Head-grouped M1 and M4 require contiguous NPU inputs with BF16 dtype, D256/R64, gate enabled, no bias, interleaved sections `(11, 11, 10)`, `eps=1e-6`, 40 vector cores and an available, valid UB capacity. M2/M4 additionally require complete Q-head tiles (`num_q_heads % min(num_q_heads, 12) == 0`) and at least 24 token rows in the smallest active per-core partition. Other valid inputs retain ordinary M1.

UB capacity comes from the shared `get_ub_size_bytes()` helper. M2 and M4 use the same integer byte model in `ac_working_set.py`:

```text
WorkingSet(M) = max(initialization, Q/K computation, output/V, complete single-row tail)
```

The model accounts for prepared weights and biases, compact and expanded coefficients, FP32 intermediates, retained K data and temporary storage. The tail uses all Q heads even when the main loop tiles them. This follows ordinary RoPE's UB-aware tiling approach, with a dedicated budget for MRoPE's three coefficient planes and Gate data. It is an empirical resource screen, not a compiler allocator bound or a guarantee that the selected route is fastest.

M2/M4 share a kernel body with Q-head tiles of `min(num_q_heads, 12)`, `multibuffer=False`, `num_stages=1` and `num_warps=32`. Remaining token rows use the complete single-row body. Ordinary M1 retains the backend's default buffering configuration and direct-half RoPE.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qkv` | Input | Fused projection `[T, (1 + has_gate) * Q + 2 * KV]`, where `Q = num_q_heads * head_size` and `KV = num_kv_heads * head_size`; with a gate, each Q head is followed by its gate head | Floating point | 2D ND, contiguous packed layout |
| `q_weight` | Input | Per-head Q RMSNorm scale `[head_size]` | Floating point | 1D ND |
| `k_weight` | Input | Per-head K RMSNorm scale `[head_size]` | Floating point | 1D ND |
| `cos_sin` | Input | Three MRoPE planes `[3, T, rope_dim]`, each containing cos and sin halves | Floating point | 3D ND |
| `num_q_heads` | Attribute | Local Q head count | Integer | Scalar |
| `num_kv_heads` | Attribute | Local K/V head count | Integer | Scalar |
| `head_size` | Attribute | Q/K/V head width and RMSNorm reduction width | Integer | Scalar |
| `eps` | Attribute | Positive RMSNorm stability constant | Float | Scalar |
| `mrope_section` | Attribute | Temporal, height, and width half-RoPE section lengths | Three integers | 1D list |
| `is_interleaved` | Attribute | Select interleaved rather than contiguous MRoPE frequency layout | Boolean | Scalar |
| `rope_dim` | Optional attribute | Rotated prefix width; defaults to `head_size` | Integer or `None` | Scalar |
| `q_bias` | Optional input | Post-norm Q bias `[head_size]` | Floating point or `None` | 1D ND |
| `k_bias` | Optional input | Post-norm K bias `[head_size]` | Floating point or `None` | 1D ND |
| `has_gate` | Attribute | Whether each Q head has a corresponding gate head in `qkv` | Boolean | Scalar |
| `q_output` | Output | Normalized and rotated Q `[T, Q]` | Same as `qkv` | 2D ND |
| `k_output` | Output | Normalized and rotated K `[T, KV]` | Same as `qkv` | 2D ND |
| `v_output` | Output | Unmodified V `[T, KV]` | Same as `qkv` | 2D ND |
| `gate_output` | Output | Unmodified gate `[T, Q]`, or `[T, 0]` when absent | Same as `qkv` | 2D ND |

## Constraints

- Inputs use the contiguous packed layouts above and reside on the same NPU. Adaptive routes additionally require matching input dtypes. Falling back to M1 does not add arbitrary-stride support.
- `num_q_heads`, `num_kv_heads` and `head_size` are positive integers. `rope_dim` is positive, even and no greater than `head_size`. `mrope_section` contains three non-negative integers with `2 * sum(mrope_section) == rope_dim`; `eps` is a finite positive float.
- The caller gathers positions into `cos_sin` before invoking this operator. Each plane contains a cosine half followed by a sine half.
- `q_bias` and `k_bias` must be supplied together or both omitted. All outputs use fresh storage; input tensors are not modified. No backward implementation is provided.
- `VLLM_ASCEND_SPLIT_QKV_RMSNORM_MROPE_BLOCK_M` is a diagnostic opt-out, not a required tuning parameter. Its default `2` enables automatic selection, including M4; `1` forces ordinary M1 and bypasses head grouping and UB queries. Other values, including `4`, raise `ValueError`. It is defined centrally in `vllm_ascend/envs.py` and also serves standalone operator calls without an LLM or `additional_config` context.
- Token count and runtime partition scalars use `do_not_specialize`. Tile size, head grouping, head geometry, RoPE settings, pointer dtypes and compiler options can still produce distinct JIT artifacts. This operator adds no warmup registration.

## Origin and Differences

- **Origin**: The existing vLLM-Ascend Split-QKV RMSNorm MRoPE operator and its Qwen3-VL/Qwen3.5 callers. The public signature and output layout are preserved.
- **Differences**:
    - Direct-half RoPE avoids constructing full-width rotation and coefficient temporaries, including on the M1 fallback.
    - UB- and workload-aware row tiling batches two or four tokens, while bounded Q-head tiles limit the live working set.
    - Single-token head parallelism increases active-core work and loads each T/H/W coefficient plane contiguously.
- **Scope**: Route selection balances resource screening and a bounded performance policy; it does not guarantee UB safety or preserve every previous route's performance on all valid shapes.

## Test Cases

Host tests cover working-set arithmetic, selector boundaries, UB-query behavior, wrapper launch configuration, and CPU kernel-body fixtures for numerical results, exactly-once output ownership and V/Gate byte copies. These fixtures do not emulate device compilation or low-precision rounding.

```bash
python3 -m unittest discover -s tests/ut/ops -p test_ac_working_set.py
python3 -m unittest discover -s tests/ut/ops -p test_split_qkv_rmsnorm_mrope_dispatch.py
python3 -m unittest discover -s tests/ut/ops -p test_split_qkv_rmsnorm_mrope_m4_probe.py
```

The single-card accuracy suite compares Q/K against a CPU PyTorch reference using FP32 normalization and rotation, cast to the output dtype. It covers BF16/FP16, gate/no-gate and both MRoPE layouts, with `atol=rtol=1e-2`. V and gate are compared with the corresponding input slices. For repository acceptance criteria, see the [Triton operator contribution guidelines](../../../../docs/source/developer_guide/contribution/triton_operators.md#reference-accuracy-criteria).

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_mrope.py
```
