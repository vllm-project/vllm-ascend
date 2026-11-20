# Split QKV, RMSNorm and MRoPE

## Description

- **Location**: `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py`.
- **Function**: shared wrapper `triton_split_qkv_rmsnorm_mrope` splits packed
  Q/K/V and optional Gate, normalizes Q/K and applies rotary encoding.
- **Formula**: normalize Q and K by `rsqrt(mean(x**2) + eps)`, multiply by their
  effective norm weights, optionally add biases, then apply MRoPE to the rotary
  dimensions. V and pre-sigmoid Gate are extracted without normalization.
- **Algorithm flow**: distribute token work across vector cores, read packed
  heads and norm weights, select rotary values, compute Q/K and write outputs.
- **Caller**: the QSA `AscendQwen4ExpQSAAttention._project_qkv_gate` uses the new
  direct-position mode. Other shared callers may retain the legacy mode.
- **Supported modes**: NPU shared operator; new QSA-mode tests select A3 only.
  No new cross-device or cross-model validation is claimed by this document.

## Parameters

| Parameter | Role | Meaning |
| --- | --- | --- |
| `qkv` | Input | Packed projection `[tokens, q_size + gate_size + 2*kv_size]` |
| `q_weight`, `k_weight` | Input | Per-channel norm weights `[head_size]` |
| `cos_sin` | Input | Legacy gathered rows, or full cache with positions |
| `num_q_heads`, `num_kv_heads`, `head_size` | Attributes | Local projection geometry |
| `eps`, `mrope_section`, `is_interleaved` | Attributes | Norm and rotary configuration |
| `rope_dim` | Optional attribute | Rotary width, defaults to head size |
| `q_bias`, `k_bias` | Optional inputs | Per-channel normalization biases |
| `has_gate` | Attribute | Whether packed Q also contains a per-head gate |
| `positions` | Optional input | int32/int64 `[tokens]` or `[3, tokens]` |
| Return | Output | Flattened Q, K, V and Gate tensors; independent allocations |

## Constraints

Legacy callers omit positions and provide gathered rotary rows and effective
norm weights (including any unit offset). In direct-position mode, QKV must be
contiguous BF16, and positions/cache must be on the same device. The full cache
is nonempty BF16 `[cache_rows, rope_dim]` with contiguous rotary width. Weights
are contiguous `[head_size]` on the QKV device in BF16, FP16 or FP32.

Direct-position mode adds one to raw Q/K weights **in their parameter dtype**
before normalization; legacy mode does not. Do not pass pre-offset weights to
the direct-position mode. Positions are clamped inside the kernel and may have
non-contiguous token/axis strides. Single-axis inputs reuse the position across
the three rotary axes.

The model integration is narrower than the shared API: enabled output gate,
head size 256, rotary width 64, NeoX-style interleaved sections `[11, 11, 10]`,
and exact packed geometry. MTP layers and unsupported layouts keep the upstream
path. Projection GEMM, KV-cache publication and sigmoid gating are not fused.

**Side effects**: inputs are not mutated; outputs are newly allocated. Empty
tokens return empty outputs without launching. The custom op has a fake
implementation; graph replay reads current positions. Callers must provide
compatible Q/K normalization epsilon because the API takes a single epsilon.

## Origin and Differences

Extends the existing shared split/norm/MRoPE operator. PR #22086 adds optional
direct rotary lookup and in-kernel unit weight offsets, and routes eligible
QSA model calls through it. Omitting positions preserves the legacy semantics.

## Test Cases

New QSA-mode tests compare against legacy gathered inputs and pre-offset
weights with `rtol=0, atol=0`. They include 0/1/32/129 tokens, 1/3 position axes,
BF16/FP32 weights, Q heads 2/12, KV heads 1, head size 256, partial width 64,
and graph replay. These cases target direct-position mode and do not validate
every shared caller.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_mrope.py
```
