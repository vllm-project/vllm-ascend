# Kimi K3 SiTU quantization operator contract

This document fixes the operator boundary needed by Kimi K3 expert fusion.
It deliberately distinguishes the A3 and A5 implementations.

## Model constants

| Item | Value |
|---|---:|
| Model hidden size | 7168 |
| Routed expert input/output hidden size | 3584 |
| Routed expert intermediate size | 3072 |
| Routed experts | 896 |
| Experts selected per token | 16 |
| Shared experts | 2 |
| Shared intermediate size per expert | 3072 |
| Combined shared intermediate size | 6144 |
| EP size | 64 |
| Local routed experts per EP rank | 14 |
| SiTU `beta` | 4.0 |
| SiTU `linear_beta` | 25.0 |

The reference SiTU calculation for a gate/up tensor `z` is:

```text
gate, up = split(z, 2, axis=-1)
gate = 4 * tanh(gate / 4) * sigmoid(gate)
up = 25 * tanh(up / 25)
situ = gate * up
```

## A3: one DequantSituQuant operator, two input modes

Both A3 shared and routed experts call the same Torch operator:

```python
torch.ops._C_ascend.dequant_situ_quant(...)
```

There is no separate A3 `SituQuant` operator in this contract.

### A3 shared experts: INT32 dequant + SiTU + dynamic INT8 quant

The shared gate/up projection is tensor-parallel. Its QMM produces a raw INT32
accumulator. `DequantSituQuant` consumes that accumulator and the two
dequantization scales.

| TP | local gate width | local up width | `x` | `y` | `scale` |
|---:|---:|---:|---|---|---|
| 1 | 6144 | 6144 | `[M, 12288]` INT32 | `[M, 6144]` INT8 | `[M]` FP32 |
| 2 | 3072 | 3072 | `[M, 6144]` INT32 | `[M, 3072]` INT8 | `[M]` FP32 |
| 4 | 1536 | 1536 | `[M, 3072]` INT32 | `[M, 1536]` INT8 | `[M]` FP32 |
| 8 | 768 | 768 | `[M, 1536]` INT32 | `[M, 768]` INT8 | `[M]` FP32 |
| 16 | 384 | 384 | `[M, 768]` INT32 | `[M, 384]` INT8 | `[M]` FP32 |

Inputs and attributes:

- `x`: contiguous ND INT32, shape `[M, 12288 / TP]`.
- `weight_scale`: contiguous FP32, shape `[12288 / TP]` (a leading singleton
  dimension is also valid).
- `activation_scale`: contiguous FP32 with one value per row, represented as
  `[M]` or `[M, 1]`.
- `bias=None`.
- `quant_scale=None`.
- `quant_offset=None`.
- `group_index=None`; shared rows form one group and are not EP-dispatched.
- `beta=4.0`.
- `linear_beta=25.0`.
- `activate_left=True`.
- `quant_mode="dynamic"`.

The decode case uses `M=1`; the prefill case in the supplied fixtures uses
`M=65`.

### A3 routed experts: pre-dequantized BF16 SiTU + dynamic INT8 quant

The W4 GMM1 has already applied weight scale, scale bias, and per-token
activation scale. Its BF16 result is therefore passed to the same
`DequantSituQuant` operator without any dequantization arguments:

```python
y, scale = torch.ops._C_ascend.dequant_situ_quant(
    x=x_bf16,
    weight_scale=None,
    activation_scale=None,
    bias=None,
    quant_scale=None,
    quant_offset=None,
    group_index=None,
    beta=4.0,
    linear_beta=25.0,
    activate_left=True,
    quant_mode="dynamic",
)
```

The routed contract is expert-parallel and TP-invariant:

| Phase | `x` | `y` | `scale` |
|---|---|---|---|
| decode | `[M, 6144]` BF16 | `[M, 3072]` INT8 | `[M]` FP32 |
| prefill | `[M, 6144]` BF16 | `[M, 3072]` INT8 | `[M]` FP32 |

`M` is the number of compacted rows on one EP rank after dispatch, not
`tokens * top_k` globally. Since each token selects distinct experts and an EP64
rank owns 14 experts:

```text
0 <= M <= tokens * min(top_k, local_experts) = tokens * 14
```

The supplied stress fixtures therefore use:

- decode, one token: `M=14`, plus the mandatory empty-rank edge `M=0`;
- prefill, 65 tokens: `M=910`.

The upstream GMM group list has 14 entries and is consumed before this
operator. It is not a `DequantSituQuant` input in BF16 mode.

### A3 dynamic quantization output

For either input mode, dynamic row scale and INT8 output are:

```text
scale = max(abs(situ), axis=-1) / 127
scale = 1 when the entire row is zero
y = clamp(round(situ / scale), -128, 127).astype(INT8)
```

## A5: SituMxQuant

A5 uses the separate `SituMxQuant` operator. It accepts an already dequantized
BF16 gate/up tensor and produces MXFP8 data plus E8M0 block scales:

```python
y, mxscale = torch.ops._C_ascend.situ_mx_quant(
    x,
    beta=4.0,
    linear_beta=25.0,
    activate_left=True,
    dst_type=36,  # FLOAT8_E4M3FN
)
```

The axis is fixed to `-1` by the Torch adapter. Kimi K3 uses destination type
36 (`FLOAT8_E4M3FN`); type 35 (`FLOAT8_E5M2`) is also supported by the operator.

For an input `[M, 2H]`:

- `x`: contiguous ND BF16 `[M, 2H]`;
- `y`: FP8 E4M3FN `[M, H]`;
- `mxscale`: FP8 E8M0 `[M, ceil(H / 64), 2]`.

Shared-expert A5 shapes are:

| TP | `x` | `y` | `mxscale` |
|---:|---|---|---|
| 1 | `[M, 12288]` | `[M, 6144]` | `[M, 96, 2]` |
| 2 | `[M, 6144]` | `[M, 3072]` | `[M, 48, 2]` |
| 4 | `[M, 3072]` | `[M, 1536]` | `[M, 24, 2]` |
| 8 | `[M, 1536]` | `[M, 768]` | `[M, 12, 2]` |
| 16 | `[M, 768]` | `[M, 384]` | `[M, 6, 2]` |

Routed-expert A5 shapes are TP-invariant: `x=[M,6144]`,
`y=[M,3072]`, and `mxscale=[M,48,2]`, with the same routed-row bounds as A3.

## Fixture policy

- Every fixture calls only the SiTU quantization operator under test; it does
  not run QMM or GMM inside the fixture.
- TP1/2/4/8/16 are represented in deterministic constructed fixtures.
- Real four-node service tracing is required only for TP16. TP1/2/4/8 shapes
  are derived from the exact shared intermediate partition and are not claimed
  as separately captured service data.
- Prefill and decode fixtures are separate so an operator implementation can
  validate both large-row and single/empty-row tiling.
