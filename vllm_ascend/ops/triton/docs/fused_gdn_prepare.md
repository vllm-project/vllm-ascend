# Fused GDN projection preparation

## Description

- **Function**: Materialize dense QKV, B and A, and zero the complete GDN
  output buffer in one launch. Z remains a view of the input.
- **Formula**: `QKV = QKVZ[:, :Q]`, `Z = QKVZ[:, Q:].reshape(T, H, D)`,
  `B = BA[:, :H]`, `A = BA[:, H:]`, `core = zeros(T, H, D)`.
- **Algorithm flow**: The generic `fused_gdn_prepare_kernel` processes one
  token and 1024 columns per program. Only column block zero copies B/A.
  The `fused_gdn_prepare_prefill_kernel` processes four complete token rows
  per program, with masks for the final row tile and column tails.
- **Supported modes**: Shared Ascend Triton implementation, without a chip-name
  gate. The four-row tile was selected for A3 UB capacity. Validation of this
  standalone revision on A2/A3/A5 and other applicable targets remains required
  before merge; no cross-device performance guarantee is implied.

Production caller: `AscendGatedDeltaNetAttention.forward` in
`vllm_ascend/attention/gdn.py`, for packed, non-interleaved projections.
Separate projections, interleaved layouts and unsupported metadata retain their
existing preparation. No FLA wheel update or GDN core change is needed.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| mixed_qkvz | Input | Packed QKV followed by Z, shape `[T,Q+H*D]` | FP16/BF16/FP32 | Unit column stride; padded row stride allowed |
| mixed_ba | Input | Packed B followed by A, shape `[T,2*H]` | FP16/BF16/FP32 | Unit column stride; padded row stride allowed |
| qkv_width | Attribute | TP-local QKV width Q | Positive integer | Scalar |
| num_v_heads | Attribute | TP-local V heads H, at most 1024 | Positive integer | Scalar |
| head_v_dim | Attribute | V head dimension D | Positive integer | Scalar |
| mixed_qkv | Output | Dense QKV `[T,Q]` | QKVZ dtype | Contiguous |
| z | Output | Z suffix `[T,H,D]`, aliases QKVZ | QKVZ dtype | View |
| b, a | Output | Dense B and A, each `[T,H]` | BA dtype | Contiguous |
| core | Output | All-zero GDN output `[T,H,D]` | QKVZ dtype | Contiguous |

## Constraints

Inputs must have matching token counts and devices. BA must already be local to
the TP rank. The model caller additionally requires QKVZ to have the hidden-state
dtype. Inputs are not modified; QKV/B/A/core are fresh allocations. Every output
row is initialized, including graph padding. Empty batches launch no kernel.

The prefill specialization is selected when T >= 1024, QKVZ is FP16/BF16, and
`max(Q,H*D) <= 8192`. Other supported inputs use the generic kernel. Selection
depends on tensor metadata, not a host-side prefill/decode flag.

Allocations and writes are enclosed in a functional custom op to avoid AOT
cloning fresh mutation arguments. Its fake implementation preserves shape,
dtype, device and dense output strides. Z is constructed outside the custom op
so its alias relationship remains visible. Address products use int64 indices.

The public wrapper is `fused_gdn_prepare`; callers must check
`supports_gdn_prepare` and preserve their fallback path. Optional caller-owned
output storage from the experimental runtime helper is not exposed here because
the production caller always requests fresh outputs.

### JIT specialization

Widths, row strides and tile sizes are constexpr. Prefill token count is a
runtime row-mask argument marked `do_not_specialize`, so different counts of
the same integer type do not add value or divisibility specializations.
Generic token count affects only the grid. Input dtypes, pointer alignment,
integer types, layouts and compiler options can still select variants.

### Startup compilation and warmup

This operator does not register with `JitWarmupRegistry`. Configured model dummy
runs can compile the variants they actually exercise; they do not guarantee
coverage of both dispatch paths or every supported dtype and layout. An unseen
variant can therefore compile on first use. Removing token-count specialization
reduces that risk but does not establish zero serving-time compilation.

The contribution guide requires complete wrapper/registration coverage **where
warmup is used**. A future registry integration must use the same
`AscendTritonJitKernel` wrappers for runtime and warmup, enumerate both generic
and prefill variants, and register during the model's active registry window;
import-time registration or custom-op registration alone is insufficient.
See [Triton JIT warmup](../../../../docs/source/developer_guide/Design_Documents/triton_jit_warmup.md).
Cold/warm cache measurements and compilation counts separated into startup,
capture and serving phases remain required before merge. If model dummy runs
leave reachable variants uncovered, add registry coverage before merging.

## Origin and Differences

Extracted from the combined A3 optimization work in #22489, originally based on
the preparation kernel in #21820. Combines the generic-only #23250 and
prefill-only #23248 into one implementation and one model integration.
Promotes address indices before multiplication and validates input metadata.
There is no Q/K normalization, gate activation, convolution or state update.

Representative Qwen3.5/Qwen3.6 shapes at TP size P are QKVZ `[T,12288/P]`,
QKV `[T,8192/P]`, BA `[T,64/P]`, and core `[T,32/P,128]`.

## Test Cases

Reference: PyTorch slicing, contiguous copies and `torch.zeros`. Data movement
and initialization require bit-exact equality (`rtol=0`, `atol=0`).

```bash
# CPU dispatch/metadata/model-branch checks; the launch is simulated.
python -m pytest --confcutdir=tests/ut/ops tests/ut/ops/test_fused_gdn_prepare.py -q

# Real NPU equality, strides, padding and changed-input graph replay.
python -m pytest tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_gdn_prepare.py -q
```

Hardware tests cover TP1/2/4/8, both kernel paths, the 1024-token boundary,
partial row tiles, row padding/storage offsets, FP16/BF16 and mixed BA dtype.
The prefill cache test holds storage, strides and dtype fixed across aligned and
unaligned token counts, checks compiled-artifact reuse, and validates outputs.
Model accuracy, MTP parity, parallel-mode integration, black-box compilation
counts and controlled operator/end-to-end performance remain merge gates.
Historical combined-runtime performance is not attributed to this PR alone.
