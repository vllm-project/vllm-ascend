# Step5 gated_scale

## Description

- **Function**: Applies the Step5 head-wise attention gate — sigmoid over
  per-head gate logits broadcast-multiplied onto the attention output — in
  one Triton kernel: `out[t, h, :] = attn[t, h, :] * sigmoid(gate[t, h])`.
  It replaces the eager post-attention chain
  `attn_output.view(t, H, D) * gate.unsqueeze(-1).sigmoid()` (two eager
  kernels plus a materializing copy) with a single fused kernel. The
  sigmoid and the multiply run in fp32, which is at least as accurate as
  the bf16 eager path. The wrapper is registered as the custom op
  `ascend_step5_gated_scale`
  (`vllm.triton_utils.direct_register_custom_op`), so it is callable
  through `torch.ops.vllm.ascend_step5_gated_scale` and safe for
  torch.compile.
- **Location**: `vllm_ascend/ops/triton/step5/gated_scale.py` — kernel
  `_gated_scale_kernel`, wrapper `gated_scale`. Production callers:
  `vllm_ascend/models/step5/model.py` (`AscendStep5DSAAttention.forward`
  and `AscendStep5Attention.forward`), selected whenever the layer's
  `use_head_wise_attn_gate` produces a gate. **Scope**: model-specific to
  Step5-Preview; not a framework-wide primitive.
- **Formula**: the gate is addressed through an **offset-0 base tensor plus
  a scalar column offset**:
    - `out[t] = attn[t] * sigmoid(base[t, gate_col_offset : gate_col_offset + H])`,
      where `attn` is `[num_tokens, H * D]` viewed per head and `base` is
      any tensor sharing the gate's storage with `storage_offset == 0`.
    - DSA layers: the gate is the tail of the fused qkv+indexer GEMM row;
      the base is the offset-0 `qkv` front view of that same row, so the
      gate columns lie beyond the base's logical width but inside its
      storage row span (`gate_col_offset = gate.storage_offset()`).
    - Sliding-window layers: the packed `qkvg` GEMM output is the base when
      `fuse_qkv_gate` packs q/k/v/gate into one GEMM, with
      `gate_col_offset = q_size + 2 * kv_size`; otherwise the standalone
      `g_proj` output is its own base with `gate_col_offset = 0`.
    - Handing a mid-storage gate view to the op boundary instead would make
      the dispatch layer materialize it first (TensorMove); callers declare
      the layout explicitly and no materializing copy is issued.
- **Algorithm flow** (one token row per program):
    1. Grid `(num_tokens, cdiv(H, BLOCK_H))`; each program owns one token
       and a `BLOCK_H`-wide head block (`BLOCK_H` sized to keep one token
       row's gate+attention lanes within a 4096-lane budget).
    2. Load the `BLOCK_H` gate logits at `base + t * stride + offset + h`,
       sigmoid in fp32.
    3. Load the `(BLOCK_H, BLOCK_D)` attention tile, multiply by the
       per-head gate in fp32, store bf16 with head/dim masking.
- **Supported modes**: Ascend 950PR/950DT (A5) on the Triton-Ascend
  backend; execution evidence on Ascend950DT / 951DT. Atlas A2/A3 are not
  validated for this operator (the Step5-Preview model itself is
  A5-targeted); no A2/A3-incompatible construct is used (plain Triton
  elementwise math), so porting is gated on hardware validation rather
  than on code changes. Eager and ACLGraph decode capture paths both work
  (the wrapper allocates the output and derives the grid from shapes
  only). Any `num_heads` / `head_dim` combination is handled through
  masking; per-token grids need no padding.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `attn_output` | Input | Attention output to gate, `[num_tokens, num_heads * head_dim]` | bf16 | ND, row-major; row stride passed to the kernel |
| `gate_base` | Input | Offset-0 tensor sharing the gate's storage; the gate is read as `gate_base[t, offset : offset + num_heads]` | bf16 | ND, row stride passed to the kernel |
| `gate_col_offset` | Input | Scalar column offset of the gate inside `gate_base`'s storage row | int | - |
| `num_heads` | Input | Gate width (heads), must satisfy `gate_col_offset + num_heads <= gate_base.stride(0)` | int | - |
| return | Output | Gated attention output, same shape/dtype as `attn_output` | bf16 | ND |

## Constraints

- `gate_col_offset >= 0` and `gate_col_offset + num_heads <= gate_base.stride(0)`:
  gate columns must stay inside the base's storage row span. Violations
  fail loudly (assertion) instead of reading garbage.
- `gate_base` must have `storage_offset == 0` and share the gate's storage;
  the row stride used for addressing is `gate_base.stride(0)`, so a front
  row-slice of a wider packed GEMM row qualifies and the gate tail is read
  in place beyond the base's logical width.
- `attn_output` is 2-D `[num_tokens, num_heads * head_dim]`;
  `head_dim` is derived as `attn_output.shape[-1] // num_heads`.
- In-place behavior: none. Inputs are read-only; the output is freshly
  allocated. The op is registered with `mutates_args=[]`.
- Specialization / recompilation profile: the grid varies with
  `num_tokens`, so batch size changes never recompile (grid changes are not
  part of the cache key). Compile-time specialization comes from the
  `constexpr` head counts (required: grid and masks derive from them) and
  from Triton's value/alignment classes of the runtime integer arguments
  (`attn_output.stride(0)`, `gate_base.stride(0)`, `gate_col_offset`,
  `out.stride(0)`; classes are `==1`, multiples of 16, other). All of those
  are fixed per layer shape and anchor kind in serving (two anchors per
  model: the DSA `qkv` view and the sliding-window `qkvg`/`g_proj` base),
  so the steady-state variant set is one per (layer shape, anchor) pair —
  no per-request compilation. `do_not_specialize` is deliberately not used:
  the specializations are few, fixed and beneficial (alignment feeds
  vectorization).

## Origin and Differences

- **Origin**: Adapted from the Step5 eager post-attention gate in vLLM
  (`attn_output.view(t, H, D) * gate.unsqueeze(-1).sigmoid()`), fused the
  same way the step4 internal `fuse-gate-chain` branch fused it for the
  Step5 tailring deployment.
- **Differences**:
    - The eager path materializes the gate view at the op boundary and
      launches a sigmoid kernel plus a broadcast multiply; the fused kernel
      reads the gate in place through the offset-0 base (no TensorMove) and
      runs both stages in fp32 in one launch.
    - Numerics: fp32 sigmoid and multiply are at least as accurate as the
      bf16 eager chain; output is rounded back to bf16 once.
    - Upstream CUDA counterpart
      (`vllm/models/step5/nvidia/ops/attention_gate.py::headwise_sigmoid_gate`):
      same post-attention fallback role, but it reproduces the eager
      numerics bit-for-bit (sigmoid re-rounded to the gate dtype before the
      multiply, `enable_fp_fusion=False`) and supports an active-row mask
      for CUDA-graph slot handling. This kernel keeps the full-fp32 chain
      (one final rounding) and handles the padded rows through the caller's
      zero-weight padding instead. The deeper upstream fusion (the gate
      folded into the attention kernel epilogue via `dsa_gate` /
      `forward_with_headwise_gate`) is a follow-up opportunity for the
      Ascend block-sparse attention operators; this op mirrors upstream's
      post-attention fallback tier first.
    - Exposed as a registered custom op (`ascend_step5_gated_scale`) so the
      fused call is torch.compile-friendly and has a registered fake impl.

## Test Cases

Single-operator accuracy tests compare the kernel against the eager path
it replaces (`attn_output.view(t, H, D) * gate.unsqueeze(-1).sigmoid()`)
under the repository bf16 criterion (`rtol = atol = 2^-6`, at least 99% of
elements within tolerance, max abs error <= 1). The gate is exercised in
every layout the model feeds the base+offset interface: a standalone
contiguous gate with offset zero (the sliding-window `g_proj` layout), a
strided column view of a wider packed buffer with the buffer itself as
base (the `qkvg` layout), and the same view with the buffer's front
row-slice as base (the DSA `qkv` layout, where the gate columns live past
the base's logical width). Shapes use the real Step5 attention geometry
(head_dim 192 plus the other supported 128; per-rank head counts 4/8/16
for TP16/TP8/TP4 on the 64-head checkpoint) with token counts spanning
single-token decode, an odd boundary and a prefill chunk; an out-of-range
offset must fail loudly, and zero-token inputs return an empty output
without launching a kernel.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_gated_scale.py
```
