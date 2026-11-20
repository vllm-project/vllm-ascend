# Step5 indexer_qkz_norm_rope

## Description

- **Function**: Fuses the Step5 sparse-indexer post-projection chain of the
  DSA layers — per-head RMSNorm on `index_q`, per-head LayerNorm on
  `index_k`, NeoX RoPE on both, and a contiguous copy of `index_z` — into a
  single Triton kernel. It replaces the eager chain in
  `AscendStep5DSAAttention._finish_indexer`
  (`k_norm -> q_norm -> rotary_emb -> contiguous(z)`), which materialized
  every intermediate back to global memory and forced contiguous copies of
  the strided `split_indexer` views. The kernel reads the three inputs in
  place through their row strides (the production inputs are the
  `split_indexer` slices of one packed qkv+indexer GEMM row), so no
  materializing copy is issued on any input. `index_q` is stored as
  `float8_e4m3fn`: its only consumer (the fused indexer scoring,
  `indexer_score_topk` via `indexer_region_score`) wants fp8, so the
  quantization rides along with the norm+RoPE store instead of a separate
  cast at the scoring entry. The wrapper is a plain Python call into the Triton
  kernel; Step5 currently runs with `do_not_compile`. Once the outer
  `step5_sparse_attention_forward` custom op from #22342 is in place, that
  op provides the torch.compile boundary.
- **Location**: `vllm_ascend/ops/triton/step5/indexer_qkz_norm_rope.py` —
  kernel `_indexer_qkz_norm_rope_kernel`, wrapper
  `fused_indexer_qkz_norm_rope`. Production caller:
  `vllm_ascend/models/step5/model.py`
  (`AscendStep5DSAAttention._finish_indexer`), selected when
  `0 < sparse_indexer_rope_dim < proxy_dim` (the production partial-RoPE
  geometry). **Scope**: model-specific to Step5-Preview's dense-sparse
  attention (DSA) layers; not a framework-wide primitive.
- **Formula**:
    - Input `index_q`: `[tokens, num_q_heads * head_dim]` (head_dim =
      `proxy_dim`, 256 in production); `index_k` / `index_z`:
      `[tokens, num_k_heads * head_dim]`. All three may be strided column
      views of one packed buffer.
    - index_q: RMSNorm per head over `D = head_dim`, fp32 reduction,
      zero-centered weights add 1.0 (`q_norm_weight_bias`), no mean
      subtraction, no bias: `y = x * rsqrt(mean(x^2) + eps) * (w + 1.0)`.
    - index_k: LayerNorm per head (mean subtracted, optional bias):
      `y = (x - mean) * rsqrt(var + eps) * w + b`.
    - NeoX RoPE over the first `rotary_dim` dims = `rotary_dim // 2` pairs
      (pair layout: dims `i` and `i + rotary_dim // 2`):
      `out1 = y1*cos - y2*sin`, `out2 = y1*sin + y2*cos`; angles are read
      from a single caller-provided `[cos | sin]` cache (last dimension
      `rotary_dim`), rows gathered by `positions`; the sin half is
      addressed as `cos + rotary_dim // 2`. The normed values are rounded to the input
      dtype (bf16) before rotation, rotated in fp32 (matches the eager
      path's dtype handoffs).
    - Stores round through the input precision first (bf16 -> output
      dtype), so index_q reproduces the unfused pipeline's bits
      (kernel-store bf16 -> `.to(float8_e4m3fn)` cast) up to one e4m3 ULP;
      same-dtype outputs take the second cast as a no-op. `index_z` is a
      bit-exact passthrough.
- **Algorithm flow** (persistent-style grid over tokens):
   1. Grid `min(max(tiles_q, tiles_kz), num_vector_cores)`;
      `ROWS_PER_PROGRAM_Q` / `ROWS_PER_PROGRAM_KZ` are selected once per
      layer shape (`head_dim`, `rotary_dim`) and reused for all token
      counts, instead of being tuned again for each token-count bucket.
   2. Q stage: each program loops over its token tiles; per tile the
       head axis is chunked (`H_CHUNK` sized to a 4096-lane budget) and
       each chunk is normed, rotated and stored through block pointers.
   3. K stage rides the same traversal and additionally copies the Z rows
       bit-exact (block-pointer load/store).
- **Supported modes**: Ascend 950PR/950DT (A5) on the Triton-Ascend
  backend; execution evidence on Ascend950DT. Atlas A2/A3 are not validated
  for this operator (it uses the Triton-Ascend CANN `extension`
  slice/compile-hint intrinsics and `num_vectorcore`), matching the other
  Step5 operators. Eager and ACLGraph decode capture both work (the
  wrapper allocates outputs and derives the grid from shapes only). The
  fused path handles the configuration's allowed even `rotary_dim` values
  up to `proxy_dim`; there is no eager fallback.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `index_q` | Input | Indexer Q logits, `[tokens, num_q_heads * head_dim]` | bf16 | ND; may be a strided view, row stride passed to the kernel |
| `index_k` | Input | Indexer K logits, `[tokens, num_k_heads * head_dim]` | bf16 | ND; may be a strided view |
| `index_z` | Input | Indexer Z values, `[tokens, num_k_heads * head_dim]` | bf16 | ND; may be a strided view |
| `q_norm_weight` | Input | Per-head RMSNorm scale for Q, `[head_dim]` | bf16 | ND (1-D) |
| `k_norm_weight` | Input | Per-head LayerNorm scale for K, `[head_dim]` | bf16 | ND (1-D) |
| `cos_sin_cache` | Input | Single `[cos \| sin]` cache for the indexer RoPE (last dim = `rotary_dim`); row stride passed to the kernel | bf16 | ND 2-D `[max_pos, rotary_dim]` |
| `positions` | Input | Absolute position of each token | int32 / int64 | ND (1-D) |
| `k_norm_bias` | Input | Required LayerNorm bias for K | bf16 | ND (1-D) |
| `head_dim` | Attribute | Per-head width (`proxy_dim`, 256 in production) | int | - |
| `num_q_heads` | Attribute | Local index Q heads (constexpr) | int | - |
| `num_k_heads` | Attribute | Local index K heads (constexpr) | int | - |
| `rotary_dim` | Attribute | Rotated dim count (`sparse_indexer_rope_dim`, 32 in production); converted to NeoX pairs internally | int | - |
| `eps` | Attribute | Norm epsilon (default `1e-6`) | float | - |
| `q_norm_weight_bias` | Attribute | Added to the Q norm scale: `1.0` for zero-centered weights, `0.0` otherwise | float | - |
| return `index_q` | Output | `[tokens, num_q_heads * head_dim]`, normed + rotated, **float8_e4m3fn** (quantized at the store) | fp8 | ND |
| return `index_k` | Output | `[tokens, num_k_heads * head_dim]`, normed + rotated | bf16 | ND |
| return `index_z` | Output | `[tokens, num_k_heads * head_dim]`, bit-exact passthrough | bf16 | ND |

## Constraints

- `rotary_dim` must be even and `> 0`; it is a **dim count** (32 in
  production), converted to `rotary_dim // 2` NeoX pairs internally. Passing
  a pair count here historically corrupted the layout (imag half on dims
  32-63 and cross-row cache reads); the pos-0 identity test guards it.
- `index_q/k/z` must share the same token count; `index_q` width must equal
  `num_q_heads * head_dim`, `index_k`/`index_z` widths `num_k_heads *
  head_dim` (validated with loud errors).
- `cos_sin_cache` must use the `[cos | sin]` layout with last dimension
  exactly `rotary_dim`; `positions` must index valid rows.
- `rotary_dim <= head_dim`.
- Specialization / recompilation profile: compile-time specialization
  comes from the `constexpr` head counts and flags (required: head chunking
  and the static head loops derive from them; fixed per layer shape) and
  from Triton's value/alignment classes of the runtime integer arguments
  (`tokens`, the strides; classes `==1`, multiples of 16, other). The
  autotuner keys only on `head_dim` and `rotary_dim`, deliberately excluding
  token counts: the first call at each layer shape benchmarks the config
  list once, and all subsequent token counts reuse that selection instead
  of tuning again per batch. The selected configuration therefore depends on
  the first batch seen by the process; this is a performance trade-off, not
  a numerical difference. `do_not_specialize` is not used (the classes are
  few, fixed in serving, and the alignment specialization feeds
  vectorization).
- In-place behavior: none. Inputs are read-only (strided reads are
  supported); outputs are freshly allocated.

## Origin and Differences

- **Origin**: Adapted from the Step5 eager indexer chain in vLLM
  (`index_k_norm -> index_q_norm -> sparse_indexer_rotary_emb` plus the Z
  copy), fused for the tailring-npu deployment (internal
  `fuse-kv-write-qknorm` / `fused-indexq-fp8` lineage) and co-designed with
  the fused indexer scoring: the e4m3-at-store contract is what
  `indexer_score_topk` (the aclnn IndexerRegionScore adapter) already
  documents and consumes.
- **Differences**:
    - The eager path materializes each stage (including contiguous copies
      of the strided split views and a final bf16 index_q that the scoring
      entry re-quantizes); the fused kernel reads the views in place and
      folds the fp8 quantization into the store.
    - RoPE angles are always read from one caller-provided `[cos | sin]` cache;
      per-layer `rope_theta` (`1e7` for the DSA indexer) is the caller's
      responsibility when building the cache.
    - Exposed as a plain Python wrapper around the Triton kernel. The
      torch.compile boundary is provided by the outer
      `step5_sparse_attention_forward` custom op once #22342 lands.

## Test Cases

Single-operator accuracy tests compare against a pure-torch reference
mirroring the eager path's numerics (fp32 RMSNorm/LayerNorm, bf16 rounding
before rotation, fp32 rotation), under the repository operator criterion:
index_k (bf16) uses `rtol = atol = 2^-6` with max abs error <= 1; index_q
lands in e4m3 (its production dtype) and is compared against the reference
cast to the same dtype under `rtol = 2^-3`, `atol = 2^-6`, max abs error
4.0 (the criterion's `max(fixed_limit, 32 * ULP(1.0))` rule with the 3-bit
e4m3 mantissa); both require at least 99% of elements within tolerance.
index_z is compared bit-exact. The inputs are the strided `split` views of one packed row (the
production layout). Real Step5 indexer shape (`proxy_dim=256`,
`sparse_indexer_rope_dim=32`, `rope_theta=1e7`) plus head-count variants,
token counts from single-token decode to odd tiles. Coverage also includes
zero-token inputs, full-RoPE geometry (`rotary_dim == head_dim`), and a
large-mean/small-spread K row for the centered LayerNorm variance. A
pos-0 identity test guards the pair/dims layout regression.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_indexer_qkz_norm_rope.py
```
