# Step5 fused_qknorm_rope

## Description

- **This will be removed after qkv_rmsnorm_rope_simt supports head_dim=192**
- **Function**: Splits the fused Step5 attention QKV projection `[ q | k | v ]`,
  applies per-head RMSNorm (optional zero-centered Gemma bias) to Q and K,
  NeoX-style RoPE to the normed Q and K, and copies V through untouched, all
  in one Triton kernel. It replaces the eager chain
  `split -> RMSNorm(q) -> RMSNorm(k) -> rotary_emb(q, k)` (plus a separate V
  copy), which wrote every intermediate back to global memory. The wrapper is
  registered as the custom op `ascend_step5_fused_qknorm_rope`
  (`vllm.triton_utils.direct_register_custom_op`), so it is callable through
  `torch.ops.vllm.ascend_step5_fused_qknorm_rope` and safe for torch.compile.
- **Location**: `vllm_ascend/ops/triton/step5/fused_qknorm_rope.py` — kernel
  `_split_qkv_norm_rope_192_kernel`, wrapper `fused_qknorm_rope`.
  Production callers: `vllm_ascend/models/step5/model.py`
  (`AscendStep5DSAAttention.forward` and `AscendStep5Attention.forward`),
  selected when `use_rope and head_dim == 192`.
- **Formula**:
    - Input `qkv`: `[num_tokens, (num_q_heads + 2 * num_kv_heads) * 192]`,
      laid out as `[ q | k | v ]`, each head a contiguous 192-element row.
    - RMSNorm per head over `D = 192`, reduction in float32:
      `rstd = rsqrt(sum(x^2) / D + eps)`,
      `y = x * rstd * (w + norm_weight_bias)` where `w` is the per-head scale
      (`q_weight` / `k_weight`) and `norm_weight_bias` is `1.0` for
      zero-centered (Gemma-style) weights, `0.0` otherwise. The normed value
      is rounded to bf16, then re-expanded to fp32 for rotation (matches the
      eager path's bf16 handoff between RMSNorm and RoPE).
    - NeoX RoPE over the first `R = 2 * rotary_pairs` elements of each head:
      `y1 = y[..., :rotary_pairs]`, `y2 = y[..., rotary_pairs:2*rotary_pairs]`,
      `out = [y1*cos - y2*sin, y1*sin + y2*cos, y[..., 2*rotary_pairs:]]`.
      `cos`/`sin` rows are gathered from the caller-provided cache by
      `positions`. With the Step5 configuration (`rotary_pairs = 96`,
      `head_dim = 192`) every lane is rotated; a smaller `rotary_pairs`
      passes the tail through un-rotated.
    - V is copied bit-exact from the packed projection, fused into the K-loop
      of the kernel (saves one kernel launch vs a separate copy).
- **Algorithm flow** (processed tile by tile; tokens independent):
    1. Grid `(cdiv(num_tokens, BLOCK_M),)`; each program owns `BLOCK_M`
       tokens (autotuned over `BLOCK_M in {4, 8, 16, 32}`, keyed on the
       token count) and loops over all Q heads, then all K heads.
    2. Hoisted out of the head loops: the `cos`/`sin` rows for the tile
       (gathered once by `positions`), the lane index pairs
       `(i, i + rotary_pairs)`, and both norm weights.
    3. Per head: one masked 256-lane load (192 live lanes), fp32 sum-of-squares
       reduction, `rsqrt`, scale by `w + bias`, round to bf16, gather the two
       rotation halves, rotate in fp32 with one FMA per output lane, store
       the rotated bf16 halves plus the un-rotated tail.
    4. V passthrough rides along the K loop: the packed V head is copied as
       raw bf16 with the same tile traversal.
- **Supported modes**: Ascend A5 only (950 series, Triton-Ascend backend;
  execution evidence on Ascend950DT). Eager and ACLGraph decode capture
  paths both work (the wrapper allocates outputs and derives the grid from
  shapes only). Other device types keep the eager split + norm + RoPE path
  (the caller's `use_fused_qknorm_rope` guard).

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qkv` | Input | Fused projection `[ q \| k \| v ]`, `[num_tokens, (num_q_heads + 2 * num_kv_heads) * 192]` | bf16 | ND, row-major; row stride passed to the kernel |
| `q_weight` | Input | RMSNorm scale for every Q head, `[192]` | bf16 | ND (1-D) |
| `k_weight` | Input | RMSNorm scale for every K head, `[192]` | bf16 | ND (1-D) |
| `cos` | Input | First half of the per-layer `cos_sin_cache` rows (`rotary_cache.chunk(2, -1)`), indexed by `positions` | bf16 | ND 2-D `[max_pos, >=rotary_pairs]` |
| `sin` | Input | Second half of the cache; must share `cos`'s row stride | bf16 | ND 2-D |
| `positions` | Input | Absolute position of each token | int32 / int64 | ND (1-D) |
| `head_dim` | Attribute | Must be `192` (validated; anything else raises `ValueError`) | int | - |
| `num_q_heads` | Attribute | Q heads in `qkv` (constexpr) | int | - |
| `num_kv_heads` | Attribute | K (and V) heads in `qkv` (constexpr) | int | - |
| `rotary_pairs` | Attribute | Rotated lane pairs per head, `rotary_dim // 2` (constexpr; `96` on Step5) | int | - |
| `eps` | Attribute | RMSNorm epsilon (constexpr; default `1e-5`) | float | - |
| `norm_weight_bias` | Attribute | Added to the scale: `1.0` zero-centered, `0.0` plain (constexpr) | float | - |
| return `q` | Output | `[num_tokens, num_q_heads * 192]`, normed + rotated | bf16 | ND |
| return `k` | Output | `[num_tokens, num_kv_heads * 192]`, normed + rotated | bf16 | ND |
| return `v` | Output | `[num_tokens, num_kv_heads * 192]`, bit-exact copy of the packed V heads | bf16 | ND |

## Constraints

- `head_dim` must be exactly `192`; the wrapper raises `ValueError`
  otherwise, and layers with other head dims keep the eager split+norm+RoPE
  path (the caller's `use_fused_qknorm_rope` guard).
- `rotary_pairs <= 128` (kernel tile `BLOCK_R = 128`) and
  `rotary_pairs <= head_dim // 2`. Step5 uses `96`.
- `cos` and `sin` must be the two halves of one cache, i.e. share the same
  row stride; each row must hold at least `rotary_pairs` elements. Rotation
  angles are read from the cache — per-layer `rope_theta` / `rope_scaling`
  are the caller's responsibility when building it.
- Position values must index valid cache rows (`positions < max_pos`).
- `num_tokens == 0` returns three empty tensors without launching.
- Specialization / recompilation profile: `tokens` is annotated
  `do_not_specialize`, so varying batch sizes do not create new kernel
  variants; the autotuner picks `BLOCK_M` per token-count bucket from the
  four pre-compiled configs (first call at a new size pays four benchmark
  launches, not a compilation). The remaining variant axes are the
  constexprs (`num_q_heads`, `num_kv_heads`, `rotary_pairs`, `eps`,
  `norm_weight_bias`) and dtype/alignment classes of the pointers — all
  fixed per model config in serving, so the steady-state variant count is
  one per layer shape (two end to end: DSA and dense attention layers when
  their head counts differ).
- In-place behavior: none. `qkv` and the weights are read-only; outputs are
  freshly allocated. The op is registered with `mutates_args=[]`.

## Origin and Differences

- **Origin**: Adapted from the Step5 eager pre-attention chain in vLLM
  (`qkv.split -> q_norm/k_norm -> rotary_emb`) and the upstream Triton fused
  counterpart `vllm.models.step5.nvidia.ops.indexer_norm_rope` (same
  per-head RMSNorm + NeoX RoPE fusion, with the indexer-side cache updates
  folded into the same launch).
- **Differences**:
    - No cache insertion: the upstream `indexer_norm_rope` also folds the
      indexer cache updates into the same kernel (`UPDATE_CSA`); on Ascend
      the cache writes stay on the existing attention backend path, so this
      kernel returns K/V tensors instead.
    - The upstream Triton kernel reproduces the native QK kernel's
      sequential-FMA / XOR-tree reduction (preserving its CuTeDSL
      numerics); this is an Ascend SIMD implementation (vector tree
      reduction), numerically equivalent within RMSNorm+RoPE tolerance but
      not bit-identical to the upstream reference.
    - Rotation angles are always read from the caller's `cos/sin` cache. (An
      earlier revision baked `theta=1e4` into the kernel, which corrupted
      layers configured with a different `rope_theta`, e.g. `1e7`; reading
      the cache keeps per-layer scaling correct.)
    - Exposed as a registered custom op (`ascend_step5_fused_qknorm_rope`)
      so the fused call is torch.compile-friendly and has a registered fake
      impl.

## Test Cases

Single-operator accuracy tests compare against a pure-torch reference that
mirrors the kernel's numerics (fp32 RMSNorm reduction, bf16 rounding before
rotation, fp32 rotation, bf16 store). Q/K use the bf16 tolerance
(`rtol = atol = 2^-6`, at least 99% of elements within the mixed tolerance,
max absolute error `<= 1`); V is compared bit-exact. Real Step5 DSA-layer
shape (`num_q_heads=16, num_kv_heads=1, head_dim=192, rotary_pairs=96`) plus
GQA and partial-RoPE cases, token counts from 0 to 4096, both
`norm_weight_bias` values, and a `rope_theta=1e7` cache.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_fused_qknorm_rope.py
```
