# LightningAttention

This document covers the four Triton kernels in `vllm_ascend/ops/triton/mamba/lightning_attn.py`:
`_fwd_diag_kernel`, `_fwd_kv_parallel`, `_fwd_kv_reduce` and `_fwd_none_diag_kernel`.
They have no individual entry point — they only ever run as one pipeline dispatched by
`_attention.forward` — so they share a single document.

## Supported Products

| Product | Supported |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 Training Series Product/Atlas A3 Inference Series Product</term>|      √     |
|<term>Atlas A2 Training Series Product/Atlas A2 Inference Series Product</term>|      √     |
|<term>Atlas 200I/500 A2 Inference Product</term>|      ×     |
|<term>Atlas Inference Series Accelerator Card Product</term>|      ×     |
|<term>Atlas Training Series Product</term>|      ×     |

> The operator does not go through `DeviceOperator`: A2/A3 and Ascend 950 receive
> identical launch parameters and there is no adaptor branch.
> On 310P, `HAS_TRITON` is False and BailingMoE linear attention does not take this path.
> `_fwd_diag_kernel` additionally passes the Triton-Ascend private compilation options
> `multibuffer` / `set_workspace_multibuffer` / `tile_mix_vector_loop` / `tile_mix_cube_loop`.
> Those are Ascend-backend specific, so the launch code cannot be reused as-is on other backends.

## Description

- **Function**: forward operator for the prefill stage of BailingMoE linear attention
  (lightning attention), the NPU replacement for `MiniMaxText01LinearKernel` on GPU.
  Given $Q, K, V$ and a per-head decay rate $s$, it computes causal linear attention with
  exponential decay and writes the KV state at the end of the sequence back into
  `kv_history` (the mamba cache), from which the decode steps continue the recurrence.
  The caller is `AscendBailingMoELinearAttention._prefill_and_mix_infer`
  (`vllm_ascend/ops/bailing_moe_linear_attn.py`), which promotes QKV to float32 before
  calling this operator.

- **Formula**: the algorithm tiles the sequence with `BLOCK = 256` and computes the
  intra-block (diagonal) and cross-block (non-diagonal) parts separately:

    $$
    O_t = \underbrace{\sum_{j \le t,\ \lfloor j/B \rfloor = \lfloor t/B \rfloor} e^{-s (t-j)} (Q_t \cdot K_j) V_j}_{\text{\_fwd\_diag\_kernel}}
        + \underbrace{\sum_{j < t,\ \lfloor j/B \rfloor \ne \lfloor t/B \rfloor} e^{-s (t-j-1)} (Q_t \cdot K_j) V_j
        + e^{-s t} \, Q_t \cdot KV_{\text{hist}}}_{\text{\_fwd\_kv\_parallel} \to \text{\_fwd\_kv\_reduce} \to \text{\_fwd\_none\_diag\_kernel}}
    $$

    The KV state is updated as:

    $$
    KV_{\text{hist}}' = e^{-s n} KV_{\text{hist}} + \sum_{j=0}^{n-1} e^{-s (n-1-j)} K_j \otimes V_j
    $$

    where $B$ is `BLOCK` and $n$ is the sequence length. **Note that the cross-block
    exponent is $t-j-1$, not $t-j$**; see Constraints.

- **Algorithm flow**:

    | Step | Kernel | Role |
    |---|---|---|
    | 1 | `_fwd_diag_kernel` | intra-block causal attention, writes `Out` directly |
    | 2 | `_fwd_kv_parallel` | each block independently computes $\sum_j e^{-s(B-1-j)} K_j \otimes V_j$ into `KV[b, h, block]` |
    | 3 | `_fwd_kv_reduce` | **exclusive** prefix scan of `KV` along the block axis: `KV[i]` is overwritten with the state in front of block `i`, and the final state is written back to `KV_HISTORY` |
    | 4 | `_fwd_none_diag_kernel` | $Q_t \cdot KV[\text{block}] \cdot e^{-s\,t_{\text{local}}}$, **accumulated** onto the `Out` of step 1 |

- **Dispatch chain**:

    ```text
    AscendBailingMoELinearAttention._prefill_and_mix_infer   vllm_ascend/ops/bailing_moe_linear_attn.py
      └─ linear_attention_prefill_and_mix(prefix_fn=...)     vLLM upstream
           └─ AscendLightningAttentionKernel.jit_linear_forward_prefix   this file
                └─ lightning_attention_npu                   this file (d-dimension chunking)
                     └─ lightning_attention_npu_ = _attention.apply      this file
                          ├─ _fwd_diag_kernel                Triton kernel
                          ├─ _fwd_kv_parallel                Triton kernel
                          ├─ _fwd_kv_reduce                  Triton kernel
                          └─ _fwd_none_diag_kernel           Triton kernel
    ```

- **Task partitioning**:

    | Kernel | Grid | Notes |
    |---|---|---|
    | `_fwd_diag_kernel` | `(b*h*NUM_BLOCK, BLOCK//32)` | dim-0 flattens batch-head and block index (`off // NUM_BLOCK` / `off % NUM_BLOCK`); dim-1 is the 32-row sub-tile within a block |
    | `_fwd_kv_parallel` | `(b*h, NUM_BLOCK, 2)` | dim-2 splits `e` into two `E_FBLOCK = e/2` column tiles to stay within UB |
    | `_fwd_kv_reduce` | `(b*h, 2)` | the block axis must stay serial (prefix scan); only batch-head and the `e` tiles are parallel |
    | `_fwd_none_diag_kernel` | `(b*h, NUM_BLOCK*(BLOCK//64), 2)` | dim-1 flattens block index and the 64-row sub-tile |

    `NUM_BLOCK = cdiv(n, 256)`. UB budget estimates are in the comments of `_attention.forward`.

## Parameters

### Python API `AscendLightningAttentionKernel.jit_linear_forward_prefix`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
|-------|------------|------|---------|-----|
|`q`|Input|Queries, shape `[h, n, d]` (unsqueezed internally when 3-D) or `[1, h, n, d]`.|fp32 / bf16 / fp16|ND|
|`k`|Input|Keys, same shape as `q`.|fp32 / bf16 / fp16|ND|
|`v`|Input|Values, shape `[h, n, e]`.|fp32 / bf16 / fp16|ND|
|`kv_caches`|Input/Output|KV state (mamba cache), shape `[h, d, e]`, **updated in place** with the state at the end of the sequence.|fp32|ND|
|`slope_rate`|Input|Per-head decay rate $s$, shape `[h]` or `[1, h, 1, 1]`, cast to float32 internally.|fp32|ND|
|`block_size`|Attribute|**Currently ignored**; the tile length is fixed at 256. See Constraints.|int32|-|
|`layer_idx`|Attribute (optional)|Unused; present only to match the upstream `prefix_fn` signature.|int32|-|
|return value|Output|Shape `[n, h*e]`, produced by `rearrange(o, "h n d -> n (h d)")`.|same as `q`|ND|

### Python API `lightning_attention_npu`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
|-------|------------|------|---------|-----|
|`q` / `k`|Input|Shape `[b, h, n, d]`.|fp32 / bf16 / fp16|ND|
|`v`|Input|Shape `[b, h, n, e]`.|fp32 / bf16 / fp16|ND|
|`ed`|Input|Decay rate; reshaped internally with `view(1, -1, 1, 1)` when 1-D.|fp32|ND|
|`block_size`|Attribute|**Ignored**.|int32|-|
|`kv_history`|Input (optional)|Shape `[b, h, d, e]`; a zero tensor is allocated when None, otherwise it is `clone()`d so the caller's tensor is not modified.|fp32|ND|
|return value 0|Output|Attention output `[b, h, n, e]`.|same as `q`|ND|
|return value 1|Output|`kv`, shape `[b, h, NUM_BLOCK+1, d, e]`; the first `NUM_BLOCK` entries are the exclusive prefix states of each block and the trailing entry is the final state.|fp32|ND|

### Kernel APIs

Common to all four kernels: `b`, `h`, `d`, `e` and `BLOCK` are `tl.constexpr`.
`n` and `NUM_BLOCK` are `tl.constexpr` in `_fwd_diag_kernel` / `_fwd_kv_parallel` and
runtime parameters in `_fwd_kv_reduce` / `_fwd_none_diag_kernel`.
**No parameter carries `do_not_specialize`**, so every new `(b, h, n, d, e)` combination
triggers a recompilation.

#### `_fwd_diag_kernel`

| Parameter | Input/Output/Attribute | Description | Data type |
|-------|------------|------|---------|
|`Q` / `K` / `V`|Input|Contiguous tensors, addressed as `off_bh * n * d` (or `* n * e`).|fp32 / bf16 / fp16|
|`Out`|Output|`[b, h, n, e]`; this kernel **writes** (does not accumulate), so it must run before `_fwd_none_diag_kernel`.|same as `Q`|
|`S`|Input|Per-head decay rate, read as a scalar at `off_h = off_bh % h`.|fp32|
|`CBLOCK`|Attribute|`tl.constexpr`, rows per sub-tile within a block; the wrapper fixes it at 32.|int32|
|`NUM_BLOCK`|Attribute|`tl.constexpr`, number of blocks; also used to recover `off_bh` / `off_block` from `program_id(0)`.|int32|

#### `_fwd_kv_parallel`

| Parameter | Input/Output/Attribute | Description | Data type |
|-------|------------|------|---------|
|`K` / `V`|Input|As above.|fp32 / bf16 / fp16|
|`K_decay`|Input|`[h, BLOCK]`, precomputed by the wrapper as `exp(-s * (BLOCK - (arange+1)))`.|fp32|
|`KV`|Output|`[b, h, NUM_BLOCK, d, e]`; each block writes its own slice.|fp32|
|`D_FBLOCK`|Attribute|`tl.constexpr`; the wrapper always passes `d` (the `d` axis is not split).|int32|
|`E_FBLOCK`|Attribute|`tl.constexpr`; the wrapper always passes `e // 2`.|int32|
|`NUM_FBLOCK`|Attribute|`tl.constexpr`; passed in but **unused** inside the kernel.|int32|
|`CBLOCK` / `NUM_CBLOCK`|Attribute|`tl.constexpr`, 64 and `BLOCK // 64 = 4`.|int32|

#### `_fwd_kv_reduce`

| Parameter | Input/Output/Attribute | Description | Data type |
|-------|------------|------|---------|
|`S`|Input|Per-head decay rate.|fp32|
|`KV`|Input/Output|`[b, h, NUM_BLOCK, d, e]`, overwritten **in place** with the exclusive prefix scan.|fp32|
|`KV_HISTORY`|Input/Output|`[b, h, d, e]`, overwritten **in place** with the final state.|fp32|
|`n` / `NUM_BLOCK`|Attribute|Runtime parameters (not `constexpr`), used to compute the real tail-block length `min(n - i*BLOCK, BLOCK)`.|int32|

#### `_fwd_none_diag_kernel`

| Parameter | Input/Output/Attribute | Description | Data type |
|-------|------------|------|---------|
|`Q`|Input|As above.|fp32 / bf16 / fp16|
|`Out`|Input/Output|Reads back the result of `_fwd_diag_kernel`, accumulates onto it and stores, so the intermediate result is rounded to `q.dtype` once.|same as `Q`|
|`S`|Input|Per-head decay rate.|fp32|
|`KV`|Input|The prefix states produced by `_fwd_kv_reduce`.|fp32|
|`E_FBLOCK` / `CBLOCK` / `NUM_CBLOCK`|Attribute|`tl.constexpr`, `e/2`, 64 and 4.|int32|

## Constraints

- Inference forward only. `_attention` subclasses `torch.autograd.Function` and calls
  `save_for_backward`, but **implements no `backward`** — a backward pass raises.
- **Cross-block decay carries an extra factor of $e^{s}$ relative to the exact recurrence.**
  The `k_decay` of `_fwd_kv_parallel` decays a block state down to the **last token** of
  that block (`exp(-s*(BLOCK-1-j))`), while `_fwd_none_diag_kernel` replays it as
  `exp(-s*t_local)`, so a cross-block token pair is weighted $e^{-s(t-j-1)}$ where an
  intra-block pair at the same distance is weighted $e^{-s(t-j)}$. This matches the
  upstream vLLM / MiniMax GPU implementation — it is porting fidelity, not something
  introduced here — so it is left unchanged, and the reference in the tests is built to
  the same convention. Sync with upstream before changing the decay convention.
- **`d` must be ≤ 128.** For `d > 128`, `lightning_attention_npu` chunks `d` with
  `m = 128`, but each iteration passes the **whole** `kv_history` (`[b, h, d, e]`) to the
  kernel, which addresses it as `off_bh * d * e` with `d = 128` — the reads and writes
  land in the wrong place. The function also returns the `kv` of the last chunk only.
  BailingMoE uses `head_dim = 128` and takes the single-chunk path, so it is unaffected.
- `d` and `e` must be powers of two (required by `tl.arange(0, d)` /
  `tl.arange(0, E_FBLOCK)`), and `e` must be divisible by 2 (`E_FBLOCK = e // 2`,
  asserted in the wrapper).
- **`block_size` is ignored.** Both `jit_linear_forward_prefix` and
  `lightning_attention_npu` accept `block_size`, but `_attention.forward` hardcodes
  `BLOCK = 256`; passing 64 gives exactly the same result as passing 256. This silent
  behaviour is worth knowing because block boundaries feed into the decay convention above.
- `kv_history` / `kv_caches` must be float32 and contiguous; `_fwd_kv_reduce` **writes
  them in place**. A test must `clone()` before dispatching, otherwise the reference is
  handed the already-updated value.
- `lightning_attention_npu_` (that is, `_attention.apply`) also **overwrites** the
  `kv_history` passed to it; only the outer `lightning_attention_npu` performs a `clone()`.
- Prefix semantics: the returned `kv[:, :, i]` is the state **in front of** block `i`
  (exclusive scan), so `kv[:, :, 0]` always equals the incoming `kv_history` and the final
  state is `kv[:, :, -1]`.
- Segmented prefill is equivalent to one-shot prefill only on a **256-aligned** split.
  Splitting elsewhere changes which token pairs count as "same block", so the result
  differs from the one-shot computation (by a factor of order $e^{s}$).
- For tail-block padding rows, `_fwd_diag_kernel` performs an **extra `tl.where` reset**
  on top of `tl.load(..., other=0.0)` (#10276): on Ascend, the vector-to-cube transfer may
  not clear out-of-bound data, and the residue reaching `tl.dot` produces NaN.
  Out-of-bound rows in `_fwd_kv_parallel` currently rely on **the mask alone, with no
  `tl.where` fallback**; if the same problem shows up there, it should be fixed the way
  #10276 did.
- For the tail block, `_fwd_kv_parallel` uses `left_shift` to move the sub-tiles left so
  they align to the end of the block, so the first sub-tile reads addresses **in front of**
  the start of the block (masked off). For the first block of a sequence with `n < 64`,
  that address is before the start of the tensor, which relies on the masked load not
  faulting.
- Variable-length / multi-sequence batching is not handled here:
  `jit_linear_forward_prefix` asserts `output.shape[0] == 1`, so the batch dimension must
  be 1; upstream `linear_attention_prefill_and_mix` splits multiple sequences and calls
  this operator per sequence.
- With no `do_not_specialize`, every change of `(b, h, n, d, e)` triggers four
  recompilations. That is a first-token latency characteristic, not a correctness issue,
  but it bounds how large a test shape grid can reasonably get.

## Test Cases

Numerical tests: `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lightning_attn.py`

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lightning_attn.py
```

The four kernels have no individual dispatch site, so the tests reach them through the
three Python entry points. The file has 13 test functions, 29 cases after
parametrization:

| Test function | Cases | Coverage |
|---|---|---|
|`test_lightning_attention_npu_single_chunk`|7|Baseline numerical comparison of `lightning_attention_npu_`: `n` below and equal to `BLOCK`, unaligned `n`, `b > 1`, `d = 64/128`, `e != d`, bf16/fp16/fp32|
|`test_lightning_attention_npu_single_chunk_with_kv_history`|2|Non-zero incoming `kv_history`, plus the final state|
|`test_lightning_attention_npu_multi_block`|2|`n > BLOCK` (300 / 768): the cross-block path and the final state, with NaN / Inf assertions|
|`test_lightning_attention_npu`|4|The outer `lightning_attention_npu` (the d-chunking entry point) with `kv_history=None`|
|`test_lightning_attention_npu_with_kv_history`|2|As above, with a non-zero `kv_history`|
|`test_ascend_lightning_attention_kernel_prefix`|5|The production entry point `jit_linear_forward_prefix`: the `[h, n, d]` layout, the `[n, h*e]` output and the in-place `kv_caches` update|
|`test_ascend_lightning_attention_kernel_prefix_with_history`|1|As above, with non-zero `kv_caches`|
|`test_ascend_lightning_attention_kernel_prefix_multi_block`|1|A multi-block sequence through the production entry point, with shape and NaN / Inf assertions|
|`test_lightning_attention_npu_output_shapes` / `test_ascend_kernel_prefix_output_shape`|2|Output shapes|
|`test_lightning_attention_output_no_nan`|1|Output free of NaN / Inf (related to #10276)|
|`test_lightning_attention_causal_property`|1|Causality: rewriting `V` after $t$ must not change the output before $t$|
|`test_lightning_attention_decay_effect`|1|Different decay rates produce different outputs, each matching the reference|

The reference is a plain PyTorch fp32 implementation that reproduces the four kernel
steps one by one (diagonal blocks → per-block KV outer product → prefix scan →
non-diagonal blocks) and mirrors the single rounding of the output to the input dtype
between the two kernels, so it follows the same tiling convention as the kernels.
The closed form given under Description is an equivalent formulation of the same
algorithm and can be used to write an independent reference when adding cases.

Tolerances:

| dtype | rtol | atol |
|---|---|---|
| float32 | 1e-2 | 1e-2 |
| float16 / bfloat16 | 5e-2 | 5e-2 |

The low-precision tiers are wider because `Out` is rounded to the input dtype once
between the two kernels (the diagonal result is stored, then read back and accumulated
by the non-diagonal kernel). Inputs are drawn at a small scale with small decay rates,
and the multi-block cases shrink `ed` further, to about 0.01, to keep the cross-block
error in range.

Per kernel:

| Kernel | Reached by |
|---|---|
|`_fwd_diag_kernel`|All numerical cases (the intra-block term is always present); `test_lightning_attention_causal_property` targets its causal mask directly; the unaligned-`n` cases exercise tail-block padding (#10276)|
|`_fwd_kv_parallel` / `_fwd_kv_reduce`|The `n > BLOCK` cases, and every case checking the final KV state (`kv[:, :, -1]`); the non-zero `kv_history` cases cover the `kv_pre` load path|
|`_fwd_none_diag_kernel`|The `n > BLOCK` cases (only they produce cross-block terms) and the non-zero `kv_history` cases (the history term is replayed through it)|

Not covered, and why:

- **The `d > 128` chunking path.** As described under Constraints, that path is
  semantically wrong today (misaligned `kv_history` offset, and only the last chunk's `kv`
  is returned). A test would lock the wrong behaviour in, so it is only recorded here; the
  deployed `head_dim = 128` never reaches it. A fix should be a separate PR, with the
  regression case added there.
- **Recompilation behaviour.** The per-shape recompilation caused by the absence of
  `do_not_specialize` is a performance characteristic, and detecting it depends on Triton's
  internal JIT cache structure, which is unstable across versions.
- **Backward.** The operator implements no `backward`; inference does not need one.
- When adding cases, keep the shape grid small: `b/h/n/d/e` are all `constexpr`, so every
  new combination costs four recompilations, and the nightly conftest skips the remaining
  cases of a file once five of them exceed 120s — a combinatorial grid ends up "passing"
  while its tail never runs.

## Change Log

| PR | Description |
|---|---|
|[#8657](https://github.com/vllm-project/vllm-ascend/pull/8657)|Introduced the four kernels for BailingMoE linear attention|
|[#8702](https://github.com/vllm-project/vllm-ascend/pull/8702)|Switched from monkey-patching to `PluggableLayer` registration; the caller became `AscendBailingMoELinearAttention`|
|[#10276](https://github.com/vllm-project/vllm-ascend/pull/10276)|Fixed the NaN caused by uncleared tail-block padding: `_fwd_diag_kernel` now re-masks `q` / `k` with `tl.where` after `tl.load`|
