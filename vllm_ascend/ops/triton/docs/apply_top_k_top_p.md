# apply_top_k_top_p

> Source: `vllm_ascend/ops/triton/sample/apply_top_k_top_p.py` (`_apply_topk_topp_tiled_kernel`, exposed as `apply_top_k_top_p`).

## Description

- **Function**: Top-k/top-p filtering of LM-head logits using CANN/upstream `topk_renorm` semantics. It backs the V2 model runner sampler, wired in `vllm_ascend/patch/worker/patch_triton.py`, when `additional_config` sets `enable_triton_apply_topk_topp: true` (default `false`). `k` and `p` accept per-request `[B]` tensors directly (vLLM `SamplingMetadata` shape), so the batch may mix different sampling parameters and no `.item()` host synchronization is needed. The op returns original-order masked logits, matching the V2 `gumbel_sample` contract.
- **Formula**: for request `b`, with `logits_sort, logits_idx = sort(logits, dim=-1, descending=false, stable=true)` and vocabulary size `V`, the operator implements the CANN/upstream `topk_renorm` semantics:
    - `topKValue[b] = logits_sort[b][V - k[b]]`;
    - `logits_sort = where(logits_sort < topKValue, -inf, logits_sort)`;
    - `probsSum = cumsum(softmax(logits_sort, dim=-1), dim=-1)` (normalized over the top-k survivors);
    - `topPMask[b][v] = probsSum[b][v] <= 1 - p[b]`, `topPMask[b][-1] = false`.

  The row is sorted ascending, so the survivor set is described by a boundary scalar `m = clamp(max(m_k, m_p), 0, V-1)` and the kernel scatters the exact suffix `offs >= m`. All `exp` / `sum` / `cumsum` run in fp32 after the usual max subtraction.
- **Algorithm flow** (rows are independent; each program owns `rows_per_prog` consecutive rows; one unified tiled kernel serves every vocabulary size):
  1. Host: validate/normalize inputs — `k` → `[B]` int32, `p` → `[B]` fp32 (`None` disables, scalar broadcasts, `[B]` tensor passes through); pre-fill `out`; derive the tile size from the runtime UB size (shared helpers in `vllm_ascend/ops/triton/triton_utils.py`, 90% UB budget, conservative 52 B/elem footprint, floored to a power of 2 in `[16, 16384]`, cached per device); pick the grid from `get_vectorcore_num()` with `rows_per_prog = cdiv(B, core_num)`; `BLOCK = min(tiled_block, next_power_of_2(V))`.
  2. Pass 1 streams the row in `BLOCK` tiles, accumulating the top-k-survivor denominator (`sum_exp_topk`) and `top_k_boundary`. All-`-inf` rows use a clamped max (`max_logits_safe`) to avoid `exp(-inf - -inf) = NaN` and are marked `all_neg_inf_row`.
  3. Pass 2 finds `top_p_boundary` (first index whose cumulative probability exceeds `1 - p[b]`) with a running cross-tile prefix (`running_sum + tl.cumsum`), dividing by `sum_exp_topk_total`.
  4. Pass 3 scatters only the surviving lanes: suffix comparison `offs >= mask_boundary`; dead rows write nothing (output keeps its prefill).
  5. All loops use the static start `0` with masks — tensor floor-division and runtime range starts are avoided. The scatter mask must never be a cumsum-derived boolean vector (triton-ascend 3.2 miscompiles it).
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT Products. Tile sizes derive from the runtime UB size rather than per-chip constants (Atlas A2/A3, 192 KB UB → block 2048; 950, 248 KB UB → 4096). 310P is explicitly rejected by `is_triton_apply_topk_topp_supported()`. The V2 sampler uses runtime dispatch: the native CANN `torch_npu.npu_top_k_top_p` op first; this Triton op only on hardware whose profile lacks `NPU_TOP_K_TOP_P` and `--additional-config '{"enable_triton_apply_topk_topp": true}'` is set; the upstream PyTorch reference last. The Triton fallback matches the native op and the upstream reference. It runs eagerly in the sampler post-processing path, outside ACL Graph capture. The Triton branch additionally applies a measured batch/vocab crossover gate (`is_triton_topk_topp_profitable`, large V requires B >= 16, small V B >= 8 on A5).

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits` | Input (sorting entries only) | Raw LM-head logits, `[B, V]`; sorted ascending with `torch.sort(..., stable=True)` on the host before the kernel | fp32 / fp16 / bf16 | ND |
| `sorted_values` (`sv_ptr`) | Input | Ascending sorted logits, `[B, V]`, contiguous | fp32 / fp16 / bf16 | ND |
| `sorted_indices` (`si_ptr`) | Input | Permutation from `torch.sort`, `[B, V]`; values `< V`, not cast on the host | int32 / int64 | ND |
| `k` (`k_ptr`) | Input | Per-request keep count, normalized to `[B]` int32; `k[b] >= V` or `None` disables top-k, `k[b] <= 0` keeps only the max | int32 / int64 scalar or tensor | ND |
| `p` (`p_ptr`) | Input | Per-request top-p threshold, normalized to `[B]` fp32; `None` or `1.0` disables top-p | fp32 scalar or tensor | ND |
| `B` | Input (attribute) | Batch size (number of rows); `do_not_specialize` | int32 | scalar |
| `V` | Input (attribute) | Vocabulary size; runtime argument, tail of each tile masked; `do_not_specialize` | int32 | scalar |
| `rows_per_prog` | Input (attribute) | Rows owned by one program, `cdiv(B, get_vectorcore_num())`; `do_not_specialize` | int32 | scalar |
| `BLOCK` | Attribute (`tl.constexpr`) | Vocabulary tile length, `min(ub-derived tiled block, next_power_of_2(V))` | int32 | scalar |
| `USE_I32` | Attribute (`tl.constexpr`) | Use int32 row offsets when `B * V < 2^31`, int64 otherwise | bool | scalar |
| `out` (`out_ptr`) | Output | `[B, V]`, pre-filled on the host with `-inf` (masked logits, input dtype); the kernel only writes surviving positions | fp32 / fp16 / bf16 | ND |

## Constraints

- `sorted_values` / `sorted_indices`: 2-D `[B, V]`, same shape, same NPU device, non-empty; non-contiguous inputs are made contiguous on the host. `sorted_values` must come from an ascending **stable** sort — bit-exactness with CANN relies on the deterministic tie order. `V < 2^31` (int32 index/offset arithmetic).
- `k`: `None`, Python int/float (integral), or `[B]` / scalar int32/int64 tensor; bool is rejected (`TypeError`), non-integral float is rejected (`ValueError`), wrong-shape tensors are rejected. `k[b] >= V` means no filtering (vLLM's disabled convention), `k[b] <= 0` keeps only the maximum.
- `p`: `None`, Python float, or `[B]` / scalar floating tensor; bool is rejected, wrong-shape tensors are rejected. The top-p comparison is strict (`cumsum > 1 - p`), and the last (maximum-probability) position always survives.
- Semantics: `topk_renorm` is bit-exact with CANN `npu_top_k_top_p` for fp32; bf16/fp16 boundary differences come from p being cast to the logits dtype by CANN and from the kernel accumulating softmax/cumsum in fp32.
- Degenerate rows: an all-`-inf` row returns all `-inf`; a row containing NaN keeps one NaN output position (fail loud); NaN inputs are not silently sanitized.
- `BLOCK` is not a free tuning knob: it is derived from the runtime UB size with a 90% budget and a 52 B/elem simultaneously-live-tensor footprint, floored to a power of 2 in `[16, 16384]` and cached per device. Raising it by hand risks a UB overflow on chips with a smaller UB.
- The tiled kernel is intentionally branchless (pass 2) and uses static loop starts with masks; tensor-valued `break` conditions, tensor floor-division and runtime range starts must not be reintroduced. The scatter mask stays the comparison form (`s >= cutoff` / `offs >= m`), never a cumsum-derived boolean.
- Graph mode: not captured. The kernel runs eagerly in the sampler post-processing path; B/V/rows_per_prog are `do_not_specialize` so dynamic batches do not trigger recompilation. Only `BLOCK` and `USE_I32` are compile-time constants.
- Inference only: there is no backward pass.
- The V2 sampler integration is additionally gated by hardware (no 310P) and by `is_triton_topk_topp_profitable(B, V)`; direct op callers are not gated.

## Origin and Differences

- **Origin**: developed from scratch as a Triton implementation for the V2 model runner top-k/top-p sampling path.
- **Differences**:
    - Single documented semantics: `topk_renorm`, matching CANN/upstream vLLM and staying bit-exact with `torch_npu.npu_top_k_top_p` for fp32.
    - NPU adaptation for performance at high batch sizes: `k`/`p` per-request `[B]` tensors are consumed directly (no `.item()`, mixed parameters allowed); tile sizes are UB-derived per device instead of chip-specific constants; a single unified tiled kernel covers all vocabulary sizes.
    - `apply_top_k_top_p` accepts original-order logits and keeps the `*_with_sorted` helper internal; the V2 sampler only needs the masked-logits form.
    - The operator is gated by `additional_config`'s `enable_triton_apply_topk_topp` (default `false`), a 310P hardware gate, and a measured B/V crossover gate; when disabled or gated off the sampler keeps the stock PyTorch path.
    - Filtered positions are pre-filled on the host; unlike `_apply_top_k_top_p_pytorch`, the Triton op never modifies its inputs in place.

## Test Cases

Accuracy tests live in `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_top_k_top_p.py`:

- `test_masked_logits_bit_exact_cann` — masked logits bit-exact against `torch_npu.npu_top_k_top_p` over `B x V` in `{(4,1000),(8,4096),(4,4097),(8,32000)}` × `{fp32,bf16,fp16}`.
- `test_degenerate_rows` — all-`-inf`, many-`-inf`, and NaN rows.
- `test_scalar_and_none_variants` — scalar / `None` / single-element forms and disabled conventions.

V2 sampler routing tests live in `tests/ut/worker/v2/test_apply_top_k_top_p.py` (CANN > Triton > PyTorch priority, gate fallback, topk_renorm routing, untouched `None` inputs).

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_top_k_top_p.py
pytest -sv tests/ut/worker/v2/test_apply_top_k_top_p.py
```
