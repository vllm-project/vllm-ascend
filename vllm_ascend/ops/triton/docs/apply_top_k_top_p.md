# apply_top_k_top_p

> Source: `vllm_ascend/ops/triton/apply_top_k_top_p.py` (`_apply_topk_topp_single_pass_kernel` / `_apply_topk_topp_tiled_kernel`, exposed as `apply_top_k_top_p`, `apply_top_k_top_p_with_sorted`, `fused_topk_topp_softmax`, `fused_topk_topp_softmax_with_sorted`).

## Description

- **Function**: Top-k/top-p filtering of LM-head logits, semantically identical to CANN `torch_npu.npu_top_k_top_p`, with an optional in-kernel fused softmax. It backs the vLLM sampler path in `vllm_ascend/sample/sampler.py` when `additional_config` sets `enable_triton_apply_topk_topp: true` (default `false`). `k` and `p` accept per-request `[B]` tensors directly (vLLM `SamplingMetadata` shape), so the batch may mix different sampling parameters and no `.item()` host synchronization is needed. The non-fused entries return original-order masked logits bit-exact with the CANN operator; the fused entries run the final softmax in the kernel and return fp32 probabilities, saving one extra softmax kernel launch and the full-vocabulary probability intermediate.
- **Formula**: for request `b`, with `sortedValue, sortedIndices = sort(logits, dim=-1, descending=false, stable=true)` and vocabulary size `V`:
    - `topKValue[b] = sortedValue[b][V - k[b]]`
    - `sortedValue = where(sortedValue < topKValue[b], -inf, sortedValue)`
    - `probsSum = cumsum(softmax(sortedValue, dim=-1), dim=-1)`
    - `topPMask[b][v] = probsSum[b][v] <= 1 - p[b]`; `topPMask[b][-1] = false`
    - `sortedValue = where(topPMask, -inf, sortedValue)`
    - `out[b][sortedIndices[b][v]] = sortedValue[b][v]`

  Because the row is sorted ascending, both the top-k and the top-p survivor sets are suffixes, so the kept set is described by a single scalar boundary `m`: `kept(v) <=> v >= m`, where `m = clamp(max(m_k, m_p), 0, V - 1)`, `m_k` is the first index with `sortedValue >= topKValue`, and `m_p` is the first index whose cumulative probability exceeds `1 - p[b]`. The kernel only computes `m` and scatters the surviving suffix; filtered positions are pre-filled on the host (`-inf` for masked logits, `0` for fused probabilities). All `exp` / `sum` / `cumsum` run in fp32 after the usual max subtraction, matching the CANN numerics.
- **Algorithm flow** (rows are independent; each program owns `rows_per_prog` consecutive rows):
  1. Host: validate/normalize inputs — `k` → `[B]` int32, `p` → `[B]` fp32 (`None` disables, scalar broadcasts, `[B]` tensor passes through); pre-fill `out`; derive tile sizes from the runtime UB size (shared helpers in `vllm_ascend/ops/triton/triton_utils.py`, 90% UB budget, conservative 36 B/elem single-pass and 48 B/elem tiled footprint, floored to a power of 2 in `[16, 16384]`, cached per process); pick the grid from `get_vectorcore_num()` with `rows_per_prog = cdiv(B, core_num)`.
  2. Small vocabularies (`V <= single_pass_block`, e.g. 4096 on Atlas A2/A3): `_apply_topk_topp_single_pass_kernel` loads the whole row in one `BLOCK = next_power_of_2(V)` tile, evaluates `topKValue`, the top-k softmax, both boundaries and the scatter in a single pass over global memory.
  3. Large vocabularies: `_apply_topk_topp_tiled_kernel` streams the row in `BLOCK` tiles. Pass 1 accumulates the top-k softmax denominator `E` and `m_k`; pass 2 finds `m_p` with a running cross-tile prefix (`running + tl.cumsum`), deliberately branchless because 0-D tensor control flow is not portable across Triton backends; pass 3 re-scans from the static loop start `0` and scatters only lanes with `offs >= m` (the fused variant splits this into pass 3a, denominator over the kept suffix, and pass 3b, scatter of `exp(s - s_max) / e_kept`).
  4. The scatter mask is always the comparison form `offs >= m`: a cumsum-derived boolean vector used directly as a scatter mask miscompiles on triton-ascend 3.2 and must not be reintroduced.
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT Products. Tile sizes are derived from the runtime UB size rather than pinned per chip (Atlas A2/A3, 192 KB UB → single-pass 4096 / tiled 2048; 950, 248 KB UB → 4096 / 4096). Enabled through `--additional-config '{"enable_triton_apply_topk_topp": true}'`; it runs in the sampler post-processing path, i.e. eagerly, outside ACL Graph capture.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits` | Input (sorting entries only) | Raw LM-head logits, `[B, V]`; sorted ascending with `torch.sort(..., stable=True)` on the host before the kernel | fp32 / fp16 / bf16 | ND |
| `sorted_values` (`sv_ptr`) | Input | Ascending sorted logits, `[B, V]`, contiguous | fp32 / fp16 / bf16 | ND |
| `sorted_indices` (`si_ptr`) | Input | Permutation from `torch.sort`, `[B, V]`; values `< V`, not cast on the host | int32 / int64 | ND |
| `k` (`k_ptr`) | Input | Per-request keep count, normalized to `[B]` int32; `k[b] >= V` or `None` disables top-k, `k[b] <= 0` clamps to keeping only the max | int32 / int64 scalar or tensor | ND |
| `p` (`p_ptr`) | Input | Per-request top-p threshold, normalized to `[B]` fp32; `None` or `1.0` disables top-p | fp32 scalar or tensor | ND |
| `B` | Input (attribute) | Batch size (number of rows) | int32 | scalar |
| `V` | Input (attribute) | Vocabulary size; runtime argument, tail of each tile is masked | int32 | scalar |
| `rows_per_prog` | Input (attribute) | Rows owned by one program, `cdiv(B, get_vectorcore_num())` | int32 | scalar |
| `BLOCK` | Attribute (`tl.constexpr`) | Vocabulary tile length; `next_power_of_2(V)` clamped to `[16, single_pass_block]` for the single-pass kernel, UB-derived `tiled_block` for the tiled kernel | int32 | scalar |
| `USE_I32` | Attribute (`tl.constexpr`) | Use int32 row offsets when `B * V < 2^31`, int64 otherwise | bool | scalar |
| `FUSED_SOFTMAX` | Attribute (`tl.constexpr`) | Scatter fp32 probabilities normalized over the kept suffix instead of raw logits | bool | scalar |
| `out` (`out_ptr`) | Output | `[B, V]`, pre-filled on the host with `-inf` (masked logits, input dtype) or `0` (fused probabilities, fp32); the kernel only writes surviving positions | fp32 / fp16 / bf16 | ND |

## Constraints

- `sorted_values` / `sorted_indices`: 2-D `[B, V]`, same shape, same NPU device, non-empty; non-contiguous inputs are made contiguous on the host. `sorted_values` must come from an ascending **stable** sort — bit-exactness with `torch_npu.npu_top_k_top_p` relies on the deterministic tie order. `V < 2^31` (int32 index/offset arithmetic).
- `k`: `None`, Python int/float (integral), or `[B]` / scalar int32/int64 tensor; bool is rejected. `k[b] >= V` means no filtering (vLLM's disabled convention), `k[b] <= 0` keeps only the maximum.
- `p`: `None`, Python float, or `[B]` / scalar floating tensor, values in `[0, 1]`; bool is rejected. The top-p comparison is strict (`cumsum > 1 - p`), and the last (maximum-probability) position always survives, matching the CANN operator.
- `BLOCK` is not a free tuning knob: it is derived from the runtime UB size with a 90% budget and a conservative simultaneously-live-tensor footprint (36 B/elem single-pass, 48 B/elem tiled), floored to a power of 2 in `[16, 16384]` and cached. Raising it by hand risks a UB overflow on chips with a smaller UB.
- The tiled kernel is intentionally branchless (pass 2) and uses a static loop start with a kept mask (pass 3); tensor-valued `break` conditions and runtime range starts are fragile across Triton backends and must not be reintroduced. Likewise the scatter mask must stay the comparison form `offs >= m` (triton-ascend 3.2 miscompiles cumsum-derived boolean scatter masks).
- Graph mode: not captured. The kernel runs eagerly in the sampler post-processing path and sees dynamic `B` on every call; only `BLOCK`, `USE_I32` and `FUSED_SOFTMAX` are compile-time constants (each distinct combination costs one compilation).
- Inference only: there is no backward pass.

## Origin and Differences

- **Origin**: developed from scratch as a Triton replacement for the CANN operator `torch_npu.npu_top_k_top_p` in the vLLM top-k/top-p sampling path (`vllm_ascend/sample/sampler.py`); the CANN operator is the semantic and bit-exactness reference.
- **Differences**:
    - NPU adaptation for performance: the stock sampler path extracts scalar `k`/`p` with `.item()` (one host synchronization per decode step) and requires uniform sampling parameters across the batch; this operator consumes the `[B]` sampling-parameter tensors directly, so no host synchronization is needed and mixed per-request `k`/`p` batches are supported. Tile sizes are derived from the runtime UB size via the shared `triton_utils` helpers instead of chip-specific constants, so one implementation serves Atlas A2/A3 and 950 products.
    - Modified for a specific vllm-ascend logic or different input parameters: the `fused_topk_topp_softmax` entries fold the final softmax into the kernel and return fp32 probabilities (masked positions `0`, each row sums to 1), equivalent to `torch_npu.npu_top_k_top_p(...).softmax(dim=-1, dtype=torch.float32)` but without the extra kernel launch and the full-vocabulary probability intermediate; the `*_with_sorted` entries accept an already-sorted input to skip the sort when the caller has one. The filtered positions are pre-filled on the host so the kernel only scatters the surviving suffix. The operator is gated by `additional_config`'s `enable_triton_apply_topk_topp` (default `false`); when disabled the sampler keeps the stock CANN path.

## Test Cases

Accuracy tests live in `tests/e2e/nightly/single_node/ops/singlecard_ops/test_apply_top_k_top_p.py`:

- `test_masked_logits_bit_exact_cann` — non-fused masked logits, bit-exact against `torch_npu.npu_top_k_top_p`, over `B x V` in `{(4, 1000), (8, 4096), (4, 4097), (8, 32000)}` (covering the single-pass/tiled split and the `V = BLOCK + 1` tail) and dtypes `{fp32, bf16}`.
- `test_fused_probs_vs_cann_softmax` — fused probabilities against `torch_npu.npu_top_k_top_p(...).softmax(dim=-1, dtype=torch.float32)`, including the Qwen2 vocabulary `V = 151936` (multi-tile path) and `V = 32000`.
- `test_scalar_and_none_variants` — Python scalar / `None` / single-element-tensor forms of `k` and `p`, and the disabled conventions (`k >= V`, `p = 1.0`).
- `test_sampler_apply_path_*` / `test_sampler_stock_still_works` — the vLLM sampler with `enable_triton_apply_topk_topp` on (top-k only, processed-logits equivalence with the stock path, mixed per-request `k`/`p` sampling) and off (stock CANN path unchanged), with the switch driven through `additional_config`.

The masked-logits path is compared bit-exactly (pure filtering/scatter, no arithmetic reordering); the fused-probability path is compared against the CANN operator followed by an fp32 softmax.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_apply_top_k_top_p.py
```
