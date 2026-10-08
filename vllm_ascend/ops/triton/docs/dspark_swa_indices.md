# dspark_swa_indices

## Description

- **Function**: Builds the non-causal visible-slot indices and visible lengths for DSpark (DeepSeek V4.1) parallel drafting over a paged sliding-window attention (SWA) cache. It replaces the eager ~15-op torch chain (`arange` / `sub` / `clamp` / `floordiv` / `gather` / `where` / `repeat_interleave` / `copy_` ...) in `build_dspark_swa_indices` of `vllm_ascend/attention/dsa_v1.py` with a single fused Triton kernel.
- **Formula** (per request `r`, all integer arithmetic):
    - `q_len = query_start_loc[r + 1] - query_start_loc[r]` (uniform across requests)
    - `prefix_len = seq_lens[r] - q_len`
    - `start_pos = max(prefix_len - window_size, 0)`
    - `visible_len = seq_lens[r] - start_pos`
    - for column `c` in `[0, index_width)`: `pos = start_pos + c`; `block_num = pos // block_size`; `block_off = pos - block_num * block_size`
    - `slot[c] = block_table[r, block_num] * block_size + block_off` if `c < visible_len`, else `-1`
    - every active output row of request `r` receives the same `slot` vector; `lens[row] = visible_len`; padded rows `[num_rows, num_rows_padded)` are reset to `slots = -1`, `lens = 0`
- **Algorithm flow** (processed per request-column-block item, independently):
  1. Compute grid: `grid_size = min(num_aiv_cores, max_num_reqs * NUM_CB)` where `NUM_CB = ceil(index_width / BLOCK_W)`; the kernel claims work items `w = pid, pid + grid_size, ...` grid-stride, with item `w` mapping to request `r = w // NUM_CB`, column block `cb = w % NUM_CB`.
  2. Distributed pad-row cleanup: each item resets the slice of `[num_rows, num_rows_padded)` inside its row band `[r * num_query_per_req, (r + 1) * num_query_per_req)`, so a captured ACL graph never replays stale rows.
  3. For active items: load the request's full block-table row (up to `ROW_POW2 = next_pow2(num_blocks)` lanes) into UB, clamp column block numbers into the table range, gather physical block IDs via `tl.gather`, and compute the slot vector for the column block.
  4. One 2D store writes the `[q_len, index_width]` output tile: the slot vector is broadcast over `Q_POW2 = next_pow2(num_speculative_tokens + 1)` rows (the uniform-query contract bounds `q_len <= Q_POW2`); the first column block also stores `visible_len` into `lens`.
- **Supported modes**: Atlas A2 (910B series; verified on 910B4). A3 and 950PR&950DT Products N/A — the host gate `dspark_swa_indices_supported` admits A2 only, and the eager torch chain remains the fallback. Used by the DSA metadata builder for DSpark non-causal parallel drafting in both eager and graph-capture modes.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `block_table` | Input | Block table of the SWA KV cache, row-sliced to the active requests `[num_reqs, num_blocks]`; row stride passed separately so a padded/sliced table works | int32 / int64 | ND |
| `query_start_loc` | Input | Cumulative query offsets `[num_reqs + 1]`; under the uniform-query contract each request contributes the same `q_len` | int32 / int64 | 1D |
| `seq_lens` | Input | Per-request context lengths `[num_reqs]` | int32 / int64 | 1D |
| `num_speculative_tokens` | Input (attribute) | Number of draft tokens per request; output row-tile height is `next_pow2(num_speculative_tokens + 1)` | int32 | scalar |
| `window_size` | Input (attribute) | Sliding-window size of the SWA cache (model constant) | int32 | scalar |
| `block_size` | Input (attribute) | DSA KV-cache block size; must be a power of two | int32 | scalar |
| `index_width` | Input (attribute) | Aligned index width of the output (`ceil((window_size + num_speculative_tokens) / 128) * 128`) | int32 | scalar |
| `num_decode_tokens` | Input (attribute) | Total active output rows (`sum(query_lens)`), supplied by the caller to avoid a D2H sync | int32 | scalar |
| `max_num_reqs` | Input (attribute) | Capture-time request-slot capacity sizing the fixed grid; optional (defaults to `num_reqs`) | int32 | scalar |
| `num_query_per_req` | Input (attribute) | Row-expansion factor per request slot; defaults to `num_speculative_tokens + 1`, anchor-sampling mode passes `num_speculative_tokens` | int32 | scalar |
| `indices_output` | Output | Visible physical slot ids `[num_rows_padded, 1, index_width]`, `-1` for invisible/padded lanes; pre-allocated persistent buffer for graph capture, or freshly allocated when omitted | int32 | ND |
| `lens_output` | Output | Visible length per row `[num_rows_padded]`, `0` on padded rows; pre-allocated persistent buffer or freshly allocated when omitted | int64 | 1D |

## Constraints

- `block_size` must be a power of two (`//` and `%` lower to shift/mask).
- `next_pow2(num_blocks)` must not exceed 8192 (UB-preload tile cap; the host gate falls back to the eager chain beyond it).
- `q_len <= num_speculative_tokens + 1` for every request (uniform-query contract of DSpark drafting); the 2D output tile relies on it.
- `indices_output` must be int32. When a persistent `indices_output` is passed for graph capture, the FULL buffer (`buffer`, not `buffer[:num_rows]`) must be provided so the pad-row cleanup stays armed; an active-sized slice disables the cleanup and triggers a `UserWarning`.
- Padded rows `[num_rows, num_rows_padded)` are reset to `slots = -1`, `lens = 0` in place — a strict superset of the eager behavior, which leaves those rows stale.
- All runtime scalars (`num_blocks`, `num_reqs`, `num_rows_padded`, `num_rows`, `num_query_per_req`, `num_slots`) are `do_not_specialize`, so no recompilation is triggered by varying batch sizes; only the constexpr set (`WINDOW_SIZE`, `BLOCK_SIZE`, `INDEX_W`, `BLOCK_W`, `ROW_POW2`, `Q_POW2`) forms the JIT key, warmed once per `(num_blocks, index_width)` combination.
- Inference-only (decode, DSpark non-causal drafting); works under both eager and ACL graph-capture modes (fixed launch grid derived from `max_num_reqs`, stable tensor addresses from persistent buffers).

## Origin and Differences

- **Origin**: Rewritten from scratch based on the eager torch implementation `build_dspark_swa_indices` in `vllm_ascend/attention/dsa_v1.py` (DSpark non-causal SWA index build), fused into a single Triton-Ascend kernel.
- **Differences**:
    - NPU adaptation for performance: fuses the ~15-op eager chain (device time 213-8046 us across 9 shapes) into one kernel launch (4.7-90.5 us, 45-89x per shape, ~70x geometric mean on 910B4); grid-stride claiming over a fixed capacity grid keeps the launch shape stable for ACL-graph capture; distributed pad-row cleanup parallelizes the graph-padding reset that the eager chain performs serially; fp32 roundtrip for `tl.gather` and window arithmetic rides the vector unit (exact below 2^24, far above any real slot id).
    - Modified for a specific vllm-ascend logic or different input parameters: adds the capacity-grid parameters (`max_num_reqs`, `num_query_per_req`, `num_slots`) and in-place `indices_output` / `lens_output` buffers required by the device-metadata graph-capture contract of the DSA metadata builder; pad rows are explicitly reset so captured graphs never replay stale rows.

## Test Cases

Accuracy tests use the real inference shapes of DeepSeek V4.1 DSpark drafting (`window_size=944`, `block_size=128`, `index_width=1088`, the DFlash uniform-query contract `num_query_per_req = num_speculative_tokens + 1`, graph-capture pad-row cleanup, the wide-window and narrow-block-table boundary cases) plus a broader generic parameter space (varying request counts, block sizes 4-128, long contexts, 8192-wide block tables). As this is an integer index-building operator, the unified precision tolerance is bit-exact (`rtol=0, atol=0`).

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dspark_swa_indices.py
```
