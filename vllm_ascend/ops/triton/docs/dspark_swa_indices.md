# dspark_swa_indices

## Description

- **Function**: Unified Triton-Ascend implementation of DSpark (DeepSeek V4 / V4.1) non-causal SWA index and length construction (`build_dspark_swa_indices_triton` and shared entry point `vllm_ascend.attention.dsa_v1.build_dspark_swa_indices`), supporting both **physical KV slots** (`output_mode="physical"`) and **request-local logical positions** (`output_mode="logical"`).
- **Formula** (per request `r`, integer arithmetic):
    - `q_len = query_start_loc[r + 1] - query_start_loc[r]`, `prefix_len = seq_lens[r] - q_len`
    - `start_pos = max(prefix_len - window_size, 0)`, `visible_len = seq_lens[r] - start_pos`
    - for `c` in `[0, index_width)`: `pos = start_pos + c`; `"logical"` sets `slot[c] = pos`, `"physical"` sets `slot[c] = block_table[r, pos // block_size] * block_size + (pos % block_size)` when `c < visible_len`, else `-1`
    - active rows `[q0, q0 + q_len)` receive `slot` and `lens[row] = visible_len`; padded rows `[active_rows, num_rows_padded)` are reset to `slots = -1`, `lens = 0` in the same launch
- **Algorithm flow** (grid-stride over request-column-block items `r = w // NUM_CB`, `cb = w % NUM_CB`):
  1. Launch `grid_size = min(get_vectorcore_num(), num_slots * NUM_CB)` (`NUM_CB = ceil(index_width / BLOCK_W)`, `num_slots = max(R_alloc, ceil(num_rows_padded / num_query_per_req))`).
  2. Load `active_rows = query_start_loc[num_reqs]` on device and reset any padded slice in `[r * num_query_per_req, (r + 1) * num_query_per_req)` so ACL graph replays never read stale rows.
  3. For active items (`r < num_reqs`, `q_len > 0`): `"logical"` uses `pos`; `"physical"` loads only the bounded block-table slice `[blk_base, blk_base + BT_WIN_POW2)` (`BT_WIN_POW2 = max(2, next_pow2((BLOCK_W + BLOCK_SIZE - 2) // BLOCK_SIZE + 1))`) and gathers block IDs via `tl.gather`.
  4. Write `[q_len, index_width]` rows in `Q_POW2 = min(16, next_pow2(num_speculative_tokens + 1))` tiles and store `visible_len` (`cb == 0`).
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT Products for `"physical"` and `"logical"` modes under eager and ACL graph capture/replay (verified on A2 eager/ACLGraph/TP8 serving and A3 eager; A3 ACLGraph and 950PR&950DT unverified directly). Triton is required on NPU; host/CPU tensors are rejected.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `block_table` | Input | SWA KV block table `[num_reqs, num_blocks]` (`"physical"` only; `None` in `"logical"`), row pitch via `stride(0)` | int32 / int64 | ND |
| `query_start_loc` | Input | Cumulative query offsets `[num_reqs + 1]` | int32 / int64 | 1D |
| `seq_lens` | Input | Per-request context lengths `[num_reqs]` | int32 / int64 | 1D |
| `num_speculative_tokens` | Input (attribute) | Non-negative draft tokens per request | int32 | scalar |
| `window_size` | Input (attribute) | Non-negative SWA sliding-window size | int32 | scalar |
| `block_size` | Input (attribute) | Positive KV block size in `"physical"` mode (non-pow2 supported; normalized to `1` in `"logical"`) | int32 | scalar |
| `index_width` | Input (attribute) | Positive aligned output width (`ceil((window_size + num_speculative_tokens) / 128) * 128` by default) | int32 | scalar |
| `num_decode_tokens` | Input (attribute) | Non-negative fixed output row count; active count read from `query_start_loc[num_reqs]` on device | int32 | scalar |
| `max_num_reqs` | Input (attribute) | Optional non-negative capture request-slot capacity | int32 | scalar |
| `num_query_per_req` | Input (attribute) | Positive row-expansion factor per request slot (defaults to `num_speculative_tokens + 1`) | int32 | scalar |
| `block_w` | Input (attribute) | Positive column-block tiling width (defaults to `1024`) | int32 | scalar |
| `output_mode` | Input (attribute) | `"physical"` (KV slot IDs) or `"logical"` (request-local positions) | str | scalar |
| `indices_output` | Output | Contiguous `[>=num_rows, 1, index_width]` buffer (`-1` on invisible/padded lanes) | int32 | ND |
| `lens_output` | Output | Contiguous `[>=num_rows]` or `[>=num_rows, 1]` buffer (`0` on padded rows; defaults `[alloc_rows]` int64 physical, `[alloc_rows, 1]` int32 logical) | int32 / int64 | 1D / 2D |

## Constraints

- Inputs must reside on the same accelerator (`npu`/`cuda`), with `int32`/`int64` metadata (`block_table` `stride(1) == 1`, no 8192-column cap) and valid scalar bounds (`window_size`/`num_speculative_tokens`/`num_decode_tokens`/`max_num_reqs >= 0`, `index_width`/`block_w`/`num_query_per_req > 0`, `block_size > 0` in `"physical"` mode).
- Under graph capture (`max_num_reqs` set), pass full persistent `indices_output`/`lens_output` buffers rather than `[:num_rows]` slices so `[active_rows, num_rows_padded)` is cleared.
- `get_vectorcore_num()` is initialized idempotently via `init_device_properties_triton()` before launch.

## Warmup Scope and Configuration

- Runtime calls invoke `_DSPARK_SWA_INDICES_KERNEL` (`DsparkSwaIndicesKernel`, compile key `(window_size, block_size, index_w, block_w, bt_win_pow2, q_pow2, bt_stride, use_logical_indices, lens_dtype)`), with runtime shape scalars in `do_not_specialize`.
- `kernel_warmup.py:_register_dspark_swa_indices_warmup` registers `AscendDSAMetadataBuilder` (`"physical"`) and `AscendDSAV41MetadataBuilder` (`"logical"`, `block_size=1`, `bt_stride=1`).

## Origin and Differences

- **Origin**: Consolidated from the prior physical-slot kernel, retired `spec_decode/dspark_swa_indices.py`, and the production PyTorch fallback in `dsa_v1.py`.
- **Differences**: Adds `output_mode`, bounded `BT_WIN_POW2` block-table loading (lifting `<= 8192` width and pow2 `block_size` caps), `Q_POW2` tiling for arbitrary `q_len`, and upfront metadata dtype/scalar validation.

## Test Cases

- Bit-exact (`rtol=0, atol=0`) against `eager_dspark_swa_indices` across `"physical"`/`"logical"`, non-pow2 `block_size`, wide `block_table` (`> 8192`), capacity/padding boundaries, reused buffers, ACLGraph replay, invalid metadata/scalar guards, and warmup registration.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dspark_swa_indices.py
pytest -sv tests/ut/ops/test_dspark_swa_indices.py tests/ut/attention/test_dsa_v1.py
pytest -sv tests/ut/model_executor/warmup/test_jit_warmup_dspark.py tests/ut/model_executor/warmup/test_kernel_warmup.py
```
