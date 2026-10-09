# dsa_local_metadata

## Description

- **Location**: `vllm_ascend/ops/triton/dsa_local_metadata.py` — `build_local_metadata_kernel`, 1:1 host wrapper `build_local_metadata`.
- **Function**: Builds the DSA context-parallel (TP/SP) per-rank local token metadata for one scheduling step. For each request, the global token interval `[query_start_loc[i], query_start_loc[i+1])` is clipped to this rank's local token slice `[local_start, local_end)`; the three outputs are the per-rank cumulative query offsets, the per-rank sequence lengths of the requests owning local tokens, and (optionally) the per-request start-position offsets used by spec-decode. It replaces the host-side `clamp` + `cumsum` + mask chain in `dsa_cp.py::_build_local_token_metadata` with a single fused Triton kernel.
- **Formula** (per request `i`, all int32 semantics):
    - `lqs[i] = clamp(query_start_loc[i], local_start, local_end)`, `lqe[i] = clamp(query_start_loc[i+1], local_start, local_end)`, `lql[i] = lqe[i] - lqs[i]`
    - `local_query_start_loc[0] = 0`, `local_query_start_loc[1+i] = Σ_{j<=i} lql[j]` (inclusive cumsum)
    - `offset[i] = query_start_loc[i+1] - lqe[i]`; `local_seq_lens[i] = (lql[i] > 0 and seq_lens[i] > 0) ? max(seq_lens[i] - offset[i], 0) : 0`
    - `start_pos_out[i] = seq_lens[i] - (query_start_loc[i+1] - query_start_loc[i])` (only when `start_pos_out` is provided)
- **Algorithm flow** (single fixed-capacity launch, one vector pass):
  1. Launch shape is compile-invariant: grid is always `(1,)` and `BLOCK` always equals the caller's buffer capacity (`scheduler_config.max_num_seqs` in production, passed as `block=local_query_start_loc.numel() - 1`). Lanes beyond `num_reqs` are disabled by a runtime mask, so no scheduling step can re-JIT or re-specialize the kernel.
  2. Load `query_start_loc[i]`, `query_start_loc[i+1]`, `seq_lens[i]` per lane (masked lanes read 0). The clamp/compare chain runs in fp32 because int32 `tl.minimum`/`tl.maximum`/compare lower to scalar loops on Ascend; token offsets are `< 2^24` and roundtrip losslessly through the fp32 mantissa. `sp` stays in the int32 domain as a pure integer expression.
  3. The 1-D cumsum is folded into a column-major `(SUB_N=8, COLS=BLOCK/8)` view so `tl.cumsum` runs on `axis=0` (the vector unit path) instead of the last dimension (which degrades to a `BLOCK`-step scalar loop on 910B); each column is then compensated with the prefix of the preceding columns' totals to recover the global inclusive scan.
  4. Store all three outputs over the full capacity: every lane writes (active lanes write computed values, tail lanes write 0), so the output buffers are deterministic on return and callers may skip their `fill_(0)` pre-zeroing.
- **Supported modes**: Atlas A2 (Ascend 910B4 measured), Atlas A3. Used by the DSA-CP metadata builder in `vllm_ascend/attention/context_parallel/dsa_cp.py`; works in both eager and graph-capture modes (the launch shape does not depend on runtime batch size, so it is cudagraph-safe).

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `query_start_loc` | Input | Inclusive token-prefix of the batch, `[num_reqs + 1]` | int32 | ND, 1D contiguous |
| `seq_lens` | Input | Global sequence length per request, `[num_reqs]` | int32 | ND, 1D contiguous |
| `local_query_start_loc` | Output | Per-rank cumulative query offsets, `[block + 1]`; element 0 is always 0 | int32 | ND, 1D |
| `local_seq_lens` | Output | Per-rank local sequence length per request, `[block]` | int32 | ND, 1D |
| `local_start` | Input | First token index of this rank's local slice (`rank_in_group * tokens_per_rank`) | int32 | runtime scalar |
| `local_end` | Input | One-past-last token index of this rank's local slice | int32 | runtime scalar |
| `num_reqs` | Input | Current batch size, `0 <= num_reqs <= block` | int32 | runtime scalar |
| `start_pos_out` | Output | Optional per-request start-position offsets `[block]`; pass `None` to skip (a 1-element dummy is substituted internally) | int32 | ND, 1D |
| `block` | Attribute | Fixed capacity of the output buffers (= `scheduler_config.max_num_seqs` in production); must be a multiple of 8 | int | compile-time |

## Constraints

- `local_query_start_loc` must be sized `[block + 1]` and `local_seq_lens` / `start_pos_out` sized `[block]`, where `block` is the same capacity on every call for a given builder; `block % 8 == 0` (asserted) because the cumsum fold requires the capacity to be a multiple of `SUB_N=8`.
- `num_reqs` must not exceed `block`; requests with `i >= num_reqs` are masked and their outputs are 0.
- All outputs are fully overwritten over the capacity on every launch — the caller must NOT pre-zero the buffers on this path (pre-zeroing is only needed by the CPU fallback).
- Token offsets must satisfy `< 2^24` for the fp32 clamp chain to be exact (guaranteed: `max_num_batched_tokens` per rank is far below this bound).
- `local_start`, `local_end`, `num_reqs` are declared `do_not_specialize`; the two `COMPUTE_START_POS` variants produce exactly two cached binaries for the lifetime of the process.
- `num_reqs == 0` is short-circuited on host: outputs are zero-filled without a kernel launch. Degenerate zero-size input tensors (NULL `data_ptr` on NPU) are substituted with a live 1-element dummy before launch.
- Only for inference (decoding/prefill) on NPU; CPU-fallback input goes through the torch path in `dsa_cp.py` instead.

## Origin and Differences

- **Origin**: Modified from the `build_local_metadata_triton` Triton kernel in `vllm_ascend/ops/triton/dsa_cp.py` (which replaces the eager clamp/cumsum chain in `dsa_cp.py::_build_local_token_metadata`). Same math, new launch contract.
- **Differences**:
    - NPU adaptation for performance: (1) fixed-capacity launch (`grid=(1,)`, `BLOCK` = buffer capacity) plus `do_not_specialize` on the three step-varying scalars removes the per-step re-JIT of the previous `next_power_of_2(num_reqs)` blocking (PR #15637 measured ~386us per recompilation); (2) the clamp/compare chain runs in fp32 to ride the vector SIMD units (int32 min/max/compare lower to scalar loops on Ascend; measured scalar-pipe share 0.969 → 0.647 and device time 33.7us → 6.2us on 910B4 at `block=1024`); (3) the 1-D cumsum is folded into a column-major `(8, BLOCK/8)` view so `tl.cumsum` runs on `axis=0` on the vector units instead of degrading to a `BLOCK`-step scalar loop; (4) full-capacity overwrite stores let callers drop the two `fill_(0)` pre-zeroing launches.
    - Modified for a specific vllm-ascend logic or different input parameters: the capacity `block` is derived from the caller's buffer (`local_query_start_loc.numel() - 1`) instead of being derived from the runtime batch size, and `start_pos_out=None` is an explicit opt-out for the spec-decode start-position output.

## Test Cases

The test covers both real inference shapes (DSA-CP decode/prefill, TP sizes 1 and 8, ranks first/middle/last, `MAX_NUM_SEQS=1024` capacity, `NUM_REQS_LIST = [1, 7, 32, 1024]`) and edge cases: graph-padding masks, no recompilation across request-count sweeps, empty batch, stale-tail overwrite on buffer reuse, and default-block/capacity consistency. As this is a pure integer metadata operator, the unified precision tolerance is bit-exact (`rtol=0, atol=0`).

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_build_local_metadata_triton.py
```
