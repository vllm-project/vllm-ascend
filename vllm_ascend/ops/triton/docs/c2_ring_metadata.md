# c2_ring_metadata

## Description

- **Function**: Publishes every control the C2 (compress-ratio-2) compressor
  needs for one decode step in a single Triton launch: the five ring-metadata
  rows, the per-token completion mask, the per-token source position, and the
  RoPE rows gathered for that source position. It replaces the eager
  `clamp`/`where`/`copy_`/`gather` chain in
  `AscendDSAV41StateMetadataBuilder._build_compressor_cache`
  (`vllm_ascend/attention/dsa_v41.py`), replacing multiple eager operations
  and RoPE gathers with one launch.
- **Formula**: With `starts = query_start_loc[req]`,
  `ends = query_start_loc[req + 1]`, `qlen = ends - starts`,
  `used = max(min(ends, num_actual_tokens) - starts, 0)`:

    | Ring row | Value |
    | --- | --- |
    | 0 | `max(seq_lens - qlen, 0)` — position where this step's ring write starts |
    | 1 | `used` — tokens written this step; `0` for a dead request or `skip_update` |
    | 2 | `starts` |
    | 3 | `starts` |
    | 4 | `block_table[req, 0]` when `used > 0`, otherwise `0` |

    Per token, with `valid_end = min(query_start_loc[num_actual_reqs], num_actual_tokens)`:

    - `complete = token < valid_end and positions[token] % 2 == 1` (and
      `skip_update == 0`)
    - `source_positions = positions - 1` when complete, otherwise `0`
    - `cos_out[token] = source_cos[source_positions]`, `sin_out[token] = source_sin[source_positions]`

- **Algorithm flow** (one grid, both axes in the same program):
  1. Grid: `max(cdiv(num_reqs, REQ_BLOCK), cdiv(num_tokens, TOKEN_BLOCK))` with
     `REQ_BLOCK = 64` and `TOKEN_BLOCK = 16`. Each program owns one request tile
     and one token tile; tiles past the real counts are masked out.
  2. Request tile: load `query_start_loc` starts/ends and `seq_lens`, derive
     `qlen` and `used`, zero `used` for requests at or past
     `num_actual_reqs` and for `skip_update`, then store the five ring rows with
     their own row stride.
  3. Token tile: load `positions`, compute `valid_end` once from
     `query_start_loc[num_actual_reqs]`, build `complete` as two low-bit /
     compare tests, and store the mask and the source position.
  4. RoPE tile: `[TOKEN_BLOCK, ROPE_BLOCK]` gather from `source_cos` /
     `source_sin` using `source_positions` as the row index and the explicit
     source row/dim strides, then store into the `[T, 1, 1, ROPE_DIM]` outputs.
     Incomplete rows therefore gather row `0` instead of being zero-filled,
     matching the eager `torch.gather` with index `0`.
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT Products. The kernel
  is selected by the presence of the full-source RoPE buffers, not by the A5
  packed-cache backend, so A2/A3 reach the same code path. Works in eager and
  ACL graph modes.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `query_start_loc` | Input | Per-request prefix offsets, at least `num_reqs + 1` entries | int32 | ND |
| `seq_lens` | Input | Sequence length per request, including the tokens of this step | int32 | ND |
| `positions` | Input | Token positions `[num_tokens]`; the low bit decides pair completion | int32 / int64 | ND |
| `block_table` | Input | Block table `[num_reqs, ...]`; only column 0 is read | int32 | ND |
| `source_cos` / `source_sin` | Input | Full-source RoPE tables `[num_rows, rope_dim]` indexed by position | fp16 / bf16 / fp32 | ND |
| `num_reqs` | Input (attribute) | Requests in the carrier, including dead ones | int32 | scalar |
| `num_tokens` | Input (attribute) | Token carrier length, including graph padding | int32 | scalar |
| `num_actual_reqs` | Input (attribute) | Number of live requests | int32 | scalar |
| `num_actual_tokens` | Input (attribute) | Number of live tokens | int32 | scalar |
| `skip_update` | Input (attribute) | When true, ring work and completion are cleared, but ring row 0 is still published | bool | scalar |
| `ring_metadata_output` | Output | Ring controls `[5, num_reqs]` | int32 | ND |
| `complete_mask_output` | Output | Completion mask `[num_tokens]` | bool | ND |
| `source_positions_output` | Output | Source positions `[num_tokens]`; `0` when incomplete | int32 / int64 | ND |
| `cos_output` / `sin_output` | Output | Gathered RoPE rows `[num_tokens, 1, 1, rope_dim]` | same as source | ND |

The launch returns `None`; all five outputs are caller-owned buffers.

## Constraints

- `ring_metadata_output.shape == (5, num_reqs)`,
  `complete_mask_output.shape == (num_tokens,)`,
  `source_positions_output.shape == (num_tokens,)` and
  `cos_output.shape == sin_output.shape == (num_tokens, 1, 1, rope_dim)`; any
  other shape raises `ValueError` before launching.
- `source_cos` and `source_sin` must have the same last-dimension width.
- `num_reqs == 0 and num_tokens == 0` returns without launching.
- `query_start_loc` must contain at least `num_reqs + 1` entries and be
  non-decreasing. Entries past `num_actual_reqs` may be stale; they are masked.
- `source_positions` indexes rows of `source_cos`/`source_sin`, so both tables
  must cover `max(positions) - 1`. Element strides of the source tables are read
  on the host, so `[num_rows, rope_dim]` and `[num_rows, 1, 1, rope_dim]` are
  both accepted.
- Incomplete rows gather row `0`. A caller that needs those rows to read as zero
  must zero the source row `0` itself; the kernel does not special-case them.
- Ring row 0 depends only on `seq_lens` and `query_start_loc`, so it stays valid
  under `skip_update`; rows 1 and 4 are the ones cleared.

## Origin and Differences

- **Origin**: Developed from scratch for vllm-ascend. The reference semantics are
  the eager branch of `_build_compressor_cache` in
  `vllm_ascend/attention/dsa_v41.py`; there is no upstream vLLM operator with
  this interface.
- **Differences**:
    - NPU adaptation for performance: eager metadata operations and RoPE gathers
      collapse into one launch, with request and token tiles sharing a grid.
    - Modified for a specific vllm-ascend logic: gating moved from
      `_uses_a5_packed_cache` to the presence of the full-source RoPE buffers,
      so A2/A3 also run this kernel. The eager chain remains for callers without
      those buffers.
    - The five outputs are caller-owned persistent buffers, so graph replay keeps
      stable addresses.

## Test Cases

- **Reference implementation**: the eager chain of `_build_compressor_cache` in
  `vllm_ascend/attention/dsa_v41.py`.
- **Accuracy criterion**: exact comparison (`rtol=0, atol=0`). Ring rows and
  source positions are integer controls, the mask is boolean, and the RoPE rows
  are gathered copies, so the general floating-point tolerance table does not
  apply.
- **Coverage**: all five outputs are compared for real inference shapes of
  DeepSeek V4.1 on Ascend — 6 requests / 6 tokens (decode), 6 requests / 2048
  tokens (prefill), a 64-token FULL-graph carrier with 37 live tokens, the
  production `[rows, 1, 1, rope_dim]` RoPE shape with `qk_rope_head_dim` 64, a
  bf16 source table written into fp32 output buffers — plus `skip_update` on and
  off, empty carriers, dead requests, multiple grid programs (`num_reqs` above
  and below 64, `num_tokens` above and below 16), a non power-of-two RoPE width,
  an fp32 source table and ACL graph replay.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_c2_ring_metadata.py
```

## Example

```python
from vllm_ascend.ops.triton.c2_ring_metadata import build_c2_ring_metadata

build_c2_ring_metadata(
    common.query_start_loc,
    req.seq_lens,
    req.positions,
    common.block_table_tensor,
    full_source_cos,
    full_source_sin,
    req.num_reqs,
    req.num_input_tokens,
    req.num_actual_reqs,
    req.num_actual_tokens,
    skip_update=skip_ring_state_update,
    ring_metadata_output=ring_meta,                                   # [5, num_reqs]
    complete_mask_output=self._c2_complete_mask[:req.num_input_tokens],
    source_positions_output=self._c2_source_positions[:req.num_input_tokens],
    cos_output=self._c2_source_cos[:req.num_input_tokens],            # [T,1,1,rope_dim]
    sin_output=self._c2_source_sin[:req.num_input_tokens],
)
```
