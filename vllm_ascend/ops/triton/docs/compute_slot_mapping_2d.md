# compute_slot_mapping_2d

## Description

- **Function**: Fuses the per-KV-cache-group DSA V4.1 slot-mapping computation into one Triton kernel. It replaces the eager PyTorch chain in `dsa_v41.py` (optional compressed-slot conversion followed by GeScalar / ClampMin / DivMod / Remainder / SWhere / ViewCopy) that previously emitted roughly ten eager operators for every non-ratio-2 cache group.
- **Formula**: Pure data movement plus integer division/remainder. For each token `i` with input slot `s = active_slots[i]`:
    - Optional compression (`ratio > 1`): `valid_comp = (s >= 0) && ((s + 1) % ratio == 0)`, `s' = valid_comp ? s // ratio : -1`
    - `valid = s' >= 0`, `physical = max(s', 0)`
    - `slot2d[i, 0] = valid ? physical // storage_block_size : -1`
    - `slot2d[i, 1] = valid ? physical %  storage_block_size : -1`
    - `flat[i]        = valid ? s' : -1`
- **Algorithm flow** (each token processed independently):
  1. Launch a 1D grid with `BLOCK = 1024`; program `pid` handles offsets `pid * BLOCK + [0, BLOCK)`.
  2. `tl.load` the slot from `active_slots` (int32 or int64) with `other=-1`.
  3. If `COMPRESSED`, map only complete compression groups to `slot // ratio`, everything else to `-1`.
  4. Compute page index and in-page offset with `max(slot, 0)` and store into the row-major `slot2d` (`offs * 2`) and the flat buffer, masking `offs < num_tokens`.
- **Supported modes**: Atlas A2 and Atlas A3. This kernel is used by the non-packed DSA V4.1 attention-metadata path (`AscendDSAV41MetadataBuilder`); the A5 packed-cache path uses a different backend. It runs in both eager and graph-capture modes.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `active_slots` | Input | Physical cache slot per logical token, shape `[num_tokens]`, contiguous | int32 / int64 | ND |
| `storage_block_size` | Input (attribute) | Number of token rows represented by one physical storage block | int32 scalar | scalar |
| `ratio` | Input (attribute) | Compression ratio; `1` means no compression, `> 1` writes only completed groups | int32 scalar | scalar |
| `slot2d` | Output | 2D slot mapping `[num_tokens, 2]` (page index, in-page offset), written in place, contiguous | int32 | ND |
| `flat` | Output | Flat slot mapping `[num_tokens]`, written in place, contiguous | int64 | ND |

## Constraints

- `active_slots` and `flat` are accessed with unit stride and must be contiguous; the wrapper asserts this. `slot2d` is flattened with `view(-1)`, which already rejects non-contiguous input.
- `num_tokens = active_slots.shape[0]`; `num_tokens == 0` returns without launching. `num_tokens` is not specialized, so token counts that change every step do not trigger recompilation.
- `ratio >= 1`. For `ratio > 1` only slots that close a complete group (`(slot + 1) % ratio == 0`) are written; incomplete groups become `-1`.
- The kernel never reads back or mutates its inputs; `slot2d` and `flat` are preallocated persistent buffers reused across steps and graph replays.
- `COMPRESSED` and `BLOCK` are `constexpr`. For the DeepSeek V4.1 `ratio in {1, 2}` configuration this yields at most two specializations (`COMPRESSED` false/true), independent of token count and block size.
- `storage_block_size` and `ratio` are runtime arguments (not `constexpr`), so different cache geometries do not recompile.

## Origin and Differences

- **Origin**: Developed from scratch; extracted from the eager slot-mapping chain in `vllm_ascend/attention/dsa_v41.py`. There is no corresponding upstream vLLM operator.
- **Differences**:
    - NPU adaptation for performance: fuses the compression, clamping, div/rem and two masked copies into a single launch, removing host-side launch overhead per KV cache group.
    - Modified for a specific vllm-ascend logic: the compressed `ratio == 2` path has additional `valid_end` / `positions` / ring-state masking and stays in the eager implementation, so this kernel handles the non-compressed groups and the plain compressed groups only.

## Test Cases

Bit-exact (`rtol=0, atol=0`) comparison against an eager PyTorch reference, as this is a pure data-movement / integer arithmetic operator. Coverage: int32 and int64 slots, `ratio` in `{1, 2}`, token counts `0, 1, 1023, 1024, 1025` (including the `BLOCK`-masked tail), negative/padded slots, values around storage-block and compression-group boundaries, repeated writes into the same output buffers (stale valid values must be overwritten with `-1`), and a compilation-reuse check that varying token counts does not recompile.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping_2d.py
```
