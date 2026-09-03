# _zero_kv_blocks_kernel

## Description

- **Function**: Zeros selected KV-cache pages across all independently allocated K/V segments in one launch.
- **Formula**: For each requested block `b` and segment `s`, `segment_s[b * PAGE_SIZE_EL : (b + 1) * PAGE_SIZE_EL] = 0`.
- **Algorithm flow** (processed work item by work item, independently): flatten `(block, segment, chunk)` work, recover its three indices, cast the stored absolute segment address to an int32 pointer, and write one zero tile.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950 standard workers. The 310P worker uses a non-Triton implementation. Graph mode is N/A because newly allocated blocks are cleared before the captured model forward.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `seg_addrs_ptr` | Input | Absolute byte address of each cache segment | uint64 | ND |
| `block_ids_ptr` | Input | Logical block IDs to clear | int64 | ND |
| `n_blocks` | Attribute | Number of active block IDs | int | scalar |
| `N_SEGS` | Attribute | Compile-time segment count | int | scalar |
| `PAGE_SIZE_EL` | Attribute | int32 elements per logical page | int | scalar |
| `BLOCK_SIZE` | Attribute | int32 elements written by one work item | int | scalar |
| `GRID_SIZE` | Attribute | Number of launched vector-core programs | int | scalar |

## Constraints

- Segment addresses must point to writable NPU allocations and block IDs must be valid for every segment.
- `PAGE_SIZE_EL` must be uniform across segments and divisible by `BLOCK_SIZE`; cache byte strides must be divisible by four.
- Metadata excludes non-full-attention and runner-only layers. The launcher is a no-op for empty block lists.
- The current Ascend implementation handles only `FullAttentionSpec`, expects separate K/V tensors, fixes the cache block dimension to 0, and requires the logical block size to be an integer multiple of the kernel block size.

## Origin and Differences

- **Origin**: Adapted from vLLM's `vllm/v1/worker/utils.py` `KVBlockZeroer` contract.
- **Differences**:
    - NPU adaptation for performance: flattens block, segment, and chunk work into a vector-core-sized 1-D grid-stride launch and writes cache storage as aligned int32 zero words.
    - Modified for vllm-ascend logic: clears dirty full-attention pages that can otherwise retain NaNs when hybrid Mamba/full-attention blocks are reassigned during multi-token speculative decoding. It uses a single uniform page size for separate K/V allocations, unlike current upstream's per-segment stride/page metadata.

## Test Cases

N/A: coverage currently runs through KV-cache integration tests; no dedicated Triton single-operator test exists.
