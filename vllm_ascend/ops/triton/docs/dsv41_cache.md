# DeepSeek V4.1 cache preparation

## Purpose and interfaces

These kernels prepare indices and packed cache records for the Ascend attention
operators. Logical token positions, physical cache slots and compressed row
indices are different coordinate systems; callers must not interchange them.

| Source / entry point | Inputs and outputs | Operation |
| --- | --- | --- |
| `packed_cache_slot_mapping.py`: `build_packed_cache_slot_mapping` | INT64 physical slots and positions, INT32 query boundaries; coordinate `[T,2]` and flat `[T]` output buffers | Convert physical slots to page/row coordinates. Ratio 2 writes only completed pairs. Mask negative slots, graph padding and skipped state updates to `-1`. |
| `build_window_indices.py`: `build_window_indices_triton` | Positions `[T]`; INT32 indices `[T,1,W]` and lengths `[T,1]` | Enumerate visible positions within each query's sliding window; unused entries are `-1`. Optional output buffers retain their addresses for graph replay. |
| `c2_ring_metadata.py`: `build_c2_ring_metadata` | Request boundaries, positions, block tables and ring controls | Derive compressor state addresses and request controls on device, avoiding per-request host synchronization. |
| `quantize_mxfp4_indexer.py`: `quantize_mxfp4_indexer` | BF16 query with final dimension 128; UINT8 packed data and scales | Quantize groups of 32 to FP4, packing two values per byte. Store power-of-two scales in E8M0. |
| `quantize_mxfp4_indexer.py`: `write_mxfp4_indexer_cache` | BF16 `[T,128]`, page/row slots `[T,2]`, strided UINT8 cache planes | Fuse quantization with cache writes. Invalid slots leave the cache unchanged. Preserve 64-bit address arithmetic for large physical page strides. |
| `fold_indexer_cache.py`: `fold_indexer_cache_rows` | K `[N,B,1,64]`, scales `[N,B,1,4]`, slots; folded `[N,B/8,1,544]` | Copy only updated rows into eight-row groups: 512 key bytes followed by 32 scale bytes. This is a layout conversion, with no requantization. |
| `prepare_indexer_indices.py`: `prepare_indexer_indices` | INT32 top-k `[T,K]`, positions and compression ratio; indices and optional lengths | Filter invisible compressed rows, sort surviving row indices chronologically and pad with `-1`. The FP32 bitwise sort keys preserve integer ordering without numeric conversion. |
| `spec_decode/dspark_swa_indices.py`: `build_dspark_swa_indices_triton` | Query boundaries and sequence lengths | Build DSpark's shared visible window for queries belonging to the same request; padded graph rows have zero valid length. |

## Current hardware usage

The table below describes current V4.1 call paths, not a claim that every kernel
supports every Ascend device. A hardware-neutral filename describes the cache
layout or operation; it does not enable a kernel on additional hardware.

| Source / entry point | Current call path |
| --- | --- |
| `packed_cache_slot_mapping.py` | A5 packed cache only, selected by the packed cache backend. A3 keeps its existing slot mapping path. |
| `build_window_indices.py` | A5 causal sliding-window indices through the packed cache backend. |
| `c2_ring_metadata.py` | A5 packed cache compressor metadata branch with device RoPE inputs. |
| `quantize_mxfp4_indexer.py` (query quantization and cache writes) | A5 MXFP4 packed indexer path. A3 uses its existing non-packed quantization path. |
| `fold_indexer_cache.py` | A5 folded packed indexer cache. |
| `prepare_indexer_indices.py` | Shared index preparation, including the A3 non-packed indexer path. |
| `spec_decode/dspark_swa_indices.py` | Shared DSpark logical-index fast path; selected by request and Triton/NPU conditions, with no A5-only gate. |
| `compressor/compressor_triton.py` | Shared compressor cache reads and updates, including A3. |

The packed cache backend is selected through `DSV41_PACKED_CACHE`, currently
provided by the A5 hardware profile. Shared call paths still require their
model and execution conditions; this table is not evidence of A3 runtime
validation or CI coverage.

## Execution and correctness

All inputs and outputs used by a launch must be on the same NPU. Cache producers
must complete before attention consumes their results. Graph callers retain the
input/output storage and update its contents before replay. Output padding is
part of the contract, because stale rows can otherwise overwrite a live cache.

The physical byte strides in cache writers and folding use 64-bit arithmetic.
Compression ratio and page size are compile-time layout parameters; token counts
and request lengths may change between steps.

## Compressor ring address regression

`compressor/compressor_triton.py` reads and updates ring pages with 64-bit
physical page offsets. Each FP32 page contains `32 * 2 * head_dim` elements;
with head dimension 512, page 65536 crosses INT32_MAX in element coordinates.
