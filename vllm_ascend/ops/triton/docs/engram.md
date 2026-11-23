# Engram kernels

## Responsibilities

Model code owns embedding tables, request state and auxiliary streams. The
Triton implementations live in `ops/triton`; moving them does not change the
table layout or arithmetic.

| Source | Operation and layout |
| --- | --- |
| `engram_lookup.py` | Gather global hash IDs from a local vocabulary shard. Device tables and registered host tables support BF16, INT8/FP32 scales and MXFP8/E8M0 scales. Each program writes one token/head row; out-of-shard IDs produce zeros and head padding remains untouched. |
| `engram_lookup.py`: `_decode_e8m0` | Decode unsigned exponent bytes. Code 0 represents `2^-127`, and 255 represents NaN. Mask sign extension before constructing FP32 bits. |
| `engram_hash.py` | Map tokens to requests, read preceding tokens from the current chunk, lookback window or slot cache, then calculate per-layer n-gram hashes. Image/dead tokens stop valid history. Empty requests must not steal another request's tokens. |
| `engram_lookback.py` | Gather the tokens preceding each request's chunk from device token history. Respect request-state reordering and fill missing history and padded requests with `-1`. |
| `engram_gate.py` | Fuse the FP32 normalization, key/query weighting, gate and residual addition used by Engram checkpoints. Rotated inputs supply a separate FP32 scoring tensor while the residual stays in its checkpoint basis. Retain the unfused operation order rather than contracting the residual into an FMA. |

## Current hardware usage

Hash generation and lookback gathering are shared model paths, including A3.
Lookup also serves A3 BF16/INT8 tables; its MXFP8 path is selected by
`ENGRAM_MXFP8`, currently an A5 capability. The fused `engram_gate.py` path is
selected by `ENGRAM_UNROTATED_GATE`, also currently an A5 capability. Moving
these kernels into `ops/triton` does not change their hardware routing.
Rotated A3 checkpoints use `ENGRAM_ROTATED_GATE`: the FP32 rotation GEMM
remains separate, then the fused gate consumes both restored and original hidden
states. Other hardware retains its existing fallback for rotated checkpoints. These
call-path descriptions do not establish A3 runtime validation or CI coverage.

## Memory and synchronization

Host lookup receives an NPU-visible pointer table created after ACL host-memory
registration. Pointers identify chunks so large tables do not require 32-bit
offset arithmetic across the whole allocation. The model must keep registrations
alive until the consuming stream finishes. Kernel execution does not register or
release host memory.

Output has shape `[tokens * pad_heads, width]`. Only `local_heads` rows per token
are written, starting at output head zero; `head_start` selects the corresponding
input ID columns. These two offsets must not be confused.

## Rotated gate validation

Run on an A3 NPU with the supported PyTorch, torch_npu and Triton stack:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_engram_gate.py
```

The test covers the profiled `[132, 4, 5120]` shape, empty input, a non-power-of-two
channel count, shared and per-block rotations, BF16/FP32, strided tensors and
masked-token residual identity. A model-dispatch test also captures and replays
the rotated path with updated inputs and masks. Kernel compilation, graph replay, model accuracy
and latency still require NPU evidence before merging. No hardware speedup is
claimed from reducing the number of launches alone. The rotation also retains
its FP32 input conversion; this is not a claim of exactly three launches.

## Gate channel tiling

The gate reduces scoring inputs in tiles of at most 1024 channels and writes
the residual in tiles of the same size. This bounds live FP32 vectors instead
of padding a 5120-channel row to 8192 channels, which exceeded A5 UB capacity
for FP32 inputs. Tile reductions accumulate in FP32; their reduction order
differs from a single full-row reduction. The public interface, output dtype,
input strides and hardware routing are unchanged. No additional kernel launch
or host synchronization is introduced. The tile size is derived in the public
wrapper from the channel dimension and adds no workload-dependent tuning key.

The gate test above runs on A3 and A5 and covers a full tile (1024 channels),
a partial second tile (1056), and the failing 5120-channel FP32 cases for all
rotation modes. Accuracy and latency comparisons on both devices remain
required before merge.
