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
| `engram_gate.py` | Fuse the FP32 normalization, key/query weighting, gate and residual addition used by native A5 Engram checkpoints. Retain the unfused operation order rather than contracting the residual into an FMA. |

## Current hardware usage

Hash generation and lookback gathering are shared model paths, including A3.
Lookup also serves A3 BF16/INT8 tables; its MXFP8 path is selected by
`ENGRAM_MXFP8`, currently an A5 capability. The fused `engram_gate.py` path is
selected by `ENGRAM_UNROTATED_GATE`, also currently an A5 capability. Moving
these kernels into `ops/triton` does not change their hardware routing. These
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
