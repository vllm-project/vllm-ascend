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

### MRV2 lookback window

`EngramModelState.prepare_engram_inputs` refreshes a persistent, contiguous
`int32` window of shape `[max_num_reqs, depth]`. The step's `idx_mapping`
contains one state slot per active request in batch order. For each active
row, the kernel reads the slot's device-side computed-token count. It gathers
from that slot's `int32` history when the count is positive. History columns
are contiguous and rows are addressed with the supplied row stride. With
PCP the mapping comes from the global batch.

For a count `k`, the reference takes the first `k` tokens in the slot,
reverses them, keeps at most `depth` and pads the rest with `-1`. A new
request has `k = 0` and fills its window with `-1` without computing a history
address. Short histories keep masked lanes inside the same slot. Rows beyond
the active batch write `-1` across the window without addressing any input.
One program handles each window row.
`DEPTH` and its power-of-two tile size are compile-time parameters.

The [single-card PR cases](../../../../tests/e2e/pull_request/one_card/test_engram_lookback.py)
compare every output and both guard regions exactly. They include the full
eight-request batch with six empty slots reported in
[#21392](https://github.com/vllm-project/vllm-ascend/issues/21392).
The [nightly case](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_engram_lookback.py)
uses eight request slots and 128K history columns. Run either file with `pytest -q` on an NPU.

### Embedding lookup

Host lookup receives an NPU-visible pointer table created after ACL host-memory
registration. Pointers identify chunks so large tables do not require 32-bit
offset arithmetic across the whole allocation. The model must keep registrations
alive until the consuming stream finishes. Kernel execution does not register or
release host memory.

Output has shape `[tokens * pad_heads, width]`. Only `local_heads` rows per token
are written, starting at output head zero; `head_start` selects the corresponding
input ID columns. These two offsets must not be confused.
