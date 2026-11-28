# Engram kernels

## Responsibilities

Model code owns embedding tables, request state and auxiliary streams. The
Triton implementations live in `ops/triton`; moving them does not change the
table layout or arithmetic.

| Source | Operation and layout |
| --- | --- |
| `engram_lookup.py` | Gather global hash IDs from a local vocabulary shard. Device tables and registered host tables support BF16, INT8/FP32 scales and MXFP8/E8M0 scales. Each program writes one token/head row; out-of-shard IDs produce zeros and head padding remains untouched. |
| `engram_lookup.py`: `_decode_e8m0` | Decode unsigned exponent bytes. Code 0 represents `2^-127`, and 255 represents NaN. Mask sign extension before constructing FP32 bits. |
| `engram_lookup.py`: `_engram_offload_encode_kernel` | Encode global hash IDs into MemFabric rank-slot entry IDs and publish the current chunk count. Invalid IDs reference a reserved zero row. |
| `engram_lookup.py`: `_engram_offload_decode_kernel` | Read rows staged by native `entry_gather`, dequantize them and scatter into the token/head output. |
| `engram_lookup.py`: `_mapped_engram_gather_dequant_kernel` | Resolve each global hash ID against owner base addresses, read the packed mapped row and dequantize directly into the output. |
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

An explicit `engram_offload` pool uses the encode, native `entry_gather`, and
decode sequence on A2/A5, and as a fallback when mapped addresses are unavailable.
On A3, a successful mapped-pool query selects `mapped_engram_gather_dequant()`.
It avoids encoded-ID and staged-row buffers. Neither path changes the local
Host UVA default when no pool backend is configured.

## MemFabric lookup layouts and specialization

### Encoded entry IDs and staged rows

For each table, owners divide the global row range into contiguous intervals
of `ROWS_PER_OWNER` rows. The encode kernel selects an owner from the `owners`
vector and computes:

```text
entry = owner_rank * ROWS_PER_SLOT + TABLE_OFFSET + local_row
```

The registered rank slot contains all configured tables and a reserved zero row.
Invalid IDs use that zero row; `count[0]` still records the current chunk's
`size`. Input IDs are INT64 with contiguous head columns and a runtime token
stride. `HEAD_START` and `LOCAL_HEADS` select this rank's input head columns.
`row_offset` selects a chunk within the flattened token/head requests.

Each staged row contains contiguous codes followed by one scale per 32 values.
Supported formats are BF16 without scales, INT8 with FP32 scales, and MXFP8
E4M3FN with E8M0 scale bytes. At width 256, the corresponding row sizes are
512, 288 and 264 bytes. Decode writes BF16 output, zeros invalid entries and
leaves padded heads untouched.

### Mapped gather and dequantization

`mapped_engram_gather_dequant()` receives an INT64 device vector of table base
addresses, in owner order, and uses the same global row IDs and packed row
formats. Each base already includes its table offset. The kernel calculates
the owning interval and a 64-bit byte address, then writes directly to a
contiguous `[tokens, padded_heads, width]` BF16 output. Empty requests return
without launching a kernel.

Mapped MXFP8 rows are read as bytes and decoded explicitly as E4M3FN. This
keeps the byte-addressed mapping kernel independent of native FP8 load/cast
support, including the handling of subnormals, signed zero and NaN encodings.
The typed device/staging kernels use native FP8 conversion. E8M0 scale
decoding is shared through `_decode_e8m0`.

### Compile-time parameters

| Kernel | `tl.constexpr` parameters |
| --- | --- |
| `_engram_offload_encode_kernel` | `ROWS_PER_OWNER`, `ROWS_PER_SLOT`, `TABLE_OFFSET`, `HEAD_START`, `LOCAL_HEADS`, `BLOCK` |
| `_engram_offload_decode_kernel` | `CODE_STRIDE`, `SCALE_STRIDE`, `WIDTH`, `GROUP`, `HEAD_START`, `LOCAL_HEADS`, `PAD_HEADS`, `QUANTIZED`, `MXFP8` |
| `_mapped_engram_gather_dequant_kernel` | `ROWS_PER_OWNER`, `ROW_BYTES`, `WEIGHT_BYTES`, `WIDTH`, `GROUP`, `HEAD_START`, `LOCAL_HEADS`, `PAD_HEADS`, `QUANTIZED`, `MXFP8` |

These values describe the configured table layout, dtype and TP head partition;
changing them can compile another kernel variant. `GROUP` is fixed at 32 by the
storage format, and encode uses a block size of 128. IDs and count contents
remain device data and do not specialize the kernels. Token count controls
the launch grid; runtime scalar arguments such as strides and offsets retain
Triton's normal specialization rules. No new warmup registry is added here.

## Memory and synchronization

Host lookup receives an NPU-visible pointer table created after ACL host-memory
registration. Pointers identify chunks so large tables do not require 32-bit
offset arithmetic across the whole allocation. The model must keep registrations
alive until the consuming stream finishes. Kernel execution does not register or
release host memory.

Output has shape `[tokens * pad_heads, width]`. Only `local_heads` rows per token
are written, starting at output head zero; `head_start` selects the corresponding
input ID columns. These two offsets must not be confused.

For MemFabric staging, the caller registers and owns the HBM destination, and
native `entry_gather` queues its completion dependency before decode consumes
the rows. For mapped lookup, the caller owns the pool and base-address vector.
Keep all referenced allocations alive through execution and graph replays;
drain execution and retire referencing graphs before releasing the pool.
The kernels do not establish model-level collective or stream ordering.

## Validation entry points

CPU unit tests cover pool routing, contributor/consumer ownership, chunk
boundaries and backend selection:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/ut/models/test_engram_offload.py \
  tests/ut/models/test_engram_unified_dispatch.py
```

Run the mapped-kernel accuracy and graph-replay tests on an NPU:

```bash
python -m pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mapped_engram_lookup.py
```

The NPU tests compare against PyTorch gather/dequantization with exact BF16
results. They cover BF16, INT8 and MXFP8, empty and nonempty requests,
noncontiguous token strides, owner boundaries, invalid IDs, padded heads, and
changing ID contents between graph replays. They use synthetic owner buffers;
real MemFabric registration and cross-node connectivity require integration
tests in addition to these kernel tests.
