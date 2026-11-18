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

For host lookup, `vocab_start` and `vocab_end` define the global ID range
owned by the caller. `ROW_OFFSET` is the row offset from the first registered
chunk in the pointer tables to the beginning of that range. For an owned ID,
the kernel indexes row `ID - vocab_start + ROW_OFFSET`, then divides by
`CHUNK` to select the pointer and row within the chunk. Codes and scales must
use matching chunk boundaries and row offsets. An out-of-range ID reads the
row at `ROW_OFFSET` and writes zeros. The caller must ensure that row is valid.
The host lookup wrapper passes `codes.row_offset`, defaulting to zero when it
is absent. `ROW_OFFSET` is a `tl.constexpr`, so different offsets produce
different JIT specializations.

### Host lookup parameters and validation

`gather_dequantize_host_uva` in `models/deepseek_v41/engram/npu.py` is the
model-specific entry point. Its host buffers contain chunk pointer tables and
two-dimensional tensors describing row count, width and dtype; IDs have shape
`[tokens, hash_columns]`. The caller supplies the output or the wrapper
allocates BF16 output. The relevant kernel attributes are:

| Parameter | Meaning |
| --- | --- |
| `CHUNK` | Rows represented by each pointer-table entry. |
| `ROW_OFFSET` | First owned row's offset relative to the first pointer-table entry, in rows. |
| `WIDTH`, `GROUP` | Elements per row and elements per quantization scale group. |
| `HEAD_START`, `LOCAL_HEADS`, `PAD_HEADS` | Input head-column start, number of heads read, and output rows reserved per token. |
| `QUANTIZED`, `MXFP8` | Select whether and how scales and codes are decoded. |

The pointer tables must cover all rows in the owned range and use the same
chunk alignment for codes and scales. The kernel assumes a non-empty owned
range, valid registered pointers, a supported power-of-two `WIDTH`, and a
compatible `GROUP` when quantized. It is derived from the existing Engram
host lookup in vLLM-Ascend; this change adds the chunk-relative row offset for
unaligned shared-pool slices.

The row-offset boundary test uses NPU-resident chunk pointers and checks BF16
and INT8 results exactly against a PyTorch reference. It does not validate
host registration or MemFabric. Run it on a supported Ascend device with:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_engram_host_lookup.py
```

Output has shape `[tokens * pad_heads, width]`. Only `local_heads` rows per token
are written, starting at output head zero; `head_start` selects the corresponding
input ID columns. These two offsets must not be confused.
