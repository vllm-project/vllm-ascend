# Small SFA indexer row store

## Description

The unquantized SFA indexer writes one key per live token after projection,
RoPE and any cache-connector wait. On Atlas A3, a small, fixed-length target
decode graph can issue a single Triton program per token. Each program loads
its slot, skips padding and copies a vector directly to the corresponding row
of the original cache allocation. For each live slot `s = slots[t]`, the
update is `cache.view(-1, width)[s, :] = updates[t, :]`.

## Parameters

| Parameter | Direction | Shape and layout | Type |
| --- | --- | --- | --- |
| `cache` | In/out | Contiguous ND [pages, tokens per page, 1, width] | FP16 or BF16 |
| `updates` | Input | ND [tokens, width], dense columns, row stride at least width | Same as cache |
| `slots` | Input | Contiguous ND [tokens] | INT32 or INT64 |

## Constraints

The wrapper accepts 1–32 rows, widths up to 2048 and whole 32-byte row
increments. All tensors have zero storage offset and share one device.
Cache and source row spans fit in 2³¹ elements. Live slots, including slot
zero, are unique. Negative and out-of-range
slots leave every cache row untouched. Updates and slots occupy storage
independent of the cache. Replay rereads the current slot values on device.

The backend selects this path while capturing a FULL target decode graph in
FULL_DECODE_ONLY or FULL_AND_PIECEWISE mode. It keeps the existing cache
writer for eager, draft, adaptive verification, CP, KV producer, quantized
and noncontiguous cache paths. Other hardware follows its existing adaptor.
The caller owns synchronization and supplies ordinary decode slot mappings.

## Origin and Differences

The original path flattens contiguous pages and calls
`DeviceOperator.scatter_cache`, which uses the SK scatter on supported A3
layouts. The specialization uses the same slots and destination storage. A
program handles one row with a bounded 32-bit address calculation and a
masked vector copy. The row count controls only the launch grid. Width,
capacity and source row stride specialize the compiled kernel at capture.

## Test Cases

Host-side tests cover layout limits and dispatch. The one-card test checks
the whole physical cache bit for bit before and after graph replay, including
padding, invalid destinations, page boundaries and changing slot values.

```bash
pytest -q tests/ut/attention/test_small_indexer_row_store.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_small_indexer_row_store.py
```
