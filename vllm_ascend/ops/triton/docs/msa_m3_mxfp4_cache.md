# MiniMax-M3 MXFP4 index cache insertion

## Description

- **Function**: `scatter_mxfp4_cache` copies packed index keys and their E8M0
  scales into caller-owned paged cache buffers. It does not quantize data.
- **Production caller**: `insert_mxfp4_index_cache` quantizes newly produced
  BF16/FP16 index keys with `npu_dynamic_mx_quant`, then calls this wrapper.
  Fused prepare uses `pack_mxfp4_query_and_insert_cache` to quantize contiguous
  index Q and new K rows in one call; it inserts K and returns packed Q/scales
  for IndexScore. Grouping stays on the last dimension in either path.
- **Algorithm**: one program handles each input token. For nonnegative slot
  `s`, copy its packed key and scale row to cache slot `s`. Negative slots
  represent graph padding and perform no writes. All other cache slots retain
  their previous bytes.
- **Supported modes**: the model's MXFP4 path is restricted to Ascend 950.
  Other devices retain their existing index-score paths.

## Parameters

| Parameter | Input/Output | Description | Data type |
| --- | --- | --- | --- |
| `packed` | Input | Packed new keys, `[T,D/2]`; two FP4 E2M1 values per byte | uint8 |
| `scales` | Input | One E8M0 scale per 32 logical elements, `[T,D/32]` | uint8 bytes or E8M0 |
| `slot_mapping` | Input | Physical slot per new token, `[T]` | int32 or int64 |
| `key_cache` | In-place output | Managed paged keys, `[P,128,D/2]` | uint8 |
| `scale_cache` | In-place output | Managed scales, `[P,128,D/64,2]` | E8M0 |

## Constraints

Logical `D` is 64 or 128. Packed data and both cache buffers must be contiguous;
scales use the same token order as keys. Nonnegative slots must be unique within
a call and below `P*128`; the model's slot allocator owns these bounds. The
wrapper avoids reading device slot values on the host. A zero-token call does
not launch a kernel. Input buffers and cache regions must not overlap.

The public wrapper supports graph capture with persistent inputs and outputs.
Replay may change packed bytes, scales, and slot values in place. The caller
orders writes before IndexScore on the execution stream. Cache allocation,
prefix reuse, and lifetime belong to vLLM's cache manager; IndexScore only reads
the packed cache and scales through its BBND input view.

## Origin and Differences

This is a model-specific copy kernel. It writes separate packed-key and scale
regions, unlike the ordinary BF16 index-key scatter. It leaves model weights
and the main attention K/V cache in their configured formats.

## JIT specialization and warmup

Packed/scale widths, their row strides, and the power-of-two copy width are
constexpr values. Supported contiguous inputs yield two width configurations.
Pointer dtype/alignment can also specialize. Token count changes the launch
grid; token data and slot values are runtime contents. No additional warmup
registry is introduced; normal model dummy execution/capture reaches the
wrapper. Compilation counts on the final main-based integration remain pending.

## Test Cases

The reference copies raw bytes into CPU cache snapshots and requires exact
equality, including untouched prefixes and padding. Cases include 64/128-wide
heads, empty/single-token input, speculative batches, and graph replay with
changed source data and slots.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_msa_m3_mxfp4_cache.py
```

CPU adapter tests check packing shapes, E8M0 order, enum selection, and the
actual FP4/E8M0 dtypes forwarded to the external score API:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q tests/ut/models/minimax_m3/test_msa_m3_mxfp4.py
```
