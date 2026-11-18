# prepare_qsa_lightning_v2_inputs

## Description

- **Function:** Prepare native LightningIndexerV2 inputs for Qwen QSA in one
  Triton launch. The production caller is
  `models/qwen4_exp/lightning_indexer.py::qsa_select_paged_tokens_lightning_v2`.
- **Formula:** For request length `L`, compression ratio `r` and query row count
  `T`, produce `L // r`, `L % r` and `min(query_start_loc, T)`. Fill weights with
  `float32(bfloat16(1 / sqrt(128)))`, preserving the existing effective value.
- **Algorithm flow:** Each vector tile independently writes request lengths,
  query boundaries and query-head weights with separate bounds masks.
- **Supported modes:** Intended for Atlas A2, Atlas A3 and 950PR/950DT through
  Triton Ascend. Hardware compilation, accuracy and performance are pending;
  CPU contract checks do not establish device support or speedup.

## Parameters

| Parameter | Direction | Shape / type | Meaning |
| --- | --- | --- | --- |
| `query` | Input | `[T, H, 128]`, model BF16 | Only shape and device are used |
| `sequence_lengths` | Input | `[B]`, INT32 or INT64 | Nonnegative original-token lengths |
| `query_start_loc` | Input | At least `[B + 1]`, INT32 or INT64 | Packed query boundaries |
| `compress_ratio` | Attribute | Positive integer; QSA uses 4 | Tokens per compressed group |
| `weights` | Output | `[T, H]`, FP32 | Uniform score weights |
| `cu_query` | Output | `[B + 1]`, boundary input dtype | Boundaries clamped to T |
| `compressed` | Output | `[B]`, length input dtype | Complete compressed-group counts |
| `residual` | Output | `[B]`, length input dtype | Remaining original-token counts |

Outputs are returned in table order: weights, cu_query, compressed, residual.

## Constraints

- Inputs share a device; metadata vectors may be strided. `H` must be positive.
  Zero query rows and zero requests are supported by the wrapper. The model
  caller bypasses preparation entirely for zero query rows.
- Outputs are fresh contiguous allocations. Inputs are not mutated. There is
  no head padding, query copy or per-token block-table gather.
- Valid model metadata describes contiguous query suffixes per request. Padded
  boundaries are clamped, without discarding the original request capacity.
- The wrapper reads no device values on the host. Graph replay must observe
  updated lengths and boundaries at the captured addresses.
- CANN schedule generation and reuse remain unchanged. The input adapter still
  prepares the reusable schedule outside the layer loop. A trimmed query causes
  the selector to regenerate it from the same fused outputs passed to Indexer.
- Output index-buffer initialization and group packing remain separate.

## Origin and Differences

- **Origin:** Replaces four PyTorch operations in the native V2 selector:
  `floor_divide`, `remainder`, `clamp_max` and `full`.
- **Differences:** V2 keeps original heads and request-level page tables.
  This operator replaces input
  preparation only; CANN still performs scoring and TopK.
- **JIT:** Compression ratio, score weight and vector tile size are compile-time
  parameters. Counts and strides are runtime integers, subject to the backend's
  ordinary scalar specialization classes. Metadata contents do not specialize.
  Pointer dtypes/alignment and compiler options can create variants. The fixed
  256-element tile is a starting configuration, not a measured optimum. No
  explicit warmup registration is added; actual compilation counts are pending.

## Test Cases

CPU tests cover decode/prefill row counts, padded requests, empty inputs,
both integer dtypes, strided metadata and schedule reuse/regeneration.
NPU e2e cases are deferred; device validation must compare all four outputs
exactly against the original PyTorch expressions, including FP32 weights,
and exercise graph replay after metadata changes.

```bash
pytest -sv tests/ut/ops/test_qsa_lightning_v2_prepare_contract.py
pytest -sv tests/ut/models/qwen4_exp/test_lightning_indexer_v2.py
```

CPU tests cover allocation, launch wiring, dtype/stride contracts, schedule
reuse, trimming and output-buffer behavior. Device arithmetic is not emulated.
NPU exact comparison, native Indexer integration, A3/A5 timing, model accuracy
and end-to-end performance remain required before claiming performance gains.
