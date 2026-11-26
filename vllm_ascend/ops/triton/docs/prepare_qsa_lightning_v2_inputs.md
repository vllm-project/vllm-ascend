# prepare_qsa_lightning_v2_inputs

## Description

- **Function:** Prepare native LightningIndexerV2 inputs for Qwen QSA in one
  Triton launch. The production caller is
  `models/qwen4_exp/lightning_indexer.py::qsa_select_paged_tokens_lightning_v2`.
- **Formula:** For request length `L`, compression ratio `r` and query row count
  `T`, produce `L // r`, `L % r` and `min(query_start_loc, T)`. Fill weights with
  `float32(bfloat16(1 / sqrt(128)))`, which is `0.08837890625`.
- **Algorithm flow:** Each vector tile independently writes request lengths,
  query boundaries and query-head weights with separate bounds masks.
- **Devices:** Requires NPU and Triton Ascend. CPU contract checks do not
  establish device support.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `query` | Input | Only shape and device are used; contents are not read | Production caller uses BF16; wrapper does not restrict dtype | `[T, H, 128]` |
| `sequence_lengths` | Input | Nonnegative original-token lengths | INT32 or INT64 | `[B]`, strided allowed |
| `query_start_loc` | Input | Packed query boundaries; only the first `B + 1` elements are used | INT32 or INT64 | 1D, length at least `B + 1`; strided allowed |
| `compress_ratio` | Attribute | Positive tokens per group; QSA uses 4 | Integer | Scalar |
| `weights` | Output | Uniform score weights | FP32 | `[T, H]`, contiguous |
| `cu_query` | Output | Each boundary is capped at `T`; no lower-bound clamp | Same dtype as `query_start_loc` | `[B + 1]`, contiguous |
| `compressed` | Output | Complete compressed-group counts | Length input dtype | `[B]`, contiguous |
| `residual` | Output | Remaining original-token counts | Length input dtype | `[B]`, contiguous |

Here `T` is the query token count, `H` is the query head count, and `B` is
`sequence_lengths.numel()`. Outputs are returned in table order:
`weights`, `cu_query`, `compressed`, `residual`.

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
  Pointer dtypes/alignment and compiler options can create variants.

## Test Cases

The NPU e2e entry point covers exact output comparison, input preservation,
both integer dtypes, strided metadata and graph replay after metadata updates.
All four outputs are compared against the original PyTorch expressions with
`rtol=0`, `atol=0`, including the BF16-rounded weight promoted to FP32.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qsa_lightning_v2_prepare.py
```
