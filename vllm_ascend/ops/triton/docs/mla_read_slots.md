# MLA cache read slots

## Description

- **Function**: Construct the physical cache slot for each token of a strided MLA history-cache read without materializing a per-token request index.
- **Formula**: For request `i` and local token `j`, let `position = offset[i] + j`; the slot is `block_table[i, floor(position / block_size)] * block_size + (position % block_size)`. A negative logical page index follows PyTorch indexing semantics.
- **Algorithm flow**: Keep sequence lengths and offsets on device. Check the first and last logical pages of every nonempty request with the fallback's PyTorch indexing operation, and use the checked page values in the slot kernel. For one or two requests and block sizes up to 512, one Triton program processes a page at a time. Other supported batches use a per-token Triton kernel. Both paths retain the reader's existing row-copy operation.
- **Supported modes**: The wrapper targets NPU tensors and returns `None` for unsupported metadata, so the caller can use the existing implementation. Cross-device correctness and performance must be validated for each supported Ascend family; historical device results are described in the PR.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `block_table` | Input | Per-request logical-to-physical page mapping; may be strided | int32 or int64 | Two-dimensional NPU tensor |
| `seq_lens` | Input | Nonnegative number of local rows per request; may be strided | int32 or int64 | One-dimensional NPU tensor |
| `seq_offset` | Input | Optional signed first position for each request; may be strided | int32 or int64 | One-dimensional NPU tensor or `None` |
| `num_tokens` | Attribute | Total output rows, equal to the sum of `seq_lens` | Python int | Scalar |
| `block_size` | Attribute | Positive tokens per physical page | Python int | Scalar |
| return value | Output | Physical slot for each output row, or `None` for fallback | int64 | One-dimensional NPU tensor or `None` |

## Constraints

- The optimized path supports one to eight requests; other batch sizes and unsupported shapes, dtypes, or devices return `None` and retain the original reader.
- The caller validates matching devices and buffer shapes. Page indices must follow PyTorch indexing rules, and sequence lengths must sum to `num_tokens`; the wrapper checks endpoint bounds and lengths without a host scalar read. Out-of-bounds device errors can be reported asynchronously, as in the original indexed path.
- Requests with zero length do not use their offsets. Negative physical page numbers are preserved in the slot output, allowing the existing copy kernel to handle negative slots.
- A zero-token input returns an empty tensor. No caller-owned tensor is modified.
- Graph replay may change lengths, offsets, and table contents while the captured tensor shapes and total output size remain fixed. NPU graph coverage is in the test below.
- Scalar values `num_tokens` and the page count avoid value specialization. Sequence count, page size, metadata strides, metadata dtypes, and presence of offsets affect compilation variants. The operator does not register a separate warmup hook.

## Origin and Differences

- **Origin**: Derived from the existing PyTorch slot construction in the strided MLA cache reader.
- **Differences**: The fast path avoids a per-token request-id tensor and repeated advanced indexing; one or two sequences additionally reuse a block-table lookup across a page. The existing reader remains the fallback for unsupported inputs.

## Test Cases

The CPU integration and validation tests isolate the device boundary. The NPU test compares exact integer slots with an independent per-token reference and covers changing metadata during graph replay.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q tests/ut/device/test_paged_mla_cache_read.py tests/ut/device/test_mla_read_slot_validation.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mla_read_slots.py
```
