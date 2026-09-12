# batch_memcpy_kernel

## Description

- **Function**: Performs a batch of independent, variable-size device-to-device byte copies.
- **Formula**: `dst[i][0:sizes[i]] = src[i][0:sizes[i]]`.
- **Algorithm flow** (processed copy by copy, independently): load one source address, destination address, and byte count; cast addresses once; then copy masked byte tiles using streaming cache accesses.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950 hardware profiles with `TRITON_BATCH_MEMCPY`; 310P uses a tensor-copy fallback. Graph mode is N/A because aligned Mamba-state copies run outside the captured model forward.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `src_ptrs` | Input | Absolute source addresses `[batch]` | int64 | ND |
| `dst_ptrs` | Input | Absolute destination addresses `[batch]` | int64 | ND |
| `sizes` | Input | Copy sizes in bytes `[batch]` | int32 | ND |
| `BLOCK_SIZE` | Attribute | Compile-time byte tile size | int | scalar |

## Constraints

- Launch grid size must equal the number of copies; every address range must be valid for its byte count.
- Source and destination ranges for a copy must not overlap. Zero-sized copies are allowed.
- Copying is bytewise and therefore independent of the tensors' element dtypes.

## Origin and Differences

- **Origin**: Adapted from vLLM's `vllm/v1/worker/mamba_utils.py`.
- **Differences**:
    - NPU adaptation for performance: hoists pointer casts outside the loop to avoid Triton-Ascend pointer-analysis failures, uses an 8192-byte tile instead of upstream's 1024-byte tile, and bypasses L1 for streaming traffic.
    - Modified behavior: unlike current upstream, this kernel has no left-overlap barrier. Callers must provide non-overlapping source and destination ranges.

## Test Cases

The NPU test copies four representative Mamba-state byte ranges `{24576, 262144, 24576, 262144}` using bf16 and fp32 backing tensors. It requires bit-exact output (`rtol=atol=0`). Overlap behavior is not tested because overlapping ranges are outside this kernel's contract.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_batch_memcpy.py
```
