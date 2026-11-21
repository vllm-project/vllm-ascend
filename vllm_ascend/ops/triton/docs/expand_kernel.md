# expand_kernel

## Description

- **Function**: Repeats one value per request over that request's flattened token range, with an optional sentinel replacement.
- **Formula**: `output[j] = replace_to if input[r] == replace_from else input[r]` for `cu[r-1] <= j < cu[r]`.
- **Algorithm flow** (processed request by request, independently): derive start/end from cumulative counts, transform the source value, and fill the request's output interval.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used to expand speculative-sampling metadata; graph mode is N/A because sampling follows the captured model-forward graph.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `output_ptr` | Output | Flattened expanded values `[num_tokens]` | same as input | ND |
| `input_ptr` | Input | One value per request `[batch_size]`; int32 and fp32 are used by serving | int32 / fp32 | ND |
| `cu_num_tokens_ptr` | Input | Inclusive cumulative token counts | int32 / int64 | ND |
| `replace_from`, `replace_to` | Attribute | Value substitution pair | same as input | scalar |
| `vec_len` | Attribute | Active request count | int | scalar |
| `MAX_NUM_TOKENS`, `BLOCK_SIZE` | Attribute | Compile-time maximum tokens per request and request tile | int | scalar |

## Constraints

- Cumulative counts must be nondecreasing and each difference must not exceed `MAX_NUM_TOKENS`.
- Output capacity must be at least the final cumulative count.

## Origin and Differences

- **Origin**: Developed for vLLM Ascend speculative sampling metadata expansion.
- **Differences**:
    - NPU adaptation for performance: fuses request-wise repeat-interleave and value replacement in one vector-core launch.
    - Modified for vllm-ascend logic: applies the speculative-sampling sentinel replacement while expanding request metadata.

## Test Cases

The NPU test expands five fp32 request values using cumulative counts `[2,2,5,6,9]`, which includes a zero-length request, and replaces `-1.0` with `99.0`. It requires bit-exact equality with the PyTorch reference. Kernel warm-up additionally covers int32 metadata values.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py::test_expand_kernel
```
