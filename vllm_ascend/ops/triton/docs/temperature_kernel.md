# _temperature_kernel

## Description

- **Function**: Applies per-request temperature scaling to expanded token logits in place.
- **Formula**: `logits[t,v] = logits[t,v] / temperature[request(t)]`; temperatures `0` and `1` leave the row unchanged.
- **Algorithm flow** (processed token row by token row, independently): resolve the request, load its temperature, skip identity/disabled scaling, and divide vocabulary tiles in fp32.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used by the V2 sampler; graph mode is N/A because sampling runs after the captured model-forward graph.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits_ptr` | Input/Output | Logits `[num_tokens, vocab_size]` | fp32 | ND |
| `logits_stride` | Attribute | Logit row stride | int | scalar |
| `expanded_idx_mapping_ptr` | Input | Token-to-request mapping | int32 | ND |
| `temperature_ptr` | Input | Temperature per request | fp32 | ND |
| `vocab_size` | Attribute | Number of valid columns | int | scalar |
| `BLOCK_SIZE` | Attribute | Compile-time vocabulary tile size (44032 in the launcher) | int | scalar |

## Constraints

- Request indices must index `temperature_ptr`; the last logits dimension must be contiguous.
- Temperature values must be finite and non-negative. Zero is treated as disabled rather than used as a divisor.
- Vocabulary tails are masked; arbitrary positive `vocab_size` values are supported.

## Origin and Differences

- **Origin**: Adapted from vLLM's `vllm/v1/worker/gpu/sample/gumbel.py`.
- **Differences**:
    - NPU adaptation for performance: uses a 44032-element fp32 tile instead of the upstream 8192-element tile to reduce vocabulary passes while fitting the Atlas A2/A3 192-KB UB, and disables multibuffering.
    - Modified behavior: avoids loading logits when the temperature is `0` or `1`; both values leave the row unchanged.

## Test Cases

The NPU test compares with a PyTorch reference for fp32 logits, 1–64 rows, vocabulary sizes `{32000, 50257, 65024, 128256, 151936}`, and temperatures including `0.0` and `1.0`. It uses `rtol=1e-5` and `atol=1e-4`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_temperature.py
```
