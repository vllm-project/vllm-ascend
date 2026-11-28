# Suffix acceptance observation

## Description

- **Function**: Accumulate accepted/observed draft counts and consume previous-draft markers.
- **Formula**: `accepted = num_sampled - 1`; `verified = accepted + num_rejected`.
- **Algorithm flow**: `launch_suffix_observe` reduces requests per statistics bucket, then clears consumed markers in a second launch on the same stream.
- **Supported modes**: Ascend A3 and A5; adaptive verification uses full decode graphs with variable-query-length attention support.

## Parameters

`B` is active batch size, `R` request capacity and `K` draft width.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `idx_mapping_ptr`, `num_sampled_ptr`, `num_rejected_ptr` | Input | Request slots and verification counts | int32 | `[B]` each |
| `stats_ptr` | Input / Output | Counts by source/length bucket, draft position and accepted/observed field | int64 | `[18, K, 2]` |
| `last_bucket_ptr`, `last_valid_ptr` | Input / Output | Previous bucket markers and valid draft counts; clearing writes only bucket markers | int32 | `[R]` each |
| `K`, `K_PO2`, `launch_options` | Attribute | Draft width, padded width and compiler options | Integers / mapping | Host values |

## Constraints

Buffers are contiguous on one device. Length buckets match upstream: 1-6 individually, then 7-10, 11-20 and 21+; two draft sources produce 18 buckets.
Each bucket has one writer. Markers are cleared after all reductions to avoid cross-bucket races.
The wrapper derives the active count and its power-of-two reduction width from the mapping; a new width can compile a new variant.

## Origin and Differences

- **Origin**: [vLLM #60339](https://github.com/vllm-project/vllm/pull/60339).
- **Differences**: Replace contended int64 atomics with int32 reductions followed by int64 updates. Non-power-of-two draft widths use scalar statistics accesses.

## Test Cases

Counters require exact equality; confidence checks use approximate floating-point comparison. Cases cover sparse slots, 64/65 requests and padded draft widths.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_suffix_observe.py
```

## Example

N/A for standalone use; the speculator dispatches observation before proposing the next draft.
