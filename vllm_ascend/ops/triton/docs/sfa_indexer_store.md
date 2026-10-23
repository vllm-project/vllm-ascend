# Native SFA indexer key/scale store

## Description

- **Function**: Stores native-quantized INT8 keys and converts their FP32 scales to the FP16 cache representation in one kernel. Quantization remains in the existing backend path, preserving its tie behavior.
- **Formula**: For each valid slot `s`, `key_cache[s, :] = quant[row, :]` and `scale_cache[s] = FP16(scales[row])`. Negative padding slots and slots outside cache capacity perform no stores.
- **Algorithm flow**: The backend computes native quantization, finishes required stream/layout dependencies and publishes key and scale together at the existing cache-write point.
- **Supported modes**: NPU eager execution and ACLGraph capture/replay. The store itself has no DCP-size dependency; Atlas A3 has operator-level execution evidence.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `quant` | Input | Native quantized keys, `[rows, 128]` | INT8 | ND, contiguous |
| `scales` | Input | Native scales before cache conversion, one per row | FP32 | ND, contiguous |
| `slots` | Input | Cache destination for each row | INT32 or INT64 | ND, contiguous |
| `cache` | Output | Key cache with trailing width128 | INT8 | ND, contiguous |
| `scale_cache` | Output | One scale per key-cache slot | FP16 | ND, contiguous |

## Constraints

- Valid slots must be unique within a launch, as guaranteed by decode slot mapping. The kernel does not resolve competing writes to the same valid slot.
- The backend selects this store only for its native C8 layout and small decode-only batches, currently up to12 rows. C4, PCP/DSA-CP layout transforms and custom cache backends retain their existing write paths.
- Layout eligibility is checked at selection time. The internal store wrapper assumes native quantization output and does not repeat those checks.
- Capacity is a runtime argument, not a constexpr specialization key. Slot bounds masking remains necessary for padded rows and memory safety.

## Origin and Differences

- **Origin**: Native indexer key/scale cache publication.
- **Differences**: Fuses scale conversion and the two scatters, not quantization or communication. Moving conversion to the publication kernel does not move publication before connector/query dependencies.

## Test Cases

The operator tests compare cache bits exactly, including invalid/padded slots, changing input graphs, capacities and both slot dtypes. CPU routing tests check native quantization and dependency ordering, with C4 and parallel-layout fallbacks. Operator coverage is distinct from model-level C8 coverage: the checkpoint used by the recorded GLM-5.2 model evaluation did not select this store.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sfa_indexer_store_triton.py
```
