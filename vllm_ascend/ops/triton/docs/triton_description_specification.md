> [!NOTE]
> **Filling Instructions**
> - Place the document under `vllm_ascend/ops/triton/docs`, named after the operator. If a same-named document already exists, add a suffix to distinguish it.
> - Fill in strictly following the template. For items that do not apply, use "N/A"; do not leave them blank or missing.

# ops

<!-- Operator name -->

## Description

- **Function**:
- **Formula**:
- **Algorithm flow** (processed row by row, independently):
- **Supported modes**:

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |

## Constraints

- Shape, dtype, value range, constraints, and graph-mode support of each input parameter

## Origin and Differences

- **Origin**: which vllm operator it is modified from, or developed from scratch
- **Differences**:
  - NPU adaptation for performance;
  - Modified for a specific vllm-ascend logic or different input parameters

## Test Cases

> [!NOTE]
> **Test Case Instructions**
> - Single-operator accuracy test cases should be placed under `tests/e2e/nightly/single_node/ops/singlecard_ops/triton`.
> - For inference scenarios, use the actual shapes and other parameters adopted by the model as single-operator test cases, rather than arbitrarily constructed ones.
> - Accuracy comparison results should use a unified precision tolerance based on the operator type and data type; example cases will be provided later.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_top_k_top_p_triton.py
```
