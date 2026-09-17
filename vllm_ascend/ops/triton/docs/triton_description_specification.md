> [!NOTE]
> **Filling Instructions**
>
> - Place the document under `vllm_ascend/ops/triton/docs`, named after the operator. If a same-named document already exists, add a suffix to distinguish it.
> - Fill in strictly following the template. For items that do not apply, use "N/A"; do not leave them blank or missing.

# Operator name

## Description

- **Function**:
- **Formula**:
- **Algorithm flow** (processed row by row, independently):
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950

## Parameters

> [!NOTE]
> All parameters are required.

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
>
> - Single-operator accuracy test cases should be placed under `tests/e2e/nightly/single_node/ops/singlecard_ops/triton`.
> - For inference scenarios, use the actual shapes and other parameters adopted by the model as single-operator test cases, rather than arbitrarily constructed ones.
> - Accuracy comparison results should use a unified precision tolerance based on the operator type and data type; example cases will be provided later.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_qkvzba_split_reshape_cat.py
```

## Example

A worked example of this template is committed alongside it in the same branch:

- **Operator doc**: `vllm_ascend/ops/triton/docs/fused_qkvzba_split_reshape.md`
- **Accuracy test**: `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_qkvzba_split_reshape_cat.py`

The example doc is filled section by section following this template; the example test uses the actual shapes adopted by the model in inference (Qwen3-GDN-10B, `gqa_interleaved_layout=True`) and applies the unified precision tolerance based on the operator type and data type (bit-exact for this pure data-movement operator).
