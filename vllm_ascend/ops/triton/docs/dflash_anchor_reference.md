# dflash_anchor_reference

## Description

- **Function**: Provides the graph-safe reference implementation for anchor-union activation routing. Anchor rows select `route_top_k` experts; suffix rows contribute `suffix_pool_top_k` candidates to the batch union and are masked to that union before the final route selection.
- **Formula**: For each row, select the top-`route_top_k` routing scores. The union contains all selected experts from role 1 rows and the first `suffix_pool_top_k` selected experts from role 2 rows. Role 2 scores outside the union are replaced with the dtype minimum.
- **Algorithm flow**: Compute row-wise top-k indices, build a boolean expert union from the role limits, mask suffix rows, then perform the model's final top-k and optional weight renormalization.
- **Supported modes**: Atlas A2, Atlas A3, and Atlas A5. The wrapper also provides a CPU reference path for unit tests and fallback validation.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| logits | Input/Output | Router scores. NPU inputs are made contiguous before the Triton launch. | FP16/BF16/FP32 | 2D [num_tokens, num_experts] |
| roles | Input | 0 padding/prefill, 1 anchor, 2 suffix. | int32/int64 | 1D [num_tokens] |
| mask | Workspace | Persistent expert-union mask. | bool | 1D [num_experts] |
| top_k | Attribute | Number of experts selected by an anchor row. | int | scalar |
| suffix_pool_top_k | Attribute | Number of candidates contributed by each suffix row. | int | scalar |

## Constraints

- roles must be contiguous, on the same device as logits, and have one entry per row.
- mask must be a boolean tensor on the logits device with one entry per expert.
- mask is a persistent workspace and is cleared and rebuilt by the NPU path.
- Padding and prefill rows do not contribute candidates or receive suffix masking.
- Contiguous NPU logits are masked in place; non-contiguous logits are copied before masking and the returned tensor owns the contiguous result. The CPU reference path returns a masked tensor without mutating its input.

## Origin and Differences

- **Origin**: Derived from the DFlash/DSpark anchor-union routing rule and retained as the exact fallback/reference implementation.
- **Differences**: The NPU path uses two Triton kernels and persistent masks for graph replay; non-contiguous logits are copied once before launch.

## Accuracy Validation

The tests compare the fused route with `route_reference` using the following criteria:

- Expert IDs are compared exactly (`rtol=0`, `atol=0`).
- Qwen fused BF16 weights use `rtol=2**-6`, `atol=2**-6`.
- D-Spark FP32 weights use `rtol=2**-16`, `atol=2**-10`, with a `0.01` maximum absolute error limit.
- Qwen BF16 weights use a `1.0` maximum absolute error limit.
- Each case prints `max_abs_error`, `max_rel_error`, and the percentage of elements satisfying `atol + rtol * abs(reference)`.

Run with `-s` to keep the precision metrics in the test log:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dflash_anchor_union.py
```

## Test Cases

```bash
pytest -sv tests/ut/test_dflash_anchor_state.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dflash_anchor_union.py
```
