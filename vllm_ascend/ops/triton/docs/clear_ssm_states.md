# clear_ssm_states

## Description

- **Function**: Clears selected rows of an SSM recurrent-state tensor in place. Rows without an initial state are set to zero before GDN or KDA prefill; rows with an initial state are preserved exactly.
- **Formula**: Let `S[r]` be one flattened state row and `h[r]` indicate whether request `r` has an initial state:

  $$
  S'[r] =
  \begin{cases}
  S[r], & h[r] = \mathrm{True}, \\
  0, & h[r] = \mathrm{False}.
  \end{cases}
  $$

- **Algorithm flow** (processed row by row, independently):
  1. Return immediately for an empty state tensor. Move `has_initial_state` to the state device when needed, convert it to `bool`, flatten it, and validate that it contains one value per state row.
  2. Flatten all dimensions after the first into `inner_size`. Launch a two-dimensional grid `(num_rows, ceil(inner_size / 4096))`.
  3. Map `program_id(0)` to one state row and `program_id(1)` to one 4096-element column tile.
  4. Preserve the row when `has_initial_state[row]` is true. Otherwise, store zeros using a tail mask for the final column tile.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. It is used by both the GDN and KDA prefill paths; the current single-operator test has been verified on Atlas A2.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `ssm_states` | Input/Output | Recurrent states whose first dimension enumerates request/state rows; modified in place. Production layouts include GDN `[num_rows, num_v_heads, head_v_dim, head_k_dim]` and KDA `[num_rows, num_heads, value_dim, key_dim]` states | fp16 / bf16 / fp32 | Contiguous ND, `[num_rows, ...]` |
| `has_initial_state` | Input | One flag per state row; false selects the row for clearing | bool, or a tensor convertible to bool | Contiguous 1-D after wrapper reshape, `[num_rows]` |

## Constraints

- For non-empty `ssm_states`, `has_initial_state.numel()` must equal `ssm_states.shape[0]`; otherwise the wrapper raises `ValueError`.
- The dimensions after the first must be dense because the kernel flattens them and only uses `ssm_states.stride(0)` for the row base address.
- `has_initial_state` may initially be on another device or use another dtype, but the wrapper converts it to a contiguous boolean tensor on the state device.
- Empty tensors and zero-sized inner states are accepted and return without launching the kernel.
- The operation mutates `ssm_states` and returns `None`. The graph-capture path should provide an already device-local boolean mask to avoid inserting conversion work during capture.

## Origin and Differences

- **Origin**: Added in vLLM-Ascend PR #7967 to remove host-device synchronization while preparing Qwen3-Next and Qwen3.5 prefill states. The surrounding utility module contains code derived from flash-linear-attention, while this device-side selective-clear path is a vLLM-Ascend adaptation.
- **Differences**:
    - NPU adaptation for performance: replaces host-side row selection and clearing with a two-dimensional vector-core Triton launch and 4096-element tiles;
    - Modified for a specific vllm-ascend logic or different input parameters: accepts the per-request `has_initial_state` metadata used by both `AscendGatedDeltaNetAttention` and the KDA recurrent-state path, and normalizes its device, dtype, and shape before launch.

## Test Cases

The test compares the complete in-place state tensor with a PyTorch boolean-indexing reference. It covers bf16 state shapes `(6, 3, 5, 7)` and `(4, 5, 25, 41)`, mixed true/false masks, preserved rows, cleared rows, and non-aligned inner sizes. The kernel only copies existing values or writes zero, while the current test invokes `torch.testing.assert_close` with its default bf16 tolerance rather than explicitly requiring `rtol=0, atol=0`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_clear_ssm_states.py
```
