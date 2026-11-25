# GDN RecoverSSM Commit

## Description

- **Function**: Reconstruct accepted GDN recurrent checkpoints across bound
  layers using a prepared plan and FP32 replay records from
  [verify](gdn_recoverssm_verify.md).
- **Formula**: Apply the accepted-prefix reconstruction below.
- **Algorithm flow**: Each program reads source records in reverse token
  order and updates its tile of a final checkpoint and, in align mode, an
  optional boundary checkpoint.
- **Supported modes**: Eager commit and graph replay with Ascend 950 and
  ordinary non-950 launch paths. See the historical results below for tested
  configurations.

### Accepted-prefix reconstruction

For accepted length `c`, state reconstruction uses reverse accumulation:

```text
F = 1; R = 0
for t = c - 1 down to 0:
    R += u[t][:, None] * (k_hat[t] * F)[None, :]
    F *= d[t]
S_accepted = S_checkpoint * F + R
```

This is algebraically the forward recurrence, but changes floating-point
evaluation order. BF16 checkpoint storage rounds the reconstructed FP32 result.

Each record contains correction `u`, effective key `k_hat`, and scalar decay
`d`. The caller prepares accepted lengths and page IDs with
[prepare_commit_plan](prepare_commit_plan.md). The
[framework context](../../../attention/recoverssm_gdn.py#L264) can also invoke
[compact_conv_state](compact_conv_state.md) for convolution history.

### Kernel launch

The flat grid has `ceil(V / BV) * L * head_groups * B` programs.

Ordinary state commit processes one head per program. The Ascend 950 grouped path
handles two adjacent heads when `K=V=128` and `HV` is even and at least two. Unlike
the KDA selection policy, this GDN selection is not restricted to FP32 checkpoints
or `HV=32/64`. The ordinary non-950 path uses a runtime reverse loop; the 950 path
can statically unroll the window.

State value tiles are limited by the reported UB capacity. The budget uses two
matrix buffers without align and six with align, FP32 element size, key tile
width, the grouped head factor, and two rows of headroom. This is a launch-sizing
heuristic, not a measurement of actual compiler allocation or occupancy.

Although the shared planner produces state-only and boundary length arrays, the
current GDN launcher uses the full accepted-length array for a single state
commit launch. It does not use KDA's two disjoint filtered state launches.

## Parameters

`B` is the batch size, `P` the page count, `Q` the window capacity, `HV`
the local value-head count, `K/V` the key/value widths, and `L` the bound
layer count. The wrapper returns `None` after scheduling the in-place update.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `state` | Input/Output | Reference checkpoint view. The kernel updates the destination checkpoints addressed by the table | BF16 or FP32 | Paged `[P, HV, V, K]` |
| `replay` | Input | Reference replay view for the bound layers | FP32 | Paged `[P, HV, Q, V + K + 1]` |
| `state_commit_cache_address_stride_table` | Input | Checkpoint and replay base addresses and page strides per layer | INT64 | Contiguous `[2, 2, L]` |
| `state_indices` | Input | Source page IDs | INT32 or INT64 | Strided `[B]` |
| `commit_lens`, `final_state_indices` | Input | Full accepted length and final page per row | INT32 | Contiguous plan vectors |
| `boundary_state_indices`, `boundary_recovery_lens` | Input | Boundary page and its accepted length per row | INT32 | Contiguous plan vectors |
| `spec_query_len` | Attribute | Replay capacity `Q` | Python int | Positive scalar |
| `align_mode` | Attribute | Enables boundary checkpoint recovery | Python bool | Scalar |

## Constraints

- All input tensors and bound layers share a device. The caller supplies
  matching cache shapes, corresponding dtypes, and state/replay inner
  strides across layers. Each layer may have its own page strides.
- Provide a prepared plan with in-range page IDs and prefix lengths within
  `Q`. A nonpositive source page ID skips the row. The caller handles empty
  batches and prevents conflicts between requests accessing the same pages.
- The source checkpoint must agree with the replay records. Keep cache and
  metadata addresses stable during execution and graph replay, and initialize
  the Triton device properties before capture. Autograd is N/A.

## Origin and Differences

- **Origin**: The accepted-prefix contract follows
  [upstream vLLM GDN RecoverSSM](https://github.com/vllm-project/vllm/pull/61309).
- **Differences**: The Ascend kernels use explicit cache strides, UB-bounded
  value tiles, and a two-head grouped path on Ascend 950. The GDN launcher can
  recover the final and boundary checkpoints in a single kernel launch.

## Test Cases

### Reference implementations

- [cpu_reference](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/gdn_recoverssm_test_utils.py#L119) builds replay records and expected checkpoints with a forward
  FP64 recurrence. The direct commit tests supply those records as FP32.
- The small-norm tests also compare the accepted checkpoints against the
  one-token AscendC recurrence using the same pre-normalized BF16 Q/K.
- The align test uses zero key and correction and a decay of `0.5`. The
  expected state after `c` steps is `S_initial * 0.5**c`.

### Accuracy criteria

[check_accuracy](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/gdn_recoverssm_test_utils.py#L31) checks shape and finiteness, then computes error on
CPU in FP64. Tolerances use the actual output's storage dtype.

| Output dtype | rtol | atol | Fixed floor for the maximum absolute error limit |
| --- | --- | --- | --- |
| FP16 | `2^-9` | `2^-9` | `0.1` |
| BF16 | `2^-6` | `2^-6` | `1.0` |
| FP32 | `2^-10` | `2^-16` | `0.01` |

Each arithmetic comparison requires at least 99% of elements to satisfy
`abs(actual - expected) <= atol + rtol * abs(expected)`, and the maximum
absolute error must be at most
`max(floor, 32 * torch.finfo(actual.dtype).eps)`. The shared checker also covers FP16 in its self-test.
Padding and cache regions expected to stay unchanged use exact checks.
The checker records per-comparison metrics as `accuracy/<label>` JUnit properties.

### Coverage

| Test function | Cases | Verification performed |
| --- | ---: | --- |
| [test_commit_preserves_head_padding](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_commit_npu.py#L60) | 1 | Padded head strides, strided source indices, and exact checks on replay and untouched checkpoint storage. |
| [test_commit_ragged_tiles_and_small_norms](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_commit_npu.py#L90) | 6 | Ragged K/V tiles and accepted prefixes of one step and the full window. |
| [test_commit_small_norms_against_ascendc](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_commit_npu.py#L117) | 4 | Accepted prefixes of 1, 3, and 5 steps across four input magnitudes. |
| [test_commit_null_rows](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_commit_npu.py#L131) | 2 | Page IDs 0 and -1 leave the cache backing unchanged. |
| [test_commit_wrapper_writes_boundary_and_final_states](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_commit_npu.py#L143) | 2 | Explicit align plan with distinct or coincident final and boundary pages. |
| [test_commit_configured_window_dispatch](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_commit_npu.py#L174) | 6 | Windows 1/2/4/5/9, FP32 and BF16 checkpoints, ordinary/grouped dispatch, grid and UB sizing, eager or graph commit. |

The 21 parameterized cases include Qwen3.6-35B-A3B TP4 (`HK/HV=4/8`, `Q=5`)
and Qwen3.8-27B TP2 (`HK/HV=8/24`, `Q=9`), both with `K=V=128`.
The separate [framework tests](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_recoverssm_context_npu.py)
cover multiple layers, cache binding, sampler mapping, plan generation,
convolution compaction, and verify followed by commit.

### Test command

Run from the repository root with the Ascend NPU test stack.

```bash
pytest -sv -o junit_family=xunit1 \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_commit_npu.py \
  --junitxml=gdn_recoverssm_commit.xml
```

### Recorded NPU results

These combined-suite reports predate the verify/commit test split. The 39 cases included four
convolution-compaction cases now maintained in
[`test_compact_conv_state_npu.py`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compact_conv_state_npu.py).
The standalone planner cases were added after that run.

The JUnit report recorded **6 passed**, **0 failed**, **0 errors**, and **0 skipped** in
**44.182 s**. The named test container used
an Ascend A3 OpenEuler image with CANN 9.2.1.

The JUnit report recorded **6 passed**, **0 failed**, **0 errors**, and **0 skipped** in
**22.357s**. The named test container used
an Ascend A5 OpenEuler image with CANN 9.2.1.

| Output dtype | Operator comparisons | Minimum mixed-tolerance pass ratio | Maximum absolute error | Absolute error limit |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 45 | 100% | `3.3789673e-3` | `1.0` |
| FP32 | 246 | 100% | `3.1493371e-7` | `0.01` |

## Example

For source page 1, accepted length 4, final page 3, boundary page 2, and
boundary recovery length 2, align-mode commit writes the state after two
steps to page 2 and the state after four steps to page 3. In the decay-only
test with initial state 8 and decay `0.5`, those pages receive 2 and 0.5
respectively. Source page 1 remains unchanged.
