# RecoverSSM Commit Planning

## Description

[`prepare_commit_plan`](../mamba/prepare_commit_plan.py#L96) translates sampler
acceptance and speculative metadata into a per-row commit plan. One Triton
program handles each speculative row. The wrapper derives the one-dimensional
grid and input strides. [GDN RecoverSSM commit](gdn_recoverssm_commit.md) consumes the plan
for state reconstruction and optional convolution compaction.

For a row with query length `q`, sampler acceptance `a`, and window capacity
`Q`, the accepted length is `c = min(max(a, 0), q, Q)`. Rows with a null
source page or zero acceptance get zero lengths and null destination indices.

Without align, the final destination is the source page. With align, let
`n` be the sampler row's prior computed length and `m` the block size.
The final column is `max(n+c-1, 0) // m`, capped at the last table column.
The next boundary is `(n // m + 1) * m`. When the accepted prefix reaches
it, the boundary page is the completed block's page. At an exact boundary,
the final and boundary page IDs coincide.

## Parameters

`B` denotes the number of speculative rows.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `num_accepted_tokens` | Input | Accepted recurrent steps, indexed by sampler row | Integer tensor | Strided 1D |
| `state_indices` | Input | Source checkpoint page per speculative row | INT32 or INT64 | Strided `[B]` |
| `query_start_loc` | Input | Cumulative query offsets | INT32 or INT64 | Strided `[B+1]` |
| `request_indices` | Input | Speculative-to-sampler row map, optional, default `None` for identity | Integer tensor or `None` | Strided 1D |
| `block_table` | Input | Sampler-row block table, optional, default `None` | INT32, INT64, or `None` | Strided 2D |
| `num_computed_tokens` | Input | Prior length per sampler row, required with a block table | Integer tensor or `None` | Strided 1D |
| `spec_query_len`, `null_block_id` | Attribute | Window capacity and null-page sentinel | Python int | Scalar |
| `mamba_block_size` | Attribute | Tokens per block, required in align mode | Python int or `None` | Scalar |
| `commit_lens` | Output | Full accepted lengths | INT32 in GDN | Contiguous vector |
| `state_commit_lens`, `boundary_commit_lens` | Output | Disjoint lengths for rows without and with a valid boundary page | INT32 in GDN | Contiguous vectors |
| `final_state_indices`, `boundary_state_indices` | Output | Final and boundary page IDs | INT32 in GDN | Contiguous vectors |
| `boundary_recovery_lens` | Output | Distance to the next boundary on live align rows, zero without align or on inactive rows | INT32 in GDN | Contiguous vector |

The wrapper returns `None` and writes all six output vectors in place.

## Constraints

The caller supplies same-device tensors, a nonempty batch, nondecreasing
offsets, valid sampler mappings and page IDs, and output capacity for at least
`B` rows. In align mode, the block table, prior lengths and block size must
be supplied together. The block size must be at least the window capacity,
and the table must cover any boundary column read by the planner. Use a
positive boundary page ID to determine whether to consume the boundary
recovery length. Output capacity beyond `B` is left untouched.

The GDN integration exercises this planner in eager and commit graph paths.
The standalone tests below run eagerly.

## Origin and Differences

The implementation was extracted from the GDN RecoverSSM Mamba helper. It
only writes the plan. The state and convolution operators consume that
plan and update their respective caches.

## Test Cases

[`test_prepare_commit_plan_npu.py`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_prepare_commit_plan_npu.py) directly calls
the wrapper and compares all six outputs against explicit integer expectations
with zero tolerance. The two NPU cases cover identity and strided sampler
mappings, oversized and negative acceptances, query and window clamping,
empty and null rows, crossing and exact boundaries, and untouched capacity.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_prepare_commit_plan_npu.py
```

## Example

With `n=6`, `m=8`, and `c=2`, final and boundary both use column 0,
with a boundary prefix of two steps. With `c=4`, column 0 still holds
the two-step boundary prefix, while column 1 holds the four-step final state.
