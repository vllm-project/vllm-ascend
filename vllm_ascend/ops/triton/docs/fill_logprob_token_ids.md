# fill_logprob_token_ids

## Description

- **Function.** `fill_logprob_token_ids` builds the logprob token ID matrix and its
  validity mask on Ascend NPUs. MRV2 `compute_topk_scores` uses it when at least
  one request specifies custom `logprob_token_ids` (`max_per_req_token_ids > 0`).
  The kernel implements the same behavior as upstream vLLM with an Ascend-specific
  adjustment for dynamic pointers. See [Origin and Differences](#origin-and-differences).

- **Output layout.** Both matrices have shape `[batch_size, 1 + num_cols]`, where
  `num_cols = max(num_logprobs, max_per_req_token_ids)`.

    - **Column 0** contains the sampled token ID (`sampled_token_ids[b]`) and is
      always valid.
    - **Columns 1 onward** contain the custom token IDs if the request has any
      (`num_custom > 0`), or the row's top-k token IDs otherwise.
      `expanded_idx_mapping` maps each row to its request state index.
      Only populated entries are marked `True` in the mask.

- **Formula.** Each row is processed independently. Let `b` be the batch row index.

    - Column 0 is always written and marked valid.

        $$
        out\_token\_ids[b, 0] = sampled\_token\_ids[b],
        \qquad out\_valid\_mask[b, 0] = 1
        $$

    - Map the row to its request state.

        $$
        req\_state\_idx = expanded\_idx\_mapping[b],
        \qquad num\_custom = num\_per\_req\_token\_ids[req\_state\_idx]
        $$

    - Determine validity and the token source for column `1 + col`, with
      `0 <= col < PADDED_COLS`.

        - When `num_custom > 0`, the custom IDs replace the top-k candidates.

            $$
            valid = col < num\_custom,
            \qquad token = per\_req\_token\_ids[req\_state\_idx, col]
            $$

        - Otherwise, use the top-k indices. When `NUM_TOPK == 0`, all candidate
          slots in this row remain invalid.

            $$
            valid = col < NUM\_TOPK,
            \qquad token = topk\_indices[b, col]
            $$

    - For columns in the output, the initialized buffers and masked stores give
      the following values.

        $$
        out\_token\_ids[b, 1 + col] = token \cdot \mathbb{1}[valid],
        \qquad out\_valid\_mask[b, 1 + col] = \mathbb{1}[valid]
        $$

    `PADDED_COLS = next_power_of_2(num_cols)` sets the total scan width. Each
    iteration processes at most `min(PADDED_COLS, COL_BLOCK_SIZE)` columns.
    The validity mask prevents stores beyond `num_cols`.

- **Algorithm flow.** The kernel uses a single pass, with `tl.program_id(0)`
  selecting the batch row.

    1. Load the sampled token ID, store it in column 0, and mark it valid.
    2. Use `expanded_idx_mapping` to find the request state, then read its custom
       token count `num_custom`.
    3. Loop over blocks of up to 4096 columns. Use `num_custom` to select the
       candidate source and form the validity mask for each block.
    4. Store the valid IDs as INT64 in `[b, 1 : 1 + num_cols]` and set the
       corresponding entries in `out_valid_mask` to 1.

## Origin and Differences

- The upstream vLLM kernel selects `src` and `valid` in an `if / else` and
  issues a single `tl.load(src + col, ...)` after the branches.
- Ascend issues `tl.load` inside each branch to work around compilation and
  execution issues with pointers selected at runtime. The output semantics
  match upstream.
- The kernel processes candidate columns in blocks to limit Unified Buffer (UB)
  use. Blocks of 8192 columns overflowed UB in the A5 large-candidate tests.
  Blocks of 4096 passed on both A3 and A5, so `COL_BLOCK_SIZE` defaults to 4096.
- The Python entry point allocates outputs, computes the launch grid, and chooses
  `PADDED_COLS`. The kernel keeps the launch arguments used by the upstream patch.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| out_token_ids_ptr | Output | Token IDs of shape `[batch_size, 1 + num_cols]`. Column 0 contains the sampled token. Subsequent columns contain custom or top-k IDs. Unfilled entries retain their initial value, typically 0. | INT64 | ND |
| out_token_ids_stride | Attribute | Row stride of `out_token_ids`, given by `stride(0)`. | INT | - |
| out_valid_mask_ptr | Output | Validity mask of shape `[batch_size, 1 + num_cols]`. Column 0 is always `True`, and each other populated entry is `True`. | BOOL | ND |
| out_valid_mask_stride | Attribute | Row stride of `out_valid_mask`. | INT | - |
| sampled_token_ids_ptr | Input | Sampled token IDs of shape `[batch_size]`. | INT64 | ND |
| topk_indices_ptr | Input | Top-k token IDs of shape `[batch_size, NUM_TOPK]`. The caller casts the top-k indices from the logits to INT32. When `NUM_TOPK == 0`, this parameter serves as an unread placeholder pointer. | INT32 | ND |
| topk_indices_stride | Attribute | Row stride of `topk_indices`. | INT | - |
| expanded_idx_mapping_ptr | Input | Mapping of shape `[batch_size]` from batch row to request state index, used to index `num_per_req_token_ids` and `per_req_token_ids`. | INT32 | ND |
| num_per_req_token_ids_ptr | Input | Custom token count per request, of shape `[max_num_reqs]`. | INT32 | ND |
| per_req_token_ids_ptr | Input | Custom token IDs of shape `[max_num_reqs, MAX_LOGPROB_TOKEN_IDS]`, stored from column 0 in each row. | INT32 | ND |
| per_req_token_ids_stride | Attribute | Row stride of `per_req_token_ids`. | INT | - |
| NUM_TOPK | Compile-time attribute | Number of top-k candidate columns, typically equal to `num_logprobs`. | INT | - |
| PADDED_COLS | Compile-time attribute | Scan width, given by `next_power_of_2(num_cols)` with `num_cols = max(num_logprobs, max_per_req_token_ids)`. | INT | - |
| COL_BLOCK_SIZE | Compile-time attribute | Maximum candidate columns processed per iteration. Defaults to 4096. The actual width is capped at `PADDED_COLS`. | INT | - |

## Constraints

- The Python entry point expects a nonempty batch and room for at least one
  custom or default candidate column. Pass `topk_indices=None` when there
  are no default candidates. All inputs must reside on the same NPU. Vectors
  must be contiguous, and matrices must have a column stride of 1.
- The caller supplies `max_per_req_token_ids`, which must cover the custom
  candidate count of every request in the batch. Each count must fit in the
  columns of `per_req_token_ids`. Every valid ID must be within the vocabulary.
- The Python entry point returns newly allocated INT64 IDs and a BOOL mask.
  Unfilled entries remain 0 and `False`, respectively. The kernel only writes
  to the output buffers allocated by the caller.
- `out_token_ids` must be a two-dimensional INT64 tensor with at least
  `1 + num_cols` columns. The wrapper uses `sampled_token_ids.new_zeros` to
  preserve the input ID dtype.
- `out_valid_mask` must be a two-dimensional BOOL tensor with the same shape as
  `out_token_ids`.
- `sampled_token_ids` must be a one-dimensional INT64 tensor of length
  `batch_size`.
- When supplied, `topk_indices` must be a two-dimensional INT32 tensor of shape
  `[batch_size, NUM_TOPK]`.
- `expanded_idx_mapping` must be a one-dimensional INT32 tensor of length
  `batch_size`. Each element must be a valid request index in `[0, max_num_reqs)`.
- `num_per_req_token_ids` must be a one-dimensional INT32 tensor of length
  `max_num_reqs`.
- `per_req_token_ids` must be a two-dimensional INT32 tensor of shape
  `[max_num_reqs, MAX_LOGPROB_TOKEN_IDS]`. The value of
  `MAX_LOGPROB_TOKEN_IDS` is 128, matching the constant in `vllm.sampling_params`.
- `PADDED_COLS` must be a power of two at least as large as `num_cols`.
  `next_power_of_2` ensures this, and the validity mask bounds stores to the
  actual columns.
- Both `num_custom` and `NUM_TOPK` must be at most `num_cols`. The definition
  `num_cols = max(num_logprobs, max_per_req_token_ids)` ensures these bounds.
- Each program processes one batch row. The launch grid is `(batch_size,)`.
- The kernel copies IDs and writes masks. Both outputs must match the CPU
  reference exactly, as checked with `torch.equal`.

## JIT Specialization

- A compile-time `NUM_TOPK` eliminates loads from `topk_indices` when there
  are zero default candidates. Different top-k counts can produce different
  specializations within the same `PADDED_COLS` bucket.
- `PADDED_COLS` rounds the scan limit up to a power of two. The production launch
  uses `COL_BLOCK_SIZE = 4096`, so `min(PADDED_COLS, COL_BLOCK_SIZE)` sets the
  width of each vector load.
- The launch grid follows the batch size. Tensor dtypes, row strides, and
  alignment can also affect Triton specialization.

## Usage Example

```python
import torch

from vllm_ascend.ops.triton.sample.fill_logprob_token_idx import fill_logprob_token_ids

device = "npu"
batch_size = 4
num_reqs = 4
max_per_req_token_ids = 3
num_logprobs = 5
MAX_LOGPROB_TOKEN_IDS = 128

sampled_token_ids = torch.randint(0, 1000, (batch_size,), dtype=torch.int64, device=device)
topk_indices = torch.randint(0, 1000, (batch_size, num_logprobs), dtype=torch.int32, device=device)

# Map each batch row to a request state. Request 0 has custom token IDs.
expanded_idx_mapping = torch.arange(batch_size, dtype=torch.int32, device=device) % num_reqs
num_per_req_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=device)
per_req_token_ids = torch.zeros(num_reqs, MAX_LOGPROB_TOKEN_IDS, dtype=torch.int32, device=device)

# Request 0 asks for scores for three custom token IDs.
num_per_req_token_ids[0] = 3
per_req_token_ids[0, 0] = 100
per_req_token_ids[0, 1] = 200
per_req_token_ids[0, 2] = 300

out_token_ids, out_valid_mask = fill_logprob_token_ids(
    sampled_token_ids,
    topk_indices,
    expanded_idx_mapping,
    num_per_req_token_ids,
    per_req_token_ids,
    max_per_req_token_ids,
)
torch.npu.synchronize()
```

## Test Cases

The kernel tests compare token IDs and validity masks element by element against a
PyTorch CPU reference and support A2, A3, and A5. Large candidate sets on the
production path were verified on A3 and A5. Through the MRV2 entry point,
`test_compute_topk_scores.py` covers custom candidates, batches mixing default
and custom requests, full-vocabulary candidates, and both logits and logprob modes.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fill_logprob_token_ids_kernel.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_scores.py
```
