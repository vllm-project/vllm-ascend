# mamba_align_block_table

## Description

- **Location**: [`mamba/align_block_table.py`](../mamba/align_block_table.py).
  `mamba_get_block_table_tensor` is the Python entry point for the private
  `_mamba_align_block_table_kernel` Triton kernel.
- **Function**: In `mamba_cache_mode="align"`, return the trailing
  `1 + num_speculative_blocks` physical state-block IDs of every request from
  the full position-indexed mamba block table, so the downstream state kernels
  know which physical blocks to read and write for speculative-decode
  alignment. One launch replaces the seven-op torch expression the upstream
  expression expands to.
- **Formula**: For request `i`, `start_i = max((seq_lens[i] - 1) //
  block_size, 0)` and `out[i, j] = block_table[i, start_i + j]` for
  `j` in `[0, 1 + num_speculative_blocks)`.
- **Algorithm flow** (processed row by row, independently): one program per
  request; `BLOCK = next_power_of_2(n_out)` lanes with a mask gather
  `n_out` consecutive entries starting at the request's anchor column.
- **Supported modes**: align-mode mamba/GDN/KDA state caches on Ascend NPU
  (A2/A3 verified). Called from host-side attention metadata builders
  (`ops/gdn_attn_builder.py`, `models/glm5next/recoverssm_metadata.py`)
  every step, outside any captured graph. `"none"` mode returns the block
  table unchanged. The kernel path requires NPU tensors.
- **Compilation**: `block_size` and `n_out` are session-constant, so the
  kernel compiles once per process. `do_not_specialize_on_alignment` on all
  three pointers keeps row-sliced caller views from JIT-recompiling mid-serve.

## Parameters

The wrapper allocates and returns `out`; all other arguments are inputs.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `block_table` | Input | Full position-indexed mamba state block table, `[R, cdiv(max_model_len, block_size)]` | INT32 | 2D, strided |
| `seq_lens` | Input | Current sequence lengths, `[R]` | INT32 | 1D, strided |
| `kv_cache_spec` | Attribute | `MambaSpec` providing `block_size` and `num_speculative_blocks` | `MambaSpec` | N/A |
| `mamba_cache_mode` | Attribute | `"align"` takes the kernel; `"none"` returns `block_table` unchanged | `str` | N/A |
| `out` (return) | Output | Trailing-window physical block IDs, `[R, 1 + num_speculative_blocks]` | INT32 | Contiguous 2D, freshly allocated |

## Constraints

- `seq_lens` and `block_table` must have the same row count `R` and reside on
  the same device (asserted).
- The kernel path requires NPU tensors. The align-mode cases of the builder
  unit tests are gated on NPU availability (`tests/ut` runners have none).
- `block_table` may be a row-sliced view (`block_table[:num_reqs]`); both
  strides are honored, including a column stride other than 1.
- `block_table_stride_0` is annotated `tl.int64`: without it the row-offset
  product wraps in i32 once a row offset reaches `2^31` elements, a limit the
  `torch.gather` it replaces does not have.
- `R == 0` returns an empty `[0, n_out]` tensor without launching a `(0,)`
  grid. Zero-length requests clamp to row 0. Reads past the request's used
  blocks but inside the row are legal and return unused entries, exactly as
  the replaced expression did.
- The output is bit-exact with the seven-op expression (`atol=0, rtol=0`).

## Origin and Differences

- **Origin**: Port of vLLM [#58737](https://github.com/vllm-project/vllm/pull/58737)'s
  CUDA fused kernel; replaces the "align" branch of vLLM's
  `mamba_get_block_table_tensor`, which on NPU always expanded to seven eager
  kernel launches because upstream gates the fused path with `is_cuda`.
- **Differences**:
    - Built for triton-ascend; keeps both load-bearing annotations from
      upstream (`tl.int64` row stride, `do_not_specialize_on_alignment`).
    - NPU-only: vllm-ascend runs on NPU, so no non-NPU fallback is carried;
      the align-mode builder unit-test cases skip on CPU runners instead.
    - The wrapper derives `num_reqs` from the block table and asserts row and
      device equality, mirroring the upstream contract.

## Test Cases

Per the repo test organization, `tests/ut` runs on CPU runners, so the split is:

- `tests/ut/ops/test_mamba_align_block_table.py` — the `"none"` passthrough
  (the only CPU-reachable behavior); hosts the shared builders (`_spec`,
  `_reference`, `_make`).

  ```bash
  pytest -sv tests/ut/ops/test_mamba_align_block_table.py
  ```

- `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_align_block_table.py`
  — on-device kernel equivalence with `rtol=0, atol=0`: parametrized request
  counts, block sizes and speculative depths, a non-contiguous block table
  (column stride 2), row-sliced views, the `seq_len == 0` clamp and the empty
  batch, reusing the UT's builders.

  ```bash
  pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_align_block_table.py
  ```

Probe and engine-level performance measurements are recorded in
[PR #21196](https://github.com/vllm-project/vllm-ascend/pull/21196).
