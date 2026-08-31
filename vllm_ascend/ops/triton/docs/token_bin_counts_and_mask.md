# token_bin_counts_and_mask

## Description

- **Location**:
`vllm_ascend/ops/triton/bincount.py` — `token_bin_counts_and_mask_kernel`, 1:1 wrapper `get_token_bin_counts_and_mask_triton`
- **Function**: Counts token occurrences per batch row over a (possibly TP-sharded) vocabulary window and returns both the histogram and a presence mask. It replaces vllm's `torch.scatter_add_` implementation of `get_token_bin_counts_and_mask` on NPU. Production entry: `apply_penalties_triton` (`vllm_ascend/ops/triton/penalty.py`) → `get_token_bin_counts_and_mask_triton` (once for prompt tokens, once for output tokens) → `token_bin_counts_and_mask_kernel`.
- **Formula** (per sequence `b`, per local vocab index `t ∈ [0, vocab_size)`):
    - `vocab_start_idx = tp_rank · vocab_size` (`tp_rank = 0` when `enable_reduce_sample` is off)
    - `bin_counts[b, t] = Σ_p 1[tokens[b, p] = vocab_start_idx + t]` for `p ∈ [0, seq_len)`
    - `mask[b, t] = bin_counts[b, t] > 0`
    - Positions whose token is outside `[vocab_start_idx, vocab_start_idx + vocab_size)` are ignored. Callers pad with `vocab_size` (not `-1`) so padding falls outside the counted window.
- **Algorithm flow** (processed per `(batch, seq_block)` work item, independently):
    1. Grid `(min(num_vectorcore, total_blocks),)` with `total_blocks = num_seqs · cdiv(seq_len, SEQ_BLOCK)` and `SEQ_BLOCK = 256`. Each program grid-stride-loops over linear blocks `pid, pid + num_programs, …` so the program count stays within the Triton-Ascend `coreDim` limit of 65535 while still covering every `(batch, seq_block)`.
    2. Decode `linear_block` into `(batch_idx, seq_block_id)`, load `SEQ_BLOCK` tokens of that row with a tail mask (`other = vocab_size + vocab_start_idx`, which is out of range and therefore ignored).
    3. Map to the local vocab: `local_token = token - vocab_start_idx`; keep only in-range, in-bounds tokens (`token_in_range`). Out-of-range indices are rewritten to 0 before the pointer is formed so the address is always in-bounds.
    4. `tl.atomic_add(bin_counts[batch_idx, local_token], 1, mask=token_in_range)`. The wrapper then returns `bin_counts` and `bin_counts > 0`.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used by the sampling-penalties path of `apply_penalties_triton`; works in eager mode (penalties run on the sampling path, not inside ACLGraph capture).

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `tokens` | Input | Token IDs `[num_seqs, seq_len]`. Padding value must be `vocab_size` and is ignored. Made contiguous by the wrapper | int64 / int32 | ND |
| `vocab_size` | Input (attribute) | Vocabulary size of the counted window (local vocab size when `enable_reduce_sample` is on) | int32 | scalar |
| `num_seqs` | Input (attribute) | Optional row count; when provided and `> 0`, asserts `tokens.shape[0] == num_seqs` | int32 / None | scalar |
| `tp_rank` | Input (attribute) | Tensor-parallel rank used to form `vocab_start_idx`; the wrapper sets this to `get_tp_group().rank_in_group` when `enable_reduce_sample` is on, otherwise `0` | int32 | scalar |
| `SEQ_BLOCK` | Input (attribute) | Sequence tile size, fixed at `256` (constexpr) | int32 | scalar |
| `bin_counts` | Output | Per-row histogram `[num_seqs, vocab_size]`; first element of the returned tuple | int32 | ND |
| `mask` | Output | Presence mask `[num_seqs, vocab_size]`, `True` where `bin_counts > 0`; second element of the returned tuple | bool | ND |

## Constraints

- `tokens`: 2D `[num_seqs, seq_len]`. Non-contiguous inputs are copied; `seq_len == 0` or `num_seqs == 0` returns zero histograms without launching the kernel.
- Padding contract: token ids `>= vocab_size + vocab_start_idx` or `< vocab_start_idx` are ignored. Callers must pad with `vocab_size`, not `-1`.
- `bin_counts` is allocated as `int32 [num_seqs, vocab_size]` (upstream uses `int64` and an extra `vocab_size + 1` padding bin, then slices it off). The mask is derived on the host as `bin_counts > 0`.
- `SEQ_BLOCK = 256` is fixed; arbitrary `seq_len` is handled by the tail mask. `tokens_batch_stride`, `batch_size`, `seq_len`, and `total_blocks` are `do_not_specialize`, so varying batch/sequence lengths do not trigger Triton recompilation.
- Grid is capped at the vector-core count (`min(num_vectorcore, total_blocks)`). Requires `init_device_properties_triton()` so `get_vectorcore_num()` is valid, and an initialized `AscendConfig` so the `enable_reduce_sample` branch can be read. Kernel is launched with `multibuffer=False`.
- When `enable_reduce_sample` is on, only the TP-local window `[tp_rank · vocab_size, (tp_rank + 1) · vocab_size)` is counted; `vocab_size` passed in is the local partition size.
- Integer histogram: comparison against a reference must be bit-exact (`torch.equal`). Only for inference (sampling penalties) on NPU; not captured into ACLGraph.

## Origin and Differences

- **Origin**: Migrated from vllm's `model_executor/layers/utils.get_token_bin_counts_and_mask` (`scatter_add_` over `vocab_size + 1` then slice). Landed with the Triton-Ascend penalties kernels (#6979 / #7569).
- **Differences**:
    - NPU adaptation for performance: replaces `scatter_add_` with a vector-core Triton kernel using `tl.atomic_add`; 1D grid-stride loop over `(batch, seq_block)` tiles keeps the program count at the vector-core count and within the 65535 `coreDim` limit; sequence tiled by `SEQ_BLOCK = 256`; counts stored as int32 rather than int64;
    - Modified for a specific vllm-ascend logic or different input parameters: when `enable_reduce_sample` is on, counts only the TP-local vocab window via `vocab_start_idx = tp_rank · vocab_size`; padding is filtered by an in-range mask instead of an extra `vocab_size + 1` bin.

## Test Cases

Covered by the penalties pipeline test, which compares `vllm_ascend.sample.penalties.apply_all_penalties` (this kernel plus `apply_all_penalties_kernel`) against vllm's `apply_all_penalties`. Shapes follow the penalties inference path (`num_seqs = 1/8/32/128`, Qwen-style `vocab_size = 151936` and `5120`, prompt/output lengths including empty and all-padding). Unified elementwise tolerances: `rtol = atol = 1e-3` for fp16, `1e-2` for bf16.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_penalties_triton.py
```
