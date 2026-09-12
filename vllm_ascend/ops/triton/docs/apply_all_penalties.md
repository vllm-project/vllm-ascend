# apply_all_penalties

## Description

- **Location**:
`vllm_ascend/ops/triton/penalty.py` — `apply_all_penalties_kernel`, 1:1 wrapper `_apply_all_penalties_triton`. The production pipeline `apply_penalties_triton` first builds prompt/output histograms via `get_token_bin_counts_and_mask_triton` (see `token_bin_counts_and_mask.md`) and then calls this kernel. Sampler entry: `vllm_ascend/sample/penalties.py::apply_all_penalties`.
- **Function**: Applies repetition, frequency, and presence penalties to logits in place, given precomputed prompt/output presence masks and output token counts. It replaces vllm's `model_executor.layers.utils.apply_penalties` torch chain on NPU.
- **Formula** (per sequence `s`, per vocab index `t`, OpenAI-style; computed independently):
    - `seen = prompt_mask[s, t] ∨ output_mask[s, t]`
    - Repetition: `x' = x / r` if `seen ∧ x > 0`, `x' = x · r` if `seen ∧ x ≤ 0`, else `x' = x`, where `x = logits[s, t]` and `r = repetition_penalties[s]`
    - Frequency: `x'' = x' − frequency_penalties[s] · output_bin_counts[s, t]`
    - Presence: `logits[s, t] = x'' − presence_penalties[s] · 1[output_mask[s, t]]`
- **Algorithm flow** (processed row by row, independently):
    1. Grid `(min(num_seqs, num_vectorcore), 1, 1)`. Sequences are split evenly: `seqs_per_program = cdiv(num_seqs, num_programs)`; program `pid` owns `[pid · seqs_per_program, min((pid + 1) · seqs_per_program, num_seqs))`.
    2. Per sequence: load the three scalar penalties, then tile the vocab axis in `BLOCK_SIZE = 2048` chunks with a tail mask.
    3. Per tile: load logits / prompt mask / output mask / output counts, apply the three penalties in order (repetition scale, frequency subtract, presence subtract), and store logits in place.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used by the sampling path of `AscendSampler` / rejection sampler; works in eager mode (penalties run on the sampling path, not inside ACLGraph capture).

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits` | Input / Output | Logits `[num_seqs, vocab_size]`, updated in place | fp16 / bf16 / fp32 | ND |
| `prompt_mask` | Input | Prompt-token presence `[num_seqs, vocab_size]`, `True` where the token appeared in the prompt | bool | ND |
| `output_mask` | Input | Output-token presence `[num_seqs, vocab_size]`, `True` where the token appeared in the generated output | bool | ND |
| `output_bin_counts` | Input | Output-token histogram `[num_seqs, vocab_size]` | int32 | ND |
| `repetition_penalties` | Input | Per-sequence repetition penalty `[num_seqs]` (typically `≥ 1`) | fp32 | ND |
| `frequency_penalties` | Input | Per-sequence frequency penalty `[num_seqs]` | fp32 | ND |
| `presence_penalties` | Input | Per-sequence presence penalty `[num_seqs]` | fp32 | ND |
| `num_seqs` | Input (attribute) | Number of sequences; `do_not_specialize` so varying batch size does not recompile | int32 | scalar |
| `vocab_size` | Input (attribute) | Vocabulary size (local partition size when `enable_reduce_sample` is on) | int32 | scalar |
| `BLOCK_SIZE` | Input (attribute) | Vocab tile size, fixed at `2048` (constexpr) | int32 | scalar |

## Constraints

- `logits`, `prompt_mask`, `output_mask`, and `output_bin_counts` must share shape `[num_seqs, vocab_size]`. Penalty vectors must be length `num_seqs`.
- `output_bin_counts` is int32 (the dtype produced by `token_bin_counts_and_mask_kernel`); the kernel casts counts to fp32 before the frequency subtract. Logits arithmetic follows the logits dtype.
- `BLOCK_SIZE = 2048` is fixed; arbitrary `vocab_size` is handled by the tail mask. `num_seqs` is `do_not_specialize`; `BLOCK_SIZE` is constexpr.
- Grid is capped at the vector-core count. Requires `init_device_properties_triton()` so `get_vectorcore_num()` is valid. Unlike the bincount wrapper, this kernel does not read `AscendConfig`.
- `num_seqs == 0` is not a no-op: `_apply_all_penalties_triton` still issues a zero-sized launch. Callers must skip an empty batch. The histogram helpers return early on empty token tensors, so an empty batch only becomes a problem if this kernel is invoked directly.
- In-place update of `logits`. Only for inference (sampling penalties) on NPU; not captured into ACLGraph.

## Origin and Differences

- **Origin**: Migrated from vllm's `model_executor/layers/utils.apply_penalties` (OpenAI-style repetition / frequency / presence). Landed with the Triton-Ascend penalties kernels (#6979 / #7569).
- **Differences**:
    - NPU adaptation for performance: fuses the three penalty updates into one vector-core kernel over vocab tiles of `BLOCK_SIZE = 2048`; sequences are block-scheduled onto `min(num_seqs, num_vectorcore)` programs instead of a host-side torch loop over the vocab;
    - Modified for a specific vllm-ascend logic or different input parameters: consumes the int32 histograms / bool masks produced by `token_bin_counts_and_mask_kernel` (upstream rebuilds them with `scatter_add_` inside `apply_penalties`); the kernel itself does not take raw token ids.

## Test Cases

Dedicated kernel test compares `_apply_all_penalties_triton` against a PyTorch OpenAI-style penalty reference, feeding synthetic prompt/output masks and output bin counts (no bincount kernel). Covers block-aligned and tail vocab (`2048` / `5120` / `32000` / `151936`), fp16 / bf16 / fp32, and mask modes mixed / none / prompt-only / output-only. Unified elementwise tolerances: `rtol = atol = 1e-3` for fp16/fp32, `1e-2` for bf16.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_all_penalties_kernel.py
```
