# prepare_gdn

## Description

- **Location**: [`v2/mamba/gdn_prepare.py`](../v2/mamba/gdn_prepare.py).
  `prepare_gdn` is the Python entry point for the private
  `_prepare_gdn_kernel` Triton kernel.
- **Function**: Build aligned-cache GDN metadata and clear the shared
  convolution output in one launch. Writes caller-provided buffers and returns `None`.
- **Formula**: For request `i`, `q_i = query[i + 1] - query[i]` and
  `c_i = max((seq_lens[i] - 1) // block_size, 0)`. Ordinary requests select
  `block_table[i, c_i]`; speculative requests select `spec_width` blocks
  starting at `c_i`. Initial-state flags are `seq_lens[i] > q_i`.
- **Algorithm flow**: Classify and partition requests on CPU, then write cache
  indices, query boundaries, lengths and token indices per request. Separate
  programs in the same launch clear `conv_output`.
- **Supported modes**: Ordinary/speculative decode, mixed prefill/decode and
  graph-padded buffers on Ascend NPU. Metadata preparation runs outside model graphs.
- **Compilation**: Request and decode counts are runtime scalars. Phase flags
  and tile sizes remain compile-time constants.

## Parameters

All arguments are required; pass `None` for disabled optional tensors.
`R = num_reqs`, `S`/`N` are speculative/non-speculative output row counts
(including applicable graph padding), and `W = spec_width`.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `block_table` | Input | Physical cache-block IDs | INT32 | 2D, strided |
| `seq_lens` | Input | Sequence lengths, at least `R` entries | INT32 | 1D, strided |
| `query` | Input | Query boundaries, `[R + 1]` | INT32 | Contiguous 1D |
| `accepted` | Input | Accepted-token counts in original request order | INT32 / `None` | 1D, strided |
| `layout` | Input | Mixed-batch group rows and token offsets, `[R, 2]` | INT32 / `None` | Contiguous ND |
| `spec_states` | Output | Speculative cache indices, `[S, W]` | INT32 | Contiguous ND |
| `non_spec_states` | Output | Ordinary cache indices, `[N]` | INT32 | Contiguous 1D |
| `conv_indices` | Output | First cache index per speculative row, `[S]` | INT32 | Contiguous 1D |
| `sequence_masks` | Output | Active speculative rows | BOOL | Contiguous 1D |
| `spec_query` | Output | Speculative query boundaries, `[S + 1]` | INT32 | Contiguous 1D |
| `non_spec_query` | Output | Non-speculative query boundaries, `[N + 1]` | INT32 | Contiguous 1D |
| `spec_lengths` | Output | Leading zero and speculative lengths, `[S + 1]` | INT32 | Contiguous 1D |
| `non_spec_lengths` | Output | Leading zero and non-speculative lengths, `[N + 1]` | INT32 | Contiguous 1D |
| `accepted_out` | Output | Accepted-token counts in group order, `[S]` | INT32 | Contiguous 1D |
| `spec_tokens` | Output | Original speculative token positions | INT32 / INT64 | Contiguous 1D |
| `non_spec_tokens` | Output | Original non-speculative token positions in mixed batches | INT64 | Contiguous 1D |
| `initial_state` | Output | Initial-state flags, `[N]` | BOOL / `None` | Contiguous 1D |
| `prefill_query` | Output | Prefill boundaries, `[N - num_decodes + 1]` | INT32 / `None` | Contiguous 1D |
| `conv_output` | Output | Shared convolution buffer to zero | Model dtype / `None` | Contiguous ND |
| `num_reqs` | Attribute | Input request count | `int` | N/A |
| `num_spec_decodes` | Attribute | Speculative request count | `int` | N/A |
| `num_decodes` | Attribute | Ordinary decode rows before prefill rows | `int` | N/A |
| `num_decode_tokens` | Attribute | Ordinary decode token count | `int` | N/A |
| `has_prefill` | Attribute | Enable prefill metadata | `bool` | N/A |
| `spec_width` | Attribute | Speculative width including the main-model token | `int` | N/A |
| `mixed_spec` | Attribute | Both speculative and non-speculative requests exist | `bool` | N/A |
| `block_size` | Attribute | Tokens per aligned cache block | `int` | N/A |
| `pad_block_id` | Attribute | Cache-index sentinel for padded rows | `int` | N/A |

## Constraints

- All tensors reside on the same NPU. Outputs and `query` are contiguous;
  `block_table`, `seq_lens` and `accepted` honor their input strides.
- Query ranges and block-table columns must cover all selected tokens and
  cache blocks. `block_size` and `spec_width` are positive.
- Mixed batches require `0 < num_spec_decodes < R`, exact group-sized output
  views and a stable, disjoint `layout`. Its first column encodes speculative
  row `d` as `d` and non-speculative row `d` as `-d - 1`; the second holds
  group token offsets. Homogeneous batches pass `layout=None`.
- Speculative batches require `accepted`; prefill requires `initial_state`
  and `prefill_query`. Unused phase outputs may be empty and must not be read.
- The host builder checks aligned caches, contiguous convolution indices and
  speculative cache capacity, and retains the generic path for unsupported cases.
- The caller owns output allocation and graph-stable buffer lifetimes.
  The wrapper introduces no device-to-host synchronization.

## Origin and Differences

- **Origin**: GDN metadata preparation in `AscendGDNAttentionMetadataBuilder`.
- **Differences**: Fuse indexing, stable token partitioning, graph padding
  and convolution-output clearing. Mixed batches transfer one compact CPU
  partition; each request writes its own token range without searches or sorting.

## Test Cases

The NPU test compares complete metadata with the generic builder using
`rtol=0, atol=0`. It covers variable speculative lengths, mixed batches,
strided inputs, shrinking batches, graph padding, shared-output clearing and
compiled-kernel reuse across changing batch counts.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_prepare.py
```

Model accuracy and performance measurements are recorded in
[PR #19954](https://github.com/vllm-project/vllm-ascend/pull/19954).
