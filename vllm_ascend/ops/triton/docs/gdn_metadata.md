# prepare_gdn_metadata

## Description

- **Location**: [`v2/mamba/gdn_metadata.py`](../v2/mamba/gdn_metadata.py).
  `prepare_gdn_metadata(...)` is the Python entry point; it computes launch
  dimensions and strides, then calls `_prepare_gdn_metadata_kernel[grid](...)`.
- **Function**: Populate aligned-cache GDN metadata and clear the shared
  convolution output in one Triton launch. The caller supplies the output
  buffers; the function returns `None`. The
  [`AscendGDNHostMetadataBuilder`](../../gdn_attn_builder.py) uses this entry
  point before model execution and attaches the resulting decode descriptors.
- **Formula**: For request `i`, let `q_i = query[i + 1] - query[i]` and
  `c_i = max((seq_lens[i] - 1) // block_size, 0)`. Ordinary requests select
  `block_table[i, c_i]`; speculative requests select the next `spec_width`
  entries starting at `c_i`. Group query boundaries use the stable partition's
  token offsets, per-request lengths equal `q_i`, and the initial-state flag
  is `seq_lens[i] > q_i`. Token indices retain their original flattened
  positions. Every element of `conv_output`, when present, is set to zero.
- **Algorithm flow**:
    1. The builder classifies requests and prepares output views, including
       graph padding. For mixed speculative/non-speculative batches it
       transfers one compact CPU partition containing group rows and offsets.
    2. Each request program writes cache indices, query boundaries, lengths,
       accepted-token counts and, where needed, stable token partitions.
       Padded request rows use `pad_block_id` and empty query ranges.
    3. Additional programs in the same launch clear the shared convolution
       output. The grid is `rows + ceil(conv_output.numel() / 1024)`, with
       `rows = max(num_reqs, spec_states.shape[0], non_spec_states.shape[0])`.
- **Supported modes**: Ordinary decode, variable speculative lengths, mixed
  prefill/decode and mixed speculative/non-speculative batches on the Ascend
  Triton host-metadata path. Hardware measurements below used Ascend950DT.
  Metadata preparation remains outside model graphs, including when serving
  uses `FULL_DECODE_ONLY`; graph-stable output buffers are reused.

## Parameters

All arguments are required. Optional tensor arguments must be passed explicitly
as `None` when their phase is disabled. Let `R = num_reqs`, `S` and `N` be the
speculative and non-speculative output row counts, and `W = spec_width`.
Homogeneous graph batches may have padded output rows; mixed batches use exact
group-sized views.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `block_table` | Input | Physical cache-block IDs; enough rows for `R` requests and columns for every selected block | INT32 | 2D; row and column strides honored |
| `seq_lens` | Input | Current sequence lengths, at least `R` entries | INT32 | 1D; stride honored |
| `query` | Input | Normalized query boundaries, length `R + 1` | INT32 | Contiguous 1D |
| `accepted` | Input | Accepted-token counts for speculative requests in original request order, or `None` without speculation | INT32 or `None` | 1D; stride honored |
| `layout` | Input | Mixed-batch plan `[R, 2]`: signed group row and group token offset; otherwise `None` | INT32 or `None` | Contiguous ND |
| `spec_states` | Output | Speculative cache-block IDs, shape `[S, W]` | INT32 | Contiguous ND |
| `non_spec_states` | Output | Ordinary cache-block IDs, shape `[N]` | INT32 | Contiguous 1D |
| `conv_indices` | Output | First cache-block ID per speculative row, capacity at least `S` | INT32 | Contiguous 1D |
| `sequence_masks` | Output | Whether each request row is active and speculative; written when speculation exists | BOOL | Contiguous 1D, capacity at least `rows` |
| `spec_query` | Output | Speculative group query boundaries, length `S + 1` | INT32 | Contiguous 1D |
| `non_spec_query` | Output | Non-speculative group query boundaries, length `N + 1` | INT32 | Contiguous 1D |
| `spec_lengths` | Output | Leading zero followed by speculative per-request lengths, length `S + 1` | INT32 | Contiguous 1D |
| `non_spec_lengths` | Output | Leading zero followed by non-speculative per-request lengths, length `N + 1` | INT32 | Contiguous 1D |
| `accepted_out` | Output | Accepted-token counts in speculative group order, capacity at least `S` | INT32 | Contiguous 1D |
| `spec_tokens` | Output | Original flattened positions of speculative tokens | INT32 for homogeneous batches; INT64 for mixed batches | Contiguous 1D |
| `non_spec_tokens` | Output | Original flattened positions of non-speculative tokens in mixed batches; otherwise unused | INT64 in mixed batches | Contiguous 1D |
| `initial_state` | Output | Non-speculative initial-state flags, length `N`, or `None` without prefill | BOOL or `None` | Contiguous 1D |
| `prefill_query` | Output | Prefill-only query boundaries, length `N - num_decodes + 1`, or `None` without prefill | INT32 or `None` | Contiguous 1D |
| `conv_output` | Output | Shared convolution buffer to clear, or `None` when disabled | Model floating-point dtype or `None` | Contiguous ND |
| `num_reqs` | Attribute | Number of normalized input request rows | Python `int` | N/A |
| `num_spec_decodes` | Attribute | Number of speculative input requests | Python `int` | N/A |
| `num_decodes` | Attribute | Number of ordinary decode rows before prefill rows in the non-speculative group | Python `int` | N/A |
| `num_decode_tokens` | Attribute | Ordinary decode token count to subtract from prefill query offsets | Python `int` | N/A |
| `has_prefill` | Attribute | Whether prefill metadata is required | Python `bool` | N/A |
| `spec_width` | Attribute | Configured speculative width, including the main-model token | Python `int` | N/A |
| `mixed_spec` | Attribute | Whether both speculative and non-speculative requests are present | Python `bool` | N/A |
| `block_size` | Attribute | Tokens represented by each aligned cache block | Python `int` | N/A |
| `pad_block_id` | Attribute | Sentinel cache-block ID written for padded rows | Python `int` | N/A |

## Constraints

- All tensors must be on the same NPU. Output tensors and `query` must be
  contiguous; only the input strides listed in the table are supported.
  Outputs must not alias inputs or each other in ways that create write races.
- Query boundaries must describe the normalized request order and valid token
  ranges. Block-table columns must cover `c_i` and, for speculative requests,
  `c_i + spec_width - 1`. `block_size` and `spec_width` must be positive.
- With `mixed_spec=True`, `0 < num_spec_decodes < R`. A nonnegative
  `layout[i, 0]` is a speculative destination row; a negative value encodes
  non-speculative row `-layout[i, 0] - 1`. `layout[i, 1]` is the corresponding
  group token offset. Rows and token ranges must form a stable, disjoint
  partition. The builder constructs this plan on CPU without reading device
  scalar values.
- Homogeneous batches pass `layout=None`. Speculative batches require
  `accepted`; prefill batches require `initial_state` and `prefill_query`.
  Phase-specific outputs are initialized only for enabled phases. Unused
  output tensors may be empty and must not be consumed.
- The host builder enables this path for aligned Mamba caches with contiguous
  convolution indices and sufficient speculative cache blocks. Other cases
  retain the generic builder. The Python entry point expects these conditions
  to have been checked by its caller.
- The wrapper allocates no output tensors and introduces no device-to-host
  synchronization. Buffer allocation, CPU classification and model graph
  lifecycle remain the builder's responsibility.

## Origin and Differences

- **Origin**: New Triton implementation of the existing GDN metadata
  preparation in `AscendGDNAttentionMetadataBuilder` and its Ascend host path.
- **Differences**: Combines cache indexing, query/length generation, stable
  token partition writes, graph padding and convolution-output clearing in
  one device launch. Mixed batches transfer one compact partition instead of
  a mask and two sequence-index arrays. Each request writes its own token
  range without token-to-request searches, token-mask expansion or sorting.
  Batch classification and descriptor attachment remain shared with the
  generic implementation. No model-name checks or new configuration flags
  are introduced.

## Test Cases

### Operator correctness

The existing [NPU comparison test](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_decode_metadata.py)
compares complete metadata against the generic builder with `rtol=0, atol=0`.
It covers ordinary decode, MTP3 and other speculative widths, variable draft
lengths, mixed prefill/decode, mixed speculative/non-speculative requests,
strided inputs, shrinking batches and graph padding. It also checks buffer
identity and clearing of a shared output prefilled with NaNs.

Run from the repository root in an initialized Ascend environment:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_decode_metadata.py
```

The hardware results below were collected before the Python-entry-point
refactor. They do not represent a new hardware run of that refactor.

### Operator performance

Measured at PR source `fbd0d17e012164a909f5618bb7f4f9d670fd5629` on one
Ascend950DT NPU: 16 requests, MTP3, aligned cache, `FULL_DECODE_ONLY`
configured, 20 warmups, three alternating rounds of 50 samples, synchronization
before and after each build, median of round means. These are metadata-only
latencies. Complete metadata matched the generic builder.

| Batch | Generic | Fused | Reduction |
| --- | ---: | ---: | ---: |
| Uniform speculative decode | 0.612 ms | 0.208 ms | 66.1% |
| Variable speculative lengths | 0.576 ms | 0.210 ms | 63.5% |
| Ordinary decode + prefill | 0.582 ms | 0.451 ms | 22.6% |
| Speculative decode + prefill + ordinary decode | 0.808 ms | 0.569 ms | 29.6% |

### Model accuracy

First paired GPQA Diamond run (2026-10-28), at the same PR source above:
GLM-5.3-Flash `GLM-5.3-Flash-W8A8-MXFP8-0918`, BF16 KDA projections,
Ascend950DT x8, TP8+EP, FP8 KV, aligned cache, MTP3, `FULL_DECODE_ONLY`,
and the V2 runner.

AISBench `5a63d936e40dd04addfcffb2401f84cd0e53bb46` evaluated all 198
Diamond questions with zero-shot CoT chat prompts and a fixed cyclic option
permutation: temperature=1.0, top_p=0.95, seed=1024, max_tokens=131072,
concurrency=32, and the weight template's default Max reasoning effort.
Warmup was excluded.

| Metric | Unfused | Fused |
| --- | ---: | ---: |
| GPQA Diamond accuracy | 90.91% | 90.91% |
| Correct answers | 180/198 | 180/198 |

Both variants used the same PR source and serving/evaluation parameters. The
validation harness forced `_build_fused_metadata` to return `None` for the
unfused variant. This was one sampled run; the zero percentage-point accuracy
difference does not establish statistical significance across seeds.

The run used fixed vLLM `db9527a46873454610df6dbedf79a36d6bf1a7f6`,
driver 26.2.0.b007, CANN 9.2.0 B237, Python 3.11.10, torch 2.10.0,
torch_npu 2.10.0.post7, triton-ascend 3.6.0+dev20261013220622 and
transformers 5.14.1. This fixed vLLM revision differs from the PR's CI main pin.

### Model performance

Earlier serving measurements compared feature
`64fc539232e36be8d5829b8a14cdbec2daa70066` with baseline
`db8836752a17abc9a26212d2b6e075b72dd4e2ab`, before the mixed-batch extension
and Python wrapper. They provide historical context rather than current-head
end-to-end measurements.

GLM-5.3-Flash, Ascend950DT x8, TP8+EP, 131072 input / 1024 output tokens,
MTP3, aligned cache, INT8 KV and `FULL_DECODE_ONLY`; physical prefix-cache
hit rate 99.3164%. Three uninstrumented rounds per concurrency measured the
interval after every request emitted at least 128 tokens and before the first
completion, excluding startup and reduced-concurrency tails.

| Concurrency | Generic throughput | Fused throughput | Change |
| --- | ---: | ---: | ---: |
| 10 | 1193.19 token/s | 1216.07 token/s | +1.92% |
| 32 | 2876.00 token/s | 2881.35 token/s | +0.19% |
| 64 | 4397.05 token/s | 4406.65 token/s | +0.22% |

These small throughput differences are descriptive and do not establish
statistical significance.

## Example

Two ordinary decode requests select cache blocks 7 and 12. This example
allocates buffers for illustration; serving reuses the builder's buffers.

```python
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.v2.mamba.gdn_metadata import prepare_gdn_metadata

device = "npu"


def empty_i32(*shape):
    return torch.empty(shape, device=device, dtype=torch.int32)


block_table = torch.tensor([[7, 8], [11, 12]], device=device, dtype=torch.int32)
seq_lens = torch.tensor([15, 17], device=device, dtype=torch.int32)
query = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
cache_indices = empty_i32(2)
non_spec_query = empty_i32(3)
non_spec_lengths = empty_i32(3)
conv_output = torch.ones((2, 16), device=device, dtype=torch.bfloat16)

prepare_gdn_metadata(
    block_table=block_table,
    seq_lens=seq_lens,
    query=query,
    accepted=None,
    layout=None,
    spec_states=empty_i32(0, 4),
    non_spec_states=cache_indices,
    conv_indices=empty_i32(0),
    sequence_masks=torch.empty(0, device=device, dtype=torch.bool),
    spec_query=empty_i32(1),
    non_spec_query=non_spec_query,
    spec_lengths=empty_i32(1),
    non_spec_lengths=non_spec_lengths,
    accepted_out=empty_i32(0),
    spec_tokens=empty_i32(0),
    non_spec_tokens=empty_i32(0),
    initial_state=None,
    prefill_query=None,
    conv_output=conv_output,
    num_reqs=2,
    num_spec_decodes=0,
    num_decodes=2,
    num_decode_tokens=2,
    has_prefill=False,
    spec_width=4,
    mixed_spec=False,
    block_size=16,
    pad_block_id=-1,
)

torch.testing.assert_close(cache_indices, block_table.new_tensor([7, 12]))
torch.testing.assert_close(non_spec_query, query)
torch.testing.assert_close(non_spec_lengths, query.new_tensor([0, 1, 1]))
torch.testing.assert_close(conv_output, torch.zeros_like(conv_output))
```
