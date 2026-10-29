# Sampling-mask packing (`SamplingMaskTensors.from_logits`)

## Description

This operator packs the finite support of processed logits into request-major
sampling masks on Ascend. A request can commit multiple tokens during speculative
decoding; each committed token needs its own source logits row and output mask.
Unused output slots contain zero masks and counts.

- **Python entry point**: `sampling_mask_from_logits_npu` in
  `vllm_ascend/ops/triton/v2/sample/sampling_mask.py`.
- **Triton kernel**: `_pack_sampling_mask_kernel_ascend`.
- **Integration**: `vllm_ascend/patch/worker/patch_v2/patch_triton.py` binds the
  wrapper as the class method `SamplingMaskTensors.from_logits` for Model Runner V2.

For request `r`, reserved output slot `s`, and vocabulary index `i`:

```text
output_row = r * rows_per_request + s
source_row = cu_num_logits[r] + s
is_active = s < num_sampled_tokens[r]
keep[output_row, i] = is_active and isfinite(logits[source_row, i])

packed_mask[output_row, i // 8] bit (i % 8) = keep[output_row, i]
counts[output_row] = sum_i keep[output_row, i]
```

This is a logical definition: inactive slots do not read their computed source
row. The load predicate combines the vocabulary bound with `is_active`; masked
lanes receive negative infinity and fail the finite-value check. This protects
inactive slots without corresponding logits rows, including empty trailing spans.

Each byte packs eight consecutive vocabulary IDs in little-bit order. NaN,
positive/negative infinity and vocabulary-tail lanes contribute no bits or counts.
The operator does not truncate supports to exactly top-k: ties retained by
upstream processing can produce more than k finite entries.

## Parameters and outputs

The class-method calling convention is:

```python
SamplingMaskTensors.from_logits(
    logits,
    cu_num_logits,
    num_sampled_tokens,
    max_num_kept,
    rows_per_request=1,
)
```

Let N be the request count, L the logits-row count, V the vocabulary size, and
R = `rows_per_request`.

| Name | Input/Output/Attribute | Shape | Data type | Description |
| --- | --- | --- | --- | --- |
| `logits` | Input | `[L, V]` | floating point | Processed logits; L need not equal N or N times R. |
| `cu_num_logits` | Input | `[N + 1]` or longer | integer | Cumulative source-row offsets; chunk metadata can contain nonzero offsets into shared logits. |
| `num_sampled_tokens` | Input | `[N]` | integer | Committed-token count per request, selecting active slots. |
| `max_num_kept` | Attribute | scalar | Python int | Accepted for upstream API compatibility; compact IDs are not generated on this path. |
| `rows_per_request` | Attribute | scalar | Python int | Positive reserved output slots per request; defaults to 1. |
| `token_ids` | Output | `[N * R, 0]` | int32 | Zero-width compact-ID buffer selecting exact packed-mask decoding. |
| `packed_mask` | Output | `[N * R, ceil(V / 8)]` | uint8 | Request-major supports, with zero inactive rows and padding bits. |
| `counts` | Output | `[N * R]` | int32 | Finite candidates per active row; zero for inactive rows. |
| `vocab_size` | Output metadata | scalar | Python int | Original vocabulary size for unpacking. |
| `rows_per_request` | Output metadata | scalar | Python int | R, retained for request-major decoding, concatenation and slicing. |

The returned structure has five fields:

```text
SamplingMaskTensors(token_ids, packed_mask, counts, vocab_size, rows_per_request)
```

With R=3 and committed counts `[1, 3, 0]`, nine output rows are reserved: one
active plus two zero rows for request 0, three active rows for request 1, and
three zero rows for request 2. Active slots read from their own cumulative offset.

This revision targets the request-major API exercised with vLLM 0.32.0. It does
not select older three-argument calls or three-/four-field return structures
based on whether `max_num_kept` is supplied.

## Data flow

The kernel runs one Triton program per reserved output row and scans
the source vocabulary in 4096-element blocks:

```text
output row -> request and slot index
    -> device-side committed count and cumulative logits offset
    -> active-slot and vocabulary-bound load mask
    -> logits tile [4096], masked lanes filled with -inf
    -> finite mask -> int32 mask
        -> sum -> output-row count
        -> reshape [512, 8] -> transpose [8, 512]
        -> multiply by bit weights [8, 1] -> reduce axis 0
        -> packed bytes [512]
```

The transpose makes the eight bits for 512 output bytes the reduction axis.
Compile-time bit weights replace the earlier variable-shift loop. Explicit int32
conversion before reduction provides correct counts on affected Triton-Ascend
versions, where boolean reduction does not produce the required integer sum.

The implementation names the byte width `_BITS_PER_BYTE`, constructed as
`tl.constexpr(8)` for JIT access. Host-side packed-width calculation uses its
integer value.

## Compact-ID fallback and stride handling

Compact token-ID generation remains unsupported on this path because its
`tl.cumsum`-based dynamic scatter can hang with Triton-Ascend 3.2.2. The wrapper
returns zero-width `token_ids` and logs the fallback once per process. Upstream
uses the exact packed mask for non-empty rows; CPU decoding may cost more than
the compact path, but the decoded support remains exact.

The kernel accepts explicit row/column strides. To avoid excessive UB allocation
from strided vocabulary loads, the wrapper makes a contiguous local copy when
`logits.stride(1) != 1`. It does not replace or mutate the caller's view, and
contiguous inputs pass through without a copy. Kernel-only profiling excludes
this wrapper-level copy.

## Constraints and graph replay

- `logits` must be rank 2, sampled counts rank 1, and cumulative offsets rank 1
  with at least N+1 entries. R must be a positive integer.
- The caller supplies valid device metadata: offsets describe valid logits
  spans, and each committed count lies between zero and both R and its span
  length. Shape checks do not validate device values or synchronize to read them.
- Output storage reserves N times R rows even when fewer tokens are committed.
  A zero-request batch returns empty outputs without launching the kernel.
- Vocabulary tails need not align to a tile or byte; tail bits are zero.
- Offsets and counts are loaded on the device, so graph replay reads updated
  metadata. R and tensor layouts remain fixed for that captured graph.
- The operator is inference-only, has no backward path, and does not consume RNG.

## Tests

Packing/layout regression tests remain in the operator directory:

```bash
python -m pytest -v \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sampling_mask.py
```

Their 28 parameterized cases cover finite-support sizes, exact counts/decoding,
ragged offsets, R=1/2/4, inactive and empty requests, non-contiguous inputs,
nonfinite values, invalid R, concatenation/slicing, processed top-k/top-p support,
and graph replay with updated metadata.

The rejection-sampling integration test belongs with speculative decoding:

```bash
python -m pytest -v \
  tests/e2e/pull_request/one_card/spec_decode/test_sampling_mask_rejection.py
```

Its eight parameter combinations cover chunked/single-chunk verification, two
seed settings and two logprob settings. Each verifies immediate rejection,
second-position rejection and full acceptance using real Ascend kernels. Mask
replay on/off must preserve committed tokens, counts, logprobs, ranks and global
NPU RNG state, while emitted masks match processed-logit supports.
