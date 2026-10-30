# Sampling-mask packing (`SamplingMaskTensors.from_logits`)

## Description

- **Function**: identifies finite logits for committed tokens, packs the boolean
  keep mask into request-major output rows, and counts the kept vocabulary entries.
- **Python entry point**: `sampling_mask_from_logits_npu` in
  `vllm_ascend/ops/triton/v2/sample/sampling_mask.py`.
- **Triton kernel**: `_pack_sampling_mask_kernel_ascend`.
- **Integration**: `vllm_ascend/patch/worker/patch_triton.py` binds the
  wrapper directly as `SamplingMaskTensors.from_logits`.

For request `r`, output slot `s` and vocabulary index `i`, the keep condition is:

```text
output_row = r * rows_per_request + s
source_row = cu_num_logits[r] + s
is_active = s < num_sampled_tokens[r]
keep[output_row, i] = is_active and isfinite(logits[source_row, i])
```

The output bit layout is:

```text
packed_mask[output_row, i // 8] bit (i % 8) = keep[output_row, i]
counts[output_row] = sum_i keep[output_row, i]
```

An inactive slot has a zero mask and count. Its logits load is masked and
filled with negative infinity, so slots without a source row are not read.
Vocabulary tails that are not divisible by eight do not contribute bits or counts.

## Data flow

The kernel processes one reserved output row per Triton program and scans its vocabulary in
4096-element blocks:

```text
logits tile [4096]
    -> finite and active mask [4096]
    -> int32 sum -> row count
    -> reshape [512, 8]
    -> transpose [8, 512]
    -> multiply by bit weights [8, 1]
    -> reduce axis 0
    -> packed bytes [512]
```

The transpose makes the eight bits for 512 output bytes the reduction axis.
The bit weights are compile-time constants, so the packing lowers to vector
transpose, multiply, and reduction operations instead of a tensor-wide loop of
variable shifts.

The 4096-element block size also limits temporary storage. The original
8192-element implementation can exceed Ascend UB capacity for a large
non-contiguous vocabulary input.

## Parameters and outputs

`num_output_rows = num_reqs * rows_per_request`. Source logits may have a
different row count, with each request starting at its cumulative offset.

| Name | Input/Output/Attribute | Shape | Data type | Description |
| --- | --- | --- | --- | --- |
| `logits` | Input | `[num_logits, vocab_size]` | floating point | Processed logits. NaN, `+inf`, and `-inf` entries are excluded. |
| `cu_num_logits` | Input | `[num_reqs + 1]` or longer | integer | Cumulative source-row offsets; chunk offsets may start above zero. |
| `num_sampled_tokens` | Input | `[num_reqs]` | integer | Committed-token count per request; slots below this count are active. |
| `max_num_kept` | Attribute | scalar | Python int | Accepted for upstream compatibility; compact IDs remain unsupported. |
| `rows_per_request` | Attribute/Output metadata | scalar | Python int | Reserved slots per request; defaults to 1 and is retained for decoding. |
| `token_ids` | Output | `[num_output_rows, 0]` | int32 | Zero-width buffer selecting exact packed-mask decoding. |
| `packed_mask` | Output | `[num_output_rows, ceil(vocab_size / 8)]` | uint8 | Little-bit-order packed finite-logit mask. |
| `counts` | Output | `[num_output_rows]` | int32 | Number of set bits in each row. |
| `vocab_size` | Output metadata | scalar | Python int | Original vocabulary size used when unpacking the mask. |

The wrapper follows the upstream request-major calling convention:

```text
SamplingMaskTensors.from_logits(
    logits, cu_num_logits, num_sampled_tokens, max_num_kept, rows_per_request=1)
```

It returns the five-field
`SamplingMaskTensors(token_ids, packed_mask, counts, vocab_size, rows_per_request)`
with zero-width `token_ids`. The upstream `tolists()` method consequently uses
exact packed-mask decoding for non-empty rows.

Compact token-ID generation is temporarily unsupported on Ascend because its
`tl.cumsum`-based dynamic scatter can hang with Triton-Ascend 3.2.2. The wrapper
logs this compatibility fallback once per process. Sampling results are
unchanged, but CPU decoding may be slower than the upstream compact path.

## Stride handling

The Triton kernel accepts explicit row and column strides, but strided loads
along the vocabulary dimension can cause excessive UB allocation in the
Triton-Ascend compiler. The wrapper therefore creates a contiguous local copy
only when:

```text
logits.stride(1) != 1
```

Rebinding the local Python variable does not mutate or replace the caller's
tensor. The caller keeps the same storage, shape, and strides. Contiguous
inputs are passed through without a copy.

## Constraints

- The caller supplies rank-2 `logits`, rank-1 sampled counts with one entry per
  request, and rank-1 cumulative offsets with at least `num_reqs + 1` entries.
- The caller ensures positive integer `rows_per_request`, valid logits spans,
  and committed counts bounded by both the reserved slots and the span length.
  The wrapper follows upstream without extra type, rank or offset-length checks.
- The operator is inference-only and has no backward path.
- The vocabulary size may be non-aligned to both 4096 elements and eight bits;
  masked tail lanes are supported.
- The finite-logit count is accumulated as int32. The boolean keep mask is
  explicitly converted before reduction because boolean reduction does not
  provide the required integer count on affected Triton-Ascend versions.
- The wrapper-level contiguous copy is separate from the Triton kernel launch;
  kernel-only profiling does not include that copy cost.
- The upstream request-major API and five-field return structure are required;
  legacy three-/four-field interface adapters are not provided.
- Empty batches skip the kernel launch. Device-side offsets/counts can change
  during graph replay while the captured layouts and `rows_per_request` stay fixed.

## Origin and differences

- **Origin**: adapts vLLM Model Runner V2
  `SamplingMaskTensors.from_logits` and its sampling-mask packing kernel for
  Ascend NPU.
- **Correctness adaptation**: casts the keep mask to int32 before summation so
  `counts` contains the number of finite logits.
- **Memory adaptation**: uses 4096-element tiles and makes only a strided
  vocabulary dimension contiguous before launch to avoid UB overflow.
- **Performance adaptation**: replaces per-element variable shifts and a
  scalar-heavy `[512, 8]` axis-1 reduction with an `[8, 512]` vector reduction
  using compile-time bit weights.
- The packed bit order, finite-support counts and final decoded token IDs are
  unchanged. Request-major row mapping additionally supports multiple committed
  tokens; the compact `token_ids` optimization remains an exact bitmask fallback.

## Test cases

The NPU regression test covers empty, single-token, compact-width boundary,
over-2048, full-vocabulary, randomly sparse, and inactive rows with a vocabulary
size not divisible by eight. It also mirrors vLLM v0.29.0's processed top-k and
top-p support test, retains the large stride-2 UB-overflow regression, adds NaN
and positive infinity exclusions, checks exact counts and decoded token IDs,
and verifies that the caller's non-contiguous `logits` view is unchanged.
Request-major cases also cover ragged offsets, inactive slots, empty batches,
concatenation/slicing and graph replay with updated device metadata.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sampling_mask.py
```

The rejection-sampling integration test covers mask replay on/off, rejection
outcomes, logprobs and RNG preservation, and is kept outside the operator suite:

```bash
pytest -sv tests/e2e/pull_request/one_card/rlhf/consistency/test_sampling_mask_rejection.py
```
