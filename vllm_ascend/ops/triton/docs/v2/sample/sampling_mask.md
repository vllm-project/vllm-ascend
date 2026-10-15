# Sampling-mask packing (`SamplingMaskTensors.from_logits`)

## Description

- **Function**: identifies finite logits for active output slots, packs the
  boolean keep mask into bytes, and counts the kept vocabulary entries.
- **Python entry point**: `sampling_mask_from_logits_npu` in
  `vllm_ascend/ops/triton/v2/sample/sampling_mask.py`.
- **Triton kernel**: `_pack_sampling_mask_kernel_ascend`.
- **Integration**: `vllm_ascend/patch/worker/patch_triton.py` binds the
  wrapper directly as `SamplingMaskTensors.from_logits`.

Outputs are laid out in request-major slots. With `R = rows_per_request`,
output row `o` belongs to request `req = o // R`, slot `s = o % R`, and reads
logits row `cu_num_logits[req] + s`. For vocabulary index `i`, the keep
condition is:

```text
active[o]  = s < num_sampled_tokens[req]
keep[o, i] = isfinite(logits[cu_num_logits[req] + s, i]) and active[o]
```

The output bit layout shared by all supported interfaces is:

```text
packed_mask[o, i // 8] bit (i % 8) = keep[o, i]
counts[o] = sum_i keep[o, i]
```

An inactive slot therefore has a zero mask and a zero count. Its source row is
clamped to `cu_num_logits[req]` and all loads are masked, so it never reads past
the last logits row; this keeps control flow uniform instead of using an early
return. Vocabulary tails that are not divisible by eight are masked and do not
contribute bits or counts.

The legacy interfaces have one logits row per request in request order. The
wrapper synthesizes `cu_num_logits = arange(num_reqs + 1)` and uses `R = 1`, so
they reduce to `keep[r, i] = isfinite(logits[r, i]) and num_sampled_tokens[r] > 0`.

## Data flow

The kernel processes one output slot per Triton program
(`num_reqs * rows_per_request` programs) and scans its source row's vocabulary
in 4096-element blocks:

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

| Name | Input/Output/Attribute | Shape | Data type | Description |
| --- | --- | --- | --- | --- |
| `logits` | Input | `[num_logits, vocab_size]` | floating point | Logits rows. NaN, `+inf`, and `-inf` entries are excluded. |
| `cu_num_logits` | Input | `[num_reqs + 1]` | int32 | Start offset of each request's logits rows. Five-field API only; synthesized as `arange(num_reqs + 1)` otherwise. |
| `num_sampled_tokens` | Input | `[num_reqs]` | integer | Number of active slots of each request. |
| `max_num_kept` | Input | scalar | Python int | Accepted for API compatibility; compact IDs are not produced (see below). |
| `rows_per_request` | Attribute | scalar | Python int | Fixed output slots per request (`num_speculative_steps + 1` for spec decode). Five-field API only; defaults to 1. Compile-time constant of the kernel. |
| `packed_mask` | Output | `[num_reqs * rows_per_request, ceil(vocab_size / 8)]` | uint8 | Little-bit-order packed finite-logit mask. |
| `counts` | Output | `[num_reqs * rows_per_request]` | int32 | Number of set bits in each row. |
| `vocab_size` | Output metadata | scalar | Python int | Original vocabulary size used when unpacking the mask. |

The wrapper accepts every dependency calling convention:

```text
SamplingMaskTensors.from_logits(logits, num_sampled_tokens)
SamplingMaskTensors.from_logits(logits, num_sampled_tokens, max_num_kept)
SamplingMaskTensors.from_logits(
    logits, cu_num_logits, num_sampled_tokens, max_num_kept, rows_per_request=1
)
```

The layout is selected from the installed `SamplingMaskTensors._fields`. The
three-field structure returns
`SamplingMaskTensors(packed_mask, counts, vocab_size)`. The four-field structure
returns `SamplingMaskTensors(token_ids, packed_mask, counts, vocab_size)`, and
the five-field structure from vLLM #59359 returns
`SamplingMaskTensors(token_ids, packed_mask, counts, vocab_size, rows_per_request)`.
In both cases `token_ids` is zero-width, so the upstream `tolists()` method uses
the exact packed mask for every non-empty row.

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

- `logits` is a rank-2 tensor. With the legacy interfaces,
  `num_sampled_tokens` has one entry per logits row; with the five-field
  interface, `num_sampled_tokens[req] <= rows_per_request` and
  `cu_num_logits` indexes rows of the passed `logits` (chunk-relative when the
  rejection sampler splits the batch).
- The operator is inference-only and has no backward path.
- The vocabulary size may be non-aligned to both 4096 elements and eight bits;
  masked tail lanes are supported.
- The finite-logit count is accumulated as int32. The boolean keep mask is
  explicitly converted before reduction because boolean reduction does not
  provide the required integer count on affected Triton-Ascend versions.
- The wrapper-level contiguous copy is separate from the Triton kernel launch;
  kernel-only profiling does not include that copy cost.
- The three-field bitmask interface, the four-field compact-ID plus bitmask
  interface, and the five-field request-major interface are all supported; all
  of them launch `_pack_sampling_mask_kernel_ascend`.
- Each distinct `rows_per_request` value compiles a separate kernel variant;
  in practice only `1` and `num_speculative_steps + 1` occur.

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
- The packed bit order, request-major slot layout, active-slot semantics,
  count values, and final decoded token IDs are unchanged. The four- and
  five-field structures remain compatible, while their compact `token_ids`
  optimization is replaced by the exact bitmask fallback.

## Test cases

The NPU regression test covers empty, single-token, compact-width boundary,
over-2048, full-vocabulary, randomly sparse, and inactive rows with a vocabulary
size not divisible by eight. It also mirrors vLLM v0.29.0's processed top-k and
top-p support test, retains the large stride-2 UB-overflow regression, adds NaN
and positive infinity exclusions, checks exact counts and decoded token IDs,
and verifies that the caller's non-contiguous `logits` view is unchanged. With
the later four-field main interface, it additionally validates the zero-width
compact buffer and exact bitmask fallback. With the five-field request-major
interface, it checks that source rows follow `cu_num_logits` request boundaries
(including an inactive request) and that a spec-decode layout with
`rows_per_request = 4` and partially active slots decodes the expected
per-token supports.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sampling_mask.py
```
