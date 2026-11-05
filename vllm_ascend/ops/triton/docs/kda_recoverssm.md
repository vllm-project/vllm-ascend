# KDA RecoverSSM

## Description

- **Function**: Verify a speculative KDA window using one persistent recurrent
  checkpoint per cache page. Record each token's correction and raw key/gate,
  then reconstruct only the accepted recurrent state after sampling. Commit also
  compacts the convolution history prepared by the caller. This supports GLM
  speculative decoding without keeping a full recurrent checkpoint per draft.
- **Formula**: For one head, let `S` have shape `[V, K]`, and normalize nonzero
  query/key vectors to unit L2 norm; zero vectors remain zero. Scale the normalized
  query by `K**-0.5`. With `a = exp(A_log)` and `x = raw_g + dt_bias`:

  ```text
  g = lower_bound * sigmoid(a * x)       if lower_bound is not None
      -a * softplus(x)                  otherwise
  decay = exp(g)
  decayed_state = S * decay[None, :]
  correction = sigmoid(raw_beta) * (v - decayed_state @ normalized_k)
  next_state = decayed_state + correction[:, None] * normalized_k[None, :]
  output = next_state @ normalized_scaled_q
  ```

  Verification records `correction` and the unnormalized `k/raw_g`. Commit
  reconstructs `next_state` through the accepted prefix from those records and
  the original checkpoint. Softplus uses its linear branch above 20.
- **Algorithm flow** (independent requests and heads, with value-dimension tiles):
  1. `_kda_recoverssm_verify_kernel` loads a request's checkpoint, advances a
     register-held recurrence, writes outputs and compact records, and leaves the
     persistent checkpoint unchanged. Null-page rows produce zero output and do
     not write records.
  2. `_prepare_commit_plan_kernel` maps speculative rows to sampler rows, clamps
     accepted lengths to the query length and window capacity, and prepares final
     and optional boundary destinations. Null pages and zero accepted lengths
     become no-op plan entries.
  3. `_compact_conv_state_kernel` copies the accepted convolution history to its
     destination. For history length `W` and accepted length `c > 0`, the source
     interval is `[c - 1, c - 1 + W)` and the destination interval is `[0, W)`.
  4. `_commit_kda_state_kernel` traverses the accepted records backwards,
     accumulates decay/products and correction outer products, and writes the
     reconstructed recurrent checkpoint. `_commit_kda_state_grouped_kernel`
     performs the same operation for two adjacent heads per program on its A5
     fast path.
- **Supported modes**: Atlas A3 and Ascend 950DT (A5), inference only, eager and
  NPU graph execution. Atlas A2 and 950PR are not validated by these tests.
  This operator-only change does not activate the GLM serving path. Model,
  configuration and worker integration will be provided in a follow-up change.
  The low-level optional align arguments do not establish serving support for
  Mamba cache mode `align`.

## Parameters

Notation: `B` speculative requests, `T` flattened tokens, `H` local heads,
`K/V` key/value widths, `P` cache pages, `Q` speculative-window capacity,
`L` local KDA layers, `C` convolution channels, `W` convolution history length.
An optional argument's default is part of its description.

### `kda_recoverssm_verify`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `q`, `k`, `raw_g` | Input | Query, raw key, raw gate, each `[1, T, H, K]` | BF16 / FP16 | ND; token-strided views allowed |
| `v` | Input | Values `[1, T, H, V]` | Same activation dtype | ND |
| `raw_beta` | Input | Raw update gate `[1, T, H]`; sigmoid is applied internally | Same activation dtype | ND |
| `A_log` | Input | Per-head log decay scale `[H]` | FP32 in validated callers | Contiguous ND |
| `dt_bias` | Input | Gate bias with `H * K` elements | FP32 in validated callers | Contiguous ND |
| `lower_bound` | Attribute | Gate lower bound; `None` selects negative softplus instead of bounded sigmoid | Python float / `None` | Scalar / N/A |
| `checkpoint_state` | Input | Persistent recurrent state `[P, H, V, K]`, read-only during verification | FP32 / BF16 | Paged ND, explicit strides |
| `correction_cache` | Output | Per-token correction records `[P, H, Q, V]` | FP32 | Paged ND, explicit strides |
| `kg_cache` | Output | Raw key and raw gate concatenated on the last axis, `[P, H, Q, 2 * K]` | Same dtype as `k` | Paged ND, explicit strides |
| `query_start_loc` | Input | Nondecreasing token offsets `[B + 1]` | INT32 / INT64 | Strided 1D |
| `state_indices` | Input | Source page per speculative row `[B]`; values at or below `NULL_BLOCK_ID` select null rows | INT32 | Strided 1D |
| `spec_query_len` | Attribute | `Q`, normally `num_speculative_tokens + 1` | Python int | Positive scalar |
| `out` | Output | Output `[1, T, H, V]`; default `None` allocates an output with `v`'s shape and dtype; returns the supplied buffer when present | Same dtype as `v` in validated callers | ND |

### `KDARecoverSSMCommitContext.create`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `layers` | Input | Nonempty sequence of layer objects exposing the fields below | Python sequence | N/A |
| `spec_query_len` | Attribute | `Q`, shared by the local layers | Python int | Positive scalar |
| `max_num_reqs` | Attribute | Capacity of the reusable commit-plan arrays | Python int | Positive scalar |
| Returned context | Output | Retains references to layer-owned pages, stacks gate parameters and allocates address/stride tables and plan arrays | Context object | N/A |

Required layer fields:

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `kv_cache[0]` | Input/Output | Convolution state; normalized view `[P, C, Q + W - 1]` | BF16 in validated callers | Paged ND; channel-first or token-first physical layout |
| `kv_cache[1]` | Input/Output | Recurrent checkpoint `[P, H, V, K]` | FP32 / BF16 | Paged ND |
| `kv_cache[2]` | Input | Correction records from verification `[P, H, Q, V]` | FP32 | Paged ND |
| `kv_cache[3]` | Input | Key/gate records from verification `[P, H, Q, 2 * K]` | BF16 / FP16 | Paged ND |
| `A_log` | Input | Decay parameter with `H` elements, stacked as `[L, H]` internally | FP32 in validated callers | ND |
| `dt_bias` | Input | Gate bias with `H * K` elements, stacked as `[L, H, K]` internally | FP32 in validated callers | ND |
| `gate_lower_bound` | Attribute | Shared `lower_bound` across layers | Python float / `None` | Scalar / N/A |
| `local_num_heads`, `head_dim` | Attribute | `H` and `K`, used to view gate parameters | Python int | Positive scalars |

### `KDARecoverSSMCommitContext.commit`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `num_accepted_tokens` | Input | Number of recurrent steps to commit per sampler row; includes any target input step in that row | INT32 | Strided 1D |
| `state_indices` | Input | Source page per speculative row `[B]` | INT32 | Strided 1D |
| `query_start_loc` | Input | Token offsets `[B + 1]`, as in verification | INT32 / INT64 | Strided 1D |
| `request_indices` | Input | Optional mapping from speculative rows to sampler rows; default `None` means identity mapping | INT32 / `None` | Strided 1D / N/A |
| `block_table` | Input | Optional request-to-page table for low-level align destinations; default `None` commits to the source page | INT32 / `None` | Strided 2D / N/A |
| `num_computed_tokens` | Input | Optional prior token count per sampler row, used with `block_table`; default `None` | INT32 / `None` | Strided 1D / N/A |
| `mamba_block_size` | Attribute | Optional token span per align block, at least `Q`; default `None` | Python int / `None` | Scalar / N/A |
| Context checkpoint/convolution pages | Input/Output | Reconstructed state and compacted history are written in place; the method returns `None` | See layer fields | Paged ND |
| Context plan arrays | Output | Accepted lengths, final page IDs, boundary page IDs and boundary recovery lengths | INT32 | Contiguous 1D |

Internal kernel arguments consist of the tensors above, INT64 base-address and
per-layer stride tables, element-stride scalars, `NULL_BLOCK_ID` and shape/tile
attributes. `K`, `V`, `BK`, `BV`, `NUM_HEADS`, `SPEC_QUERY_LEN`, `UNROLL_COMMIT`, gate/align flags,
`BLOCK_D`, `BLOCK_HISTORY`, `DIM_STRIDE`, `USE_1D_CHANNEL_GATHER` and grouped-head tile
`BH` specialize their respective kernels. The plan's `HAS_REQUEST_INDICES` and `ALIGN_MODE` are derived
from optional pointer presence by `triton.heuristics`. Address tables and context
buffers are implementation-owned, not user outputs to replace between replays.

## Constraints

- All tensor arguments to a call must be on the same NPU. Shapes and floating
  dtypes must match the parameter tables. The tensor batch dimension for
  activations is 1: requests are represented by flattened token offsets.
- `K`, `V` and `Q` are positive. Every query span is in `[0, Q]`, offsets begin at
  zero and end at `T`, and live page IDs and sampler-row mappings are in range.
  Concurrent live rows must not write overlapping cache pages.
- The last two strides of `q/k/raw_g` are `(K, 1)` and those of `v/out` are
  `(V, 1)`; the head stride of `raw_beta` is 1. Token strides may differ.
  `A_log` and `dt_bias` are contiguous. Cache pages may include page/head padding;
  supplied strides must describe allocated, nonoverlapping storage.
- Layer checkpoint, correction and key/gate inner strides must match within a
  context; page strides may differ by layer. Convolution layout is derived from
  `is_conv_state_dim_first()` and converted to a view, without changing caller
  storage. `W = conv_state_len - Q + 1` must be positive.
- `B` must not exceed context capacity. Accepted counts are clamped to
  `[0, min(query_length, Q)]`. Null pages and zero accepted lengths do not modify
  persistent state/history. Rejected suffix records are not read by commit.
- The caller prepares convolution state before commit. Commit writes only the
  accepted checkpoint destinations and the first `W` convolution positions;
  verification writes only `out` and the valid correction/key/gate record spans.
  Input/output buffers must not alias these write regions unexpectedly.
- `block_table`, `num_computed_tokens` and `mamba_block_size` must either all be
  absent or all be supplied. Low-level align can additionally write a boundary
  checkpoint/history; full align commit is not covered by the GLM `none`-mode
  serving validation. Compact's align copy path has separate exact-copy tests.
- Normalization and nonlinear arithmetic operate on FP32 values after existing
  load-time conversions. Output is stored in the activation dtype; checkpoints
  retain their storage dtype. This does not change caller tensor dtype/layout.
  Finite, non-overflowing inputs/intermediates are the validated numerical domain.
- A3 verification/commit use value tiles no larger than 64. A5 can select 128
  where its occupancy/layout rules permit. Grouped commit requires A5, FP32 state,
  no align table, `K = V = 128` and `H` in `{32, 64}`; the window is supplied by
  the caller rather than restricted to six tokens. Other supported cases use
  ordinary commit. A3 strided convolution compaction uses 256-channel
  tiles; A5 and contiguous-channel compaction retain 4096-channel tiles.
  Ordinary commit further caps its value tile using the existing
  `triton_utils.get_ub_size_bytes()` result, the key tile and align buffers.
  A3 iterates accepted records at runtime to avoid compiler-expanded matrix
  lifetimes; A5 retains static unrolling for both ordinary and grouped commit.
  The estimate covers two matrix buffers normally and six with align enabled.
  Two rows' worth of vector temporaries
  are reserved before rounding down to a power of two. Context creation initializes
  the shared device-property cache outside graph capture; kernel launch reads its UB budget.
  Missing device properties use the utility's existing
  conservative 192 KiB fallback, including on A5; this is not a hardware-size claim.
  A3 strided compaction copies one history position at a time, preserving exact
  in-place shifts and graph support on Triton-Ascend 3.2.2.
- Create the context before graph capture and retain its buffers and layer
  allocations across replays. Speculative capacity and specialization attributes
  remain fixed for each captured graph; an empty batch is a no-op.
- Hardware evidence records each validated per-device software stack. The
  ordinary commit budget also covers the TP4/H16 case that previously exceeded
  A3 UB capacity with Triton-Ascend 3.2.2.

## Origin and Differences

- **Origin**: Adapted from upstream vLLM KDA RecoverSSM at `c4df37dfcc`, as recorded
  in `vllm_ascend/ops/triton/kda/recoverssm.py`. It implements the paged verify and
  accepted-state commit contracts for Ascend GLM KDA.
- **Differences**:
    - Q/K normalization follows AscendC KDA: no squared-norm epsilon is added to
    nonzero vectors, and zero vectors normalize to zero.
    - FP32 recurrent arithmetic, explicit strided paged records and batched
    multi-layer commit are used for Ascend. Persistent state is not advanced
    during verification.
    - Verify avoids the final unnecessary full-state update. A3 ordinary commit
    reconstructs accepted prefixes with runtime iteration; A5 retains static
    window expansion and can process two heads together.
    - Convolution history is compacted by an Ascend kernel with device/layout-aware
    tiles. Commit strides remain runtime arguments where static shapes are not
    required. There is no autograd implementation (N/A for inference).

## Test Cases

The NPU test file uses an independent CPU FP64 forward recurrence for verify
outputs, FP32 correction records and accepted states. Raw key/gate records and
convolution compaction are compared exactly. AscendC is an additional reference
for zero/small-norm inputs, rather than the only mathematical reference.

Each arithmetic comparison uses the **actual output storage dtype**, computes
errors on CPU in FP64, and enforces both criteria:

| Output dtype | rtol | atol | Fixed maximum absolute error |
| --- | --- | --- | --- |
| FP16 | `2^-9` | `2^-9` | 0.1 |
| BF16 | `2^-6` | `2^-6` | 1 |
| FP32 | `2^-10` | `2^-16` | 0.01 |

At least 99% of elements must satisfy
`abs(actual - golden) <= atol + rtol * abs(golden)`, and **all** elements must have
absolute error at most `max(fixed_limit, 32 * ULP_1.0(output_dtype))`. Non-finite
results/references fail. Pure data movement and unchanged regions require exact
agreement. The small-norm AscendC tests also retain their stronger regression
tolerances. JUnit properties record each comparison's dtype, element counts,
tolerances, pass ratio, maximum error and error limit; comparisons are not pooled
to conceal a failed output.

Coverage includes H32/H64 and K=V128 model dimensions, a recorded B11/L11 compact
shape, FP16/BF16 activations, FP32/BF16 state, windows of 1/2/4/6/8 tokens,
multi-step recovery, sampler-row mappings, page/head padding, metadata strides,
null/empty rows, rejected-all and rejected-suffix records, ragged tiles,
sigmoid/softplus gates, both convolution layouts and exact untouched regions.
The same test file also covers full align commit across/exactly at/before a page
boundary, mixed per-layer convolution layouts, and ordinary/grouped commit with
the model's B11/B12 and L11/L12 pool strides. The L34 cases are capacity stress
tests: actual GLM contexts split the 34 KDA layers into groups of 11/11/12.
Pool-stride cases retain the recorded strides with B+2 pages and independent
per-layer allocations; they do not reproduce every shared-pool address alias.
The criteria themselves test the 99% boundary and independently failing the
ratio or maximum-error gate.
Window regressions explicitly check the selected ordinary/grouped path for
Q1/Q2/Q4/Q6/Q8, both local head counts and eager/graph execution, with additional
FP16 and unbounded-gate cases. This prevents a fixed-window fallback from masking
missing grouped-kernel coverage.

```bash
pytest -sv -o junit_family=xunit1 \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_recoverssm_npu.py \
  --junitxml=kda_recoverssm.xml
```

Earlier validation on 2026-10-28 with verified vLLM `544f182d9` passed on A3
with Triton-Ascend 3.2.2 and 3.6, and on A5 with 3.2.2. The validated
earlier operator source SHA256 (LF line endings) is
`ef436f90cea2db8e251cb1a0de91acd61b586d460c096af5c22634cb082de903`.
Each stack passed all 144 cases with no failures or skips, including H16
eager/graph and align-boundary regressions. Each produced 1280 arithmetic
comparisons. Every comparison independently passed both accuracy gates;
the A3 results below are identical on both tested Triton versions.

| Device | Output dtype | Comparisons | Minimum pass ratio | Maximum absolute error | Error limit |
| --- | --- | --- | --- | --- | --- |
| A3 | BF16 | 252 | 100% | `0.00195286687` | 1 |
| A3 | FP16 | 48 | 100% | `2.28976024e-05` | 0.1 |
| A3 | FP32 | 980 | 100% | `4.11211216e-07` | 0.01 |
| A5 | BF16 | 252 | 100% | `0.00321757621` | 1 |
| A5 | FP16 | 48 | 100% | `1.51970713e-05` | 0.1 |
| A5 | FP32 | 980 | 100% | `4.04177257e-07` | 0.01 |

A3 used torch 2.10.0, torch-npu 2.10.0.post7, and either an isolated
triton-ascend 3.2.2 package or 3.6.0+dev20261013220622. A5 used torch
2.10.0+cpu, torch-npu 2.10.0.post7.dev20261020 and triton-ascend 3.2.2.
The isolated 3.2.2 run does not replace the A3 container's installed package.

Operator accuracy is evaluated against the independent references above.
RecoverSSM off/on changes the model's algorithm path and is not a before/after
comparison of the same Triton optimization. It is therefore not used to judge
whether that optimization preserves numerical accuracy. For the runtime-stride
change, the same-algorithm microbenchmarks below additionally compared all
committed checkpoint tensors: the maximum before/after absolute difference was
zero in all 24 A3/A5, L11/L12, B11/B12 and accepted-length cases. This checkpoint
comparison complements, rather than replaces, the independent golden checks.

Single-operator graph microbenchmarks compared the candidate with `b5326cd6b`
(before runtime-stride changes), alternating both implementations for nine
rounds on B11/B12, L11/L12 and accepted lengths 1/3/6. Commit latency changes
ranged from -5.30% to +5.39% on A3 and -1.49% to +3.62% on A5. Event-averaged
timings include graph submission gaps and are distinct from pure kernel time.

Separately, the RecoverSSM algorithm-switch comparison used GLM-5.3-Flash W8A8
on A3 (DP8/TP2) and local MXFP8 on A5 (DP8/TP1), MTP5 and FULL_DECODE_ONLY.
At actual decode concurrency 90, three-trial median TPOT changed from 34.5218
to 33.0748 ms/token on A3 and 21.1032 to 20.9797 on A5; A5 trial ranges overlap.
Full GSM8K mixed-prefill workload TPOT regressed by 3.26% and 1.62%, respectively.
These off/on results describe algorithm integration and are not attributed to
the runtime-stride optimization.

The 2026-10-27 dataset audit covered all four active RecoverSSM kernels, recording
shape/dtype/stride, scalar specialization parameters and compiled identifiers.
After warmup, GSM8K introduced no JIT memory misses or new/changed persistent
compiler artifacts: A3 retained five variants and A5 six. This is a warm-cache
result for the tested configuration, not a guarantee for arbitrary shapes or
low-level align serving. Explicit specialization attributes are listed above;
runtime strides may still trigger Triton type/alignment specialization.

### Extracted-source validation

This split preserves the latest operator and tests from PR #19133 at
`423bc7e0da6f1d17699a7d4bfa6d2438506101bc`. The operator source SHA256
(LF line endings) is
`4e0e6b174d07931f56df9a69a0ad2e9714808fa23f6bb1e204fe019680b7d1fa`.
The same source passed 144/144 A3 cases and 1280 arithmetic comparisons,
with a 100% mixed-tolerance pass ratio for every comparison and the same A3
maximum errors listed above. The current A5 sequence stopped during the
commit microbenchmark because the device ran out of memory; it did not
complete the full accuracy suite. Earlier A5 results above apply to their
recorded source, rather than proving this extracted version passed.
Model dataset results and compilation audits above are historical integration
evidence; the operator-only split does not enable or revalidate serving.

### Configured-window regression

Both commit kernels require the caller's `SPEC_QUERY_LEN`; neither supplies a
six-token default. Grouped dispatch and compilation have no six-token-only
condition. The configured positive window bounds A3 runtime iteration
and controls A5 `tl.static_range`.
The 30 path cases check Q1/Q2/Q4/Q6/Q8, H16/H32/H64, eager/graph execution,
FP16/BF16 activations, bounded/unbounded gates, null/zero-acceptance rows,
accepted-prefix recovery, and the actual selected kernel.

For the original Q6 configuration, six A5 B11/B12 and accepted-length 1/3/6
cases were measured against `b7ec33b35` using the same alternating graph
microbenchmark. Committed checkpoints were identical; grouped commit latency
changes ranged from -0.52% to +0.21%. The arithmetic bodies and Q6
dispatch are unchanged; the source comparison is recorded with the tests.

### UB compatibility regression

The original TP4/H16 ordinary commit required 260 KiB on A3 with
Triton-Ascend 3.2.2, exceeding its 192 KiB UB. Accepted-prefix runtime
iteration and the shared UB budget fix that compilation failure without
halving the normal A3 value tile. The A3 strided-copy path is also validated
for exact results and graph capture. A5 retains static commit expansion.

Final-source graph microbenchmarks against `e6ac48c00` use B12, L11, Q6,
K=V128, FP32 checkpoints/corrections and BF16 activations. Each value is
the median of nine alternating samples, with eight calls per graph capture
and eight replays per timed sample. A3 uses Triton 3.6 for a valid pre-fix
comparison; A5 uses 3.2.2. H32/H64 use recorded serving strides. H16 keeps
those padded strides to exercise ordinary A5 commit. Checkpoints are
bitwise identical in all nine cases.

| Device/path | H | Accepted steps | Before (us) | After (us) | Latency change |
| --- | --- | --- | --- | --- | --- |
| A3 ordinary | 32 | 1 | 705.819 | 574.976 | -18.54% |
| A3 ordinary | 32 | 3 | 962.567 | 763.802 | -20.65% |
| A3 ordinary | 32 | 6 | 1290.735 | 1135.232 | -12.05% |
| A5 grouped | 64 | 1 | 375.868 | 376.889 | +0.27% |
| A5 grouped | 64 | 3 | 478.538 | 479.389 | +0.18% |
| A5 grouped | 64 | 6 | 744.741 | 745.046 | +0.04% |
| A5 ordinary | 16 | 1 | 93.820 | 93.595 | -0.24% |
| A5 ordinary | 16 | 3 | 130.539 | 131.881 | +1.03% |
| A5 ordinary | 16 | 6 | 189.493 | 190.258 | +0.40% |

Including plan and compact, context latency changes range from -20.03% to
-10.44% on A3, -0.11% to +0.19% for A5 grouped, and -1.34% to +0.81%
for A5 ordinary. These are operator graph measurements; serving TPOT/ITL
was not remeasured for this UB compatibility change. `UNROLL_COMMIT` and
`USE_1D_CHANNEL_GATHER` are fixed by device/layout; UB selection reads the shared
device-property cache and does not create a new per-iteration compilation parameter.

## Example

The multi-layer and multi-step cases in
`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_recoverssm_npu.py`
show how to allocate page views, call `kda_recoverssm_verify`, construct a commit
context and commit accepted prefixes. Model activation and configuration will be
provided by the framework follow-up. This operator test does not establish
model dataset accuracy or serving performance.
