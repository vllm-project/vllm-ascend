# GDN RecoverSSM

## Description

- **Function**: Verify a speculative Gated DeltaNet (GDN) window from a persistent
  recurrent checkpoint. Return per-token outputs and record the effective
  recurrence inputs. After sampling, reconstruct the accepted prefix and
  optionally compact the convolution history prepared by the caller.
- **Formula**: The forward recurrence and accepted-prefix reconstruction are
  given below.
- **Algorithm flow**: Verify runs independently per request and value head,
  with value-dimension tiling. Commit first plans the destination pages, then
  optionally copies convolution history, and finally updates the checkpoint
  for each layer bound to the context.
- **Supported modes**: The implementation has an Ascend 950 path and an ordinary
  non-950 path. The test suite defines eager verification, eager commit, and
  captured commit replay. This source review does not establish hardware
  qualification for Atlas A2, Atlas A3, 950PR, or 950DT.

The operator interfaces are defined in
[`gdn/recoverssm_gdn.py`](../gdn/recoverssm_gdn.py#L585).
The verify wrapper and commit context can be exercised directly with the
single-device operator suite. Qwen model integration is a follow-up to this
operator change.

### Recurrence

For one value head, let `S` be `[V, K]`. Its query/key head is
`value_head // (HV / HK)`. The gate is scalar per value head:

```text
x = a + dt_bias
softplus_x = log(1 + exp(softplus_beta * x)) / softplus_beta
             if softplus_beta * x <= softplus_threshold else x
g = -exp(A_log) * softplus_x
d = exp(g)
beta = sigmoid(b)

q_hat = q * rsqrt(sum(q * q) + 1e-6)  # if normalization enabled
k_hat = k * rsqrt(sum(k * k) + 1e-6)  # if normalization enabled
q_scaled = q_hat * scale
D = S * d
u = beta * (v - D @ k_hat)
S_next = D + u[:, None] * k_hat[None, :]
output = S_next @ q_scaled
```

Without normalization, `q_hat=q` and `k_hat=k`. Default scale is `K**-0.5`.
State arithmetic and normalized keys are FP32; normalized keys are not rounded
back to activation dtype before recording. Outputs are stored in the output
tensor's dtype. BF16 checkpoint storage rounds the reconstructed FP32 result.

For accepted length `c`, state reconstruction uses reverse accumulation:

```text
F = 1; R = 0
for t = c - 1 down to 0:
    R += u[t][:, None] * (k_hat[t] * F)[None, :]
    F *= d[t]
S_accepted = S_checkpoint * F + R
```

This is algebraically the forward recurrence, but changes floating-point
evaluation order; bitwise equality with a forward implementation is not promised.

### Algorithm flow

1. [Verify](../gdn/recoverssm_gdn.py#L585) advances a temporary state, writes
   token outputs and replay records, and leaves the checkpoint unchanged.
2. After sampling, [`prepare_commit_plan`](../mamba/recoverssm.py#L228) maps each
   speculative row to its sampler row and writes the accepted lengths and
   destination page IDs. Accepted lengths count recurrent steps, including the
   target step where applicable.
3. [`compact_conv_state`](../mamba/recoverssm.py#L277) copies the accepted
   convolution history when `commit_conv=True`.
4. [State commit](../gdn/recoverssm_gdn.py#L1043) reads the accepted records
   backwards and writes final and optional boundary checkpoints across all
   bound layers.

### Kernel launches

| Stage | Grid / strategy |
| --- | --- |
| Verify | `(ceil(V / BV), HV, B)`; sequential recurrence within each request |
| Commit plan | `(B,)`, one program per request, one warp |
| Convolution | `(ceil(C / BLOCK_D), B, L)`, layer-specific address/stride tables |
| State commit | Flat grid of `ceil(V / BV) * L * head_groups * B` programs |

Verify uses `BK=next_power_of_2(K)` and normally
`BV=min(next_power_of_2(V), 64)`. The Ascend 950 branch can select a 128-wide value
tile when its occupancy checks retain at least 96 programs. Verify uses four
warps and disables FP fusion. Shapes, inner strides, head geometry and activation
options participate in specialization; variable metadata values remain on device.

Ordinary state commit processes one head per program. The Ascend 950 grouped path
handles two adjacent heads when `K=V=128` and `HV` is even and at least two. Unlike
the KDA selection policy, this GDN selection is not restricted to FP32 checkpoints
or `HV=32/64`. The ordinary non-950 path uses a runtime reverse loop; the 950 path
can statically unroll the window.

State value tiles are limited by the reported UB capacity. The budget uses two
matrix buffers without align and six with align, FP32 element size, key tile
width, the grouped head factor, and two rows of headroom. This is a launch-sizing
heuristic, not a measurement of actual compiler allocation or occupancy.

Convolution uses a 4096-channel tile on the 950 branch or with shared unit channel
stride; otherwise it uses 256. Mixed per-layer strides use the address/stride
table. The non-950 path supports the one-dimensional channel gather strategy.

Although the shared planner produces state-only and boundary length arrays, the
current GDN launcher uses the full accepted-length array for a single state
commit launch. It does not use KDA's two disjoint filtered state launches.

## Parameters

`B` is the number of speculative rows, `T` the flattened token count, `P`
the page count, `Q` the speculative window capacity, `HK/HV` the local
query-key/value head counts, `K/V` the key/value widths, `L` the bound layer
count, and `W` the retained convolution history. The number of convolution
channels is `C = 2 * HK * K + HV * V`.

### `gdn_recoverssm_verify`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `A_log`, `dt_bias` | Input | Per-value-head log decay scale and gate bias, each `[HV]` | Floating point, converted to FP32 in the kernel | Contiguous 1D |
| `a`, `b` | Input | Raw gate and beta inputs, each with `T * HV` elements, reshaped to `[T, HV]` | Floating point, converted to FP32 in the kernel | ND with unit head stride after reshape |
| `q`, `k` | Input | Query and key, each `[1, T, HK, K]` | BF16 in the NPU tests, converted to FP32 in the kernel | ND with inner strides `(K, 1)` |
| `v` | Input | Values `[1, T, HV, V]` | BF16 in the NPU tests, converted to FP32 in the kernel | ND with inner strides `(V, 1)` |
| `checkpoint_state` | Input | Persistent state `[P, HV, V, K]`, read during verification | FP32 or BF16 | Paged ND with explicit strides |
| `replay_cache` | Output | Caller-supplied buffer `[P, HV, Q, V + K + 1]`, written for valid pages and tokens | FP32 | Paged ND with explicit strides |
| `query_start_loc` | Input | Cumulative token offsets `[B + 1]` | INT32 or INT64 | Strided 1D |
| `state_indices` | Input | Source page IDs `[B]`. Nonpositive IDs select inactive rows | INT32 or INT64 | Strided 1D |
| `spec_query_len` | Attribute | Window capacity `Q` | Python int | Positive scalar |
| `scale` | Attribute | Query scale. Default `None` selects `K**-0.5` | Python float or `None` | Scalar / N/A |
| `softplus_beta` | Attribute | Positive softplus beta, default `1.0` | Python float | Scalar |
| `softplus_threshold` | Attribute | Threshold for the softplus linear branch, default `20.0` | Python float | Scalar |
| `use_qk_l2norm_in_kernel` | Attribute | Whether to normalize Q/K, default `True` | Python bool | Scalar |
| `out` | Output | Returns the supplied output buffer. Default `None` allocates a contiguous tensor with `v`'s shape and dtype | Supplied buffer dtype or `v.dtype` | `[1, T, HV, V]` with inner strides `(V, 1)` |

For each page, head and token, the FP32 replay record is:

| Slice | Stored value |
| --- | --- |
| `[..., :V]` | Correction `u` |
| `[..., V:V + K]` | Effective key `k_hat`, repeated across associated value heads |
| `[..., V + K]` | Effective scalar decay `d = exp(g)`, not the log gate |

Only the first value tile writes key and decay; value tiles write disjoint
correction slices. No full `[V, K]` state is stored per speculative token.
Replay payload per page is `4 * HV * Q * (V + K + 1)` bytes, excluding checkpoint,
convolution, outputs and metadata. This is a storage formula, not measured memory.

### `GDNRecoverSSMCommitContext.create`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `layers` | Input | Nonempty sequence of layers exposing the fields below | Python sequence | N/A |
| `spec_query_len` | Attribute | Shared window capacity `Q` | Python int | Positive scalar |
| `max_num_reqs` | Attribute | Capacity of the reusable commit plan | Python int | Positive scalar |
| Returned context | Output | Binds layer-owned cache views and allocates launch metadata | Context object | N/A |

| Layer field | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `kv_cache[0]` | Input/Output | Convolution history with logical shape `[P, C, Q + W - 1]` | BF16 in the NPU tests | Paged ND, normalized from channel-first or token-first |
| `kv_cache[1]` | Input/Output | Recurrent checkpoint `[P, HV, V, K]` | FP32 or BF16 | Paged ND |
| `kv_cache[2]` | Input | Verification records `[P, HV, Q, V + K + 1]` | FP32 | Paged ND |
| `num_k_heads`, `tp_size` | Attribute | Derive local head count as `HK = num_k_heads // tp_size` | Python int | Scalar |

The context retains cache references and preallocates:

- An INT64 `[2, 2, L]` state/replay address-and-page-stride table.
- An INT64 `[4, L]` convolution address, page, channel and token stride table.
- An INT32 `[6, max_num_reqs]` plan buffer, viewed as accepted lengths,
  state-only lengths, boundary lengths, final pages, boundary pages and boundary
  recovery lengths.

### `GDNRecoverSSMCommitContext.commit`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `num_accepted_tokens` | Input | Accepted recurrent steps per sampler row | INT32 in the NPU tests | Strided 1D |
| `state_indices` | Input | Source page IDs in speculative-row order | INT32 or INT64 | Strided 1D, `[B]` |
| `query_start_loc` | Input | Token offsets in speculative-row order | INT32 or INT64 | Strided 1D, `[B + 1]` |
| `request_indices` | Input | Speculative-row to sampler-row mapping. Default `None` uses identity mapping | INT32 in the NPU tests or `None` | Strided 1D / N/A |
| `block_table` | Input | Sampler-row to page table for align mode. Default `None` commits to the source page | INT32, INT64, or `None` | Strided 2D / N/A |
| `num_computed_tokens` | Input | Prior computed lengths in sampler-row order for align mode, default `None` | INT32 in the NPU tests or `None` | Strided 1D / N/A |
| `mamba_block_size` | Attribute | Token span of an align block, at least `Q`, default `None` | Python int or `None` | Scalar / N/A |
| `commit_conv` | Attribute | Whether to compact convolution history, default `True` | Python bool | Scalar |
| Context checkpoint and convolution pages | Input/Output | Commit updates destination checkpoints and, when enabled, convolution histories in place. The method returns `None` | See layer fields | Paged ND |
| Context plan arrays | Output | Internal accepted lengths, final and boundary page IDs, and boundary recovery lengths | INT32 | Contiguous rows of the reusable plan buffer |

## Constraints

- All tensors in a call share a device. The verify wrapper checks metadata
  dtype, checkpoint and replay dtype, and activation/output inner strides.
  Callers supply contiguous `A_log` and `dt_bias` because the kernel indexes
  them directly by head. The wrapper checks output shape and strides.
- Query offsets start at zero, end at `T`, and are nondecreasing. Every request
  span is at most `Q`. The wrapper checks aggregate capacity `T <= B * Q`.
  Positive page IDs and sampler mappings must be in range.
- Bound layers share cache shapes, corresponding dtypes and device, local head
  geometry, and state/replay inner stride signatures. Page strides and
  per-layer convolution inner strides may differ. The context requires
  `W > 0` and initializes NPU device properties before graph capture.
- A nonempty query on a null page produces zero output and leaves replay
  unchanged. Empty verification returns its output buffer without launching.
  Verify must finish before commit consumes records from the same checkpoint
  and window. The caller prevents cross-request cache read/write conflicts.
- Cache storage and metadata addresses stay live during execution and graph
  replay. Rebuild the context if a bound cache is reallocated. Order accesses
  to its mutable plan and pages on the appropriate streams. Autograd is N/A.

### Commit destinations

The three align arguments must be all present or all absent. Tensor metadata must
be on the cache device. Counts are clamped to
`c = max(0, min(count, query_length, Q))`. Null source pages and zero counts are
no-ops. Without align, the destination is the source page.

With align, let `n` be prior computed length and `m` block size. The shared planner
chooses final table column `max(n + c - 1, 0) // m`, capped to the last available
column. The next boundary is `(n // m + 1) * m`. If reached, it also reconstructs
the prefix of length `boundary - n` into column `boundary // m - 1`.
At an exact boundary, final and boundary destinations coincide; the next block's
page is not advanced merely because the preceding block has just completed.

For accepted length `c > 0`, convolution compaction copies source
`[c - 1 : c - 1 + W]` into destination `[0 : W]`; the optional boundary copy uses
its own accepted length. Convolution contents must already represent the verified
window. Turning compaction off does not disable recurrent reconstruction.

## Origin and Differences

- **Origin**: Ascend implementation of the paged GDN RecoverSSM verify and
  accepted-prefix commit contract. The raw-gate interface and page layout
  follow [upstream vLLM GDN RecoverSSM](https://github.com/vllm-project/vllm/pull/61309).
- **Differences**: The Ascend kernels use explicit page strides, UB-bounded
  value tiles, and a two-head grouped commit path on Ascend 950 for eligible
  geometry. Batched commit uses Ascend-specific launch sizing and shared
  convolution compaction.
- **Relationship to KDA**: GDN uses the planner and convolution compactor in
  [`mamba/recoverssm.py`](../mamba/recoverssm.py). GDN stores an effective key
  and scalar decay for each value head. The existing
  [KDA RecoverSSM](kda_recoverssm.md) implementation continues to use its own
  planning, convolution compaction, and raw key/gate records.

## Test Cases

The [single-device test suite](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py) uses synthetic tensors with
model-derived shapes. The module requires `torch_npu`. Device cases also
require a usable Ascend NPU and the matching PyTorch, CANN, and Triton stack.

### Reference implementations

- [`cpu_reference`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L122) evaluates the forward GDN recurrence on CPU in
  FP64. It expands grouped Q/K heads, applies the configured normalization and
  raw gate transforms, and returns token outputs, per-step checkpoints, and
  packed records. The tests compare correction, effective key, and decay
  separately, and use the checkpoint at the accepted step as the commit target.
- The AscendC comparison runs `recurrent_gated_delta_rule` one token at a
  time. Both paths receive the same pre-normalized BF16 Q/K, with internal
  verify normalization disabled. The CPU FP64 cases separately exercise
  internal normalization.
- The packed-page commit test sets effective key and correction to zero and
  decay to `0.75`. After `c` accepted steps, the expected checkpoint is
  `S_initial * 0.75**c`. Expected convolution histories come from direct
  slices of the original cache.

### Accuracy criteria

[`check_accuracy`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L24) checks shape and finiteness, then computes error on
CPU in FP64. Tolerances use the actual output's storage dtype.

| Output dtype | rtol | atol | Fixed floor for the maximum absolute error limit |
| --- | --- | --- | --- |
| FP16 | `2^-9` | `2^-9` | `0.1` |
| BF16 | `2^-6` | `2^-6` | `1.0` |
| FP32 | `2^-10` | `2^-16` | `0.01` |

Each arithmetic comparison requires at least 99% of elements to satisfy
`abs(actual - expected) <= atol + rtol * abs(expected)`, and the maximum
absolute error must be at most
`max(floor, 32 * torch.finfo(actual.dtype).eps)`. The FP16 row also serves
the checker self-test. Model-shaped verify cases use BF16 activations, BF16
or FP32 checkpoints, and FP32 replay. Convolution copies, padding, and cache
regions expected to stay unchanged use exact checks where asserted. The
checker records per-comparison metrics as `accuracy/<label>` JUnit properties.

### Coverage

| Test function | Cases | Verification performed |
| --- | ---: | --- |
| [`test_accuracy_gates`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L57) | 3 | Self-tests the 99% boundary and independent worst-element limit for FP16, BF16, and FP32. |
| [`test_multilayer_verify_and_commit_across_steps`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L204) | 2 | Two layers and two rounds, sampler remapping, query-span clamping, rejected-suffix NaNs, null output, and page padding. |
| [`test_head_padding_survives_verify_and_commit`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L253) | 1 | Padded head strides, strided source indices, and exact checks on padding and untouched checkpoint pages. |
| [`test_compaction_page_routing`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L290) | 4 | Direct convolution copy with 127/2048/2560/5120 channels, both physical layouts, in-place/final/boundary destinations, and whole-backing checks. |
| [`test_verify_ragged_tiles_and_small_norms`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L354) | 6 | Masked K/V tiles, small and zero Q/K norms, custom gate parameters, strided INT64 offsets and supplied output, and commit with convolution disabled. |
| [`test_small_norms_against_ascendc`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L402) | 4 | Single-token AscendC recurrence as an additional reference for verify and accepted prefixes of 1, 3, and 5 steps. |
| [`test_empty_and_null_rows`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L453) | 3 | Empty batch and page IDs 0 or -1. Checks zero/empty outputs and an unchanged cache backing. |
| [`test_context_rejects_invalid_capacity_or_no_layers`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L468) | 3 | Rejects zero window or request capacity and an empty layer list. |
| [`test_align_final_and_boundary_routes`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L486) | 5 | Crossing, exact boundary, relocation, in-place, and zero acceptance, with strided metadata, mixed convolution layouts, and eager or graph commit. |
| [`test_model_packed_page_commit`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L552) | 2 | Per-rank model shapes, a shared byte backing per layer, and a closed-form decay check over 30 or 48 layers. |
| [`test_configured_window_dispatch`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L610) | 6 | Windows 1/2/4/5/9, verify tile and grid checks, ordinary/grouped dispatch, commit UB budget, and eager or graph commit. |
| **Total static expansion** | **39** | Runtime collection and skips depend on the environment. |

Both target models use `K=V=128` and convolution width four, retaining
three history slots. The model-shaped cases use these per-rank geometries.

| Model | TP | Local HK/HV | Speculative window in representative cases |
| --- | ---: | --- | --- |
| Qwen3.6-35B-A3B | 4 | 4/8 | 5, corresponding to MTP with four draft tokens |
| Qwen3.8-27B | 2 | 8/24 | 9, corresponding to DFlash with eight draft tokens |
| Qwen3.8-27B | 4 | 4/12 | 9 in the packed-page test, 5 in the align test |

The packed-page tests use synthetic page strides and padding with convolution,
checkpoint, and replay views over one backing per layer. The graph cases
capture commit, restore the original backing, and replay the graph. Verify,
where used, runs eagerly. The suite describes single-rank operator coverage.

### Test commands

From the repository root in the Ascend test environment, run the full suite
with JUnit properties or select the exact-boundary case.

```bash
pytest -sv -o junit_family=xunit1 \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py \
  --junitxml=gdn_recoverssm.xml

pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py \
  -k 'test_align_final_and_boundary_routes and exact'
```

### Recorded NPU results

The JUnit report `gdn_recoverssm.xml` from the 2026-11-13 run
recorded **39 passed**, **0 failed**, **0 errors**, and **0 skipped** in
**84.782 s**. The remote checkout inspected with the report was
`3fa454efbfefb996611d124ba200082915d1398d`. The named test container used
an Ascend A3 OpenEuler image with CANN 9.2.1.

| Output dtype | Operator comparisons | Minimum mixed-tolerance pass ratio | Maximum absolute error | Absolute error limit |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 45 | 100% | `3.3789673e-3` | `1.0` |
| FP32 | 246 | 100% | `3.1493371e-7` | `0.01` |

The table summarizes the 291 operator comparisons recorded as
`accuracy/<label>` properties. The three passing accuracy-checker self-test
cases intentionally construct failing comparisons, so their diagnostic metrics
are excluded from the table. The 39-case suite includes these self-tests,
configuration checks, and eager and graph operator cases. The duration is the
pytest suite time.

#### Multi-step recovery and model-shaped pages

The values below are maximum absolute errors across the recorded comparisons
for each case. Verify output was BF16 and replay records were FP32. All
listed arithmetic comparisons had a 100% mixed-tolerance pass ratio.

| Two-layer, two-round case | Verify output (4 comparisons) | Replay fields (24 comparisons) | Committed checkpoint (4 comparisons) |
| --- | ---: | ---: | ---: |
| Qwen3.6-35B-A3B TP4, MTP4 | `1.0540717e-4` | `2.7508018e-7` | `1.1920929e-7` (FP32) |
| Qwen3.8-27B TP2, DFlash8 | `1.2110923e-4` | `3.1493371e-7` | `1.9531250e-3` (BF16) |

The packed-page tests compared each layer's FP32 checkpoint against the
closed-form decay reference. Qwen3.6 TP4 used 11 requests and 30 layers in
eager mode. Qwen3.8 TP4 used 12 requests and 48 layers with graph replay.
Both recorded a maximum checkpoint error of `1.8626451e-9`, and their exact
checks on convolution histories, untouched pages, replay data, and page
padding passed.

#### Align routing

Each row summarizes the full checkpoint comparisons for both layers. In the
zero-acceptance case, the other request still commits, so the row reports a
batch-level maximum.

| Route | Commit mode | Checkpoint dtype | Maximum absolute checkpoint error |
| --- | --- | --- | ---: |
| Cross a boundary | Eager | FP32 | `1.1920929e-7` |
| End exactly at a boundary | Graph replay | BF16 | `4.8828125e-4` |
| Relocate within a block | Eager | BF16 | `4.8828125e-4` |
| Update in place | Graph replay | FP32 | `1.1920929e-7` |
| Zero acceptance for one request | Eager | FP32 | `1.1920929e-7` |

All five cases passed the exact convolution and untouched-checkpoint checks.

#### Device dispatch and small-norm inputs

The six configured-window cases recorded `_commit_gdn_state_kernel` on
this run. Their verify output error was at most `2.4221478e-4` (BF16).
The maximum commit error was `1.1920929e-7` for FP32 checkpoints and
`9.7656250e-4` for BF16 checkpoints. The recorded dispatch result covers
the ordinary kernel on A3.

For the AscendC comparison, Q/K scaled by `1e-8` gave a BF16 verify
error of `2.2737368e-13` and a maximum FP32 commit error of
`7.4505806e-9` across accepted prefixes of 1, 3, and 5 steps. With zero
Q/K, the verify error was zero and the maximum commit error was again
`7.4505806e-9`.

## Example

For `W=3` and a verified five-token window, an accepted length of two commits
records at offsets 0 and 1. The convolution copy takes source slots `[1:4]`
and writes them into destination slots `[0:3]`. In align mode with six prior
tokens and block size eight, the final column is `(6 + 2 - 1) // 8 = 0`.
The boundary is reached at the same accepted length, so the recurrent kernel
writes one checkpoint to the shared final/boundary page. The next page is
left untouched.

The
[multi-layer test](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L204)
shows how to allocate paged caches, run verify, bind a commit context, and
commit accepted lengths with a sampler-row mapping.
