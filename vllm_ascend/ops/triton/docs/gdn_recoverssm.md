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
For Model Runner V2, the GDN attention layer calls verify and the GDN
attention metadata calls the commit context after sampling.
See the [Qwen GDN integration guide](../../../../docs/source/developer_guide/Design_Documents/qwen_gdn_recoverssm.md)
for serving configuration.

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
- **Relationship to KDA**: Both use the planner and convolution compactor in
  [`mamba/recoverssm.py`](../mamba/recoverssm.py). GDN stores an effective key
  and scalar decay for each value head. KDA stores raw key/gate data and uses
  its own commit tiling, request filters and accuracy checks. See
  [KDA RecoverSSM](kda_recoverssm.md).

## Test Cases

The numerical operator suite is
[`test_gdn_recoverssm_npu.py`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py).
It requires `torch_npu`, Triton/NPU dependencies and usable NPU hardware. It does
not load a full language model. The following counts are static parameter
expansions, not collected, executed or passing counts.

| Test function (without `test_`) | Cases | Verification performed |
| --- | ---: | --- |
| `verify_and_commit_support_padded_inner_page_strides` | 1 | Verify and two-layer commit with padded layouts and differing page strides; irregular `K=33,V=47` |
| `commit_without_conv_still_updates_checkpoint` | 1 | State updates while convolution storage stays exactly unchanged |
| `commit_context_rejects_mismatched_inner_page_strides` | 1 | Incompatible state/replay inner layouts are rejected |
| `verify_and_multilayer_commit_preserve_accepted_prefix` | 6 | Qwen3.6/Qwen3-Next and Qwen3.5 head geometries, MTP windows 2/4/5, zero/negative/partial/full/oversized counts, two layers and rounds, page permutation and sampler remapping |
| `align_commit_restores_boundary_and_final_pages` | 8 | Crossing, exact, relocation, in-place, zero, and full cases, with eager and graph commit, mixed convolution strides, and strided INT64 align metadata |
| `reverse_commit_grouped_and_ordinary_match_forward_reference` | 5 | A5-only ordinary/grouped comparison across three head ratios, including exact boundary and full 16-token windows |
| `verify_raw_interface_defaults_custom_options_and_out` | 2 | Allocated and strided caller-supplied output, strided activations and INT64 offsets, zero-norm Q/K, default and custom gate options |
| `empty_verification_does_not_launch_or_mutate` | 1 | Empty verify preserves buffers |
| `nonempty_capture_rows_with_null_pages_produce_zero_outputs` | 1 | Null pages yield zero outputs without replay/checkpoint mutation |
| **Total static expansion** | **26** | Device skips and collection failures can change actual execution counts |

Representative model shapes use the local key/value head counts `4/8` and
`16/32` from Qwen3.6-35B-A3B or Qwen3-Next-80B-A3B at TP4 and TP1, and
`4/12` from Qwen3.5-27B at TP4. Both head widths are 128, and the
convolution kernel width is four (three retained history tokens).
These single-device tests reproduce per-rank tensors and do not run TP collectives.

### Reference and test procedure

The main `reference` function runs a PyTorch forward recurrence in FP32 on the
input NPU, explicitly expands grouped keys/queries and saves every timestep's
state. It is not a CPU FP64 reference. The activated-input helper transforms gates
and beta back to raw inputs using inverse softplus and logit before invoking the
raw operator; this round trip can itself introduce floating-point differences.
The dedicated raw-interface test instead constructs its reference directly from
the raw gate, activation and normalization settings.

Accepted-prefix tests initialize checkpoints and convolution histories, fill
replay storage with NaNs, compute the forward reference, and run verification.
They compare outputs and selected replay fields, confirm checkpoints remain
unchanged, then commit mapped acceptance counts and compare complete destination
states/histories. Zero acceptance, inactive rows and untouched pages must retain
their previous contents. A second round checks cache reuse and page permutation.
After verification, the test also fills the rejected suffix with NaNs before
commit to check that only accepted records are consumed.

Align tests construct expected final and boundary pages, exercise mixed
convolution strides and observe the selected state kernel. In graph mode only
commit is captured: verification is eager. State and convolution buffers are
restored before graph replay. This does not validate an entire captured
verify-sample-commit model execution.

The ordinary/grouped comparison manually constructs replay records and computes
forward reconstruction with NPU double tensors. This is a separate higher
precision reference for commit, not a CPU FP64 end-to-end verify oracle.

| Comparison | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: |
| Verify output versus main FP32 reference | `3e-3` | `2e-2` |
| FP32 committed state | `3e-5` | `3e-4` |
| BF16 committed state | `2e-3` | `1e-2` |
| Recorded decay | `2e-3` | `2e-3` |
| Recorded key, copied convolution and unchanged data where checked | `0` | `0` |

Assertions use `torch.testing.assert_close`, not KDA's separate aggregate
accuracy metrics. BF16 expected states are stored through BF16 reference buffers.
There is no independent direct comparison of every correction record in the
main accepted-prefix test; correction correctness is exercised through commit.

The exact-boundary case in
[`test_align_commit_restores_boundary_and_final_pages`](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L304)
uses `(n + count - 1) // block_size` for the expected final page, matching the
[shared planner](../mamba/recoverssm.py#L101). These checks are test definitions.
No NPU numerical, graph, or performance run was performed for this document update.

Run from the repository root in a supported Ascend environment with matching
PyTorch, `torch_npu`, CANN and Triton dependencies:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py --junitxml=gdn_recoverssm.xml
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py -k 'align and exact'
```

## Example

For `W=3` and a verified five-token window, an accepted length of two commits
records at offsets 0 and 1. The convolution copy takes source slots `[1:4]`
and writes them into destination slots `[0:3]`. In align mode with six prior
tokens and block size eight, the final column is `(6 + 2 - 1) // 8 = 0`.
The boundary is reached at the same accepted length, so the recurrent kernel
writes one checkpoint to the shared final/boundary page. The next page is
left untouched.

The
[multi-layer test](../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py#L203)
shows how to allocate paged caches, run verify, bind a commit context, and
commit accepted lengths with a sampler-row mapping.
