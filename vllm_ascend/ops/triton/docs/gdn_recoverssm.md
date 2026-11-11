# GDN RecoverSSM

## Description

This document specifies the Triton operators in
[`gdn/recoverssm_gdn.py`](../gdn/recoverssm_gdn.py), including verification,
accepted-prefix state reconstruction, convolution-history compaction, and their
current validation coverage. The implementation baseline is commit
`33c9dff99701fd64dd43946c3a65effcefd6dd97`.

**Validation status:** this report is based on source inspection. No NPU accuracy,
graph, or performance run was performed to produce this document. Existing test
assertions are described as test specifications, not reported passing results.
An exact-boundary discrepancy in the align test oracle remains unresolved below.

### Function and recurrence

Verification advances a temporary recurrent state over a speculative window,
emits outputs, and saves compact per-token records without changing the persistent
checkpoint. After sampling, commit reconstructs only the accepted prefix from
the original checkpoint and those records. The rejected suffix does not become
the next persistent state.

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

### Execution flow

1. `gdn_recoverssm_verify` reads a checkpoint, writes outputs and replay records.
2. The caller samples and provides accepted lengths, including the target
   recurrent step where applicable; these are not simply accepted-draft counts.
3. The shared commit-plan kernel resolves counts and destination pages.
4. The shared convolution kernel compacts the caller-prepared convolution history,
   unless `commit_conv=False`.
5. The GDN state kernel reconstructs final and optional boundary checkpoints for
   all layers bound to the context.

The caller owns convolution computation/history preparation. These operators do
not perform the model's convolution, draft generation, sampling, or page allocation.
Operator align support alone does not establish model-level prefix-cache or PD
support. See the separate
[Qwen GDN integration guide](../../../../docs/source/developer_guide/Design_Documents/qwen_gdn_recoverssm.md)
for model configuration; this report does not certify those integration paths.

## Parameters

Notation: `B` speculative requests, `T` flattened tokens, `P` pages, `Q` window
capacity, `HK/HV` local query-key/value heads, `K/V` head widths, `L` layers,
`W` retained convolution history, and `C = 2 * HK * K + HV * V` channels.

### Verification interface

```python
gdn_recoverssm_verify(
    A_log, a, b, dt_bias, q, k, v,
    *, checkpoint_state, replay_cache, query_start_loc, state_indices,
    spec_query_len, scale=None, softplus_beta=1.0,
    softplus_threshold=20.0, use_qk_l2norm_in_kernel=True, out=None,
)
```

| Parameter | Shape / meaning | Contract |
| --- | --- | --- |
| `q`, `k` | `[1, T, HK, K]` | Inner strides `(K, 1)`; token stride may vary |
| `v` | `[1, T, HV, V]` | Inner strides `(V, 1)`; `HV` divisible by `HK` |
| `a`, `b` | `T * HV` elements, reshaped to `[T, HV]` | Raw gate and beta inputs; resulting head stride is one |
| `A_log`, `dt_bias` | `[HV]` | Per-value-head parameters; kernel assumes contiguous storage |
| `checkpoint_state` | `[P, HV, V, K]` | FP32 or BF16; verify does not modify it |
| `replay_cache` | `[P, HV, Q, V + K + 1]` | FP32; stores effective recurrence data |
| `query_start_loc` | `[B + 1]` | INT32/INT64 cumulative token offsets |
| `state_indices` | `[B]` | INT32/INT64 source pages; nonpositive pages are inactive |
| `spec_query_len` | `Q` | Positive replay capacity |
| `scale` | Scalar or `None` | Defaults to `K**-0.5` |
| `softplus_beta`, `softplus_threshold` | Scalars | Gate activation controls; use meaningful positive beta |
| `use_qk_l2norm_in_kernel` | Boolean | Epsilon-based FP32 normalization |
| `out` | Optional tensor shaped like `v` | Same device and required inner strides; otherwise allocated like `v` |

All tensors must share the cache device. The wrapper checks shape/stride/dtype
requirements for the state and replay cache, but does not comprehensively validate
every input dtype or value. In particular, output dtype and contiguous parameter
storage should not be inferred to have complete explicit validation. Existing
numerical tests use BF16 activations; they are not evidence for other activation
dtypes. A reshape of `a` or `b` can allocate when their original layout requires it.

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

### Layer caches and context creation

```python
GDNRecoverSSMCommitContext.create(
    layers, *, spec_query_len, max_num_reqs,
)
```

Each layer supplies `num_k_heads`, `tp_size` and
`kv_cache=(conv, checkpoint, replay)`; local `HK=num_k_heads // tp_size`.
The context normalizes convolution orientation according to the platform setting
to logical `[P, C, Q + W - 1]`, where `W > 0`.

Layers must have compatible corresponding shapes, devices, dtypes and head
geometry. State/replay inner stride signatures must match, ignoring singleton
dimensions; page strides may differ. Convolution inner strides may differ between
layers and are represented explicitly. Context creation requires nonempty layers
and positive request/window capacities. On NPU it initializes device properties
before subsequent graph capture.

The context retains cache references and preallocates:

- An INT64 `[2, 2, L]` state/replay address-and-page-stride table.
- An INT64 `[4, L]` convolution address, page, channel and token stride table.
- An INT32 `[6, max_num_reqs]` plan buffer, viewed as accepted lengths,
  state-only lengths, boundary lengths, final pages, boundary pages and boundary
  recovery lengths.

Cache storage must remain alive and at the bound addresses. Reallocation requires
context rebuilding. The mutable plan is reused: do not concurrently reuse one
context without appropriate stream ordering/synchronization.

### Commit interface

```python
context.commit(
    num_accepted_tokens, state_indices, query_start_loc,
    request_indices=None, block_table=None, num_computed_tokens=None,
    mamba_block_size=None, commit_conv=True,
)
```

| Parameter | Meaning |
| --- | --- |
| `num_accepted_tokens` | Integer accepted recurrent lengths in sampler-row order |
| `state_indices`, `query_start_loc` | Source pages and token spans for speculative rows |
| `request_indices` | Optional speculative-row to sampler-row mapping; covers at least `B` rows |
| `block_table` | Align-mode sampler-row to destination-page table, INT32/INT64, two-dimensional |
| `num_computed_tokens` | Prior computed lengths in sampler-row order |
| `mamba_block_size` | Align block size, at least `Q` |
| `commit_conv` | Whether to compact convolution history; state commit remains enabled |

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

## Kernel Design

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

## Constraints and Caller Responsibilities

- Offsets must be monotonic, within the flattened token range, and each request's
  length must not exceed `Q`. The wrapper's aggregate `T <= B * Q` check does not
  replace per-request validation.
- Positive page indices, sampler mappings and block-table values must be in
  range. Host-side checks do not inspect every device-resident index.
- No cross-request write/read conflicts may exist in checkpoint, replay or
  convolution pages. Supported in-place commit within one request does not make
  arbitrary inter-request aliasing safe.
- Verify must complete before commit consumes its records. Every accepted record
  must belong to the current checkpoint/window. Rejected and unwritten records
  must not be consumed.
- Verify with no tokens/requests returns without a kernel launch. Nonempty rows
  with null pages produce zero outputs and do not write replay records.
- Allocate the context and stable graph inputs before capture. Use stream ordering
  and stable addresses for graph replay; repeated replay mutates state and must
  not be treated as independent measurements without restoration.
- These are inference operators. No backward/autograd contract is provided.
- Device-specific branches are implementation intent, not proof of validation on
  A2, A3, 950DT or 950PR. Hardware qualification requires actual runs.

## Relationship to KDA RecoverSSM

Both implementations use the shared
[`mamba/recoverssm.py`](../mamba/recoverssm.py) planner and convolution compactor,
but retain separate verify kernels, state reconstruction kernels and contexts.
GDN uses a scalar decay per value head and grouped query/key heads; its replay
stores effective keys and decays. KDA uses its own gate semantics and raw replay
inputs. Consequently KDA activation processing, launch heuristics, tolerances and
historical performance results must not be copied as GDN validation evidence.

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
| `verify_and_multilayer_commit_preserve_accepted_prefix` | 48 | Six accepted lengths, two convolution orientations, two head geometries and two checkpoint dtypes; multiple layers, two rounds, page permutation and sampler remapping |
| `align_commit_restores_boundary_and_final_pages` | 144 | Six align modes, two orientations, eager/graph commit, three geometries and two checkpoint dtypes; mixed layer convolution strides and launch-path checks |
| `reverse_commit_grouped_and_ordinary_match_forward_reference` | 18 | A5-only ordinary/grouped comparison, three head ratios, align off/on, accepted lengths 1/5/16 |
| `verify_raw_interface_defaults_custom_options_and_out` | 2 | Raw gate API, default/custom activation and scaling, normalization choice and supplied output buffer |
| `empty_verification_does_not_launch_or_mutate` | 1 | Empty verify preserves buffers |
| `nonempty_capture_rows_with_null_pages_produce_zero_outputs` | 1 | Null pages yield zero outputs without replay/checkpoint mutation |
| **Total static expansion** | **217** | Device skips and collection failures can change actual execution counts |

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
NaN initialization occurs before verification; this is not an explicit poisoning
of every freshly written rejected-suffix record after verification.

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

The separate
[unit test file](../../../../tests/ut/ops/test_gdn_recoverssm.py) checks context
validation and mocked launch/metadata behavior. Passing such tests would not
establish NPU numerical correctness.

### Known validation discrepancy: exact block boundary

The current shared planner uses `(n + c - 1) // m` for the final destination.
The align test's expected-target construction still uses `(n + count) // m`
(test source line 326 at the baseline revision). For example, with `n=6`, `c=2`
and `m=8`, the planner writes final/boundary state to column 0, while the test
expects an additional final write to column 1.

This is a source-level oracle/implementation mismatch, not an observed NPU test
failure in this report. The `exact` mode expands to 24 cases. Confirm the intended
page ownership contract, update the appropriate oracle or implementation, and
rerun these cases before declaring align validation complete. This documentation
change deliberately does not alter either implementation or tests.

### Validation report and remaining coverage

| Item | Evidence / status |
| --- | --- |
| Interface, replay layout, recurrence and launch review | Inspected at the baseline revision |
| Numerical test definitions | Present; 217 statically expanded cases |
| Eager NPU numerical execution | Not run for this report |
| NPU graph execution | Not run; existing graph cases cover commit only |
| Exact-boundary oracle agreement | Known discrepancy; unresolved |
| A3 versus A5 qualification | No run logs supplied or produced for this report |
| Full-model token equality / draft activity | Outside this operator suite; no result asserted here |
| Latency, bandwidth, UB occupancy and peak allocated memory | No GDN measurement reported |

Additional useful coverage includes production head counts (existing geometries
are small), a broader window-length matrix, explicit post-verify rejected-record
poisoning, invalid metadata/index rejection where intended, full verify-plus-commit
graph capture, and independent high-precision verification references. These are
follow-up proposals, not claims about current test coverage.

### Reproduction

Run from the repository root in a supported Ascend environment with matching
PyTorch, `torch_npu`, CANN and Triton dependencies:

```bash
pytest -sv tests/ut/ops/test_gdn_recoverssm.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py --junitxml=gdn_recoverssm.xml
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py -k 'align and exact'
```

Record the Git revision and dirty diff, device model, driver/firmware, CANN,
PyTorch/`torch_npu`/Triton versions, commands, collected/pass/fail/skip counts and
failure details. A skipped suite or successful collection is not a numerical pass.
Resolve the documented boundary discrepancy before using a green result as an
align correctness claim.

For performance qualification, add a dedicated benchmark rather than infer
latency from pytest duration. Separate verify, plan, convolution, state commit and
end-to-end measurements; warm up compilation and execution, synchronize timing,
restore mutable caches outside timed regions, and compare equivalent workloads.
Report eager and graph results separately with hardware, shapes, strides, dtype,
accepted lengths, layer counts, iteration counts and distribution statistics.
Compare against a defined conventional state-storage/recovery baseline, and check
correctness before timing. No speedup or memory reduction percentage is claimed
by this document.
