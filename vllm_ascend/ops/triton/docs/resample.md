# Resample (`_resample_kernel` / `_npu_gumbel_block_argmax`)

## Description

- **Function**: Resamples the rejected token, or the bonus token when every
  draft token is accepted, during speculative decoding. The host entry point is
  `rejection_sample` in
  `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py`.
  `_resample_kernel` computes one local maximum and argmax for each vocabulary
  block, and the upstream `_insert_resampled_kernel` reduces those block-local
  results and writes the selected token to `sampled`.
- **Formula**: Let `target_logits` and `draft_logits` be
  \(\ell^t\) and \(\ell^d\), and let \(Z_t\) and \(Z_d\) be their
  log-sum-exp values. The residual logits are

  $$
  \operatorname{residual}_v =
  \begin{cases}
    \ell^t_v, & \text{bonus token}, \\
    (\ell^t_v-Z_t)+\log\!\left(1-
      \exp((\ell^d_v-Z_d)-(\ell^t_v-Z_t))\right),
      & \text{draft logits are available and }q_v/p_v<1, \\
    -\infty, & \text{draft logits are available and }q_v/p_v\geq1, \\
    \ell^t_v, & \text{one-hot draft and }v\neq v_{\mathrm{rejected}}, \\
    -\infty, & \text{one-hot draft and }v=v_{\mathrm{rejected}}.
  \end{cases}
  $$

  For sampling requests, `_npu_gumbel_block_argmax` adds Gumbel noise and
  finds the maximum within each vocabulary block:

  $$
  g_v=-\log(-\log(u_v+10^{-20})+10^{-20}),\qquad u_v\sim U[0,1),
  $$

  $$
  (\operatorname{value},\operatorname{idx})=
  \max_{v\in\mathrm{block}}(\operatorname{residual}_v+g_v).
  $$

  When `temperature == 0`, no noise is added and the operation reduces to
  argmax.
- **Algorithm flow**:
  1. `rejection_sample` launches `_probabilistic_rejection_kernel` to determine
     the rejected step and to compute the target and draft log-sum-exp values.
  2. `_resample_kernel` launches a two-dimensional grid of
     `(num_reqs, ceil(vocab_size / 1024))`. Each program handles one request and
     one vocabulary block.
  3. The kernel maps the request-local rejected step to a global logit row and
     then to the request-state row used by the temperature, seed, and draft
     tensors.
  4. It selects the bonus, full-draft residual, or one-hot residual branch.
     Greedy non-bonus requests return without writing because their target
     argmax was already written by `_probabilistic_rejection_kernel`.
  5. `_npu_gumbel_block_argmax` optionally adds Gumbel noise and returns the
     block-relative argmax. `_resample_kernel` adds the vocabulary-block offset
     before storing the token ID and local maximum.
  6. The upstream `_insert_resampled_kernel` reduces the per-block results and
     writes the final token to `sampled`.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. The operator is used
  only for NPU inference in model runner v2. It has no backward path. Ascend
  310P does not use this implementation because Triton and model runner v2 are
  not enabled for this path.

## Parameters

### Python entry point: `rejection_sample`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `target_logits` | Input | Processed target logits with shape `[num_logits, vocab_size]` | fp32 | ND |
| `draft_logits` | Input | Draft logits with shape `[max_num_reqs, num_speculative_steps, vocab_size]`; may be `None` for a one-hot draft | fp32 | ND |
| `draft_sampled` | Input | Draft token stream with shape `[num_logits]`; it is shifted by one position relative to the logit rows | int32 | ND |
| `cu_num_logits` | Input | Prefix sum of the logit counts, with shape `[num_reqs + 1]` | int32 | ND |
| `pos` | Input | Global position of each logit row, used as the Philox offset | int64 | ND |
| `idx_mapping` | Input | Maps `req_idx` to `req_state_idx`; used by `_probabilistic_rejection_kernel` | int32 | ND |
| `expanded_idx_mapping` | Input | Maps each global logit row to `req_state_idx` | int32 | ND |
| `expanded_local_pos` | Input | Request-local position of each global logit row; used by `_probabilistic_rejection_kernel` | int32 | ND |
| `temperature` | Input | Per-request-state temperature; zero selects greedy decoding | fp32 | ND |
| `seed` | Input | Per-request-state random seed | int64 | ND |
| `num_speculative_steps` | Attribute | Maximum number of speculative tokens per request | int32 | scalar |
| `use_fp64` | Attribute | Must be `False`; the NPU implementation raises `NotImplementedError` otherwise | bool | scalar |
| `synthetic_conditional_rates` | Attribute | Must be `None`; synthetic rejection sampling is not implemented on this path | fp32 | ND |
| `use_block_verification` | Attribute | Accepted for API compatibility but not implemented on the NPU path | bool | scalar |
| `sampled` | Output | Selected tokens with shape `[num_reqs, num_speculative_steps + 1]`; only `sampled[i, :num_sampled[i]]` is valid for request `i` | int64 | ND |
| `num_sampled` | Output | Number of output tokens per request, including the rejected or bonus token | int32 | ND |

### Triton kernel: `_resample_kernel`

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `resampled_local_argmax_ptr` | Output | Global token ID selected in each vocabulary block, shape `[num_reqs, num_blocks]` | int64 | ND |
| `resampled_local_argmax_stride` | Attribute | Row stride of `resampled_local_argmax_ptr` | int32 | scalar |
| `resampled_local_max_ptr` | Output | Maximum score in each vocabulary block, shape `[num_reqs, num_blocks]` | fp32 | ND |
| `resampled_local_max_stride` | Attribute | Row stride of `resampled_local_max_ptr` | int32 | scalar |
| `target_logits_ptr` | Input | Target logits, indexed by the global resample-token row | fp32 | ND |
| `target_logits_stride` | Attribute | Row stride of `target_logits_ptr` | int32 | scalar |
| `target_rejected_logsumexp_ptr` | Input | Target log-sum-exp for each request | fp32 | ND |
| `draft_logits_ptr` | Input | Draft logits, indexed by `[req_state_idx, resample_idx, :]` | fp32 | ND |
| `draft_logits_stride_0` | Attribute | Request-state stride of `draft_logits_ptr` | int32 | scalar |
| `draft_logits_stride_1` | Attribute | Speculative-step stride of `draft_logits_ptr` | int32 | scalar |
| `draft_rejected_logsumexp_ptr` | Input | Draft log-sum-exp for each request | fp32 | ND |
| `rejected_step_ptr` | Input | Rejected step for each request | int32 | ND |
| `cu_num_logits_ptr` | Input | Prefix sum of per-request logit counts | int32 | ND |
| `expanded_idx_mapping_ptr` | Input | Maps a global logit row to its request-state row | int32 | ND |
| `draft_sampled_ptr` | Input | Shifted draft token stream | int32 | ND |
| `temp_ptr` | Input | Per-request-state temperature | fp32 | ND |
| `seed_ptr` | Input | Per-request-state random seed | int64 | ND |
| `pos_ptr` | Input | Position used as the Philox offset for each logit row | int64 | ND |
| `vocab_size` | Attribute | Runtime vocabulary size | int32 | scalar |
| `BLOCK_SIZE` | Attribute | Compile-time vocabulary-block width; the wrapper uses 1024 | int32 | scalar |
| `HAS_DRAFT_LOGITS` | Attribute | Compile-time selector for full draft logits versus a one-hot draft | bool | scalar |

### Triton device function: `_npu_gumbel_block_argmax`

Not a separately dispatched operator: it has no `tl.program_id` and receives `logits` as a
value rather than a pointer, so it is inlined into `_resample_kernel`, its only caller in
this repository. The table below documents the internal contract.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits` | Input | Logit values already loaded for one vocabulary block | fp32 | ND |
| `block` | Input | Global vocabulary indices for the current block; also used as Philox offsets | int32 | ND |
| `mask` | Input | Marks indices smaller than `vocab_size` | bool | ND |
| `token_idx` | Input | Global logit-row index | int32 | scalar |
| `expanded_idx_mapping_ptr` | Input | Maps `token_idx` to `req_state_idx` | int32 | ND |
| `temp_ptr` | Input | Per-request-state temperature | fp32 | ND |
| `seeds_ptr` | Input | Per-request-state seed | int64 | ND |
| `pos_ptr` | Input | Per-logit-row position | int64 | ND |
| `processed_logits_ptr` | Output | Optional processed-logit buffer written before noise is added; `None` disables the store | fp32 | ND |
| `processed_logits_stride` | Attribute | Request-state row stride of `processed_logits_ptr` | int32 | scalar |
| `processed_logits_col_ptr` | Input | Optional scalar column index; `None` selects column zero | int32 | scalar |
| `vocab_size` | Attribute | Runtime vocabulary size used by the optional output layout | int32 | scalar |
| `APPLY_TEMPERATURE` | Attribute | Compile-time flag that divides logits by temperature before sampling | bool | scalar |
| return `value`, `idx` | Output | Maximum score and block-relative argmax | fp32, int32 | scalar |

## Constraints

- `target_logits` and `draft_logits`, when provided, must use fp32 because the
  residual computation and Gumbel sampling are performed in fp32.
- `draft_sampled` is shifted by one position relative to `target_logits`.
  The rejected token is loaded from `draft_sampled[resample_token_idx + 1]`.
  The bonus branch must remain short-circuited so the last request never reads
  beyond the end of this tensor.
- `req_idx`, `req_state_idx`, and `resample_token_idx` address different
  tensors and must not be used interchangeably.
- Greedy non-bonus programs leave `resampled_local_argmax` and
  `resampled_local_max` unwritten. `_insert_resampled_kernel` depends on the
  same early-return condition and must not read those entries.
- `_npu_gumbel_block_argmax` returns an index relative to the current block.
  Its caller must add `block_idx * BLOCK_SIZE` before storing a token ID.
- Invalid vocabulary lanes and excluded tokens must use negative infinity.
  In the padded tail of the full-draft branch, `-inf - -inf` produces NaN;
  `ratio < 1` evaluates to false and therefore maps those lanes back to
  negative infinity.
- The NPU implementation does not implement the upstream
  `req_state_idx >= 0` padding guard. Current callers must provide only
  non-negative request-state indices.
- `pos` is cast to int32 because the Ascend vector-core Philox path does not
  support uint64 multiplication. Positions must fit in int32.
- `use_fp64=True` and non-`None` `synthetic_conditional_rates` are unsupported
  and raise `NotImplementedError`. `use_block_verification=True` is currently
  accepted but has no implementation on this path.
- The operator supports dynamic request counts, token counts, and vocabulary
  sizes. `BLOCK_SIZE` and `HAS_DRAFT_LOGITS` are compile-time constants.
- The operator is inference-only. There is no backward implementation.

## Origin and Differences

- **Origin**: `_resample_kernel` follows
  `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`.
  `_npu_gumbel_block_argmax` is adapted from
  `vllm/v1/worker/gpu/sample/gumbel.py::gumbel_block_argmax`. The Ascend
  implementation is installed by
  `vllm_ascend/patch/worker/patch_v2/patch_triton.py`.
- **Differences**:
    - NPU does not support the fp64 random path used upstream. Local maxima and
      Gumbel noise use fp32, and `use_fp64=True` is rejected explicitly.
    - `pos` is cast to int32 so Philox uses the int32/uint32 multiplication path
      supported by Ascend vector cores.
    - The implementation uses `tl.rand` and
      `-log(-log(u + 1e-20) + 1e-20)` instead of upstream `tl_rand64` or the
      numerically stronger fp32 `-log(-log1p(-u))` formulation. This gives the
      large-noise tail lower resolution than upstream.
    - The upstream negative-request-state mask was not carried over. Current
      Ascend callers do not generate negative request-state indices.
    - The residual maximum is stored as fp32 rather than fp64.

## Test Cases

The accuracy test uses an independent PyTorch fp32 reference for the
block-local maxima and argmax indices of the greedy paths, which carry no Gumbel
noise. `_npu_gumbel_block_argmax`
is a device function inlined into `_resample_kernel` and is not tested on its
own; it is covered through `_resample_kernel`, its only caller. Its
`processed_logits` store and its `APPLY_TEMPERATURE=True` path are unreachable
from this repository, because the call site passes `None, 0, None` and `False`,
and are therefore not covered.

The test file contains no Triton kernel of its own and does not reproduce any
part of the operator. Sampling requests are checked in two ways that are both
independent of the Gumbel draw:

- Cases that need an exact expected token are built so that each vocabulary
  block holds exactly one finite residual, which the noise cannot displace.
- Cases that need the residual values themselves compare sampling frequencies
  over 16,384 draws against the analytic distribution: `softmax(logits)` for the
  bonus branch, and the normalised `(p - q)+` residual for the draft-logits
  branch.

The cases cover greedy and sampling requests, all three residual branches, the
greedy early return, ragged vocabulary tails, shuffled request-state rows,
deterministic seeds and positions, both sampling distributions, and an
end-to-end greedy call through `rejection_sample`.

The greedy paths compare block maxima with `rtol=1e-5` and `atol=1e-5`, and
token IDs exactly. The sampling paths compare token IDs exactly against the
single finite candidate of each block, and the statistical frequency comparison
uses `atol=0.02`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_resample.py
```
