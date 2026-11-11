# Resample (`resample` / `_resample_kernel` / `_categorical_finalize_kernel`)

## Description

- **Function**: resamples the first rejected token, or the bonus token when all draft tokens are accepted, during Model Runner V2 speculative decoding on Ascend NPU.
- **Python entry point**: `resample` in `vllm_ascend/ops/triton/spec_decode/resample.py`.
- **Implementation**: two Triton kernels. `_resample_kernel` scans the vocabulary and produces one block statistic per request/vocabulary block. `_categorical_finalize_kernel` uses one request-level random threshold to select the final categorical token.
- **Integration**: the surrounding rejection kernel still owns draft-token verification and the greedy rejected-token argmax. `resample` updates `sampled` and `num_sampled` in place.

The previous NPU implementation converted the rejected distribution to residual logits, generated one Gumbel value for every vocabulary token, reduced block-local maxima, and then used `_insert_resampled_kernel` to select the final token. This implementation samples the same categorical distributions directly from probability mass and does not materialize full-vocabulary Gumbel noise.

### Supported resample branches

For request `r`, let

```text
start = cu_num_logits[r]
step  = num_sampled[r]              # rejected step on entry
row   = start + step
end   = cu_num_logits[r + 1]
```

`row == end - 1` is the bonus row.

The operator supports:

1. **Random full-draft residual**
   \[
   m_i = \max(p_i-q_i, 0),
   \quad
   p_i = \exp(\ell^t_i-\operatorname{LSE}_t),
   \quad
   q_i = \exp(\ell^d_i-\operatorname{LSE}_d).
   \]

2. **Random one-hot residual**
   \[
   m_i =
   \begin{cases}
   p_i, & i \ne i_\text{draft},\\
   0, & i = i_\text{draft}.
   \end{cases}
   \]

3. **Random bonus**
   \[
   P(i) = \operatorname{softmax}(\ell^t)_i.
   \]

4. **Greedy bonus**: selects the global target argmax.

5. **Greedy non-bonus rejection**: does not overwrite the target argmax already written by the verification kernel; it only advances `num_sampled`.

Individual `-inf` vocabulary entries are valid and contribute zero probability mass.

### Two-stage categorical flow

With `_RESAMPLE_BLOCK_SIZE = 1024`:

```text
num_blocks = ceil(vocab_size / 1024)
total_tasks = num_reqs * num_blocks
num_workers = min(vector_core_num, total_tasks)
```

`_resample_kernel` flattens `(req_idx, block_idx)` into one task space. Tasks are split into contiguous ranges so workers differ by at most one task:

```text
base  = total_tasks // num_workers
extra = total_tasks % num_workers
```

The first `extra` workers execute `base + 1` tasks and the rest execute `base` tasks. Every workspace cell has one writer, so atomics are unnecessary.

For residual rows the first stage stores the sum of token masses in each block. For bonus rows it stores:

```text
block_max
block_argmax
block_sumexp = sum(exp(logit - block_max))
```

The finalize kernel converts bonus block sums to a common global-max scale.

For random requests it draws one value

```text
u ~ Uniform[0, 1)
threshold = u * total_mass
```

and uses the same global threshold for both hierarchy levels:

1. select the first block whose cumulative mass crosses `threshold`;
2. subtract the mass before that block;
3. rebuild only the selected 1024-token block;
4. select the first token whose cumulative mass crosses the remaining threshold.

No second random draw is needed.

## Parameters

### Python entry point: `resample`

```python
def resample(
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    target_logits: torch.Tensor,
    target_rejected_logsumexp: torch.Tensor,
    draft_logits: torch.Tensor | None,
    draft_rejected_logsumexp: torch.Tensor,
    cu_num_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    draft_sampled: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    has_draft_logits: bool | None = None,
) -> None:
```

| Parameter | Input/Output/Attribute | Shape | Contract |
| --- | --- | --- | --- |
| `sampled` | Input/Output | `[num_reqs, num_speculative_steps + 1]` | int64 output buffer. Random rejected/bonus tokens and greedy bonus tokens are written in place. |
| `num_sampled` | Input/Output | `[num_reqs]` | int32. On entry it is the rejected/bonus step. On return it is `step + 1`. |
| `target_logits` | Input | `[num_logits, vocab_size]` | fp16, bf16, or fp32. Vocabulary dimension must be contiguous. |
| `target_rejected_logsumexp` | Input | `[num_reqs]` | fp32 LSE for each rejected row. |
| `draft_logits` | Input | `[max_num_reqs, num_speculative_steps, vocab_size]` or `None` | fp16, bf16, or fp32. `None` selects one-hot draft semantics. Vocabulary dimension must be contiguous when present. |
| `draft_rejected_logsumexp` | Input | `[num_reqs]` | fp32 LSE for each rejected draft row. |
| `cu_num_logits` | Input | `[num_reqs + 1]` | int32 prefix sum of logit rows. Request lengths may be ragged. |
| `expanded_idx_mapping` | Input | `[num_logits]` | int32 mapping from global logit row to request-state row. |
| `draft_sampled` | Input | `[num_logits]` | int32 shifted draft-token stream. One-hot rejection reads `draft_sampled[row + 1]`. |
| `temperature` | Input | `[max_num_reqs]` | fp32. Zero is the greedy sentinel; non-zero selects random sampling. |
| `seed` | Input | `[max_num_reqs]` | int64 request-state seed. |
| `pos` | Input | `[num_logits]` | int64 logical position; currently cast to int32 by the NPU random path. |
| `has_draft_logits` | Attribute | scalar | If `None`, inferred from `draft_logits is not None`. Pass `False` when an upper layer replaced `None` with a dummy tensor. |

`temperature` is used here to distinguish greedy and random rows. The surrounding rejection path owns the logits/LSE preprocessing contract.

### Internal workspace

| Workspace | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `local_argmax` | `[num_reqs, num_blocks]` | int64 | Global token ID of each bonus block argmax; consumed by greedy bonus finalize. |
| `local_max` | `[num_reqs, num_blocks]` | fp32 | Bonus block maximum. |
| `local_mass` | `[num_reqs, num_blocks]` | fp32 | Residual block mass, or bonus block local sum-exp. |

Workspace dtype is independent of the input logits dtype. Target/draft logits are converted to fp32 before exponential, subtraction, and reduction operations.

## Constraints

- The operator is inference-only and has no backward path.
- `target_logits` and full `draft_logits` support fp16, bf16, and fp32.
- Probability-mass arithmetic and block statistics use fp32.
- The vocabulary dimension must be contiguous:
  ```text
  target_logits.stride(-1) == 1
  draft_logits.stride(-1) == 1  # when full draft logits are used
  ```
  Earlier dimensions may use non-default strides because their strides are passed explicitly.
- `draft_logits=None` selects one-hot draft semantics. If an upper layer already replaced `None` with a dummy tensor, it must pass the original semantic flag through `has_draft_logits=False`.
- `has_draft_logits=True` with `draft_logits=None` is invalid.
- `num_reqs == 0` is supported as a no-op.
- `vocab_size == 0` is rejected.
- Individual `-inf` logits are supported and represent zero-mass tokens. Padded vocabulary lanes are masked to zero mass and cannot be sampled.
- NaN, `+inf`, and rows containing only `-inf` do not receive a stronger exceptional-value contract than the previous Triton Gumbel resample path. Normal masked `-inf` entries are part of the supported path.
- Full-draft residual mass is fp32 `max(p - q, 0)`. Extremely small positive residuals can be lost when `p` and `q` are nearly equal; there is no fp64 residual mode in this Triton operator.
- `pos` is cast to int32 for the current Ascend Philox path, so valid logical positions must fit in int32.
- Callers own index validity. In particular:
  ```text
  0 <= num_sampled[r] < cu_num_logits[r + 1] - cu_num_logits[r]
  0 <= expanded_idx_mapping[row] < max_num_reqs
  ```
- For one-hot non-bonus rejection, `draft_sampled[row + 1]` must be valid.
- Inactive branches may leave some workspace cells unwritten. `_categorical_finalize_kernel` masks those loads. Greedy non-bonus rows must never consume those workspace cells.

## Origin and Differences

- **Origin**: replaces the previous NPU MRV2 `_resample_kernel` / `_npu_gumbel_block_argmax` / upstream `_insert_resampled_kernel` path in `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py`.
- **Previous random algorithm**:
  ```text
  residual logits
      -> one random Gumbel value per vocabulary token
      -> block-local argmax
      -> global max over block winners
  ```
- **Current random algorithm**:
  ```text
  probability mass
      -> block mass
      -> one uniform value per request
      -> block threshold selection
      -> selected-block token threshold selection
  ```
- Both algorithms target the same categorical distribution for valid finite inputs, but they do not preserve an identical token stream for the same `seed` and `pos`.
- The categorical implementation avoids the full-vocabulary `-log(-log(u))` chain.
- `_resample_kernel` uses a one-dimensional Vector-Core-aligned grid instead of one program per `(request, vocab block)`.
- The upper layer no longer allocates `resampled_local_argmax` / `resampled_local_max` or invokes `_insert_resampled_kernel`; the standalone `resample` API owns all temporary workspace.
- Greedy rejected-token ownership remains in the verification kernel.

### Integration

The upper layer must preserve whether real draft logits existed before creating a dummy pointer:

```python
has_draft_logits = draft_logits is not None
if draft_logits is None:
    draft_logits = target_logits.new_empty(1, 1, 1)
```

After rejection verification produces `sampled`, `num_sampled`, `target_rejected_logsumexp`, and `draft_rejected_logsumexp`, integration is:

```python
resample(
    sampled,
    num_sampled,
    target_logits,
    target_rejected_logsumexp,
    draft_logits,
    draft_rejected_logsumexp,
    cu_num_logits,
    expanded_idx_mapping,
    draft_sampled,
    temperature,
    seed,
    pos,
    has_draft_logits=has_draft_logits,
)
```

## Test Cases

The focused test replaces the previous Gumbel-specific `test_resample.py` and tests the public categorical `resample()` API rather than launching a private Triton stage directly.

It defines no Triton kernel and does not reproduce the operator. Deterministic cases construct exact support; random cases compare empirical frequencies against independent analytic probability vectors.

Covered behavior:

| Area | Coverage |
| --- | --- |
| Finite logits | deterministic and statistical random cases |
| Partial `-inf` | zero-mass entries are never sampled; finite support keeps the expected distribution |
| Vocabulary tail | `vocab_size = 1023, 1024, 1025` |
| Full-draft residual | exact positive residual and normalized `(p-q)+` statistical distribution |
| Bonus | random softmax distribution and greedy global argmax |
| One-hot draft | rejected token exclusion, renormalized distribution, and `None`/dummy compatibility |
| Greedy rejection | verification result is preserved and `num_sampled` advances |
| Logits dtype | fp16, bf16, fp32 for bonus and full-draft residual |
| Ragged requests | different request lengths and rejected steps |
| Request-state mapping | shuffled/non-contiguous `expanded_idx_mapping` |
| Business shape | `num_reqs=8`, `vocab_size=151936` |
| Mixed branches | greedy rejection, greedy bonus, random bonus, and random residual in one launch |
| Numerical residual | moderate fp32-resolvable `p≈q` case |
| RNG | identical seed/position inputs are deterministic within the categorical implementation |
| Layout | non-contiguous target/draft vocabulary dimensions are rejected |
| Wrapper contract | empty batch, zero vocabulary, and contradictory `has_draft_logits=True` + `draft_logits=None` |
| End-to-end | greedy `rejection_sample` covers immediate rejection, later rejection, and fully accepted bonus requests |

Run on an Ascend environment with Triton enabled:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_resample.py
```

Performance is validated separately with `msprof op`; the numerical UT does not use Python wall time as an operator-performance metric.
