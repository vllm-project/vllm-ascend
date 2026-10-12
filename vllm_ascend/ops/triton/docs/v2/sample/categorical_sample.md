# categorical_sample

## Description

- **Function**: Samples one token per input row from the categorical distribution represented by `logits`. It is used as the Ascend NPU replacement for vLLM `gumbel_sample`: `temperature == 0` performs greedy argmax; non-zero temperature performs random categorical sampling. The implementation keeps the existing sampling interface, including optional raw-logit caching for speculative decoding.
- **Formula**:
    - Let the effective logit for token `i` be `x_i = logits_i / temperature` when `apply_temperature=True` and `temperature != 0`, otherwise `x_i = logits_i`.
    - Greedy mode (`temperature == 0`): `sample = argmax_i x_i`.
    - Random mode (`temperature != 0`): `P(sample=i) = exp(x_i) / sum_j exp(x_j)`.
    - For numerical stability, each coarse block `b` uses `m_b = max_{i in b} x_i` and `S_b = sum_{i in b} exp(x_i - m_b)`. With `M = max_b m_b`, the global block mass is `W_b = S_b * exp(m_b - M)`.
    - A stateless uniform draw `u(seed, pos)` defines `threshold = u * sum_b W_b`. The same threshold is propagated through coarse-block, fine-block, and token-level cumulative masses; no additional random draw is used.
- **Algorithm flow** (processed row by row, independently):
  1. Split each vocabulary row into 8192-element coarse blocks. Flatten `(token, coarse_block)` tasks and distribute them across at most the available Vector Cores.
  2. `_categorical_prepare_mass_kernel` loads each coarse block, optionally stores raw pre-temperature logits into `logits_cache`, applies temperature when requested, computes the block max/argmax, and computes probability masses. Each 8192-element block is further reduced into eight 1024-element fine-block masses.
  3. `_categorical_sample_kernel` processes token rows across at most the available Vector Cores. For random rows, it uses the prepared block masses to select one 8192-element coarse block, then uses the eight prepared fine-block masses to select one 1024-element fine block.
  4. Reload only the selected 1024 logits, apply the same temperature rule, compute the token masses and cumulative sum, and return the token whose cumulative mass crosses the propagated threshold.
  5. For greedy rows, use the prepared block maxima/argmax metadata to return the global argmax.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits` | Input | Logits with shape `[num_tokens, vocab_size]`; vocabulary dimension must be contiguous | fp32 / bf16 | ND |
| `expanded_idx_mapping` | Input | Maps each token row to a request-state index, shape `[num_tokens]` | int32 | ND |
| `temperature` | Input | Per-request temperature, shape `[max_num_reqs]`; `0` selects greedy mode | fp32 | ND |
| `seed` | Input | Per-request stateless RNG seed, shape `[max_num_reqs]` | int64 | ND |
| `pos` | Input | Per-token RNG position, shape `[num_tokens]`; converted to int32 inside the Triton sampling kernel | int32 / int64 | ND |
| `apply_temperature` | Attribute | If `True`, non-zero-temperature rows use `logits / temperature`; if `False`, logits are sampled as provided | bool | scalar |
| `is_drafting` | Attribute | If `True`, adds a fixed RNG-position salt to keep the optional drafting stream separate | bool | scalar |
| `logits_cache` | Input/Output | Optional cache buffer `[max_num_reqs, num_cols, cache_vocab_size]`; stores raw logits before temperature scaling | fp32 / bf16 | ND |
| `logits_cache_col` | Input | Optional cache-column selector; either a scalar or shape `[num_tokens]` | int32 | scalar / ND |
| `use_fp64` | Attribute | Compatibility argument; must be `False` on this NPU implementation | bool | scalar |
| `sampled_token_ids` | Output | Sampled token ID for each input row, shape `[num_tokens]` | int64 | ND |

## Constraints

- `logits` must be rank 2 with shape `[num_tokens, vocab_size]`, `vocab_size > 0`, and `logits.stride(-1) == 1`. The current single-operator tests cover fp32 and bf16.
- `expanded_idx_mapping.shape == [num_tokens]`; valid rows index `temperature` and `seed` by request state. A negative mapping is treated as an invalid/padded request by the kernel and must not be used to write `logits_cache`.
- `temperature.shape[0]` and `seed.shape[0]` must cover every non-negative request index referenced by `expanded_idx_mapping`. Temperature values are expected to be non-negative.
- `temperature == 0` is greedy mode. For non-zero temperature, random sampling is categorical. `apply_temperature=False` means the caller has already applied any required temperature scaling.
- `pos.shape == [num_tokens]`. The NPU RNG path converts positions to int32; inference positions must therefore remain within the supported int32 range.
- The vocabulary size does not need to be divisible by 8192 or 1024. Tail elements are masked. The long-vocabulary inference shape `vocab_size=151936` therefore exercises both coarse and fine tail handling.
- `logits_cache`, when provided, must satisfy `logits_cache.size(-1) >= vocab_size`. Cache values are written before temperature scaling.
- `logits_cache_col`, when provided, must be either a 0-D scalar tensor or a tensor with one column index per token. If multiple tokens map to the same `(request, cache_col)` location, their cache writes alias; callers requiring deterministic cache contents must avoid that mapping.
- `use_fp64=True` is not supported and raises `NotImplementedError`.
- Finite logits and `-inf` masking are supported. NaN and `+inf` inputs do not have an additional operator-specific normalization contract.
- The Python wrapper obtains `num_tokens` and `vocab_size` from tensor shape metadata; it does not perform a device-to-host `.item()` synchronization. Runtime task counts are passed to Triton as scalar kernel arguments.
- The implementation introduces no explicit device-to-host synchronization. The single-operator accuracy tests run eagerly; graph-capture behavior is validated by higher-level vLLM/vLLM-Ascend integration tests rather than by this single-operator test.

## Origin and Differences

- **Origin**: Replaces the `gumbel_sample` path from `vllm/v1/worker/gpu/sample/gumbel.py` for Ascend NPU. Gumbel-Max and direct categorical sampling represent the same categorical distribution for non-zero temperature.
- **Differences**:
    - NPU adaptation for performance: replaces per-vocabulary Gumbel RNG and two logarithms with one stateless uniform draw per sampled row plus hierarchical probability-mass selection. The prepare stage uses 8192-element coarse blocks and stores eight 1024-element fine-block masses so the final sampling stage reloads only one 1024-element block.
    - NPU adaptation for performance: flattens `(token, coarse_block)` work and limits launches to the available Vector Core count; the sampling stage similarly distributes token rows over the available Vector Cores.
    - Modified for vllm-ascend logic: preserves the existing sampling call contract, including per-request temperature/seed, request-index mapping, optional pre-temperature `logits_cache`, scalar or per-token cache columns, and greedy rows mixed with random rows.
    - Random sampling is distribution-equivalent to Gumbel-Max but does not consume random numbers in the same way. Therefore the same `seed`/`pos` is deterministic within `categorical_sample`, but is not required to return the exact same token sequence as the previous Gumbel implementation.
    - FP64/fixed-point sampling is not added by this operator; `use_fp64=True` remains unsupported on NPU.

## Test Cases

The single-operator tests use the long-vocabulary inference shape `vocab_size=151936`, which is not divisible by either the 8192 coarse block or the 1024 fine block and therefore covers hierarchy-tail masking. Batch/token counts include `1`, `16`, and `64`, matching the operator's long-vocabulary performance validation range. Greedy and cache results are checked bit-exact. Random sampling cannot be validated by exact token equality against Gumbel-Max because the RNG consumption pattern is intentionally different; the tests instead cover deterministic replay, mixed greedy/random rows, exact single-support sampling across hierarchy boundaries, temperature-scaling equivalence, cache semantics, and an equal-mass finite-support distribution sanity check.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_categorical_sample.py
```
