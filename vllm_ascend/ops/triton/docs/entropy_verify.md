# MRV2 entropy verification

## Description

- **Function**: Verify a prefix of speculative tokens using a per-position entropy-adjusted acceptance threshold. This is an opt-in accuracy/performance trade-off. It does not preserve the target sampling distribution when enabled. The entropy kernels are in `ops/triton/v2/spec_decode/entropy_verify.py`; the callable entry point is `rejection_sample` in `worker/v2/spec_decode/rejection_sampler_utils.py`.
- **Formula**: For already processed target logits `z`, `p = softmax(z)` and `H = sum(-p * log(p + 1e-10))`. At each draft position, `threshold = min(exp(-alpha * H), posterior_threshold)`; accept when `q > 0` and `p(token) / q(token) >= threshold * u`, where `u` is the seeded uniform verification draw. Full draft probabilities use `softmax(draft_logits / temperature)`; one-hot drafts use `q(token) = 1`. When `alpha = 0`, use the constant `posterior_threshold`. Verification stops at the first rejection. Recovery uses the existing, unscaled residual distribution; no entropy scaling is applied to recovery logits.
- **Algorithm flow** (processed row by row, independently): Compute block max/sum-exp statistics; compute FP32 partial entropy in vocabulary blocks of 8192; sum partials and compute the threshold inside the request's rejection kernel; write the accepted prefix and sample one recovery/bonus token. Entropy rows are independent; prefix verification is sequential within each request. Greedy requests compare argmax tokens, and synthetic acceptance uses its configured rates. Disabled entropy and `alpha = 0` avoid the entropy scan; the threshold reduction is fused into verification.
- **Supported modes**: Ascend A2/A3/A5 device families (Atlas A2/A3 and 950 products) are included in the existing test guards. Eager and `FULL_DECODE_ONLY` graph execution are supported. The evidence linked below was collected only on Ascend910_9392 with DFlash and DSpark; it does not certify every device family or model. FP64 mode is unsupported.

## Parameters

The table describes the host entry point. Tensor inputs are required unless explicitly optional. `R` is the active request count, `M` the request-state capacity, `K` the speculative step count, `L` the packed logit row count, and `V` the effective vocabulary size.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `target_logits` | Input | Target logits after temperature, penalties and top-k/top-p processing | FP16/BF16/FP32 | `[L, V_target]`, NPU ND, vocabulary contiguous |
| `draft_logits` | Input, optional | Unscaled draft logits; `None` selects one-hot draft | FP16/BF16/FP32 or None | `[M, K, V_draft]`, NPU ND, vocabulary contiguous |
| `draft_sampled` | Input | Packed draft token IDs; first entry per request is a placeholder, later entries are candidates | INT64 | `[L]`, contiguous |
| `cu_num_logits` | Input | Exclusive packed-row offsets, starts at 0 and ends at L | INT32 | `[R + 1]`, contiguous |
| `pos` | Input | Positions used with request seed to generate verification and recovery draws | INT32 | `[L]`, contiguous |
| `idx_mapping` | Input | Active request index to state slot | INT32 | `[R]`, contiguous |
| `expanded_idx_mapping` | Input | Packed row to state slot; consistent with offsets and `idx_mapping` | INT32 | `[L]`, contiguous |
| `expanded_local_pos` | Input | Row-local draft position; consistent with the packed request | INT32 | `[L]`, contiguous |
| `temperature` | Input | Per-state temperature; 0 selects greedy, positive selects probabilistic | FP32 | `[M]`, contiguous |
| `seed` | Input | Per-state deterministic RNG seed | INT64 | `[M]`, contiguous |
| `num_speculative_steps` | Attribute | K, maximum draft length in this invocation | Python int | Scalar, positive |
| `synthetic_conditional_rates` | Input, optional | Synthetic per-step rates, overrides entropy when present; default None | FP32 or None | `[K]`, contiguous |
| `use_fp64` | Attribute | Default False; True raises `NotImplementedError` before launch | bool | Scalar |
| `use_block_verification` | Attribute | Existing upstream block mode; default False, mutually exclusive with entropy | bool | Scalar |
| `use_entropy_verify` | Attribute | Default None resolves worker configuration on every call; explicit bool overrides it | bool or None | Scalar |
| `posterior_threshold` | Attribute | Default 0.95; used when an explicit entropy flag is supplied, otherwise loaded from configuration | Python float | Scalar in `(0, 1]` |
| `posterior_alpha` | Attribute | Default 0.4; 0 selects constant threshold; otherwise loaded from configuration as above | Python float | Scalar, nonnegative; use finite values |
| `sampled` | Output | Accepted prefix followed by one recovery/bonus token | INT64 | New `[R, K + 1]` tensor; only `:num_sampled[r]` is valid |
| `num_sampled` | Output | Number of valid output tokens, including recovery/bonus | INT32 | New `[R]` tensor |

The entropy scan is internal. Its complete low-level argument contract is:

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits_ptr` | Input | Processed target logits | FP16/BF16/FP32 pointer | `[L, V_target]` |
| `logits_stride` | Attribute | Row stride in elements | Integer | Scalar |
| `max_ptr` | Input | Per-block target maxima from the logits-statistics kernel | FP32 pointer | `[L, N]` |
| `sumexp_ptr` | Input | Per-block shifted exponential sums from the same kernel | FP32 pointer | `[L, N]` |
| `stats_stride` | Attribute | Row stride shared by statistics and entropy partials | Integer | Scalar, N in the host allocation |
| `entropy_ptr` | Output | Per-block entropy; zeros for greedy or bonus rows | FP32 pointer | New `[L, N]` |
| `mapping_ptr` | Input | Expanded request-state mapping | INT32 pointer | `[L]` |
| `local_pos_ptr` | Input | Expanded local position | INT32 pointer | `[L]` |
| `temp_ptr` | Input | Per-state temperature | FP32 pointer | `[M]` |
| `vocab_size` | Attribute | `min(V_target, V_draft)` for full draft, else V_target | Integer | Scalar, positive |
| `vocab_num_blocks` | Attribute | N = ceil(V / 8192) | Integer | Scalar |
| `num_speculative_steps` | Attribute | K; local positions at or above K skip entropy work | Integer | Scalar |
| `BLOCK_SIZE` | Attribute | Compile-time vocabulary block width | constexpr integer | 8192 |
| `PADDED_VOCAB_NUM_BLOCKS` | Attribute | Compile-time reduction width | constexpr integer | next power of two of N |
| `EPSILON` | Attribute | Stabilizer inside entropy logarithm | constexpr float | 1e-10 |

The `_entropy_verify` helper is inlined into `_probabilistic_rejection_kernel`; it has no separate launch or externally visible output buffer. Its inputs are `entropy_partials_ptr` (FP32 `[L,N]`, or None when alpha is zero), `target_local_sumexp_stride` (integer row stride), `logit_idx` (integer row), `vocab_num_blocks` (integer N), `target_log_prob`, `draft_log_prob`, `u` (FP32 scalars). Its attributes `PADDED_VOCAB_NUM_BLOCKS`, `POSTERIOR_THRESHOLD`, `POSTERIOR_ALPHA` are constexpr; its result is a scalar boolean. The containing rejection kernel's other pointers are the host inputs above or fresh internal statistics/output buffers; entropy adds only the partials pointer and these threshold constants to the existing verification path.

## Constraints

- All tensor arguments reside on the same NPU. Vocabulary axes must have unit stride; padded/noncontiguous row strides are supported. Integer mapping and offset arrays must be contiguous and mutually consistent. This internal API trusts scheduler-provided indices; it is not a validator for arbitrary malformed mappings, device mismatches, or transposed vocabulary axes.
- Each request has 1 through `K + 1` packed rows, including the final bonus row. Bonus-only requests are valid. Offsets are monotonic, in `[0,L]`; state mappings are in `[0,M)`, local positions are nonnegative and bounded by K. Position and seed arrays must match the same logical requests after chunking/reordering.
- Candidate token IDs are in `[0,V)` or the supported `-1` placeholder, which forces rejection and target recovery. No out-of-range positive ID is supported. A smaller draft vocabulary restricts verification to the shared vocabulary; padding outside it must not be treated as a valid candidate.
- Every sampled distribution must contain at least one finite supported logit. Partial `-inf` masking is valid, including single support. All-`-inf`, NaN and `+inf` distributions are outside the supported contract. Entropy compares against a golden calculated from the actual quantized input, not an earlier FP32 source.
- Temperature must be finite and nonnegative, rates must be finite and in `[0,1]`, and threshold/alpha must satisfy the ranges above. The host rejects invalid threshold/negative or NaN alpha when entropy is active. Finiteness of positive alpha is a caller precondition, not an additional check implemented by this change.
- Full and one-hot draft, mixed greedy/probabilistic requests, ragged packing, reordered states and graph replay are supported. Synthetic plus upstream block mode is invalid; entropy plus upstream block mode raises `ValueError`. Synthetic mode suppresses entropy. FP64 is explicitly unsupported.
- The caller must split work to respect the current Triton Ascend flattened grid bound. Both `L * ceil(V / 8192)` and the recovery kernel's `R * ceil(V / 1024)` must fit the backend limit (65535 in the validated stack). An empty batch is handled by the lower-level resample API; the full host wrapper is not certified for zero-request launch grids.
- Only the returned valid prefix may be read. Unwritten suffix elements are intentionally unspecified. Inputs retain their storage, dtype, shape, strides and values; outputs and temporary buffers are newly allocated. FP32 promotion is local arithmetic, not an in-place conversion of caller tensors.
- Fixed deployment settings reuse compiled kernels across observed dynamic batch sizes. New dtype, mode or floating threshold/alpha settings can create a new specialization. Warm actual entropy requests before measuring or serving traffic; server readiness alone does not guarantee that these kernels have been compiled.

## Origin and Differences

- **Origin**: The enclosing sampling path adapts upstream vLLM's `v1/worker/gpu/spec_decode/rejection_sampler_utils.py`. The entropy threshold is the Ascend MRV1 feature ported to the MRV2 packed-row flow. It is separate from upstream's distribution-preserving block verification.
- **Differences**: Target entropy uses already processed logits, computed in FP32 vocabulary blocks on NPU. Threshold reduction is fused into rejection; alpha zero skips the vocabulary scan. The existing MRV2 recovery implementation is reused without entropy scaling. The stats call supports the upstream argument change between vLLM 0.31.0 and the validated newer main revision.
- **Development reason**: Preserve the explicitly enabled MRV1 accuracy/performance option in MRV2, while avoiding an extra threshold kernel and unnecessary alpha-zero work. This is a permanent optional feature, not a temporary model-weight workaround. Removal deadline: N/A. If upstream provides equivalent configurable verification semantics, replace the local implementation through the existing extension point and retain the oracle/compatibility tests. The inherited upstream `has_auto_blockify_blacklist_op` workaround is separate; the validated compiler ignores it and derives safety from TTIR.
- **Integration**: Uses the existing MRV2 verification integration and worker configuration; this documentation introduces no new monkey patch, environment variable, or model-runner behavior. User-facing semantics and opt-in configuration are in the [feature guide](../../../../docs/source/user_guide/feature_guide/speculative_decoding.md#block-verify-and-entropy-verify).

## Test Cases

Tests are split by responsibility to avoid duplicating the same combinations. The numerical golden, external-input protection and entropy-specific vocabulary-padding cases below were run as supplemental validation; their scripts and quality report are retained separately and are not included in this documentation commit:

| Input equivalence class / requirement | Test and assertion |
| --- | --- |
| Config lookup per call, explicit True/False, None | `test_entropy_verify.py`: worker config and override tests |
| Disabled, alpha zero, synthetic bypass | CPU allocation/launch tests; `test_synthetic_entropy_is_a_noop` on NPU |
| Invalid threshold/alpha, conflicting block flag | CPU fail-before-launch tests, including FP64 rejection and block/synthetic conflict in `test_entropy_verify.py` |
| FP16/BF16/FP32, one-hot/full draft, vocabulary tail | `test_entropy_verify_matches_fp32_oracle`: exact prefix counts and accepted IDs |
| Real Qwen3 vocabulary 151936, K=7, batch 1/8, mixed temperature | `test_entropy_real_shape_unified_tolerance`: FP64 CPU golden for every FP32 entropy partial and row sum |
| Uniform, peaked, partially masked/single-support distributions | Same numerical test, BF16 input; output precision is FP32 in all cases |
| Threshold variation, alpha zero/positive, per-position entropy | `test_entropy_parameters_and_per_position_thresholds`; numerical golden above |
| Multiple entropy blocks and final partial block | `test_entropy_verify_reduces_all_vocab_blocks`, real-shape golden |
| Placeholder, bonus-only, greedy/probabilistic | `test_entropy_verify_placeholder_and_bonus_only` and recovery tests |
| Offsets, position, mapping, seed; chunking/reordered requests | `test_entropy_verify_chunking_and_request_mapping`; resample deterministic and ragged tests |
| Narrow draft vocabulary | Existing standard/block vocabulary test, entropy-specific contract test |
| Recovery probabilities, dtype, residual independence | `test_resample.py`: distribution and `test_entropy_verify_uses_unscaled_recovery` tests |
| External tensor data, dtype, shape, stride, offset, address; output aliasing | `test_verification_preserves_external_inputs`: seven mode/layout classes |
| Empty batch and invalid low-level recovery inputs | `test_resample.py`; not a promise of full-wrapper support for malformed input |

The FP32 intermediate must pass both `abs(actual-golden) <= 2^-16 + 2^-10 * abs(golden)` for at least 99% of elements and maximum absolute error `<= max(0.01, 32 * ULP_FP32(1)) = 0.01`. These rules also apply to the summed entropy. Integer token IDs/counts use exact comparison to the oracle on the valid prefix, not floating tolerances. Recovery distributions use statistical tests. Model quality is evaluated separately because enabled entropy intentionally changes sampling.

Set `NPU_DEVICE_ID` to an available NPU on the test host. Run the repository tests with:

```bash
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE_ID:?Set NPU_DEVICE_ID to an available NPU}"
export VLLM_USE_V2_MODEL_RUNNER=1
python -m pytest -q tests/ut/worker/v2/test_entropy_verify.py
python -m pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample_v2.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_resample.py \
  -o junit_family=xunit1 --junitxml=entropy-contract.xml
```

Validation results apply to the recorded configurations, not all possible combinations of dimensions and hardware. Model quality and performance must be evaluated separately on the intended workload.

## Example

Use already prepared MRV2 inputs; the caller must not apply target temperature twice:

```python
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

sampled, counts = rejection_sample(
    target_logits=processed_target_logits,
    draft_logits=draft_logits,  # Or None for a one-hot/greedy draft.
    draft_sampled=draft_sampled,
    cu_num_logits=cu_num_logits,
    pos=positions,
    idx_mapping=request_mapping,
    expanded_idx_mapping=row_mapping,
    expanded_local_pos=local_positions,
    temperature=temperatures,
    seed=seeds,
    num_speculative_steps=7,
    use_entropy_verify=True,
    posterior_threshold=0.95,
    posterior_alpha=0.4,
)
```

For serving, retain the model's supported proposer configuration, select `rejection_sample_method="standard"`, and add `--additional-config '{"rejection_sampler_config":{"enable_entropy_verify":true,"posterior_threshold":0.95,"posterior_alpha":0.4}}'`. Omit it to retain the default disabled behavior. Tuning changes both acceptance and model quality; benchmark the intended workload before selecting a setting.
