# deepseek_expert_pool

## Description

- **Function**: Fused Triton-Ascend expert routing for the DeepSeek (DSpark "L") single-device expert-offload verification segment. For each token row of a prepared batch it computes the sqrt-softplus router scores, builds a batch-wide expert pool from the votes of the verification rows, and emits logical expert IDs with renormalized FP32 routing weights. Rows with role CONTRIBUTOR (protected/anchor positions, role 1) each vote their top-6 experts into the pool; rows with role RESTRICTED (suffix draft positions, role 2) each vote their top-`suffix_experts` experts; unrestricted rows (role 0) do not vote. In the final routing, restricted rows may only select experts that received at least one vote, while all other rows route over the full expert set. The caller (`DSparkExpertPool.merge_native`) overwrites the outputs of unrestricted rows with the native router results, so in production only the verification rows consume the fused outputs. The path is opt-in via `additional_config.dspark_expert_pool` (requires enabled expert offload and DSpark speculation).
- **Formula**: For row `r` and expert `e` (all arithmetic in FP32):
    - raw score: `raw[r,e] = sqrt(log(exp(x[r,e]) + 1.0))` where `x = float32(logits[r,e])`. The softplus is written verbatim (no overflow-stable rewrite) to preserve the frozen L score rounding, including its effect on ties.
    - selection score: `score[r,e] = raw[r,e] + bias[e]`; the correction bias is used for selection only, never for weights.
    - pool votes: `votes[e] =` number of verification rows whose top-k selection (top-6 for role 1, top-`SUFFIX_EXPERTS` for role 2) contains `e`; selection runs K rounds of argmax over `score` and always breaks ties by the smallest expert ID.
    - final route: candidates for row `r` are all experts when `role != RESTRICTED`, otherwise `{e : votes[e] > 0}`; the top `K = 6` candidates by `score` (same smallest-ID tie-break) give `ids[r,0..5]`.
    - weights: `weights[r,i] = raw[r, ids[r,i]] / max(sum_j raw[r, ids[r,j]], 1e-20) * scaling` — bias-free raw scores, L1-renormalized over the selected experts, scaled by `routed_scaling_factor`.
- **Algorithm flow** (processed row by row, independently, across two kernel launches):
  1. The wrapper `route_experts` zeroes the `votes` scratch and derives `BLOCK = next_power_of_2(experts)`.
  2. `_build_pool` launches one program per row on the vector core (`core_mode="vector"`). Each program loads its row of `logits` (masked to `EXPERTS`), casts to FP32, computes and stores the raw scores into the `raw` scratch, loads its role, and, when the role's vote limit is positive, runs K argmax rounds over `raw + bias` (smallest-ID tie-break, one expert removed per round) and atomically adds one relaxed vote per selected expert.
  3. `_finish_route` launches one program per row. Each program loads its role and the votes (`allowed = votes > 0`), restricts candidates to the pool only for RESTRICTED rows, runs K argmax rounds over `raw + bias` to produce the IDs, gathers the bias-free raw scores of those IDs, L1-normalizes them (denominator clamped at 1e-20), scales, and stores the FP32 weights and int32 IDs.
  The launch boundary orders the two kernels: kernel 2 reads votes only after kernel 1's atomics have completed for the whole batch.
- **Supported modes**: Triton-Ascend vector-core execution on the Ascend 950 (Atlas A5) family, which the expert-offload configuration enforces at startup, in eager mode and ACL graph capture (compilation mode NONE, `FULL_DECODE_ONLY` decode graphs; persistent scratch buffers keep captured addresses stable). The committed unit test is CPU-only and launches no kernel; NPU operator-level accuracy is exercised by the nightly test at `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dspark_expert_pool.py`, added in parallel with this operator. Device-by-device validation evidence is recorded there; none is claimed here. Inference only.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits` | Input | Router logits for the padded routed batch; cast to FP32 inside the kernel | FP32 (enforced by the caller) | 2D `[rows, experts]`, arbitrary strides |
| `roles` | Input | Per-token role codes: 0 unrestricted, 1 CONTRIBUTOR, 2 RESTRICTED; any other value behaves as 0 | int32 | 1D `[rows]`, contiguous |
| `bias` | Input | Expert correction bias added to the selection score only; the caller substitutes zeros when the router has none and casts to the logits dtype | FP32 (same as `logits`) | 1D `[experts]`, contiguous |
| `raw` | Input/Output (scratch) | Raw sqrt-softplus scores written by `_build_pool` and re-read by `_finish_route`; persistent graph workspace sliced to `[rows]` when the batch fits the workspace capacity | FP32 | 2D `[rows, experts]`, contiguous row-major |
| `votes` | Input/Output (scratch) | Batch pool votes; zeroed by the wrapper on entry, accumulated by `_build_pool` | int32 | 1D `[experts]`, contiguous |
| `suffix_experts` | Attribute | Number of pool candidates each RESTRICTED row votes for | Python int | Scalar in `[0, 6]` |
| `scaling` | Attribute | `routed_scaling_factor` applied after weight normalization | Python float | Scalar |
| `weights` | Output | Renormalized routing weights on the bias-free raw basis, multiplied by `scaling`; freshly allocated by the wrapper | FP32 | 2D `[rows, 6]`, contiguous |
| `ids` | Output | Selected logical expert IDs; freshly allocated by the wrapper | int32 | 2D `[rows, 6]`, contiguous |

## Constraints

- `logits` must be FP32 and 2D with `experts == router.num_logical_experts` and one row per prepared token role; `DSparkExpertPool.route` raises `ValueError` otherwise. `logits` may carry arbitrary row/expert strides (both are passed to the kernel); every other tensor must be contiguous.
- `roles` must be int32, contiguous, with values in `{0, 1, 2}`. The caller's role construction (`build_roles` with `protected_positions` in `[1, num_speculative_tokens + 1]`) always pairs RESTRICTED rows with CONTRIBUTOR rows, and each CONTRIBUTOR row votes 6 distinct experts, so the pool always holds at least `K` candidates and the emitted IDs stay in `[0, experts)`.
- `bias` must be a contiguous `[experts]` tensor in the logits dtype. `raw` must be a contiguous FP32 `[rows, experts]` buffer (a leading-dimension slice of a persistent `[workspace_rows, experts]` workspace satisfies this); `votes` must be a contiguous int32 `[experts]` buffer. The wrapper zeroes `votes` per call, so the scratch is reusable across calls but must not be shared by concurrent calls.
- `K` is fixed at `TOP_K = 6`; the router gate (`supports_router`) additionally requires `scoring_func == "sqrtsoftplus"`, `top_k == 6`, `num_logical_experts >= 6`, renormalization enabled, and no tid2eid/bias_vl/custom routing/grouped-topk or EPLB log2phy mapping. `suffix_experts` is validated at config time to `[0, 6]`.
- The softplus is evaluated verbatim as `log(exp(x) + 1.0)` in FP32 with no clamping; logits large enough to overflow FP32 `exp` (roughly `x > 88.7`) produce inf raw scores, exactly as the frozen expression dictates.
- `EXPERTS`, `BLOCK = next_power_of_2(experts)`, both `logits` strides, `SUFFIX_EXPERTS`, the two role codes, `K`, and `SCALING` are compile-time constants: each distinct `(experts, strides, suffix_experts, scaling)` tuple compiles a separate variant. `rows` only changes the launch grid (`multibuffer=False`). Output staging uses 8 register lanes with stores masked to `K = 6`.
- Scratch placement: batches with `rows <= min(capacity, num_device_experts // 6)` reuse a persistent per-expert-count workspace (stable addresses for graph capture); larger eager batches allocate fresh contiguous scratch per call.
- Configuration: the feature requires `additional_config.dspark_expert_pool.enabled` with enabled expert offload, DSpark speculation with a positive `num_speculative_tokens`, `protected_positions` in `[1, num_speculative_tokens + 1]`, and `suffix_experts` in `[0, 6]`. Graphs at the full verify width record the merged routing sequence (fused route, native router, `merge_native`), so replays handle any role layout the per-step staging produces — structured output, rejected drafts, or prefilling requests included; narrower graphs record native-only routing, and uniform steps with short drafts (`has_short_drafts`) run eager instead of replaying them.
- Ties in every argmax round are broken by the smallest expert ID; `votes` uses relaxed atomics that are ordered by the kernel launch boundary. `raw` and `votes` are intentional in-place scratch; `logits`, `roles`, and `bias` are not modified.

## Origin and Differences

- **Origin**: Developed from scratch as a native Triton-Ascend implementation for the opt-in DSpark L routing. The per-row scoring and normalization contract it reproduces is the DeepSeek sqrtsoftplus route of `AscendFusedTopKRouter` in `vllm_ascend/ops/fused_moe/router/fused_topk_router.py` (`softplus(logits).sqrt()`, correction bias for selection only, renormalize, `routed_scaling_factor`), which remains the native/fallback path and still serves unrestricted rows via `merge_native`.
- **Differences**:
    - NPU adaptation for performance: the eager per-row top-K chain is fused into two vector-core kernel launches — one pass computes raw scores and accumulates the batch pool with relaxed atomics, the second performs the pooled top-K, normalization, and scaling — with persistent scratch buffers that keep tensor addresses stable for ACL graph capture.
    - Modified for a specific vllm-ascend logic: new pool semantics (RESTRICTED rows may only use pooled experts), role-coded inputs, in/out scratch (`raw`, `votes`) instead of a pure-function signature, a fixed `K = 6`, and tie/rounding preservation — the sqrt-softplus expression is kept verbatim and ties always resolve to the smallest expert ID so routing matches the reference bit-for-bit where scores do.

## Test Cases

- CPU unit contract (launches no NPU kernel): `pytest -sv tests/ut/expert_offload/test_dspark_expert_pool.py` asserts role construction (protected/suffix split, padding rows left at 0, prefilling requests excluded, the protected-position clamp), short-draft eager detection (`has_short_drafts`), same-graph-bucket padding refresh with role-buffer address reuse and stale-role clearing on inactive steps, structured-output deactivation with cleared roles, native-output preservation outside the verification segment (`merge_native`), and capture-time uniform verify-width roles.
- NPU nightly accuracy (added in parallel with this operator): `pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dspark_expert_pool.py` compares the fused route against the reference router path on the model's DSpark verification shapes; the comparison cases and tolerance live in that file.

```bash
pytest -sv tests/ut/expert_offload/test_dspark_expert_pool.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dspark_expert_pool.py
```
