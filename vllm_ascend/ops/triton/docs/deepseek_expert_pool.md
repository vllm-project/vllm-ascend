# DeepSeek expert-pool routing

## Description

- **Function**: `route_experts` builds a batch-wide expert candidate pool and
  returns logical expert IDs and normalized FP32 weights for each row. This is
  a model-specific, lossy optimization for compatible DeepSeek DSpark routing.
- **Formula**: unbiased scores are `sqrt(softplus(logits))`. Selection ranks
  `score + bias`; weights use the unbiased selected scores, divided by
  `max(sum(selected_scores), 1e-20)` and multiplied by `scaling`. Equal selection
  scores prefer the lower expert ID.
- **Algorithm flow**:
    1. `_build_pool` computes scores and votes. Role 1 (`CONTRIBUTOR`) contributes
       TopK experts, role 2 (`RESTRICTED`) contributes `suffix_experts`, and role
       0 (`UNRESTRICTED`) contributes none. Votes are shared across requests.
    2. `_finish_route` selects TopK from the pool for restricted rows and from all
       experts for other rows, then computes normalized routing weights.
    3. `AscendFusedTopKRouter._compute_routing` selects the pool or native compute
       path. The common router entry captures final logical IDs and performs
       its normal post-processing after routing or any required row-wise merge.
- **Supported modes**: operator kernels target Atlas A2, Atlas A3, and Ascend
  950PR/950DT. Production expert-offload admission currently requires Ascend 950;
  operator availability does not enable offload on other hardware.

Protected positions include the anchor. P1 protects only the anchor; P2 protects
the anchor and first draft position. Protected rows contribute TopK experts and
select from all experts. Later verification rows contribute E candidates and
select TopK from the resulting pool, even when E is zero. The current supported
model uses TopK = 6.

In batches without prefill, a real one-token request with zero scheduled drafts
also becomes a contributor. Its selected experts expand the pool available to
other requests. Such decode/verify batches use the pool computation without an
additional native routing pass. Any genuinely unclassified real rows retain
native routing through a paired weights/IDs merge. Padding stays role 0 and
cannot expand the pool. A batch containing prefill uses native routing entirely.

## Parameters

All parameters are required. Let `R` be the row count and `E` the expert count.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits` | Input | Router logits, shape `[R, E]` | FP32 | Strided 2D tensor |
| `roles` | Input | Row roles 0, 1, or 2, shape `[R]` | INT32 | Contiguous |
| `bias` | Input | Selection-only correction, shape `[E]` | FP32 | Contiguous |
| `raw` | Input/Output | Score scratch, shape `[R, E]`; overwritten | FP32 | Contiguous |
| `votes` | Input/Output | Pool scratch, shape `[E]`; reset and overwritten | INT32 | Contiguous |
| `suffix_experts` | Attribute | Candidates per restricted row, in `[0, TopK]` | Python int | Scalar |
| `scaling` | Attribute | Finite routed scaling factor | Python int or float | Scalar |
| Returned weights | Output | Newly allocated tensor, shape `[R, TopK]` | FP32 | Contiguous |
| Returned IDs | Output | Newly allocated logical IDs, shape `[R, TopK]` | INT32 | Contiguous |

## Constraints

- All tensors share a device, `E >= TopK`, and scratch does not alias inputs or
  other scratch. Logits may be noncontiguous; the other tensors are contiguous.
  Unsupported metadata is rejected before scratch mutation or kernel launch.
- The caller provides valid role values. Production configuration requires at
  least one protected position, so a normal valid verification pool contains at
  least TopK distinct experts. P0 is rejected. The operator does not copy roles
  or scores to the host to validate their contents.
- Softplus uses its linear branch above 20 and bounds the exponential branch
  to avoid FP32 overflow. Below this threshold it retains the original
  `log(exp(x) + 1)` rounding, including zero scores and tie behavior.
- Nonfinite selection scores are excluded. If invalid data or an insufficient
  pool leaves fewer than TopK candidates, missing slots receive expert ID 0 and
  weight 0. Gather indices remain in range. This prevents invalid memory access;
  it does not establish meaningful model accuracy for invalid inputs.
- Inputs `logits`, `roles`, and `bias` remain unchanged. Only `raw` and `votes`
  are modified in place; returned outputs are new allocations. Empty batches
  return empty outputs without changing scratch or launching kernels.
- The manager owns the reusable roles and scratch. Uniform eligible verify
  batches support full decode graph capture/replay. Mixed or short verify
  batches retain eager execution. Role buffers are refreshed in place; offload
  separately makes padding reuse the first real row's IDs with zero weights
  before deciding which expert weights to load.
- Draft-model execution, profiling, disabled pools, and incompatible routers
  use native routing. Warmup registration for this operator: N/A.

## Origin and Differences

- **Origin**: developed for the DSpark expert-pool routing feature in PR #21680.
  Native `sqrtsoftplus` TopK routing supplies the unrestricted baseline; the
  test file supplies an independent PyTorch sorting/voting reference.
- **Differences**: two Triton kernels add batch-level voting and restrict suffix
  rows to the shared pool. Correction bias affects selection only. Stable
  softplus and safe missing-candidate handling cover numerical boundaries.
- **JIT specialization**: expert count, power-of-two block size, logits strides,
  suffix candidate count, TopK, role constants, and scaling are compile-time
  parameters. Pointer element types and alignment can affect specialization.
  P changes runtime role contents; role values, logits contents, and launch-grid
  changes alone do not require another specialization. Actual compilation also
  depends on the Triton/backend version, device target, and cache state.

## Test Cases

The NPU reference compares expert IDs exactly. FP32 weights require at least
99% of elements within `atol + rtol * abs(reference)`, with `rtol = 2^-10`,
`atol = 2^-16`, and maximum absolute error no greater than 0.01.

Coverage includes real verification shapes `[6, 256]`, `[12, 256]`, and
`[18, 256]`; strided and contiguous logits; ties and zero scores; P/E boundaries;
real zero-draft contributors; padding isolation; large finite logits; NaN
scores; and insufficient pools with minimum and non-power-of-two expert counts.
CPU tests cover role preparation, prefill transitions, router capture, native
fallback, configuration validation, and unchanged wrapper inputs.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dspark_expert_pool.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/distributed/expert_transfer tests/ut/ops/test_deepseek_expert_pool.py
```
