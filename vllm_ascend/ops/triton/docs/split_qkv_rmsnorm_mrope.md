# Split-QKV RMSNorm MRoPE

## Description

- **Function**: Split a fused Q/K/V projection, apply per-head RMSNorm and multimodal RoPE to Q and K, and copy V and an optional Q gate to separate outputs. The public op is used by the Qwen3-VL and Qwen3.5 multimodal attention paths; it is not the TP-global RMSNorm operator.
- **Formula**: For each Q or K head `x` of width `D`, compute `rstd = rsqrt(sum(x^2) / D + eps)` in fp32 and `y = x * rstd * weight [+ bias]`. MRoPE selects temporal, height, and width cos/sin lanes from `cos_sin` according to `mrope_section`, then rotates the first `R = rope_dim` elements of `y` in two halves: `[y_1 * cos - y_2 * sin, y_2 * cos + y_1 * sin]`. Elements beyond `R` remain unrotated. V and the optional gate are copied, not normalized or rotated.
- **Algorithm flow** (tokens and heads are independent):
  1. The wrapper partitions `T = qkv.shape[0]` tokens across the initialized vector-core count `P`. Each active core owns a contiguous token range; the public op allocates Q, K, V, and gate outputs and launches one kernel instance.
  2. With the default `BLOCK_M=2` request, a scalar selector checks the input/layout contract (G1), compares a shape-derived UB demand estimate with the community UB helper's routing capacity (G2), and checks that there is enough pair work per active core (G3). Failed gates select the M1 single-row execution path.
  3. Both M1 and M2 compute the two RoPE halves directly as `x_1 * cos - x_2 * sin` and `x_2 * cos + x_1 * sin`, removing the concatenated rotation tensor and full-width cos/sin broadcast temporaries used by the main-branch baseline. A selected pair-capable M2 instance handles two token rows per loop iteration. It bounds the Q head tile to `min(num_q_heads, 12)` and processes full head tiles separately. An odd token uses the single-row tail path. The M1 path processes one complete row per iteration. Both paths preserve the same Q/K normalization, MRoPE, V, and gate semantics.
- **Supported modes**: This is a vLLM-Ascend Triton NPU op called from the Qwen3-VL and Qwen3.5 multimodal paths with optional gate, optional paired Q/K biases, partial RoPE, and interleaved or contiguous MRoPE sections. The adaptive M2 screening domain is bf16/fp16. Bounded single-operator numerical and performance evidence in this PR comes from one Ascend910B3/P40 (Atlas A2) environment; Atlas A3/950, other dtypes, and model/graph execution are not qualified by those measurements. Dispatch does not use a SoC-name or CANN-version allowlist.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qkv` | Input | Fused projection `[T, (1 + has_gate) * Q + 2 * KV]`, where `Q = num_q_heads * head_size` and `KV = num_kv_heads * head_size`; with a gate, each Q head is followed by its gate head | Floating point | 2D ND, contiguous layout for M2 |
| `q_weight` | Input | Per-head Q RMSNorm scale `[head_size]` | Floating point | 1D ND |
| `k_weight` | Input | Per-head K RMSNorm scale `[head_size]` | Floating point | 1D ND |
| `cos_sin` | Input | Three MRoPE planes `[3, T, rope_dim]`, each containing cos and sin halves | Floating point | 3D ND |
| `num_q_heads` | Attribute | Local Q head count | Integer | Scalar |
| `num_kv_heads` | Attribute | Local K/V head count | Integer | Scalar |
| `head_size` | Attribute | Q/K/V head width and RMSNorm reduction width | Integer | Scalar |
| `eps` | Attribute | Positive RMSNorm stability constant | Float | Scalar |
| `mrope_section` | Attribute | Temporal, height, and width half-RoPE section lengths | Three integers | 1D list |
| `is_interleaved` | Attribute | Select interleaved rather than contiguous MRoPE frequency layout | Boolean | Scalar |
| `rope_dim` | Optional attribute | Rotated prefix width; defaults to `head_size` | Integer or `None` | Scalar |
| `q_bias` | Optional input | Post-norm Q bias `[head_size]` | Floating point or `None` | 1D ND |
| `k_bias` | Optional input | Post-norm K bias `[head_size]` | Floating point or `None` | 1D ND |
| `has_gate` | Attribute | Whether each Q head has a corresponding gate head in `qkv` | Boolean | Scalar |
| `q_output` | Output | Normalized and rotated Q `[T, Q]` | Same as `qkv` | 2D ND |
| `k_output` | Output | Normalized and rotated K `[T, KV]` | Same as `qkv` | 2D ND |
| `v_output` | Output | Unmodified V `[T, KV]` | Same as `qkv` | 2D ND |
| `gate_output` | Output | Unmodified gate `[T, Q]`, or `[T, 0]` when absent | Same as `qkv` | 2D ND |

## Constraints

- `num_q_heads`, `num_kv_heads`, and `head_size` are positive. `rope_dim` is positive, even, and no greater than `head_size`; `2 * sum(mrope_section) == rope_dim`. `eps` is finite and positive. Positions have already been gathered into the three-plane `cos_sin` tensor by the caller.
- `q_bias` and `k_bias` must be supplied together or both omitted. The kernel enables both bias loads when `q_bias` is not `None`. The M2 selector checks the live tensor layout contract; an invalid layout falls back to M1, which is not a promise that arbitrary malformed inputs are valid for the original kernel.
- `VLLM_ASCEND_SPLIT_QKV_RMSNORM_MROPE_BLOCK_M` defaults to `2` and is read on each public call. Only `1` and `2` are accepted: `1` explicitly selects M1 without the adaptive M2 screen; `2` requests adaptive selection. The unqualified M4 kernel path has been removed; setting `4` raises `ValueError`. The variable is defined in `vllm_ascend/envs.py` and is not sensitive.
- For the default request, G1 checks semantics and tensor layout. G2 reuses the community `get_ub_size_bytes()` helper in `triton_utils.py`, as LayerNorm and Q-RMSNorm do, and a shape-derived *screening estimate*. The helper uses worker-initialized properties and its existing 192 KiB default / `VLLM_ASCEND_ROPE_UB_SIZE_KB` override. This capacity can be detected, defaulted, or overridden; it is **not an independently measured compiler allocator budget**. The wrapper does not import private backend UB fields or add another fallback. Helper errors/invalid values select M1, as does estimated demand at or above capacity. The estimate and its difference from capacity are **not measured UB usage or free headroom**.
- Until boundary liveness is calibrated, pair-capable M2 requires `num_q_heads` to be divisible by `min(num_q_heads, 12)`; a positive remainder selects M1. This structural guard follows an Hq16/D256 allocator overflow in a simulator compile probe. An Hq16/D128 boundary compiled in a separate simulator probe, so the guard is intentionally conservative rather than a claim that every boundary shape overflows on device.
- G3 selects M1 for tail-only work or when `floor(min_active_tokens_per_core / 2) < 12`; otherwise a G1/G2-cleared pair path selects M2. At `P=40`, `T=959` falls below this trial threshold and `T=960` reaches it. This is a workload policy supported by bounded tests, not a universal optimal crossover. A one-token call remains on M1.
- The sole M2 launch uses `multibuffer=False`, `num_stages=1`, and `num_warps=32`; an M1 fallback uses the original kernel/backend default. There is no separate M2 tail-only specialization, but M2 still processes an odd final token through its single-row tail body. Runtime selection does not load an evidence registry, hash source files, inspect SoC/toolchain identity, or maintain a process-global decision record. Experiment evidence remains in the research repository and Git history, not in the per-call production path. No route is claimed fastest for every supported input or hardware configuration.

## Origin and Differences

- **Origin**: The pre-existing `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py` public operator and its Qwen3-VL/Qwen3.5 callers. This PR combines the Phase 1 kernel work with WP1 adaptive dispatch; it does not introduce a second public operator.
- **Differences**:
    - Compute RoPE halves directly in the single-row and paired paths, preserving the unrotated suffix for partial RoPE. A fallback to M1 retains this optimization.
    - Pair two token rows on suitable long partitions and bound Q head tiles to reduce the per-instance live tensor extent; preserve the single-row M1 path for unsupported or less useful work.
    - Separate semantic/layout, resource feasibility, and workload-benefit decisions. The resource guard prevents selecting the known-unsafe D256 pair-boundary class without converting measured cases, SoC names, or toolchain versions into an enablement whitelist.
    - Preserve the public call signature and output contract while allowing the selected kernel instance and launch configuration to change.

## Test Cases

- Host-only selector tests check representative B3 routes, the Hq24/T960 long-partition boundary, singleton partitions, unknown capacity, invalid layout, and the centralized override without importing an NPU runtime:

  ```bash
  python3 -m unittest discover -s tests/ut/ops -p test_split_qkv_rmsnorm_mrope_dispatch.py
  ```

- Host-only checks cover the shared UB helper, invalid capacity, exact-fit rejection, tensor layout, semantic constraints, the centralized override, and the real wrapper's Python selection/launch wiring using tensor/launch stubs. The production cleanup removes evidence/identity machinery and the unused M2 tail-only specialization, reuses the existing token partition, and preserves the G2 estimate and G3 threshold. Structural comparison with the pre-cleanup source checks the M1, M2 pair and odd-tail computation bodies. The device results below cover the cleaned-up routes and a same-run four-route ablation of the current PR kernel.

- The pre-existing single-card accuracy suite covers bf16/fp16, gate/no-gate, interleaved/contiguous MRoPE, two token counts, and two head configurations. It was not run on the exact current PR head in this work:

  ```bash
  pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_mrope.py
  ```

## Bounded offline validation

### Same-run main-baseline ablation

The primary baseline is the unmodified main/Phase 0 operator, byte-identical at the
PR's local main base `be0b61da22778b043eb2369b5f922b2a39ddc46d` (source SHA-256
`ba3c7e740c3a2a9c8850733d3be77fefac79bc3ed02a45b4be076b58d6f7632d`).
It constructs the rotation tensors and full-width cos/sin temporaries removed by
direct half RoPE. The candidate measured here is PR source commit `5b09d01c`
(wrapper SHA-256 `6089f901fb35ddcbe6a6af698384029a43a2d30451f14bd9a40f4f0e6514eea1`).
Later documentation-only changes do not change these measured kernel bytes.

The Qwen3.5 caller and frozen model configuration define the following local head
geometries. Inputs are fixed-seed **synthetic representative tensors**, not
captured model activations; TP describes the source geometry, not a distributed
experiment. P1/P4 retain their B4-origin input bytes with a B3 execution overlay;
P5 originated as B3.

| Case | Model geometry / TP | Tokens | Local Q/KV heads | Fused QKV shape |
| --- | --- | ---: | --- | --- |
| P1 | Qwen3.5-27B / TP4 | 192 | 6 / 1 | [192, 3584] |
| P2 | Qwen3.5-27B / TP1 | 192 | 24 / 4 | [192, 14336] |
| P3 | Qwen3.5-27B / TP4 | 1 | 6 / 1 | [1, 3584] |
| P4 | Qwen3.5-27B / TP4 | 4096 | 6 / 1 | [4096, 3584] |
| P5 | Qwen3.5-122B-A10B / TP8 | 8192 | 4 / 1 | [8192, 2560] |

All cases use bf16, per-head RMSNorm (`eps=1e-6`), `head_size=256`,
`rope_dim=64`, `mrope_section=(11, 11, 10)`, gate enabled, interleaved
Q/gate and MRoPE, and no Q/K bias.

Four routes use the same input and seed `20260812` within
`run_20261009T120233461334Z_all_c87fd60a` (`pkg_e0b2a8d76a0073b7`):

- **BASE**: unmodified main kernel, single row, backend-default multibuffer.
- **M1**: current PR direct-half kernel, explicit single row, backend default.
- **M2/off**: current PR kernel, forced M2 only inside the experiment by bypassing
  G2/G3, not G1; `multibuffer=False`, `num_stages=1`, `num_warps=32`.
- **PR auto**: unmodified production wrapper with BLOCK/UB overrides unset, selecting M1 for
  P1/P2/P3 and M2/off for P4/P5. Production gates are not bypassed.

This ablation directly invokes the same wrapper; it does not add another public-op
registration test. Separate public-entry numerical protection is recorded below.

Each cell reports **median device duration (us) / speedup over BASE**:
`BASE median / route median`, calculated before rounding. BASE is 1x.

| Case | BASE | M1 | Forced M2/off | PR auto (actual route) |
| --- | --- | --- | --- | --- |
| P1 | 14.230 / 1.0000x | 13.860 / 1.0267x | 13.050 / 1.0904x | 13.650 / 1.0425x (M1) |
| P2 | 20.340 / 1.0000x | 18.270 / 1.1133x | 18.560 / 1.0959x | 18.410 / 1.1048x (M1) |
| P3 | 6.520 / 1.0000x | 5.860 / 1.1126x | 6.490 / 1.0046x | 5.740 / 1.1359x (M1) |
| P4 | 193.330 / 1.0000x | 183.870 / 1.0514x | 155.200 / 1.2457x | 155.990 / 1.2394x (M2) |
| P5 | 360.450 / 1.0000x | 346.060 / 1.0416x | 282.700 / 1.2750x | 283.700 / 1.2705x (M2) |

The run reported 20/20 numerical passes, 20/20 terminal canaries and 120/120 valid
timings. All Q/K outputs were finite and passed `atol=rtol=0.02`; V/Gate were
byte-exact. The received configuration summaries confirm the expected B3
M1/default-on or M2/off instances, stages 1 and warps 32.

P4/P5 are the main measured long-partition gains. M1 retains direct half RoPE even
when adaptive dispatch falls back. The M1 control is a route-level ablation, not
an isolation of every Python-wrapper difference. Forced M2 is not uniformly
better than M1: P2 is slightly slower and P3 is about 10.8% longer; P3 has only
a single-row tail, no pair work. P1's forced-M2 result is better than the selected
M1, so the conservative workload threshold is not claimed universally optimal.
These five M2 successes do not invalidate the Hq16/D256 boundary overflow or
justify relaxing G2.

### Measurement and evidence limits

- One Ascend910B3/P40 device (host card 3, logical card 0), the rc1 image ID
  `sha256:13315b656180f24489bb0076ff449fd5dcc15f84f447872ce38bf7e43606489a`,
  and the provisional OBSFS mount-managed profile were used. This SHA is a local
  **image ID**, not a registry manifest digest. The older image was staged with
  the exact main UB helper and environment getters, not a mock capacity.
- Each route has six samples from `msprof op`, five warm-ups and one profiled
  target-kernel launch. These are same-run fixed-order rounds, **not randomized
  or ABBA-paired estimates**. Raw samples, including spikes, are retained.
  Maximum min/max spread across all cells was 9.7561%; PR P4/P5 spreads were
  1.2565%/0.7261%. A single seed/run is not a significance or equivalence test.
  The auxiliary 1.05x gain / 0.98x regression labels are not statistical bounds;
  P1's 1.0425x does not clear the gain label.
- The environment, six-sample protocol and BASE metric were confirmed before
  packaging, but the additional fixed-order / 10% spread-hint confirmation was
  still pending when the user executed `run-all`. The research archive records
  that process deviation; the executed protocol is not relabeled as fully
  pre-confirmed.
- Received compact fragments, supplemental per-item summaries, completed/committed
  status and the mutation-free host handoff form **report-level evidence**.
  The full target result tree and its checksums were not independently rehashed
  locally. The target-reported result manifest SHA-256 is
  `c8dd6761c0df38ecadf0d67bb51a5cf8d696b9fb65e6ad90052aa64c3a872009`.
  Input/source identities, all samples and reconstruction boundaries are retained
  in the research archive; no target artifacts were deleted.
- Prior default-versus-off P4/P5 measurements had mixed directions. Keeping
  M2/off preserves the tested configuration, not a claim that off is always
  fastest or that the backend default is unsafe. No selector or resource-model
  change is inferred from this ablation.

### Supplementary route protection

The earlier cleaned-up source `e0f26aeb` passed a three-point public-entry numerical
run (`run_20261009T072317173553Z_numerics_aaee9b8d`,
`pkg_3a16bd076842b8c8`) on the same B3/P40 runtime, with all overrides unset:

| Case | Tokens | Q/KV heads | Actual route | Q / K maximum absolute error | V / Gate |
| --- | ---: | --- | --- | --- | --- |
| HQ24_T1024_B3 | 1024 | 24 / 4 | M2, shape_resource, multibuffer off | 0.015625 / 0.00390625 | Byte-exact |
| P3_B3 | 1 | 6 / 1 | M1, g3_workload, multibuffer on | 0 / 0 | Byte-exact |
| E2C_B3 | 192 | 16 / 4 | M1, g2_resource, multibuffer on | 0.0078125 / 0.0078125 | Byte-exact |

All outputs were finite and passed the stated tolerances. HQ24/T1024 includes odd
final rows and protects M2's single-row tail; the other points protect singleton
and pair-boundary fallbacks. This earlier run is numerical protection, not a
timing measurement of the exact current source.

T1024 with 6/1 heads is a separate long-partition transition probe, not one of
P1-P5. An earlier source `fd671ea7` measured 1.0447x incremental paired gain over
an already direct-half M1 in `run_20261008T073722771387Z_all_9e4872ed`; this is
below the 1.05x auxiliary gain label, not a main-baseline speedup. It remains
eligible under the workload policy without a claim of clear performance benefit.
Historical cross-run P5 ratios are superseded by the same-run table above.

No matching-main NPU Nightly, model/graph-replay test, measured UB-liveness upper
bound, or broader hardware/dtype/runtime qualification is claimed. Simulator IR
and resource screening do not establish device numerical correctness or actual
free UB headroom.
