# Split-QKV RMSNorm MRoPE

## Description

- **Function**: Split a fused Q/K/V projection, apply per-head RMSNorm and multimodal RoPE to Q and K, and copy V and an optional Q gate to separate outputs. The public op is used by the Qwen3-VL and Qwen3.5 multimodal attention paths; it is not the TP-global RMSNorm operator.
- **Formula**: For each Q or K head `x` of width `D`, compute `rstd = rsqrt(sum(x^2) / D + eps)` in fp32 and `y = x * rstd * weight [+ bias]`. MRoPE selects temporal, height, and width cos/sin lanes from `cos_sin` according to `mrope_section`, then rotates the first `R = rope_dim` elements of `y` in two halves: `[y_1 * cos - y_2 * sin, y_2 * cos + y_1 * sin]`. Elements beyond `R` remain unrotated. V and the optional gate are copied, not normalized or rotated.
- **Algorithm flow** (tokens and heads are independent):
  1. The public op allocates Q, K, V, and gate outputs and launches one kernel instance. Ordinarily the wrapper partitions `T = qkv.shape[0]` tokens across the initialized vector-core count `P`; each active core owns a contiguous token range.
  2. With the default `BLOCK_M=2` request, qualified single-token inputs instead distribute `max(num_q_heads, num_kv_heads)` independent head jobs across cores. Each job handles one Q/gate head and, where present, one K/V head. Three contiguous RoPE-plane loads replace six masked coefficient loads. Other inputs retain the scalar selector: input/layout contract (G1), shape-derived UB screening versus the community helper's routing capacity (G2), and sufficient pair work per active core (G3). Layout/resource/workload rejection selects ordinary M1; invalid scalar parameters can raise before selection.
  3. Both M1 and M2 compute the two RoPE halves directly as `x_1 * cos - x_2 * sin` and `x_2 * cos + x_1 * sin`, removing the concatenated rotation tensor and full-width cos/sin broadcast temporaries used by the main-branch baseline. A selected pair-capable M2 instance handles two token rows per loop iteration. It bounds the Q head tile to `min(num_q_heads, 12)` and processes full head tiles separately. An odd token uses the single-row tail path. The M1 path processes one complete row per iteration. Both paths preserve the same Q/K normalization, MRoPE, V, and gate semantics.
- **Supported modes**: This is a vLLM-Ascend Triton NPU op called from the Qwen3-VL and Qwen3.5 multimodal paths with optional gate, optional paired Q/K biases, partial RoPE, and interleaved or contiguous MRoPE sections. The adaptive M2 screening domain is bf16/fp16. Bounded single-operator numerical and performance evidence in this PR comes from one Ascend910B3/P40 (Atlas A2) environment; Atlas A3/950, other dtypes, and model/graph execution are not qualified by those measurements. Dispatch does not use a SoC-name or CANN-version allowlist.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qkv` | Input | Fused projection `[T, (1 + has_gate) * Q + 2 * KV]`, where `Q = num_q_heads * head_size` and `KV = num_kv_heads * head_size`; with a gate, each Q head is followed by its gate head | Floating point | 2D ND, contiguous packed layout |
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
- `VLLM_ASCEND_SPLIT_QKV_RMSNORM_MROPE_BLOCK_M` defaults to `2` and is read once on each public call. Only `1` and `2` are accepted: `1` explicitly selects ordinary M1, bypassing both head grouping and UB queries; `2` requests adaptive selection. The unqualified M4 kernel path has been removed; setting `4` raises `ValueError`. The variable is defined in `vllm_ascend/envs.py` and is not sensitive.
- All routes use contiguous packed addressing, including ordinary M1; fallback is not arbitrary-stride support or a repair for malformed inputs. M2/grouped additionally check tensor shapes, contiguity, dtype and device before selection.
- Single-token head grouping requires real contiguous NPU tensors, bf16, `head_size=256`, `rope_dim=64`, gate enabled, no bias, interleaved MRoPE sections `(11, 11, 10)`, `eps=1e-6`, an initialized vector-core count of exactly 40, and `2 <= max(num_q_heads, num_kv_heads) <= 40`. All shapes, dtypes and devices must match the existing layout contract. UB capacity must be available from the helper and within the selector's plausibility range (64 KiB to 1 MiB, integer bytes); the whole-head M2 demand estimate is not a grouped-kernel allocator budget. This is a parameter-domain guard, not a model/case whitelist. Other geometries, unknown capacity, P48, and other semantics retain the previous routing. Only 4/1, 6/1 and 24/4 head geometries have bounded device performance evidence; the broader parameter predicate is not exhaustive device qualification.
- For the default request, G1 checks semantics and tensor layout. G2 reuses the community `get_ub_size_bytes()` helper in `triton_utils.py`, as LayerNorm and Q-RMSNorm do, and a shape-derived *screening estimate*. The helper uses worker-initialized properties or its 192 KiB default; upstream #18851 removed the UB-size environment override. Neither detected nor defaulted capacity is **an independently measured compiler allocator budget**. The wrapper does not import private backend UB fields or add another fallback. Helper errors/invalid values select M1, as does estimated demand at or above capacity. The estimate and its difference from capacity are **not measured UB usage or free headroom**.
- Until boundary liveness is established, pair-capable M2 requires `num_q_heads` to be divisible by `min(num_q_heads, 12)`; a positive remainder selects M1. This structural guard follows an Hq16/D256 allocator overflow in a simulator compile probe. An Hq16/D128 boundary compiled in a separate simulator probe, so the guard is intentionally conservative rather than a claim that every boundary shape overflows on device.
- Outside the single-token grouped path, G3 selects ordinary M1 for tail-only work or when `floor(min_active_tokens_per_core / 2) < 12`; otherwise a G1/G2-cleared pair path selects M2. At `P=40`, `T=959` falls below this trial threshold and `T=960` reaches it. This is a workload policy supported by bounded tests, not a universal optimal crossover. T4 retains ordinary M1, including its historical negative evidence for head grouping.
- The sole M2 launch uses `multibuffer=False`, `num_stages=1`, and `num_warps=32`; an M1 fallback uses the original kernel/backend default. There is no separate M2 tail-only specialization, but M2 still processes an odd final token through its single-row tail body. Runtime selection does not load an evidence registry, hash source files, inspect SoC/toolchain identity, or maintain a process-global decision record. Experiment evidence remains in the research repository and Git history, not in the per-call production path. No route is claimed fastest for every supported input or hardware configuration.

## Origin and Differences

- **Origin**: The pre-existing `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py` public operator and its Qwen3-VL/Qwen3.5 callers. This PR combines direct-half RoPE, WP1 paired-row dispatch and WP3 single-token head grouping; it does not introduce a second public operator.
- **Differences**:
    - Compute RoPE halves directly in the single-row and paired paths, preserving the unrotated suffix for partial RoPE. A fallback to M1 retains this optimization.
    - Pair two token rows on suitable long partitions and bound Q head tiles to reduce the per-instance live tensor extent; preserve the single-row M1 path for unsupported or less useful work.
    - Distribute qualified single-token head work across cores and load each T/H/W coefficient plane contiguously; mask nonexistent Q or K/V heads independently without changing output ownership.
    - Separate semantic/layout, resource feasibility, and workload-benefit decisions. The resource guard prevents selecting the known-unsafe D256 pair-boundary class without converting measured cases, SoC names, or toolchain versions into an enablement whitelist.
    - Preserve the public call signature and output contract while allowing the selected kernel instance and launch configuration to change.

## Test Cases

- Host-only selector tests check representative B3 routes, the Hq24/T960 long-partition boundary, singleton partitions, unknown capacity, invalid layout, and the centralized override without importing an NPU runtime:

  ```bash
  python3 -m unittest discover -s tests/ut/ops -p test_split_qkv_rmsnorm_mrope_dispatch.py
  ```

- Host-only checks cover shared-helper detection/defaults, ignored legacy UB overrides, invalid capacity, exact-fit rejection, tensor layout, semantic constraints, the centralized BLOCK_M opt-out and real wrapper selection/launch wiring. They cover the three measured T1 geometries, asymmetric head counts, P48 fallback and the unchanged fake ABI. A small NumPy fixture executes the actual grouped and M2 JIT function bodies, including complete Q-head tiles and odd token tails, checking mathematical results, exactly-once input/output ownership and V/Gate byte copies including signed zero. Forced M2 head remainders are rejected while the same M1 geometries remain supported. This FP32 CPU fixture does not emulate BF16 rounding, compilation, NPU timing or device accuracy. The historical four-route ablation below applies to the earlier WP1 source, not the new grouped route.

- Head-boundary contracts additionally connect the real wrapper to the CPU kernel-body fixture for 2/1, 1/2, 40/1 and 1/40 Q/KV heads. Valid-layout 1/1 and 41/41 inputs are excluded from grouping and retain ordinary M1; their rejection is not hidden by a mismatched packed shape. These checks establish host mapping and ownership, not device or compiler qualification for these six shapes.

- The pre-existing single-card accuracy suite covers bf16/fp16, gate/no-gate, interleaved/contiguous MRoPE, two token counts, and two head configurations. It was not run on the exact current PR head in this work:

  ```bash
  pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_mrope.py
  ```

## Bounded offline validation

### Wide-head executability

For eligible single-token bf16 inputs, head grouping bounds the explicit per-program head tiles. The integrated implementation passes device numerical checks on Ascend910B3 for three tested, shape-valid wide-head inputs whose frozen baseline is rejected for insufficient UB during default B3-target compilation. An 8/32 case additionally checks the reverse asymmetric mask, without a BASE comparison.

| Q/KV heads | Frozen BASE default B3-target compilation | Integrated public AUTO on B3 |
| --- | --- | --- |
| 32/32 | UB required 303,264 B > 196,608 B | Three numerical calls and identity canary pass |
| 40/40 | UB required 378,112 B > 196,608 B | Three numerical calls and identity canary pass |
| 32/16 | UB required 233,632 B > 196,608 B | Three numerical calls and identity canary pass |
| 8/32 | Not tested | Three numerical calls and identity canary pass |

These are synthetic operator-contract inputs, not confirmed model deployments: T1, D256/R64, gate, no bias, bf16, eps=1e-6 and interleaved sections (11,11,10), within the grouped conditions above. Q/K pass finite and atol/rtol 0.02 checks; V/gate are byte-exact and repeated calls are deterministic. Device reports identify the current operator SHA `ccacdcb6f7e428d89cc2b37be8175b623813ed5650f9cd4489e213af582b2c24`; the performance tables below retain their earlier source attribution.

The 40/40 and 32/16 BASE/current compilation comparisons used the same fixed rc1-a3 image targeting B3, with backend-default multibuffer, stages 1 and warps 32. Their candidate objects and metadata match the device-reported identities from the rc1 image; this is not a claim that both images contain identical toolchains. The 32/32 BASE rejection and earlier grouped compilation remain historical evidence, separately attributed from the current device run. Results are report-level, not a local rehash of all target artifacts. They support bounded executability coverage, not an all-shape UB guarantee, measured UB-peak reduction, isolated attribution to grouping, or a speedup for a baseline that cannot compile. Ordinary M1 outside the grouped domain is not a general UB-safe fallback.

### Same-run main-baseline ablation

The primary baseline is the unmodified main/Phase 0 operator, byte-identical at the
PR's local main base `be0b61da22778b043eb2369b5f922b2a39ddc46d` (source SHA-256
`ba3c7e740c3a2a9c8850733d3be77fefac79bc3ed02a45b4be076b58d6f7632d`).
It constructs the rotation tensors and full-width cos/sin temporaries removed by
direct half RoPE. The candidate measured here is PR source commit `5b09d01c`
(wrapper SHA-256 `6089f901fb35ddcbe6a6af698384029a43a2d30451f14bd9a40f4f0e6514eea1`).
This historical table is retained for attribution; it does not measure the final single-token grouped source.

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

- **BASE**: frozen main/phase0 kernel, single row, backend-default multibuffer.
- **M1**: historical WP1 candidate `5b09d01c`, direct-half kernel, explicit single row, backend default.
- **M2/off**: historical WP1 candidate `5b09d01c`, forced M2 only inside the experiment by bypassing
  G2/G3, not G1; `multibuffer=False`, `num_stages=1`, `num_warps=32`.
- **PR auto**: that historical WP1 wrapper with BLOCK/UB overrides unset, selecting M1 for
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

### Integrated single-token route

The performance tables describe the integrated implementation tested as frozen source SHA-256 `c81f8302f733311380bc13ed2d28285f1525d6ff526d3fe33d34bd6ce9b9d6dd` in the same B3/P40 rc1 environment. A subsequent cleanup removes the publicly unreachable M2 head-boundary branch and requires complete Q-head tiles for internal M2 calls. Public routing and retained math are unchanged.

A compile-only comparison of that frozen source and the cleaned source SHA-256 `64fccc08a4a2603ad028e6aab365e61a56056b6a60f299899b0480b8e8b1b2f6` covers one B4 specialization: HQ24/T1024, M2 with `multibuffer=False`, `num_stages=1` and `num_warps=32`. Both generated byte-identical 16,296-byte npubin objects (SHA-256 `e63a5a6a01e6821ff4ce466d885e9c587b1ced96ab1947df5c4813b2190d0209`); TTIR/TTAdapter differ only in operator source debug locations, and metadata differs only in the cache hash. This establishes object correspondence for that specialization with the same ABI and launch configuration. It adds no kernel execution, B3 numerical or performance measurement, other-route qualification, or proof of UB safety. The tables retain their frozen-source attribution.

The table below reuses the historical BASE medians above and the integrated candidate's medians from later runs. Speedup is `historical BASE median / integrated median`, not a same-run paired estimate. Inputs, scalar semantics, environment and timing settings are comparable, but inter-run variation remains possible; P1's small difference is not evidence of a clear gain. Synthetic model-derived geometries are not captured model execution.

| Case | Historical BASE (us) | Integrated (us) | Cross-run speedup | Actual route |
| --- | ---: | ---: | ---: | --- |
| P1 | 14.230 | 13.940 | 1.021x | Ordinary M1 |
| P2 | 20.340 | 18.200 | 1.118x | Ordinary M1 |
| P3 | 6.520 | 3.550 | 1.837x | T1 grouped |
| P4 | 193.330 | 155.700 | 1.242x | M2/off |
| P5 | 360.450 | 283.120 | 1.273x | M2/off |

The integrated T1 measurements in `run_20261016T063356025991Z_all_de63e61f` used six independent pairs against frozen WP1 (`6089f901...`), not BASE. Median pair ratios were 1.738x for 4/1 heads, 1.642x for 6/1 and 1.148x for 24/4. A corresponding historical BASE measurement has not been verified for T1 4/1 or 24/4; those BASE speedups are intentionally omitted rather than borrowed from another geometry. These ratios must not be multiplied with older ratios and presented as direct measurements.

The seven-point protection run `run_20261016T111952768021Z_all_fe4f200f` covered P1/P2/P4/P5, T4, Hq24/T1024's M2 odd tail and Hq16's resource fallback. All numerical and profile-identity checks passed. P2 included one valid 0.943x pair below the prespecified 0.95 minimum, retained as a negative observation. A separate full six-pair P2 rerun, `run_20261016T233003405339Z_all_0fa1269c`, passed with median ratio 0.997732x, minimum 0.987991x and all six ratios at least 0.98. No old samples were replaced or pooled; the displayed P2 median above remains from the original protection run.

The three benefit points plus seven protection points close the measured ten-point engineering gate only. The received evidence is report-level: target-side committed/checksum checks passed, but the full raw result trees were not independently rehashed locally. The measurements are target-kernel duration, not Python public-call latency or end-to-end model throughput. P48/A3, all parameter combinations and matching-main Nightly/model/graph tests remain outside this bounded validation.
