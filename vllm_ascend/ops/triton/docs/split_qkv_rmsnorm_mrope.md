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
- For the default request, G1 checks the semantic/layout and compiler identity. G2 reuses the community `get_ub_size_bytes()` helper in `triton_utils.py`, as LayerNorm and Q-RMSNorm do, and a shape-derived *screening estimate*. The helper uses worker-initialized properties and its existing 192 KiB default / `VLLM_ASCEND_ROPE_UB_SIZE_KB` override. This capacity can be detected, defaulted, or overridden; it is **not an independently measured compiler allocator budget**. The wrapper does not import private backend UB fields or add another fallback. Helper errors/invalid values select M1, as does estimated demand at or above capacity. The estimate and its difference from capacity are **not measured UB usage or free headroom**.
- Until boundary liveness is calibrated, pair-capable M2 requires `num_q_heads` to be divisible by `min(num_q_heads, 12)`; a positive remainder selects M1. This structural guard follows an Hq16/D256 allocator overflow in a simulator compile probe. An Hq16/D128 boundary compiled in a separate simulator probe, so the guard is intentionally conservative rather than a claim that every boundary shape overflows on device.
- G3 selects M1 for tail-only work or when `floor(min_active_tokens_per_core / 2) < 12`; otherwise a G1/G2-cleared pair path selects M2. At `P=40`, `T=959` falls below this trial threshold and `T=960` reaches it. This is a workload policy supported by bounded tests, not a universal optimal crossover. A one-token call remains on M1.
- The M2 launch uses `PAIR_CAPABLE=True`, `multibuffer=False`, `num_stages=1`, and `num_warps=32`. An M1 fallback uses the original kernel/backend default. The empty evidence registry is diagnostic and does not gate runtime selection; unknown SoC identity alone does not force fallback. No route is claimed fastest for every supported input or hardware configuration.

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

- The shared UB helper refactor and M4 removal are covered by host-only checks, including invalid helper values, capacity-dependent selection, and rejection of the removed override. AST comparison preserves the M1/M2 computation and existing resource/workload rules. The archived device results below precede this capacity-source refactor; they do not constitute an on-device validation of the new helper connection.

- The pre-existing single-card accuracy suite covers bf16/fp16, gate/no-gate, interleaved/contiguous MRoPE, two token counts, and two head configurations. It was not run on the exact current PR head in this work:

  ```bash
  pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_mrope.py
  ```

## Bounded offline validation

- The primary performance baseline for this PR is the **unmodified main/Phase 0 operator**, compared with the complete candidate including direct half RoPE and adaptive M1/M2 routing. The speedup metric is `baseline median device duration / candidate median device duration`; above 1 favors the candidate. At main commit `0015065d`, the operator source is byte-identical to the frozen Phase 0 `v0.23.0rc1` baseline (SHA-256 `ba3c7e740c3a2a9c8850733d3be77fefac79bc3ed02a45b4be076b58d6f7632d`); it still constructs `cat_x`/`cat_y` and broadcasts cos/sin to the full RoPE width.

  | Case | Tokens | Q/KV heads | Phase 0 median duration (us) | Complete candidate median duration (us) | Historical baseline/candidate ratio |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | P5 | 8,192 | 4 / 1 | 362.069992 | 283.090012 | 1.2790x |

  This is a **descriptive cross-run ratio**, not a same-run paired result. The baseline comes from `run_20260903T030801685837Z_all_0ae57a2d`; the candidate is the measured earlier port source `fd671ea7` in `run_20261008T073722771387Z_all_9e4872ed`, not the exact current PR head. Both use Ascend910B3/P40, the same image ID (`sha256:13315b656180f24489bb0076ff449fd5dcc15f84f447872ce38bf7e43606489a`), the same defined P5 shape/modes and seed `20260812`, six timing samples per subject, and `msprof op` device duration with one profiled launch. However, the runs use different dates and zero versus five profiler warm-ups; physical-card identity and byte-identical input packs were not matched across runs. These differences can affect the ratio, so it is not a controlled estimate or a non-regression guarantee. Comparable B3 Phase 0 durations for T1024/P4 have not been identified; A3 or simulator timings are not mixed into this table.
- On one Ascend910B3/P40 device in the `v0.23.0rc1` image, parent commit `c1469f99` completed `HQ24_T1024_B3` through the public entry (`run_20261009T022037682885Z_numerics_20830d57`). With the override unset, the wrapper selected M2 pair-capable and produced a compiled object. Q/K passed `atol=rtol=0.02` against the frozen CPU reference; V/Gate were byte-exact. This one-item run has no timing result. The later `de2b1605` commit changes only the policy's approved `regex` import and a docstring spelling; its exact policy bytes were not rerun on device.
- As a supplementary comparison, an earlier port source, `fd671ea7`, completed a seven-point B3 numerical matrix and the separate `run_20261008T073722771387Z_all_9e4872ed` paired long-workload run. The latter used frozen M1/on **already containing direct half RoPE** as A (source SHA-256 `ad28c9afec42629ee21ed9ddf50c6665cd46e94533b5b80a8470580b6c04fc0d`) and the ported public wrapper as B, with the same inputs, six valid pairs per point (18/18 total), three ABBA/BAAB/ABBA blocks, five warm-ups, and one profiled target-kernel launch. This comparison measures the additional benefit of the integrated paired path over the optimized M1 control; it does not measure the total improvement over main/Phase 0. The metric below is the median of individual A/B device-duration ratios; above 1 favors B.

  All three sampled inputs are bf16, `head_size=256`, `rope_dim=64`, `mrope_section=(11, 11, 10)`, with gate enabled, interleaved Q/gate layout, interleaved MRoPE, and no Q/K bias. They use 40 active vector cores. T1024 and P4 use six local Q heads and one KV head; P5 uses four local Q heads and one KV head. The frozen T1024/P4 B4 input definitions were overlaid to B3 by changing only the case ID and execution SoC; P5 originated as a B3 input.

  | Case | Tokens | Q/KV heads | Direct-half M1 / candidate median duration (us) | Valid pairs | Incremental paired A/B | Interpretation |
  | --- | ---: | ---: | ---: | ---: | ---: | --- |
  | T1024 | 1,024 | 6 / 1 | 59.210 / 56.670 | 6/6 | 1.0447x | `no_clear_change`; below the pre-registered 1.05x gain threshold |
  | P4 | 4,096 | 6 / 1 | 185.130 / 156.160 | 6/6 | 1.1861x | `faster` at this sampled shape |
  | P5 | 8,192 | 4 / 1 | 347.070 / 283.090 | 6/6 | 1.2269x | `faster` at this sampled shape |

- All six numerical items in that paired run passed; no failure or diagnostic was reported. These timings belong to the **earlier source and that B3 runtime**, not to the exact current PR head or a model-throughput claim. Their ratios must not be relabeled as main/Phase 0 speedups or multiplied by historical Phase 1 ratios to estimate a total gain. Earlier P1/P2/P3 results with no clear incremental M2 benefit likewise do not establish that the complete PR has no benefit over main: the current M1 fallback retains direct half RoPE. The full target result tree was not copied locally; the received compact handoff, target-side committed status, and mutation-free handoff form report-level evidence. The target-side original result remains the source for independent checksum review.
- The exact current PR head has host-only tests but no matching-main NPU Nightly, model/graph-replay test, or fresh paired performance run. The trial workload threshold and conservative boundary guard remain explicit qualification limits.
