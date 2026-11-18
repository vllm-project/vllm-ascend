# Gated LayerNorm / RMSNorm

## Description

- **Function**: Normalize each row and group, apply per-channel weight and optional bias, and optionally gate with `silu(z)` (activation `swish` or `silu`) or `sigmoid(z)` before or after normalization. Entry: `layer_norm_fwd_npu` in `vllm_ascend/ops/triton/layernorm_gated.py`.
- **Formula** (per group, computed in fp32):
    - LayerNorm: `mean = sum(x) / N_group`, `rstd = rsqrt(sum((x - mean)^2) / N_group + eps)`, `y = (x - mean) * rstd * weight + bias`.
    - RMSNorm: `rstd = rsqrt(sum(x^2) / N_group + eps)`, `y = x * rstd * weight + bias`; bias is omitted when absent.
    - With `z`, let `gate(z) = z * sigmoid(z)` for `swish`/`silu` (default `swish`), or `gate(z) = sigmoid(z)` for `sigmoid`. Apply `x *= gate(z)` before normalization when `norm_before_gate=False`, or `y *= gate(z)` after the affine transform when true.
- **Algorithm flow** (rows and groups are independent):
  1. Validate the two-dimensional input layout, allocate output/statistics, and select a BASE row tile or HOIST32 execution.
  2. BASE processes BM16 or BM64 row tiles. HOIST32 uses BM32 tiles in a grid-stride loop with the grid capped at the vector-core count; weight and optional bias are loaded before the loop.
  3. Normalize and apply the affine transform and optional gate in fp32, then store masked outputs and group-major statistics.
- **Supported modes**: Ascend NPU inference with LayerNorm or RMSNorm, optional bias, and pre/post gate. The single-card test retains the upstream Atlas A2/A3 device guard. Compatibility on other applicable products (including 950PR/950DT) remains unvalidated; the runtime selector introduces no product-name or dtype allowlist. Historical single-operator device evidence is limited to Ascend910B4 and does not qualify the current merged source.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | Activations `[M,N]` | fp16 / bf16 / fp32 | 2D ND |
| `weight` | Input | Per-channel scale `[N]` | fp16 / bf16 / fp32 | 1D ND |
| `bias` | Optional input | Per-channel bias `[N]`, or `None` | fp16 / bf16 / fp32 | 1D ND |
| `eps` | Attribute | Stability constant added before reciprocal square root | float | scalar |
| `z` | Optional input | Gate activations `[M,N]`, or `None` | fp16 / bf16 / fp32 | 2D ND |
| `out` | Optional input/output | Caller-provided output, otherwise allocated by the wrapper | same as `x` | 2D ND |
| `group_size` | Optional attribute | Per-group width `N_group`; `None` defaults to `N` | int | scalar |
| `norm_before_gate` | Attribute | Gate after normalization when true, before when false; default true | bool | scalar |
| `is_rms_norm` | Attribute | RMSNorm when true, LayerNorm when false; default false | bool | scalar |
| `activation` | Optional attribute | `swish` (default), `silu` (alias), or `sigmoid`; unsupported names raise `ValueError`, also when `z=None` | str | scalar |
| return value | Output | `(out, mean, rstd)`; `mean=None` for RMSNorm | output dtype / fp32 statistics | `[M,N]`, `[G*M]` |

## Constraints

- `M` and `N_group` must be positive, and `N` divisible by `group_size`. `weight` and optional `bias` have shape `[N]`; optional `z` and `out` match `x`. All require contiguous last dimensions.
- `G=N/N_group` counts normalization groups within each row. For GDN output normalization, `[T,H_local,D_v]` is flattened to `[M,N]` with `M=T*H_local` and `N=D_v`: each row is one token/value-head pair, and model heads do not become normalization groups. `group_size=None` gives `G=1`.
- Statistics use group-major indexing `group*M+row`. Caller-provided `out` is reused and overwritten; input `x`, `weight`, `bias`, and `z` are otherwise read only. Allocate a separate `out`; overlapping output/input storage is not qualified. No caller-visible layout or dtype conversion is introduced by this wrapper. Device, dtype compatibility and non-overlapping storage remain caller preconditions inherited from the upstream wrapper.
- Multi-group inputs retain BASE64. Single-group NPU inputs below width 128 use BASE16; width 128 uses HOIST32 when `4*ceil(M/32)>=P`, otherwise BASE16, where `P` is the initialized vector-core count. Widths 129–512 use BASE16 when the UB getter reports at least 192 KiB; other inputs retain BASE64.
- Single-group NPU calls require device properties initialized through the existing `init_device_properties_triton()` contract. The UB getter may use its existing compatibility default or debugging override. Grouped calls do not query these properties.
- The single-group BASE16 resource envelope ends at `N_group=512` (`BLOCK_N<=512`). Wider groups retain BASE64 and may still overflow UB; this change does not make them resource-safe. The limit applies to per-group width, not total `N`. BASE's existing `65536 / element_size` feature-width guard is separate from the compiler's full-tile UB limit.
- Forward inference only; positive finite `eps` and finite floating inputs are the intended domain. There is no backward implementation in this module. Graph replay and full-model qualification are not established by the recorded single-operator tests.

## Origin and Differences

- **Origin**: The existing vLLM-Ascend gated normalization implementation is adapted from Flash Linear Attention's gated LayerNorm and the Triton LayerNorm tutorial; see the source-file header.
- **Differences**:
    - NPU adaptation for performance: selects smaller BASE row tiles for eligible single-group inputs and uses persistent execution with parameter reuse for suitable N128 workloads.
    - The merged public signature, activation validation and BASE kernel match main. HOIST implements both upstream activation IDs without changing the default swish arithmetic, normalization, output aliasing or statistics layout. Grouped inputs keep the original BASE64 launch. This PR does not add adaptive tiling, wider HOIST routes or new resource getters.

## Test Cases

The single-card test compares output and group-major statistics against a CPU fp32 reference. It covers LayerNorm/RMSNorm, optional bias/gate, all activation names, pre/post gating, grouping, caller-provided `out`, BASE16/64, the HOIST32 dispatch boundary, N192/N512 resource-envelope points, and flattened GDN shapes including M20449/N128. Synthetic boundaries supplement model-shape coverage; they do not establish all reachable model configurations.

For every compared tensor, both finite-value checks and both contribution-guide accuracy conditions must pass: at least 99% of elements satisfy `abs(actual-reference) <= atol + rtol*abs(reference)`, and the maximum absolute error does not exceed the bound below.

| Compared tensor dtype | rtol | atol | Maximum absolute error |
| --- | --- | --- | --- |
| fp16 | 2^-9 | 2^-9 | 0.1 |
| bf16 | 2^-6 | 2^-6 | 1 |
| fp32 | 2^-10 | 2^-16 | 0.01 |

Output uses its own dtype; `mean` and `rstd` always use the fp32 row of this table, independently of input dtype. These are the current test criteria, not retroactive reinterpretations of historical measurements.

```bash
python3 -m unittest discover -s tests/ut/ops -p 'test_layernorm_gated_*.py' -v
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_layernorm_gated.py
```

The host suite exercises scalar dispatch, wrapper launch arguments and the actual pre/post gate branch AST for each kernel. It does not compile Triton or emulate its vector/reduction execution. The module-level A2/A3 skip guard covers every device test, and the autouse initialization fixture also skips unsupported devices before initializing properties.

## Production integration and initialization

`register_ascend_customop` in `vllm_ascend/utils.py` maps vLLM's `RMSNormGated` to `AscendRMSNormGated` in `vllm_ascend/ops/layernorm.py`. Its `forward_oot` passes the activation through `LayerNormFn.forward`, which flattens leading dimensions, ensures contiguous last dimensions/parameters, calls `layer_norm_fwd_npu`, and restores the original output shape. This is a shared model-layer operator, used by gated delta-network output normalization (Qwen3-Next/GDN); it is not a runner-only kernel. FLA's separately registered `FusedRMSNormGated` is a different path and is not covered by this optimization.

The production worker calls the existing `init_device_properties_triton()` during device initialization (`vllm_ascend/worker/worker.py`) before runtime use. Direct single-group NPU wrapper callers must honor the same initialization contract. The wrapper does not initialize devices, synchronize, perform host/device copies, mutate global configuration or register warmup variants. It allocates the output when absent and fp32 statistics on the input device. Callers use the public wrapper rather than launching either private kernel.

## JIT specialization and recompilation

White-box analysis of the merged source:

| Argument/configuration | Domain and specialization behavior |
| --- | --- |
| `x`, `out`, `weight`, optional `bias`/`z`, statistics pointers | Floating pointer dtype and alignment classes can specialize. Optional `None` versus Tensor also changes signatures; heuristics derive `HAS_BIAS` and `HAS_Z`. Contents and full pointer addresses are not keys. |
| Runtime row strides, `M`, per-group `N`, `eps` | Existing `do_not_specialize` annotations are preserved in both kernels. Runtime type differences still matter; explicit constexpr values are unaffected. Shape changes can still change tiling and selected kernel. |
| HOIST `num_m_blocks` | Runtime loop bound; remains in `do_not_specialize`. Changing the grid alone is not a new specialization. |
| `BLOCK_M`, `BLOCK_N` | Compile-time tile extents required for `tl.arange` and layout generation; values change with the existing selector and power-of-two feature width. |
| `NORM_BEFORE_GATE`, `IS_RMS_NORM` | Compile-time control flow removes unused normalization/gating operations. |
| `ACTIVATION` | Compile-time branch ID: 0 for both `swish` and `silu`, 1 for `sigmoid`. The alias shares an ID; sigmoid introduces a necessary math variant in both kernels. Validation still applies when no gate is supplied. |
| Compiler/target/source/cache | Effective compiler options, backend/target, source and cache state can require fresh compilation. No autotuning or new compiler option is introduced. |

The bounded variant space is justified by algorithm/tile differences; this is a source analysis, not an observed compilation count. No specialization annotation is changed as part of conflict resolution. The persistent kernel's runtime control flow and activation variants still require compilation, accuracy and performance validation on the supported backend.

## Validation status and remaining evidence

- **Host validation**: stdlib dispatch/wrapper/gate-branch tests are available and executed for this local merge. They check host contracts only.
- **Current merged-source operator compilation and NPU accuracy**: NOT_RUN, including sigmoid/silu propagation through HOIST, all output dtypes, fp32 statistics under the new bounds, and A2/A3 coverage. Historical default-swish single-operator results and older tolerances do not close these gaps.
- **Current merged-source operator performance**: NOT_RUN. Recompare with matching baseline/source, device, toolchain, shape/dtype/flags and timing protocol. Historical route gains do not establish a sigmoid gain or current-tip performance.
- **Model accuracy**: NOT_RUN. Record a baseline/candidate evaluation such as GPQA, model/revision, dtype, parallelism, dataset settings and score comparison for actual callers.
- **Model performance**: NOT_RUN. Record a matched workload such as GSM8K, latency/throughput, versions, devices, model configuration and measurement method. No model-level speedup is claimed.
- **Black-box specialization and compilation counts**: NOT_RUN. Capture reachable model shapes, scalar/optional/activation metadata, startup compile counts and serving compile counts separately; distinguish cache hits, recompilation and initial compilation.
- **Warmup**: no registration added or claimed. Warmup execution/reuse validation is NOT_RUN; registration-window and runtime-coverage tests are N/A to this unchanged integration. If required variants are registered later, validate them through the same public wrapper and active component registry window after measuring specialization.
- **Graph replay and other applicable hardware**: NOT_RUN. Functional compatibility and resource safety beyond the documented envelope remain open validation items.

These pending device/model gates prevent treating the local host merge as complete Triton contribution qualification.
