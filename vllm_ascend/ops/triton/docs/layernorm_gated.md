# Gated LayerNorm / RMSNorm

## Description

- **Function**: Normalize each row and group, apply per-channel weight and optional bias, and optionally gate with `silu(z)` before or after normalization.
- **Formula**: For a group of width `N_group`, LayerNorm uses `mean = sum(x) / N_group`, `rstd = rsqrt(sum((x - mean)^2) / N_group + eps)` and `y = (x - mean) * rstd * weight + bias`. RMSNorm omits the mean and uses `rstd = rsqrt(sum(x^2) / N_group + eps)`. If `z` is present, `x *= silu(z)` before normalization when `norm_before_gate=False`; otherwise `y *= silu(z)` after the affine transform.
- **Algorithm flow** (rows and groups are independent):
  1. The wrapper validates the input layout, allocates `out`, `mean` (LayerNorm only), and `rstd`, and obtains the initialized vector-core count for single-group NPU calls. For their qualified wide per-group domain, it also reads UB through the existing initialized-properties getter. Multi-group calls retain the baseline launch without these property queries.
  2. A scalar selector chooses a BASE row tile or, for a qualified single group, a HOIST32 M-axis launch. BASE uses the existing two-dimensional `(row tiles, groups)` grid with `BLOCK_M=16` or `64`. All multi-group inputs retain BASE64.
  3. HOIST32 caps its one-dimensional grid at the vector-core count, walks M-axis tiles with a grid-stride loop, and loads the single group's weight and optional bias before that loop.
  4. Each tile computes normalization in fp32, applies the affine transform and optional gate, and stores masked outputs. Statistics are stored group-major as `[group, row]`.
- **Supported modes**: Inference-time LayerNorm and RMSNorm, with optional bias and pre/post gate. The public signature, calling convention, output aliasing, and normalization/gating semantics are unchanged.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | Activations `[M, N]` | Floating point | 2D ND; contiguous last dimension |
| `weight` | Input | Per-channel scale `[N]` | Floating point | 1D contiguous |
| `bias` | Optional input | Per-channel bias `[N]` | Floating point | 1D contiguous, or `None` |
| `eps` | Attribute | Stability constant added before reciprocal square root | Float | Scalar |
| `z` | Optional input | Gate activations `[M, N]` | Floating point | 2D ND with contiguous last dimension, or `None` |
| `out` | Optional input/output | Caller-provided output, or allocated by the wrapper | Same as `x` | `[M, N]` with contiguous last dimension |
| `group_size` | Optional attribute | Per-group width `N_group`; defaults to `N` | Integer | Scalar dividing `N` |
| `norm_before_gate` | Attribute | Apply the gate after normalization when true, before when false | Boolean | Scalar |
| `is_rms_norm` | Attribute | Select RMSNorm rather than mean-subtracting LayerNorm | Boolean | Scalar |
| return value | Output | `(out, mean, rstd)`; `mean=None` for RMSNorm | Output dtype / fp32 statistics | `[M, N]`, `[ngroups * M]` |

## Constraints

- `x` is two-dimensional `[M,N]`, with positive `M` and `N_group`. `N` must be divisible by `group_size`; optional `z` and `out` match `x`, and `weight` and optional `bias` have shape `[N]`. These tensors require contiguous last dimensions. The wrapper checks shapes and strides before dispatch.
- `group_size=None` means `N_group=N` and `G=1`. Tiling uses the per-group width, not total `N`: `[M,384]` with `group_size=128` has `G=3`.
- LayerNorm returns fp32 `mean` and `rstd` in group-major order, with flattened index `group*M+row`; RMSNorm returns `mean=None`. The caller-provided `out` is reused when supplied.
- BASE retains the existing `65536 / element_size` feature-width guard. This input-width guard is separate from the compiler's UB requirements.

## Model input shapes

vLLM's [Qwen3.5 linear-attention layers](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/qwen3_5.py#L144-L151) use Qwen GDN. Its [output norm](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L487-L494) uses `group_size=None`, `norm_before_gate=True`, and RMSNorm with a gate and no bias. On Ascend, the [output-projection path](https://github.com/c5566b/vllm-ascend/blob/1097a8d766ec8abb4e3bfb9fba72b10bbe0beeb2/vllm_ascend/ops/gdn.py#L333-L341) reshapes both the attention output and its gate from `[T, H_local, D_v]` to `[M, N]`. The [Ascend gated-norm wrapper](https://github.com/c5566b/vllm-ascend/blob/1097a8d766ec8abb4e3bfb9fba72b10bbe0beeb2/vllm_ascend/ops/layernorm.py#L125-L147) then calls `layer_norm_fwd_npu`. Here:

- `T` is the number of token-activation rows passed to this projection on the current rank, including any rows present in its input tensor. It is not a fixed request batch size or the model's maximum sequence length.
- `H_local` is its local value-head count (`num_v_heads / TP` in this path).
- `M = T * H_local`, and `N = D_v`, the value-head width.
- `group_size=None` gives `N_group=N` and `G=1`: model heads have been folded into the row dimension, rather than becoming normalization groups.

| Model | Value-head width `D_v` | Value heads at TP1 `H_local` | Flattened row count at TP1 | GDN norm epsilon |
| --- | ---: | ---: | --- | --- |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B/blob/af65380a20d418eeb0a2bcb784dd43e9b76c4a2e/config.json) | 128 | 48 | `M=48*T` | `1e-6` |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/41adbc1e50345066ac5153217957e9910204e632/config.json) | 128 | 32 | `M=32*T` | `1e-6` |

Each flattened row is one `(token, local value head)` pair: `row=t*H_local+h`. The normalization weight (and optional bias) has width `D_v` and is reused across these rows. `G=N/N_group` counts groups within a row; it is not the number of model value heads.

For example, four token rows at TP1 give `M=192` for Qwen3.5-27B and `M=128` for Qwen3.5-35B-A3B. At TP1, the Qwen3.5-35B-A3B head count also maps 2048 token rows to `M=65536`. These are shape mappings, not captured model benchmarks. Both Qwen configurations use `rms_norm_eps=1e-6`. Operator performance cases are stated in flattened `[M,N]` coordinates; boundary and tail controls need not map to an integer token count for a particular model. They are single-operator measurements, not model-throughput measurements.

[OLMo-Hybrid-7B](https://huggingface.co/allenai/Olmo-Hybrid-7B/blob/712e06263b8656a2ea6ec2c395106f19bba2f50f/config.json) has 30 value heads of width 192. Its [GDN output norm](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py#L149-L157) also uses single-group RMSNorm with `norm_before_gate=True`, and its [output-projection reshape](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py#L279-L288) gives `M=30*T` at TP1 and `N=N_group=192`. Two token rows therefore map to `[60,192]`, motivating the resource control below. The pinned GDN implementation sets its output-norm epsilon to `1e-5`; the resource controls use `1e-6` and test both gate orders. They exercise this width and the operator's supported semantics, rather than reproducing all OLMo call parameters.

## Execution paths and dispatch

The existing BASE kernel is launched with BM16 (BASE16) or BM64 (BASE64). HOIST32 uses BM32, caps its grid at the initialized vector-core count, and processes row blocks in a grid-stride loop. Loading weight and optional bias outside that loop reuses them when a program processes multiple blocks.

| Input condition | Route | Row block | Grid |
| --- | --- | ---: | --- |
| `G>1`, or non-NPU input | BASE64 | 64 | `(ceil(M/64), G)` |
| NPU, `G=1`, `N_group<128` | BASE16 | 16 | `(ceil(M/16), 1)` |
| NPU, `G=1`, `N_group=128`, `4*ceil(M/32)<P` | BASE16 | 16 | `(ceil(M/16), 1)` |
| NPU, `G=1`, `N_group=128`, `4*ceil(M/32)>=P` | HOIST32 | 32 | `(min(P, ceil(M/32)),)` |
| NPU, `G=1`, `129<=N_group<=512`, UB getter reports at least 192 KiB | BASE16 | 16 | `(ceil(M/16), 1)` |
| Remaining inputs | BASE64 | 64 | `(ceil(M/64), G)` |

`P` is supplied by the initialized device properties; it is not a hardcoded core count. The N128 boundary is M=289 at P=40 and M=353 at P=48. This tile-count rule is supported by the sampled cases, rather than a precisely measured universal crossover. Route selection uses shape and resource properties, not tensor values, and does not impose a device-name or dtype allowlist.

## Origin and Differences

The BASE implementation is adapted from Flash Linear Attention's gated LayerNorm and the Triton LayerNorm tutorial. This change reuses its mathematical body, selects smaller row tiles for eligible single-group inputs, and adds persistent execution with parameter reuse for N128. All grouped inputs retain the original BASE64 execution.

## Test Cases

- Host selector and wrapper-route tests check the integer quarter-wave tile boundary and nearby rows, grouped BASE64 across narrow and wide group widths without resource queries, the single-group BASE16 `N_group`/UB envelope (including N513), non-NPU fallbacks, and launch arguments without an NPU:

  ```bash
  python -m unittest discover -s tests/ut/ops -p 'test_layernorm_gated_*.py'
  ```

- The existing single-card operator test checks LayerNorm/RMSNorm, optional bias/gate, grouping, dtype tolerances, and `out` behavior against a CPU reference:

  ```bash
  pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_layernorm_gated.py
  ```

- Additional cases derive M from the initialized vector-core count to exercise BASE16 immediately below the threshold, HOIST32 at and above the threshold, and HOIST32 at M=65536. They record the actual JIT launch through the public wrapper and compare `out`, `mean`, and `rstd` with the same CPU reference for limited BF16/FP16 RMSNorm/LayerNorm and gate combinations. These cases still require execution on a matching-main NPU environment before claiming in-tree NPU coverage of the new routes.

## On-device validation

### Qwen-aligned N128 route comparison

Six batches on Ascend910B4/P40, 192 KiB UB, and CANN 8.5 compare frozen BASE64 with the selected public route and the alternative route. All use N128/G1/BF16 RMSNorm, `z` present, no bias, `eps=1e-6`, and `norm_before_gate=True`, matching the Qwen GDN normalization parameters described above. M128 tests small M; M288/289 bracket dispatch; M639/M20449 check tails; M20449/M65536 cover larger workloads.

All 36 numerical checks passed (two seeds per route and shape), with outputs and applicable statistics compared against the CPU reference. The offline tolerances are `atol=rtol=0.03` for output and `0.005` for statistics. These are the recorded offline tolerances; the in-tree Nightly test defines its own dtype-specific tolerances.

| M | N_group | G | Dtype | Norm | z / bias | norm_before_gate | BASE64 (frozen baseline) | BASE16 (route 1) | HOIST32 (route 2) | Selected route |
| ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 128 | 128 | 1 | BF16 | RMSNorm | Yes / No | True (post-gate) | 8.310 / 1.000x | **5.590 / 1.487x** | 6.030 / 1.378x | **BASE16** |
| 288 | 128 | 1 | BF16 | RMSNorm | Yes / No | True (post-gate) | 7.670 / 1.000x | **6.430 / 1.193x** | 6.550 / 1.171x | **BASE16** |
| 289 | 128 | 1 | BF16 | RMSNorm | Yes / No | True (post-gate) | 7.440 / 1.000x | 6.600 / 1.127x | **6.620 / 1.124x** | **HOIST32** |
| 639 | 128 | 1 | BF16 | RMSNorm | Yes / No | True (post-gate) | 7.750 / 1.000x | 8.190 / 0.946x | **7.530 / 1.029x** | **HOIST32** |
| 20449 | 128 | 1 | BF16 | RMSNorm | Yes / No | True (post-gate) | 53.121 / 1.000x | 142.033 / 0.374x | **41.071 / 1.293x** | **HOIST32** |
| 65536 | 128 | 1 | BF16 | RMSNorm | Yes / No | True (post-gate) | 160.553 / 1.000x | 444.989 / 0.361x | **117.612 / 1.365x** | **HOIST32** |

Route cells show median device duration (us) and speedup relative to BASE64, with BASE64 normalized to `1.000x`. Bold cells mark the selected route. Timing uses `msprof op` target-kernel device-task duration, five warm-ups, one profiled launch, and three serial A-B-B-A blocks per comparison. The six batches contain 144 timing samples. Kernel route, row tile, grid, and mathematical flags were verified. Each shape has a BASE64/public-route comparison and an alternative/public-route comparison. The table uses the public-route median from the former and the alternative-route median from the latter; each speedup is the BASE64 duration median divided by that route's median. No acceleration ratios are multiplied or pooled across gate orders.

The selected routes improve over BASE64 at all six points. Near the boundary, the two candidate routes have similar timings; the threshold is not a universal fastest-route claim. At M20449 and M65536, HOIST32 improves over BASE64 by 1.293x and 1.365x and also substantially outperforms the BASE16 controls. These comparisons evaluate persistent execution and parameter reuse together, rather than isolating parameter hoisting.

### N192 resource control

M60/N192/G1/BF16 RMSNorm with `z`, no bias, and `eps=1e-6` was tested with the same baseline/candidate inputs for both gate orders in the same B4/P40 environment. Each uses two seeds.

| M | N_group | G | norm_before_gate | Upstream BASE64 (BM64/BN256) | Public PR1 BASE16 (BM16/BN256) |
| ---: | ---: | ---: | --- | --- | --- |
| 60 | 192 | 1 | True (post-gate) | Compile failed: 321 KiB required > 192 KiB available | Compiled; numerical checks passed |
| 60 | 192 | 1 | False (pre-gate) | Compile failed: 321 KiB required > 192 KiB available | Compiled; numerical checks passed |

BASE64 requires 321 KiB on the 192 KiB target; it fails compilation and has no numerical or performance result. BASE16 compiles and passes numerical checks for both orders. Independent fresh-process NPU health probes passed after every baseline compile failure. These measurements establish the resource repair for the tested inputs, not an N192 speedup.

### Evidence reuse

The native packages use PR1 runtime checkpoint `bdbdc07a9f268751bed01dff3a0b300b0005cb0f` and frozen upstream `5f8a1286a2d04d35b94ebc8a961057c48a7b82d2`. Subsequent grouped BASE64 fallback and BASE16 naming cleanup preserve these single-group kernel bodies and measured launch configurations. The revised wrapper as a whole has not been rerun on device. Earlier gate-before-normalization measurements remain separate historical evidence and are not pooled into the Qwen-aligned table. The earlier 24-case BF16/FP16 public-route regression and N192/256/384/512 single-group numerical controls also remain bounded evidence for their unchanged paths.

## Known limitations and remaining validation

- **Grouped inputs**: All `G>1` cases retain BASE64. No grouped performance improvement or grouped UB repair is claimed; the baseline's resource limitations remain.
- **Wide inputs**: HOIST32 is selected only at single-group N128. At N256/G1, experimental forced HOIST32 probes for M20449 and M2560 required 209.125 KiB UB on a 192 KiB target and failed compilation; the BASE16 numerical controls passed. There is no wide-HOIST timing or numerical result. These negatives do not show that every possible wide-N implementation is infeasible.
- **UB envelope**: Qualified single-group widths 129–512 use BASE16; `BLOCK_N` is 256 for widths 129–256 and 512 for widths 257–512. Missing or lower UB in the scalar selector, or widths above 512, retain BASE64. A BASE fallback does not guarantee compilation for arbitrary widths, dtypes, or mathematical modes.
- **Runtime properties**: Single-group NPU calls require the existing device-property initialization contract. An uninitialized vector-core getter raises; it does not silently select BASE64. Grouped calls do not query these properties. The UB getter may return its compatibility default or the existing debugging override, so its routing value is not an independent measurement of the compiler's resource use.
- **Performance scope**: The displayed gains are BF16 single-operator measurements on B4/P40 with the stated semantics. Host route tests and BF16/FP16 numerical checks do not establish speedup for every dtype, shape, device, or gate order. Simulator traces explain execution mechanisms but do not prove numerical correctness or device speedup.
- **Remaining validation**: Matching-main in-tree NPU CI/Nightly, full model execution, and graph replay remain unverified by these offline runs. The earlier M288 first-launch NaN did not recur in the final tests; its cause remains unresolved.
