# Triton JIT warmup on Ascend

This guide describes how to register and precompile Triton kernels during vLLM-Ascend startup. Use it with the upstream [JIT Kernel Warmup guide](https://docs.vllm.ai/en/latest/contributing/jit_kernel_warmup/) and the [vLLM revision verified by the target branch](https://github.com/vllm-project/vllm-ascend/blob/main/.github/vllm-main-verified.commit).

## Integration mechanism

| Component | Responsibility |
| --- | --- |
| vLLM `JitWarmupRegistry` | Collect registrations, expand compile keys and deduplicate them per wrapper. |
| vLLM Triton helper | Reuse the launch specification for warmup compilation and runtime execution. |
| Ascend runner | Activate the registry during component initialization. |
| `AscendTritonJitKernel` | Adapt the parallel-compilation gate for CANN. |
| Ascend `kernel_warmup()` | Complete late registration, retain Ascend wrappers and invoke registry warmup. |

Register an `AscendTritonJitKernel` instance inside the runner's active window. Registration outside a window is a no-op; the current orchestrator removes instances that do not derive from the Ascend base.

The [worker](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/worker.py) executes configured model dummy runs, then [JIT warmup](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/model_executor/warmup/kernel_warmup.py), followed by graph capture when enabled. Dummy runs may already trigger compilation.

For a non-autotuned wrapper, warmup follows `compile → warmup_inputs → __call__ → launch → kernel.warmup` without executing the kernel. Runtime calls use real inputs and the actual grid. `CompileKey` deduplicates warmup work; Triton's cache determines runtime reuse.

## Integration steps

### 1. Define the warmup domain

List the supported dtypes, constexpr values, scalar specialization classes, layouts and compilation options. Generate keys for reachable variants, including applicable out-of-graph paths. See [JIT specialization and recompilation](triton_operators.md#jit-specialization-and-recompilation).

### 2. Implement the Ascend wrapper

Derive the wrapper from [AscendTritonJitKernel](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/ascend_jit_warmup.py). For an explicit-key wrapper, implement:

| Member | Requirement |
| --- | --- |
| `kernel` | Reference the JIT kernel. |
| `CompileKey` | A hashable key identifying a required variant. |
| `dispatch` | Map concrete dispatch inputs to a key. |
| `get_warmup_keys` | Generate keys from the supported configuration domain. |
| `warmup_inputs` | Supply every required `__call__` argument, including launch-only arguments. |
| `__call__` | Use `@kernel_launcher`; return the grid and launch arguments. |

Use `TritonWarmupTensor` for compile-only tensor metadata. Match runtime dtype, strides, alignment and scalar specialization. Keep operations that require tensor contents outside this path.

### 3. Connect runtime calls

Create one wrapper instance for registration and execution. In the public operator function, replace direct kernel launch with the wrapper call. Preserve output allocation, grid dimensions, argument binding and return values.

### 4. Register in the selected component

Register during model-layer or metadata-builder initialization inside the runner window. Ordinary integrations do not need a central operator list. If the key depends on final cache metadata, register after that metadata is available and before `registry.warmup()`, using the same runner registry.

## Example: Q-RMS

The existing [Q-RMS implementation](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/rms_norm.py) shows the complete wrapper and runtime call:

1. `TritonQRmsKernel.CompileKey` contains `dim`, `block_m` and `dtype`.
2. `get_warmup_keys` enumerates reachable power-of-two `block_m` values up to the UB-derived bound.
3. `warmup_inputs` supplies tensor metadata and all launcher arguments, including `num_vectorcore`.
4. `triton_q_rms` calculates the runtime tile, allocates the output and calls `_TRITON_Q_RMS_KERNEL`.

[AscendDSAMetadataBuilder](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/dsa_v1.py) registers that instance during construction:

```python
from vllm_ascend.ops.triton.rms_norm import _TRITON_Q_RMS_KERNEL

_TRITON_Q_RMS_KERNEL.register_warmup(vllm_config)
```

## Validation

- Verify expected keys for supported configurations and specialization boundaries.
- Verify registration inside the runner window and no registration outside it.
- Verify warmup argument binding and unchanged runtime output after wrapper migration.
- On the target device, test cold and warm caches; record compilation separately for dummy runs, explicit warmup, graph capture and serving.

Add tests under `tests/ut/model_executor/warmup/`. Refer to [test_jit_warmup_rms.py](https://github.com/vllm-project/vllm-ascend/blob/main/tests/ut/model_executor/warmup/test_jit_warmup_rms.py) for keys and registration, and [test_kernel_warmup.py](https://github.com/vllm-project/vllm-ascend/blob/main/tests/ut/model_executor/warmup/test_kernel_warmup.py) for orchestration. In a prepared test environment, run:

```bash
python -m pytest tests/ut/model_executor/warmup/test_jit_warmup_rms.py tests/ut/model_executor/warmup/test_kernel_warmup.py -q
```

Mock tests validate registration and scheduling; use device tests for compilation coverage and numerical correctness.

## Configuration and troubleshooting

| Setting | Behavior |
| --- | --- |
| `kernel_config.enable_jit_warmup` | Enables registry warmup; defaults to `True`. |
| `VLLM_TRITON_JIT_WARMUP_NUM_THREADS` | Defaults to `4`; set to `1` for serial compilation. |

Parallel compilation requires `AsyncCompileMode` and a non-autotuned wrapper. Autotuning may execute kernels and requires suitable real inputs. `enforce_eager` does not globally disable registry warmup.

| Symptom | Check |
| --- | --- |
| Empty registry | Component selection, active window and Ascend base class |
| Warmup `TypeError` | Missing or mismatched `__call__` arguments |
| Compilation during serving | Uncovered specialization classes or missing registration |
| Incorrect runtime output | Grid, argument binding, masks and output writes |
| Parallel compilation failure | Retry with one thread and check compiler compatibility |
