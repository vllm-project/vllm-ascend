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

## Startup and execution flow

1. **Initialize and register.** The runner activates its `jit_warmup_registry` while constructing selected components. Their `register_warmup(...)` calls record the wrapper and configuration; registration itself does not compile kernels.
2. **Run configured dummy inputs.** The [worker](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/worker.py) runs the configured model dummy inputs. These executions can already trigger Triton compilation.
3. **Complete registration.** The worker calls [kernel_warmup(worker)](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/model_executor/warmup/kernel_warmup.py). It adds registrations that need final runtime metadata and removes wrappers that cannot compile through the Ascend base.
4. **Compile required variants.** `registry.warmup()` expands registered configurations through `get_warmup_keys`, deduplicates keys per wrapper and compiles each required variant. For a non-autotuned wrapper, the path is `compile → warmup_inputs → __call__ → launch → kernel.warmup`; it compiles without executing the kernel.
5. **Capture and serve.** The worker captures graphs when enabled. Runtime wrapper calls use real tensors and the actual grid; Triton reuses matching cached artifacts. A variant not covered by warmup can still compile on first use.

`CompileKey` deduplicates warmup work; it does not replace Triton's runtime specialization and cache keys.

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

Place `register_warmup(...)` in the initialization path of the component that selects and uses the operator. Pass the configuration expected by the wrapper's `get_warmup_keys`.

| Operator owner | Where to register or integrate | Source reference |
| --- | --- | --- |
| Model | Register in the model layer's `__init__`, while the runner constructs the model inside its registry window. | `DeepseekV41Indexer.__init__` registers indexer kernels in [indexer.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/deepseek_v41/indexer.py). |
| Attention | Register in the selected metadata builder's `__init__`. If keys need allocated cache layouts, register after allocation in the runner's cache initialization window. | `AscendDSAMetadataBuilder.__init__` in [dsa_v1.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/dsa_v1.py); `initialize_kv_cache` in the [runner](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/model_runner.py). |
| Sampling | Register when the selected sampling stack is initialized. Keep registrations specific to that stack. | The [v1 runner](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v1/model_runner.py) calls `register_v1_sampling_warmup(vllm_config)` inside its registry window. |
| Worker / runner | Own registry windows and compilation timing. For keys requiring final component or input-batch metadata, call a late-registration helper before `registry.warmup()`. | `_register_dspark_swa_indices_warmup` and `kernel_warmup` in [kernel_warmup.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/model_executor/warmup/kernel_warmup.py). |
| Compilation pass | If a pass introduces a wrapped Triton operator, provide its key domain through an initialization or late-registration path above, once its configuration is known. A graph rewrite alone does not register the wrapper. | Follow the wrapper and registration steps in this guide; the current Ascend compilation directory has no dedicated `register_warmup` entry point. |
| Runtime | Call the same wrapper instance with real tensors. Register variants before serving; a runtime call alone does not populate the warmup registry. | `triton_q_rms` in [rms_norm.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/rms_norm.py). |

For the Q-RMS path, the runner's `_initialize_kv_cache_upstream` activates the registry around the upstream cache initialization that constructs attention metadata builders. The builder then registers the same wrapper instance used by `triton_q_rms`.

For a new initialization path, check that the caller encloses component construction in `with runner.jit_warmup_registry.activate():`. If it does not, add the window at the runner call site using its existing registry. A late-registration helper must also activate that same registry. Registration outside a window is a no-op; importing a wrapper alone does not register it.

Ordinary component registration does not require adding the operator to a central list or calling `registry.warmup()` from the component. The worker owns compilation timing.

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
