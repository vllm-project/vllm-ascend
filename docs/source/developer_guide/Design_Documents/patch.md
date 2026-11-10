# Patch in vLLM Ascend

vLLM Ascend is a platform plugin for vLLM. Due to the different release cycle of vLLM and vLLM Ascend and their hardware limitations, we need to patch some code in vLLM to make it compatible with vLLM Ascend.

In vLLM Ascend code, we provide a patch module `vllm_ascend/patch` to adapt to changes in vLLM.

## Principle

We should keep in mind that Patch is not the best way to make vLLM Ascend compatible. It's just a temporary solution. The best way is to contribute the change to vLLM to make it compatible with vLLM Ascend initially. In vLLM Ascend, we have the basic principle for Patch strategy:

1. Less is more. Please do not patch unless it's the only way currently.
2. Once a patch is added, it's required to describe the future plan for removing the patch.
3. Anytime, cleaning the patch code is welcome.

## How it works

In `vllm_ascend/patch`, you can see the code structure as follows:

```shell
vllm_ascend/
└── patch/
    ├── platform/
    │   └── patch_xxx.py
    └── worker/
        └── patch_yyy.py
```

- **platform**: The patch code in this directory is for patching the code in vLLM Main process. It's called by `vllm_ascend/platform::NPUPlatform::pre_register_and_update` very early when vLLM is initialized.
    - For online mode, vLLM process calls the platform patch in `vllm/vllm/engine/arg_utils.py::AsyncEngineArgs.add_cli_args` when parsing the CLI args.
    - For offline mode, vLLM process calls the platform patch in `vllm/vllm/engine/arg_utils.py::EngineArgs.create_engine_config` when parsing the input parameters.
- **worker**: The patch code in this directory is for patching the code in vLLM worker process. It's called by `vllm_ascend/worker/worker::NPUWorker::__init__` when the vLLM Worker process is initialized.
    - For both online and offline mode, vLLM EngineCore process calls the worker patch in `vllm/vllm/worker/worker_base.py::WorkerWrapperBase.init_worker` when initializing the worker process.

## How to write a patch

Before writing a patch, following the principle above, we should patch the least code. If it's necessary, we can patch the code in either **platform** or **worker** folder. Here is an example to patch `distributed` module in vLLM.

1. Decide which version of vLLM we should patch. For example, after analysis, here we want to patch both `0.10.0` and `main` of vLLM.
2. Decide which process we should patch. For example, here `distributed` belongs to the vLLM main process, so we should patch `platform`.
3. Create the patch file in the right folder. The file should be named as `patch_{module_name}.py`. The example here is `vllm_ascend/patch/platform/patch_distributed.py`.
4. Write your patch code in the new file. Here is an example:

    ```python
    import vllm

    def patch_destroy_model_parallel():
        # your patch code
        ...

    vllm.distributed.parallel_state.destroy_model_parallel = patch_destroy_model_parallel
    ```

5. Import the patch file in `__init__.py`. In this example, add `import vllm_ascend.patch.platform.patch_distributed` into `vllm_ascend/patch/platform/__init__.py`.
6. Add the description of the patch in `vllm_ascend/patch/__init__.py`. The description format is as follows:

    ```python
    # ** File: <The patch file name> **
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   1. `<The target patch module in vLLM>`
    #    Why:
    #       <Describe the reason why we need to patch>
    #    How:
    #       <Describe the way to patch>
    #    Related PR (if no, explain why):
    #       <Add a link to the related PR in vLLM. If there is no related PR, explain why>
    #    Future Plan:
    #       <Describe the future plan to remove the patch>
    ```

7. Add the Unit Test and E2E Test. Any newly added code in vLLM Ascend should contain the Unit Test and E2E Test as well. You can find more details in [test guide](../contribution/testing.md)

## Limitations

1. In V1 Engine, vLLM starts three kinds of processes: Main process, EngineCore process and Worker process. Now vLLM Ascend can only patch the code in Main process and Worker process by default. If you want to patch the code running in EngineCore process, you should patch EngineCore process entirely during setup. Find the entire code in `vllm.v1.engine.core`. Please override `EngineCoreProc` and `DPEngineCoreProc` entirely.
2. If you are running edited vLLM code, the version of vLLM may be changed automatically. For example, if you run the edited vLLM based on v0.9.n, the version of vLLM may be changed to v0.9.nxxx. In this case, the patch for v0.9.n in vLLM Ascend would not work as expected, because vLLM Ascend can't distinguish the version of the vLLM you're using. In this case, you can set the environment variable `VLLM_VERSION` to specify the version of the vLLM you're using, and then the patch for that version (e.g., v0.9.n) should work.

## Gate Triton patches before importing them

Use `vllm_ascend.utils.supports_triton()` when selecting a Triton implementation.
It combines Python package availability with the hardware profile's
`TRITON_KERNELS` capability. Package discovery alone must not enable Triton on
310P, even if `triton` or `triton-ascend` is installed.

Keep Triton imports inside the capability branch. Apply the same rule before
worker patch registration, custom-op registration, and kernel warmup. The worker
bundle imports `patch_ascend_ops` before `patch_triton` on supported hardware
because the latter overrides the causal-convolution implementation.

Select a native fallback when one exists. If a requested feature has no fallback,
validate it before initializing the model runner or allocating device resources,
and include a supported alternative in the error. For example, a KDA state-copy
plan rejects hardware without Triton support before importing its compiled
launchers. Select a model supported by 310P.
Environment switches may select optimizations only within the hardware's
capabilities; they must not override an unsupported capability.

Cover patch order and combinations in `tests/ut/patch/worker/test_triton_gate.py`.
Test 310P with both an absent Triton package and a discoverable package, and test
supported hardware with Triton enabled and disabled. Use real hardware E2E runs
to verify the selected fallback; mocked capabilities do not establish kernel
correctness or performance.

Run the capability, patch-combination, warmup, and fallback tests together:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q \
  tests/ut/device/test_hardware_profile.py \
  tests/ut/device/test_pcp_kv_cache_gate.py \
  tests/ut/device/test_paged_mla_cache_read.py \
  tests/ut/patch/worker/test_triton_gate.py \
  tests/ut/model_executor/warmup/test_sampling_warmup_gate.py \
  tests/ut/ops/test_kda_state_copy_gate.py \
  tests/ut/models/test_triton_fallbacks.py \
  tests/ut/attention/test_sfa_dcp.py
```

SFA and paged MLA cache preparation must import the device-level PCP wrapper,
which selects the Triton implementation lazily. Its native path preserves
strided cache rows and packed FP8 bytes, and fills negative padding slots with
zeros. Importing the Triton PCP module directly bypasses the hardware gate.

For hardware validation, record the plugin and vLLM revisions, CANN, Torch,
TorchNPU, Triton versions, model revision, device, and complete engine settings.
On 310P, inspect the worker process as well as the parent: `supports_triton()`
must be false and no `vllm_ascend.ops.triton.*` implementation should be loaded.
Compare generated token IDs and log probabilities in eager and supported graph
modes. A text-only run of a vision model does not validate its vision path.

On supported hardware, verify that the selected operator actually invokes its
Triton implementation and compare its output with a native reference. Benchmark
the baseline and candidate with the same dependencies, model, inputs, engine
settings, and device. Warm up before collecting repeated operator timings,
throughput, and request latency. Keep compilation and model-loading time separate
from steady-state measurements; retain raw samples and report measurement noise
before judging a performance regression.
