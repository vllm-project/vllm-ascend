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
vllm_ascend
├── patch
│   ├── platform
│   │   ├── patch_xxx.py
│   ├── worker
│   │   ├── patch_yyy.py
└───────────
```

- **platform**: The patch code in this directory is for patching the code in vLLM main process. It's called by `vllm_ascend/platform::NPUPlatform::pre_register_and_update` very early when vLLM is initialized.
    - For online mode, vLLM process calls the platform patch in `vllm/vllm/engine/arg_utils.py::AsyncEngineArgs.add_cli_args` when parsing the cli args.
    - For offline mode, vLLM process calls the platform patch in `vllm/vllm/engine/arg_utils.py::EngineArgs.create_engine_config` when parsing the input parameters.
- **worker**: The patch code in this directory is for patching the code in vLLM worker process. It's called by `vllm_ascend/worker/worker::NPUWorker::__init__` when the vLLM worker process is initialized.
    - For both online and offline mode, vLLM engine core process calls the worker patch in `vllm/vllm/worker/worker_base.py::WorkerWrapperBase.init_worker` when initializing the worker process.

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
    #    How：
    #       <Describe the way to patch>
    #    Related PR (if no, explain why):
    #       <Add a link to the related PR in vLLM. If there is no related PR, explain why>
    #    Future Plan:
    #       <Describe the future plan to remove the patch>
    ```

7. Add the Unit Test and E2E Test. Any newly added code in vLLM Ascend should contain the Unit Test and E2E Test as well. You can find more details in [test guide](../contribution/testing.md)

## Limitations

1. In V1 Engine, vLLM starts three kinds of process: Main process, EngineCore process and Worker process. Now vLLM Ascend only can patch the code in Main process and Worker process by default. If you want to patch the code running in EngineCore process, you should patch EngineCore process entirely during setup. Find the entire code in `vllm.v1.engine.core`. Please override `EngineCoreProc` and `DPEngineCoreProc` entirely.
2. If you are running edited vLLM code, the version of vLLM may be changed automatically. For example, if you run the edited vLLM based on v0.9.n, the version of vLLM may be changed to v0.9.nxxx. In this case, the patch for v0.9.n in vLLM Ascend would not work as expected, because vLLM Ascend can't distinguish the version of the vLLM you're using. In this case, you can set the environment variable `VLLM_VERSION` to specify the version of the vLLM you're using, and then the patch for v0.10.0 should work.
