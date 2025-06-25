# Patch in vLLM Ascend

vLLM Ascend is a platform plugin for vLLM. Due to the release cycle of vLLM and vLLM Ascend is different, and the hardware limitation in some case, we need to patch some code in vLLM to make it compatible with vLLM Ascend.

In vLLM Ascend code, we provide a patch module `vllm_ascend/patch` to address the change for vLLM.

## Principle

We should keep in mind that Patch is not the best way to make vLLM Ascend compatible. It's just a temporary solution. The best way is to contribute the change to vLLM to make it compatible with vLLM Ascend originally. In vLLM Ascend, we have the basic principle for Patch strategy:

1. Less is more. Please do not patch unless it's the only way currently.
2. Once a patch is added, it's required to describe the future plan for removing the patch.
3. Anytime, clean the patch code is welcome.

## How it work

In `vllm_ascend/patch`, you can see the code structure as follows:

```
vllm_ascend
├── patch
│   ├── platform
│   │   ├── patch_0_9_1
│   │   ├── patch_common
│   │   ├── patch_main
│   ├── worker
│   │   ├── patch_0_9_1
│   │   ├── patch_common
│   │   ├── patch_main
└───────────
```

- **platform**: The patch code in this directory is for patching the code in vLLM main process. It's called by `vllm_ascend/platform::NPUPlatform::pre_register_and_update` very early when vLLM is initialized.
  - for online mode, vLLM process calls the platform patch here `vllm/vllm/engine/arg_utils.py::AsyncEngineArgs.add_cli_args` when parsing the cli args.
  - for offline mode, vLLM process calls the platform patch here `vllm/vllm/engine/arg_utils.py::EngineArgs.create_engine_config` when parsing the input parameters.
- **worker**: The patch code in this directory is for patching the code in vLLM worker process. It's called by `vllm_ascend/worker/worker_v1::NPUWorker::__init__` when the vLLM worker process is initialized.
  - for both online and offline mode, vLLM engine core process calls the worker patch here `vllm/vllm/worker/worker_base.py::WorkerWrapperBase.init_worker` when initializing the worker process.

In both **platform** and **worker** folder, there are several patch module. They are used for patching different version of vLLM.

- `patch_0_9_1`: This module is used for patching vLLM 0.9.1. The version is always the nearest version of vLLM. Once vLLM is released, we will drop this patch module and bump a new version. For example, `patch_0_9_2` is used for patching vLLM 0.9.2.
- `patch_main`: This module is used for patching the code in vLLM main branch.
- `patch_common`: This module is used for patching both vLLM 0.9.1 and vLLM main branch.

## How to write a patch

Before writing a patch, following the principle above, we should patch the least code. If it's necessary, we can patch the code in either **platform** and **worker** folder. Here is an example to patch `distributed` module in vLLM.

1. Decide which version of vLLM we should patch. For example, after analysis, here we want to patch both 0.9.1 and main of vLLM.
2. Decide which process we should patch. For example, here `distributed` belongs to the vLLM main process, so we should patch `platform`.
3. Create the patch file in the write folder. The file should be named as `patch_{module_name}.py`. The example here is `vllm_ascend/patch/platform/patch_common/patch_distributed.py`.
4. Write your patch code in the new file. Here is an example:
    ```python
    import vllm

    def patch_destroy_model_parallel():
        # your patch code
        ...

    vllm.distributed.parallel_state.destroy_model_parallel = patch_destroy_model_parallel
    ```
5. Import the patch file in `__init__.py`. In this example, add `import vllm_ascend.patch.platform.patch_common.patch_distributed` into `vllm_ascend/patch/platform/patch_common/__init__.py`.
6. Add the description of the patch in `vllm_ascend/patch/__init__.py`. The description format is as follows:
    ```
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
7. Add the Unit Test and E2E Test. Any new added code in vLLM Ascend should contain the Unit Test and E2E Test as well. You can find more detail in [test guide](../contribution/testing.md)


## Limitation
1. In V1 Engine, vLLM start three kinds for process: Main process, EngineCore process and Worker process. Now vLLM Ascend only support patch the code in Main process and Worker process by default. If you want to patch the code runs in EngineCore process, you should patch EngineCore process totally during setup, the entry code is here `vllm.v1.engine.core`. Please override `EngineCoreProc` and `DPEngineCoreProc` totally.
2. If you are running an edited vLLM code, the version of the vLLM may be changed automatically. For example, if you runs an edited vLLM basing on v0.9.1, the version of vLLM may be change to v0.9.2xxx, in this case, the patch for v0.9.1 in vLLM Ascend would not work as expect, because that vLLM Ascend can't distinguish the version of vLLM you're using. In this case, you can set the environment variable `VLLM_VERSION` to specify the version of vLLM you're using, then the patch for v0.9.1 should work.
