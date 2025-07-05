# Environment Variables

vllm-ascend uses the following environment variables to configure the system:

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| ASCEND_HOME_PATH | string | None | The home path for CANN toolkit. If not set, the default value is `/usr/local/Ascend/ascend-toolkit/latest`. |
| CMAKE_BUILD_TYPE | string | None | The build type of the package. It can be one of the following values: Release, Debug, RelWithDebugInfo. If not set, the default value is Release. |
| COMPILE_CUSTOM_KERNELS | bool | 1 | Whether to compile custom kernels. If not set, the default value is True. If set to False, the custom kernels will not be compiled. Please note that the sleep mode feature will be disabled as well if custom kernels are not compiled. |
| CXX_COMPILER | string | None | The CXX compiler used for compiling the package. If not set, the default value is None, which means the system default CXX compiler will be used. |
| C_COMPILER | string | None | The C compiler used for compiling the package. If not set, the default value is None, which means the system default C compiler will be used. |
| DECODE_DEVICE_ID | string | None | The decode device id for disaggregated prefilling case. |
| HCCL_SO_PATH | string | None | The path for HCCL library, it's used by pyhccl communicator backend. If not set, the default value is `libhccl.so`. |
| HCCN_PATH | string | /usr/local/Ascend/driver/tools/hccn_tool | The path for HCCN Tool, the tool will be called by disaggregated prefilling case. |
| LLMDATADIST_COMM_PORT | int | 26000 | The port number for llmdatadist communication. If not set, the default value is 26000. |
| LLMDATADIST_SYNC_CACHE_WAIT_TIME | int | 5000 | The wait time for llmdatadist sync cache. If not set, the default value is 5000ms. |
| MAX_JOBS | int | None | Max compile thread number for package building. Usually, it is set to the number of CPU cores. If not set, the default value is None, which means all number of CPU cores will be used. |
| MOE_ALL2ALL_BUFFER | bool | 0 | MOE_ALL2ALL_BUFFER: `0`: default, normal init; `1`: enable moe_all2all_buffer. |
| PROMPT_DEVICE_ID | string | None | The prefill device id for disaggregated prefilling case. |
| SOC_VERSION | string | ASCEND910B1 | The version of the Ascend chip. If not set, the default value is ASCEND910B1. It's used for package building. Please make sure that the version is correct. |
| USE_OPTIMIZED_MODEL | bool | 1 | Some models are optimized by vllm ascend. While in some case, e.g. rlhf training, the optimized model may not be suitable. In this case, set this value to False to disable the optimized model. |
| VERBOSE | bool | 0 | If set, vllm-ascend will print verbose logs during compilation. |
| VLLM_ASCEND_ENABLE_DBO | bool | 0 |  |
| VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE | bool | 0 | Whether to enable the topk optimization. It's disabled by default for experimental support We'll make it enabled by default in the future. |
| VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE | bool | 0 | Whether to enable the model execute time observe profile. Disable it when running vllm ascend in production environment. |
| VLLM_ASCEND_TRACE_RECOMPILES | bool | 0 | Whether to enable the trace recompiles from pytorch. |
| VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP | bool | 0 | Whether to enable fused_experts_allgather_ep. `MoeInitRoutingV3` and `GroupedMatmulFinalizeRouting` operators are combined to implement EP. |
| VLLM_VERSION | string | None | The version of vllm is installed. This value is used for developers who installed vllm from source locally. In this case, the version of vllm is usually changed. For example, if the version of vllm is `0.9.0`, but when it's installed from source, the version of vllm is usually set to `0.9.1`. In this case, developers need to set this value to `0.9.0` to make sure that the correct package is installed. |
