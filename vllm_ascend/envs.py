#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from typing import Any, Callable, Dict

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: Dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled. Please note that
    # the sleep mode feature will be disabled as well if custom kernels are not
    # compiled.
    "COMPILE_CUSTOM_KERNELS":
    lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER":
    lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER":
    lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. If not set, the default value is
    # ASCEND910B1(Available for A2 and A3 series). It's used for package building.
    # Please make sure that the version is correct.
    "SOC_VERSION":
    lambda: os.getenv("SOC_VERSION", "ASCEND910B1"),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH":
    lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so。
    "HCCL_SO_PATH":
    lambda: os.environ.get("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION":
    lambda: os.getenv("VLLM_VERSION", None),
    # Whether to enable the trace recompiles from pytorch.
    "VLLM_ASCEND_TRACE_RECOMPILES":
    lambda: bool(int(os.getenv("VLLM_ASCEND_TRACE_RECOMPILES", '0'))),
    # Whether to enable fused_experts_allgather_ep. MoeInitRoutingV3 and
    # GroupedMatmulFinalizeRouting operators are combined to implement EP.
    "VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP":
    lambda: bool(int(os.getenv("VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP", '0'))
                 ),
    # Whether to enable DBO feature for deepseek model.
    "VLLM_ASCEND_ENABLE_DBO":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_DBO", '0'))),
    # Whether to enable the model execute time observe profile. Disable it when
    # running vllm ascend in production environment.
    "VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE", '0'))
                 ),
    # Some models are optimized by vllm ascend. While in some case, e.g. rlhf
    # training, the optimized model may not be suitable. In this case, set this
    # value to False to disable the optimized model.
    "USE_OPTIMIZED_MODEL":
    lambda: bool(int(os.getenv('USE_OPTIMIZED_MODEL', '1'))),
    # The tolerance of the kv cache size, if the difference between the
    # actual kv cache size and the cached kv cache size is less than this value,
    # then the cached kv cache size will be used.
    "VLLM_ASCEND_KV_CACHE_MEGABYTES_FLOATING_TOLERANCE":
    lambda: int(
        os.getenv("VLLM_ASCEND_KV_CACHE_MEGABYTES_FLOATING_TOLERANCE", 64)),
    # Whether to enable the topk optimization. It's enabled by default. Please set to False if you hit any issue.
    # We'll remove this flag in the future once it's stable enough.
    "VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION":
    lambda: bool(
        int(os.getenv("VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION", '1'))),
    # `LLMDataDistCMgrConnector` required variable. `DISAGGREGATED_PREFILL_RANK_TABLE_PATH` is
    # used for llmdatadist to build the communication topology for kv cache transfer, it is
    # a required variable if `LLMDataDistCMgrConnector` is used as kv connector for disaggregated
    # pd. The rank table can be generated by adopting the script `gen_ranktable.sh`
    # in vllm_ascend's example folder.
    "DISAGGREGATED_PREFILL_RANK_TABLE_PATH":
    lambda: os.getenv("DISAGGREGATED_PREFILL_RANK_TABLE_PATH", None),
    # `LLMDataDistCMgrConnector` required variable. `VLLM_ASCEND_LLMDD_RPC_IP` is used as the
    # rpc communication listening ip, which will be used to receive the agent metadata from the
    # remote worker.
    "VLLM_ASCEND_LLMDD_RPC_IP":
    lambda: os.getenv("VLLM_ASCEND_LLMDD_RPC_IP", "0.0.0.0"),
    # `LLMDataDistCMgrConnector` required variable. `VLLM_ASCEND_LLMDD_RPC_PORT` is used as the
    # rpc communication listening port, which will be used to receive the agent metadata from the
    # remote worker.
    "VLLM_ASCEND_LLMDD_RPC_PORT":
    lambda: int(os.getenv("VLLM_ASCEND_LLMDD_RPC_PORT", 5557)),
    # Whether to enable mla_pa for deepseek mla decode, this flag will be removed after its available torch_npu is public accessible
    # and the mla_pa will be the default path of deepseek decode path.
    "VLLM_ASCEND_MLA_PA":
    lambda: int(os.getenv("VLLM_ASCEND_MLA_PA", 0)),
    # Whether to enable MatmulAllReduce fusion kernel when tensor parallel is enabled.
    # this feature is supported in A2, and eager mode will get better performance.
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", '0'))),
    # Whether to enable FlashComm optimization when tensor parallel is enabled.
    # This feature will get better performance when concurrency is large.
    "VLLM_ASCEND_ENABLE_FLASHCOMM":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", '0'))),
    # Whether to enable MLP weight prefetch, only used in small concurrency.
    "VLLM_ASCEND_ENABLE_PREFETCH_MLP":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_PREFETCH_MLP", '0'))),
    # buffer size for gate up prefetch
    "VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE":
    lambda: int(
        os.getenv("VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE", 18 * 1024 * 1024)),
    # buffer size for down proj prefetch
    "VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE":
    lambda: int(
        os.getenv("VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE", 18 * 1024 * 1024)),
    # Whether to enable dense model and general optimizations for better performance.
    # Since we modified the base parent class `linear`, this optimization is also applicable to other model types.
    # However, there might be hidden issues, and it is currently recommended to prioritize its use with dense models.
    "VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE", '0'))),
    # Whether to enable mlp optimize when tensor parallel is enabled.
    # this feature in eager mode will get better performance.
    "VLLM_ASCEND_ENABLE_MLP_OPTIMIZE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLP_OPTIMIZE", '0'))),
    # Determine the number of physical devices in a non-full-use scenario
    # caused by the initialization of the Mooncake connector.
    "PHYSICAL_DEVICES":
    lambda: os.getenv("PHYSICAL_DEVICES", None),
    # Timeout (in seconds) for delayed KVCache block release. In the prefill
    # node, if a request is marked for delayed KV block release and the blocks
    # are not freed within this timeout, they will be forcibly released.
    "VLLM_ASCEND_KVCACHE_DELAY_FREE_TIMEOUT":
    lambda: int(os.getenv("VLLM_ASCEND_KVCACHE_DELAY_FREE_TIMEOUT", 250)),
    # Decide whether we should enable CP parallelism and SP parallelism.
    "VLLM_ASCEND_ENABLE_SP":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_SP", '0'))),
    "VLLM_ASCEND_ENABLE_CP":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_CP", '0')))
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
