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
    # Whether to enable MC2 for DeepSeek. If not set, the default value is False.
    # MC2 is a fusion operator provided by Ascend to speed up computing and communication.
    # Find more detail here: https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0043.html
    "VLLM_ENABLE_MC2":
    lambda: bool(int(os.getenv("VLLM_ENABLE_MC2", '0'))),
    # Whether to enable the topk optimization. It's disabled by default for experimental support
    # We'll make it enabled by default in the future.
    "VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE", '0'))),
    # Whether to use LCCL communication. If not set, the default value is False.
    "USING_LCCL_COM":
    lambda: bool(int(os.getenv("USING_LCCL_COM", '0'))),
    # The version of the Ascend chip. If not set, the default value is
    # ASCEND910B1. It's used for package building. Please make sure that the
    # version is correct.
    "SOC_VERSION":
    lambda: os.getenv("SOC_VERSION", "ASCEND910B1"),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH":
    lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCN Tool, the tool will be called by disaggregated prefilling
    # case.
    "HCCN_PATH":
    lambda: os.getenv("HCCN_PATH", "/usr/local/Ascend/driver/tools/hccn_tool"),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so。
    "HCCL_SO_PATH":
    # The prefill device id for disaggregated prefilling case.
    lambda: os.environ.get("HCCL_SO_PATH", None),
    "PROMPT_DEVICE_ID":
    lambda: os.getenv("PROMPT_DEVICE_ID", None),
    # The decode device id for disaggregated prefilling case.
    "DECODE_DEVICE_ID":
    lambda: os.getenv("DECODE_DEVICE_ID", None),
    # The port number for llmdatadist communication. If not set, the default
    # value is 26000.
    "LLMDATADIST_COMM_PORT":
    lambda: os.getenv("LLMDATADIST_COMM_PORT", "26000"),
    # The wait time for llmdatadist sync cache. If not set, the default value is
    # 5000ms.
    "LLMDATADIST_SYNC_CACHE_WAIT_TIME":
    lambda: os.getenv("LLMDATADIST_SYNC_CACHE_WAIT_TIME", "5000"),
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
    "VLLM_ASCEND_ENABLE_DBO":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_DBO", '0'))),
    # Whether to enable the model execute time observe profile. Disable it when
    # running vllm ascend in production environment.
    "VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE", '0'))
                 ),
    # MOE_ALL2ALL_BUFFER:
    #   0: default, normal init.
    #   1: enable moe_all2all_buffer.
    "MOE_ALL2ALL_BUFFER":
    lambda: bool(int(os.getenv("MOE_ALL2ALL_BUFFER", '0'))),
    # VLLM_ASCEND_ACL_OP_INIT_MODE:
    #   0: default, normal init.
    #   1: delay init until launch aclops.
    #   2: forbid aclops init and launch.
    # Find more details at https://gitee.com/ascend/pytorch/pulls/18094
    # We set this var default to `1` in vllm-ascend to avoid segment fault when
    # enable `pin_memory` while creating a tensor using `torch.tensor`.
    "VLLM_ASCEND_ACL_OP_INIT_MODE":
    lambda: os.getenv("VLLM_ASCEND_ACL_OP_INIT_MODE", '1'),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
