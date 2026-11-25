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
from collections.abc import Callable
from typing import Any


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable as ``1`` or ``0``.

    Only ``1`` and ``0`` are accepted (surrounding whitespace is ignored).
    Unset or empty values fall back to ``default``, which keeps
    ``export VAR=`` from crashing. Any other value raises ValueError naming the
    variable, instead of failing inside ``int()`` with an opaque message.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip()
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"Invalid value for environment variable {name}: {raw!r}. Expected '0' or '1'.")


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

def _moonep_prefetch_side_stream_overlap() -> str:
    value = os.getenv("MOONEP_PREFETCH_SIDE_STREAM_OVERLAP", "dispatch")
    if value not in ("none", "dispatch", "dispatch|home_gemm1"):
        raise ValueError(
            "MOONEP_PREFETCH_SIDE_STREAM_OVERLAP must be none, dispatch, "
            "or dispatch|home_gemm1"
        )
    return value


def _moonep_cached_redundant_expert_count() -> int:
    value = int(os.getenv("MOONEP_CACHED_REDUNDANT_EXPERT_COUNT", "0"))
    if value < 0:
        raise ValueError(f"MOONEP_CACHED_REDUNDANT_EXPERT_COUNT must be >= 0, got {value}")
    return value


def _moonep_shmem_bootstrap_type() -> str:
    value = os.getenv("MOONEP_SHMEM_BOOTSTRAP_TYPE", "ipport")
    if value not in ("ipport", "uid"):
        raise ValueError(
            f"MOONEP_SHMEM_BOOTSTRAP_TYPE must be ipport or uid, got {value!r}"
        )
    return value


def _moonep_transport_type() -> str:
    value = os.getenv("MOONEP_TRANSPORT_TYPE", "mte")
    if value not in ("mte", "udma"):
        raise ValueError(
            f"MOONEP_TRANSPORT_TYPE must be mte or udma, got {value!r}"
        )
    return value


# begin-env-vars-definition


env_variables: dict[str, Callable[[], Any]] = {
    # Existing MoonEP configuration; defaults match moonep_v0.23.0.
    # MoonEP dispatch/combine/weight-prefetch transport; planning stays native.
    # Non-sensitive; exact lowercase mte (default) or udma. Read at startup,
    # without whitespace/case normalization or fallback to the removed flag.
    "MOONEP_TRANSPORT_TYPE": _moonep_transport_type,
    # SHMEM ipport bootstrap's TCP port only; IP comes from vLLM's DP master.
    # Non-sensitive; integer, default 28765. Read only for the ipport branch;
    # preserve existing integer parsing without adding port-range validation.
    "MOONEP_SHMEM_BOOTSTRAP_PORT": lambda: int(os.getenv("MOONEP_SHMEM_BOOTSTRAP_PORT", "28765")),
    # SHMEM startup bootstrap: ipport (default, TCP config store) or uid.
    # Non-sensitive; exact lowercase values only, no whitespace normalization.
    "MOONEP_SHMEM_BOOTSTRAP_TYPE": _moonep_shmem_bootstrap_type,
    # Manual MoonEP pool token-capacity tiers: comma-separated integers.
    # Non-sensitive; unset/empty selects automatic generation. Parsing, sorting,
    # deduplication and maximum-capacity handling stay in the ladder builder.
    "MOONEP_POOL_TOKEN_CAPACITY_TIERS": lambda: os.getenv("MOONEP_POOL_TOKEN_CAPACITY_TIERS"),
    # Automatic pool tier-count target, integer, default 32. Non-sensitive.
    # Existing base tiers and maximum capacity are retained; not a hard limit.
    "MOONEP_POOL_MAX_TIER_COUNT": lambda: int(os.getenv("MOONEP_POOL_MAX_TIER_COUNT", "32")),
    # MoonEP automatic capacity-ladder memory estimate, in MiB despite the MB
    # suffix. Non-sensitive; integer, default 2048; <=0 disables the budget.
    # Per-shape ladder heuristic, not a total device-memory allocation limit.
    "MOONEP_POOL_MEMORY_BUDGET_MB": lambda: int(os.getenv("MOONEP_POOL_MEMORY_BUDGET_MB", "2048")),
    # Persistent remote-expert cache capacity per EP rank and MoE layer.
    # Non-sensitive; integer >=0, default 0 disables caching. No E/R or Slot
    # capacity upper bound; available memory still limits actual allocation.
    "MOONEP_CACHED_REDUNDANT_EXPERT_COUNT": _moonep_cached_redundant_expert_count,
    # MoonEP prefetch overlap: none, dispatch (default), dispatch|home_gemm1.
    # Non-sensitive; set before worker startup. Home GEMM1 requires cache off
    # and an already-supported Home GEMM backend/operator library.
    "MOONEP_PREFETCH_SIDE_STREAM_OVERLAP": _moonep_prefetch_side_stream_overlap,
    # MoonEP small-step policy: when enable_moonep is on, keep the steps whose
    # SoC-default selection is MC2 (num_tokens <= mc2_tokens_capacity) on MC2 and
    # replace only the all-gather / all-to-all selections with SHMEM. Default
    # off: every step within the SHMEM capacity S goes to MoonEP. MC2 needs the
    # HCCL UB_MEM link, which a cross-UB-domain card group does not have
    # (aclnnMoeDistributeDispatchV4 561000), so enable this only on one UB domain.
    "MOONEP_KEEP_MC2": lambda: bool(int(os.getenv("MOONEP_KEEP_MC2", "0"))),
    # Redundant expert prefetch slots per EP rank, not the persistent cache size.
    # Unset is resolved per layer as E/EP; the current planner requires B == E/EP.
    # Runtime accepts integer values in [1, E/EP] before that planner check.
    # Non-sensitive. This does not change the SDK's B parameter.
    "MOONEP_REDUNDANT_EXPERT_SLOT_COUNT": lambda: os.getenv("MOONEP_REDUNDANT_EXPERT_SLOT_COUNT", None),

    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: _bool_env("COMPILE_CUSTOM_KERNELS", True),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: _bool_env("VERBOSE", False),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
