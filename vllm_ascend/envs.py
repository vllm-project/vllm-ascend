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

# begin-env-vars-definition


env_variables: dict[str, Callable[[], Any]] = {
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
    # PCP prefill KV placement. "replicated" (default) keeps one physical copy
    # on every PCP rank. "owner" backs the DSV4.1 KV cache with a VMM slab whose
    # physical 2 MiB chunks are split 1/pcp_size per rank while every rank maps
    # the complete logical address range. Reads and stores therefore address
    # the normal cache tensor; only the physical footprint changes. Only the
    # prefill PCP path is affected; decode keeps the replicated cache contract.
    "VLLM_ASCEND_PCP_KV_SHARING": lambda: os.getenv("VLLM_ASCEND_PCP_KV_SHARING", "replicated"),
    # Turn the HBM freed by owner sharing into extra KV blocks. Off by default.
    # When "1"/"true"/"yes"/"on" and a supported owner configuration is active,
    # the planner divides each owner-sharded tuple's per-block bytes by
    # pcp_size (replicated tensors, such as the LOCAL_INDEX copy, keep the full
    # page). The enabled staging scratch must fit in the
    # gpu-memory-utilization headroom; it is deliberately not subtracted from
    # the capacity because vLLM's rank re-plan assumes a linear divisor. Owner
    # sharding must actually be used: if the slab cannot be mapped, the rank
    # raises instead of falling back to a replicated cache that no longer fits.
    # Total KV HBM returns to the replicated budget; the win is block count.
    "VLLM_ASCEND_PCP_OWNER_CAPACITY": lambda: os.getenv("VLLM_ASCEND_PCP_OWNER_CAPACITY", "0"),
    # Per-chunk SWA staging for PCP owner-sharded KV. Off by default. When
    # "1"/"true"/"yes"/"on", a pure-prefill step copies only the rows each
    # virtual segment can see -- ``[chunk_start - sliding_window, chunk_end)``
    # of every local chunk -- from the owner slab into local scratch, and runs
    # the sparse-attention kernel against that bounded view. Inactive when no
    # owner slab backs the SWA cache, for mixed/decode batches, or when
    # bounded replay is in use.
    "VLLM_ASCEND_PCP_OWNER_STAGE_SWA_CHUNK": lambda: os.getenv("VLLM_ASCEND_PCP_OWNER_STAGE_SWA_CHUNK", "0"),
    # Local scratch budget in bytes for the per-chunk SWA staging. One request
    # stages at most ``2 * (chunk_len + sliding_window)`` SWA rows per layer,
    # so the default 256 MiB covers many more requests than max_num_seqs.
    "VLLM_ASCEND_PCP_OWNER_STAGE_SWA_CHUNK_BYTES": lambda: os.getenv(
        "VLLM_ASCEND_PCP_OWNER_STAGE_SWA_CHUNK_BYTES", "268435456"
    ),
    # Print the rank-local chunk layout once per process (num_rows,
    # query_start_loc, seq_lens, and each row's chunk_start/len/end). First
    # "1"/"true"/"yes"/"on" only; for validating the one-row-per-chunk
    # assumption on a live prompt.
    "VLLM_ASCEND_PCP_OWNER_STAGE_SWA_CHUNK_DEBUG": lambda: os.getenv(
        "VLLM_ASCEND_PCP_OWNER_STAGE_SWA_CHUNK_DEBUG", "0"
    ),
    # Keep the DSV4.1 indexer K cache out of the owner-sharded long-KV tuple.
    # Off by default. When "1"/"true"/"yes"/"on", the index plane gets its own
    # replicated tensor (page = its unpadded size) while long KV, compressor
    # state and the SWA aliases stay in the owner-sharded tuple. The indexer
    # then reads local HBM instead of peer pages in ``select``. Costs one full
    # index copy per rank (about 1.2 GiB) and lowers the global block capacity
    # by roughly 4-5%.
    "VLLM_ASCEND_PCP_LOCAL_INDEX": lambda: os.getenv("VLLM_ASCEND_PCP_LOCAL_INDEX", "0"),
    # Write prefill KV for each rank's own DualChunkSwap chunks from the
    # rank-local hidden states instead of the gathered view. Off by default and
    # ignored unless an owner slab mapped successfully: on a replicated
    # allocation the local write would lose the other ranks' rows. Only the
    # prefill PCP path is affected; decode keeps the gathered write.
    "VLLM_ASCEND_PCP_LOCAL_KV_WRITE": lambda: os.getenv("VLLM_ASCEND_PCP_LOCAL_KV_WRITE", "0"),
    # Also write the compressed source planes from local tokens. Off by
    # default. Requires VLLM_ASCEND_PCP_LOCAL_KV_WRITE and assumes chunk-aligned
    # ratio-2 pairing. Inactive while VLLM_ASCEND_PCP_LOCAL_INDEX keeps the
    # index plane replicated, because that copy needs the complete global rows.
    "VLLM_ASCEND_PCP_LOCAL_KV_WRITE_SOURCE": lambda: os.getenv("VLLM_ASCEND_PCP_LOCAL_KV_WRITE_SOURCE", "0"),
    # Memoize long-KV staging per source layer for owner-sharded prefill: copy
    # the referenced slab blocks into local scratch once per forward instead of
    # paying per-row peer reads in the sparse-attention kernel. Off by default;
    # overflow falls back to direct peer reads. Only pure-prefill steps stage.
    "VLLM_ASCEND_PCP_OWNER_STAGE": lambda: os.getenv("VLLM_ASCEND_PCP_OWNER_STAGE", "0"),
    # Local scratch budget in bytes for long-KV staging. Default 1 GiB.
    "VLLM_ASCEND_PCP_OWNER_STAGE_BYTES": lambda: os.getenv("VLLM_ASCEND_PCP_OWNER_STAGE_BYTES", "1073741824"),
    # Stage the whole live sliding-window prefix as well. Off by default and
    # slower than the bounded per-chunk path; kept as a diagnostic.
    "VLLM_ASCEND_PCP_OWNER_STAGE_SWA": lambda: os.getenv("VLLM_ASCEND_PCP_OWNER_STAGE_SWA", "0"),
    # Plan staged tables from host-side block counts instead of device masks.
    # Diagnostic and off by default; the mask path is the validated one.
    "VLLM_ASCEND_PCP_OWNER_STAGE_CPUPLAN": lambda: os.getenv("VLLM_ASCEND_PCP_OWNER_STAGE_CPUPLAN", "0"),
    # Diagnostic: shard only the first N owner slabs and keep the rest
    # replicated, to measure the HBM/latency tradeoff. Unset means all slabs.
    "VLLM_ASCEND_PCP_OWNER_MAX_SLABS": lambda: os.getenv("VLLM_ASCEND_PCP_OWNER_MAX_SLABS", None),
    # Diagnostic control mode: allocate the full slab locally on every rank
    # (same virtual mapping, no peer export/import, no HBM saving).
    "VLLM_ASCEND_PCP_OWNER_LOCAL_VMM": lambda: os.getenv("VLLM_ASCEND_PCP_OWNER_LOCAL_VMM", "0"),
    # Allow an external KV-transfer connector alongside owner sharding. Off by
    # default; the owner slab is not validated with KV pooling or Mooncake.
    "VLLM_ASCEND_PCP_OWNER_ALLOW_KV_TRANSFER": lambda: os.getenv("VLLM_ASCEND_PCP_OWNER_ALLOW_KV_TRANSFER", "0"),
    # Log per-region device time for the DSV4.1 attention forward. Off by
    # default; synchronizes only in batches.
    "VLLM_ASCEND_PCP_TIMING": lambda: os.getenv("VLLM_ASCEND_PCP_TIMING", "0"),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
