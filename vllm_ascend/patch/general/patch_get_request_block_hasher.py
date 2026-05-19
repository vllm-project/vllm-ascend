#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
# See the License for the specific governing permissions and
# limitations under the License.

"""Scale ``get_request_block_hasher`` block size by CP world size when needed.

``EngineCore`` passes ``hash_block_size`` from ``resolve_kv_cache_block_sizes``,
which is already ``cache_config.block_size * pcp * dcp`` in the common case.
If anything still passes the physical cache block size while CP is enabled,
multiply once by ``pcp * dcp`` so hashing aligns with the scheduler virtual block.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import vllm.v1.engine.core as engine_core_module
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.scheduler import Scheduler

_ORIG_SCHEDULER_INIT = Scheduler.__init__
_ORIG_GET_REQUEST_BLOCK_HASHER = kv_cache_utils.get_request_block_hasher
_VLLM_CONFIG_SNAPSHOT: dict[str, Any] = {}


def _set_vllm_config_snapshot(vllm_config: Any) -> None:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    _VLLM_CONFIG_SNAPSHOT.clear()
    _VLLM_CONFIG_SNAPSHOT.update(
        {
            "decode_context_parallel_size": getattr(parallel_config, "decode_context_parallel_size", None),
            "prefill_context_parallel_size": getattr(parallel_config, "prefill_context_parallel_size", None),
            "block_size": getattr(cache_config, "block_size", None),
        }
    )


def _patched_scheduler_init(self, vllm_config, *args, **kwargs):
    _set_vllm_config_snapshot(vllm_config)
    return _ORIG_SCHEDULER_INIT(self, vllm_config, *args, **kwargs)


def get_hash_size(block_size: int) -> int:
    dcp_size: int = int(_VLLM_CONFIG_SNAPSHOT.get("decode_context_parallel_size", 1) or 1)
    pcp_size: int = int(_VLLM_CONFIG_SNAPSHOT.get("prefill_context_parallel_size", 1) or 1)
    cache_block_size: int = int(_VLLM_CONFIG_SNAPSHOT.get("block_size", 1) or 1)

    if block_size == cache_block_size * pcp_size * dcp_size:
        block_size //= dcp_size
    return block_size


def ascend_get_request_block_hasher(
    block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Any], list[BlockHash]]:
    hash_block_size = get_hash_size(block_size)
    return _ORIG_GET_REQUEST_BLOCK_HASHER(hash_block_size, caching_hash_fn)


kv_cache_utils.get_request_block_hasher = ascend_get_request_block_hasher
engine_core_module.get_request_block_hasher = ascend_get_request_block_hasher
Scheduler.__init__ = _patched_scheduler_init
