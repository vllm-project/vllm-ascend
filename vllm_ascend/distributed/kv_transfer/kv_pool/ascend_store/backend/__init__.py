#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

backend_map = {
    "mooncake": {
        "name": "MooncakeBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend",
    },
    "memcache": {
        "name": "MemcacheBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend",
    },
    "yuanrong": {
        "name": "YuanrongBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend",
    },
}

# Optional protocol families implemented by each backend. This table is the
# single source of truth for backend-specific feature gating in the generic
# layers; keep it in sync with the backend classes in backend_map
# (tests/ut/distributed/ascend_store/test_backend.py asserts the consistency
# between this table and GVALayerwiseCapable subclasses).
_BACKEND_CAPABILITIES: dict[str, frozenset[str]] = {
    "mooncake": frozenset(),
    "memcache": frozenset({"gva_layerwise"}),
    "yuanrong": frozenset(),
}


def backend_supports(backend_name: str, capability: str) -> bool:
    """Return True if the backend registered under ``backend_name``
    implements the given optional capability (e.g. ``"gva_layerwise"``).
    """
    return capability in _BACKEND_CAPABILITIES.get(backend_name, frozenset())


def use_gva_layerwise(use_layerwise: bool, backend_name: str) -> bool:
    """Single derivation point for the GVA layerwise transfer mode.

    The layerwise GVA fast path is a memcache-specific protocol, so every
    call site must derive the flag from here instead of re-spelling the
    backend string comparison. Duplicated derivations have already caused a
    live regression: #14465 deleted one copy as dead code while a reader
    still consumed it.
    """
    return use_layerwise and backend_supports(backend_name, "gva_layerwise")
