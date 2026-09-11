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

from typing import Any

# Entries are heterogeneous (a marker boolean next to path/name strings), so
# the registry is annotated explicitly: without it mypy joins the per-backend
# dicts to object and every .get()/[] on an entry fails to type-check.
backend_map: dict[str, dict[str, Any]] = {
    "mooncake": {
        "name": "MooncakeBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend",
    },
    "memcache": {
        "name": "MemcacheBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend",
        # The backend module opts into the layerwise transfer protocol:
        # it exposes make_full_key / make_partial_key / make_hit_check_keys /
        # extract_layout_config at module level. Generic layers resolve the
        # module through get_layerwise_protocol() and never import it by name.
        "layerwise_protocol": True,
    },
    "yuanrong": {
        "name": "YuanrongBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend",
    },
}


def get_layerwise_protocol(backend_name: str):
    """Return the backend module carrying the layerwise transfer protocol
    registered under ``backend_name`` (None when the backend opts out).

    The protocol functions live in the backend module itself, so resolving
    reuses the registered ``path``: no second module path to drift, and
    backends without the marker (e.g. mooncake, whose module pulls heavy
    third-party imports at top level) are never imported here."""
    normalized_name = backend_name.strip().lower()
    backend = backend_map.get(normalized_name, {})
    if not backend.get("layerwise_protocol"):
        return None
    import importlib

    return importlib.import_module(backend["path"])
