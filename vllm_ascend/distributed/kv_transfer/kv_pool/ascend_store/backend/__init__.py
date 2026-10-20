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
        "layerwise_protocol": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_layerwise",
    },
    "memcache": {
        "name": "MemcacheBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend",
        "layerwise_protocol": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend",
    },
    "yuanrong": {
        "name": "YuanrongBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend",
    },
}


def get_layerwise_protocol(backend_name: str):
    """Return a backend's lightweight layerwise protocol module, if any."""
    normalized_name = backend_name.strip().lower()
    backend = backend_map.get(normalized_name, {})
    protocol_path = backend.get("layerwise_protocol")
    if not protocol_path:
        return None
    import importlib

    return importlib.import_module(protocol_path)


def get_layerwise_data_plane(protocol: Any) -> str | None:
    """Return the protocol's common data-plane capability."""
    data_plane = getattr(protocol, "LAYERWISE_DATA_PLANE", None)
    return data_plane if data_plane in ("block_key", "gva") else None


def validate_layerwise_topology(protocol: Any, parallel_config: Any, use_layerwise: bool) -> None:
    """Let a backend validate coordinates represented by its wire keys."""
    if not use_layerwise or protocol is None:
        return
    validate = getattr(protocol, "validate_topology", None)
    if callable(validate):
        validate(parallel_config)


def validate_layerwise_runtime(protocol: Any, **runtime: Any) -> None:
    """Let a backend reject unsupported runtime layout combinations."""
    if protocol is None:
        return
    validate = getattr(protocol, "validate_runtime", None)
    if callable(validate):
        validate(**runtime)
