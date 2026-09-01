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
        # The memcache backend carries an exclusive layerwise transfer
        # protocol (module path). Generic layers resolve it through
        # get_layerwise_protocol() and never import the protocol module
        # by name.
        "layerwise_protocol": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.gva_protocol",
    },
    "yuanrong": {
        "name": "YuanrongBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend",
    },
}


def get_layerwise_protocol(backend_name: str):
    """Import and return the layerwise protocol module of the backend
    registered under ``backend_name`` (None when the backend carries
    none)."""
    normalized_name = backend_name.strip().lower()
    module_path = backend_map.get(normalized_name, {}).get("layerwise_protocol")
    if module_path is None:
        return None
    import importlib

    return importlib.import_module(module_path)
