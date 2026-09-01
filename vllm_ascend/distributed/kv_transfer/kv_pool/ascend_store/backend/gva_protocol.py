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
# This file is a part of the vllm-ascend project.
#
"""Backward-compatible alias for the layerwise transfer protocol.

The protocol implementation moved into the memcache backend module; this
module keeps the pre-move import surface (``GVAKeyFactory`` /
``extract_layout_config``) working until the generic layers switch to the
protocol functions and this file is deleted.
"""

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    extract_layout_config,
    make_full_key,
    make_hit_check_keys,
    make_partial_key,
)

__all__ = ["GVAKeyFactory", "extract_layout_config"]


class GVAKeyFactory:
    """Static-method view over the memcache layerwise key functions."""

    full_key = staticmethod(make_full_key)
    partial_key = staticmethod(make_partial_key)
    hit_check_keys = staticmethod(make_hit_check_keys)
