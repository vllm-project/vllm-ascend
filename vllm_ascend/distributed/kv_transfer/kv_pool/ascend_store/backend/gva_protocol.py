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
"""Layerwise GVA transfer protocol (memcache backend).

GVA is a memcache-exclusive protocol. The backend registry entry for
memcache points at this module (``layerwise_protocol``), and the generic
layers resolve it through backend/__init__.py — they never import this
module by name. Key formats are centralized here so the worker-side and
scheduler-side constructions cannot drift apart; the strings are
byte-for-byte identical to the pre-refactor ``pool_worker`` /
``pool_scheduler`` implementations.
``tests/ut/distributed/ascend_store/test_gva_protocol.py`` locks the
memcache exclusivity of the GVA store methods and the key formats with
snapshot assertions.
"""

from __future__ import annotations


class GVAKeyFactory:
    """String formats for the layerwise GVA keys.

    Single-group models use the PR #11585 format (model@hash@rank) for
    backward compatibility. Multi-group models include group_id
    (model@group_id@hash@rank) to distinguish groups.
    """

    @staticmethod
    def full_key(
        model_name: str,
        group_id: int,
        block_hash_hex: str,
        head_or_tp_rank: int,
        num_groups: int,
    ) -> str:
        if num_groups > 1:
            return f"{model_name}@{group_id}@{block_hash_hex}@{head_or_tp_rank}"
        else:
            return f"{model_name}@{block_hash_hex}@{head_or_tp_rank}"

    @staticmethod
    def partial_key(
        model_name: str,
        req_id: str,
        group_id: int,
        block_index: int,
        end_token: int,
        head_or_tp_rank: int,
    ) -> str:
        return f"{model_name}@partial@{req_id}@{group_id}@{block_index}@{end_token}@{head_or_tp_rank}"

    @staticmethod
    def hit_check_keys(
        model_name: str,
        group_id: int,
        block_hash_hex: str,
        num_ranks: int,
        num_groups: int,
    ) -> list[str]:
        """All-rank GVA keys for scheduler-side hit check.

        Returns one key per head_or_tp_rank (ranks in the same put_step
        group share one key for MLA).
        """
        if num_groups > 1:
            return [f"{model_name}@{group_id}@{block_hash_hex}@{h}" for h in range(num_ranks)]
        else:
            return [f"{model_name}@{block_hash_hex}@{h}" for h in range(num_ranks)]
