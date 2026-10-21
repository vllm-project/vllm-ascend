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
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import wraps
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.parallel_state import get_pcp_group

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _apply_patch() -> None:
    original_sync = kv_transfer_state._sync_engine_id_across_tp
    if getattr(original_sync, "_ascend_pcp_engine_id_sync", False):
        return

    @wraps(original_sync)
    def sync_engine_id_with_pcp(vllm_config: "VllmConfig") -> None:
        original_sync(vllm_config)
        if vllm_config.parallel_config.prefill_context_parallel_size > 1:
            kv_config = vllm_config.kv_transfer_config
            assert kv_config is not None
            # TP/PP broadcasts leave independent PCP replicas with different
            # IDs. Select PCP rank 0 without crossing DP engine boundaries.
            kv_config.engine_id = get_pcp_group().broadcast_object(kv_config.engine_id, src=0)

    sync_engine_id_with_pcp._ascend_pcp_engine_id_sync = True
    kv_transfer_state._sync_engine_id_across_tp = sync_engine_id_with_pcp


_apply_patch()
