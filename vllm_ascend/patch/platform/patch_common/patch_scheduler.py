#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
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
from vllm.v1.core.sched.scheduler import Scheduler


def ascend_update_waiting_for_remote_kv(self, request) -> bool:
    """
    P/D: check if the request_id is finished_recving.

    The finished_recving_kv_req_ids list is populated
    on the previous steps()'s update_from_output based
    on the worker side connector.

    When the kv transfer is ready, we cache the blocks
    and the request state will be moved back to WAITING from
    WAITING_FOR_REMOTE_KV.
    """
    if request.request_id not in self.finished_recving_kv_req_ids:
        return False
    assert len(self.kv_cache_config.kv_cache_groups
               ) == 1, "KV connector only supports one KV cache group now"
    # Now that the blocks are ready, actually cache them.
    # In order to make decode node always do the decode step, we transfer every block as long as it contains the
    # data computed by prefill node.
    num_computed_tokens = request.num_tokens
    if num_computed_tokens == request.num_tokens:
        num_computed_tokens -= 1
    self.kv_cache_manager.cache_blocks(
        request,
        self.kv_cache_manager.req_to_block_hashes[request.request_id],
        num_computed_tokens,
    )

    # Update the request state for scheduling.
    request.num_computed_tokens = num_computed_tokens

    # Return that we are ready.
    self.finished_recving_kv_req_ids.remove(request.request_id)
    return True


Scheduler._update_waiting_for_remote_kv = ascend_update_waiting_for_remote_kv
