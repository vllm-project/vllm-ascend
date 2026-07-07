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
# Patch vLLM v0.24.0+ ``get_physical_gpu_ids_for_local_dp_rank`` so that it
# tolerates a pre-sharded ASCEND_RT_VISIBLE_DEVICES env var (one slice per
# DP rank), instead of unconditionally applying ``local_dp_rank * world_size``
# as an offset into it.
#
# Background:
# PR #45026 removed the per-process device isolation that older vLLM
# versions performed internally. Application-level DP (e.g.
# ``offline_data_parallel.py``) now has to slice ASCEND_RT_VISIBLE_DEVICES
# per rank itself, but the upstream helper still expects the env var to
# contain ALL devices for ALL ranks and tries to read it with the
# ``local_dp_rank * world_size`` offset. With a sharded env var, that
# offset is out of range and the helper raises ``IndexError`` (wrapped in
# the user-facing "Error computing device indices for ..." message).

from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.23.0"):
    import os

    from vllm.platforms import current_platform
    from vllm.v1.engine import utils as _engine_utils

    _original_get_physical_gpu_ids = _engine_utils.get_physical_gpu_ids_for_local_dp_rank

    def _patched_get_physical_gpu_ids_for_local_dp_rank(
        device_control_env_var,
        local_dp_rank,
        world_size,
        local_world_size=None,
        user_assigned_gpu_ids=None,
    ):
        if local_world_size is None:
            local_world_size = world_size

        # If the caller did not pass --device-ids and the env var has
        # fewer devices than the full DP range expects, the env var has
        # already been pre-sharded per rank by the caller. Use it
        # directly from index 0 instead of applying the DP offset again.
        if user_assigned_gpu_ids is None and device_control_env_var in os.environ:
            visible = [d for d in os.environ[device_control_env_var].split(",") if d]
            if local_dp_rank * world_size + local_world_size > len(visible):
                return [
                    current_platform.device_control_id_to_physical_device_id(visible[device_id])
                    for device_id in range(local_world_size)
                ]

        return _original_get_physical_gpu_ids(
            device_control_env_var,
            local_dp_rank,
            world_size,
            local_world_size,
            user_assigned_gpu_ids,
        )

    _engine_utils.get_physical_gpu_ids_for_local_dp_rank = _patched_get_physical_gpu_ids_for_local_dp_rank
