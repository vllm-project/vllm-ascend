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
# Patch DP device selection for the supported vLLM versions.
#
# CatCCOS on vLLM 0.23 needs every local DP process to see the full NPU list
# for SHMEM initialization. Its worker then selects the DP-rank NPU itself.
# vLLM v0.24.0+ also needs to tolerate a pre-sharded
# ASCEND_RT_VISIBLE_DEVICES env var (one slice per DP rank), instead of
# unconditionally applying ``local_dp_rank * world_size`` as an offset into it.
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

if vllm_version_is("0.23.0"):
    import os
    from contextlib import contextmanager

    from vllm.platforms import current_platform
    from vllm.v1.engine import utils as _engine_utils

    _original_set_device_control_env_var = _engine_utils.set_device_control_env_var

    def _uses_catccos_local_dp(vllm_config) -> bool:
        parallel_config = vllm_config.parallel_config
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        backend = additional_config.get("fused_mc2_backend")
        if backend is None:
            try:
                from vllm_ascend.ascend_config import get_ascend_config

                backend = get_ascend_config().fused_mc2_backend
            except RuntimeError:
                return False
        return (
            backend == "catccos"
            and parallel_config.data_parallel_size > 1
            and getattr(parallel_config, "distributed_executor_backend", None) not in ("ray", "external_launcher")
            and getattr(parallel_config, "data_parallel_backend", None) != "ray"
            and getattr(parallel_config, "nnodes_within_dp", 1) == 1
            and getattr(parallel_config, "assigned_physical_gpu_ids", None) is None
        )

    def _all_local_device_ids(vllm_config, device_control_env_var: str) -> str:
        parallel_config = vllm_config.parallel_config
        visible = os.environ.get(device_control_env_var)
        if visible:
            device_ids = [device.strip() for device in visible.split(",") if device.strip()]
            local_dp_size = (
                getattr(parallel_config, "data_parallel_size_local", None) or parallel_config.data_parallel_size
            )
            required = parallel_config.world_size * local_dp_size
            if len(device_ids) < required:
                raise RuntimeError(
                    f"CatCCOS local DP needs {required} visible devices, but "
                    f"{device_control_env_var} contains {len(device_ids)}: {visible!r}"
                )
            return ",".join(device_ids)

        local_dp_size = getattr(parallel_config, "data_parallel_size_local", None) or parallel_config.data_parallel_size
        device_count = parallel_config.world_size * local_dp_size
        return ",".join(str(current_platform.device_id_to_physical_device_id(i)) for i in range(device_count))

    @contextmanager
    def _set_device_control_env_var(vllm_config, local_dp_rank):
        if not _uses_catccos_local_dp(vllm_config):
            with _original_set_device_control_env_var(vllm_config, local_dp_rank):
                yield
            return

        device_control_env_var = current_platform.device_control_env_var
        visible_device_ids = _all_local_device_ids(vllm_config, device_control_env_var)
        previous_value = os.environ.get(device_control_env_var)
        os.environ[device_control_env_var] = visible_device_ids
        try:
            yield
        finally:
            if previous_value is None:
                os.environ.pop(device_control_env_var, None)
            else:
                os.environ[device_control_env_var] = previous_value

    _engine_utils.set_device_control_env_var = _set_device_control_env_var

elif not vllm_version_is("0.23.0"):
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
