# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.utils.mem_constants import GiB_bytes


def test_310p_non_rc_memory_does_not_charge_other_instance() -> None:
    """Do not subtract device-global allocations from a second 310P instance."""
    from vllm_ascend._310p.worker_310p import NPUWorker310

    worker = object.__new__(NPUWorker310)
    worker.init_snapshot = SimpleNamespace(
        free_memory=32 * GiB_bytes,
        total_memory=64 * GiB_bytes,
    )
    worker.requested_memory = int(64 * 0.4 * GiB_bytes)
    worker.model_runner = MagicMock()
    worker.model_runner.model_memory_usage = int(0.5 * GiB_bytes)

    profile_result = MagicMock()
    profile_result.after_profile.free_memory = worker.init_snapshot.free_memory - int(0.5 * GiB_bytes)
    profile_result.non_kv_cache_memory = int(0.5 * GiB_bytes)
    profile_result.non_torch_increase = 0
    profile_result.torch_peak_increase = 0

    context = MagicMock()
    context.__enter__.return_value = profile_result
    context.__exit__.return_value = False

    with (
        patch("vllm_ascend._310p.worker_310p.is_rc_device", return_value=False),
        patch("vllm_ascend._310p.worker_310p.memory_profiling", return_value=context),
    ):
        result = worker.determine_available_memory()

    expected = (worker.requested_memory - profile_result.non_kv_cache_memory) // 2
    assert result == expected
    assert result > 0
    worker.model_runner.profile_run.assert_called_once()
