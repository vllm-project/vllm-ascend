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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.utils.mem_constants import GiB_bytes

from vllm_ascend._310p.worker_310p import NPUWorker310


def _make_worker_and_profile_result(after_profile_free_memory: int):
    worker = object.__new__(NPUWorker310)
    worker.init_snapshot = SimpleNamespace(free_memory=60 * GiB_bytes)
    worker.requested_memory = 48 * GiB_bytes
    worker.model_runner = MagicMock()
    worker._scale_kv_cache_memory_for_multi_group = MagicMock(side_effect=lambda available_memory: available_memory)

    profile_result = SimpleNamespace(
        after_profile=SimpleNamespace(free_memory=after_profile_free_memory),
        non_torch_increase=0,
        torch_peak_increase=0,
        non_kv_cache_memory=2 * GiB_bytes,
    )
    return worker, profile_result


def test_determine_available_memory_accepts_equal_snapshot():
    worker, profile_result = _make_worker_and_profile_result(60 * GiB_bytes)
    profile_context = MagicMock()
    profile_context.__enter__.return_value = profile_result
    profile_context.__exit__.return_value = False

    with (
        patch("vllm_ascend._310p.worker_310p.memory_profiling", return_value=profile_context),
        patch("vllm_ascend._310p.worker_310p.is_rc_device", return_value=False),
        patch("vllm_ascend._310p.worker_310p.torch.npu.mem_get_info", return_value=(60 * GiB_bytes, 64 * GiB_bytes)),
        patch("vllm_ascend._310p.worker_310p.torch.npu.memory_reserved", return_value=4 * GiB_bytes),
    ):
        result = worker.determine_available_memory()

    assert result == 23 * GiB_bytes
    worker.model_runner.profile_run.assert_called_once_with()


def test_determine_available_memory_rejects_increased_snapshot():
    worker, profile_result = _make_worker_and_profile_result(61 * GiB_bytes)
    profile_context = MagicMock()
    profile_context.__enter__.return_value = profile_result
    profile_context.__exit__.return_value = False

    with (
        patch("vllm_ascend._310p.worker_310p.memory_profiling", return_value=profile_context),
        patch("vllm_ascend._310p.worker_310p.is_rc_device", return_value=False),
        patch("vllm_ascend._310p.worker_310p.torch.npu.mem_get_info", return_value=(60 * GiB_bytes, 64 * GiB_bytes)),
        patch("vllm_ascend._310p.worker_310p.torch.npu.memory_reserved", return_value=4 * GiB_bytes),
        pytest.raises(ValueError, match="Initial free memory 60.0 GiB, current free memory 61.0 GiB"),
    ):
        worker.determine_available_memory()
