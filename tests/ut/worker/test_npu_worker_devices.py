# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Regression tests for NPUWorker device sufficiency check.

Covers https://github.com/vllm-project/vllm-ascend/issues/14631
("NPU 环境下设备不足与显存不足错误提示对 NPU 用户不友好").

The check is exposed as ``NPUWorker._raise_if_world_size_exceeds_devices``,
a pure static method, so it can be exercised without instantiating the full
worker (which requires NPU hardware and distributed init).
"""

import unittest

import pytest

from vllm_ascend.worker.worker import NPUWorker


class TestRaiseIfWorldSizeExceedsDevices(unittest.TestCase):
    """Regression tests for the friendly insufficient-devices error."""

    def test_returns_none_when_world_size_within_limit(self):
        """No raise when local_world_size <= visible_device_count."""
        # Equal to visible count: still valid.
        NPUWorker._raise_if_world_size_exceeds_devices(local_world_size=2, visible_device_count=2)
        # Below visible count: valid.
        NPUWorker._raise_if_world_size_exceeds_devices(local_world_size=1, visible_device_count=8)

    def test_raises_runtime_error_not_assertion(self):
        """Bug fix: must raise RuntimeError, not AssertionError, so external
        callers can catch the failure mode without inheriting from AssertionError
        (which is conventionally reserved for internal invariant checks).
        """
        with pytest.raises(RuntimeError, match="Insufficient NPU devices"):
            NPUWorker._raise_if_world_size_exceeds_devices(local_world_size=4, visible_device_count=2)

    def test_omits_suggestion_5_when_no_devices_visible(self):
        """When visible_device_count == 0 (driver-less / CI), suggestion 5
        (reduce TP/DP) is not actionable and must be omitted to avoid
        misleading the user into futile TP/DP tuning.
        """
        with pytest.raises(RuntimeError) as exc_info:
            NPUWorker._raise_if_world_size_exceeds_devices(local_world_size=2, visible_device_count=0)
        message = str(exc_info.value)
        # Suggestions 1-4 must be present.
        assert "npu-smi info" in message
        assert "/dev/davinci*" in message
        assert "CANN toolkit" in message
        assert "permission to access NPU devices" in message
        # Suggestion 5 must NOT be present.
        assert "--tensor-parallel-size" not in message
        assert "--data-parallel-size" not in message

    def test_includes_suggestion_5_when_some_devices_visible(self):
        """When at least one NPU is visible, suggestion 5 is actionable and
        must be included so the user can reduce parallel degree to match.
        """
        with pytest.raises(RuntimeError) as exc_info:
            NPUWorker._raise_if_world_size_exceeds_devices(local_world_size=8, visible_device_count=2)
        message = str(exc_info.value)
        assert "--tensor-parallel-size" in message
        assert "--data-parallel-size" in message
        # Suggestion must reflect the actual visible count.
        assert "2 available device(s)" in message

    def test_includes_local_world_size_and_visible_count(self):
        """Error message must include both numbers so the user can diagnose."""
        with pytest.raises(RuntimeError) as exc_info:
            NPUWorker._raise_if_world_size_exceeds_devices(local_world_size=16, visible_device_count=4)
        message = str(exc_info.value)
        assert "local_world_size=16" in message
        assert "4 NPU device(s)" in message


if __name__ == "__main__":
    unittest.main()
