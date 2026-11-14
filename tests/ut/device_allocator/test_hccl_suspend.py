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

from vllm_ascend.device_allocator.hccl_suspend import (
    HCCL_E_SUSPENDING,
    HCCL_OK,
    HcclSuspendBackend,
    HcclSuspendStats,
    comm_to_int,
    get_group_comm_handles,
    resume_group,
    suspend_group,
)


def _backend_ready(suspend_rc=HCCL_OK, resume_rc=HCCL_OK, support_rc=1):
    backend = HcclSuspendBackend(HcclSuspendStats())
    backend._loaded = True
    backend._suspend_fn = MagicMock(return_value=suspend_rc)
    backend._resume_fn = MagicMock(return_value=resume_rc)
    if support_rc is not None:
        backend._support_resume_fn = MagicMock(return_value=support_rc)
    return backend


def test_comm_to_int_accepts_int_and_wrapped_handle():
    assert comm_to_int(None) == 0
    assert comm_to_int(1234) == 1234
    assert comm_to_int(SimpleNamespace(value=99)) == 99
    assert comm_to_int(object()) == 0


def test_backend_unavailable_when_disabled_by_env(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_SLEEP_HCCL_DISABLE", "1")
    backend = HcclSuspendBackend()
    assert backend.available() is False
    assert backend.suspend(123) is False
    assert backend.resume(123) is False


def test_backend_unavailable_when_capability_flag_is_zero():
    backend = _backend_ready(support_rc=0)
    assert backend.available() is False


def test_backend_suspend_accepts_suspending_code():
    backend = _backend_ready(suspend_rc=HCCL_E_SUSPENDING)
    assert backend.available() is True
    assert backend.suspend(11) is True
    assert backend.stats.suspend_ok == 1
    assert backend.stats.suspend_suspending == 1
    assert backend.stats.suspend_failed == 0


def test_backend_suspend_and_resume_failures_are_counted():
    backend = _backend_ready(suspend_rc=5, resume_rc=7)
    assert backend.suspend(12) is False
    assert backend.stats.suspend_failed == 1
    assert backend.resume(12) is False
    assert backend.stats.resume_failed == 1


def test_get_group_comm_handles_uses_rank_argument():
    backend = MagicMock()
    device_group = MagicMock(rank=3)
    device_group._get_backend.return_value = backend

    with patch.object(backend, "get_hccl_comm", side_effect=[TypeError(), 777]) as getter:
        assert get_group_comm_handles(device_group) == [777]
        assert getter.call_args_list[1].args == (3, False)


def test_get_group_comm_handles_without_backend_returns_empty():
    assert get_group_comm_handles(None) == []
    assert get_group_comm_handles(object()) == []


def test_suspend_group_rolls_back_partial_failure():
    backend = _backend_ready()
    backend.suspend = MagicMock(side_effect=[True, False])
    backend.resume = MagicMock(return_value=True)

    with patch(
        "vllm_ascend.device_allocator.hccl_suspend.get_group_comm_handles",
        return_value=[101, 102],
    ):
        ok, handles = suspend_group(object(), backend)

    assert (ok, handles) == (False, [])
    backend.resume.assert_called_once_with(101)
    assert backend.stats.suspended_handles == []


def test_suspend_group_records_successful_handles():
    backend = _backend_ready()
    with patch(
        "vllm_ascend.device_allocator.hccl_suspend.get_group_comm_handles",
        return_value=[201],
    ):
        ok, handles = suspend_group(object(), backend)

    assert (ok, handles) == (True, [201])
    assert backend.stats.suspended_handles == [201]


def test_resume_group_prefers_process_group_resume():
    backend = _backend_ready()
    backend.resume = MagicMock(return_value=True)
    backend_impl = MagicMock()
    device_group = MagicMock()
    device_group._get_backend.return_value = backend_impl

    with patch(
        "vllm_ascend.device_allocator.hccl_suspend.torch.npu.current_device",
        return_value=2,
    ):
        assert resume_group(device_group, [301], backend) is True

    backend_impl.resume_hccl_comm.assert_called_once_with(2)
    backend.resume.assert_not_called()


def test_resume_group_falls_back_to_raw_resume():
    backend = _backend_ready()
    backend.resume = MagicMock(return_value=True)
    device_group = MagicMock()
    device_group._get_backend.return_value = SimpleNamespace()

    assert resume_group(device_group, [401, 402], backend) is True
    assert backend.resume.call_count == 2


if __name__ == "__main__":
    sys_exit = pytest.main([__file__])
    raise SystemExit(sys_exit)
