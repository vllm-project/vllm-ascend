# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.utils.mem_constants import GiB_bytes


def _make_worker(requested_memory: int, init_free_memory: int, model_memory_usage: int):
    from vllm_ascend._310p.worker_310p import NPUWorker310

    worker = object.__new__(NPUWorker310)
    worker.init_snapshot = SimpleNamespace(free_memory=init_free_memory)
    worker.requested_memory = requested_memory
    worker.model_runner = MagicMock()
    worker.model_runner.model_memory_usage = model_memory_usage
    worker.vllm_config = SimpleNamespace(cache_config=SimpleNamespace())
    worker.get_kv_cache_spec = MagicMock(return_value=None)
    return worker


def _profile_result(free_memory_after: int, non_kv_cache_memory: int):
    return SimpleNamespace(
        after_profile=SimpleNamespace(free_memory=free_memory_after),
        non_kv_cache_memory=non_kv_cache_memory,
        non_torch_increase=64 * 1024**2,
        torch_peak_increase=128 * 1024**2,
    )


def _profiling_context(profile_result):
    context = MagicMock()
    context.__enter__.return_value = profile_result
    context.__exit__.return_value = False
    return context


@pytest.mark.parametrize(
    ("is_rc", "requested_memory", "host_total", "host_available", "non_kv_cache_memory", "raw_budget"),
    [
        pytest.param(
            False,
            24 * GiB_bytes,
            64 * GiB_bytes,
            48 * GiB_bytes,
            2 * GiB_bytes,
            11 * GiB_bytes,
            id="non-rc-uses-worker-profile-only",
        ),
        pytest.param(
            True,
            32 * GiB_bytes,
            64 * GiB_bytes,
            48 * GiB_bytes,
            2 * GiB_bytes,
            8 * GiB_bytes,
            id="rc-preserves-shared-host-accounting",
        ),
    ],
)
def test_memory_budget_avoids_device_global_hbm_and_preserves_scaling(
    is_rc: bool,
    requested_memory: int,
    host_total: int,
    host_available: int,
    non_kv_cache_memory: int,
    raw_budget: int,
):
    init_free_memory = 28 * GiB_bytes
    model_memory_usage = 2 * GiB_bytes
    profile_result = _profile_result(
        free_memory_after=init_free_memory - non_kv_cache_memory,
        non_kv_cache_memory=non_kv_cache_memory,
    )
    worker = _make_worker(requested_memory, init_free_memory, model_memory_usage)
    scaled_budget = raw_budget - GiB_bytes
    worker._scale_kv_cache_memory_for_multi_group = MagicMock(return_value=scaled_budget)
    host_memory = SimpleNamespace(total=host_total, available=host_available)

    with (
        patch("vllm_ascend._310p.worker_310p.is_rc_device", return_value=is_rc),
        patch(
            "vllm_ascend._310p.worker_310p.memory_profiling",
            return_value=_profiling_context(profile_result),
        ) as mock_memory_profiling,
        patch(
            "vllm_ascend._310p.worker_310p.psutil.virtual_memory",
            return_value=host_memory,
        ) as mock_virtual_memory,
        patch(
            "vllm_ascend._310p.worker_310p.torch.npu.mem_get_info",
            side_effect=AssertionError("device-global memory must not be queried"),
        ) as mock_mem_get_info,
        patch(
            "vllm_ascend._310p.worker_310p.torch.npu.memory_reserved",
            side_effect=AssertionError("device-global memory must not be queried"),
        ) as mock_memory_reserved,
    ):
        result = worker.determine_available_memory()

    assert result == scaled_budget
    assert worker.non_torch_memory == profile_result.non_torch_increase
    assert worker.peak_activation_memory == profile_result.torch_peak_increase
    worker.model_runner.profile_run.assert_called_once_with()
    worker._scale_kv_cache_memory_for_multi_group.assert_called_once_with(raw_budget)
    mock_memory_profiling.assert_called_once_with(
        worker.init_snapshot,
        weights_memory=model_memory_usage,
    )
    mock_mem_get_info.assert_not_called()
    mock_memory_reserved.assert_not_called()
    if is_rc:
        mock_virtual_memory.assert_called_once_with()
    else:
        mock_virtual_memory.assert_not_called()
