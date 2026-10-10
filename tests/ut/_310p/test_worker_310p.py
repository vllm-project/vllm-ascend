#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.utils.mem_constants import GiB_bytes

from tests.ut.base import TestBase


class TestDetermineAvailableMemory310P(TestBase):
    @staticmethod
    def _make_worker(requested_memory: int, init_free_memory: int, model_memory_usage: int):
        from vllm_ascend._310p.worker_310p import NPUWorker310

        worker = object.__new__(NPUWorker310)
        worker.init_snapshot = SimpleNamespace(free_memory=init_free_memory)
        worker.requested_memory = requested_memory
        worker.model_runner = MagicMock()
        worker.model_runner.model_memory_usage = model_memory_usage
        worker.get_kv_cache_spec = MagicMock(return_value=None)
        return worker

    @staticmethod
    def _profile_result(
        free_memory_after: int,
        non_kv_cache_memory: int,
        non_torch_increase: int,
        torch_peak_increase: int,
    ):
        return SimpleNamespace(
            after_profile=SimpleNamespace(free_memory=free_memory_after),
            non_kv_cache_memory=non_kv_cache_memory,
            non_torch_increase=non_torch_increase,
            torch_peak_increase=torch_peak_increase,
        )

    @staticmethod
    def _profiling_context(profile_result):
        context = MagicMock()
        context.__enter__.return_value = profile_result
        context.__exit__.return_value = False
        return context

    def test_ep_does_not_charge_device_global_memory(self):
        total_memory = int(43.24 * GiB_bytes)
        init_free_memory = int(23.45 * GiB_bytes)
        requested_memory = int(total_memory * 0.5)
        model_memory_usage = int(1.43 * GiB_bytes)
        non_kv_cache_memory = int(1.71 * GiB_bytes)
        profile_result = self._profile_result(
            free_memory_after=init_free_memory - non_kv_cache_memory,
            non_kv_cache_memory=non_kv_cache_memory,
            non_torch_increase=int(0.06 * GiB_bytes),
            torch_peak_increase=int(0.07 * GiB_bytes),
        )
        worker = self._make_worker(requested_memory, init_free_memory, model_memory_usage)

        with (
            patch("vllm_ascend._310p.worker_310p.is_rc_device", return_value=False),
            patch(
                "vllm_ascend._310p.worker_310p.memory_profiling",
                return_value=self._profiling_context(profile_result),
            ) as mock_memory_profiling,
            patch(
                "vllm_ascend._310p.worker_310p.torch.npu.mem_get_info",
                side_effect=AssertionError("device-global memory must not be queried"),
            ),
            patch(
                "vllm_ascend._310p.worker_310p.torch.npu.memory_reserved",
                side_effect=AssertionError("device-global memory must not be queried"),
            ),
        ):
            result = worker.determine_available_memory()

        expected = (requested_memory - non_kv_cache_memory) // 2
        self.assertEqual(result, expected)
        self.assertAlmostEqual(result / GiB_bytes, 9.95, delta=0.1)
        self.assertEqual(worker.non_torch_memory, profile_result.non_torch_increase)
        self.assertEqual(worker.peak_activation_memory, profile_result.torch_peak_increase)
        worker.model_runner.profile_run.assert_called_once_with()
        worker.get_kv_cache_spec.assert_called_once_with()
        mock_memory_profiling.assert_called_once_with(
            worker.init_snapshot,
            weights_memory=model_memory_usage,
        )

    def test_rc_keeps_shared_host_memory_calculation(self):
        requested_memory = 32 * GiB_bytes
        init_free_memory = 48 * GiB_bytes
        model_memory_usage = int(1.43 * GiB_bytes)
        non_kv_cache_memory = int(1.71 * GiB_bytes)
        profile_result = self._profile_result(
            free_memory_after=init_free_memory - non_kv_cache_memory,
            non_kv_cache_memory=non_kv_cache_memory,
            non_torch_increase=int(0.06 * GiB_bytes),
            torch_peak_increase=int(0.07 * GiB_bytes),
        )
        worker = self._make_worker(requested_memory, init_free_memory, model_memory_usage)
        host_memory = SimpleNamespace(total=64 * GiB_bytes, available=48 * GiB_bytes)

        with (
            patch("vllm_ascend._310p.worker_310p.is_rc_device", return_value=True),
            patch(
                "vllm_ascend._310p.worker_310p.memory_profiling",
                return_value=self._profiling_context(profile_result),
            ),
            patch("vllm_ascend._310p.worker_310p.psutil.virtual_memory", return_value=host_memory),
        ):
            result = worker.determine_available_memory()

        expected = (requested_memory - (host_memory.total - host_memory.available)) // 2
        self.assertEqual(result, expected)
        self.assertEqual(result, 8 * GiB_bytes)
        worker.get_kv_cache_spec.assert_called_once_with()
