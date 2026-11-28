# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from vllm.utils.mem_constants import GiB_bytes

from vllm_ascend._310p import worker_310p


class Test310PMemoryBudget(TestCase):
    def setUp(self):
        self.worker = object.__new__(worker_310p.NPUWorker310)
        self.worker.model_runner = MagicMock()
        self.worker.model_runner.model_memory_usage = 2 * GiB_bytes
        self.worker.requested_memory = 20 * GiB_bytes
        self.worker.init_snapshot = self.snapshot(24, 3, 6)
        self.worker._scale_kv_cache_memory_for_multi_group = lambda budget: budget
        self.profile = SimpleNamespace(
            after_profile=self.snapshot(24, 5, 6),
            non_torch_increase=0,
            torch_peak_increase=3 * GiB_bytes,
            weights_memory=2 * GiB_bytes,
            transient_peak_headroom=3 * GiB_bytes,
            non_kv_cache_memory=3 * GiB_bytes,
        )
        profiling = patch.object(worker_310p, "memory_profiling", return_value=nullcontext(self.profile))
        profiling.start()
        self.addCleanup(profiling.stop)

    @staticmethod
    def snapshot(free, allocated, reserved):
        return SimpleNamespace(
            total_memory=32 * GiB_bytes,
            free_memory=free * GiB_bytes,
            torch_allocated=allocated * GiB_bytes,
            torch_memory=reserved * GiB_bytes,
        )

    def test_ep_accounts_for_retained_and_transient_memory(self):
        with patch.object(worker_310p, "is_rc_device", return_value=False):
            self.assertEqual(self.worker.determine_available_memory(), (20 - 2 - 3) * GiB_bytes // 2)
            self.profile.after_profile = self.snapshot(24, 5, 10)
            with self.assertRaisesRegex(ValueError, "allocator statistics"):
                self.worker.determine_available_memory()
            self.profile.after_profile = self.snapshot(25, 5, 6)
            with self.assertRaisesRegex(ValueError, "Free NPU memory increased"):
                self.worker.determine_available_memory()
            self.profile.after_profile = self.snapshot(22, 5, 6)
            self.profile.non_kv_cache_memory = 5 * GiB_bytes
            self.assertEqual(self.worker.determine_available_memory(), (20 - 5) * GiB_bytes // 2)

    def test_rc_compares_host_available_memory_and_budgets_from_host_usage(self):
        self.profile.after_profile = self.snapshot(28, 5, 6)
        with (
            patch.object(worker_310p, "is_rc_device", return_value=True),
            patch.object(worker_310p.psutil, "virtual_memory") as host_memory,
        ):
            for available in (24, 22):
                with self.subTest(host_available=available):
                    host_memory.return_value = SimpleNamespace(total=32 * GiB_bytes, available=available * GiB_bytes)
                    self.assertEqual(self.worker.determine_available_memory(), (20 - (32 - available)) * GiB_bytes // 2)
            host_memory.return_value = SimpleNamespace(total=32 * GiB_bytes, available=25 * GiB_bytes)
            with self.assertRaisesRegex(ValueError, "Free host memory increased"):
                self.worker.determine_available_memory()
