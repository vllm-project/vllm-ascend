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

from unittest.mock import MagicMock, patch

from vllm.utils.mem_constants import GiB_bytes

from tests.ut.base import TestBase


class TestDetermineAvailableMemoryMultiInstance(TestBase):
    """Tests for determine_available_memory() focusing on the multi-instance
    OOM regression (PR #7427)."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _make_worker(
        self,
        requested_memory: int,
        init_free_memory: int,
        init_total_memory: int,
        model_memory_usage: int | None = None,
    ):
        """Return a minimally-configured NPUWorker mock with memory state set."""
        from vllm_ascend.worker.worker import NPUWorker

        if model_memory_usage is None:
            model_memory_usage = int(0.5 * GiB_bytes)  # Qwen3-0.6B ~0.5 GiB

        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()

        worker.model_runner = MagicMock()
        worker.model_runner.model_memory_usage = model_memory_usage

        mock_snapshot = MagicMock()
        mock_snapshot.free_memory = init_free_memory
        mock_snapshot.total_memory = init_total_memory
        worker.init_snapshot = mock_snapshot

        worker.requested_memory = requested_memory
        return worker

    @staticmethod
    def _make_profile_result(free_memory_after: int, non_kv_cache_memory: int):
        """Return a mock profile_result compatible with memory_profiling output."""
        profile_result = MagicMock()
        profile_result.after_profile.free_memory = free_memory_after
        profile_result.non_kv_cache_memory = non_kv_cache_memory
        return profile_result

    @staticmethod
    def _patch_memory_profiling(profile_result):
        """Return a mock for `memory_profiling` that yields *profile_result*."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=profile_result)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_profiling = MagicMock(return_value=mock_ctx)
        return patch("vllm_ascend.worker.worker.memory_profiling", mock_profiling)

    # ------------------------------------------------------------------ #
    # Tests
    # ------------------------------------------------------------------ #

    @patch("vllm_ascend.worker.worker.logger")
    def test_single_instance_positive_kv_cache(self, mock_logger):
        """Baseline: single instance on an empty card yields positive KV cache."""
        total = int(64 * GiB_bytes)
        gpu_util = 0.9
        requested_memory = int(total * gpu_util)   # 57.6 GiB
        init_free = int(62 * GiB_bytes)            # almost all free
        non_kv_cache = int(0.5 * GiB_bytes)        # Qwen3-0.6B weights

        worker = self._make_worker(requested_memory, init_free, total)
        profile_result = self._make_profile_result(
            free_memory_after=init_free - non_kv_cache,
            non_kv_cache_memory=non_kv_cache,
        )

        with self._patch_memory_profiling(profile_result):
            result = worker.determine_available_memory()

        expected = requested_memory - non_kv_cache
        self.assertEqual(result, expected)
        self.assertGreater(result, 0)

    @patch("vllm_ascend.worker.worker.logger")
    def test_second_instance_on_same_card_positive_kv_cache(self, mock_logger):
        """
        Regression test for PR #7427.

        Scenario (64 GiB Ascend 910B card, two Qwen3-0.6B instances,
        gpu_memory_utilization=0.4):

          ┌───────────────────────────────────────────────────────────────┐
          │ Card total:  64 GiB                                           │
          │ Instance 1:  requested_memory = 64 * 0.4 = 25.6 GiB (in use) │
          │ Instance 2 start: init_snapshot.free_memory ≈ 38.4 GiB       │
          │ Instance 2: requested_memory = 25.6 GiB                      │
          │ Profiling (fixed):  non_kv_cache_memory = 0.5 GiB (weights)  │
          │ available = 25.6 - 0.5 = 25.1 GiB  → must be > 0  ✓         │
          └───────────────────────────────────────────────────────────────┘

        Before the fix, non_kv_cache_memory was inflated to include first
        instance memory (~25.6 GiB), yielding available ≈ -1.32 GiB (OOM).
        """
        total = int(64 * GiB_bytes)
        gpu_util = 0.4
        requested_memory = int(total * gpu_util)          # 25.6 GiB

        # First instance already occupies its full requested_memory slice
        first_instance_used = requested_memory            # 25.6 GiB
        init_free = total - first_instance_used           # ~38.4 GiB

        # After the fix: profiling correctly reports only the second
        # instance's own model weights, not the first instance's memory.
        non_kv_cache = int(0.5 * GiB_bytes)              # Qwen3-0.6B weights

        worker = self._make_worker(requested_memory, init_free, total)
        profile_result = self._make_profile_result(
            free_memory_after=init_free - non_kv_cache,
            non_kv_cache_memory=non_kv_cache,
        )

        with self._patch_memory_profiling(profile_result):
            result = worker.determine_available_memory()

        self.assertGreater(
            result, 0,
            "Second instance must have positive KV cache memory. "
            "A non-positive value means the multi-instance OOM bug "
            "(PR #7427) has regressed.",
        )
        expected = requested_memory - non_kv_cache
        self.assertEqual(result, expected)
        # Verify model_runner.profile_run() was called during profiling
        worker.model_runner.profile_run.assert_called_once()

    @patch("vllm_ascend.worker.worker.logger")
    def test_second_instance_buggy_non_kv_cache_gives_negative(self, mock_logger):
        """
        Documents the *pre-fix* buggy behaviour that PR #7427 addresses.

        When non_kv_cache_memory is erroneously inflated to include memory
        already held by the first instance (~25.6 GiB extra), the formula
            available = requested_memory - non_kv_cache_memory
        yields a negative value, confirming why the fix was necessary.

        This test is intentionally asserting the *negative* outcome to
        document the regressed state; it is NOT testing the fix itself.
        """
        total = int(64 * GiB_bytes)
        gpu_util = 0.4
        requested_memory = int(total * gpu_util)   # 25.6 GiB

        first_instance_used = requested_memory     # 25.6 GiB
        init_free = total - first_instance_used    # ~38.4 GiB

        # Buggy: non_kv_cache_memory = first-instance memory + second-instance weights
        buggy_non_kv_cache = int((25.6 + 0.5) * GiB_bytes)  # ~26.1 GiB

        worker = self._make_worker(requested_memory, init_free, total)
        profile_result = self._make_profile_result(
            # free_memory decreased only by the actual new allocation (weights)
            free_memory_after=init_free - int(0.5 * GiB_bytes),
            non_kv_cache_memory=buggy_non_kv_cache,
        )

        with self._patch_memory_profiling(profile_result):
            result = worker.determine_available_memory()

        # Pre-fix: 25.6 GiB - 26.1 GiB = -0.5 GiB  (negative → OOM)
        self.assertLess(
            result, 0,
            "With the pre-fix (buggy) non_kv_cache_memory the result must be "
            "negative; this documents the OOM regression that PR #7427 fixed.",
        )

    @patch("vllm_ascend.worker.worker.logger")
    def test_assert_raises_when_free_memory_increases_after_profile(self, mock_logger):
        """
        determine_available_memory() must raise AssertionError when free memory
        after profiling is greater than before (external process released memory
        during profiling, invalidating the measurement).
        """
        total = int(64 * GiB_bytes)
        requested_memory = int(total * 0.9)
        init_free = int(60 * GiB_bytes)

        worker = self._make_worker(requested_memory, init_free, total)
        # Abnormal: free memory increased after profiling
        profile_result = self._make_profile_result(
            free_memory_after=init_free + int(1 * GiB_bytes),  # went UP
            non_kv_cache_memory=int(0.5 * GiB_bytes),
        )

        with self._patch_memory_profiling(profile_result):
            with self.assertRaises(AssertionError) as ctx:
                worker.determine_available_memory()

        self.assertIn("Error in memory profiling", str(ctx.exception))

    @patch("vllm_ascend.worker.worker.logger")
    def test_second_instance_tight_memory_still_positive(self, mock_logger):
        """
        Edge case: card is almost full when second instance starts.

        Even with very little free memory left, as long as requested_memory >
        non_kv_cache_memory (i.e. there is room for at least some KV blocks),
        the result must be positive.
        """
        total = int(32 * GiB_bytes)       # smaller card (e.g. 910B1)
        gpu_util = 0.3
        requested_memory = int(total * gpu_util)   # 9.6 GiB

        # First instance has consumed most of its requested slice
        first_instance_used = requested_memory     # 9.6 GiB
        init_free = total - first_instance_used    # 22.4 GiB

        non_kv_cache = int(0.5 * GiB_bytes)  # Qwen3-0.6B

        worker = self._make_worker(requested_memory, init_free, total)
        profile_result = self._make_profile_result(
            free_memory_after=init_free - non_kv_cache,
            non_kv_cache_memory=non_kv_cache,
        )

        with self._patch_memory_profiling(profile_result):
            result = worker.determine_available_memory()

        self.assertGreater(result, 0)
        self.assertEqual(result, requested_memory - non_kv_cache)
