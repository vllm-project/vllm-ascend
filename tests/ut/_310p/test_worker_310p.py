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

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from tests.ut.base import TestBase


class TestDetermineAvailableMemory310P(TestBase):
    @staticmethod
    def _make_worker(
        requested_memory: float,
        init_free_memory: int,
        model_memory_usage: int,
        kv_cache_groups: list[KVCacheGroupSpec],
        layout: KVCacheLayout = KVCacheLayout.LBHNC,
    ):
        from vllm_ascend._310p.worker_310p import NPUWorker310

        with patch.object(NPUWorker310, "__init__", lambda worker: None):
            worker = NPUWorker310()

        worker.init_snapshot = SimpleNamespace(free_memory=init_free_memory)
        worker.requested_memory = requested_memory
        worker.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: layout),
            kv_transfer_config=None,
        )
        worker.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3"))
        worker.model_runner = SimpleNamespace(
            model_memory_usage=model_memory_usage,
            profile_run=MagicMock(),
            supports_standardized_shared_kv_backing=False,
            use_sparse=False,
            use_compress=False,
        )
        worker.get_kv_cache_spec = MagicMock(
            return_value={
                layer_name: group.kv_cache_spec for group in kv_cache_groups for layer_name in group.layer_names
            },
        )
        return worker

    @staticmethod
    def _make_attention_groups():
        spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float16)
        return [KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=spec)]

    @staticmethod
    @contextmanager
    def _mock_kv_cache_groups(groups):
        # Provide the planner's groups while exercising the real 310P and
        # parent worker budget scaling with concrete specs and cache layouts.
        with (
            patch("vllm_ascend._310p.worker_310p.get_kv_cache_groups", return_value=groups),
            patch("vllm_ascend.worker.worker.get_kv_cache_groups", return_value=groups),
        ):
            yield

    @staticmethod
    def _make_profile_result(
        free_memory_after: int,
        model_memory_usage: int,
        non_torch_increase: int,
        torch_peak_increase: int,
    ):
        return SimpleNamespace(
            after_profile=SimpleNamespace(free_memory=free_memory_after),
            non_kv_cache_memory=model_memory_usage + non_torch_increase + torch_peak_increase,
            non_torch_increase=non_torch_increase,
            torch_peak_increase=torch_peak_increase,
        )

    @staticmethod
    def _mock_memory_profiling(profile_result):
        context = MagicMock()
        context.__enter__.return_value = profile_result
        context.__exit__.return_value = False
        return MagicMock(return_value=context)

    def test_ep_ignores_pre_existing_process_memory(self):
        total_memory = int(43.24 * GiB_bytes)
        requested_memory = total_memory * 0.5
        model_memory_usage = int(1.43 * GiB_bytes)
        non_torch_increase = int(0.06 * GiB_bytes)
        torch_peak_increase = int(0.07 * GiB_bytes)
        non_kv_cache_memory = model_memory_usage + non_torch_increase + torch_peak_increase
        groups = self._make_attention_groups()
        expected = int((requested_memory - non_kv_cache_memory) // 2)

        # All instances have enough startup free memory for the same requested
        # budget. Other processes must not be charged against that budget again.
        for external_memory in (0, 10 * GiB_bytes, int(19.79 * GiB_bytes)):
            with self.subTest(external_memory=external_memory):
                init_free_memory = total_memory - external_memory
                self.assertGreater(init_free_memory, requested_memory)
                worker = self._make_worker(
                    requested_memory=requested_memory,
                    init_free_memory=init_free_memory,
                    model_memory_usage=model_memory_usage,
                    kv_cache_groups=groups,
                )
                profile_result = self._make_profile_result(
                    free_memory_after=init_free_memory - non_kv_cache_memory,
                    model_memory_usage=model_memory_usage,
                    non_torch_increase=non_torch_increase,
                    torch_peak_increase=torch_peak_increase,
                )
                mock_memory_profiling = self._mock_memory_profiling(profile_result)

                with (
                    patch("vllm_ascend._310p.worker_310p.is_rc_device", return_value=False),
                    patch("vllm_ascend._310p.worker_310p.memory_profiling", mock_memory_profiling),
                    patch(
                        "vllm_ascend._310p.worker_310p.torch.npu.mem_get_info",
                        return_value=(profile_result.after_profile.free_memory, total_memory),
                    ) as mock_mem_get_info,
                    patch(
                        "vllm_ascend._310p.worker_310p.torch.npu.memory_reserved",
                        return_value=model_memory_usage + torch_peak_increase,
                    ) as mock_memory_reserved,
                    self._mock_kv_cache_groups(groups),
                ):
                    result = worker.determine_available_memory()

                self.assertEqual(result, expected)
                self.assertEqual(worker.available_kv_cache_memory_bytes, expected)
                self.assertAlmostEqual(result / GiB_bytes, 10.03, delta=0.01)
                self.assertEqual(worker.non_torch_memory, non_torch_increase)
                self.assertEqual(worker.peak_activation_memory, torch_peak_increase)
                worker.model_runner.profile_run.assert_called_once_with()
                mock_mem_get_info.assert_not_called()
                mock_memory_reserved.assert_not_called()
                mock_memory_profiling.assert_called_once_with(
                    worker.init_snapshot,
                    weights_memory=model_memory_usage,
                )

    def test_rc_keeps_shared_host_memory_calculation(self):
        requested_memory = 32 * GiB_bytes
        init_free_memory = 52 * GiB_bytes
        model_memory_usage = 2 * GiB_bytes
        non_torch_increase = GiB_bytes
        torch_peak_increase = GiB_bytes
        non_kv_cache_memory = model_memory_usage + non_torch_increase + torch_peak_increase
        groups = self._make_attention_groups()

        worker = self._make_worker(
            requested_memory=requested_memory,
            init_free_memory=init_free_memory,
            model_memory_usage=model_memory_usage,
            kv_cache_groups=groups,
        )
        profile_result = self._make_profile_result(
            free_memory_after=init_free_memory - non_kv_cache_memory,
            model_memory_usage=model_memory_usage,
            non_torch_increase=non_torch_increase,
            torch_peak_increase=torch_peak_increase,
        )
        mock_memory_profiling = self._mock_memory_profiling(profile_result)
        host_memory = SimpleNamespace(
            total=64 * GiB_bytes,
            available=profile_result.after_profile.free_memory,
        )

        with (
            patch("vllm_ascend._310p.worker_310p.is_rc_device", return_value=True),
            patch("vllm_ascend._310p.worker_310p.memory_profiling", mock_memory_profiling),
            patch("vllm_ascend._310p.worker_310p.psutil.virtual_memory", return_value=host_memory),
            self._mock_kv_cache_groups(groups),
        ):
            result = worker.determine_available_memory()

        expected = int((requested_memory - (host_memory.total - host_memory.available)) // 2)
        self.assertEqual(result, expected)
        self.assertEqual(result, 8 * GiB_bytes)
        self.assertEqual(worker.available_kv_cache_memory_bytes, expected)

    def test_profile_rejects_free_memory_that_does_not_decrease(self):
        init_free_memory = 24 * GiB_bytes
        groups = self._make_attention_groups()

        for released_memory in (0, GiB_bytes):
            with self.subTest(released_memory=released_memory):
                worker = self._make_worker(
                    requested_memory=16 * GiB_bytes,
                    init_free_memory=init_free_memory,
                    model_memory_usage=2 * GiB_bytes,
                    kv_cache_groups=groups,
                )
                profile_result = self._make_profile_result(
                    free_memory_after=init_free_memory + released_memory,
                    model_memory_usage=2 * GiB_bytes,
                    non_torch_increase=GiB_bytes,
                    torch_peak_increase=GiB_bytes,
                )

                with (
                    patch(
                        "vllm_ascend._310p.worker_310p.memory_profiling",
                        self._mock_memory_profiling(profile_result),
                    ),
                    self.assertRaisesRegex(AssertionError, "Error in memory profiling"),
                ):
                    worker.determine_available_memory()

                worker.model_runner.profile_run.assert_called_once_with()
                worker.get_kv_cache_spec.assert_not_called()

    def test_multi_group_budget_scales_after_workspace_reservation(self):
        full_attention = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float16)
        sliding_attention = SlidingWindowSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float16,
            sliding_window=8,
        )
        self.assertEqual(full_attention.page_size_bytes, sliding_attention.page_size_bytes)
        groups = [
            KVCacheGroupSpec(layer_names=["full_attn.0", "full_attn.1"], kv_cache_spec=full_attention),
            KVCacheGroupSpec(layer_names=["sliding_attn"], kv_cache_spec=sliding_attention),
        ]
        requested_memory = 32 * GiB_bytes
        init_free_memory = 48 * GiB_bytes
        profile_result = self._make_profile_result(
            free_memory_after=44 * GiB_bytes,
            model_memory_usage=2 * GiB_bytes,
            non_torch_increase=GiB_bytes,
            torch_peak_increase=GiB_bytes,
        )
        host_memory = SimpleNamespace(total=64 * GiB_bytes, available=profile_result.after_profile.free_memory)

        for is_rc in (False, True):
            with self.subTest(is_rc=is_rc):
                worker = self._make_worker(
                    requested_memory=requested_memory,
                    init_free_memory=init_free_memory,
                    model_memory_usage=2 * GiB_bytes,
                    kv_cache_groups=groups,
                )
                with (
                    patch("vllm_ascend._310p.worker_310p.is_rc_device", return_value=is_rc),
                    patch(
                        "vllm_ascend._310p.worker_310p.memory_profiling",
                        self._mock_memory_profiling(profile_result),
                    ),
                    patch("vllm_ascend._310p.worker_310p.psutil.virtual_memory", return_value=host_memory),
                    self._mock_kv_cache_groups(groups),
                ):
                    result = worker.determine_available_memory()

                occupied_memory = (
                    host_memory.total - host_memory.available if is_rc else profile_result.non_kv_cache_memory
                )
                workspace_budget = (requested_memory - occupied_memory) // 2
                self.assertEqual(result, int(workspace_budget * (2 / 3)))
                self.assertEqual(worker.available_kv_cache_memory_bytes, result)
                # The resulting block count must fit all three private layer
                # pages within the budget remaining after workspace reservation.
                num_blocks = result // (2 * full_attention.page_size_bytes)
                self.assertLessEqual(num_blocks * 3 * full_attention.page_size_bytes, workspace_budget)

    def test_shared_mamba_slots_do_not_scale_budget_again(self):
        attention = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float16)
        mamba = MambaSpec(block_size=4, shapes=((16,),), dtypes=(torch.float32,))
        self.assertEqual(attention.page_size_bytes, mamba.page_size_bytes)
        groups = [
            KVCacheGroupSpec(["L3", "L7"], attention),
            KVCacheGroupSpec(["L0", "L4"], mamba),
            KVCacheGroupSpec(["L1", "L5"], mamba),
            KVCacheGroupSpec(["L2", "L6"], mamba),
        ]
        requested_memory = 32 * GiB_bytes
        profile_result = self._make_profile_result(
            free_memory_after=44 * GiB_bytes,
            model_memory_usage=2 * GiB_bytes,
            non_torch_increase=GiB_bytes,
            torch_peak_increase=GiB_bytes,
        )

        for layout in (KVCacheLayout.LBHNC, KVCacheLayout.LBNHC):
            with self.subTest(layout=layout):
                worker = self._make_worker(
                    requested_memory=requested_memory,
                    init_free_memory=48 * GiB_bytes,
                    model_memory_usage=2 * GiB_bytes,
                    kv_cache_groups=groups,
                    layout=layout,
                )
                with (
                    patch("vllm_ascend._310p.worker_310p.is_rc_device", return_value=False),
                    patch(
                        "vllm_ascend._310p.worker_310p.memory_profiling",
                        self._mock_memory_profiling(profile_result),
                    ),
                    self._mock_kv_cache_groups(groups),
                ):
                    result = worker.determine_available_memory()

                expected = (requested_memory - profile_result.non_kv_cache_memory) // 2
                self.assertEqual(result, expected)
                self.assertEqual(worker.available_kv_cache_memory_bytes, expected)
