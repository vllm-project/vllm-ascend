"""Unit tests for the V2 (attn_utils) DSA PD mooncake offload allocation path.

The V1 port is covered by tests/ut/kv_offload/test_dsa_pd_mooncake.py; this
file covers the V2 function-style port in vllm_ascend/worker/v2/attn_utils.py:

* swapped allocations are aligned and recorded for final-view registration;
* pending offloaded ranges are bound to the final reshaped views and cleared;
* the fused_overlap registry reset hook runs before allocation.
"""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

import vllm_ascend.worker.v2.attn_utils as attn_utils
from vllm_ascend.kv_offload import dsa_pd_mooncake
from vllm_ascend.kv_offload.dsa_pd_mooncake import (
    clear_dsa_pd_mooncake_kv_buffer_registries_for_testing,
    should_use_dsa_pd_mooncake_cpu_kv,
)


class TestV2MooncakeAllocatorRouting(TestCase):
    def setUp(self):
        clear_dsa_pd_mooncake_kv_buffer_registries_for_testing()
        attn_utils._dsa_pd_mooncake_pending_offloaded_ranges.clear()
        self.fake_tensors = []

    def tearDown(self):
        attn_utils._dsa_pd_mooncake_pending_offloaded_ranges.clear()

    def _fake_swapped(self, shape, *, dtype):
        numel = shape[0]
        tensor = torch.zeros(numel, dtype=dtype)
        # Keep the CPU storage alive while the allocation helper aligns views.
        self.fake_tensors.append(tensor)
        return tensor

    def test_swapped_allocation_records_aligned_range(self):
        with patch.object(
            attn_utils,
            "empty_mooncake_cpu_kv_memory",
            side_effect=self._fake_swapped,
        ):
            k_tensor = attn_utils._allocate_dsa_pd_mooncake_kv_tensor(48, 4096)
            v_tensor = attn_utils._allocate_dsa_pd_mooncake_kv_tensor(32, 4096)

        self.assertEqual(k_tensor.data_ptr() % 4096, 0)
        self.assertEqual(v_tensor.data_ptr() % 4096, 0)
        self.assertEqual(k_tensor.numel(), 48)
        self.assertEqual(v_tensor.numel(), 32)
        self.assertEqual(
            attn_utils._dsa_pd_mooncake_pending_offloaded_ranges,
            [(int(k_tensor.data_ptr()), 48), (int(v_tensor.data_ptr()), 32)],
        )

    def test_final_views_bind_and_clear_pending(self):
        vllm_config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="auto"))
        kv_config = SimpleNamespace(kv_cache_groups=[])
        ascend_config = SimpleNamespace(dsa_sparse_attention_config=SimpleNamespace(mode="baseline"))

        def allocate(*args, **kwargs):
            raw = attn_utils._allocate_dsa_pd_mooncake_kv_tensor(64, 4096)
            return {"model.layers.0": raw}

        def reshape(**kwargs):
            raw = kwargs["kv_cache_raw_tensors"]["model.layers.0"]
            return {"model.layers.0": (raw.view(2, 32),)}

        with (
            patch("vllm_ascend.ascend_config.get_ascend_config", return_value=ascend_config),
            patch.object(attn_utils, "get_current_vllm_config", return_value=vllm_config),
            patch.object(attn_utils, "get_layers_from_vllm_config", return_value={}),
            patch.object(attn_utils, "empty_mooncake_cpu_kv_memory", side_effect=self._fake_swapped),
            patch.object(attn_utils, "_allocate_kv_cache", side_effect=allocate),
            patch.object(attn_utils, "_reshape_kv_cache_v2", side_effect=reshape),
        ):
            caches = attn_utils.allocate_kv_cache_main(kv_config, torch.device("cpu"), None, [])

        view = caches["model.layers.0"][0]
        self.assertEqual(tuple(view.shape), (2, 32))
        resolved = dsa_pd_mooncake.get_dsa_pd_mooncake_offloaded_kv_tensor(view.data_ptr(), 64)
        self.assertIsNotNone(resolved)
        tensor, offset = resolved
        self.assertIs(tensor, view)
        self.assertEqual(offset, 0)
        self.assertEqual(attn_utils._dsa_pd_mooncake_pending_offloaded_ranges, [])

    def test_unmatched_pending_range_raises(self):
        attn_utils._dsa_pd_mooncake_pending_offloaded_ranges.append((1 << 40, 4096))
        other = torch.zeros(4096, dtype=torch.int8)
        with self.assertRaises(RuntimeError):
            attn_utils._register_dsa_pd_mooncake_final_kv_views({"x": (other,)})
        attn_utils._dsa_pd_mooncake_pending_offloaded_ranges.clear()

    def test_predicate_reads_ascend_config(self):
        vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                is_kv_consumer=True,
                kv_connector="MooncakeLayerwiseConnector",
            )
        )
        ascend_config = SimpleNamespace(
            dsa_sparse_attention_config=SimpleNamespace(enable_cpu_kv_store=True, mode="fused_overlap")
        )
        with (
            patch.object(
                attn_utils,
                "should_use_dsa_pd_mooncake_cpu_kv",
                wraps=should_use_dsa_pd_mooncake_cpu_kv,
            ) as pred,
            patch(
                "vllm_ascend.ascend_config.get_ascend_config",
                return_value=ascend_config,
            ),
        ):
            ok = attn_utils._use_dsa_pd_mooncake_cpu_kv(vllm_config, is_sparse_kv_cache=True)
        self.assertTrue(pred.called)
        self.assertIsInstance(ok, bool)


class TestV2RegistryResetHook(TestCase):
    def test_fused_overlap_resets_registry(self):
        clear_dsa_pd_mooncake_kv_buffer_registries_for_testing()
        stale = torch.zeros(64, dtype=torch.int8)
        dsa_pd_mooncake.register_dsa_pd_mooncake_offloaded_kv_tensor(stale)
        attn_utils._dsa_pd_mooncake_pending_offloaded_ranges.append((stale.data_ptr(), 64))
        self.addCleanup(clear_dsa_pd_mooncake_kv_buffer_registries_for_testing)
        self.addCleanup(attn_utils._dsa_pd_mooncake_pending_offloaded_ranges.clear)

        def fake_alloc(kv_cache_config, *, shared_layers, device):
            self.assertIsNone(dsa_pd_mooncake.get_dsa_pd_mooncake_offloaded_kv_tensor(stale.data_ptr(), 64))
            self.assertEqual(attn_utils._dsa_pd_mooncake_pending_offloaded_ranges, [])
            return {}

        vllm_config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="auto"))
        kv_config = SimpleNamespace(kv_cache_groups=[])
        ascend_config = SimpleNamespace(dsa_sparse_attention_config=SimpleNamespace(mode="fused_overlap"))
        with (
            patch(
                "vllm_ascend.ascend_config.get_ascend_config",
                return_value=ascend_config,
            ),
            patch.object(
                attn_utils,
                "reset_dsa_pd_mooncake_kv_registries",
                wraps=dsa_pd_mooncake.reset_dsa_pd_mooncake_kv_registries,
            ) as reset,
            patch.object(
                attn_utils,
                "_allocate_kv_cache",
                side_effect=fake_alloc,
            ),
            patch.object(attn_utils, "get_current_vllm_config", return_value=vllm_config),
            patch.object(attn_utils, "get_layers_from_vllm_config", return_value={}),
            patch.object(attn_utils, "_reshape_kv_cache_v2", return_value={}),
        ):
            self.assertEqual(attn_utils.allocate_kv_cache_main(kv_config, torch.device("cpu"), None, []), {})
        reset.assert_called_once_with()
