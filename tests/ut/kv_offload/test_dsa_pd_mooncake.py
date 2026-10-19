"""Unit tests for the DSA PD mooncake offload address book.

Ported from the glm52 branch (commit chain 1e9ebb64..8956d8f3,
tests/ut/kv_offload/test_dsa_pd_mooncake.py), trimmed to the ported scope:
the two AscendSFAImpl tests covering the fused-overlap compute path
(sfa_v1.py) were dropped because that compute-side code is not part of
this port.
"""

import gc
import sys
import weakref
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CUDAGraphMode
from vllm_ascend.kv_offload.dsa_pd_mooncake import (
    clear_dsa_pd_mooncake_kv_buffer_registries_for_testing,
    empty_mooncake_cpu_kv_memory,
    get_dsa_pd_mooncake_kv_register_location,
    get_dsa_pd_mooncake_offloaded_kv_tensor,
    kv_transfer_uses_mooncake,
    register_dsa_pd_mooncake_cpu_kv_buffer,
    register_dsa_pd_mooncake_offloaded_kv_tensor,
    should_use_dsa_pd_mooncake_cpu_kv,
    should_use_dsa_pd_mooncake_long_context_kv,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestDSAPDMooncakeCPUKV(TestCase):
    def setUp(self):
        clear_dsa_pd_mooncake_kv_buffer_registries_for_testing()

    def test_detects_nested_mooncake_connector(self):
        kv_transfer_config = SimpleNamespace(
            kv_connector="",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "P2pNcclConnector"},
                    {"kv_connector": "MooncakeLayerwiseConnector"},
                ],
            },
        )

        self.assertTrue(kv_transfer_uses_mooncake(kv_transfer_config))

    def test_long_context_kv_uses_dsa_additional_config(self):
        kv_transfer_config = SimpleNamespace(
            is_kv_consumer=True,
            kv_connector="MooncakeLayerwiseConnector",
            kv_connector_extra_config={},
        )
        vllm_config = SimpleNamespace(
            additional_config={
                "dsa_sparse_attention_config": {
                    "mode": "fused_overlap",
                    "enable_cpu_kv_store": True,
                },
            },
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(index_topk=2048),
            ),
            kv_transfer_config=kv_transfer_config,
        )

        self.assertTrue(should_use_dsa_pd_mooncake_long_context_kv(vllm_config))

        vllm_config.additional_config["dsa_sparse_attention_config"]["mode"] = "baseline"
        self.assertFalse(should_use_dsa_pd_mooncake_long_context_kv(vllm_config))

        vllm_config.additional_config["dsa_sparse_attention_config"]["mode"] = "fused_overlap"
        kv_transfer_config.is_kv_consumer = False
        self.assertFalse(should_use_dsa_pd_mooncake_long_context_kv(vllm_config))

    def test_cpu_kv_requires_decode_consumer_sparse_fused_mooncake(self):
        kv_transfer_config = SimpleNamespace(
            is_kv_consumer=True,
            kv_connector="MooncakeConnector",
            kv_connector_extra_config={},
        )
        vllm_config = SimpleNamespace(
            kv_transfer_config=kv_transfer_config,
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        )

        self.assertTrue(
            should_use_dsa_pd_mooncake_cpu_kv(
                vllm_config,
                use_sparse=True,
                is_kv_consumer=True,
                enable_cpu_kv_store=True,
                dsa_sparse_attention_mode="fused_overlap",
            )
        )
        self.assertFalse(
            should_use_dsa_pd_mooncake_cpu_kv(
                vllm_config,
                use_sparse=True,
                is_kv_consumer=True,
                enable_cpu_kv_store=False,
                dsa_sparse_attention_mode="fused_overlap",
            )
        )
        self.assertFalse(
            should_use_dsa_pd_mooncake_cpu_kv(
                vllm_config,
                use_sparse=True,
                is_kv_consumer=True,
                enable_cpu_kv_store=True,
                dsa_sparse_attention_mode="baseline",
            )
        )

    def test_cpu_kv_predicate_allows_graph_mode(self):
        kv_transfer_config = SimpleNamespace(
            is_kv_consumer=True,
            kv_connector="MooncakeConnector",
            kv_connector_extra_config={},
        )
        vllm_config = SimpleNamespace(
            kv_transfer_config=kv_transfer_config,
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        )

        self.assertTrue(
            should_use_dsa_pd_mooncake_cpu_kv(
                vllm_config,
                use_sparse=True,
                is_kv_consumer=True,
                enable_cpu_kv_store=True,
                dsa_sparse_attention_mode="fused_overlap",
            )
        )


class TestDSAPDMooncakeCPUKVRuntimeCaches(TestCase):
    def setUp(self):
        clear_dsa_pd_mooncake_kv_buffer_registries_for_testing()

    def test_registered_kv_buffer_is_marked_as_cpu_location(self):
        kv_tensor = torch.zeros((2, 4, 1, 3), dtype=torch.float32)

        register_dsa_pd_mooncake_cpu_kv_buffer(
            kv_tensor.data_ptr(),
            kv_tensor.element_size() * kv_tensor.numel(),
        )

        self.assertEqual(
            get_dsa_pd_mooncake_kv_register_location(
                kv_tensor.data_ptr(),
                kv_tensor.element_size() * kv_tensor.numel(),
            ),
            "cpu",
        )

    def test_registry_does_not_keep_obsolete_offloaded_tensor_alive(self):
        kv_tensor = torch.empty((128,), dtype=torch.int8)
        ptr = kv_tensor.data_ptr()
        size = kv_tensor.numel()
        tensor_ref = weakref.ref(kv_tensor)
        register_dsa_pd_mooncake_offloaded_kv_tensor(kv_tensor)

        del kv_tensor
        gc.collect()

        self.assertIsNone(tensor_ref())
        self.assertIsNone(get_dsa_pd_mooncake_offloaded_kv_tensor(ptr, size))

    def test_mooncake_cpu_kv_memory_uses_swapped_allocator(self):
        fake_torch_npu = SimpleNamespace(
            empty_with_swapped_memory=MagicMock(return_value=torch.empty((128,), dtype=torch.int8))
        )
        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            kv_tensor = empty_mooncake_cpu_kv_memory((128,), dtype=torch.int8)

        self.assertEqual(kv_tensor.dtype, torch.int8)
        self.assertEqual(kv_tensor.numel(), 128)
        fake_torch_npu.empty_with_swapped_memory.assert_called_once_with(
            (128,),
            dtype=torch.int8,
            device="npu",
        )

    def test_final_kv_view_remains_registered_after_raw_view_is_released(self):
        impl = object.__new__(NPUModelRunner)

        raw_storage = torch.empty((4096 + 64 + 1,), dtype=torch.int8)
        raw_offset = 1 if raw_storage.data_ptr() % 64 == 0 else 0
        raw_tensor = raw_storage[raw_offset:]
        self.assertNotEqual(raw_tensor.data_ptr() % 64, 0)
        with patch(
            "vllm_ascend.worker.model_runner_v1.empty_mooncake_cpu_kv_memory",
            return_value=raw_tensor,
        ):
            raw_kv_tensor = NPUModelRunner._allocate_dsa_pd_mooncake_kv_tensor(impl, 4096, 64)

        self.assertEqual(raw_kv_tensor.numel(), 4096)
        self.assertEqual(raw_kv_tensor.data_ptr() % 64, 0)
        self.assertIsNone(
            get_dsa_pd_mooncake_kv_register_location(
                raw_kv_tensor.data_ptr(),
                raw_kv_tensor.element_size() * raw_kv_tensor.numel(),
            )
        )
        self.assertIsNone(
            get_dsa_pd_mooncake_offloaded_kv_tensor(
                raw_kv_tensor.data_ptr(),
                raw_kv_tensor.element_size() * raw_kv_tensor.numel(),
            )
        )

        final_kv_view = raw_kv_tensor.view(torch.float32).view(16, 64)
        NPUModelRunner._register_dsa_pd_mooncake_final_kv_views(
            impl,
            {"model.layers.0.self_attn.attn": (final_kv_view,)},
        )
        raw_view_ref = weakref.ref(raw_kv_tensor)
        del raw_kv_tensor
        gc.collect()

        self.assertIsNone(raw_view_ref())
        resolved = get_dsa_pd_mooncake_offloaded_kv_tensor(
            final_kv_view.data_ptr(),
            final_kv_view.element_size() * final_kv_view.numel(),
        )
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertIs(resolved[0], final_kv_view)
        self.assertEqual(resolved[1], 0)
