# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPImpl
from vllm_ascend.device.hardware_profile import HardwareCapability
from vllm_ascend.weight_switch import (
    WeightSwitchConfig,
    WeightSwitchGatherSpec,
    WeightSwitchMixin,
)


class _OProjLinearMethod(WeightSwitchMixin):
    supports_weight_switch = True
    weight_switch_gather_specs = (
        WeightSwitchGatherSpec("weight"),
        WeightSwitchGatherSpec("weight_scale"),
    )


class TestAscendDSACPOProjWeightSwitch(unittest.TestCase):
    class _OProj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_size = 8
            self.input_size_per_partition = 4
            self.output_size = 3
            self.output_size_per_partition = 3
            self.weight = torch.nn.Parameter(torch.randn(4, 3), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(torch.randn(2, 3), requires_grad=False)
            self.quant_method: Any = SimpleNamespace(quant_method=_OProjLinearMethod())

    def setUp(self):
        AscendDSACPImpl.o_proj_full_pools.clear()

    def _make_impl(self):
        impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
        impl.tp_size = 2
        impl.tp_group = object()
        impl.o_proj_weight_switch_config = WeightSwitchConfig(
            group=impl.tp_group,
            world_size=impl.tp_size,
            rank=0,
        )
        impl.wo_a = self._OProj()
        impl.wo_b = self._OProj()
        impl._o_proj_weight_switch_enabled = False
        return impl

    def test_enablement_is_not_gated_by_hardware_family(self):
        profile = MagicMock()
        profile.supports.return_value = False
        tp_group = SimpleNamespace(world_size=2, rank_in_group=0)
        layer = self._OProj()
        with (
            patch(
                "vllm_ascend.attention.context_parallel.dsa_cp.enable_dsa_cp_full_o_proj",
                return_value=True,
            ),
            patch("vllm_ascend.attention.context_parallel.dsa_cp.get_tp_group", return_value=tp_group),
            patch(
                "vllm_ascend.attention.context_parallel.dsa_cp.get_current_hardware_profile",
                return_value=profile,
            ),
            patch(
                "vllm_ascend.attention.context_parallel.dsa_cp.get_current_vllm_config",
                return_value=SimpleNamespace(),
            ),
        ):
            impl = AscendDSACPImpl(
                n_heads=2,
                scale=1.0,
                n_local_heads=1,
                q_lora_rank=1,
                o_lora_rank=1,
                head_dim=2,
                rope_head_dim=1,
                nope_head_dim=1,
                n_groups=2,
                n_local_groups=1,
                window_size=1,
                compress_ratio=1,
                wq_a=object(),
                wq_b=object(),
                wkv=object(),
                q_norm=object(),
                kv_norm=object(),
                swa_cache_layer=SimpleNamespace(prefix="swa"),
                wo_a=layer,
                wo_b=layer,
                eps=1e-6,
                attn_sink=torch.empty(2),
            )

        self.assertTrue(impl.enable_dsa_cp_full_o_proj)
        profile.supports.assert_called_once_with(HardwareCapability.FP8_ATTENTION)

    def test_get_weight_switch_method_unwraps_adapter_and_rejects_unsupported(self):
        layer = self._OProj()

        method = AscendDSACPImpl._get_weight_switch_method(layer)

        self.assertIs(method, layer.quant_method.quant_method)
        layer.quant_method = object()
        with self.assertRaisesRegex(RuntimeError, "weight-switch capable"):
            AscendDSACPImpl._get_weight_switch_method(layer)

    def test_enable_o_proj_switch_initializes_both_layers_once_with_cloned_local_storage(self):
        impl = self._make_impl()
        original_ptrs = (impl.wo_a.weight.data_ptr(), impl.wo_b.weight.data_ptr())

        impl._enable_o_proj_full_weight_switch()

        self.assertTrue(impl._o_proj_weight_switch_enabled)
        self.assertNotEqual(impl.wo_a.weight.data_ptr(), original_ptrs[0])
        self.assertNotEqual(impl.wo_b.weight.data_ptr(), original_ptrs[1])
        self.assertEqual(
            impl.wo_a.weight.data_ptr(),
            impl.wo_a_weight_state.gather_parts["weight"].local_tensor.data_ptr(),
        )
        self.assertEqual(
            impl.wo_b.weight.data_ptr(),
            impl.wo_b_weight_state.gather_parts["weight"].local_tensor.data_ptr(),
        )
        self.assertEqual(len(AscendDSACPImpl.o_proj_full_pools), 4)

        wo_a_state = impl.wo_a_weight_state
        wo_b_state = impl.wo_b_weight_state
        impl._enable_o_proj_full_weight_switch()
        self.assertIs(impl.wo_a_weight_state, wo_a_state)
        self.assertIs(impl.wo_b_weight_state, wo_b_state)

    def test_maybe_all_gather_honors_enable_flag_for_both_layers(self):
        impl = self._make_impl()
        impl._enable_o_proj_full_weight_switch()
        impl.wo_a_weight_method.all_gather_weight = MagicMock()
        impl.wo_b_weight_method.all_gather_weight = MagicMock()

        impl._maybe_all_gather_o_proj_full_weight(False)

        impl.wo_a_weight_method.all_gather_weight.assert_not_called()
        impl.wo_b_weight_method.all_gather_weight.assert_not_called()

        impl._maybe_all_gather_o_proj_full_weight(True)

        impl.wo_a_weight_method.all_gather_weight.assert_called_once_with(
            impl.wo_a_weight_state,
            impl.o_proj_weight_switch_config,
        )
        impl.wo_b_weight_method.all_gather_weight.assert_called_once_with(
            impl.wo_b_weight_state,
            impl.o_proj_weight_switch_config,
        )

    def test_switch_o_proj_between_full_and_local_storage(self):
        impl = self._make_impl()
        impl._enable_o_proj_full_weight_switch()
        local_ptrs = (impl.wo_a.weight.data_ptr(), impl.wo_b.weight.data_ptr())
        full_ptrs = (
            impl.wo_a_weight_state.gather_parts["weight"].full_tensor.data_ptr(),
            impl.wo_b_weight_state.gather_parts["weight"].full_tensor.data_ptr(),
        )

        impl._switch_o_proj_to_full_weight()

        self.assertEqual(impl.wo_a.weight.data_ptr(), full_ptrs[0])
        self.assertEqual(impl.wo_b.weight.data_ptr(), full_ptrs[1])

        impl._switch_o_proj_to_local_weight()

        self.assertEqual(impl.wo_a.weight.data_ptr(), local_ptrs[0])
        self.assertEqual(impl.wo_b.weight.data_ptr(), local_ptrs[1])
