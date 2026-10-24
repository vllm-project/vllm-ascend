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
    ParallelOProjection,
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
        impl.wo_a = self._OProj()
        impl.wo_b = self._OProj()
        impl._parallel_o_proj = ParallelOProjection(
            4,
            impl.wo_b,
            config=WeightSwitchConfig(group=impl.tp_group, world_size=impl.tp_size, rank=0),
            tp_group=impl.tp_group,
            pool=impl.o_proj_full_pools,
            pool_key=impl.o_proj_weight_switch_pool_key,
            project_full=MagicMock(),
            decode_sharded=False,
            input_layer=impl.wo_a,
            clone_local_tensors=True,
        )
        impl._parallel_o_proj.enabled = True
        return impl

    def test_enablement_is_limited_to_a5(self):
        profile = MagicMock()
        profile.supports.return_value = False
        tp_group = SimpleNamespace(world_size=2, rank_in_group=0)
        layer = self._OProj()
        with (
            patch(
                "vllm_ascend.attention.context_parallel.dsa_cp.get_ascend_config",
                return_value=SimpleNamespace(multistream_dsv4_dsa_overlap=True),
            ),
            patch(
                "vllm_ascend.attention.context_parallel.dsa_cp.is_a5_bf16_kv_enabled",
                return_value=False,
            ),
            patch(
                "vllm_ascend.attention.context_parallel.dsa_cp.enable_dsa_cp_full_o_proj",
                return_value=True,
            ),
            patch(
                "vllm_ascend.attention.context_parallel.dsa_cp.is_950",
                side_effect=(False, True),
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
            impls = [
                AscendDSACPImpl(
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
                    wq_a=layer,
                    wq_b=layer,
                    wkv=layer,
                    q_norm=object(),
                    kv_norm=object(),
                    swa_cache_layer=SimpleNamespace(prefix="swa"),
                    wo_a=layer,
                    wo_b=layer,
                    eps=1e-6,
                    attn_sink=torch.empty(2),
                )
                for _ in range(2)
            ]

        self.assertFalse(impls[0].enable_dsa_cp_full_o_proj)
        self.assertTrue(impls[1].enable_dsa_cp_full_o_proj)
        for impl in impls:
            self.assertTrue(impl.multistream_dsv4_dsa_overlap)
            self.assertIs(impl.cv_wq_a.linear, layer)
            self.assertIs(impl.cv_wkv.linear, layer)
            self.assertIs(impl.cv_wq_b.linear, layer)
        self.assertEqual(profile.supports.call_args_list, [unittest.mock.call(HardwareCapability.FP8_ATTENTION)] * 2)

    def test_resolve_o_proj_weight_switch_method_unwraps_adapter_and_rejects_unsupported(self):
        layer = self._OProj()

        method = ParallelOProjection._resolve_o_proj_weight_switch_method(layer)

        self.assertIs(method, layer.quant_method.quant_method)
        layer.quant_method = object()
        with self.assertRaisesRegex(RuntimeError, "weight-switch capable"):
            ParallelOProjection._resolve_o_proj_weight_switch_method(layer)

    def test_enable_o_proj_switch_initializes_both_layers_once_with_cloned_local_storage(self):
        impl = self._make_impl()
        original_ptrs = (impl.wo_a.weight.data_ptr(), impl.wo_b.weight.data_ptr())

        impl._parallel_o_proj.enable()

        self.assertTrue(impl._parallel_o_proj._weight_switch_enabled)
        self.assertNotEqual(impl.wo_a.weight.data_ptr(), original_ptrs[0])
        self.assertNotEqual(impl.wo_b.weight.data_ptr(), original_ptrs[1])
        self.assertEqual(
            impl.wo_a.weight.data_ptr(),
            impl._parallel_o_proj._weight_switch_states[0][2].gather_parts["weight"].local_tensor.data_ptr(),
        )
        self.assertEqual(
            impl.wo_b.weight.data_ptr(),
            impl._parallel_o_proj._weight_switch_states[1][2].gather_parts["weight"].local_tensor.data_ptr(),
        )
        self.assertEqual(len(AscendDSACPImpl.o_proj_full_pools), 4)

        wo_a_state = impl._parallel_o_proj._weight_switch_states[0][2]
        wo_b_state = impl._parallel_o_proj._weight_switch_states[1][2]
        impl._parallel_o_proj.enable()
        self.assertIs(impl._parallel_o_proj._weight_switch_states[0][2], wo_a_state)
        self.assertIs(impl._parallel_o_proj._weight_switch_states[1][2], wo_b_state)

    def test_explicit_full_mode_gather_honors_enable_flag_for_both_layers(self):
        impl = self._make_impl()
        impl._parallel_o_proj.enable()
        impl._parallel_o_proj._weight_switch_states[0][1].all_gather_weight = MagicMock()
        impl._parallel_o_proj._weight_switch_states[1][1].all_gather_weight = MagicMock()

        impl._parallel_o_proj.set_mode(use_full_weight=False, exchange_tokens=False)
        impl._parallel_o_proj.all_gather_weights()

        impl._parallel_o_proj._weight_switch_states[0][1].all_gather_weight.assert_not_called()
        impl._parallel_o_proj._weight_switch_states[1][1].all_gather_weight.assert_not_called()

        impl._parallel_o_proj.set_mode(use_full_weight=True, exchange_tokens=False)
        impl._parallel_o_proj.all_gather_weights()

        config = impl._parallel_o_proj.config
        impl._parallel_o_proj._weight_switch_states[0][1].all_gather_weight.assert_called_once_with(
            impl._parallel_o_proj._weight_switch_states[0][2],
            config,
        )
        impl._parallel_o_proj._weight_switch_states[1][1].all_gather_weight.assert_called_once_with(
            impl._parallel_o_proj._weight_switch_states[1][2],
            config,
        )

    def test_switch_o_proj_between_full_and_local_storage(self):
        impl = self._make_impl()
        impl._parallel_o_proj.enable()
        local_ptrs = (impl.wo_a.weight.data_ptr(), impl.wo_b.weight.data_ptr())
        full_ptrs = (
            impl._parallel_o_proj._weight_switch_states[0][2].gather_parts["weight"].full_tensor.data_ptr(),
            impl._parallel_o_proj._weight_switch_states[1][2].gather_parts["weight"].full_tensor.data_ptr(),
        )

        with impl._parallel_o_proj.use_full_weights():
            self.assertEqual(impl.wo_a.weight.data_ptr(), full_ptrs[0])
            self.assertEqual(impl.wo_b.weight.data_ptr(), full_ptrs[1])

        self.assertEqual(impl.wo_a.weight.data_ptr(), local_ptrs[0])
        self.assertEqual(impl.wo_b.weight.data_ptr(), local_ptrs[1])

    def test_restore_tp_heads_preserves_inverse_rope_with_hardware_fallback(self):
        for supports_negate in (False, True):
            for tp_size, skip_all_to_all in ((1, False), (2, False), (2, True)):
                with self.subTest(supports_negate=supports_negate, tp_size=tp_size, skip=skip_all_to_all):
                    impl = self._make_impl()
                    impl.tp_size = tp_size
                    impl.nope_head_dim, impl.head_dim = 2, 4
                    output = torch.randn(3, 2, 4)
                    sin = torch.randn(3, 2)
                    cos = torch.randn(3, 2)
                    metadata = SimpleNamespace(
                        req_metadata=SimpleNamespace(
                            cp_metadata=SimpleNamespace(local_sin={"layer": sin}, local_cos={"layer": cos})
                        )
                    )
                    with (
                        patch("vllm_ascend.attention.context_parallel.dsa_cp.get_current_hardware_profile") as profile,
                        patch("torch.ops._C_ascend.inplace_partial_rotary_mul", create=True) as rotary,
                        patch("vllm_ascend.attention.context_parallel.dsa_cp.restore_tp_heads") as restore,
                    ):
                        profile.return_value.supports.return_value = supports_negate
                        result = impl._restore_tp_head_layout(output, "layer", metadata, skip_all_to_all)

                    profile.return_value.supports.assert_called_once_with(
                        HardwareCapability.INPLACE_PARTIAL_ROTARY_MUL_NEGATE_SIN
                    )
                    rotary.assert_called_once()
                    args, kwargs = rotary.call_args
                    self.assertEqual(args[0].data_ptr(), output.data_ptr())
                    self.assertIs(args[1], cos)
                    self.assertEqual(kwargs["partial_slice"], [2, 4])
                    self.assertEqual(kwargs["rotary_mode"], "interleave")
                    effective_sin = -args[2] if kwargs["negate_sin"] else args[2]
                    torch.testing.assert_close(effective_sin, -sin)
                    if tp_size == 1 or skip_all_to_all:
                        restore.assert_not_called()
                        self.assertIs(result, output)
                    else:
                        restore.assert_called_once_with(output, impl.tp_group)
                        self.assertIs(result, restore.return_value)
