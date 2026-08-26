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
from vllm_ascend.quantization.tp_weight_switch import (
    TPWeightGatherSpec,
    TPWeightSwitchMixin,
)


class _OProjLinearMethod(TPWeightSwitchMixin):
    supports_tp_weight_switch = True
    tp_weight_gather_specs = (
        TPWeightGatherSpec("weight"),
        TPWeightGatherSpec("weight_scale"),
    )


class TestAscendDSACPOProjTPParams(unittest.TestCase):
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
        impl._o_proj_tp_weight_switch_enabled = False
        return impl

    def test_get_tp_weight_switch_method_unwraps_adapter_and_rejects_unsupported(self):
        layer = self._OProj()

        method = AscendDSACPImpl._get_tp_weight_switch_method(layer)

        self.assertIs(method, layer.quant_method.quant_method)
        layer.quant_method = object()
        with self.assertRaisesRegex(RuntimeError, "TP weight-switch capable"):
            AscendDSACPImpl._get_tp_weight_switch_method(layer)

    def test_enable_o_proj_switch_initializes_both_layers_once_with_cloned_tp_storage(self):
        impl = self._make_impl()
        original_ptrs = (impl.wo_a.weight.data_ptr(), impl.wo_b.weight.data_ptr())

        impl._enable_o_proj_tp_full_weight_switch()

        self.assertTrue(impl._o_proj_tp_weight_switch_enabled)
        self.assertNotEqual(impl.wo_a.weight.data_ptr(), original_ptrs[0])
        self.assertNotEqual(impl.wo_b.weight.data_ptr(), original_ptrs[1])
        self.assertEqual(
            impl.wo_a.weight.data_ptr(),
            impl.wo_a_tp_weight_state.gather_parts["weight"].tp_tensor.data_ptr(),
        )
        self.assertEqual(
            impl.wo_b.weight.data_ptr(),
            impl.wo_b_tp_weight_state.gather_parts["weight"].tp_tensor.data_ptr(),
        )
        self.assertEqual(len(AscendDSACPImpl.o_proj_full_pools), 4)

        wo_a_state = impl.wo_a_tp_weight_state
        wo_b_state = impl.wo_b_tp_weight_state
        impl._enable_o_proj_tp_full_weight_switch()
        self.assertIs(impl.wo_a_tp_weight_state, wo_a_state)
        self.assertIs(impl.wo_b_tp_weight_state, wo_b_state)

    def test_maybe_all_gather_schedules_weights_and_scales_separately(self):
        impl = self._make_impl()
        impl._enable_o_proj_tp_full_weight_switch()
        handle = MagicMock()
        with patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.all_gather_async",
            return_value=(None, handle),
        ) as mock_all_gather:
            self.assertEqual(impl._maybe_all_gather_wo_a_weight(False), [])
            self.assertEqual(impl._maybe_all_gather_wo_b_weight(False), [])
            self.assertEqual(impl._maybe_all_gather_o_proj_scales(False), ([], []))
            mock_all_gather.assert_not_called()

            wo_a_weight_handles = impl._maybe_all_gather_wo_a_weight(True)
            wo_a_scale_handles, wo_b_scale_handles = impl._maybe_all_gather_o_proj_scales(True)
            wo_b_weight_handles = impl._maybe_all_gather_wo_b_weight(True)

        self.assertEqual(wo_a_weight_handles, [handle])
        self.assertEqual(wo_a_scale_handles, [handle])
        self.assertEqual(wo_b_scale_handles, [handle])
        self.assertEqual(wo_b_weight_handles, [handle])
        self.assertEqual(mock_all_gather.call_count, 4)
        gathered_outputs = [call.kwargs["output"] for call in mock_all_gather.call_args_list]
        self.assertIs(gathered_outputs[0], impl.wo_a_tp_weight_state.gather_parts["weight"].gather_output)
        self.assertIs(gathered_outputs[1], impl.wo_a_tp_weight_state.gather_parts["weight_scale"].gather_output)
        self.assertIs(gathered_outputs[2], impl.wo_b_tp_weight_state.gather_parts["weight_scale"].gather_output)
        self.assertIs(gathered_outputs[3], impl.wo_b_tp_weight_state.gather_parts["weight"].gather_output)

    def test_switch_o_proj_between_full_and_tp_storage(self):
        impl = self._make_impl()
        impl._enable_o_proj_tp_full_weight_switch()
        tp_ptrs = (impl.wo_a.weight.data_ptr(), impl.wo_b.weight.data_ptr())
        full_ptrs = (
            impl.wo_a_tp_weight_state.gather_parts["weight"].full_tensor.data_ptr(),
            impl.wo_b_tp_weight_state.gather_parts["weight"].full_tensor.data_ptr(),
        )

        wo_a_handle = MagicMock()
        wo_b_handle = MagicMock()
        impl._switch_wo_a_to_full_weight([wo_a_handle])
        impl._switch_wo_b_to_full_weight([wo_b_handle])

        self.assertEqual(impl.wo_a.weight.data_ptr(), full_ptrs[0])
        self.assertEqual(impl.wo_b.weight.data_ptr(), full_ptrs[1])
        wo_a_handle.wait.assert_called_once_with()
        wo_b_handle.wait.assert_called_once_with()

        impl._switch_o_proj_to_tp_weight()

        self.assertEqual(impl.wo_a.weight.data_ptr(), tp_ptrs[0])
        self.assertEqual(impl.wo_b.weight.data_ptr(), tp_ptrs[1])

    def test_activation_all_gather_is_async_and_unpads_after_wait(self):
        impl = self._make_impl()
        hidden_states_local = torch.randn(2, 4)
        gathered = torch.randn(4, 4)
        handle = MagicMock()

        with patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.all_gather_async",
            return_value=(gathered, handle),
        ) as mock_all_gather:
            output, output_handle = impl._maybe_all_gather_activation_async(hidden_states_local, True)

        self.assertIs(output, gathered)
        self.assertIs(output_handle, handle)
        mock_all_gather.assert_called_once()
        self.assertTrue(mock_all_gather.call_args.args[0].is_contiguous())
        self.assertIs(mock_all_gather.call_args.args[1], impl.tp_group)

        with patch(
            "vllm_ascend.attention.context_parallel.dsa_cp._EXTRA_CTX",
            SimpleNamespace(pad_size=1),
        ):
            output = impl._finish_activation_all_gather(output, output_handle)

        handle.wait.assert_called_once_with()
        self.assertEqual(tuple(output.shape), (3, 4))
