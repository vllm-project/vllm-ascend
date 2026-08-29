# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import torch

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPImpl, DSACPMetadata
from vllm_ascend.weight_switch import (
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

    def test_split_full_hidden_states_for_cp_uses_metadata_range(self):
        hidden_states = torch.arange(32).reshape(8, 4)
        cp_metadata = DSACPMetadata(
            local_query_start_loc=torch.tensor([0, 2]),
            local_seq_lens=torch.tensor([2]),
            local_start=4,
            local_end=6,
            tokens_per_rank=2,
            num_tokens_pad=8,
        )

        local_hidden_states = AscendDSACPImpl._split_full_hidden_states_for_cp(hidden_states, cp_metadata)

        torch.testing.assert_close(local_hidden_states, hidden_states[4:6])

    def test_split_full_hidden_states_for_cp_pads_unaligned_input(self):
        cp_metadata = DSACPMetadata(
            local_query_start_loc=torch.tensor([0, 2]),
            local_seq_lens=torch.tensor([2]),
            local_start=6,
            local_end=8,
            tokens_per_rank=2,
            num_tokens_pad=8,
        )

        hidden_states = torch.arange(28).reshape(7, 4)
        local_hidden_states = AscendDSACPImpl._split_full_hidden_states_for_cp(hidden_states, cp_metadata)

        torch.testing.assert_close(local_hidden_states[0], hidden_states[-1])
        torch.testing.assert_close(local_hidden_states[1], torch.zeros(4, dtype=hidden_states.dtype))

    def test_split_full_hidden_states_for_cp_rejects_oversized_input(self):
        cp_metadata = DSACPMetadata(
            local_query_start_loc=torch.tensor([0, 2]),
            local_seq_lens=torch.tensor([2]),
            local_start=4,
            local_end=6,
            tokens_per_rank=2,
            num_tokens_pad=8,
        )

        with self.assertRaisesRegex(RuntimeError, "exceeds its TP-aligned metadata"):
            AscendDSACPImpl._split_full_hidden_states_for_cp(torch.empty(9, 4), cp_metadata)

    def test_gather_cp_output_restores_rank_ordered_full_state(self):
        impl = self._make_impl()
        local_output = torch.arange(8).reshape(2, 4)
        gathered_output = torch.arange(16).reshape(4, 4)
        impl.tp_group = SimpleNamespace(all_gather=MagicMock(return_value=gathered_output))
        cp_metadata = DSACPMetadata(
            local_query_start_loc=torch.tensor([0, 2]),
            local_seq_lens=torch.tensor([2]),
            local_start=2,
            local_end=4,
            tokens_per_rank=2,
            num_tokens_pad=4,
        )

        output = impl._gather_cp_output(local_output, cp_metadata)

        impl.tp_group.all_gather.assert_called_once_with(local_output, dim=0)
        self.assertIs(output, gathered_output)

    def test_gather_cp_output_preserves_already_restored_full_state(self):
        impl = self._make_impl()
        impl.tp_group = SimpleNamespace(all_gather=MagicMock())
        full_output = torch.arange(16).reshape(4, 4)
        cp_metadata = DSACPMetadata(
            local_query_start_loc=torch.tensor([0, 2]),
            local_seq_lens=torch.tensor([2]),
            local_start=2,
            local_end=4,
            tokens_per_rank=2,
            num_tokens_pad=4,
        )

        output = impl._gather_cp_output(full_output, cp_metadata)

        impl.tp_group.all_gather.assert_not_called()
        self.assertIs(output, full_output)

    def test_gather_cp_output_removes_tp_padding(self):
        impl = self._make_impl()
        full_output = torch.arange(16).reshape(4, 4)
        cp_metadata = DSACPMetadata(
            local_query_start_loc=torch.tensor([0, 2]),
            local_seq_lens=torch.tensor([2]),
            local_start=2,
            local_end=4,
            tokens_per_rank=2,
            num_tokens_pad=4,
        )

        output = impl._gather_cp_output(full_output, cp_metadata, num_output_tokens=3)

        torch.testing.assert_close(output, full_output[:3])

    def test_gather_cp_output_rejects_wrong_full_size(self):
        impl = self._make_impl()
        impl.tp_group = SimpleNamespace(all_gather=MagicMock(return_value=torch.empty(3, 4)))
        cp_metadata = DSACPMetadata(
            local_query_start_loc=torch.tensor([0, 2]),
            local_seq_lens=torch.tensor([2]),
            local_start=0,
            local_end=2,
            tokens_per_rank=2,
            num_tokens_pad=4,
        )

        with self.assertRaisesRegex(RuntimeError, "gathered output does not match"):
            impl._gather_cp_output(torch.empty(2, 4), cp_metadata)

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

    def test_maybe_all_gather_honors_enable_flag_for_both_layers(self):
        impl = self._make_impl()
        impl._enable_o_proj_tp_full_weight_switch()
        impl.wo_a_tp_weight_method.all_gather_tp_weight = MagicMock()
        impl.wo_b_tp_weight_method.all_gather_tp_weight = MagicMock()

        impl._maybe_all_gather_o_proj_full_weight(False)

        impl.wo_a_tp_weight_method.all_gather_tp_weight.assert_not_called()
        impl.wo_b_tp_weight_method.all_gather_tp_weight.assert_not_called()

        impl._maybe_all_gather_o_proj_full_weight(True)

        impl.wo_a_tp_weight_method.all_gather_tp_weight.assert_called_once_with(
            impl.wo_a_tp_weight_state,
            impl.tp_group,
        )
        impl.wo_b_tp_weight_method.all_gather_tp_weight.assert_called_once_with(
            impl.wo_b_tp_weight_state,
            impl.tp_group,
        )

    def test_switch_o_proj_between_full_and_tp_storage(self):
        impl = self._make_impl()
        impl._enable_o_proj_tp_full_weight_switch()
        tp_ptrs = (impl.wo_a.weight.data_ptr(), impl.wo_b.weight.data_ptr())
        full_ptrs = (
            impl.wo_a_tp_weight_state.gather_parts["weight"].full_tensor.data_ptr(),
            impl.wo_b_tp_weight_state.gather_parts["weight"].full_tensor.data_ptr(),
        )

        impl._switch_o_proj_to_full_weight()

        self.assertEqual(impl.wo_a.weight.data_ptr(), full_ptrs[0])
        self.assertEqual(impl.wo_b.weight.data_ptr(), full_ptrs[1])

        impl._switch_o_proj_to_tp_weight()

        self.assertEqual(impl.wo_a.weight.data_ptr(), tp_ptrs[0])
        self.assertEqual(impl.wo_b.weight.data_ptr(), tp_ptrs[1])
