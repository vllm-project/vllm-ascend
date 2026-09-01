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
# This file is a part of the vllm-ascend project.
#
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl
from vllm_ascend.attention.sfa_v1 import PreprocessType, SFAForwardContext
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


class _UnsupportedOProjLinearMethod(TPWeightSwitchMixin):
    pass


class TestAscendSFAOProjTPParams(TestBase):
    class _OProj(torch.nn.Module):
        def __init__(self, linear_method):
            super().__init__()
            self.input_size = 8
            self.input_size_per_partition = 4
            self.output_size = 3
            self.output_size_per_partition = 3
            self.weight = torch.nn.Parameter(torch.randn(4, 3), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(torch.randn(2, 3), requires_grad=False)
            self.quant_method = linear_method

    def setUp(self):
        AscendSFADSACPImpl.o_proj_full_pools.clear()

    def _make_impl(self, linear_method=None):
        impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
        impl.tp_size = 2
        impl.o_proj = self._OProj(linear_method or _OProjLinearMethod())
        impl._o_proj_tp_weight_switch_enabled = False
        impl.o_proj_tp_weight_state = None
        return impl

    def test_enable_o_proj_switch_uses_mixin_state_and_is_idempotent(self):
        impl = self._make_impl()
        original_weight_ptr = impl.o_proj.weight.data_ptr()
        original_scale_ptr = impl.o_proj.weight_scale.data_ptr()

        impl._enable_o_proj_tp_full_weight_switch()

        state = impl.o_proj_tp_weight_state
        self.assertTrue(impl._o_proj_tp_weight_switch_enabled)
        self.assertEqual(state.gather_parts["weight"].tp_tensor.data_ptr(), original_weight_ptr)
        self.assertEqual(state.gather_parts["weight_scale"].tp_tensor.data_ptr(), original_scale_ptr)
        self.assertEqual(state.gather_parts["weight"].full_tensor.shape, (8, 3))
        self.assertEqual(state.gather_parts["weight_scale"].full_tensor.shape, (4, 3))
        self.assertEqual(len(AscendSFADSACPImpl.o_proj_full_pools), 2)

        impl._enable_o_proj_tp_full_weight_switch()
        self.assertIs(impl.o_proj_tp_weight_state, state)

    def test_o_proj_full_weight_forward_restores_tp_storage(self):
        impl = self._make_impl()
        impl._enable_o_proj_tp_full_weight_switch()
        state = impl.o_proj_tp_weight_state
        original_weight_ptr = impl.o_proj.weight.data_ptr()
        original_scale_ptr = impl.o_proj.weight_scale.data_ptr()
        full_weight_ptr = state.gather_parts["weight"].full_tensor.data_ptr()
        full_scale_ptr = state.gather_parts["weight_scale"].full_tensor.data_ptr()

        def _apply_with_full_weight(_attn_output):
            self.assertEqual(impl.o_proj.weight.data_ptr(), full_weight_ptr)
            self.assertEqual(impl.o_proj.weight_scale.data_ptr(), full_scale_ptr)
            return torch.ones(2, 3)

        impl._apply_o_proj_full_weight = MagicMock(side_effect=_apply_with_full_weight)

        impl.enable_dsa_cp_with_o_proj_tp = True
        gathered_output = torch.cat((torch.ones(2, 3), torch.full((2, 3), 2.0)))
        tp_group = SimpleNamespace(all_gather=MagicMock(return_value=gathered_output))
        with patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=tp_group):
            output = impl._finalize_o_proj(
                attn_output=torch.randn(2, 8),
                output=torch.empty(3, 3),
                gather_full_o_proj=True,
            )

        self.assertEqual(impl.o_proj.weight.data_ptr(), original_weight_ptr)
        self.assertEqual(impl.o_proj.weight_scale.data_ptr(), original_scale_ptr)
        tp_group.all_gather.assert_called_once()
        self.assertTrue(torch.equal(output, gathered_output[:3]))

    def test_prepare_native_hidden_states_slices_replicated_token_state(self):
        impl = self._make_impl()
        hidden_states = torch.arange(24).reshape(6, 4)
        attn_metadata = SimpleNamespace(
            dsa_cp_context=SimpleNamespace(
                num_tokens_pad=6,
                local_start=3,
                local_end_with_pad=6,
            )
        )

        local_hidden_states = impl._prepare_native_hidden_states(hidden_states, attn_metadata)

        torch.testing.assert_close(local_hidden_states, hidden_states[3:6])

    def test_prepare_native_hidden_states_pads_unaligned_token_state(self):
        impl = self._make_impl()
        hidden_states = torch.arange(20).reshape(5, 4)
        attn_metadata = SimpleNamespace(
            dsa_cp_context=SimpleNamespace(
                num_tokens_pad=6,
                local_start=3,
                local_end_with_pad=6,
            )
        )

        local_hidden_states = impl._prepare_native_hidden_states(hidden_states, attn_metadata)

        torch.testing.assert_close(local_hidden_states[:2], hidden_states[3:5])
        torch.testing.assert_close(local_hidden_states[2], torch.zeros(4, dtype=hidden_states.dtype))

    def test_enable_o_proj_switch_rejects_unsupported_method(self):
        impl = self._make_impl(_UnsupportedOProjLinearMethod())

        with self.assertRaisesRegex(RuntimeError, "TP weight-switch capable"):
            impl._enable_o_proj_tp_full_weight_switch()

    def test_no_indexer_full_o_proj_still_opens_gate_and_saves_layer(self):
        impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
        impl.enable_dsa_cp_with_o_proj_tp = True
        impl.enable_sp = False
        impl.has_indexer = False
        impl.skip_topk = True
        impl.enable_sparse_sfa_c8 = False
        impl.is_kv_producer = True
        impl.preprocess_type = PreprocessType.NATIVE
        impl.tp_size = 2
        impl.q_lora_rank = 8
        impl.kv_lora_rank = 4
        impl.qk_rope_head_dim = 2
        impl.layer_name = "layers.0.attn"

        q_c = MagicMock()
        qkv_lora = MagicMock()
        qkv_lora.split.return_value = (q_c, MagicMock())
        impl.fused_qkv_a_proj = MagicMock(return_value=(qkv_lora,))
        impl.q_a_layernorm = MagicMock(return_value=q_c)
        impl.exec_kv = MagicMock(return_value=(MagicMock(), MagicMock()))
        impl._q_proj_and_k_up_proj = MagicMock(return_value=(MagicMock(), MagicMock()))
        impl.rope_single = MagicMock(return_value=MagicMock())
        impl._record_query_gather_context = MagicMock()
        impl._prepare_kv_for_parallel = MagicMock(return_value=(None, None, None, []))
        impl._store_parallel_kv = MagicMock(return_value=(None, None, None))
        impl._get_indexcache_topk_indices = MagicMock(return_value=MagicMock())
        impl._execute_sparse_flash_attention_process = MagicMock(return_value=MagicMock())
        attn_output = MagicMock()
        impl._v_up_proj = MagicMock(return_value=attn_output)
        impl.o_proj = MagicMock()
        impl._prepare_native_hidden_states = MagicMock(side_effect=lambda hidden_states, _: hidden_states)

        output = MagicMock()
        finalized_output = MagicMock()
        kv_cache = (MagicMock(), MagicMock())
        impl._compose_sfa_kv_cache = MagicMock(return_value=kv_cache)
        impl._finalize_o_proj = MagicMock(return_value=finalized_output)

        attn_metadata = MagicMock()
        attn_metadata.dcp_context = None
        attn_metadata.dsa_cp_context = None
        attn_metadata.num_input_tokens = 1
        impl._get_parallel_forward_context = MagicMock(
            return_value=SFAForwardContext(
                actual_seq_lengths_query=MagicMock(),
                actual_seq_lengths_key=MagicMock(),
                kv_slot_mapping=MagicMock(),
                topk_num_tokens=1,
                gather_full_o_proj=True,
            )
        )

        with (
            patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector"),
            patch("vllm_ascend.attention.sfa_v1.record_attention_compute_start") as record_gate,
            patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector") as save_layer,
            patch("vllm_ascend.attention.sfa_v1.notify_kv_cache_written") as notify_cache_written,
        ):
            result = impl.forward(
                layer_name=impl.layer_name,
                hidden_states=MagicMock(),
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        self.assertIs(result, finalized_output)
        impl._finalize_o_proj.assert_called_once_with(attn_output, output, True)
        notify_cache_written.assert_called_once_with(impl.layer_name)
        record_gate.assert_called_once_with()
        save_layer.assert_called_once_with(impl.layer_name, list(kv_cache))
        impl._prepare_native_hidden_states.assert_called_once()
        impl.o_proj.assert_not_called()
