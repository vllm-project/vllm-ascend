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
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


class TestAscendSFAOProjTPParams(TestBase):
    class _OProj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 3), requires_grad=False)
            self.aclnn_input_scale = torch.nn.Parameter(torch.randn(3), requires_grad=False)
            self.weight_scale_second = torch.nn.Parameter(torch.randn(4, 2), requires_grad=False)
            self.weight_scale_second.input_dim = 1
            self.weight_offset_second = torch.nn.Parameter(torch.randn(4, 2), requires_grad=False)
            self.weight_offset_second.input_dim = 1
            self.extra_input_scale = torch.nn.Parameter(torch.randn(4, 2), requires_grad=False)
            self.extra_input_scale.input_dim = 1
            self.weight_scale = torch.nn.Parameter(torch.randn(4), requires_grad=False)

    def setUp(self):
        AscendSFAImpl.o_proj_full_pools.clear()

    def _make_impl(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.tp_size = 2
        impl.o_proj = self._OProj()
        impl._is_o_proj_unquantized = lambda: False
        return impl

    def test_o_proj_tp_params_alias_original_storage(self):
        impl = self._make_impl()
        o_proj = impl.o_proj

        impl._init_o_proj_tp_full_params()

        self.assertEqual(impl.o_proj_tp_weight.data_ptr(), o_proj.weight.data_ptr())
        self.assertEqual(
            impl.o_proj_tp_aclnn_input_params["aclnn_input_scale"].data_ptr(),
            o_proj.aclnn_input_scale.data_ptr(),
        )
        self.assertEqual(
            impl.o_proj_tp_input_sharded_quant_params["weight_scale_second"].data_ptr(),
            o_proj.weight_scale_second.data_ptr(),
        )
        self.assertEqual(
            impl.o_proj_tp_input_sharded_quant_params["weight_offset_second"].data_ptr(),
            o_proj.weight_offset_second.data_ptr(),
        )
        self.assertEqual(
            impl.o_proj_tp_input_sharded_quant_params["extra_input_scale"].data_ptr(),
            o_proj.extra_input_scale.data_ptr(),
        )
        self.assertNotIn("weight_scale", impl.o_proj_tp_input_sharded_quant_params)

    def test_o_proj_full_weight_forward_restores_tp_storage(self):
        impl = self._make_impl()
        impl._init_o_proj_tp_full_params()
        original_weight_ptr = impl.o_proj.weight.data_ptr()
        original_scale_ptr = impl.o_proj.weight_scale_second.data_ptr()
        full_weight_ptr = impl.o_proj_full_pool.data_ptr()
        full_scale_ptr = impl.o_proj_full_input_sharded_quant_params["weight_scale_second"].data_ptr()

        def _apply_with_full_weight(_attn_output):
            self.assertEqual(impl.o_proj.weight.data_ptr(), full_weight_ptr)
            self.assertEqual(impl.o_proj.weight_scale_second.data_ptr(), full_scale_ptr)
            return torch.ones(2, 4)

        impl._apply_o_proj_full_weight = MagicMock(side_effect=_apply_with_full_weight)

        output, require_o_proj_forward = impl._handle_o_proj_weight_switch_and_forward(
            attn_output=torch.randn(2, 3),
            output=torch.empty(2, 4),
            o_proj_full_handle=None,
            o_proj_full_param_handles=[],
            should_shard_weight=True,
        )

        self.assertEqual(impl.o_proj.weight.data_ptr(), original_weight_ptr)
        self.assertEqual(impl.o_proj.weight_scale_second.data_ptr(), original_scale_ptr)
        self.assertFalse(require_o_proj_forward)
        self.assertTrue(torch.equal(output, torch.ones(2, 4)))

    def test_no_indexer_full_o_proj_still_opens_gate_and_saves_layer(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.enable_dsa_cp = False
        impl.enable_dsa_cp_with_o_proj_tp = True
        impl.enable_sfa_prolog_v3 = False
        impl.enable_mlapo = False
        impl.enable_sp = False
        impl.has_indexer = False
        impl.skip_topk = True
        impl.use_sparse_c8_sfa = False
        impl.is_kv_producer = False
        impl.q_lora_rank = 8
        impl.kv_lora_rank = 4
        impl.qk_rope_head_dim = 2
        impl.layer_name = "layers.0.attn"

        q_c = MagicMock()
        kv_no_split = MagicMock()
        qkv_lora = MagicMock()
        qkv_lora.split.return_value = (q_c, kv_no_split)
        impl.fused_qkv_a_proj = MagicMock(return_value=(qkv_lora,))
        impl.q_a_layernorm = MagicMock(return_value=q_c)
        impl.exec_kv = MagicMock(return_value=(MagicMock(), MagicMock()))
        impl._q_proj_and_k_up_proj = MagicMock(return_value=(MagicMock(), MagicMock()))
        impl.rope_single = MagicMock(return_value=MagicMock())
        impl._record_dcp_query_gather_context = MagicMock()
        impl._get_indexcache_topk_indices = MagicMock(return_value=MagicMock())
        impl._execute_sparse_flash_attention_process = MagicMock(return_value=MagicMock())
        impl._v_up_proj = MagicMock(return_value=MagicMock())
        impl.o_proj = MagicMock()

        output = MagicMock()
        kv_cache = (MagicMock(), MagicMock())
        impl._compose_sfa_kv_cache = MagicMock(return_value=kv_cache)
        impl._handle_o_proj_weight_switch_and_forward = MagicMock(return_value=(output, False))

        attn_metadata = MagicMock()
        attn_metadata.dcp_context = None
        attn_metadata.num_input_tokens = 1

        with (
            patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector"),
            patch("vllm_ascend.attention.sfa_v1.record_attention_compute_start") as record_gate,
            patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector") as save_layer,
            patch("vllm_ascend.attention.sfa_v1.get_weight_prefetch_method", return_value=MagicMock()),
        ):
            result = impl.forward(
                layer_name=impl.layer_name,
                hidden_states=MagicMock(),
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        self.assertIs(result, output)
        record_gate.assert_called_once_with()
        save_layer.assert_called_once_with(impl.layer_name, list(kv_cache))
        impl.o_proj.assert_not_called()
