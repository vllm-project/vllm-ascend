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
from unittest.mock import MagicMock

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
