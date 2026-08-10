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
from contextlib import ExitStack, contextmanager
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


class TestAscendSFAForwardOProjTP(TestBase):
    """_forward_o_proj_tp: static exchange buffers pin the HCCL collective
    shapes to potential_max_tokens, so uneven num_tokens across the OTP ranks
    cannot shape-mismatch (sync timeout), and the buffers keep a stable device
    address for ACL graph replay."""

    def _make_impl(self, tp_size, chunk, capacity):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        o_proj = torch.nn.Module()
        o_proj.input_size_per_partition = chunk
        o_proj.skip_bias_add = False
        o_proj.bias = None
        o_proj.quant_method = MagicMock()
        o_proj.quant_method.apply = MagicMock(side_effect=lambda layer, x, bias=None: x * 2)
        impl.o_proj = o_proj
        return impl

    @contextmanager
    def _patch_env(self, tp_size, capacity, *extra_patches):
        """Enter the OTP-group and potential-max-tokens patches, then any extra
        patches, so the call site can use a plain ``with`` (mypy 1.11 rejects
        starred ``with`` and a list has no context manager protocol)."""
        group = MagicMock()
        group.world_size = tp_size
        group.rank_in_group = 0
        group.device_group = object()
        with ExitStack() as stack:
            stack.enter_context(patch("vllm_ascend.attention.sfa_v1.get_otp_group", return_value=group))
            stack.enter_context(patch("vllm_ascend.attention.sfa_v1.get_potential_max_tokens", return_value=capacity))
            for patcher in extra_patches:
                stack.enter_context(patcher)
            yield

    def test_exchange_shapes_and_buffers_pinned_to_capacity(self):
        """Exchange shapes and static buffer addresses are pinned to capacity:
        uneven num_tokens across the OTP ranks cannot shape-mismatch, and ACL
        graph replay keeps stable addresses."""
        tp_size, chunk, capacity = 2, 4, 8
        impl = self._make_impl(tp_size, chunk, capacity)
        a2a_send_shapes, rs_in_shapes = [], []

        def _a2a(recv, send, group):
            a2a_send_shapes.append(tuple(send.shape))
            recv.copy_(send)

        def _rs(out, inp, group):
            rs_in_shapes.append(tuple(inp.shape))
            out.copy_(inp[: out.shape[0]])

        with self._patch_env(
            tp_size,
            capacity,
            patch("torch.distributed.all_to_all_single", side_effect=_a2a),
            patch("torch.distributed.reduce_scatter_tensor", side_effect=_rs),
        ):
            send_ptr = None
            for num_tokens in (3, capacity):
                attn_output = torch.randn(num_tokens, tp_size * chunk)
                output = torch.zeros(num_tokens, chunk)
                impl._forward_o_proj_tp(attn_output, output)
                # With the identity-exchange mock, rank-local output rows are
                # 2 * attn_output[:, :chunk] (this rank's own shard).
                self.assertTrue(torch.equal(output, attn_output[:, :chunk] * 2))
                if send_ptr is None:
                    send_ptr = impl._o_proj_tp_send_buf.data_ptr()
                    recv_ptr = impl._o_proj_tp_recv_buf.data_ptr()
                    rs_ptr = impl._o_proj_tp_rs_out_buf.data_ptr()
                    self.assertEqual(impl._o_proj_tp_send_buf.shape, (tp_size, capacity, chunk))
                    self.assertEqual(impl._o_proj_tp_rs_out_buf.shape, (capacity, chunk))
                else:
                    # Buffers are allocated once and reused across calls.
                    self.assertEqual(impl._o_proj_tp_send_buf.data_ptr(), send_ptr)
                    self.assertEqual(impl._o_proj_tp_recv_buf.data_ptr(), recv_ptr)
                    self.assertEqual(impl._o_proj_tp_rs_out_buf.data_ptr(), rs_ptr)

            # Exchange shapes never depend on num_tokens.
            self.assertEqual(a2a_send_shapes, [(tp_size * capacity * chunk,)] * 2)
            self.assertEqual(rs_in_shapes, [(tp_size * capacity, chunk)] * 2)

    def test_capacity_exceeded_raises(self):
        tp_size, chunk, capacity = 2, 4, 8
        impl = self._make_impl(tp_size, chunk, capacity)
        with self._patch_env(tp_size, capacity), self.assertRaises(ValueError):
            impl._forward_o_proj_tp(torch.randn(capacity + 1, tp_size * chunk), torch.zeros(capacity + 1, chunk))

    def test_profiling_run_oproj_tp_collectives_dispatch(self):
        """attn_metadata=None (profiling / dummy accompanying ranks) must still
        run the OTP collectives on zero input when oproj_tp is on, otherwise
        the group deadlocks when other DP ranks execute the real o_proj;
        without oproj_tp it keeps the plain fill_(0). Mirrors dsa_v1.forward()."""
        tp_size, chunk, capacity = 2, 4, 8
        impl = self._make_impl(tp_size, chunk, capacity)
        impl.o_proj.input_size = tp_size * chunk
        group = MagicMock()
        group.world_size = tp_size
        group.rank_in_group = 0
        group.device_group = object()
        with (
            patch("vllm_ascend.attention.sfa_v1.oproj_tp_enable", return_value=True),
            patch("vllm_ascend.attention.sfa_v1.get_otp_group", return_value=group),
            patch("vllm_ascend.attention.sfa_v1.get_potential_max_tokens", return_value=capacity),
            patch("torch.distributed.all_to_all_single") as mock_a2a,
            patch("torch.distributed.reduce_scatter_tensor") as mock_rs,
        ):
            mock_a2a.side_effect = lambda recv, send, group: recv.copy_(send)
            mock_rs.side_effect = lambda out, inp, group: out.copy_(inp[: out.shape[0]])

            impl.forward("layer0", torch.randn(3, 8), None, None, output=torch.zeros(3, 8))

        mock_a2a.assert_called_once()
        mock_rs.assert_called_once()
