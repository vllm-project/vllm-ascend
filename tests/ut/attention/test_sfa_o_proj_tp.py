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

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, PreprocessType
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
        AscendSFAImpl.o_proj_full_pools.clear()

    def _make_impl(self, linear_method=None):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
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
        self.assertEqual(len(AscendSFAImpl.o_proj_full_pools), 2)

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

        output, require_o_proj_forward = impl._handle_o_proj_weight_switch_and_forward(
            attn_output=torch.randn(2, 8),
            output=torch.empty(2, 3),
            should_shard_weight=True,
        )

        self.assertEqual(impl.o_proj.weight.data_ptr(), original_weight_ptr)
        self.assertEqual(impl.o_proj.weight_scale.data_ptr(), original_scale_ptr)
        self.assertFalse(require_o_proj_forward)
        self.assertTrue(torch.equal(output, torch.ones(2, 3)))

    def test_enable_o_proj_switch_rejects_unsupported_method(self):
        impl = self._make_impl(_UnsupportedOProjLinearMethod())

        with self.assertRaisesRegex(RuntimeError, "TP weight-switch capable"):
            impl._enable_o_proj_tp_full_weight_switch()

    def test_no_indexer_full_o_proj_still_opens_gate_and_saves_layer(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.enable_dsa_cp = False
        impl.enable_dsa_cp_with_o_proj_tp = True
        impl.enable_sp = False
        impl.has_indexer = False
        impl.skip_topk = True
        impl.enable_sparse_sfa_c8 = False
        impl.is_kv_producer = False
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
        attn_metadata.dsa_cp_context = None
        attn_metadata.num_input_tokens = 1

        with (
            patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector"),
            patch("vllm_ascend.attention.sfa_v1.record_attention_compute_start") as record_gate,
            patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector") as save_layer,
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


class TestAscendSFAForwardOProjTP(TestBase):
    """_forward_o_proj_tp: static exchange buffers pin the HCCL collective
    shapes to potential_max_tokens, so uneven num_tokens across the OTP ranks
    cannot shape-mismatch (sync timeout), and the buffers keep a stable device
    address for ACL graph replay."""

    def _make_impl(self, tp_size, chunk, capacity):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        o_proj = torch.nn.Module()
        o_proj.input_size_per_partition = chunk
        o_proj.quant_method = MagicMock()
        o_proj.quant_method.apply = MagicMock(side_effect=lambda layer, x, bias=None: x * 2)
        impl.o_proj = o_proj
        return impl

    def _patch_env(self, tp_size, capacity):
        group = MagicMock()
        group.world_size = tp_size
        group.device_group = object()
        return [
            patch("vllm_ascend.attention.sfa_v1.get_otp_group", return_value=group),
            patch("vllm_ascend.attention.sfa_v1.get_potential_max_tokens", return_value=capacity),
        ]

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

        with (
            *self._patch_env(tp_size, capacity),
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
        for oproj_enabled, expect_collectives in ((True, True), (False, False)):
            with self.subTest(oproj_enabled=oproj_enabled):
                impl = self._make_impl(tp_size, chunk, capacity)
                impl.o_proj.input_size = tp_size * chunk
                group = MagicMock()
                group.world_size = tp_size
                group.device_group = object()
                with (
                    patch("vllm_ascend.attention.sfa_v1.oproj_tp_enable", return_value=oproj_enabled),
                    patch("vllm_ascend.attention.sfa_v1.get_otp_group", return_value=group),
                    patch("vllm_ascend.attention.sfa_v1.get_potential_max_tokens", return_value=capacity),
                    patch("torch.distributed.all_to_all_single") as mock_a2a,
                    patch("torch.distributed.reduce_scatter_tensor") as mock_rs,
                ):
                    mock_a2a.side_effect = lambda recv, send, group: recv.copy_(send)
                    mock_rs.side_effect = lambda out, inp, group: out.copy_(inp[: out.shape[0]])

                    impl.forward("layer0", torch.randn(3, 8), None, None, output=torch.zeros(3, 8))

                if expect_collectives:
                    mock_a2a.assert_called_once()
                    mock_rs.assert_called_once()
                else:
                    mock_a2a.assert_not_called()
                    mock_rs.assert_not_called()
