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

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import get_cp_local_query_key_lens
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl
from vllm_ascend.attention.sfa_v1 import PreprocessType, SFAForwardContext
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


class _UnsupportedOProjLinearMethod(WeightSwitchMixin):
    pass


class TestAscendSFAOProjWeightSwitch(TestBase):
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
            self.reduce_results = True

    def setUp(self):
        AscendSFADSACPImpl.o_proj_full_pools.clear()

    def _make_impl(self, linear_method=None):
        impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
        impl.layerwise_kv_cache_hook = None
        impl.tp_size = 2
        impl.o_proj = self._OProj(linear_method or _OProjLinearMethod())
        impl._o_proj_weight_switch_enabled = False
        impl.o_proj_weight_state = None
        impl.o_proj_weight_switch_config = WeightSwitchConfig(
            group=object(),
            world_size=2,
            rank=0,
        )
        return impl

    def test_enable_o_proj_switch_uses_mixin_state_and_is_idempotent(self):
        impl = self._make_impl()
        original_weight_ptr = impl.o_proj.weight.data_ptr()
        original_scale_ptr = impl.o_proj.weight_scale.data_ptr()

        impl._enable_o_proj_full_weight_switch()

        state = impl.o_proj_weight_state
        self.assertTrue(impl._o_proj_weight_switch_enabled)
        self.assertEqual(state.gather_parts["weight"].local_tensor.data_ptr(), original_weight_ptr)
        self.assertEqual(state.gather_parts["weight_scale"].local_tensor.data_ptr(), original_scale_ptr)
        self.assertEqual(state.gather_parts["weight"].full_tensor.shape, (8, 3))
        self.assertEqual(state.gather_parts["weight_scale"].full_tensor.shape, (4, 3))
        self.assertEqual(len(AscendSFADSACPImpl.o_proj_full_pools), 2)

        impl._enable_o_proj_full_weight_switch()
        self.assertIs(impl.o_proj_weight_state, state)

    def test_o_proj_full_weight_forward_restores_local_storage(self):
        impl = self._make_impl()
        impl._enable_o_proj_full_weight_switch()
        state = impl.o_proj_weight_state
        original_weight_ptr = impl.o_proj.weight.data_ptr()
        original_scale_ptr = impl.o_proj.weight_scale.data_ptr()
        full_weight_ptr = state.gather_parts["weight"].full_tensor.data_ptr()
        full_scale_ptr = state.gather_parts["weight_scale"].full_tensor.data_ptr()

        def _apply_with_full_weight(_attn_output):
            self.assertEqual(impl.o_proj.weight.data_ptr(), full_weight_ptr)
            self.assertEqual(impl.o_proj.weight_scale.data_ptr(), full_scale_ptr)
            return torch.ones(2, 3)

        impl._apply_o_proj_full_weight = MagicMock(side_effect=_apply_with_full_weight)

        impl.enable_dsa_cp_full_o_proj = True
        gathered_output = torch.cat((torch.ones(2, 3), torch.full((2, 3), 2.0)))
        tp_group = SimpleNamespace(all_gather=MagicMock(return_value=gathered_output))
        with patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group",
            return_value=tp_group,
        ):
            output = impl._finalize_o_proj(
                attn_output=torch.randn(2, 8),
                output=torch.empty(3, 3),
                gather_full_o_proj=True,
            )

        self.assertEqual(impl.o_proj.weight.data_ptr(), original_weight_ptr)
        self.assertEqual(impl.o_proj.weight_scale.data_ptr(), original_scale_ptr)
        tp_group.all_gather.assert_called_once()
        self.assertTrue(torch.equal(output, gathered_output[:3]))

    def test_o_proj_full_weight_stages_local_result_for_upstream_sp(self):
        impl = self._make_impl()
        impl._enable_o_proj_full_weight_switch()
        impl.enable_dsa_cp_full_o_proj = True
        impl.o_proj.reduce_results = False
        local_output = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        impl._apply_o_proj_full_weight = MagicMock(return_value=local_output)
        tp_group = SimpleNamespace(rank_in_group=1, all_gather=MagicMock())

        with patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=tp_group):
            output = impl._finalize_o_proj(
                attn_output=torch.randn(2, 8),
                output=torch.full((3, 3), -1.0),
                gather_full_o_proj=True,
            )

        tp_group.all_gather.assert_not_called()
        torch.testing.assert_close(
            output,
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 4.0, 5.0]]),
        )

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

        with self.assertRaisesRegex(RuntimeError, "weight-switch capable"):
            impl._enable_o_proj_full_weight_switch()

    def test_mixed_sp_input_matches_replicated_path_on_every_rank(self):
        impl = self._make_impl()
        impl.tp_size = 4
        impl.enable_dsa_cp_full_o_proj = True
        query_starts = torch.tensor([0, 1, 2, 3, 4, 10, 14])
        seq_lens = torch.tensor([10, 20, 30, 40, 16, 24])
        # N=14 exercises TP padding; N=16 retains two existing graph/DP rows.
        for num_tokens in (14, 16):
            full_input = torch.arange(num_tokens * 4).reshape(num_tokens, 4)
            padded = torch.nn.functional.pad(full_input, (0, 0, 0, 16 - num_tokens))
            for rank in range(4):
                start, end = rank * 4, rank * 4 + 4
                query_lens, key_lens = get_cp_local_query_key_lens(query_starts, query_starts[1:], seq_lens, start, end)
                context = SimpleNamespace(
                    num_tokens=num_tokens,
                    num_tokens_pad=16,
                    local_start=start,
                    local_end_with_pad=end,
                    actual_seq_lengths_query=query_lens,
                    actual_seq_lengths_key=key_lens,
                    slot_mapping_cp=torch.arange(start, end),
                )
                metadata = SimpleNamespace(
                    num_input_tokens=num_tokens,
                    num_actual_tokens=14,
                    num_decode_tokens=4,
                    attn_state=AscendAttentionState.ChunkedPrefill,
                    dsa_cp_context=context,
                )
                local_input = padded[start:end].clone()
                if num_tokens == 14 and rank == 3:
                    local_input[2:].fill_(99)
                original = local_input.clone()
                with patch(
                    "vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group",
                    return_value=SimpleNamespace(rank_in_group=rank),
                ):
                    actual = impl._prepare_sequence_parallel_hidden_states(local_input, metadata)
                expected = impl._prepare_native_hidden_states(full_input, metadata)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(local_input, original)
                forward_context = impl._get_parallel_forward_context(metadata, num_tokens, actual)
                # Rank 0 owns only decode queries, but must join the global mixed path.
                self.assertTrue(forward_context.gather_full_o_proj)
                if rank == 0:
                    torch.testing.assert_close(query_lens, torch.tensor([1, 2, 3, 4, 4, 4]))
                if num_tokens == 16 or rank < 3:
                    self.assertEqual(actual.data_ptr(), local_input.data_ptr())

    def test_local_full_projection_matches_legacy_zero_staging(self):
        impl = self._make_impl()
        impl._enable_o_proj_full_weight_switch()
        impl.enable_dsa_cp_full_o_proj = True
        impl.o_proj.reduce_results = False
        original_weight_ptr = impl.o_proj.weight.data_ptr()
        for rank in range(2):
            projected = torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank * 10
            impl._apply_o_proj_full_weight = MagicMock(return_value=projected)
            group = SimpleNamespace(rank_in_group=rank, all_gather=MagicMock())
            metadata = SimpleNamespace(dsa_cp_context=SimpleNamespace(num_tokens=3, local_start=rank * 2))
            with (
                patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=group),
                patch("torch.distributed.reduce_scatter_tensor") as scatter,
            ):
                staged = impl._finalize_o_proj(torch.empty(2, 8), torch.empty(3, 3), True)
                expected = torch.nn.functional.pad(staged, (0, 0, 0, 1))[rank * 2 : rank * 2 + 2]
                actual = impl._finalize_sequence_parallel_o_proj(
                    torch.empty(2, 8), torch.empty(2, 3), True, metadata, 1.0
                )
                scaled = impl._finalize_sequence_parallel_o_proj(
                    torch.empty(2, 8), torch.empty(2, 3), True, metadata, 0.5
                )
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(scaled, expected * 0.5)
            group.all_gather.assert_not_called()
            scatter.assert_not_called()
            self.assertEqual(impl.o_proj.weight.data_ptr(), original_weight_ptr)

    def test_local_sharded_projection_preserves_tp_sum_and_zero_padding(self):
        for full_o_proj in (False, True):
            for tp_size, num_tokens in ((2, 3), (4, 6), (4, 8)):
                with self.subTest(full_o_proj=full_o_proj, tp_size=tp_size, num_tokens=num_tokens):
                    self._check_local_sharded_projection(full_o_proj, tp_size, num_tokens)

    def _check_local_sharded_projection(self, full_o_proj, tp_size, num_tokens):
        impl = self._make_impl()
        impl.enable_dsa_cp_full_o_proj = full_o_proj
        impl.tp_size = tp_size
        impl.num_heads = 1
        impl.v_head_dim = 2
        local_tokens = (num_tokens + tp_size - 1) // tp_size
        padded_tokens = local_tokens * tp_size
        attention = torch.arange(padded_tokens * tp_size * 2, dtype=torch.float32).reshape(padded_tokens, tp_size * 2)
        weight = torch.arange(3 * tp_size * 2, dtype=torch.float32).reshape(3, tp_size * 2)
        complete = attention @ weight.T
        expected = complete.clone()
        expected[num_tokens:].zero_()
        for rank in range(tp_size):
            local_attention = attention[rank * local_tokens : (rank + 1) * local_tokens]
            head_shard = attention[:, rank * 2 : rank * 2 + 2]
            weight_shard = weight[:, rank * 2 : rank * 2 + 2]
            impl.o_proj = MagicMock(side_effect=lambda x, weight_shard=weight_shard: (x @ weight_shard.T,))
            impl.o_proj.reduce_results = False
            impl.o_proj.custom_op = None
            impl.o_proj.tp_size = tp_size
            output = torch.empty(local_tokens, 3)
            metadata = SimpleNamespace(
                dsa_cp_context=SimpleNamespace(num_tokens=num_tokens, local_start=rank * local_tokens)
            )
            group = SimpleNamespace(device_group=object())

            def exchange(recv, send, *, group, local_attention=local_attention, head_shard=head_shard):
                # Each destination receives its heads from every token owner.
                torch.testing.assert_close(send, torch.cat(local_attention.split(2, dim=-1), dim=0))
                recv.copy_(head_shard)

            def scatter(out, input_, *, group, rank=rank, head_shard=head_shard, weight_shard=weight_shard):
                torch.testing.assert_close(input_, head_shard @ weight_shard.T)
                out.copy_(complete[rank * local_tokens : (rank + 1) * local_tokens])

            with (
                patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=group),
                patch("torch.distributed.all_to_all_single", side_effect=exchange) as a2a,
                patch("torch.distributed.reduce_scatter_tensor", side_effect=scatter) as reduce,
            ):
                actual = impl._finalize_sequence_parallel_o_proj(local_attention, output, False, metadata, 1.0)
            self.assertIs(actual, output)
            self.assertIs(reduce.call_args.args[0], output)
            torch.testing.assert_close(actual, expected[rank * local_tokens : (rank + 1) * local_tokens])
            a2a.assert_called_once()
            reduce.assert_called_once()

    def test_local_fp16_projection_scales_before_tp_sum(self):
        impl = self._make_impl()
        impl.enable_dsa_cp_full_o_proj = True
        impl.num_heads = 1
        impl.v_head_dim = 2
        partial = torch.full((4, 3), 50000, dtype=torch.float16)
        expected = partial.clone()
        expected[3:].zero_()
        for rank in range(2):
            impl.o_proj = MagicMock(return_value=(partial.clone(),))
            impl.o_proj.reduce_results = False
            output = torch.empty(2, 3, dtype=torch.float16)
            metadata = SimpleNamespace(dsa_cp_context=SimpleNamespace(num_tokens=3, local_start=rank * 2))

            def scatter(out, input_, *, group, rank=rank):
                other = partial * 0.5
                other[3:].zero_()
                out.copy_((input_ + other)[rank * 2 : rank * 2 + 2])

            with (
                patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group"),
                patch("torch.distributed.all_to_all_single", side_effect=lambda recv, send, **kw: recv.copy_(send)),
                patch("torch.distributed.reduce_scatter_tensor", side_effect=scatter) as reduce,
            ):
                actual = impl._finalize_sequence_parallel_o_proj(torch.ones(2, 4), output, False, metadata, 0.5)
            torch.testing.assert_close(actual, expected[rank * 2 : rank * 2 + 2])
            torch.testing.assert_close(reduce.call_args.args[1][0], partial[0] * 0.5)

    def test_legacy_custom_projection_keeps_its_own_communication(self):
        impl = self._make_impl()
        impl.enable_dsa_cp_full_o_proj = False
        hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        impl.o_proj = MagicMock(return_value=(expected,))
        impl.o_proj.custom_op = object()
        impl.o_proj.tp_size = 2
        output = torch.empty_like(expected)
        with (
            patch("torch.distributed.all_to_all_single") as exchange,
            patch("torch.distributed.reduce_scatter_tensor") as reduce,
        ):
            actual = impl._finalize_o_proj(hidden, output, False)
        self.assertIs(actual, output)
        torch.testing.assert_close(actual, expected)
        impl.o_proj.assert_called_once_with(hidden)
        exchange.assert_not_called()
        reduce.assert_not_called()

    def test_local_input_rejects_inconsistent_layout_metadata(self):
        impl = self._make_impl()
        for invalid in (
            {"num_tokens": 2},
            {"num_tokens_pad": 6},
            {"local_start": 2},
            {"local_end_with_pad": 4},
        ):
            context = dict(num_tokens=3, num_tokens_pad=4, local_start=0, local_end_with_pad=2)
            context.update(invalid)
            metadata = SimpleNamespace(num_input_tokens=3, dsa_cp_context=SimpleNamespace(**context))
            with (
                self.subTest(invalid=invalid),
                patch(
                    "vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group",
                    return_value=SimpleNamespace(rank_in_group=0),
                ),
                self.assertRaisesRegex(RuntimeError, "sequence-parallel shard"),
            ):
                impl._prepare_sequence_parallel_hidden_states(torch.ones(2, 4), metadata)

    def test_no_indexer_full_o_proj_still_opens_gate_and_saves_layer(self):
        impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
        impl.layerwise_kv_cache_hook = None
        impl.enable_dsa_cp_full_o_proj = True
        impl.enable_sp = False
        impl.has_indexer = False
        impl.skip_topk = True
        impl.enable_sparse_sfa_c8 = False
        impl.enable_sparse_sfa_turboquant = False
        impl.is_kv_producer = True
        impl.preprocess_type = PreprocessType.NATIVE
        impl.tp_size = 2
        impl.q_lora_rank = 8
        impl.kv_lora_rank = 4
        impl.qk_rope_head_dim = 2
        impl.g_proj = None
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
        impl._prepare_kv_for_parallel = MagicMock(return_value=(None, []))
        impl._store_parallel_kv = MagicMock(return_value=(None, None))
        impl._get_indexcache_topk_indices = MagicMock(return_value=MagicMock())
        impl._execute_sparse_flash_attention_process = MagicMock(return_value=MagicMock())
        attn_output = MagicMock()
        impl._v_up_proj = MagicMock(return_value=attn_output)
        impl.o_proj = MagicMock()
        impl._prepare_native_hidden_states = MagicMock(side_effect=lambda hidden_states, _: hidden_states)

        hidden_states = torch.empty(2, 1)
        output = torch.full((2, 1), 123.0)
        kv_cache = (MagicMock(), MagicMock())
        impl._compose_sfa_kv_cache = MagicMock(return_value=kv_cache)
        impl._finalize_o_proj = MagicMock(side_effect=lambda attn_output, output, gather: output.fill_(1.0))

        attn_metadata = MagicMock()
        attn_metadata.dcp_context = None
        attn_metadata.dsa_cp_context = None
        attn_metadata.num_input_tokens = 1
        attn_metadata.num_actual_tokens = 1
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
            patch("vllm_ascend.attention.sfa_v1.attention_transfer_window") as transfer_window,
            patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector") as save_layer,
            patch("vllm_ascend.attention.sfa_v1.notify_kv_cache_written") as notify_cache_written,
        ):
            result = impl.forward(
                layer_name=impl.layer_name,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        self.assertIs(result, output)
        impl._finalize_o_proj.assert_called_once()
        finalize_args = impl._finalize_o_proj.call_args.args
        self.assertIs(finalize_args[0], attn_output)
        self.assertEqual(finalize_args[1].shape, (1, 1))
        self.assertEqual(finalize_args[1].data_ptr(), output.data_ptr())
        self.assertTrue(finalize_args[2])
        torch.testing.assert_close(result, torch.tensor([[1.0], [123.0]]))
        projected_input = impl.fused_qkv_a_proj.call_args.args[0]
        self.assertEqual(projected_input.shape, (1, 1))
        self.assertEqual(projected_input.data_ptr(), hidden_states.data_ptr())
        notify_cache_written.assert_called_once_with(impl.layer_name)
        transfer_window.assert_called_once_with()
        save_layer.assert_called_once_with(impl.layer_name, list(kv_cache))
        impl._prepare_native_hidden_states.assert_called_once()
        impl.o_proj.assert_not_called()
