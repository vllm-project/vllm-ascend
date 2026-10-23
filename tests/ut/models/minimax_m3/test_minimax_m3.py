# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.minimax_m3 import (
    MiniMaxM3Attention,
    MiniMaxM3MoE,
    _get_rope_parameters,
    _sparse_attention_layer_ids,
)
from vllm_ascend.models.minimax_m3.minimax_m3 import (
    MiniMaxM3DecoderLayer,
    MiniMaxM3MLP,
    MiniMaxM3Model,
    MiniMaxM3SwiGLUOAI,
)


class _FakeQKVProj(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        num_tokens = hidden_states.shape[0]
        qkv = torch.arange(
            num_tokens * 6,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).reshape(num_tokens, 6)
        return qkv, None


class _IdentityRotary(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k


class _AssertContiguousAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.saw_contiguous_v = False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        self.saw_contiguous_v = v.is_contiguous()
        if not self.saw_contiguous_v:
            raise AssertionError(f"Expected contiguous v, got stride {v.stride()}")
        return q + k + v


class _FakeOProj(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden_states, None


def _make_attention() -> MiniMaxM3Attention:
    attention = MiniMaxM3Attention.__new__(MiniMaxM3Attention)
    nn.Module.__init__(attention)
    attention.q_size = 2
    attention.kv_size = 2
    attention.head_dim = 2
    attention.qkv_proj = _FakeQKVProj()
    attention.q_norm = nn.Identity()
    attention.k_norm = nn.Identity()
    attention.rotary_emb = _IdentityRotary()
    attention.attn = _AssertContiguousAttention()
    attention.o_proj = _FakeOProj()
    return attention


class _PairNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _RecordingAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_shapes: list[torch.Size] = []

    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del positions
        self.input_shapes.append(hidden_states.shape)
        return hidden_states


def _make_decoder_layer(*, use_sequence_parallel: bool, is_moe: bool = True) -> MiniMaxM3DecoderLayer:
    layer = MiniMaxM3DecoderLayer.__new__(MiniMaxM3DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = use_sequence_parallel
    layer.is_layer_sparse = is_moe
    layer.input_layernorm = _PairNorm()
    layer.post_attention_layernorm = _PairNorm()
    layer.self_attn = _RecordingAttention()
    if is_moe:
        layer.block_sparse_moe = nn.Identity()
    else:
        layer.mlp = nn.Identity()
    return layer


class TestMiniMaxM3Modeling(unittest.TestCase):
    @patch("vllm_ascend.models.minimax_m3.minimax_m3.sp_shard")
    @patch("vllm_ascend.models.minimax_m3.minimax_m3.sp_reduce_scatter")
    def test_decoder_enters_sequence_parallel(
        self,
        reduce_scatter: Mock,
        shard: Mock,
    ) -> None:
        layer = _make_decoder_layer(use_sequence_parallel=True)
        hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        residual = hidden_states + 10
        reduce_scatter.side_effect = lambda value: torch.nn.functional.pad(value, (0, 0, 0, 1))[:2]
        shard.side_effect = lambda value: torch.nn.functional.pad(value, (0, 0, 0, 1))[:2]

        output, output_residual = layer(
            positions=torch.arange(3),
            hidden_states=hidden_states,
            residual=residual,
        )

        reduce_scatter.assert_called_once()
        shard.assert_called_once()
        torch.testing.assert_close(shard.call_args.args[0], residual)
        self.assertEqual(layer.self_attn.input_shapes, [torch.Size([3, 2])])
        self.assertEqual(output.shape, torch.Size([2, 2]))
        self.assertEqual(output_residual.shape, torch.Size([2, 2]))

    @patch("vllm_ascend.models.minimax_m3.minimax_m3.sp_reduce_scatter")
    @patch("vllm_ascend.models.minimax_m3.minimax_m3.sp_all_gather")
    def test_consecutive_moe_keeps_residual_sharded(
        self,
        all_gather: Mock,
        reduce_scatter: Mock,
    ) -> None:
        layer = _make_decoder_layer(use_sequence_parallel=True)
        hidden_states = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        residual = hidden_states + 10
        all_gather.side_effect = lambda value: torch.cat((value, value + 100), dim=0)
        reduce_scatter.side_effect = lambda value: value[:2]

        output, output_residual = layer(
            positions=torch.arange(3),
            hidden_states=hidden_states,
            residual=residual,
            input_is_sequence_parallel=True,
        )

        # Only the normalized attention input is gathered. The residual stays
        # sharded and the attention partial output is reduce-scattered.
        all_gather.assert_called_once()
        reduce_scatter.assert_called_once()
        self.assertEqual(layer.self_attn.input_shapes, [torch.Size([3, 2])])
        torch.testing.assert_close(output_residual, residual)
        self.assertEqual(output.shape, torch.Size([2, 2]))

    @patch("vllm_ascend.models.minimax_m3.minimax_m3.sp_reduce_scatter")
    @patch("vllm_ascend.models.minimax_m3.minimax_m3.sp_all_gather")
    def test_dense_decoder_keeps_sequence_parallel_region(
        self,
        all_gather: Mock,
        reduce_scatter: Mock,
    ) -> None:
        layer = _make_decoder_layer(use_sequence_parallel=True, is_moe=False)
        hidden_states = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        residual = hidden_states + 10
        all_gather.side_effect = lambda value: torch.cat((value, value + 100), dim=0)
        reduce_scatter.side_effect = lambda value: value[:2]

        output, output_residual = layer(
            positions=torch.arange(3),
            hidden_states=hidden_states,
            residual=residual,
            input_is_sequence_parallel=True,
        )

        all_gather.assert_called_once()
        reduce_scatter.assert_called_once()
        self.assertEqual(layer.self_attn.input_shapes, [torch.Size([3, 2])])
        self.assertEqual(output.shape, torch.Size([2, 2]))
        self.assertEqual(output_residual.shape, torch.Size([2, 2]))

    @patch("vllm_ascend.models.minimax_m3.minimax_m3.sp_reduce_scatter")
    @patch("vllm_ascend.models.minimax_m3.minimax_m3.sp_all_gather")
    def test_dense_mlp_gathers_and_scatters_sequence_shards(
        self,
        all_gather: Mock,
        reduce_scatter: Mock,
    ) -> None:
        class _Linear(nn.Module):
            def forward(self, value):
                return value + 1, None

        mlp = MiniMaxM3MLP.__new__(MiniMaxM3MLP)
        nn.Module.__init__(mlp)
        mlp.is_sequence_parallel = True
        mlp.gate_up_proj = _Linear()
        mlp.act_fn = nn.Identity()
        mlp.down_proj = _Linear()
        hidden_states = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        all_gather.side_effect = lambda value: torch.cat((value, value + 100), dim=0)
        reduce_scatter.side_effect = lambda value: value[:2]

        output = mlp(hidden_states)

        all_gather.assert_called_once_with(hidden_states)
        reduce_scatter.assert_called_once()
        torch.testing.assert_close(output, hidden_states + 2)

    @patch("vllm_ascend.models.minimax_m3.minimax_m3.sp_all_gather")
    @patch("vllm_ascend.models.minimax_m3.minimax_m3.get_pp_group")
    def test_pipeline_boundary_restores_full_token_layout(
        self,
        get_pp_group: Mock,
        all_gather: Mock,
    ) -> None:
        class _SequenceParallelLayer(nn.Module):
            use_sequence_parallel = True

            def forward(self, positions, hidden_states, residual, input_is_sequence_parallel=False):
                del positions, input_is_sequence_parallel
                return hidden_states[:2], hidden_states[:2] if residual is None else residual[:2]

        model = MiniMaxM3Model.__new__(MiniMaxM3Model)
        nn.Module.__init__(model)
        model.do_not_compile = True
        model.start_layer = 0
        model.end_layer = 1
        model.layers = nn.ModuleList([_SequenceParallelLayer()])
        model.aux_hidden_state_layers = set()
        model._maybe_add_hidden_state = MethodType(lambda self, states, *_args: states, model)
        get_pp_group.return_value = SimpleNamespace(is_first_rank=True, is_last_rank=False)
        all_gather.side_effect = lambda value: torch.cat((value, value + 100), dim=0)

        output = model(
            input_ids=None,
            positions=torch.arange(3),
            intermediate_tensors=None,
            inputs_embeds=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        )

        self.assertEqual(all_gather.call_count, 2)
        self.assertEqual(output["hidden_states"].shape, torch.Size([3, 2]))
        self.assertEqual(output["residual"].shape, torch.Size([3, 2]))

    def test_mxfp8_swiglu_uses_small_ops_then_dynamic_quant(self) -> None:
        activation = MiniMaxM3SwiGLUOAI(
            alpha=1.702,
            beta=1.0,
            limit=7.0,
            use_mx_quant=True,
        )
        gate_up = torch.tensor(
            [[-2.0, 8.0, -9.0, 3.0]],
            dtype=torch.bfloat16,
        )
        quantized = torch.randn(1, 2)
        output_scale = torch.randn(1, 1)

        with patch.object(
            DeviceOperator,
            "npu_dynamic_quant",
            return_value=(quantized, output_scale),
        ) as dynamic_quant:
            output, actual_scale = activation(gate_up)

        gate = gate_up[..., :2].clamp(max=7.0)
        up = gate_up[..., 2:].clamp(min=-7.0, max=7.0)
        expected_activation = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
        actual_activation = dynamic_quant.call_args.args[0]
        torch.testing.assert_close(actual_activation, expected_activation)
        self.assertEqual(dynamic_quant.call_args.kwargs["act_quant_type"], torch.float8_e4m3fn)
        self.assertTrue(dynamic_quant.call_args.kwargs["use_mxfp_quant"])
        self.assertIs(output, quantized)
        self.assertIs(actual_scale, output_scale)

    def test_moe_passes_swigluoai_config_during_construction(self) -> None:
        config = PretrainedConfig(
            hidden_size=64,
            intermediate_size=32,
            num_local_experts=8,
            num_experts_per_tok=2,
            scoring_func="sigmoid",
            n_shared_experts=1,
            use_routing_bias=False,
            routed_scaling_factor=1.0,
            swiglu_limit=7.0,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        )
        gate = nn.Identity()
        gate.out_dtype = torch.float32
        experts = nn.Identity()
        experts.moe_config = Mock()
        parallel_config = Mock()
        parallel_config.enable_eplb = False
        parallel_config.eplb_config.num_redundant_experts = 0
        ep_group = Mock()
        ep_group.device_group.size.return_value = 1
        ep_group.rank_in_group = 0

        with (
            patch(
                "vllm_ascend.models.minimax_m3.minimax_m3.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm_ascend.models.minimax_m3.minimax_m3.get_ep_group",
                return_value=ep_group,
            ),
            patch(
                "vllm_ascend.models.minimax_m3.minimax_m3.MiniMaxM3MLP",
                return_value=nn.Identity(),
            ),
            patch(
                "vllm_ascend.models.minimax_m3.minimax_m3.GateLinear",
                return_value=gate,
            ),
            patch(
                "vllm_ascend.models.minimax_m3.minimax_m3.FusedMoEFactory",
                return_value=experts,
            ) as fused_moe,
        ):
            MiniMaxM3MoE(
                config=config,
                parallel_config=parallel_config,
                prefix="model.layers.0.block_sparse_moe",
                is_sequence_parallel=True,
            )

        kwargs = fused_moe.call_args.kwargs
        self.assertTrue(kwargs["is_sequence_parallel"])
        self.assertEqual(kwargs["activation"], "swigluoai_uninterleave")
        self.assertEqual(kwargs["swiglu_alpha"], config.swiglu_alpha)
        self.assertEqual(kwargs["swiglu_beta"], config.swiglu_beta)

    def test_attention_makes_value_states_contiguous(self) -> None:
        attention = _make_attention()
        hidden_states = torch.zeros(3, 4)
        positions = torch.arange(3)

        output = attention(positions=positions, hidden_states=hidden_states)

        self.assertTrue(attention.attn.saw_contiguous_v)
        self.assertEqual(output.shape, (3, 2))

    def test_sparse_attention_layer_ids_ignores_missing_config(self) -> None:
        self.assertEqual(_sparse_attention_layer_ids(PretrainedConfig()), set())

        config = PretrainedConfig(sparse_attention_config={"sparse_attention_freq": [0, 1, 0, 2]})

        self.assertEqual(_sparse_attention_layer_ids(config), {1, 3})

    def test_rope_parameters_are_copied_from_config(self) -> None:
        config = PretrainedConfig(
            rope_parameters={
                "rope_theta": 1000000,
                "partial_rotary_factor": 0.5,
            }
        )

        rope_parameters = _get_rope_parameters(config)
        assert rope_parameters is not None
        rope_parameters["partial_rotary_factor"] = 1.0

        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 0.5)


if __name__ == "__main__":
    unittest.main()
