# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import vllm
from torch import nn
from transformers import PretrainedConfig

from vllm_ascend.models.minimax_m3 import (
    MiniMaxM3Attention,
    MiniMaxM3MoE,
    MiniMaxVLVisionModel,
    _get_rope_parameters,
    _sparse_attention_layer_ids,
    minimax_m3_model,
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


class TestMiniMaxM3Modeling(unittest.TestCase):
    def test_vision_tower_uses_vllm_common_implementation(self) -> None:
        source_file = inspect.getsourcefile(MiniMaxVLVisionModel)
        self.assertIsNotNone(source_file)
        self.assertIsNotNone(vllm.__file__)
        expected_source = Path(vllm.__file__).resolve().parent / "models" / "minimax_m3" / "common" / "vision_tower.py"
        self.assertTrue(Path(source_file).samefile(expected_source))

    def test_moe_wrapper_reuses_vllm_native_implementation(self) -> None:
        config = PretrainedConfig()

        with patch.object(minimax_m3_model.MiniMaxM3MoE, "__init__", return_value=None) as native_init:
            MiniMaxM3MoE(config=config, prefix="model.layers.0.block_sparse_moe")

        self.assertTrue(issubclass(MiniMaxM3MoE, minimax_m3_model.MiniMaxM3MoE))
        self.assertEqual(native_init.call_args.kwargs["config"], config)
        self.assertEqual(native_init.call_args.kwargs["layer_id"], 0)
        self.assertEqual(native_init.call_args.kwargs["prefix"], "model.layers.0.block_sparse_moe")

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
