# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""M3 checkpoint names must still reach base parameters after LoRA wrapping."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)

from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3Model


class _QKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(6, 2, dtype=torch.bfloat16), requires_grad=False)
        self.weight.weight_loader = self.load

    @staticmethod
    def load(param, loaded_weight, shard_id):
        start, width = {"q": (0, 2), "k": (2, 1), "v": (3, 1), "index_q": (4, 1), "index_k": (5, 1)}[shard_id]
        param[start : start + width].copy_(loaded_weight)


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.routed_experts = nn.Module()
        for name, shape in (("w13_weight", (2, 4, 2)), ("w2_weight", (2, 2, 2))):
            param = nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False)
            param.weight_loader = self.load
            self.routed_experts.register_parameter(name, param)

    @staticmethod
    def load(param, loaded_weight, name, shard_id, expert_id, return_success=False):
        target = param[expert_id]
        if shard_id != "w2":
            target = target[0:2] if shard_id == "w1" else target[2:4]
        target.copy_(loaded_weight)
        return True if return_success else None


class _ReloadModel(nn.Module):
    get_expert_mapping = MiniMaxM3Model.get_expert_mapping
    load_weights = MiniMaxM3Model.load_weights

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_local_experts=2)
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = _QKV()
        layer.self_attn.o_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        layer.block_sparse_moe = nn.Module()
        layer.block_sparse_moe.experts = _Experts()
        self.layers = nn.ModuleList([layer])


def checkpoint(offset):
    weights = []
    for i, (name, width) in enumerate((("q", 2), ("k", 1), ("v", 1), ("index_q", 1), ("index_k", 1), ("o", 2))):
        weights.append(
            (f"layers.0.self_attn.{name}_proj.weight", torch.full((width, 2), offset + i, dtype=torch.bfloat16))
        )
    for expert in range(2):
        for index, name in enumerate(("w1", "w2", "w3")):
            weights.append(
                (
                    f"layers.0.block_sparse_moe.experts.{expert}.{name}.weight",
                    torch.full((2, 2), offset + 10 + expert * 3 + index, dtype=torch.bfloat16),
                )
            )
    return weights


@pytest.mark.parametrize("wrapping", ["plain", "attention-only", "all"])
@torch.no_grad()
def test_checkpoint_reload_updates_wrapped_base_and_preserves_storage(wrapping):
    """Exercise two real upstream layerwise updates, including indexer and expert triplets."""
    model = _ReloadModel()
    model.load_weights(checkpoint(1))
    record_metadata_for_reloading(model)
    original = {name: param.data_ptr() for name, param in model.named_parameters()}
    layer = model.layers[0]
    targets = [(layer.self_attn, "qkv_proj"), (layer.self_attn, "o_proj")]
    if wrapping == "all":
        targets.append((layer.block_sparse_moe, "experts"))
    if wrapping != "plain":
        for parent, name in targets:
            wrapper = BaseLayerWithLoRA()
            wrapper.base_layer = getattr(parent, name)
            setattr(parent, name, wrapper)
    for offset in (20, 40):
        initialize_layerwise_reload(model)
        loaded = model.load_weights(checkpoint(offset))
        finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.bfloat16))
        params = {name.replace(".base_layer.", "."): param for name, param in model.named_parameters()}
        assert loaded == set(original)
        assert {name: param.data_ptr() for name, param in params.items()} == original
        source = dict(checkpoint(offset))
        torch.testing.assert_close(
            params["layers.0.self_attn.qkv_proj.weight"],
            torch.cat(
                [source[f"layers.0.self_attn.{name}_proj.weight"] for name in ("q", "k", "v", "index_q", "index_k")]
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            params["layers.0.self_attn.o_proj.weight"], source["layers.0.self_attn.o_proj.weight"], rtol=0, atol=0
        )
        for expert in range(2):
            prefix = f"layers.0.block_sparse_moe.experts.{expert}"
            torch.testing.assert_close(
                params["layers.0.block_sparse_moe.experts.routed_experts.w13_weight"][expert],
                torch.cat([source[f"{prefix}.w1.weight"], source[f"{prefix}.w3.weight"]]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                params["layers.0.block_sparse_moe.experts.routed_experts.w2_weight"][expert],
                source[f"{prefix}.w2.weight"],
                rtol=0,
                atol=0,
            )
