# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear

from vllm_ascend.models.kimi_k3.kimi_k3_mtp import AscendKimiK3MTP
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig


@pytest.mark.parametrize("checkpoint_prefix", ["", "language_model."])
@pytest.mark.parametrize("gate_type", ["FLOAT", "W8A8_DYNAMIC"])
def test_mtp_loads_standalone_gate_without_fused_shard_two(checkpoint_prefix, gate_type):
    """MTP name rewriting must preserve mixed gates and their TP scale slices."""
    prefix = "model.layers.1.self_attn"
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3_mtp")))
    quant_config = AscendModelSlimConfig({f"{prefix}.g_proj.weight": gate_type})
    with (
        patch("vllm_ascend.quantization.configs.modelslim_config.get_current_vllm_config", return_value=config),
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=2),
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=1),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=1),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=2),
        patch(
            "vllm_ascend.ops.linear.get_parallel_op",
            side_effect=lambda disable_tp, *args: (None, 0, 1) if disable_tp else (None, 1, 2),
        ),
    ):
        attention = nn.Module()
        attention.fused_qkv_a_proj = MergedColumnParallelLinear(
            4, [2, 2], bias=False, disable_tp=True, params_dtype=torch.bfloat16
        )
        attention.g_proj = ColumnParallelLinear(
            4, 8, bias=False, params_dtype=torch.bfloat16, quant_config=quant_config, prefix=f"{prefix}.g_proj"
        )

    model = AscendKimiK3MTP.__new__(AscendKimiK3MTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        linear_attn_config=None,
        q_lora_rank=2,
        mla_use_output_gate=True,
        is_moe=False,
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
    )
    model.num_redundant_experts = 0
    model.model = nn.Module()
    model.model.mtp_start_layer_idx = 1
    model.model.num_mtp_layers = 1
    layer = nn.Module()
    layer.mtp_block = nn.Module()
    layer.mtp_block.self_attn = attention
    model.model.layers = nn.ModuleDict({"1": layer})

    q = torch.full((2, 4), 2, dtype=torch.bfloat16)
    kv = torch.full((2, 4), 3, dtype=torch.bfloat16)
    gate = torch.arange(-16, 16, dtype=torch.float32).reshape(8, 4)
    gate = (gate / 64).to(torch.bfloat16) if gate_type == "FLOAT" else gate.to(torch.int8)
    weights = [
        (f"{checkpoint_prefix}{prefix}.q_a_proj.weight", q),
        (f"{checkpoint_prefix}{prefix}.kv_a_proj_with_mqa.weight", kv),
        (f"{checkpoint_prefix}{prefix}.g_proj.weight", gate),
        # Verifier weights are not MTP parameters and must remain skipped.
        (f"{checkpoint_prefix}model.layers.0.self_attn.g_proj.weight", torch.zeros_like(gate)),
    ]
    expected_names = {
        "model.layers.1.mtp_block.self_attn.fused_qkv_a_proj.weight",
        "model.layers.1.mtp_block.self_attn.g_proj.weight",
    }
    scale = torch.arange(1, 9, dtype=torch.bfloat16).reshape(8, 1)
    if gate_type != "FLOAT":
        for suffix, value in (("weight_scale", scale), ("weight_offset", scale + 1)):
            weights.append((f"{checkpoint_prefix}{prefix}.g_proj.{suffix}", value))
            expected_names.add(f"model.layers.1.mtp_block.self_attn.g_proj.{suffix}")

    loaded = model.load_weights(iter(weights))

    assert loaded == expected_names
    torch.testing.assert_close(attention.fused_qkv_a_proj.weight, torch.cat((q, kv)))
    torch.testing.assert_close(attention.g_proj.weight, gate[4:])
    if gate_type != "FLOAT":
        torch.testing.assert_close(attention.g_proj.weight_scale, scale[4:])
        torch.testing.assert_close(attention.g_proj.weight_offset, (scale + 1)[4:])
