# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Checkpoint payload coverage without downloading checkpoint weights."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import PretrainedConfig

from tests.e2e.common.rlhf.weight_transfer import weight_source as utils
from tests.e2e.common.rlhf.weight_transfer.case import MINIMAX_M3_CASE, MODEL_CASES
from tests.e2e.common.rlhf.weight_transfer.checkpoint_adapters import minimax_m3_checkpoint_name


def _checkpoint_source(model, case):
    # Only supply num_experts when the case does not override an expert field.
    # Otherwise an incorrect fallback could pass by reading this default.
    overrides = (
        case.hf_overrides.get(case.meta_config_attribute, {}) if case.meta_config_attribute else case.hf_overrides
    )
    meta_config = PretrainedConfig()
    if not any(name in overrides for name in ("n_routed_experts", "num_local_experts", "num_experts")):
        meta_config.num_experts = 8
    config = (
        PretrainedConfig(**{case.meta_config_attribute: meta_config}) if case.meta_config_attribute else meta_config
    )
    with (
        patch.object(utils.AutoConfig, "from_pretrained", return_value=config) as config_loader,
        patch.object(utils.AutoModelForCausalLM, "from_config", return_value=model) as model_loader,
    ):
        source = utils.FixedRandomWeightSource(case, torch.device("cpu"))
    config_loader.assert_called_once_with(case.model, trust_remote_code=case.trust_remote_code)
    assert model_loader.call_args.kwargs["trust_remote_code"] == case.trust_remote_code
    assert source._num_experts == 8
    return source


def _router_meta_model(layer_index):
    model = torch.nn.Module()
    parent = model
    for component in ("model", "layers", str(layer_index), "mlp"):
        child = torch.nn.Module()
        parent.add_module(component, child)
        parent = child
    gate = torch.nn.Linear(16, 8, bias=False, device="meta")
    parent.add_module("gate", gate)
    return model, gate


@pytest.mark.parametrize(
    ("model_name", "layer_index", "checkpoint_gate", "buffers", "legacy_buffers", "added_buffers"),
    [
        pytest.param(
            "Qwen/Qwen3.5-35B-A3B",
            3,
            "model.language_model.layers.3.mlp.gate",
            (),
            {},
            {},
            id="qwen3.5-unchanged",
        ),
        pytest.param(
            "kylesayrs/DeepSeek-V4-Flash-bf16",
            0,
            "model.layers.0.mlp.gate",
            (("tid2eid", (16, 2), torch.int64, False),),
            {"model.layers.0.mlp.gate.tid2eid": ((16, 2), torch.int64)},
            {},
            id="deepseek-v4-integer-table",
        ),
        pytest.param(
            "zai-org/GLM-5.1",
            3,
            "model.layers.3.mlp.gate",
            (("e_score_correction_bias", (8,), torch.float32, True),),
            {},
            {"model.layers.3.mlp.gate.e_score_correction_bias": ((8,), torch.float32)},
            id="glm-5.1-fp32-router-bias",
        ),
        pytest.param(
            MINIMAX_M3_CASE.model,
            3,
            "language_model.model.layers.3.block_sparse_moe.gate",
            (("e_score_correction_bias", (8,), torch.float32, True),),
            {},
            {"language_model.model.layers.3.block_sparse_moe.e_score_correction_bias": ((8,), torch.float32)},
            id="minimax-m3-fp32-router-bias",
        ),
    ],
)
def test_checkpoint_source_buffer_metadata(
    model_name, layer_index, checkpoint_gate, buffers, legacy_buffers, added_buffers
):
    """Pin the metadata delta using representative HF state, without downloads.

    Real buffer registration exercises state_dict persistence. The DeepSeek
    table is deliberately non-persistent to preserve the old integer-buffer
    contract regardless of how a Transformers version registers it.
    """
    case = next(case for case in MODEL_CASES if case.model == model_name)
    model, gate = _router_meta_model(layer_index)
    for name, shape, dtype, persistent in buffers:
        gate.register_buffer(name, torch.empty(shape, dtype=dtype, device="meta"), persistent=persistent)
    # Runtime-only floating/complex buffers must never enter the payload.
    model.register_buffer("inv_freq", torch.empty(8, device="meta"), persistent=False)
    model.register_buffer("complex_runtime", torch.empty(8, dtype=torch.complex64, device="meta"), persistent=False)

    unchanged_baseline_metadata = {f"{checkpoint_gate}.weight": ((8, 16), torch.bfloat16), **legacy_buffers}
    source = _checkpoint_source(model, case)
    actual = {meta.name: (meta.shape, meta.dtype) for meta in source.metadata}
    assert actual == unchanged_baseline_metadata | added_buffers
    with patch.object(torch.npu, "manual_seed"):
        weights = dict(source)
    assert {name: (tuple(tensor.shape), tensor.dtype) for name, tensor in weights.items()} == actual
    # Exercise expert-count resolution even for models without integer buffers.
    with patch.object(torch.npu, "manual_seed"):
        table = source._make_tensor("synthetic.tid2eid", (16, 2), torch.int64)
    assert torch.all((table >= 0) & (table < source._num_experts))
    assert torch.all(table[:, 0] != table[:, 1])


def test_minimax_checkpoint_source_includes_parameter_router_bias():
    """Keep the name mapping valid if HF represents the bias as a parameter."""
    model, gate = _router_meta_model(3)
    gate.register_parameter(
        "e_score_correction_bias", torch.nn.Parameter(torch.empty(8, dtype=torch.float32, device="meta"))
    )
    source = _checkpoint_source(model, MINIMAX_M3_CASE)
    prefix = "language_model.model.layers.3.block_sparse_moe"
    expected = {
        f"{prefix}.gate.weight": ((8, 16), torch.bfloat16),
        f"{prefix}.e_score_correction_bias": ((8,), torch.bfloat16),
    }
    assert {meta.name: (meta.shape, meta.dtype) for meta in source.metadata} == expected
    with patch.object(torch.npu, "manual_seed"):
        assert {name: (tuple(tensor.shape), tensor.dtype) for name, tensor in source} == expected


def test_minimax_checkpoint_source_splits_fused_projections():
    model = MagicMock()
    model.named_parameters.return_value = [
        ("model.layers.0.mlp.gate_up_proj.weight", torch.empty((12, 16), device="meta")),
        ("model.layers.3.mlp.shared_experts.gate_up_proj.weight", torch.empty((12, 16), device="meta")),
        ("model.layers.3.mlp.experts.gate_up_proj", torch.empty((2, 12, 16), device="meta")),
    ]
    model.named_buffers.return_value = []
    model.state_dict.return_value = {}
    source = _checkpoint_source(model, MINIMAX_M3_CASE)
    actual = {meta.name: (meta.shape, meta.dtype) for meta in source.metadata}
    expected_names = {
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.3.block_sparse_moe.shared_experts.gate_proj.weight",
        "language_model.model.layers.3.block_sparse_moe.shared_experts.up_proj.weight",
        *{
            f"language_model.model.layers.3.block_sparse_moe.experts.{expert}.{shard}.weight"
            for expert in range(2)
            for shard in ("w1", "w3")
        },
    }
    assert actual == {name: ((6, 16), torch.bfloat16) for name in expected_names}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("lm_head.weight", "language_model.lm_head.weight"),
        (
            "language_model.model.layers.3.self_attn.indexer.q_proj.weight",
            "language_model.model.layers.3.self_attn.index_q_proj.weight",
        ),
        ("language_model.model.layers.0.mlp.gate_proj.weight", "language_model.model.layers.0.mlp.gate_proj.weight"),
        (
            "language_model.model.layers.3.mlp.shared_experts.down_proj.weight",
            "language_model.model.layers.3.block_sparse_moe.shared_experts.down_proj.weight",
        ),
        (
            "language_model.model.layers.3.mlp.experts.7.down_proj.weight",
            "language_model.model.layers.3.block_sparse_moe.experts.7.w2.weight",
        ),
    ],
)
def test_minimax_checkpoint_name(name, expected):
    assert minimax_m3_checkpoint_name(name) == expected
    assert minimax_m3_checkpoint_name(expected) == expected


def test_minimax_fused_projection_rejects_odd_dimension():
    with pytest.raises(AssertionError, match="odd fused gate/up dimension"):
        utils.split_fused_mlp_params("model.layers.0.mlp.gate_up_proj.weight", (7, 16))
