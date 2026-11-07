# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_ascend.models.minimax_m3.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig


@pytest.mark.parametrize("namespace", ["legacy", "canonical", "model_prefix"])
@pytest.mark.parametrize(
    ("legacy_name", "canonical_name", "shape"),
    [
        (
            "vision_tower.embeddings.proj.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            (2, 2),
        ),
        (
            "vision_tower.pre_layrnorm.bias",
            "vision_tower.vision_model.pre_layrnorm.bias",
            (2,),
        ),
        (
            "vision_tower.layers.0.self_attn.q_proj.weight",
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
            (2, 2),
        ),
    ],
)
def test_vision_checkpoint_weights_load_into_model_parameters(legacy_name, canonical_name, shape, namespace):
    checkpoint_name = {
        "legacy": legacy_name,
        "canonical": canonical_name,
        "model_prefix": f"model.{canonical_name}",
    }[namespace]
    target_name = f"model.{canonical_name}"
    model = nn.Module()
    leaf = model
    *module_names, parameter_name = target_name.split(".")
    for module_name in module_names:
        child = nn.Module()
        leaf.add_module(module_name, child)
        leaf = child
    leaf.register_parameter(parameter_name, nn.Parameter(torch.zeros(shape)))
    weight = torch.arange(1, model.get_parameter(target_name).numel() + 1, dtype=torch.float32).reshape(shape)

    loaded = AutoWeightsLoader(model).load_weights(
        [(checkpoint_name, weight)], mapper=MiniMaxM3SparseForConditionalGeneration.hf_to_vllm_mapper
    )

    assert loaded == {target_name}
    torch.testing.assert_close(model.get_parameter(target_name), weight)


@pytest.mark.parametrize("namespace", ["legacy", "canonical", "model_prefix"])
def test_vision_quantization_description_uses_same_model_names(namespace):
    names = [
        (
            "vision_tower.embeddings.proj.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "W8A8",
        ),
        (
            "vision_tower.embeddings.proj.weight_scale",
            "vision_tower.vision_model.embeddings.patch_embedding.weight_scale",
            "W8A8",
        ),
        ("vision_tower.pre_layrnorm.weight", "vision_tower.vision_model.pre_layrnorm.weight", "FLOAT"),
        ("vision_tower.pre_layrnorm.bias", "vision_tower.vision_model.pre_layrnorm.bias", "FLOAT"),
        (
            "vision_tower.layers.0.self_attn.q_proj.weight",
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
            "W8A8",
        ),
        (
            "vision_tower.layers.0.self_attn.q_proj.weight_scale",
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight_scale",
            "W8A8",
        ),
    ]
    checkpoint_description = {
        {"legacy": legacy, "canonical": canonical, "model_prefix": f"model.{canonical}"}[namespace]: quant_type
        for legacy, canonical, quant_type in names
    }
    expected = {f"model.{canonical}": quant_type for _, canonical, quant_type in names}
    optional_metadata = {"producer": "test"}
    config = AscendModelSlimConfig({**checkpoint_description, "optional": optional_metadata})
    mapper = MiniMaxM3SparseForConditionalGeneration.hf_to_vllm_mapper

    config.apply_vllm_mapper(mapper)
    config.apply_vllm_mapper(mapper)

    assert config.quant_description == {**expected, "optional": optional_metadata}
