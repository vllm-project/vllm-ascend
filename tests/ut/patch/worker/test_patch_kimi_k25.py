# SPDX-License-Identifier: Apache-2.0

import runpy

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.models.kimi_k25_vit import (
    Learnable2DInterpPosEmbDivided_fixed,
    MoonViT3dPretrainedModel,
)

from vllm_ascend import utils as ascend_utils
from vllm_ascend.patch.worker import patch_kimi_k25


@pytest.mark.parametrize(
    "grid_thws",
    [
        pytest.param([[1, 2, 2], [2, 1, 1]], id="list"),
        pytest.param(torch.tensor([[1, 2, 2], [2, 1, 1]]), id="tensor"),
    ],
)
def test_position_embedding_patch_uses_current_vllm_contract(grid_thws):
    assert (
        Learnable2DInterpPosEmbDivided_fixed.forward
        is patch_kimi_k25.AscendLearnable2DInterpPosEmbDivided_fixed.forward
    )

    pos_emb = Learnable2DInterpPosEmbDivided_fixed(
        height=2,
        width=2,
        num_frames=2,
        dim=4,
        interpolation_mode="nearest",
    )
    weight = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    time_weight = torch.tensor([[[10.0, 20.0, 30.0, 40.0]], [[50.0, 60.0, 70.0, 80.0]]])
    with torch.no_grad():
        pos_emb.weight.copy_(weight)
        pos_emb.time_weight.copy_(time_weight)

    output = pos_emb(torch.zeros(6, 4), grid_thws)

    expected = torch.cat(
        [
            weight.flatten(end_dim=1),
            weight[0, 0].unsqueeze(0) + time_weight[0],
            weight[0, 0].unsqueeze(0) + time_weight[1],
        ]
    )
    torch.testing.assert_close(output, expected)


def test_a5_moonvit_to_patch_uses_current_vllm_contract(monkeypatch):
    original_to = MoonViT3dPretrainedModel.to
    original_forward = Learnable2DInterpPosEmbDivided_fixed.forward
    monkeypatch.setattr(ascend_utils, "_ascend_device_type", ascend_utils.AscendDeviceType.A5)

    try:
        patch_namespace = runpy.run_path(patch_kimi_k25.__file__)
        assert MoonViT3dPretrainedModel.to is patch_namespace["_patched_moonvit_to"]

        vision_tower = MoonViT3dPretrainedModel.__new__(MoonViT3dPretrainedModel)
        nn.Module.__init__(vision_tower)
        vision_tower.register_parameter(
            "quantized_weight",
            nn.Parameter(torch.ones(1, dtype=torch.float32)),
        )

        assert vision_tower.to(torch.float16) is vision_tower
        assert vision_tower.to(dtype=torch.bfloat16) is vision_tower
        assert vision_tower.quantized_weight.dtype == torch.float32
    finally:
        MoonViT3dPretrainedModel.to = original_to
        Learnable2DInterpPosEmbDivided_fixed.forward = original_forward
