# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import torch
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    pos_embed_interpolate_native,
)

from vllm_ascend._310p.ops.qwen3vl_310 import fast_pos_embed_interpolate_310
from vllm_ascend.patch.worker.patch_idex_310 import Qwen3_VisionTransformer as PatchedVisionTransformer


def test_310p_fast_pos_embed_uses_native_interpolate(monkeypatch):
    captured = {}

    def fake_native(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
        captured["args"] = (t, h, w, num_grid_per_side, m_size, dtype)
        return torch.ones(t * h * w, embed_weight.shape[1], dtype=dtype)

    monkeypatch.setattr(
        "vllm_ascend._310p.ops.qwen3vl_310.pos_embed_interpolate_native",
        fake_native,
    )
    vision = SimpleNamespace(
        pos_embed=SimpleNamespace(weight=torch.zeros(9, 4)),
        num_grid_per_side=3,
        spatial_merge_size=2,
        dtype=torch.float16,
    )

    out = fast_pos_embed_interpolate_310(vision, [[1, 4, 4], [1, 2, 2]])

    assert out.shape == (20, 4)
    assert captured["args"] == (1, 2, 2, 3, 2, torch.float16)


def test_310p_patches_qwen3_vision_pos_embed_interpolate():
    assert PatchedVisionTransformer is Qwen3_VisionTransformer
    assert Qwen3_VisionTransformer.fast_pos_embed_interpolate is fast_pos_embed_interpolate_310
    assert Qwen3_VisionTransformer.fast_pos_embed_interpolate is not pos_embed_interpolate_native
