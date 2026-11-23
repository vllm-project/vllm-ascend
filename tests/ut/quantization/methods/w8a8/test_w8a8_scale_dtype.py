# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""A8W8 per-token quant requires fp32 dequant scales.

Qwen3.5/3.6 W8A8 checkpoints ship fp16 scales. Passing them to
``npu_quant_matmul`` / ``npu_grouped_matmul`` fails with aclnn 161002
("scale[0] dtype FLOAT16 and y dtype FLOAT16 are not supported when A8W8
per-token quant"). The loader therefore keeps fp32 copies and every consumer
picks them up, including the grouped-matmul w2 path.
"""

from types import SimpleNamespace

import torch

from vllm_ascend.eplb.adaptor.vllm_adaptor import EPLB_EXPERT_WEIGHT_NAMES
from vllm_ascend.quantization.methods.registry import get_scheme_class
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    AscendW8A8DynamicLinearMethod,
    _dequant_scale,
)
from vllm_ascend.quantization.quant_type import QuantType


class _LayerWithFp32:
    def __init__(self):
        self.weight_scale = torch.ones(4, 1, dtype=torch.float16)
        self.weight_scale_fp32 = self.weight_scale.to(torch.float32)


class _LayerWithoutFp32:
    def __init__(self):
        self.weight_scale = torch.ones(4, 1, dtype=torch.float16)


def test_prefers_fp32_scale_when_available():
    scale = _dequant_scale(_LayerWithFp32())
    assert scale.dtype == torch.float32


def test_falls_back_to_weight_scale_when_fp32_copy_is_absent():
    scale = _dequant_scale(_LayerWithoutFp32())
    assert scale.dtype == torch.float16


def test_linear_scheme_is_registered_to_the_class():
    """A helper inserted above the class must not steal its registration."""
    assert get_scheme_class("W8A8_DYNAMIC", "linear") is AscendW8A8DynamicLinearMethod


def _moe_layer(stacked: bool) -> SimpleNamespace:
    w2_scale_fp32 = [torch.ones(2, 1, dtype=torch.float32) for _ in range(3)]
    if stacked:
        return SimpleNamespace(
            w13_weight=torch.zeros(1),
            w13_weight_scale_fp32=torch.ones(1),
            w2_weight=torch.zeros(1),
            w2_weight_scale_fp32=torch.ones(1),
        )
    return SimpleNamespace(
        w13_weight_list=[torch.zeros(1) for _ in range(3)],
        w13_weight_scale_fp32_list=[torch.ones(1) for _ in range(3)],
        w2_weight_list=[torch.zeros(1) for _ in range(3)],
        w2_weight_scale_fp32_list=w2_scale_fp32,
    )


def test_mlp_weights_use_the_fp32_w2_scale_when_stacked():
    method = SimpleNamespace(use_expert_weight_list=False)
    _, _, _, w2_scale = AscendW8A8DynamicFusedMoEMethod._get_mlp_weights(method, _moe_layer(stacked=True))
    assert w2_scale[0] is not None
    assert w2_scale[0].dtype == torch.float32


def test_mlp_weights_use_the_fp32_w2_scale_list_for_eplb():
    method = SimpleNamespace(use_expert_weight_list=True)
    _, _, _, w2_scale = AscendW8A8DynamicFusedMoEMethod._get_mlp_weights(method, _moe_layer(stacked=False))
    assert len(w2_scale) == 3
    assert all(scale.dtype == torch.float32 for scale in w2_scale)


def test_eplb_weight_views_include_the_fp32_w2_scales():
    """EPLB reads and writes the same fp32 scale tensors, so they must exist."""
    layer = _moe_layer(stacked=False)
    views = AscendW8A8DynamicFusedMoEMethod.get_eplb_weight_views(layer)
    assert any(view is layer.w2_weight_scale_fp32_list for view in views)


def test_eplb_name_map_matches_the_fp32_scale_attributes():
    """EPLB reads attributes by name; keep the map in sync with the loader."""
    for fused in (False, True):
        names = EPLB_EXPERT_WEIGHT_NAMES[(QuantType.W8A8, fused)]
        assert "w2_weight_scale_fp32_list" in names
        assert "w2_weight_scale_list" not in names


def _loader_layer(num_experts: int = 2, hidden: int = 4, intermediate: int = 3) -> SimpleNamespace:
    """Build the tensor layout the loader hands over, before any view happens.

    ``get_dynamic_quant_param`` allocates the per-channel scales as [E, N, 1], so
    the fp32 copy must be taken *after* the [E, N] view; otherwise grouped matmul
    receives a 3-D scale that cannot broadcast over the per-channel axis.
    """
    scale = torch.float16
    return SimpleNamespace(
        w13_weight=torch.zeros(num_experts, 2 * intermediate, hidden, dtype=torch.int8),
        w2_weight=torch.zeros(num_experts, hidden, intermediate, dtype=torch.int8),
        w13_weight_scale=torch.ones(num_experts, 2 * intermediate, 1, dtype=scale),
        w2_weight_scale=torch.ones(num_experts, hidden, 1, dtype=scale),
        w13_weight_offset=torch.zeros(num_experts, 2 * intermediate, 1, dtype=scale),
        w2_weight_offset=torch.zeros(num_experts, hidden, 1, dtype=scale),
    )


def _run_loader(monkeypatch, layer: SimpleNamespace, use_expert_weight_list: bool):
    import vllm_ascend.quantization.methods.w8a8.w8a8_dynamic as w8a8_dynamic

    method = SimpleNamespace(quant_type=QuantType.W8A8, use_expert_weight_list=use_expert_weight_list)
    monkeypatch.setattr(w8a8_dynamic, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=0))
    monkeypatch.setattr(w8a8_dynamic, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(w8a8_dynamic, "use_cann_megamoe", lambda *_: False)
    monkeypatch.setattr(w8a8_dynamic.torch_npu, "npu_format_cast", lambda tensor, _fmt: tensor)
    AscendW8A8DynamicFusedMoEMethod.process_weights_after_loading(method, layer)
    return method


def test_fp32_w2_scale_keeps_the_per_channel_layout_when_stacked(monkeypatch):
    num_experts, hidden, intermediate = 2, 4, 3
    layer = _loader_layer(num_experts, hidden, intermediate)
    method = _run_loader(monkeypatch, layer, use_expert_weight_list=False)

    assert layer.w13_weight_scale.data.shape == (num_experts, 2 * intermediate)
    assert layer.w2_weight_scale.data.shape == (num_experts, hidden)
    assert layer.w2_weight_scale_fp32.shape == (num_experts, hidden)
    assert layer.w2_weight_scale_fp32.dtype == torch.float32

    _, _, _, w2_scale = AscendW8A8DynamicFusedMoEMethod._get_mlp_weights(method, layer)
    assert w2_scale[0].shape == (num_experts, hidden)


def test_fp32_w2_scale_keeps_the_per_channel_layout_for_eplb(monkeypatch):
    num_experts, hidden, intermediate = 2, 4, 3
    layer = _loader_layer(num_experts, hidden, intermediate)
    method = _run_loader(monkeypatch, layer, use_expert_weight_list=True)

    assert len(layer.w2_weight_scale_fp32_list) == num_experts
    assert all(scale.shape == (hidden,) for scale in layer.w2_weight_scale_fp32_list)
    assert all(scale.dtype == torch.float32 for scale in layer.w2_weight_scale_fp32_list)
    assert all(scale.shape == (2 * intermediate,) for scale in layer.w13_weight_scale_fp32_list)

    _, _, _, w2_scale = AscendW8A8DynamicFusedMoEMethod._get_mlp_weights(method, layer)
    assert all(scale.shape == (hidden,) for scale in w2_scale)


def test_gmm2_output_dtype_follows_the_model_dtype(monkeypatch):
    """The fp32 dequant scale must not leak into the grouped-matmul output dtype.

    aclnn rejects a float32 output for A8W8 per-token quant ("y[0] ... has
    incorrect dtype FLOAT"), so both grouped-matmul calls take the output dtype
    from the model dtype instead of ``scale[0].dtype``.
    """
    import vllm_ascend.quantization.methods.w8a8.w8a8_dynamic as w8a8_dynamic

    captured = {}

    def fake_grouped_matmul(**kwargs):
        captured.update(kwargs)
        return [torch.zeros(1)]

    monkeypatch.setattr(w8a8_dynamic.torch_npu, "npu_grouped_matmul", fake_grouped_matmul)

    method = SimpleNamespace(
        in_dtype=torch.float16,
        _get_mlp_weights=lambda layer: (
            [torch.zeros(1)],
            [torch.ones(1)],
            [torch.zeros(1)],
            [torch.ones(1, dtype=torch.float32)],
        ),
    )
    mlp = SimpleNamespace(
        layer=SimpleNamespace(),
        hidden_states=torch.zeros(1, dtype=torch.int8),
        group_list=None,
        group_list_type=0,
    )
    AscendW8A8DynamicFusedMoEMethod.apply_gmm2(method, mlp, torch.zeros(1, dtype=torch.int8), torch.ones(1))

    assert captured["output_dtype"] == torch.float16
    assert captured["scale"][0].dtype == torch.float32
