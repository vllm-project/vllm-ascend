from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.models.gemma4_mm import (
    AscendGemma4ForConditionalGeneration,
    _get_tower_quant_config,
    _is_ascend_mxfp4_quant,
    _patch_gemma4_vision_patch_embedder,
)
from vllm_ascend.ops.linear import AscendReplicatedLinear
from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig


def _vllm_config(quant_config, hidden_size=1152, intermediate_size=4304):
    tower_config = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    return SimpleNamespace(
        quant_config=quant_config,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(vision_config=tower_config),
        ),
    ), tower_config


def test_ascend_quantization_is_enabled_for_unaligned_gemma4_vit():
    quant_config = Mock()
    quant_config.get_name.return_value = "ascend"
    quant_config.quant_description = {"model.vision_tower.layer.weight": "W4A4_MXFP4"}
    vllm_config, tower_config = _vllm_config(quant_config)

    assert _get_tower_quant_config(vllm_config, tower_config, enable_ascend=True) is quant_config
    assert _get_tower_quant_config(vllm_config, tower_config) is None


def test_other_ascend_quantization_keeps_alignment_guard():
    quant_config = Mock()
    quant_config.get_name.return_value = "ascend"
    quant_config.quant_description = {"model.vision_tower.layer.weight": "W8A8_DYNAMIC"}
    unaligned_config, unaligned_tower_config = _vllm_config(quant_config)
    aligned_config, aligned_tower_config = _vllm_config(quant_config, intermediate_size=4352)

    assert _get_tower_quant_config(unaligned_config, unaligned_tower_config, enable_ascend=True) is None
    assert _get_tower_quant_config(aligned_config, aligned_tower_config, enable_ascend=True) is quant_config


def test_unquantized_gemma4_vit_remains_unquantized():
    vllm_config, tower_config = _vllm_config(None)
    assert _get_tower_quant_config(vllm_config, tower_config) is None


def test_other_quantization_keeps_upstream_dimension_guard():
    quant_config = Mock()
    quant_config.get_name.return_value = "some-64-aligned-backend"
    unaligned_config, unaligned_tower_config = _vllm_config(quant_config)
    aligned_config, aligned_tower_config = _vllm_config(quant_config, intermediate_size=4352)

    assert _get_tower_quant_config(unaligned_config, unaligned_tower_config) is None
    assert _get_tower_quant_config(aligned_config, aligned_tower_config) is quant_config


def test_tower_quantization_uses_passed_tower_config():
    quant_config = Mock()
    quant_config.get_name.return_value = "some-64-aligned-backend"
    vllm_config, _ = _vllm_config(quant_config)
    audio_config = SimpleNamespace(hidden_size=1024, intermediate_size=4096)

    assert _get_tower_quant_config(vllm_config, audio_config) is quant_config


def test_patch_embedder_guard_only_matches_ascend_mxfp4():
    quant_config = Mock()
    quant_config.get_name.return_value = "ascend"
    quant_config.quant_description = {"model.vision_tower.layer.weight": "W8A8_DYNAMIC"}
    assert not _is_ascend_mxfp4_quant(quant_config)

    quant_config.quant_description = {"model.vision_tower.layer.weight": "W4A4_MXFP4"}
    assert _is_ascend_mxfp4_quant(quant_config)


def test_mxfp4_vit_linear_registers_and_loads_weight_scale():
    prefix = "vision_tower.encoder.layers.0.mlp.down_proj.linear"
    quant_config = AscendModelSlimConfig(
        {
            f"model.{prefix}.weight": "W4A4_MXFP4",
            "group_size": 32,
        }
    )
    quant_config.apply_vllm_mapper(AscendGemma4ForConditionalGeneration.hf_to_vllm_mapper)
    assert f"{prefix}.weight" in quant_config.quant_description
    current_config = SimpleNamespace(
        quant_config=quant_config,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="gemma4"),
        ),
    )

    with (
        patch(
            "vllm_ascend.quantization.modelslim_config.get_current_vllm_config",
            return_value=current_config,
        ),
        patch(
            "vllm_ascend.quantization.methods.w4a4_mxfp4.get_current_vllm_config",
            return_value=current_config,
        ),
        patch(
            "vllm_ascend.quantization.methods.w4a4_mxfp4.ensure_mxfp4_linear_available",
        ),
    ):
        linear = AscendReplicatedLinear(
            input_size=96,
            output_size=64,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=True,
        )

    parameters = dict(linear.named_parameters())
    assert parameters["weight"].shape == (64, 48)
    assert parameters["weight_scale"].shape == (64, 3)

    loaded_scale = torch.randint(0, 255, (64, 3), dtype=torch.uint8)
    parameters["weight_scale"].weight_loader(parameters["weight_scale"], loaded_scale)
    torch.testing.assert_close(parameters["weight_scale"], loaded_scale)


def test_gemma4_vit_patch_embedder_uses_activation_dtype_not_quant_weight_dtype():
    class FakeInputProj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.empty((1, 1), dtype=torch.float8_e4m3fn)
            self.input_dtype = None

        def forward(self, x):
            self.input_dtype = x.dtype
            return x

    class FakePatchEmbedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = FakeInputProj()

        def _position_embeddings(self, pixel_position_ids, padding_positions):
            return torch.zeros_like(pixel_position_ids, dtype=torch.float32)

    patch_embedder = FakePatchEmbedder()
    _patch_gemma4_vision_patch_embedder(patch_embedder, torch.bfloat16)

    pixel_values = torch.ones((2, 3), dtype=torch.float32)
    pixel_position_ids = torch.zeros((2, 3), dtype=torch.bfloat16)
    padding_positions = torch.zeros((2, 3), dtype=torch.int64)

    output = patch_embedder(pixel_values, pixel_position_ids, padding_positions)

    assert patch_embedder._ascend_activation_dtype is torch.bfloat16
    assert patch_embedder.input_proj.input_dtype is torch.bfloat16
    assert output.dtype is torch.bfloat16
