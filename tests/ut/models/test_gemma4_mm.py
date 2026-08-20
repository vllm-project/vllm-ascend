from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.models.gemma4_mm import (
    AscendGemma4ForConditionalGeneration,
    _get_tower_quant_config,
    _patch_gemma4_vision_patch_embedder,
)
from vllm_ascend.ops.linear import AscendReplicatedLinear
from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig


def _vllm_config(quant_config, hidden_size=1152, intermediate_size=4304):
    vision_config = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    return SimpleNamespace(
        quant_config=quant_config,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(vision_config=vision_config),
        ),
    )


def _modelslim_quant_config(quant_scheme, prefix="model.vision_tower.encoder.layers.0.mlp.down_proj.linear"):
    return AscendModelSlimConfig(
        {
            f"{prefix}.weight": quant_scheme,
            "group_size": 32,
        }
    )


def test_mxfp4_quantization_is_enabled_for_unaligned_gemma4_vit():
    quant_config = _modelslim_quant_config("W4A4_MXFP4")

    assert _get_tower_quant_config(_vllm_config(quant_config), allow_unaligned_mxfp4=True) is quant_config
    assert _get_tower_quant_config(_vllm_config(quant_config)) is None


def test_non_mxfp4_ascend_quantization_keeps_dimension_guard():
    quant_config = _modelslim_quant_config("W8A8")

    assert _get_tower_quant_config(_vllm_config(quant_config), allow_unaligned_mxfp4=True) is None


def test_language_only_mxfp4_does_not_enable_vision_quantization():
    quant_config = _modelslim_quant_config(
        "W4A4_MXFP4",
        prefix="model.language_model.layers.0.mlp.down_proj",
    )

    assert _get_tower_quant_config(_vllm_config(quant_config), allow_unaligned_mxfp4=True) is None


def test_unquantized_gemma4_vit_remains_unquantized():
    assert _get_tower_quant_config(_vllm_config(None)) is None


def test_other_quantization_keeps_upstream_dimension_guard():
    quant_config = Mock()
    quant_config.get_name.return_value = "some-64-aligned-backend"

    assert _get_tower_quant_config(_vllm_config(quant_config)) is None
    assert _get_tower_quant_config(_vllm_config(quant_config, intermediate_size=4352)) is quant_config


def test_patched_vision_embedder_uses_activation_dtype_and_preserves_output_dtype():
    class FakePatchEmbedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = torch.nn.Identity()

        def _position_embeddings(self, pixel_position_ids, padding_positions):
            return torch.ones((*pixel_position_ids.shape[:-1], 2), dtype=torch.float32)

    patch_embedder = FakePatchEmbedder()
    _patch_gemma4_vision_patch_embedder(patch_embedder, torch.bfloat16)

    pixel_values = torch.tensor([[[0.5, 1.0]]], dtype=torch.float32)
    pixel_position_ids = torch.zeros((1, 1, 2), dtype=torch.long)
    padding_positions = torch.zeros((1, 1), dtype=torch.bool)
    output = patch_embedder(pixel_values, pixel_position_ids, padding_positions)

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16))


def test_mxfp4_vit_linear_registers_and_loads_weight_scale():
    prefix = "vision_tower.encoder.layers.0.mlp.down_proj.linear"
    quant_config = _modelslim_quant_config("W4A4_MXFP4", prefix=f"model.{prefix}")
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
