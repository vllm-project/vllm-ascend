from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.models.gemma4_mm import (
    AscendGemma4ForConditionalGeneration,
    _patch_gemma4_vision_patch_embedder,
)
from vllm_ascend.ops.linear import AscendReplicatedLinear
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig
from vllm_ascend.quantization.methods import (
    AscendW4A4MXFP4DynamicLinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)


def test_gemma4_mm_lora_flag_follows_lora_config():
    language_model = SimpleNamespace(
        make_empty_intermediate_tensors=None,
        moe_layers=None,
        num_moe_layers=0,
        num_logical_experts=0,
        num_physical_experts=0,
        num_local_physical_experts=0,
        num_routed_experts=0,
        num_expert_groups=0,
        num_shared_experts=0,
        num_redundant_experts=0,
    )
    vision_tower = SimpleNamespace(
        patch_embedder=SimpleNamespace(
            input_proj=SimpleNamespace(quant_method=None),
        ),
    )

    for lora_config, expected in (
        (None, False),
        (SimpleNamespace(enable_tower_connector_lora=False), False),
        (SimpleNamespace(enable_tower_connector_lora=True), True),
    ):
        model_config = SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                vision_config=SimpleNamespace(),
                audio_config=None,
                text_config=SimpleNamespace(
                    hidden_size_per_layer_input=None,
                    use_bidirectional_attention=None,
                ),
            ),
            multimodal_config=SimpleNamespace(),
            try_get_generation_config=lambda: None,
        )
        vllm_config = SimpleNamespace(
            model_config=model_config,
            quant_config=None,
            lora_config=lora_config,
        )

        with (
            patch.object(
                AscendGemma4ForConditionalGeneration,
                "_mark_tower_model",
                return_value=nullcontext(),
            ),
            patch.object(
                AscendGemma4ForConditionalGeneration,
                "_mark_language_model",
                return_value=nullcontext(),
            ),
            patch(
                "vllm_ascend.models.gemma4_mm.AutoModel.from_config",
                return_value=vision_tower,
            ),
            patch("vllm_ascend.models.gemma4_mm.Gemma4MultimodalEmbedder"),
            patch("vllm_ascend.models.gemma4_mm.recursive_replace_linear"),
            patch(
                "vllm_ascend.models.gemma4_mm.init_vllm_registered_model",
                return_value=language_model,
            ),
        ):
            model = AscendGemma4ForConditionalGeneration(vllm_config=vllm_config)

        assert model._enable_mm_lora is expected


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
            "vllm_ascend.quantization.configs.modelslim_config.get_current_vllm_config",
            return_value=current_config,
        ),
        patch(
            "vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4.get_current_vllm_config",
            return_value=current_config,
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

    inner_quant_method = getattr(
        linear.quant_method,
        "quant_method",
        linear.quant_method,
    )

    assert isinstance(
        inner_quant_method,
        AscendW4A4MXFP4DynamicLinearMethod,
    )

    parameters = dict(linear.named_parameters())
    assert parameters["weight"].shape == (64, 48)
    assert parameters["weight_scale"].shape == (64, 3)

    loaded_scale = torch.randint(0, 255, (64, 3), dtype=torch.uint8)
    parameters["weight_scale"].weight_loader(
        parameters["weight_scale"],
        loaded_scale,
    )
    torch.testing.assert_close(
        parameters["weight_scale"],
        loaded_scale,
    )


def test_mxfp8_vit_linear_uses_mxfp8_quant_method():
    prefix = "vision_tower.patch_embedder.input_proj"

    quant_config = AscendModelSlimConfig(
        {
            f"model.{prefix}.weight": "W8A8_MXFP8",
        }
    )
    quant_config.apply_vllm_mapper(AscendGemma4ForConditionalGeneration.hf_to_vllm_mapper)

    current_config = SimpleNamespace(
        quant_config=quant_config,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="gemma4"),
        ),
    )

    with (
        patch(
            "vllm_ascend.quantization.configs.modelslim_config.get_current_vllm_config",
            return_value=current_config,
        ),
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
            return_value=current_config,
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

    inner_quant_method = getattr(
        linear.quant_method,
        "quant_method",
        linear.quant_method,
    )

    assert isinstance(
        inner_quant_method,
        AscendW8A8MXFP8DynamicLinearMethod,
    )


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
    _patch_gemma4_vision_patch_embedder(
        patch_embedder,
        torch.bfloat16,
    )

    pixel_values = torch.ones((2, 3), dtype=torch.float32)
    pixel_position_ids = torch.zeros((2, 3), dtype=torch.bfloat16)
    padding_positions = torch.zeros((2, 3), dtype=torch.int64)

    output = patch_embedder(
        pixel_values,
        pixel_position_ids,
        padding_positions,
    )

    assert patch_embedder._ascend_activation_dtype is torch.bfloat16
    assert patch_embedder.input_proj.input_dtype is torch.bfloat16
    assert output.dtype is torch.bfloat16
