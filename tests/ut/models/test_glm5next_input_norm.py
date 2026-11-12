# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real processor, normalizer factory and vision-tower entry on CPU."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fusion.mm_input_norm import (
    FusedMMInputNorm,
    IdentityInputNorm,
    build_mm_input_norm,
)

from vllm_ascend.models.glm5next import model as glm5next_model
from vllm_ascend.models.glm5next import multimodal as vision
from vllm_ascend.models.glm5next.processor import Glm5NextImageProcessor, Glm5NextVideoProcessor


@pytest.fixture(autouse=True)
def cpu_vllm_config():
    with patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"):
        config = VllmConfig()
    with set_current_vllm_config(config), torch.device("cpu"):
        yield


@pytest.fixture
def model_config(tmp_path):
    processor = Glm5NextImageProcessor()
    image_config = {
        "do_rescale": processor.do_rescale,
        "do_normalize": processor.do_normalize,
        "rescale_factor": processor.rescale_factor,
        "image_mean": processor.image_mean,
        "image_std": processor.image_std,
    }
    (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": image_config}))

    def make(enabled):
        return SimpleNamespace(
            model=str(tmp_path),
            revision=None,
            hf_image_processor_config=image_config,
            hf_config=SimpleNamespace(text_config=None, vision_config=SimpleNamespace(rms_norm_eps=1e-6)),
            multimodal_config=SimpleNamespace(
                mm_device_do_normalize=enabled,
                mm_encoder_tp_mode="weights",
                is_multimodal_pruning_enabled=lambda: False,
            ),
        )

    return make


@pytest.mark.parametrize("enabled", [True, False])
def test_multimodal_model_passes_factory_normalizer_to_vision(monkeypatch, model_config, enabled):
    def capture_tower(*args, **kwargs):
        tower = nn.Module()
        tower.input_norm = kwargs["input_norm"]
        return tower

    cls = glm5next_model.Glm5NextForConditionalGeneration
    monkeypatch.setattr(cls, "_mark_tower_model", lambda *args: nullcontext())
    monkeypatch.setattr(cls, "_mark_language_model", lambda *args: nullcontext())
    monkeypatch.setattr(glm5next_model, "Glm5NextVisionTransformer", capture_tower)
    language_model = nn.Module()
    language_model.make_empty_intermediate_tensors = Mock()
    monkeypatch.setattr(glm5next_model, "init_vllm_registered_model", lambda **kwargs: language_model)

    model = cls(vllm_config=SimpleNamespace(model_config=model_config(enabled)))
    expected = FusedMMInputNorm if enabled else IdentityInputNorm
    assert isinstance(model.visual.input_norm, expected)
    assert model.visual.input_norm.input_dtype == (torch.uint8 if enabled else None)


class _ReachedPatchEmbed(Exception):
    pass


class _CapturePatchEmbed(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.proj = nn.Linear(1, 1, bias=False, dtype=dtype)
        self.value = None

    def forward(self, pixels):
        self.value = pixels.detach().clone()
        raise _ReachedPatchEmbed


@pytest.fixture
def make_tower(monkeypatch):
    # Stop at patch embedding: attention/merger weights and NPU kernels are
    # unrelated to the input contract. Keep the actual tower ctor/forward.
    monkeypatch.setattr(vision, "is_vit_use_data_parallel", lambda: False)
    monkeypatch.setattr(vision, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(vision, "get_rope", lambda **kwargs: nn.Identity())
    monkeypatch.setattr(vision, "get_vit_attn_backend", lambda **kwargs: None)
    monkeypatch.setattr(vision, "Glm5NextPatchMerger", lambda **kwargs: nn.Identity())
    monkeypatch.setattr(vision, "Conv2dLayer", lambda **kwargs: nn.Identity())
    monkeypatch.setattr(vision, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    config = SimpleNamespace(
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        depth=0,
        hidden_size=4,
        num_heads=1,
        spatial_merge_size=2,
        out_hidden_size=4,
        swiglu_limit=7,
        projection_intermediate_size=4,
        rms_norm_eps=1e-6,
    )

    def make(input_norm=None, dtype=torch.bfloat16):
        monkeypatch.setattr(vision, "Glm5NextVisionPatchEmbed", lambda **kwargs: _CapturePatchEmbed(dtype))
        return vision.Glm5NextVisionTransformer(SimpleNamespace(), config, input_norm=input_norm)

    return make


def test_vision_tower_defaults_to_identity(make_tower):
    assert isinstance(make_tower().input_norm, IdentityInputNorm)


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_encoder_metadata", [False, True])
def test_raw_pixels_are_normalized_before_patch_embed(make_tower, model_config, modality, dtype, with_encoder_metadata):
    pixels = (torch.arange(4 * 3 * 28 * 28, dtype=torch.int32) % 256).to(torch.uint8).reshape(4, 3, 28, 28)
    if modality == "image":
        processor = Glm5NextImageProcessor()
        kwargs = {"images": pixels[0], "do_resize": False, "return_tensors": "pt"}
        pixel_key, grid_key = "pixel_values", "image_grid_thw"
    else:
        processor = Glm5NextVideoProcessor()
        kwargs = {"videos": [pixels], "do_resize": False, "do_sample_frames": False, "return_tensors": "pt"}
        pixel_key, grid_key = "pixel_values_videos", "video_grid_thw"
    raw = processor(**kwargs, do_rescale=False, do_normalize=False)
    normalized = processor(**kwargs)
    assert raw[pixel_key].dtype == torch.uint8
    assert torch.equal(raw[grid_key], normalized[grid_key])
    norm = build_mm_input_norm(model_config(True))
    tower = make_tower(norm, dtype)
    seen = []
    norm.register_forward_pre_hook(lambda module, args: seen.append((args[0].dtype, args[1])))

    with pytest.raises(_ReachedPatchEmbed):
        tower(raw[pixel_key], raw[grid_key], encoder_metadata={} if with_encoder_metadata else None)

    assert seen == [(torch.uint8, dtype)]
    assert tower.patch_embed.value.dtype == dtype
    torch.testing.assert_close(
        tower.patch_embed.value, normalized[pixel_key].to(dtype), rtol=0, atol=1e-6 if dtype == torch.float32 else 0
    )


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpu_normalized_pixels_are_only_cast_when_disabled(make_tower, model_config, input_dtype):
    norm = build_mm_input_norm(model_config(False))
    tower = make_tower(norm)
    pixels = torch.linspace(-2, 2, 3 * 2 * 14 * 14).reshape(1, -1).to(input_dtype)
    with pytest.raises(_ReachedPatchEmbed):
        tower(pixels, [[1, 1, 1]])
    torch.testing.assert_close(tower.patch_embed.value, pixels.to(tower.dtype), rtol=0, atol=0)
