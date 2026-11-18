# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp registration, patch-size-16 preprocessing and normalization."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from vllm.model_executor.layers.fusion.mm_input_norm import IdentityInputNorm
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

from vllm_ascend.models import qwen3_5_vl
from vllm_ascend.models.qwen3_5_vl import AscendQwen2VLImageProcessor, AscendQwen3_5VLProcessor
from vllm_ascend.models.qwen4_exp import model as qwen4

PATCH_SIZE = 16
PATCH_DIM = 3 * 2 * PATCH_SIZE**2


def test_qwen4_registers_raw_processor_without_losing_config_type():
    model_cls = qwen4.AscendQwen4ExpForConditionalGeneration
    assert "_processor_factory" in model_cls.__dict__
    assert issubclass(qwen4.AscendQwen4ExpProcessingInfo, qwen4.upstream_model.Qwen4ExpProcessingInfo)
    factory = model_cls._processor_factory
    assert factory.info is qwen4.AscendQwen4ExpProcessingInfo
    assert factory.processor is AscendQwen3_5VLProcessor
    ctx = Mock()
    info = qwen4.AscendQwen4ExpProcessingInfo(ctx)
    info.get_hf_config()
    ctx.get_hf_config.assert_called_once_with(qwen4.upstream_model.Qwen4ExpConfig)


class ReachedPatchEmbed(Exception):
    pass


class VisionProbe:
    device = torch.device("cpu")
    dtype = torch.bfloat16

    def __init__(self, norm):
        self.input_norm = norm
        self.seen = None

    def patch_embed(self, pixels):
        self.seen = pixels
        raise ReachedPatchEmbed

    def __call__(self, *args, **kwargs):
        return Qwen3_VisionTransformer.forward(self, *args, **kwargs)


def make_model(monkeypatch, enabled):
    norm = Mock(side_effect=lambda pixels, dtype: ((pixels.float() / 255 - 0.5) / 0.25).to(dtype))

    def build_norm(config):
        return norm if enabled else IdentityInputNorm()

    def init_upstream(self, *, vllm_config, prefix):
        torch.nn.Module.__init__(self)
        self.model_config = SimpleNamespace(multimodal_config=SimpleNamespace(mm_device_do_normalize=enabled))
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(
                in_channels=3, patch_size=PATCH_SIZE, temporal_patch_size=2, spatial_merge_size=2
            )
        )
        self.visual = VisionProbe(build_norm(self.model_config))
        self.use_data_parallel = False

    monkeypatch.setattr(qwen4.upstream_model.Qwen4ExpForConditionalGeneration, "__init__", init_upstream)
    monkeypatch.setattr(qwen3_5_vl, "build_mm_input_norm", build_norm)
    return qwen4.AscendQwen4ExpForConditionalGeneration(vllm_config=None), norm


def make_processor():
    return AscendQwen2VLImageProcessor(
        patch_size=PATCH_SIZE,
        temporal_patch_size=2,
        merge_size=2,
        size={"shortest_edge": 32 * 32, "longest_edge": 4096 * 4096},
        image_mean=[0.5] * 3,
        image_std=[0.25] * 3,
    )


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("graph_selection", [True, False])
def test_qwen4_processor_to_vision_normalizes_once(monkeypatch, enabled, graph_selection):
    model, norm = make_model(monkeypatch, enabled)
    processor = make_processor()
    kwargs = {"do_rescale": False, "do_normalize": False} if enabled else {}
    images = [Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))]
    inputs = dict(processor(images=images, return_tensors="pt", **kwargs))
    assert ("image_hw" in inputs) == enabled
    torch.testing.assert_close(inputs["image_grid_thw"], torch.tensor([[1, 4, 4]]))
    with pytest.raises(ReachedPatchEmbed):
        if graph_selection:
            selected = model.select_encoder_cudagraph_items(inputs, [0])
            selected = model.select_encoder_cudagraph_items(selected, [0])
            model.visual(selected["pixel_values"], grid_thw=selected["image_grid_thw"])
        else:
            parsed = model._parse_and_validate_image_input(**inputs)
            model._process_image_input(parsed)
    torch.testing.assert_close(model.visual.seen, torch.full((16, PATCH_DIM), 2.0, dtype=torch.bfloat16))
    assert norm.call_count == int(enabled)


def test_patch_size_16_layout_matches_stock_processor(monkeypatch):
    model, _ = make_model(monkeypatch, True)
    # No resize: exact patch order and temporal replication must match stock.
    pixels = np.random.default_rng(0).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    processor = make_processor()
    raw = dict(processor(images=[image], return_tensors="pt", do_rescale=False, do_normalize=False))
    stock = processor(images=[image], return_tensors="pt", do_resize=False, do_rescale=False, do_normalize=False)
    actual = model._resize_and_patchify(raw, raw["image_grid_thw"])
    torch.testing.assert_close(actual, stock["pixel_values"].float(), rtol=0, atol=0)
    assert actual.shape == (24, PATCH_DIM)


def test_qwen4_text_only_and_image_embeddings(monkeypatch):
    model, norm = make_model(monkeypatch, True)
    assert model._parse_and_validate_image_input() is None
    embeds = torch.randn(6, 2560)
    outputs = model._process_image_input(
        {"type": "image_embeds", "image_embeds": embeds, "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 4]])}
    )
    assert [part.shape for part in outputs] == [torch.Size([4, 2560]), torch.Size([2, 2560])]
    torch.testing.assert_close(torch.cat(outputs), embeds.to(torch.bfloat16))
    norm.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_qwen4_data_parallel_receives_normalized_patches(monkeypatch, enabled):
    model, norm = make_model(monkeypatch, enabled)
    model.use_data_parallel = True
    processor = make_processor()
    kwargs = {"do_rescale": False, "do_normalize": False} if enabled else {}
    images = [Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))]
    inputs = dict(processor(images=images, return_tensors="pt", **kwargs))
    expected = (torch.ones(4, 2560),)

    def run_dp(visual, pixels, grid, rope_type):
        assert visual is model.visual
        assert grid == [[1, 4, 4]]
        assert rope_type == "rope_3d"
        torch.testing.assert_close(pixels, torch.full((16, PATCH_DIM), 2.0, dtype=torch.bfloat16))
        # The DP encoder can enter the ViT directly: no second normalization.
        torch.testing.assert_close(visual.input_norm(pixels, visual.dtype), pixels)
        return expected

    monkeypatch.setattr(qwen3_5_vl, "run_dp_sharded_mrope_vision_model", run_dp)
    actual = model._process_image_input(model._parse_and_validate_image_input(**inputs))
    assert actual is expected
    assert norm.call_count == int(enabled)


def test_qwen4_raw_processor_keeps_grid_and_payload_for_multiple_images():
    processor = make_processor()
    images = [Image.fromarray(np.full((h, w, 3), 127, dtype=np.uint8)) for h, w in [(45, 67), (81, 99)]]
    raw = processor(images=images, return_tensors="pt", do_rescale=False, do_normalize=False)
    stock = processor(images=images, return_tensors="pt")
    torch.testing.assert_close(raw["image_grid_thw"], stock["image_grid_thw"])
    assert raw["pixel_values"].dtype == torch.uint8
    assert raw["pixel_values"].numel() == sum(h * w * 3 for h, w in [(45, 67), (81, 99)])
    assert raw["image_hw"].tolist() == [[45, 67], [81, 99]]
