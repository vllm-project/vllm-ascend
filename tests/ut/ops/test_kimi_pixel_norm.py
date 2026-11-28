# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Processor-metadata regression coverage for Kimi device normalization."""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.kimi_pixel_norm import build_kimi_pixel_norm


def _model_config(path, image_config):
    return SimpleNamespace(
        model=str(path),
        revision=None,
        multimodal_config=SimpleNamespace(mm_device_do_normalize=True),
        hf_config=SimpleNamespace(model_type="kimi_k3", architectures=["KimiK3ForConditionalGeneration"]),
        hf_image_processor_config=image_config,
    )


def _vision_tower():
    projection = torch.nn.Conv2d(3, 8, 14, stride=14, bias=False, dtype=torch.bfloat16, device="cpu")
    return SimpleNamespace(patch_embed=SimpleNamespace(proj=projection))


@pytest.mark.parametrize("source", ["legacy", "processor", "nested", "cached"])
def test_build_reads_processor_metadata_without_requiring_processor_class(tmp_path, source):
    media = {
        "patch_size": 14,
        "image_mean": [0.5] * 3,
        "image_std": [0.5] * 3,
        "merge_kernel_size": 2,
        "temporal_merge_kernel_size": 4,
    }
    image_config = {"media_proc_cfg": media}
    cached_config = {}
    if source == "legacy":
        # Like the K3 dummy checkpoint, legacy metadata can advertise auto_map
        # without processor_class; get_processor_config ignores this file.
        legacy = {**image_config, "auto_map": {"AutoImageProcessor": "image_processing.KimiK25FusedVisionProcessor"}}
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(legacy))
    elif source == "cached":
        cached_config = image_config
    else:
        processor: dict = {"image_processor": image_config} if source == "nested" else image_config
        (tmp_path / "processor_config.json").write_text(json.dumps({"processor_class": "KimiK3Processor", **processor}))

    norm = build_kimi_pixel_norm(_model_config(tmp_path, cached_config), _vision_tower())
    assert norm is not None
    expected = ((torch.arange(256, dtype=torch.float32, device="cpu") / 255.0 - 0.5) * 2).bfloat16()
    assert torch.equal(norm.lut.view(torch.int16), expected.view(torch.int16))


def test_build_keeps_missing_media_config_error(tmp_path):
    with pytest.raises(KeyError, match="media_proc_cfg"):
        build_kimi_pixel_norm(_model_config(tmp_path, {}), _vision_tower())
