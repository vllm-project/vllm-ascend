# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_processing,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader
from vllm.model_executor.models.vision import FusedInputNorm

from vllm_ascend.patch.worker.patch_mm_input_norm import patch_mm_input_norm


@pytest.fixture(params=[True, False], ids=["v030-persistent", "already-nonpersistent"])
def norm(request, monkeypatch):
    original_init = FusedInputNorm.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Reproduce v0.30 persistence even if another patch already fixed it.
        for name in ("weight", "bias"):
            self.register_buffer(name, getattr(self, name), persistent=request.param)

    monkeypatch.setattr(FusedInputNorm, "__init__", init)
    patch_mm_input_norm()
    patched_init = FusedInputNorm.__init__
    patch_mm_input_norm()
    assert FusedInputNorm.__init__ is patched_init
    with torch.device("cpu"):
        yield FusedInputNorm([0.2, 0.4, 0.6], [0.25, 0.5, 0.75], 1 / 255)


def test_legacy_shard_loads_and_missing_parameter_still_fails(norm, tmp_path, monkeypatch):
    model = torch.nn.Module()
    # Ascend's Qwen3.5-VL preprocessing mixin owns normalization at model level.
    model.input_norm = norm
    model.visual = torch.nn.Module()
    model.visual.proj = torch.nn.Linear(2, 2, bias=False)
    legacy_state = {"visual.proj.weight": torch.ones(2, 2)}
    save_file(legacy_state, tmp_path / "model-rank-0-part-0.safetensors")
    loader = ShardedStateLoader(LoadConfig(load_format="sharded_state"))
    # Rank resolution is the only mocked part of the real v0.30 loader.
    monkeypatch.setattr("vllm.distributed.get_tensor_model_parallel_rank", lambda: 0)
    config = SimpleNamespace(model=str(tmp_path), model_weights=None, revision=None)
    loader.load_weights(model, config)
    torch.testing.assert_close(model.visual.proj.weight, legacy_state["visual.proj.weight"])
    assert set(model.state_dict()) == set(legacy_state)
    assert set(dict(norm.named_buffers())) == {"weight", "bias"}

    model.visual.other = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(ValueError, match=r"Missing keys.*visual.other.weight"):
        loader.load_weights(model, config)


def test_dummy_loading_preserves_normalization(norm):
    weight, bias = norm.weight.clone(), norm.bias.clone()
    DummyModelLoader(LoadConfig(load_format="dummy")).load_weights(norm, Mock())
    torch.testing.assert_close(norm.weight, weight, rtol=0, atol=0)
    torch.testing.assert_close(norm.bias, bias, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_device_normalization_matches_processor_transform(norm, dtype):
    # Full byte range and different channels catch missing/double normalization.
    pixels = torch.arange(256, dtype=torch.uint8).repeat(3).reshape(1, -1)
    mean = torch.tensor([0.2, 0.4, 0.6]).view(1, 3, 1)
    std = torch.tensor([0.25, 0.5, 0.75]).view(1, 3, 1)
    expected = ((pixels.float().reshape(1, 3, -1) / 255 - mean) / std).reshape_as(pixels)
    torch.testing.assert_close(norm(pixels, dtype), expected.to(dtype))


def test_identity_normalization_retains_none_buffers():
    norm = FusedInputNorm.identity()
    assert norm.weight is None
    assert norm.bias is None
    assert {"weight", "bias"} <= norm._non_persistent_buffers_set
    assert not norm.state_dict()
    pixels = torch.arange(24, dtype=torch.uint8).reshape(2, 12)
    torch.testing.assert_close(norm(pixels, torch.float16), pixels.to(torch.float16))


def test_layerwise_reload_preserves_normalization(norm):
    model = torch.nn.Module()
    model.input_norm = norm
    model.visual = torch.nn.Module()
    model.visual.proj = torch.nn.Linear(2, 2, bias=False)
    original_buffers = (norm.weight, norm.bias)
    original_values = tuple(buffer.clone() for buffer in original_buffers)
    original_pointers = tuple(buffer.data_ptr() for buffer in original_buffers)
    record_metadata_for_reloading(model)

    for updated_value in (2.0, 3.0):
        initialize_layerwise_reload(model)
        weight = model.visual.proj.weight
        updated_weight = torch.full(weight.shape, updated_value, device="cpu")
        weight.weight_loader(weight, updated_weight)
        finalize_layerwise_processing(model, SimpleNamespace(dtype=torch.float32))

        torch.testing.assert_close(model.visual.proj.weight, updated_weight)
        for name, original, value, pointer in zip(
            ("weight", "bias"), original_buffers, original_values, original_pointers
        ):
            buffer = getattr(norm, name)
            assert buffer is original
            assert buffer.data_ptr() == pointer
            torch.testing.assert_close(buffer, value, rtol=0, atol=0)
        assert {"weight", "bias"} <= norm._non_persistent_buffers_set
        assert set(model.state_dict()) == {"visual.proj.weight"}
