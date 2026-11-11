# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import save_file
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.fusion.mm_input_norm import FusedMMInputNorm
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader

from vllm_ascend.patch.worker.patch_mm_input_norm import patch_mm_input_norm


@pytest.fixture(params=[True, False], ids=["v032-persistent", "upstream-nonpersistent"])
def norm(request, monkeypatch):
    original_init = FusedMMInputNorm.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Exercise both upstream contracts even when CI already has #62767.
        for name in ("weight", "bias"):
            self.register_buffer(name, getattr(self, name), persistent=request.param)

    monkeypatch.setattr(FusedMMInputNorm, "__init__", init)
    patch_mm_input_norm()
    patched_init = FusedMMInputNorm.__init__
    patch_mm_input_norm()
    assert FusedMMInputNorm.__init__ is patched_init
    # Config construction must not initialize Ascend logging or device state.
    with patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"):
        vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config), torch.device("cpu"):
        yield FusedMMInputNorm([0.2, 0.4, 0.6], [0.25, 0.5, 0.75], 1 / 255)


def test_legacy_shard_loads_and_missing_parameter_still_fails(norm, tmp_path, monkeypatch):
    model = torch.nn.Module()
    model.visual = torch.nn.Module()
    model.visual.input_norm = norm
    model.visual.proj = torch.nn.Linear(2, 2, bias=False)
    legacy_state = {"visual.proj.weight": torch.ones(2, 2)}
    save_file(legacy_state, tmp_path / "model-rank-0-part-0.safetensors")
    loader = ShardedStateLoader(LoadConfig(load_format="sharded_state"))
    # Rank resolution is the only mocked part of the real safetensors loader.
    monkeypatch.setattr(
        ShardedStateLoader,
        "_render_default_pattern",
        staticmethod(lambda *args, **kwargs: "model-rank-0-part-*.safetensors"),
    )
    config = SimpleNamespace(model=str(tmp_path), model_weights=None)
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
