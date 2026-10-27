# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Legacy 310P Qwen3-VL shards require CPU normalization on vLLM 0.32+."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import save_file
from transformers import Qwen2VLImageProcessor
from vllm.config import MultiModalConfig
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.fusion.mm_input_norm import IdentityInputNorm, build_mm_input_norm
from vllm.model_executor.model_loader import ShardedStateLoader
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.fixture(params=["Qwen3vl-8B-acc", "Qwen3vl-8B-perf"])
def legacy_config(request):
    config_path = Path(__file__).resolve().parents[3] / "tests/e2e/cases/models/configs/Qwen/Qwen3VL-8B-310p.yaml"
    cases = yaml.safe_load(config_path.read_text(encoding="utf-8"))["test_cases"]
    case = next(case for case in cases if case["name"] == request.param)
    command = case["server_cmd"] + case.get("server_cmd_extra", [])
    assert command[command.index("--load-format") + 1] == "sharded_state"

    # Parse the actual normalization CLI option without downloading a model.
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([arg for arg in command if "mm-device-do-normalize" in arg])
    assert args.mm_device_do_normalize is False
    mm_config = MultiModalConfig(mm_device_do_normalize=args.mm_device_do_normalize)
    return SimpleNamespace(multimodal_config=mm_config)


def test_legacy_shard_loads_without_input_norm_buffers(legacy_config, tmp_path, monkeypatch):
    model = torch.nn.Module()
    model.visual = torch.nn.Module()
    model.visual.proj = torch.nn.Linear(2, 2, bias=False)
    legacy_state = {"visual.proj.weight": torch.ones(2, 2)}
    save_file(legacy_state, tmp_path / "model-rank-0-part-0.safetensors")

    model.visual.input_norm = build_mm_input_norm(legacy_config)
    assert isinstance(model.visual.input_norm, IdentityInputNorm)
    assert set(model.state_dict()) == set(legacy_state)
    loader = ShardedStateLoader(LoadConfig(load_format="sharded_state"))
    # Only rank resolution is mocked; exercise the real safetensors loader and
    # its strict missing-key check with a shard saved before input_norm existed.
    monkeypatch.setattr(
        ShardedStateLoader,
        "_render_default_pattern",
        staticmethod(lambda *args, **kwargs: "model-rank-0-part-*.safetensors"),
    )
    model_config = SimpleNamespace(model=str(tmp_path), model_weights=None)
    loader.load_weights(model, model_config)
    torch.testing.assert_close(model.visual.proj.weight, legacy_state["visual.proj.weight"])

    # A genuinely missing learned weight must still fail.
    model.visual.other = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(ValueError, match=r"Missing keys.*visual.other.weight"):
        loader.load_weights(model, model_config)


def test_cpu_normalizes_exactly_once(legacy_config):
    mm_config = legacy_config.multimodal_config
    kwargs = mm_config.merge_mm_processor_kwargs({})
    assert kwargs.get("do_rescale", True)
    assert kwargs.get("do_normalize", True)
    processor = Qwen2VLImageProcessor(min_pixels=56 * 56, max_pixels=56 * 56)
    # Different channels and the full byte range catch omitted normalization,
    # wrong channel layout, and applying the affine transform twice.
    image = (np.arange(56 * 56 * 3).reshape(56, 56, 3) % 256).astype(np.uint8)
    pixels = processor(images=[image], return_tensors="pt", **kwargs)["pixel_values"]
    raw = processor(images=[image], return_tensors="pt", do_rescale=False, do_normalize=False)["pixel_values"]
    channels = raw.float().reshape(raw.shape[0], 3, -1)
    mean = torch.tensor(processor.image_mean).view(1, 3, 1)
    std = torch.tensor(processor.image_std).view(1, 3, 1)
    expected = ((channels * processor.rescale_factor - mean) / std).reshape_as(pixels)
    torch.testing.assert_close(pixels, expected)

    norm = build_mm_input_norm(legacy_config)
    # This is the normalization called by Qwen3-VL before patch embedding.
    torch.testing.assert_close(norm(pixels, torch.float16), expected.to(torch.float16))
