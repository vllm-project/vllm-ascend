# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the checkpoint/oracle boundaries shared by vision reload tests."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import load_file

from tests.e2e.common.rlhf.weight_transfer.checkpoint_adapters import deepseek_v4_vision_params
from tests.e2e.common.rlhf.weight_transfer.deepseek_vision import deepseek_vision_case
from tests.e2e.common.rlhf.weight_transfer.reference_check import generation_signature, reference_serve_args
from tests.e2e.common.rlhf.weight_transfer.weight_source import FixedRandomWeightSource


def test_graph_modes_are_case_local(monkeypatch):
    monkeypatch.setenv("VLLM_RL_DSV4_CUDAGRAPH_MODE", "NONE")
    eager = deepseek_vision_case()
    monkeypatch.setenv("VLLM_RL_DSV4_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
    graph = deepseek_vision_case()
    for case, expected in ((eager, "NONE"), (graph, "FULL_DECODE_ONLY")):
        args = reference_serve_args(case, port=8000, gpu_memory_utilization=0.45)
        assert json.loads(args[args.index("--compilation-config") + 1])["cudagraph_mode"] == expected
    assert eager.id != graph.id


def test_vision_checkpoint_metadata_includes_tower_sentinels_and_router_state():
    config = SimpleNamespace(
        vision_dim=16,
        vision_inter_dim=32,
        vision_patch_size=2,
        vision_downsample_ratio=2,
        hidden_size=32,
        vision_n_layers=2,
        num_hidden_layers=2,
        n_routed_experts=8,
        mlp_layer_types=["hash_moe", "moe"],
    )
    params = {name: (shape, dtype) for name, shape, dtype in deepseek_v4_vision_params(config)}
    assert params["vision.blocks.1.attn.wqkv.weight"] == ((48, 16), torch.bfloat16)
    assert params["aligner.w1.weight"] == ((32, 64), torch.bfloat16)
    assert params["image_start"] == ((32,), torch.bfloat16)
    assert params["model.layers.0.mlp.gate.bias_vl"] == ((8,), torch.float32)
    assert params["model.layers.1.mlp.gate.e_score_correction_bias"] == ((8,), torch.float32)
    assert "model.layers.0.mlp.gate.e_score_correction_bias" not in params


def test_local_startup_checkpoint_uses_same_payload_as_transfer(tmp_path):
    config_bytes = b'{"model_type": "deepseek_v4", "num_hidden_layers": 1}\n'
    (tmp_path / "config.json").write_bytes(config_bytes)
    case = replace(deepseek_vision_case(), model=str(tmp_path), extra_param_builder=None)
    source = object.__new__(FixedRandomWeightSource)
    source._case = case
    source._device = torch.device("cpu")
    source._num_experts = 8
    source._parameters = [("image_start", (4,), torch.bfloat16), ("gate.tid2eid", (4, 6), torch.int64)]
    with patch.object(torch.npu, "manual_seed"):
        expected = dict(source)
        with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("local config must not use the Hub")):
            source.write_checkpoint(tmp_path / "startup")
    assert (tmp_path / "startup/config.json").read_bytes() == config_bytes
    actual = load_file(str(tmp_path / "startup/model.safetensors"))
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name])
    assert all(row.unique().numel() == 6 for row in actual["gate.tid2eid"])


def test_image_signature_exercises_chat_and_records_logprobs():
    client = MagicMock()
    client.completions.create.return_value.choices = [
        SimpleNamespace(text="text", logprobs=SimpleNamespace(token_logprobs=[-0.5]))
    ]
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(
            message=SimpleNamespace(content="image"), logprobs=SimpleNamespace(content=[SimpleNamespace(logprob=-0.75)])
        )
    ]
    signature = generation_signature(client, "vision", image_prompt=True)
    assert signature == [("text", (-0.5,)), ("text", (-0.5,)), ("image", (-0.75,))]
    content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("layers,vision,experts", [(0, 2, 8), (4, 0, 8), (4, 2, 5)])
def test_invalid_vision_reduction_fails_before_starting_server(monkeypatch, layers, vision, experts):
    monkeypatch.setenv("VLLM_RL_DSV4_TEXT_LAYERS", str(layers))
    monkeypatch.setenv("VLLM_RL_DSV4_VISION_LAYERS", str(vision))
    monkeypatch.setenv("VLLM_RL_DSV4_EXPERTS", str(experts))
    with pytest.raises(ValueError, match="Vision reload"):
        deepseek_vision_case()
