from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

import vllm_ascend.models as ascend_models
import vllm_ascend.models.qwen3_dspark as qwen3_dspark
from vllm_ascend.models.qwen3_dspark import (
    AscendQwen3DSparkForCausalLM,
    process_weight,
)


def test_qwen3_dspark_model_is_registered(monkeypatch):
    register_model = MagicMock()
    monkeypatch.setattr(
        ascend_models.ModelRegistry,
        "register_model",
        register_model,
    )

    ascend_models.register_model()

    register_model.assert_any_call(
        "Qwen3DSparkModel",
        "vllm_ascend.models.qwen3_dspark:AscendQwen3DSparkForCausalLM",
    )


def test_process_weight_rotates_each_hidden_size_block():
    linear_weight = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=torch.bfloat16,
    )
    rotation_weight = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )

    actual = process_weight(linear_weight, rotation_weight)

    expected = torch.tensor(
        [
            [2.0, 1.0, 4.0, 3.0],
            [6.0, 5.0, 8.0, 7.0],
        ],
        dtype=torch.bfloat16,
    )
    assert actual.dtype == linear_weight.dtype
    torch.testing.assert_close(actual, expected)


def test_process_weight_rejects_non_divisible_input_width():
    with pytest.raises(AssertionError, match="multiple of rotation weight"):
        process_weight(
            torch.ones((2, 3), dtype=torch.float32),
            torch.eye(2, dtype=torch.float32),
        )


@pytest.mark.parametrize(
    ("quant_config", "expected_rotation_path"),
    [
        (object(), Path("/tmp/quarot.safetensors")),
        (None, None),
    ],
)
def test_qwen3_dspark_initializes_rotation_path_only_for_quantized_target(
    monkeypatch,
    quant_config,
    expected_rotation_path,
):
    def fake_base_init(self, **kwargs):
        torch.nn.Module.__init__(self)

    get_rotation_path = MagicMock(return_value=expected_rotation_path)
    monkeypatch.setattr(
        Qwen3DSparkForCausalLM,
        "__init__",
        fake_base_init,
    )
    monkeypatch.setattr(
        qwen3_dspark,
        "get_rotation_path",
        get_rotation_path,
    )
    vllm_config = SimpleNamespace(quant_config=quant_config)

    model = AscendQwen3DSparkForCausalLM(vllm_config=vllm_config)

    assert model.rotation_path == expected_rotation_path
    if quant_config is None:
        get_rotation_path.assert_not_called()
    else:
        get_rotation_path.assert_called_once_with(vllm_config)


def test_qwen3_dspark_load_weights_rotates_only_fc(monkeypatch):
    captured_weights = []

    def fake_load_weights(self, weights):
        captured_weights.extend(list(weights))

    monkeypatch.setattr(
        Qwen3DSparkForCausalLM,
        "load_weights",
        fake_load_weights,
    )
    monkeypatch.setattr(
        qwen3_dspark,
        "get_rotataion_matrix",
        lambda _: torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    model = AscendQwen3DSparkForCausalLM.__new__(AscendQwen3DSparkForCausalLM)
    torch.nn.Module.__init__(model)
    model.rotation_path = Path("/tmp/quarot.safetensors")
    fc_weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    embed_weight = torch.tensor([[5.0, 6.0, 7.0, 8.0]])

    model.load_weights(
        [
            ("model.fc.weight", fc_weight),
            ("model.embed_tokens.weight", embed_weight),
        ]
    )

    assert [name for name, _ in captured_weights] == [
        "model.fc.weight",
        "model.embed_tokens.weight",
    ]
    torch.testing.assert_close(
        captured_weights[0][1],
        torch.tensor([[2.0, 1.0, 4.0, 3.0]]),
    )
    torch.testing.assert_close(captured_weights[1][1], embed_weight)


def test_qwen3_dspark_load_weights_without_rotation_is_passthrough(
    monkeypatch,
):
    captured_weights = []

    def fake_load_weights(self, weights):
        captured_weights.extend(list(weights))

    monkeypatch.setattr(
        Qwen3DSparkForCausalLM,
        "load_weights",
        fake_load_weights,
    )
    model = AscendQwen3DSparkForCausalLM.__new__(AscendQwen3DSparkForCausalLM)
    torch.nn.Module.__init__(model)
    model.rotation_path = None
    weights = [("model.fc.weight", torch.tensor([[1.0, 2.0]]))]

    model.load_weights(iter(weights))

    assert captured_weights == weights
