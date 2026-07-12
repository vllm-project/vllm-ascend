import json

import pytest
import torch

from vllm_ascend.models.dspark_quarot import (
    resolve_quarot_rotation_path,
    transform_fc_weight_for_quarot,
)
from vllm_ascend.models.qwen3_dspark import DSparkQwen3ForCausalLM


def test_transform_fc_weight_matches_runtime_unrotate():
    torch.manual_seed(7)
    hidden_size = 8
    num_features = 5
    output_size = 6

    rotation, _ = torch.linalg.qr(torch.randn(hidden_size, hidden_size))
    fc_weight = torch.randn(output_size, num_features * hidden_size)
    rotated_hidden = torch.randn(4, num_features * hidden_size)

    hidden_chunks = rotated_hidden.view(4, num_features, hidden_size)
    unrotated_hidden = torch.matmul(hidden_chunks, rotation.T).reshape(4, -1)
    expected = torch.nn.functional.linear(unrotated_hidden, fc_weight)

    transformed_weight = transform_fc_weight_for_quarot(fc_weight, rotation)
    actual = torch.nn.functional.linear(rotated_hidden, transformed_weight)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_resolve_quarot_rotation_path(tmp_path):
    rotation_path = tmp_path / "optional" / "quarot.safetensors"
    rotation_path.parent.mkdir()
    rotation_path.touch()
    description = {
        "is_rot_used": True,
        "optional": {
            "quarot": {
                "rotation_map": {
                    "global_rotation": "optional/quarot.safetensors",
                }
            }
        },
    }
    (tmp_path / "quant_model_description.json").write_text(
        json.dumps(description),
        encoding="utf-8",
    )

    assert resolve_quarot_rotation_path(tmp_path) == rotation_path


def test_transform_fc_weight_rejects_mismatched_width():
    with pytest.raises(ValueError, match="multiple"):
        transform_fc_weight_for_quarot(torch.randn(3, 10), torch.eye(4))


def test_dspark_quarot_load_requires_fc_weight(tmp_path):
    model = object.__new__(DSparkQwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    model._quarot_rotation_path = tmp_path / "quarot.safetensors"

    with pytest.raises(ValueError, match="did not provide fc.weight"):
        model.load_weights([])
