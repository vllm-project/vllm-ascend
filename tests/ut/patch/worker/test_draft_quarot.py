from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.patch.worker import patch_draft_quarot
from vllm_ascend.patch.worker.patch_draft_quarot import (
    get_rotation_path,
    make_qwen3_dspark_load_weights,
    transform_quarot_linear_weight,
)


def test_get_rotation_path_falls_back_to_model_description(tmp_path):
    description_path = tmp_path / "quant_model_description.json"
    description_path.write_text(
        '{"is_rot_used": true, "optional": {"quarot": {"rotation_map": '
        '{"global_rotation": "optional/quarot.safetensors"}}}}',
        encoding="utf-8",
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path)),
        quant_config=None,
    )

    assert get_rotation_path(config) == tmp_path / "optional/quarot.safetensors"


def test_transform_quarot_linear_weight_rotates_each_input_block():
    weight = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=torch.float32,
    )
    rotation = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    actual = transform_quarot_linear_weight(weight, rotation)
    expected = torch.tensor(
        [
            [2.0, 1.0, 4.0, 3.0],
            [6.0, 5.0, 8.0, 7.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)


def test_transform_quarot_linear_weight_rejects_invalid_input_width():
    weight = torch.ones((2, 3), dtype=torch.float32)
    rotation = torch.eye(2, dtype=torch.float32)

    with pytest.raises(ValueError, match="multiple of the QuaRot hidden size"):
        transform_quarot_linear_weight(weight, rotation)


def test_transform_quarot_linear_weight_preserves_dtype():
    weight = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    rotation = torch.eye(2, dtype=torch.float32)

    actual = transform_quarot_linear_weight(weight, rotation)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, weight)


def test_qwen3_dspark_wrapper_rotates_only_fc_weight(monkeypatch):
    captured_weights = {}
    rotation = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    def original_load_weights(_self, weights):
        captured_weights.update(dict(weights))
        return "loaded"

    monkeypatch.setattr(
        patch_draft_quarot,
        "get_rotataion_matrix",
        lambda _path: rotation,
    )

    model = SimpleNamespace(
        model=SimpleNamespace(
            fc=SimpleNamespace(weight=torch.empty(0)),
        )
    )
    fc_weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    lm_head_weight = torch.tensor([[5.0, 6.0]], dtype=torch.float32)

    result = make_qwen3_dspark_load_weights(
        "rotation.safetensors",
        original_load_weights,
    )(
        model,
        [
            ("fc.weight", fc_weight),
            ("lm_head.weight", lm_head_weight),
        ],
    )

    assert result == "loaded"
    torch.testing.assert_close(
        captured_weights["fc.weight"],
        torch.tensor([[2.0, 1.0, 4.0, 3.0]], dtype=torch.float32),
    )
    assert captured_weights["lm_head.weight"] is lm_head_weight
