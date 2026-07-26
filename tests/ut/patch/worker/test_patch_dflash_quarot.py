import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend


MODULE_PATH = (
    Path(vllm_ascend.__file__).resolve().parent
    / "patch"
    / "worker"
    / "patch_draft_quarot.py"
)
SPEC = importlib.util.spec_from_file_location(
    "dflash_quarot_module_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
patch_draft_quarot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch_draft_quarot)


def test_transform_quarot_linear_weight_matches_block_diagonal():
    torch.manual_seed(0)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float32))
    weight = torch.randn(3, 12, dtype=torch.float32)

    actual = patch_draft_quarot.transform_quarot_linear_weight(
        weight, rotation
    )
    expected = weight @ torch.block_diag(rotation, rotation, rotation)

    torch.testing.assert_close(actual, expected)


def test_transform_quarot_linear_weight_rejects_invalid_width():
    weight = torch.randn(3, 10)
    rotation = torch.eye(4)

    with pytest.raises(ValueError, match="multiple of the QuaRot hidden size"):
        patch_draft_quarot.transform_quarot_linear_weight(weight, rotation)


def test_dflash_wrapper_transforms_only_fc_weight(monkeypatch):
    rotation = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    fc_weight = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    other_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    captured = {}

    monkeypatch.setattr(
        patch_draft_quarot,
        "get_rotataion_matrix",
        lambda _: rotation,
    )

    def original_load_weights(_self, weights):
        captured.update(dict(weights))
        return {"loaded": True}

    load_weights = patch_draft_quarot.make_dflash_load_weights(
        "unused.safetensors", original_load_weights
    )
    result = load_weights(
        object(),
        [("fc.weight", fc_weight), ("model.layers.0.weight", other_weight)],
    )

    expected_fc = fc_weight @ torch.block_diag(
        rotation, rotation, rotation, rotation
    )
    torch.testing.assert_close(captured["fc.weight"], expected_fc)
    torch.testing.assert_close(captured["model.layers.0.weight"], other_weight)
    assert result == {"loaded": True}


def test_dflash_wrapper_requires_fc_weight(monkeypatch):
    monkeypatch.setattr(
        patch_draft_quarot,
        "get_rotataion_matrix",
        lambda _: torch.eye(2),
    )

    def original_load_weights(_self, weights):
        list(weights)

    load_weights = patch_draft_quarot.make_dflash_load_weights(
        "unused.safetensors", original_load_weights
    )

    with pytest.raises(RuntimeError, match="did not provide fc.weight"):
        load_weights(object(), [("model.layers.0.weight", torch.eye(2))])


def test_non_quarot_target_leaves_dflash_loader_unchanged():
    target_config = SimpleNamespace(
        model_config=SimpleNamespace(model="unused"),
        quant_config=SimpleNamespace(quant_description={}),
    )
    original = patch_draft_quarot.DFlashQwen3ForCausalLM.load_weights

    patch_draft_quarot.patch_load_weights(target_config)

    assert patch_draft_quarot.DFlashQwen3ForCausalLM.load_weights is original
