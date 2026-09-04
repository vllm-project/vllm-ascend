import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.patch.platform.patch_speculative_config import (
    _checkpoint_has_qwen3_5_mtp_weights,
    _validate_qwen3_5_mtp_checkpoint,
)


def _write_index(checkpoint_dir, weight_names):
    weight_map = {name: "model-00001-of-00001.safetensors" for name in weight_names}
    index_path = checkpoint_dir / "model.safetensors.index.json"
    with index_path.open("w", encoding="utf-8") as index_file:
        json.dump({"metadata": {}, "weight_map": weight_map}, index_file)


def _make_speculative_config(model_path, method="qwen3_5_mtp"):
    return SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(model=str(model_path)),
    )


def test_qwen3_5_mtp_preflight_accepts_indexed_checkpoint(tmp_path):
    _write_index(
        tmp_path,
        ["model.layers.0.mlp.down_proj.weight", "mtp.fc.weight"],
    )

    assert _checkpoint_has_qwen3_5_mtp_weights(tmp_path) is True
    _validate_qwen3_5_mtp_checkpoint(_make_speculative_config(tmp_path))


def test_qwen3_5_mtp_preflight_rejects_sft_checkpoint_without_mtp(tmp_path):
    _write_index(tmp_path, ["model.layers.0.mlp.down_proj.weight"])

    assert _checkpoint_has_qwen3_5_mtp_weights(tmp_path) is False
    with pytest.raises(ValueError, match=r"does not contain any 'mtp\.\*' tensors"):
        _validate_qwen3_5_mtp_checkpoint(_make_speculative_config(tmp_path))


def test_qwen3_5_mtp_preflight_reads_single_safetensors_file(tmp_path):
    save_file(
        {"mtp.layers.0.input_layernorm.weight": torch.zeros(1)},
        tmp_path / "model.safetensors",
    )

    assert _checkpoint_has_qwen3_5_mtp_weights(tmp_path) is True


def test_qwen3_5_mtp_preflight_skips_malformed_safetensors_file(tmp_path):
    (tmp_path / "model.safetensors").write_text("not a safetensors file")

    assert _checkpoint_has_qwen3_5_mtp_weights(tmp_path) is None


def test_qwen3_5_mtp_preflight_skips_remote_or_uninspectable_checkpoint():
    config = _make_speculative_config("Qwen/Qwen3.5-9B")

    assert _checkpoint_has_qwen3_5_mtp_weights("Qwen/Qwen3.5-9B") is None
    _validate_qwen3_5_mtp_checkpoint(config)


def test_qwen3_5_mtp_preflight_does_not_affect_other_methods(tmp_path):
    _write_index(tmp_path, ["model.layers.0.mlp.down_proj.weight"])

    _validate_qwen3_5_mtp_checkpoint(_make_speculative_config(tmp_path, method="deepseek_mtp"))
