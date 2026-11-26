# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

import vllm_ascend.utils as ascend_utils
from vllm_ascend.models import quarot_utils, qwen3_dflash2
from vllm_ascend.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM

QUAROT_LOGGER = "vllm_ascend.models.quarot_utils"


@pytest.fixture(autouse=True)
def reset_quarot_globals(monkeypatch):
    monkeypatch.setattr(quarot_utils._warn_rotation_once, "_warnings", set(), raising=False)
    monkeypatch.setattr(ascend_utils, "_IS_ROT_WEIGHT_USED", None)


def make_config(model_dir, *, is_rot_used=None, declare_rotation=False, hidden_size=8):
    quant_description = {} if is_rot_used is None else {"is_rot_used": is_rot_used}
    if declare_rotation:
        quant_description["optional"] = {"quarot": {"rotation_map": {"global_rotation": "declared/quarot.safetensors"}}}
    return SimpleNamespace(
        quant_config=SimpleNamespace(quant_description=quant_description),
        model_config=SimpleNamespace(model=str(model_dir), hf_text_config=SimpleNamespace(hidden_size=hidden_size)),
    )


def write_rotation_file(model_dir, relative_path, hidden_size=8):
    path = model_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"global_rotation": torch.eye(hidden_size, dtype=torch.float32)}, path)
    return path


def make_dflash2_draft():
    draft = DFlash2Qwen3ForCausalLM.__new__(DFlash2Qwen3ForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.model = torch.nn.Module()
    draft.model.use_aux_hidden_state = True
    draft.model.fc = torch.nn.Linear(4, 2, bias=False)
    return draft


def test_declared_rotation_path_returned_unchanged(tmp_path, caplog):
    config = make_config(tmp_path, is_rot_used=True, declare_rotation=True)
    declared = tmp_path / "declared" / "quarot.safetensors"
    with caplog.at_level(logging.WARNING, logger=QUAROT_LOGGER):
        assert quarot_utils.resolve_rotation_path(config) == declared
        assert quarot_utils.resolve_rotation_path(config, allow_fallback=True) == declared
    assert caplog.records == []


def test_config_without_quant_config_stays_silent(tmp_path, caplog):
    config = SimpleNamespace(
        quant_config=None,
        model_config=SimpleNamespace(model=str(tmp_path), hf_text_config=SimpleNamespace(hidden_size=8)),
    )
    with caplog.at_level(logging.WARNING, logger=QUAROT_LOGGER):
        assert quarot_utils.resolve_rotation_path(config) is None
        assert quarot_utils.resolve_rotation_path(config, allow_fallback=True) is None
    assert caplog.records == []


def test_non_rotated_target_never_falls_back(tmp_path, caplog):
    write_rotation_file(tmp_path, "optional/quarot.safetensors")
    config = make_config(tmp_path, is_rot_used=False)
    with caplog.at_level(logging.WARNING, logger=QUAROT_LOGGER):
        assert quarot_utils.resolve_rotation_path(config, allow_fallback=True) is None
    assert caplog.records == []


def test_missing_map_without_fallback_permission_is_silent(tmp_path, caplog):
    write_rotation_file(tmp_path, "optional/quarot.safetensors")
    config = make_config(tmp_path, is_rot_used=True)
    with caplog.at_level(logging.WARNING, logger=QUAROT_LOGGER):
        assert quarot_utils.resolve_rotation_path(config) is None
        assert quarot_utils.resolve_rotation_path(config, allow_fallback=False) is None
    assert caplog.records == []


def test_fallback_used_when_conventional_file_exists(tmp_path, caplog):
    write_rotation_file(tmp_path, "optional/quarot.safetensors")
    config = make_config(tmp_path, is_rot_used=True)
    with caplog.at_level(logging.WARNING, logger=QUAROT_LOGGER):
        resolved = quarot_utils.resolve_rotation_path(config, allow_fallback=True)
    assert resolved == tmp_path / "optional" / "quarot.safetensors"
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "falling back to the conventional" in messages[0]


def test_fallback_rejected_on_shape_mismatch(tmp_path, caplog):
    write_rotation_file(tmp_path, "optional/quarot.safetensors", hidden_size=4)
    config = make_config(tmp_path, is_rot_used=True, hidden_size=8)
    with caplog.at_level(logging.WARNING, logger=QUAROT_LOGGER):
        assert quarot_utils.resolve_rotation_path(config, allow_fallback=True) is None
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "refusing to align" in messages[0]


def test_fallback_missing_file_warns_without_raising(tmp_path, caplog):
    config = make_config(tmp_path, is_rot_used=True)
    with caplog.at_level(logging.WARNING, logger=QUAROT_LOGGER):
        assert quarot_utils.resolve_rotation_path(config, allow_fallback=True) is None
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "does not exist" in messages[0]
    assert "accept ~0 tokens" in messages[0]


def test_fallback_warning_emitted_once(tmp_path, caplog):
    write_rotation_file(tmp_path, "optional/quarot.safetensors")
    config = make_config(tmp_path, is_rot_used=True)
    with caplog.at_level(logging.WARNING, logger=QUAROT_LOGGER):
        quarot_utils.resolve_rotation_path(config, allow_fallback=True)
        quarot_utils.resolve_rotation_path(config, allow_fallback=True)
    assert len(caplog.records) == 1


def test_align_draft_weights_forwards_explicit_allow_fallback(monkeypatch):
    seen = {}

    def fake_resolve(config, *, allow_fallback=False):
        seen["allow_fallback"] = allow_fallback
        return None

    monkeypatch.setattr(quarot_utils, "resolve_rotation_path", fake_resolve)
    monkeypatch.setattr(quarot_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    projection = torch.nn.Linear(2, 2, bias=False)
    quarot_utils.align_draft_weights(SimpleNamespace(), projection, SimpleNamespace(), allow_fallback=True)
    assert seen["allow_fallback"] is True


def test_align_draft_weights_defaults_to_fallback_enabled(monkeypatch):
    seen = {}

    def fake_resolve(config, *, allow_fallback=False):
        seen["allow_fallback"] = allow_fallback
        return None

    monkeypatch.setattr(quarot_utils, "resolve_rotation_path", fake_resolve)
    monkeypatch.setattr(quarot_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    projection = torch.nn.Linear(2, 2, bias=False)
    quarot_utils.align_draft_weights(SimpleNamespace(), projection, SimpleNamespace())
    assert seen["allow_fallback"] is True


def test_dflash2_post_process_aligns_via_fallback(tmp_path, monkeypatch):
    write_rotation_file(tmp_path, "optional/quarot.safetensors")
    config = make_config(tmp_path, is_rot_used=True)
    align = MagicMock()
    monkeypatch.setattr(qwen3_dflash2, "align_draft_weights", align)
    draft = make_dflash2_draft()
    draft.post_process(config)
    align.assert_called_once_with(
        draft,
        draft.model.fc,
        config,
        allow_fallback=True,
        rotation_path=tmp_path / "optional" / "quarot.safetensors",
    )


def test_dflash2_post_process_survives_missing_fallback_file(tmp_path, caplog, monkeypatch):
    config = make_config(tmp_path, is_rot_used=True)
    align = MagicMock()
    monkeypatch.setattr(qwen3_dflash2, "align_draft_weights", align)
    draft = make_dflash2_draft()
    with caplog.at_level(logging.WARNING, logger=QUAROT_LOGGER):
        draft.post_process(config)
    align.assert_not_called()
    assert any("does not exist" in record.getMessage() for record in caplog.records)


def test_align_draft_weights_applies_fallback_for_dspark_callers(tmp_path, monkeypatch):
    """The default-on fallback must reach the DSpark call sites that pass no flag."""
    write_rotation_file(tmp_path, "optional/quarot.safetensors")
    config = make_config(tmp_path, is_rot_used=True)
    projection = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
    with torch.no_grad():
        projection.weight.copy_(torch.ones_like(projection.weight))
    draft = SimpleNamespace(model=SimpleNamespace(), has_own_embed_tokens=True, has_own_lm_head=True)
    monkeypatch.setattr(quarot_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(quarot_utils, "get_rotation_matrix", lambda path: 2 * torch.eye(8))
    quarot_utils.align_draft_weights(draft, projection, config)
    torch.testing.assert_close(projection.weight, 2 * torch.ones_like(projection.weight))


def test_align_draft_weights_explicit_false_restores_silent_skip(tmp_path, caplog, monkeypatch):
    write_rotation_file(tmp_path, "optional/quarot.safetensors")
    config = make_config(tmp_path, is_rot_used=True)
    projection = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
    with torch.no_grad():
        projection.weight.copy_(torch.ones_like(projection.weight))
    monkeypatch.setattr(quarot_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    with caplog.at_level(logging.WARNING, logger=QUAROT_LOGGER):
        quarot_utils.align_draft_weights(SimpleNamespace(), projection, config, allow_fallback=False)
    torch.testing.assert_close(projection.weight, torch.ones_like(projection.weight))
    assert caplog.records == []


def test_align_draft_weights_skips_resolution_for_pre_resolved_path(tmp_path, monkeypatch):
    rotation_path = write_rotation_file(tmp_path, "optional/quarot.safetensors")
    resolve = MagicMock()
    monkeypatch.setattr(quarot_utils, "resolve_rotation_path", resolve)
    projection = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
    with torch.no_grad():
        projection.weight.copy_(torch.ones_like(projection.weight))
    draft = SimpleNamespace(model=SimpleNamespace(), has_own_embed_tokens=True, has_own_lm_head=True)
    monkeypatch.setattr(quarot_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(quarot_utils, "get_rotation_matrix", lambda path: 2 * torch.eye(8))
    quarot_utils.align_draft_weights(
        draft, projection, make_config(tmp_path, is_rot_used=True), rotation_path=rotation_path
    )
    resolve.assert_not_called()
    torch.testing.assert_close(projection.weight, 2 * torch.ones_like(projection.weight))
