# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MRV2 GQA DSpark target/draft contract."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.models.qwen3_dspark import (
    AscendQwen3DSparkForCausalLM,
    _get_draft_rotation_path,
    process_weight,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED,
    AscendDSparkSpeculator,
)

_HIDDEN = 8


def _spec(vllm_config, draft_hf_config) -> AscendDSparkSpeculator:
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.vllm_config = vllm_config
    spec.draft_model_config = SimpleNamespace(hf_config=draft_hf_config)
    return spec


def _target() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(embed_tokens=object()),
        lm_head=object(),
        set_dspark_aux_capture_materialized=MagicMock(),
    )


def _gqa_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["Qwen3DSparkModel"],
        model_type="qwen3",
        dspark_aux_hidden_state_format=DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED,
    )


def _vllm_config(*, quarot: bool) -> SimpleNamespace:
    quant_config = None
    if quarot:
        quant_config = SimpleNamespace(
            quant_description={"optional": {"quarot": {"rotation_map": {"global_rotation": "rotation.safetensors"}}}}
        )
    return SimpleNamespace(
        quant_config=quant_config,
        model_config=SimpleNamespace(model="/target"),
    )


def test_qwen3_gqa_draft_declares_materialized_format():
    assert AscendQwen3DSparkForCausalLM.dspark_aux_hidden_state_format == "materialized"


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(architectures=["Qwen3DSparkModel"]),
        SimpleNamespace(architectures=["DSparkDraftModel"], model_type="qwen3"),
        SimpleNamespace(model_type="qwen3"),
    ],
)
def test_qwen3_config_selects_materialized_target_capture(config):
    target = _target()
    AscendDSparkSpeculator._configure_target_aux_hidden_state_format(target, config)
    target.set_dspark_aux_capture_materialized.assert_called_once_with(True)


def test_undeclared_format_preserves_target_capture_mode():
    target = _target()
    AscendDSparkSpeculator._configure_target_aux_hidden_state_format(target, SimpleNamespace())
    target.set_dspark_aux_capture_materialized.assert_not_called()


def test_rejects_unknown_gqa_aux_hidden_format():
    with pytest.raises(ValueError, match="Unsupported GQA DSpark"):
        AscendDSparkSpeculator._configure_target_aux_hidden_state_format(
            _target(),
            SimpleNamespace(dspark_aux_hidden_state_format="raw"),
        )


def test_configures_capture_before_loading_draft(monkeypatch):
    events = []
    target = _target()
    target.set_dspark_aux_capture_materialized = lambda enabled: events.append(("capture", enabled))
    draft = SimpleNamespace(model=SimpleNamespace(embed_tokens=object()), lm_head=object())

    def _load(self, target_model, target_attn_layer_names):
        events.append(("load", None))
        return draft

    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", _load)
    spec = _spec(_vllm_config(quarot=False), _gqa_config())

    assert spec.load_draft_model(target, set()) is draft
    assert events == [("capture", True), ("load", None)]


def test_injects_rotation_before_draft_construction(monkeypatch):
    draft_config = _gqa_config()
    target = _target()
    draft = SimpleNamespace(model=SimpleNamespace(embed_tokens=object()), lm_head=object())

    def _load(self, target_model, target_attn_layer_names):
        assert draft_config._ascend_target_rotation_path == "/rotation"
        return draft

    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", _load)
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_rotation_path",
        lambda config: "/rotation",
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_target_lm_head",
        lambda *args: target.lm_head,
    )
    spec = _spec(_vllm_config(quarot=True), draft_config)

    assert spec.load_draft_model(target, set()) is draft
    assert not hasattr(draft_config, "_ascend_target_rotation_path")


def test_rejects_shared_quarot_embedding(monkeypatch):
    draft_config = _gqa_config()
    target = _target()
    draft = SimpleNamespace(model=SimpleNamespace(embed_tokens=target.model.embed_tokens), lm_head=object())
    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", lambda *args: draft)
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_rotation_path",
        lambda config: "/rotation",
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_target_lm_head",
        lambda *args: target.lm_head,
    )
    spec = _spec(_vllm_config(quarot=True), draft_config)

    with pytest.raises(RuntimeError, match="must not share target embed_tokens"):
        spec.load_draft_model(target, set())


def test_injected_rotation_path_does_not_require_draft_quant_config():
    config = SimpleNamespace(_ascend_target_rotation_path="/rotation")
    assert _get_draft_rotation_path(SimpleNamespace(quant_config=None), config) == Path("/rotation")


def test_process_weight_preserves_the_unrotated_projection():
    generator = torch.Generator().manual_seed(7)
    rotation, _ = torch.linalg.qr(torch.randn(_HIDDEN, _HIDDEN, dtype=torch.float64, generator=generator))
    inputs = torch.randn(3, 5, _HIDDEN, dtype=torch.float64, generator=generator)
    weight = torch.randn(_HIDDEN, 5 * _HIDDEN, dtype=torch.float64, generator=generator)

    expected = torch.nn.functional.linear(inputs.reshape(3, -1), weight)
    actual = torch.nn.functional.linear((inputs @ rotation).reshape(3, -1), process_weight(weight, rotation))

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-6)
