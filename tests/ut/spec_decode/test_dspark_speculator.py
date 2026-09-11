# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MRV2 GQA DSpark target/draft contract."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils
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
    spec = _spec(_vllm_config(quarot=True), draft_config)

    assert spec.load_draft_model(target, set()) is draft
    assert not hasattr(draft_config, "_ascend_target_rotation_path")


def test_injected_rotation_path_is_removed_when_loading_fails(monkeypatch):
    draft_config = _gqa_config()
    target = _target()

    def fail_load(*args):
        raise ValueError("checkpoint load failed")

    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", fail_load)
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_rotation_path",
        lambda config: "/rotation",
    )
    spec = _spec(_vllm_config(quarot=True), draft_config)

    with pytest.raises(ValueError, match="checkpoint load failed"):
        spec.load_draft_model(target, set())
    assert not hasattr(draft_config, "_ascend_target_rotation_path")


def test_injected_rotation_path_does_not_require_draft_quant_config():
    config = SimpleNamespace(_ascend_target_rotation_path="/rotation")
    assert _get_draft_rotation_path(SimpleNamespace(quant_config=None), config) == Path("/rotation")


def test_quarot_loaded_weights_survive_upstream_sharing(monkeypatch):
    draft = AscendQwen3DSparkForCausalLM.__new__(AscendQwen3DSparkForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.model = torch.nn.Module()
    draft.model.embed_tokens = torch.nn.Embedding(4, 2)
    draft.lm_head = torch.nn.Linear(2, 4, bias=False)
    draft.rotation_path = Path("/rotation")
    draft.target_model_path = Path("/target")
    draft.has_own_embed_tokens = False
    draft.has_own_lm_head = False
    loaded_layers = []

    def load_layer(layer, *args):
        loaded_layers.append(layer)
        with torch.no_grad():
            layer.weight.fill_(len(loaded_layers))

    monkeypatch.setattr(Qwen3DSparkForCausalLM, "load_weights", lambda self, weights: set())
    monkeypatch.setattr("vllm_ascend.models.qwen3_dspark.get_rotation_matrix", lambda path: torch.eye(2))
    monkeypatch.setattr("vllm_ascend.models.qwen3_dspark.load_quarot_target_layer", load_layer)
    draft.load_weights([])
    assert draft.has_own_embed_tokens and draft.has_own_lm_head

    target = _target()
    target.model.embed_tokens = torch.nn.Embedding(4, 2)
    target.lm_head = torch.nn.Linear(2, 4, bias=False)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=_gqa_config(), model="/draft"),
            attention_backend=None,
            kv_cache_dtype=None,
        ),
        attention_config=SimpleNamespace(backend=None),
        cache_config=object(),
        model_config=SimpleNamespace(model="/target"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    monkeypatch.setattr(dspark_utils, "replace", lambda obj, **kwargs: SimpleNamespace(**(vars(obj) | kwargs)))
    monkeypatch.setattr(dspark_utils, "get_model", lambda **kwargs: draft)
    monkeypatch.setattr(dspark_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(dspark_utils, "get_target_lm_head", lambda *args: target.lm_head)
    monkeypatch.setattr("vllm.compilation.backends.set_model_tag", lambda *args: nullcontext())
    monkeypatch.setattr("vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal", lambda config: False)
    monkeypatch.setattr("vllm.model_executor.models.utils.get_draft_quant_config", lambda config: None)

    result = dspark_utils.load_dspark_model(target, config)
    assert result.model.embed_tokens is loaded_layers[0]
    assert result.lm_head is loaded_layers[1]
    torch.testing.assert_close(result.model.embed_tokens.weight, torch.ones(4, 2))
    torch.testing.assert_close(result.lm_head.weight, torch.full((4, 2), 2.0))


def test_process_weight_preserves_the_unrotated_projection():
    generator = torch.Generator().manual_seed(7)
    rotation, _ = torch.linalg.qr(torch.randn(_HIDDEN, _HIDDEN, dtype=torch.float64, generator=generator))
    inputs = torch.randn(3, 5, _HIDDEN, dtype=torch.float64, generator=generator)
    weight = torch.randn(_HIDDEN, 5 * _HIDDEN, dtype=torch.float64, generator=generator)

    expected = torch.nn.functional.linear(inputs.reshape(3, -1), weight)
    actual = torch.nn.functional.linear((inputs @ rotation).reshape(3, -1), process_weight(weight, rotation))

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-6)
