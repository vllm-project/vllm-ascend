# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.models import kimi_k3_dspark
from vllm_ascend.models.kimi_k3 import AscendKimiLinearModel
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dspark import mla
from vllm_ascend.worker.v2.spec_decode.dspark.mla import AscendMLADSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


def make_speculator():
    spec = AscendMLADSparkSpeculator.__new__(AscendMLADSparkSpeculator)
    spec.vllm_config = SimpleNamespace()
    spec.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(target_layer_ids=[0, 2], target_hidden_size=4, num_target_layers=2)
    )
    spec.num_query_per_req = 5
    spec.input_buffers = SimpleNamespace(positions=torch.arange(20))
    return spec


def make_target():
    return SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=4, hidden_size=4),
            aux_hidden_state_layers=(1, 3),
        ),
        set_dspark_aux_capture_materialized=MagicMock(),
    )


@pytest.mark.parametrize("architecture", ["K3DSparkModel", "Qwen3DSparkModel", "DSparkDraftModel"])
def test_routes_only_k3_mla_to_specialization(monkeypatch, architecture):
    import vllm_ascend.worker.v2.spec_decode.dspark.speculator as shared

    mla_constructor = MagicMock()
    shared_constructor = MagicMock()
    monkeypatch.setattr(mla, "AscendMLADSparkSpeculator", mla_constructor)
    monkeypatch.setattr(shared, "AscendDSparkSpeculator", shared_constructor)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            use_dspark=lambda: True,
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
        )
    )
    init_speculator(config, torch.device("cpu"))
    assert mla_constructor.call_count == (architecture == "K3DSparkModel")
    assert shared_constructor.call_count == (architecture != "K3DSparkModel")


@pytest.mark.parametrize("wrapped", [False, True])
def test_raw_contract_and_rotation_preserved_during_loading(monkeypatch, wrapped):
    spec, target = make_speculator(), make_target()
    config = spec.draft_model_config.hf_config
    monkeypatch.setattr(mla, "get_rotation_path", lambda _: "/rotation")
    draft = object()

    def load(*args):
        assert config._ascend_target_rotation_path == "/rotation"
        return draft

    monkeypatch.setattr(AscendDSparkSpeculator, "load_draft_model", load)
    outer = SimpleNamespace(get_language_model=lambda: target) if wrapped else target
    assert spec.load_draft_model(outer, set()) is draft
    target.set_dspark_aux_capture_materialized.assert_called_once_with(False)
    assert not hasattr(config, "_ascend_target_rotation_path")


def test_rotation_restored_after_loader_failure(monkeypatch):
    spec = make_speculator()
    config = spec.draft_model_config.hf_config
    config._ascend_target_rotation_path = "original"
    monkeypatch.setattr(mla, "get_rotation_path", lambda _: None)
    monkeypatch.setattr(AscendDSparkSpeculator, "load_draft_model", MagicMock(side_effect=RuntimeError("load failed")))
    with pytest.raises(RuntimeError, match="load failed"):
        spec.load_draft_model(make_target(), set())
    assert config._ascend_target_rotation_path == "original"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("target_layer_ids", [], "requires target_layer_ids"),
        ("target_layer_ids", [0, 0], "Invalid"),
        ("target_layer_ids", [-1, 2], "Invalid"),
        ("target_layer_ids", [0, 9], "Invalid"),
        ("target_layer_ids", [1, 2], "boundaries do not match"),
        ("target_hidden_size", 8, "hidden sizes"),
        ("num_target_layers", 3, "num_target_layers"),
    ],
)
def test_rejects_invalid_raw_contract(monkeypatch, field, value, message):
    spec = make_speculator()
    setattr(spec.draft_model_config.hf_config, field, value)
    monkeypatch.setattr(mla, "get_rotation_path", lambda _: None)
    monkeypatch.setattr(AscendDSparkSpeculator, "load_draft_model", lambda *args: object())
    with pytest.raises(ValueError, match=message):
        spec.load_draft_model(make_target(), set())


def test_raw_prefix_capture_does_not_add_attnres_bank():
    state = torch.arange(8).view(2, 4)
    target = SimpleNamespace(aux_hidden_state_layers=(1, 3))
    captured = AscendKimiLinearModel._capture_raw_dspark_aux_hidden_state(target, [], 1, state)
    assert captured[0] is state
    assert captured[0].ndim == 2
    assert AscendKimiLinearModel._capture_raw_dspark_aux_hidden_state(target, [], 2, state) == []


def test_padded_mla_query_lengths_are_nested():
    spec = make_speculator()
    metadata = {"draft.0": SimpleNamespace(decode=SimpleNamespace(actual_seq_lengths_q=[5, 5]))}
    assert spec._update_draft_attn_metadata(metadata, 2) is metadata
    assert metadata["draft.0"].decode.actual_seq_lengths_q == [5, 10]
    assert not hasattr(metadata["draft.0"], "actual_seq_lengths_q")


@pytest.mark.parametrize("metadata", [{}, {"draft": SimpleNamespace(decode=None)}])
def test_rejects_missing_mla_decode_metadata(metadata):
    with pytest.raises((RuntimeError, TypeError)):
        make_speculator()._update_draft_attn_metadata(metadata, 1)


def test_capture_uses_descriptor_positions_and_restores_on_error(monkeypatch):
    spec = make_speculator()
    original = mla.dflash_cudagraph.build_attn_metadata
    builder = MagicMock()
    monkeypatch.setattr(mla, "build_attn_metadata", builder)
    with pytest.raises(RuntimeError, match="capture failed"), spec.draft_capture_context():
        mla.dflash_cudagraph.build_attn_metadata(num_tokens=10, num_reqs=2, causal=False)
        kwargs = builder.call_args.kwargs
        torch.testing.assert_close(kwargs["positions"], torch.arange(10))
        assert kwargs["is_prefilling"].tolist() == [False, False]
        assert kwargs["attn_state"] == AscendAttentionState.SpecDecoding
        assert kwargs["causal"] is False
        raise RuntimeError("capture failed")
    assert mla.dflash_cudagraph.build_attn_metadata is original


def test_reuses_shared_graph_initialization_and_propose():
    assert AscendMLADSparkSpeculator.init_cudagraph_manager is AscendDSparkSpeculator.init_cudagraph_manager
    assert AscendMLADSparkSpeculator.propose is AscendDSparkSpeculator.propose


def test_replay_metadata_clears_prefill_flags_and_preserves_causality(monkeypatch):
    spec = make_speculator()
    spec.input_batch = SimpleNamespace(num_reqs=1)
    spec._group_causal = {0: False}
    metadata = {"draft": SimpleNamespace(decode=SimpleNamespace(actual_seq_lengths_q=[5, 5]))}
    spec._build_draft_attn_metadata = MagicMock(return_value=metadata)
    captured = {}

    @contextmanager
    def factory(positions, pad, is_prefilling):
        captured.update(pad=pad, is_prefilling=is_prefilling)
        yield

    monkeypatch.setattr(mla, "build_draft_attn_metadata_factory", factory)
    result = spec.build_draft_attn_metadatas(2, torch.tensor([128]))
    assert captured["pad"] == 10
    assert captured["is_prefilling"].tolist() == [False, False]
    assert result[0]["draft"].decode.actual_seq_lengths_q == [5, 10]
    kwargs = spec._build_draft_attn_metadata.call_args.kwargs
    assert kwargs["num_reqs"] == 1
    assert kwargs["num_reqs_padded"] == 2
    assert kwargs["causal"] == {0: False}


@pytest.mark.parametrize(
    "direct,preserved,expected", [("target", "fallback", "target"), (None, "fallback", "fallback"), (None, None, None)]
)
def test_mla_model_recovers_target_rotation(monkeypatch, direct, preserved, expected):
    draft_config = SimpleNamespace(hf_config=SimpleNamespace(_ascend_target_rotation_path=preserved))
    config = SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft_config))
    monkeypatch.setattr(kimi_k3_dspark, "get_rotation_path", lambda _: direct)
    assert kimi_k3_dspark._get_target_rotation_path(config) == expected
