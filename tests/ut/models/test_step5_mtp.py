# SPDX-License-Identifier: Apache-2.0
# UTs for vllm_ascend/models/step5/mtp.py -- CPU only.
#
# The heavy collaborators (VocabParallelEmbedding, LogitsProcessor, the
# shared head and the Ascend draft decoder block) are patched in the module
# namespace; only the Ascend MTP wiring itself is exercised.

import dataclasses
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

import vllm_ascend.models.step5.mtp as step5_mtp
from vllm_ascend.models.step5.mtp import (
    AscendStep5MTP,
    AscendStep5MultiTokenPredictor,
    AscendStep5MultiTokenPredictorLayer,
)


def _mtp_config(**overrides):
    kwargs = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=4,
        num_nextn_predict_layers=2,
        rms_norm_eps=1e-6,
        zero_centered=True,
        norm_dtype="fp32",
        fp32_residual_connection=False,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# AscendStep5MultiTokenPredictorLayer
# ---------------------------------------------------------------------------


@pytest.fixture
def layer_env(monkeypatch):
    built: dict[str, Any] = {}

    class _Norm(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            built.setdefault("norms", []).append((args, kwargs))

    def _shared_head(config, quant_config, prefix):
        built["shared_head"] = SimpleNamespace(config=config, quant_config=quant_config, prefix=prefix)
        return SimpleNamespace(head=MagicMock())

    def _decoder_layer(vllm_config, prefix):
        built["decoder"] = (vllm_config, prefix)
        return SimpleNamespace()

    mtp_vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=_mtp_config()))
    monkeypatch.setattr(step5_mtp, "AscendStep5RMSNorm", _Norm)
    monkeypatch.setattr(step5_mtp, "Step5MTPSharedHead", _shared_head)
    monkeypatch.setattr(step5_mtp, "AscendStep5DecoderLayer", _decoder_layer)
    monkeypatch.setattr(step5_mtp, "_get_mtp_vllm_config", lambda vllm_config: mtp_vllm_config)
    built["mtp_vllm_config"] = mtp_vllm_config
    return built


def test_predictor_layer_wires_ascend_modules(layer_env):
    target_vllm_config = SimpleNamespace(marker="target", quant_config=None)
    layer = AscendStep5MultiTokenPredictorLayer(target_vllm_config, "model.layers.4")

    assert layer.fp32_residual_connection is False

    # enorm/hnorm are AscendStep5RMSNorm with the fp32 norm dtype
    norms = layer_env["norms"]
    assert len(norms) == 2
    for args, kwargs in norms:
        assert args == (32,)
        assert kwargs == {"eps": 1e-6, "zero_centered": True, "dtype": torch.float32}

    # eh_proj is a plain unfused projection over [e; h]
    assert isinstance(layer.eh_proj, nn.Linear)
    assert layer.eh_proj.in_features == 64
    assert layer.eh_proj.out_features == 32
    assert layer.eh_proj.bias is None

    assert layer_env["shared_head"].prefix == "model.layers.4.shared_head"

    # the draft block is built from the MTP vllm config, not the target's
    assert layer_env["decoder"] == (layer_env["mtp_vllm_config"], "model.layers.4.mtp_block")


# ---------------------------------------------------------------------------
# AscendStep5MultiTokenPredictor
# ---------------------------------------------------------------------------


@pytest.fixture
def predictor_env(monkeypatch):
    built: dict[str, Any] = {}

    class _FakeLayer(nn.Module):
        def __init__(self, *, vllm_config, prefix):
            super().__init__()
            self.vllm_config = vllm_config
            self.prefix = prefix

    def _embed(*args, **kwargs):
        built["embed"] = (args, kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(step5_mtp, "_get_mtp_config", lambda vllm_config: _mtp_config())
    monkeypatch.setattr(step5_mtp, "VocabParallelEmbedding", _embed)
    monkeypatch.setattr(step5_mtp, "AscendStep5MultiTokenPredictorLayer", _FakeLayer)
    monkeypatch.setattr(step5_mtp, "LogitsProcessor", lambda vocab_size: SimpleNamespace(vocab_size=vocab_size))
    built["layer_cls"] = _FakeLayer
    return built


def test_predictor_builds_one_layer_per_nextn(predictor_env):
    vllm_config = SimpleNamespace(marker="vc")
    # the "model." prefix normally comes from the MTP wrapper's maybe_prefix
    predictor = AscendStep5MultiTokenPredictor(vllm_config=vllm_config, prefix="model")

    assert predictor.mtp_start_layer_idx == 4
    assert list(predictor.layers.keys()) == ["4", "5"]
    for idx, layer in predictor.layers.items():
        assert isinstance(layer, predictor_env["layer_cls"])
        assert layer.vllm_config is vllm_config
        assert layer.prefix == f"model.layers.{idx}"
    assert predictor_env["embed"][0] == (128, 32)
    assert predictor.logits_processor.vocab_size == 128


def test_predictor_rejects_missing_nextn_layers(predictor_env, monkeypatch):
    monkeypatch.setattr(step5_mtp, "_get_mtp_config", lambda vllm_config: _mtp_config(num_nextn_predict_layers=0))
    with pytest.raises(ValueError, match="num_nextn_predict_layers > 0"):
        AscendStep5MultiTokenPredictor(vllm_config=SimpleNamespace())


# ---------------------------------------------------------------------------
# AscendStep5MTP
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeVllmConfig:
    # a real dataclass so vllm's config replace() works on it
    quant_config: Any = None
    model_config: Any = None
    speculative_config: Any = None


@pytest.fixture
def mtp_env(monkeypatch):
    calls: dict[str, Any] = {}
    head = MagicMock(name="first_draft_head")

    class _FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_head = SimpleNamespace(head=head)

    class _FakePredictor(nn.Module):
        def __init__(self, *, vllm_config, prefix):
            super().__init__()
            calls["predictor"] = (vllm_config, prefix)
            self.mtp_start_layer_idx = 4
            self.layers = nn.ModuleDict({"4": _FakeLayer(), "5": _FakeLayer()})

    def _set_meta(model, example_layer):
        calls["meta"] = (model, example_layer)

    monkeypatch.setattr(step5_mtp, "_get_mtp_config", lambda vllm_config: _mtp_config())
    monkeypatch.setattr(step5_mtp, "AscendStep5MultiTokenPredictor", _FakePredictor)
    monkeypatch.setattr(step5_mtp, "_is_step5_mtp_dense", lambda model: True)
    monkeypatch.setattr(step5_mtp, "_set_step5_moe_protocol_metadata", _set_meta)
    return calls, head


def test_mtp_without_quant_config_keeps_target_vllm_config(mtp_env):
    calls, head = mtp_env
    vllm_config = _FakeVllmConfig(quant_config=None)

    mtp = AscendStep5MTP(vllm_config=vllm_config, prefix="")

    assert mtp.vllm_config is vllm_config
    assert calls["predictor"][0] is vllm_config
    assert calls["predictor"][1] == "model"
    # the exposed lm_head is the first draft layer's checkpoint head
    assert mtp.lm_head is head
    assert mtp.is_dense_mtp is True
    assert mtp.moe_layers == []
    assert calls["meta"][0] is mtp
    assert calls["meta"][1] is None


def test_mtp_maps_quant_exclusions_without_mutating_target(mtp_env):
    calls, _ = mtp_env
    mapper_calls = []

    def _apply(mapper):
        mapper_calls.append(mapper)

    quant_config = SimpleNamespace(apply_vllm_mapper=_apply, marker="original")
    vllm_config = _FakeVllmConfig(quant_config=quant_config)

    mtp = AscendStep5MTP(vllm_config=vllm_config, prefix="")

    # language_model.* quant exclusions were rewritten to the draft's model.*
    # namespace on a deep copy
    assert len(mapper_calls) == 1
    assert mapper_calls[0].orig_to_new_prefix == {"language_model.": ""}

    # the predictor and the wrapper see the mapped copy, not the target config
    assert mtp.vllm_config is not vllm_config
    assert mtp.vllm_config.quant_config is not quant_config
    assert mtp.vllm_config.quant_config.marker == "original"
    assert calls["predictor"][0] is mtp.vllm_config

    # the target config object itself is untouched
    assert vllm_config.quant_config is quant_config


def test_mtp_rejects_non_dense_draft_layers(mtp_env, monkeypatch):
    monkeypatch.setattr(step5_mtp, "_is_step5_mtp_dense", lambda model: False)
    with pytest.raises(NotImplementedError, match="dense draft layers only"):
        AscendStep5MTP(vllm_config=_FakeVllmConfig(), prefix="")
