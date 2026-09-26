# SPDX-License-Identifier: Apache-2.0
"""CPU-only gate contract tests; synthetic examples are not accuracy evidence."""

import copy
import importlib
import shutil
import sys
from types import SimpleNamespace

import pytest

from tools.glm_reduced import run_logits_gate
from tools.glm_reduced.logits_gate import DATA, compare, load_case, validate_model


def sample(count=10):
    record = {"id": "first", "input_sha256": "abc", "argmax": [1] * count}
    request = {"id": "first", "input_sha256": "abc", "answer_token_ids": [2] * count}
    return record, request


@pytest.mark.parametrize(("hits", "passed"), [(10, True), (9, True), (8, False), (0, False)])
def test_inclusive_ninety_percent(hits, passed):
    reference, request = sample()
    candidate = {**reference, "argmax": [1] * hits + [2] * (10 - hits)}
    result = compare([reference], [candidate], [request])
    assert result["passed"] is passed
    assert result["matched"] == hits
    assert result["total"] == 10


def test_position_weighting_and_first_sample():
    first, request = sample(1)
    second = {**first, "id": "second", "argmax": [1] * 99}
    next_request = {**request, "id": "second", "answer_token_ids": [2] * 99}
    result = compare([first, second], [{**first, "argmax": [2]}, second], [request, next_request])
    assert result["agreement"] == 0.99
    assert result["samples"][0]["matched"] == 0
    assert result["total"] == 100


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_sample",
        "missing_position",
        "extra_position",
        "wrong_id",
        "wrong_hash",
        "negative",
        "out_of_vocab",
        "float",
        "bool",
    ],
)
def test_invalid_candidate_fails_closed(mutation):
    reference, request = sample()
    candidate = copy.deepcopy(reference)
    if mutation == "missing_sample":
        rows = []
    else:
        if mutation == "missing_position":
            candidate["argmax"].pop()
        elif mutation == "extra_position":
            candidate["argmax"].append(1)
        elif mutation == "wrong_id":
            candidate["id"] = "other"
        elif mutation == "wrong_hash":
            candidate["input_sha256"] = "changed"
        else:
            candidate["argmax"][0] = {"negative": -1, "out_of_vocab": 154880, "float": 1.0, "bool": True}[mutation]
        rows = [candidate]
    with pytest.raises(ValueError):
        compare([reference], rows, [request])


def test_empty_and_duplicate_samples_rejected():
    reference, request = sample()
    with pytest.raises(ValueError):
        compare([], [], [])
    with pytest.raises(ValueError):
        compare([reference] * 2, [reference] * 2, [request] * 2)


def test_pinned_fixture_complete():
    case, dataset, reference = load_case()
    assert case["model"] == "Eco-Tech/GLM-5.2-w4a8"
    assert dataset["samples"][0]["id"] == "gsm8k-test-0015"
    result = compare(reference["samples"], reference["samples"], dataset["samples"])
    assert result["total"] == 20693
    assert len(result["samples"]) == 200


def test_fixture_tampering_rejected(tmp_path):
    for name in ("case.json", "dataset.json", "A1.json"):
        shutil.copyfile(DATA / name, tmp_path / name)
    with (tmp_path / "A1.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        load_case(tmp_path)


def model_config():
    return {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 11,
        "num_nextn_predict_layers": 1,
        "vocab_size": 154880,
        "hidden_size": 6144,
    }


def test_model_scope():
    validate_model(model_config(), "Eco-Tech/GLM-5.2-w4a8")
    for model in ("GLM-5.3-Flash", "GLM-4.7"):
        with pytest.raises(ValueError):
            validate_model(model_config(), model)
    for override in (
        {"num_hidden_layers": 78},
        {"num_nextn_predict_layers": 0},
        {"layer_types": ["linear_attention"]},
        {"model_type": "glm4_moe"},
    ):
        with pytest.raises(ValueError):
            validate_model({**model_config(), **override}, "GLM-5.2")


def test_yaml_loader_and_nightly_dispatch(monkeypatch, tmp_path):
    # Only the port helper is stubbed; exercise the repository's actual loader.
    monkeypatch.setitem(sys.modules, "vllm.utils.network_utils", SimpleNamespace(get_open_port=lambda: 8000))
    loader = importlib.import_module("tests.e2e.nightly.single_node.models.scripts.single_node_config")
    config = loader.SingleNodeConfigLoader.from_yaml_cases("GLM-5.2-W4A8-A3-Logits11.yaml")[0]
    assert "glm5x_logits_gate" in config.extra_config
    calls = []
    monkeypatch.setattr(run_logits_gate, "provision", lambda options, model: tmp_path / "model")
    monkeypatch.setattr(run_logits_gate.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.chdir(tmp_path)
    run_logits_gate.run_nightly(config)
    assert len(calls) == 1
    assert "tools.glm_reduced.run_logits_gate" in calls[0][0]
    assert calls[0][1]["check"] is True
    assert calls[0][1]["env"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_nightly_failure_is_not_retried(monkeypatch, tmp_path):
    calls = []

    def fail(options, model):
        calls.append(model)
        raise ValueError("unavailable model")

    monkeypatch.setattr(run_logits_gate, "provision", fail)
    monkeypatch.chdir(tmp_path)
    config = SimpleNamespace(name="test", model="Eco-Tech/GLM-5.2-w4a8", extra_config={"glm5x_logits_gate": {}})
    with pytest.raises(ValueError, match="unavailable"):
        run_logits_gate.run_nightly(config)
    assert len(calls) == 1
    assert len(list(tmp_path.glob("benchmark_results/test/*/failure.json"))) == 1
