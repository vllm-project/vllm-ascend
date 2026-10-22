# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import subprocess
from pathlib import Path

import pytest

from tests.e2e.glm53_flash.logprobs import (
    LOGPROB_ATOL,
    OUTPUT_TOKENS,
    compare_completion,
    completion_request,
    curl_completion,
    normalize_completion,
    validate_golden,
)


@pytest.fixture
def response():
    candidates = {f"token_id:{index}": -float(index) for index in range(1, 6)}
    return {
        "model": "glm53-flash-ci",
        "usage": {"prompt_tokens": 2, "completion_tokens": OUTPUT_TOKENS},
        "choices": [
            {
                "finish_reason": "length",
                "token_ids": [1] * OUTPUT_TOKENS,
                "logprobs": {
                    "tokens": ["token_id:1"] * OUTPUT_TOKENS,
                    "token_logprobs": [-1.0] * OUTPUT_TOKENS,
                    "top_logprobs": [dict(candidates) for _ in range(OUTPUT_TOKENS)],
                },
            }
        ],
    }


def test_identical_and_candidate_order(response):
    expected = normalize_completion(response)
    for candidates in response["choices"][0]["logprobs"]["top_logprobs"]:
        items = list(candidates.items())[::-1]
        candidates.clear()
        candidates.update(items)
    assert compare_completion(expected, normalize_completion(response), "same") == 0


def test_absolute_tolerance_only(response):
    expected = normalize_completion(response)
    actual = copy.deepcopy(expected)
    actual["steps"][2]["top_logprobs"]["2"] += LOGPROB_ATOL / 2
    assert compare_completion(expected, actual, "close") < LOGPROB_ATOL
    actual["steps"][2]["top_logprobs"]["2"] += LOGPROB_ATOL
    with pytest.raises(AssertionError, match="decode step=2.*rtol=0"):
        compare_completion(expected, actual, "changed")
    expected["steps"][0]["token_logprob"] = -9999
    actual = copy.deepcopy(expected)
    actual["steps"][0]["token_logprob"] += 0.01
    with pytest.raises(AssertionError, match="prefill step=0"):
        compare_completion(expected, actual, "no-relative-tolerance")


def test_first_token_divergence(response):
    expected = normalize_completion(response)
    actual = copy.deepcopy(expected)
    actual["token_ids"][3] = 2
    actual["steps"][4]["token_logprob"] = 42
    with pytest.raises(AssertionError, match="step=3: token mismatch"):
        compare_completion(expected, actual, "divergence")


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf"), True])
def test_nonfinite_response_fails(response, bad):
    response["choices"][0]["logprobs"]["token_logprobs"][0] = bad
    with pytest.raises(AssertionError, match="non-finite"):
        normalize_completion(response)


def test_candidate_set_difference_fails(response):
    expected = normalize_completion(response)
    actual = copy.deepcopy(expected)
    actual["steps"][0]["top_logprobs"]["6"] = actual["steps"][0]["top_logprobs"].pop("5")
    with pytest.raises(AssertionError, match="candidate token-ID set mismatch"):
        compare_completion(expected, actual, "changed")


@pytest.mark.parametrize("field", ["token_ids", "logprobs"])
def test_missing_response_fields(response, field):
    del response["choices"][0][field]
    with pytest.raises(KeyError):
        normalize_completion(response)


def test_response_token_alias_collision_rejected(response):
    response["choices"][0]["logprobs"]["top_logprobs"][0]["中文"] = -2.0
    with pytest.raises(AssertionError, match="token ID"):
        normalize_completion(response)


def test_wrong_finish_or_length(response):
    response["choices"][0]["finish_reason"] = "stop"
    with pytest.raises(AssertionError, match="finish reason"):
        normalize_completion(response)
    response["choices"][0]["finish_reason"] = "length"
    response["choices"][0]["token_ids"].pop()
    with pytest.raises(AssertionError, match="output token IDs"):
        normalize_completion(response)


def test_noncanonical_candidate_id_rejected(response):
    candidates = response["choices"][0]["logprobs"]["top_logprobs"][0]
    candidates["token_id:01"] = candidates.pop("token_id:5")
    with pytest.raises(AssertionError, match="token ID"):
        normalize_completion(response)


def test_reviewed_prompt_fixture():
    fixtures = Path(__file__).resolve().parents[2] / "e2e/glm53_flash/fixtures"
    prompts = json.loads((fixtures / "prompts.json").read_text(encoding="utf-8"))
    source = json.loads((fixtures / "source.json").read_text(encoding="utf-8"))
    assert prompts["tokenizer_sha256"] == source["files"]["tokenizer.json"]["sha256"]
    cases = prompts["cases"]
    assert len(cases) == len({case["name"] for case in cases}) == 11
    assert [len(case["token_ids"]) for case in cases[3:]] == [127, 128, 129, 2047, 2048, 2051, 2052, 4097]
    for case in cases:
        assert all(type(token) is int and 0 <= token < 154880 for token in case["token_ids"])


@pytest.mark.parametrize(
    "mutation", ["contract", "cases", "cold_starts", "rounds", "requests", "failed", "nonfinite", "workers"]
)
def test_invalid_golden_fails(response, mutation):
    contract = {"source_id": "fixed-test-source"}
    golden = {
        "schema_version": 1,
        "contract": dict(contract),
        "cases": {"case": normalize_completion(response)},
        "validation": [
            {"max_abs_error": 0.0, "errors": [], "completed_rounds": 2, "completed_requests": 2} for _ in range(3)
        ],
        "workers": [{"rank": 0}, {"rank": 1}],
    }
    validate_golden(golden, contract, {"case"})
    if mutation == "contract":
        golden["contract"]["source_id"] = "other"
    elif mutation == "cases":
        golden["cases"].clear()
    elif mutation == "cold_starts":
        golden["validation"].pop()
    elif mutation == "rounds":
        golden["validation"][0]["completed_rounds"] = 1
    elif mutation == "requests":
        golden["validation"][0]["completed_requests"] = 1
    elif mutation == "failed":
        golden["validation"][0]["fatal_error"] = "server failed"
    elif mutation == "nonfinite":
        golden["cases"]["case"]["steps"][0]["token_logprob"] = float("nan")
    else:
        golden["workers"].pop()
    with pytest.raises(AssertionError):
        validate_golden(golden, contract, {"case"})


def test_curl_invocation_and_artifacts(response, tmp_path, monkeypatch):
    def run(command, **kwargs):
        assert command[0] == "curl" and "--fail-with-body" in command
        assert command[command.index("--noproxy") + 1] == "*"
        assert not kwargs.get("shell")
        request = Path(command[command.index("--data-binary") + 1][1:])
        assert json.loads(request.read_text()) == completion_request([17, 23])
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(response))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    result = curl_completion("http://127.0.0.1:1234/v1/completions", [17, 23], tmp_path / "request")
    assert result == normalize_completion(response)
    assert (tmp_path / "request/curl.stderr").is_file()


@pytest.mark.parametrize("failure", ["http", "timeout", "invalid-json"])
def test_curl_failure_is_not_skipped(tmp_path, monkeypatch, failure):
    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 180)
        Path(command[command.index("--output") + 1]).write_text("not JSON")
        return subprocess.CompletedProcess(command, 22 if failure == "http" else 0, "", "HTTP failure")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(AssertionError):
        curl_completion("http://127.0.0.1:1234/v1/completions", [17], tmp_path / "request")
    assert (tmp_path / "request/curl.stderr").is_file()
