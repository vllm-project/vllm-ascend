# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import copy
import json
from typing import Any

import pytest

from tests.e2e.runner_parity.compare import (
    compare_precision,
    compare_provenance,
    load_upstream_rule,
    normalize,
    read_records,
    token_id,
)


@pytest.fixture
def record():
    return {
        "key": "zh-128-0/greedy/c1",
        "response": {
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            "choices": [
                {
                    "text": "ab",
                    "token_ids": [1, 2],
                    "logprobs": {
                        "top_logprobs": [{"token_id:1": -0.1, "token_id:9": -2}, {"token_id:2": -0.2}],
                    },
                }
            ],
        },
    }


def test_token_ids_cannot_be_inferred_from_ambiguous_text():
    assert token_id("token_id:12") == 12
    with pytest.raises(AssertionError):
        token_id(" twelve")


@pytest.mark.parametrize("field", ["token_ids", "logprobs"])
def test_missing_precision_evidence_is_rejected(record, field):
    record["response"]["choices"][0].pop(field)
    with pytest.raises(AssertionError):
        normalize(record)


def test_shortened_logprob_sequence_is_rejected(record):
    record["response"]["choices"][0]["logprobs"]["top_logprobs"].pop()
    with pytest.raises(AssertionError):
        normalize(record)


def test_non_finite_logprob_cannot_pass_top_k_membership(record):
    record["response"]["choices"][0]["logprobs"]["top_logprobs"][0]["token_id:1"] = float("nan")
    with pytest.raises(AssertionError, match="Non-finite"):
        normalize(record)


def test_duplicate_cases_are_rejected(tmp_path, record):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(AssertionError, match="Duplicate"):
        read_records(path)


def test_fixed_history_uses_completion_choice_schema(record):
    record["request"] = {"prompt_logprobs": 5}
    record["response"]["choices"][0]["prompt_logprobs"] = [None]
    assert normalize(record)[3] == [None]
    record["response"]["choices"][0].pop("prompt_logprobs")
    with pytest.raises(AssertionError, match="missing from the choice"):
        normalize(record)


def test_pinned_rule_rejects_incompatible_distribution(record, upstream_utils):
    rule, _ = load_upstream_rule(upstream_utils)
    baseline = normalize(record)
    candidate = copy.deepcopy(record)
    candidate["response"]["choices"][0]["token_ids"] = [3, 4]
    rule(outputs_0_lst=[baseline], outputs_1_lst=[baseline], name_0="V1", name_1="V1", always_check_logprobs=True)
    with pytest.raises(AssertionError):
        rule(
            outputs_0_lst=[baseline],
            outputs_1_lst=[normalize(candidate)],
            name_0="V1",
            name_1="V2",
            always_check_logprobs=True,
        )


def test_different_runtime_configuration_is_rejected(tmp_path):
    for arm, selector in (("v1", "0"), ("v2", "1")):
        root = tmp_path / arm
        root.mkdir()
        record: dict[str, Any] = {
            "code_hashes": {"runner.py": "same"},
            "ascend_head": "same",
            "vllm_head": "same",
            "container": {
                "image": "same",
                "mounts": [],
                "command": [
                    "docker",
                    "run",
                    "--name",
                    arm,
                    "-e",
                    "VLLM_USE_V2_MODEL_RUNNER=" + selector,
                    "image",
                    "serve",
                    "--max-model-len",
                    "40960",
                ],
            },
        }
        (root / "provenance-round0.json").write_text(json.dumps(record))
    assert compare_provenance(tmp_path / "v1", tmp_path / "v2")["checked_arm_rounds"] == 2
    path = tmp_path / "v2/provenance-round0.json"
    record["container"]["command"][-1] = "32768"
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="configuration differs"):
        compare_provenance(tmp_path / "v1", tmp_path / "v2")


def test_later_divergence_is_not_hidden_by_matching_first_token(tmp_path, record, upstream_utils):
    record.update(input_sha256="same-input", request={"temperature": 0})
    candidate = copy.deepcopy(record)
    choice = candidate["response"]["choices"][0]
    choice["token_ids"] = [1, 3]
    choice["text"] = "ac"
    choice["logprobs"]["top_logprobs"][1] = {"token_id:3": -0.1}
    paths = [tmp_path / "v1.jsonl", tmp_path / "v2.jsonl"]
    for path, item in zip(paths, [record, candidate]):
        path.write_text(json.dumps(item) + "\n")
    report = compare_precision(paths[0], paths[1], upstream_utils)
    assert report["failed"] == 1, "A matching first token cannot bypass incompatible later distributions"
