# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.e2e.models import glm52_dcp_probe as probe


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return [13] * len(text)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False
        return [1] + self.encode(messages[0]["content"]) + [2] * 128


class MergingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False
        return [1] + [13] * (len(messages[0]["content"]) // 8) + [2] * 128


def test_build_cases_accounts_for_cross_boundary_token_merging(monkeypatch):
    monkeypatch.setattr(probe.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: MergingTokenizer())
    cases = probe.build_cases("unused", [4096])
    for name, case in cases.items():
        if not name.endswith("short"):
            assert len(case["prompt"]) == 4096
            assert case["prompt"][-128:] == [2] * 128


@pytest.mark.parametrize("broken_output", [{"input_ids": [1, 2]}, [1, 2]])
def test_build_cases_rejects_invalid_or_nonexpanding_tokenization(monkeypatch, broken_output):
    tokenizer = FakeTokenizer()
    tokenizer.apply_chat_template = lambda *args, **kwargs: broken_output
    monkeypatch.setattr(probe.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    with pytest.raises((AssertionError, ValueError)):
        probe.build_cases("unused", [256])


def test_build_cases_preserves_lengths_and_generation_suffix(monkeypatch):
    monkeypatch.setattr(probe.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    cases = probe.build_cases("unused", [256, 512])
    assert len(cases) == 9
    for name, case in cases.items():
        if not name.endswith("short"):
            assert case["input_tokens"] == int(name.rsplit("-", 1)[1])
            assert case["prompt"][0] == 1
            assert case["prompt"][-128:] == [2] * 128
        assert len(case["prompt_sha256"]) == 64


def test_requests_pin_the_selected_data_parallel_replica(monkeypatch, tmp_path):
    session = MagicMock()
    session.__enter__.return_value = session
    session.headers = {}
    session.post.return_value.json.return_value = {"choices": [{"logprobs": {"tokens": ["token_id:1"]}}]}
    monkeypatch.setattr(probe.requests, "Session", lambda: session)
    case = {"prompt": [1], "input_tokens": 1, "prompt_sha256": "same"}
    args = SimpleNamespace(request_dp_rank=1, port=8000, output=tmp_path / "report.json")
    probe.probe_requests({"english-short": case}, args, {"results": {}})
    assert session.headers == {"X-data-parallel-rank": "1"}
    assert session.post.call_count == 2


def _report():
    result = {
        "prompt_sha256": "same-prompt",
        "response": {
            "choices": [
                {
                    "logprobs": {
                        "tokens": ["token_id:1"],
                        "top_logprobs": [{"token_id:1": -0.5, "token_id:2": -0.6}],
                    }
                }
            ]
        },
    }
    return {
        "model": "same-model",
        "tp": 8,
        "dp": 2,
        "pp": 1,
        "pp_layer_partition": None,
        "pcp": 1,
        "dcp": 1,
        "mtp": False,
        "graph": False,
        "kv_cache_bytes": 4 * 1024**3,
        "request_dp_rank": 0,
        "kv_cache_dtype": "int8",
        "indexer_kv_dtype": "int8",
        "disable_mlapo": False,
        "runtime_env": {key: None for key in probe.RUNTIME_ENV_KEYS},
        "passed": True,
        "results": {"english-greedy": copy.deepcopy(result), "english-sampled": copy.deepcopy(result)},
    }


@pytest.mark.parametrize("sampling,expected", [("greedy", False), ("sampled", True)])
def test_comparison_keeps_greedy_strict_and_sampling_topk(tmp_path, sampling, expected):
    baseline = _report()
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))
    actual = copy.deepcopy(baseline)
    actual["dcp"] = 8
    actual["results"][f"english-{sampling}"]["response"]["choices"][0]["logprobs"]["tokens"] = ["token_id:2"]
    assert probe.compare_baseline(actual, path)[f"english-{sampling}"]["passed"] is expected


@pytest.mark.parametrize(
    "field",
    [
        "tp",
        "mtp",
        "graph",
        "kv_cache_bytes",
        "request_dp_rank",
        "pp_layer_partition",
        "kv_cache_dtype",
        "indexer_kv_dtype",
        "disable_mlapo",
        "runtime_env",
        "prompt",
        "case-set",
    ],
)
def test_comparison_rejects_incomparable_runs(tmp_path, field):
    baseline = _report()
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))
    actual = copy.deepcopy(baseline)
    if field == "tp":
        actual["tp"] = 4
    elif field in ("mtp", "graph", "disable_mlapo"):
        actual[field] = True
    elif field == "kv_cache_bytes":
        actual[field] //= 2
    elif field == "request_dp_rank":
        actual[field] = 1
    elif field == "pp_layer_partition":
        actual[field] = "38,40"
    elif field in ("kv_cache_dtype", "indexer_kv_dtype"):
        actual[field] = "bf16" if field == "indexer_kv_dtype" else "auto"
    elif field == "runtime_env":
        actual[field]["LCCL_DETERMINISTIC"] = "1"
    elif field == "prompt":
        actual["results"]["english-greedy"]["prompt_sha256"] = "different"
    else:
        actual["results"].pop("english-greedy")
    with pytest.raises(AssertionError):
        probe.compare_baseline(actual, path)


def test_comparison_rejects_legacy_unpinned_baseline(tmp_path):
    actual = _report()
    baseline = copy.deepcopy(actual)
    del baseline["request_dp_rank"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(baseline))
    with pytest.raises(AssertionError, match="rerun with the same probe protocol"):
        probe.compare_baseline(actual, path)


@pytest.mark.parametrize("cache_dtype,indexer_dtype,disable_mlapo", [("int8", "int8", False), ("auto", "bf16", True)])
def test_serve_command_preserves_cache_and_preprocess_controls(cache_dtype, indexer_dtype, disable_mlapo):
    args = SimpleNamespace(
        model="checkpoint",
        port=8000,
        tp=8,
        dp=2,
        pp=1,
        pcp=1,
        dcp=8,
        lengths=[4096],
        kv_cache_bytes=4 * 1024**3,
        kv_cache_dtype=cache_dtype,
        indexer_kv_dtype=indexer_dtype,
        disable_mlapo=disable_mlapo,
        graph=False,
        mtp=False,
    )
    command = probe.command_for(args)
    assert command[command.index("--kv-cache-dtype") + 1] == cache_dtype
    assert command[command.index("--attention_config.indexer_kv_dtype") + 1] == indexer_dtype
    additional = json.loads(command[command.index("--additional-config") + 1])
    assert additional == {"enable_fused_mc2": 0, "enable_mlapo": not disable_mlapo}
