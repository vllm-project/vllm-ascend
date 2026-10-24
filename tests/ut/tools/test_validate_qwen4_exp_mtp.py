# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from tools import validate_qwen4_exp_mtp as validator
from tools.validate_qwen4_exp_mtp import (
    build_engine_args,
    compare_results,
    counter_delta,
    coverage_estimate,
    first_difference,
    read_counters,
)


@pytest.fixture
def config():
    return {
        "model": "qwen-model",
        "seed": 42,
        "batch_sizes": [1, 8, 3, 8],
        "num_speculative_tokens": 3,
        "use_local_argmax_reduction": True,
        "engine_args": {
            "tensor_parallel_size": 8,
            "compilation_config": {"cudagraph_capture_sizes": [4, 8, 32]},
        },
    }


def test_modes_isolate_draft_graph_and_preserve_original_config(config):
    original = copy.deepcopy(config)
    base = build_engine_args(config, "base")
    eager = build_engine_args(config, "draft_eager")
    graph = build_engine_args(config, "draft_graph")
    assert "speculative_config" not in base
    assert eager["speculative_config"]["enforce_eager"] is True
    assert graph["speculative_config"]["enforce_eager"] is False
    assert graph["speculative_config"]["use_local_argmax_reduction"] is True
    assert eager["compilation_config"] == base["compilation_config"] == graph["compilation_config"]
    assert config == original


def test_counter_deltas_exclude_warmup():
    def snapshot(drafts, accepted):
        return read_counters(
            [
                SimpleNamespace(name="vllm:spec_decode_num_drafts", value=drafts),
                SimpleNamespace(name="vllm:spec_decode_num_accepted_tokens", value=accepted),
                SimpleNamespace(name="vllm:spec_decode_num_draft_tokens", value=drafts * 3),
                SimpleNamespace(
                    name="vllm:spec_decode_num_accepted_tokens_per_pos", value=accepted, labels={"position": 0}
                ),
            ]
        )

    delta = counter_delta(snapshot(100, 200), snapshot(120, 230))
    assert delta["drafts"] == 20
    assert delta["draft_tokens"] == 60
    assert delta["accepted_tokens"] == 30
    assert delta["accepted_per_position"] == {"0": 30}
    assert delta["mean_acceptance_length"] == 2.5


def test_missing_metrics_remain_unknown():
    delta = counter_delta(read_counters([]), read_counters([]))
    assert delta["drafts"] is None
    assert delta["accepted_tokens"] is None
    assert delta["mean_acceptance_length"] is None


def test_coverage_uses_verify_tokens_and_excludes_oversized_rounding(config):
    kwargs = build_engine_args(config, "draft_graph")
    assert coverage_estimate(kwargs, 8, 3)["target_capture_tokens"] == 32
    assert coverage_estimate(kwargs, 32, 3)["target_capture_tokens"] is None
    kwargs["compilation_config"] = {"cudagraph_capture_sizes": [31], "max_cudagraph_capture_size": 31}
    assert coverage_estimate(kwargs, 7, 3)["target_capture_tokens"] is None
    kwargs["compilation_config"] = {"cudagraph_capture_sizes": [64]}
    assert coverage_estimate(kwargs, 8, 3)["target_capture_tokens"] is None


def test_compare_reports_drift_and_aggregates_mal_by_draft_count():
    def run(index, ids, drafts, accepted):
        return {
            "run_index": index,
            "batch_size": 1,
            "requests": [{"max_tokens": 3, "output_token_ids": ids}],
            "speculative_counters": {"drafts": drafts, "accepted_tokens": accepted},
        }

    results = {
        "base": {"runs": [run(0, [1, 2, 3], None, None), run(1, [1, 2, 3], None, None)]},
        "draft_eager": {"runs": [run(0, [1, 2, 3], 10, 20), run(1, [1, 2, 3], 30, 60)]},
        "draft_graph": {"runs": [run(0, [1, 2, 3], 10, 10), run(1, [1, 4, 3], 30, 30)]},
    }
    summary = compare_results(results)
    assert summary["acceptance"]["draft_eager"]["mean_acceptance_length"] == 3
    assert summary["acceptance"]["draft_graph"]["mean_acceptance_length"] == 2
    assert summary["graph_to_eager_acceptance_ratio"] == pytest.approx(2 / 3)
    assert summary["token_mismatches_against_base"] == [
        {"mode": "draft_graph", "run_index": 1, "request_index": 0, "first_different_token": 1}
    ]
    assert summary["repeated_run_token_mismatches"] == summary["token_mismatches_against_base"]


@pytest.mark.parametrize("left,right,expected", [([1], [1], None), ([1], [1, 2], 1), ([1, 2], [1, 3], 1)])
def test_first_difference(left, right, expected):
    assert first_difference(left, right) == expected


def test_dry_run_writes_modes_without_spawning(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["validate", "--model", "model", "--output-dir", str(tmp_path), "--dry-run"])
    monkeypatch.setattr(validator.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run spawned a process"))
    assert validator.main() == 0
    modes = json.loads((tmp_path / "engine_args_by_mode.json").read_text(encoding="utf-8"))
    assert "speculative_config" not in modes["base"]
    assert modes["draft_eager"]["speculative_config"]["enforce_eager"] is True
    assert modes["draft_graph"]["speculative_config"]["enforce_eager"] is False


def test_rejects_v1_environment_before_loading_a_model(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["validate", "--model", "model", "--dry-run"])
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    with pytest.raises(SystemExit, match="must be 1"):
        validator.main()


def test_check_fails_on_low_acceptance_despite_equal_outputs(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        mode = command[command.index("--worker-mode") + 1]
        accepted = 10 if mode == "draft_graph" else 20
        report = {
            "runs": [
                {
                    "run_index": 0,
                    "batch_size": 1,
                    "requests": [{"max_tokens": 8, "output_token_ids": [1, 2, 3]}],
                    "speculative_counters": {"drafts": 10, "accepted_tokens": accepted},
                }
            ]
        }
        (tmp_path / f"{mode}.json").write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["validate", "--model", "model", "--output-dir", str(tmp_path), "--check"])
    monkeypatch.setattr(validator.subprocess, "run", fake_run)
    assert validator.main() == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["token_mismatches_against_base"] == []
    assert summary["acceptance_ratio_passed"] is False
