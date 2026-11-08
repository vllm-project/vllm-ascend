#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""DSpark speculative decoding: Model Runner V2 must match V1 under greedy decoding.

Refs issue #13745.

``VLLM_USE_V2_MODEL_RUNNER`` is read when the vLLM config is built, so the two
runners cannot be exercised from one process. Each arm therefore runs
``dspark_v1_v2_collect.py`` in its own subprocess and the dumps are compared
afterwards.

Default model pair is the single-card DSpark combination already used by
``tests/e2e/pull_request/one_card/spec_decode/test_dspark.py``. Override with
``DSPARK_PARITY_MODEL`` / ``DSPARK_PARITY_DRAFT_MODEL`` (plus
``DSPARK_PARITY_TP`` / ``DSPARK_PARITY_DP`` / ``DSPARK_PARITY_QUANTIZATION``)
to run the same checks against Kimi-K3 on a multi-node topology.

Run::

    pytest tests/e2e/cases/features/spec_decode/test_dspark_v1_v2_parity.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.cases.features.spec_decode.dspark_v1_v2_collect import (
    _extract_remote_token_ids,
    _first_remote_choice,
    _runner_verification,
    _select_prompts,
    _server_speculative_config,
)
from tests.e2e.cases.features.spec_decode.dspark_v1_v2_compare import (
    ACCEPTANCE_TOLERANCE,
    _compare_outputs,
    _degenerate_prompts,
    _markdown,
)
from tests.e2e.cases.features.spec_decode.dspark_v1_v2_perf_summary import _ratio

COLLECT_SCRIPT = Path(__file__).parent / "dspark_v1_v2_collect.py"

MODEL = os.environ.get("DSPARK_PARITY_MODEL", "Qwen/Qwen3-8B")
DRAFT_MODEL = os.environ.get("DSPARK_PARITY_DRAFT_MODEL", "deepseek-ai/dspark_qwen3_8b_block7")
TP = os.environ.get("DSPARK_PARITY_TP", "1")
DP = os.environ.get("DSPARK_PARITY_DP", "1")
QUANTIZATION = os.environ.get("DSPARK_PARITY_QUANTIZATION", "")

# Keep the default run small enough for a CI slot; the full prompt suite and
# longer outputs belong to the manual sweep that produces the task report.
PROMPTS = os.environ.get("DSPARK_PARITY_PROMPTS", "short_en,short_zh,code")
MAX_TOKENS = int(os.environ.get("DSPARK_PARITY_MAX_TOKENS", "128"))
NUM_SPECULATIVE_TOKENS = int(os.environ.get("DSPARK_PARITY_NUM_SPEC_TOKENS", "7"))


def _collect(tmp_path: Path, runner_env: str, temperature: float, draft_sample_method: str) -> Path:
    """Run one arm and return its dump path."""
    out_path = tmp_path / f"runner{runner_env}_t{temperature}_{draft_sample_method}.json"
    cmd = [
        sys.executable,
        str(COLLECT_SCRIPT),
        "--model",
        MODEL,
        "--draft-model",
        DRAFT_MODEL,
        "--num-speculative-tokens",
        str(NUM_SPECULATIVE_TOKENS),
        "--draft-sample-method",
        draft_sample_method,
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(MAX_TOKENS),
        "--prompts",
        PROMPTS,
        "--tensor-parallel-size",
        TP,
        "--data-parallel-size",
        DP,
        "--out",
        str(out_path),
    ]
    if QUANTIZATION:
        cmd += ["--quantization", QUANTIZATION]

    env = dict(os.environ)
    # Set explicitly on both arms: an unset value selects V2, the current
    # default, which would silently compare V2 against itself.
    env["VLLM_USE_V2_MODEL_RUNNER"] = runner_env
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        pytest.fail(
            f"collection failed for VLLM_USE_V2_MODEL_RUNNER={runner_env}\n"
            f"--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}"
        )
    assert out_path.exists(), f"collector exited 0 but wrote no dump: {out_path}"
    return out_path


def test_dspark_v1_v2_greedy_token_parity(tmp_path: Path):
    """Greedy decoding is the hard equivalence check between the two runners.

    With ``temperature=0`` and a fixed seed both runners take the argmax path.
    A mismatch is a regression signal only after both outputs pass the
    degeneration guard; a degenerate V1 output is an invalid baseline.
    """
    v1_path = _collect(tmp_path, "0", 0.0, "greedy")
    v2_path = _collect(tmp_path, "1", 0.0, "greedy")

    with open(v1_path, encoding="utf-8") as handle:
        v1 = json.load(handle)
    with open(v2_path, encoding="utf-8") as handle:
        v2 = json.load(handle)

    v1_degenerate = _degenerate_prompts(v1)
    v2_degenerate = _degenerate_prompts(v2)
    assert not v1_degenerate, f"invalid V1 baseline: degenerate prompts {v1_degenerate}"
    assert not v2_degenerate, f"V2 regression signal: degenerate prompts {v2_degenerate}"

    rows = _compare_outputs(v1, v2)
    mismatched = [
        f"{row['prompt_id']} (first divergence at index {row['first_divergence']}, "
        f"lengths v1={row['len_v1']} v2={row['len_v2']})"
        for row in rows
        if not row["identical"]
    ]
    assert not mismatched, "V2 diverged from V1 under greedy decoding: " + "; ".join(mismatched)


def test_dspark_v1_v2_nongreedy_acceptance_distribution(tmp_path: Path):
    """Non-greedy acceptance rates must stay in the same band across runners.

    Token sequences legitimately differ here (independent randomness), so the
    assertion is on the acceptance-rate distribution. The run also guards
    against a recurrence of #13730, whose signature was every position
    accepting with probability 1.0; that defect was fixed by #13470.
    """
    v1_path = _collect(tmp_path, "0", 1.0, "greedy")
    v2_path = _collect(tmp_path, "1", 1.0, "greedy")

    with open(v1_path, encoding="utf-8") as handle:
        v1 = json.load(handle)
    with open(v2_path, encoding="utf-8") as handle:
        v2 = json.load(handle)

    v1_degenerate = _degenerate_prompts(v1)
    v2_degenerate = _degenerate_prompts(v2)
    assert not v1_degenerate, f"invalid V1 baseline: degenerate prompts {v1_degenerate}"
    assert not v2_degenerate, f"V2 regression signal: degenerate prompts {v2_degenerate}"
    v1_acceptance = v1["acceptance"]
    v2_acceptance = v2["acceptance"]

    if v2_acceptance["num_drafts"] == 0:
        pytest.skip("no drafts were produced; acceptance comparison is not meaningful")

    v2_per_pos = v2_acceptance["acceptance_per_pos"]
    assert not all(rate >= 0.999 for rate in v2_per_pos), (
        "V2 accepted every draft token at every position under temperature>0. "
        "This is the signature of the u=0.0 rejection-sampler defect (#13730, "
        f"fixed by #13470); acceptance_per_pos={v2_per_pos}"
    )

    deviations = [
        f"pos {pos}: v1={a:.4f} v2={b:.4f} (delta {b - a:+.4f})"
        for pos, (a, b) in enumerate(zip(v1_acceptance["acceptance_per_pos"], v2_per_pos))
        if abs(b - a) > ACCEPTANCE_TOLERANCE
    ]
    assert not deviations, f"acceptance rates differ by more than {ACCEPTANCE_TOLERANCE} between runners: " + "; ".join(
        deviations
    )


def _minimal_dump(output_count: int = 1, acceptance_positions: int = 2) -> dict:
    outputs = [
        {
            "prompt_id": f"p{index}",
            "prompt_num_tokens": 1,
            "token_ids": None,
            "text": "same",
            "top_logprobs": [],
        }
        for index in range(output_count)
    ]
    return {
        "runner": "v1",
        "mode": "remote",
        "config": {
            "model": "model",
            "draft_model": "draft",
            "num_speculative_tokens": 2,
            "draft_sample_method": "greedy",
            "enforce_eager_draft": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_tokens": 4,
            "logprobs": 0,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "batch_size": output_count,
            "endpoint": "http://endpoint",
        },
        "acceptance": {
            "num_drafts": 1,
            "num_draft_tokens": acceptance_positions,
            "num_accepted_tokens": 1,
            "acceptance_per_pos": [0.5] * acceptance_positions,
            "mean_acceptance_length": 1.5,
        },
        "timing": {
            "generate_wall_seconds": 1.0,
            "total_output_tokens": 1,
            "output_tokens_per_second": 1.0,
            "remote_mode": True,
        },
        "outputs": outputs,
    }


def test_compare_rejects_mismatched_output_counts():
    left = _minimal_dump(output_count=1)
    right = _minimal_dump(output_count=2)
    with pytest.raises(SystemExit, match="output counts differ"):
        _compare_outputs(left, right)


def test_compare_rejects_mismatched_acceptance_positions():
    left = _minimal_dump(acceptance_positions=2)
    right = _minimal_dump(acceptance_positions=3)
    with pytest.raises(SystemExit, match="acceptance position counts differ"):
        _compare_outputs(left, right)


def test_compare_reports_logprobs_unavailable_without_top_zero_table():
    left = _minimal_dump()
    right = _minimal_dump()
    right["runner"] = "v2"
    rows = _compare_outputs(left, right)
    report, _ = _markdown(left, right, rows)
    assert "UNAVAILABLE" in report
    assert "Mean top-k overlap" not in report
    assert "| Prompt | Compared positions | Mean top-k overlap | Mean KL" not in report


def test_remote_response_token_ids_are_preserved():
    assert _extract_remote_token_ids({"token_ids": [17374, 0, "0"]}) == [17374, 0, 0]
    assert _extract_remote_token_ids({}) is None


def test_remote_response_requires_a_usable_choice():
    with pytest.raises(RuntimeError, match="no usable choice"):
        _first_remote_choice({"choices": []})
    with pytest.raises(RuntimeError, match="no usable choice"):
        _first_remote_choice({"choices": [None]})


def test_repeated_prompts_receive_unique_ids():
    selected = _select_prompts("short_en", 2)
    assert [prompt_id for prompt_id, _ in selected] == ["short_en#1", "short_en#2"]
    with pytest.raises(SystemExit, match="at least 1"):
        _select_prompts("short_en", 0)


def test_compare_allows_different_remote_endpoints():
    left = _minimal_dump()
    right = _minimal_dump()
    right["runner"] = "v2"
    right["config"]["endpoint"] = "http://other-endpoint"
    assert _compare_outputs(left, right)[0]["identical"] is True


def test_compare_rejects_empty_token_ids_with_nonempty_text():
    left = _minimal_dump()
    right = _minimal_dump()
    right["runner"] = "v2"
    left["outputs"][0]["token_ids"] = []
    right["outputs"][0]["token_ids"] = []
    with pytest.raises(SystemExit, match="nonempty text but empty token ids"):
        _compare_outputs(left, right)


def test_perf_threshold_controls_row_marker():
    assert _ratio(100, 92, True, 10) == "0.920 (-8.0% PASS)"
    assert _ratio(100, 92, True, 5) == "0.920 (-8.0% REGRESSION)"


def test_remote_runner_without_evidence_is_explicitly_unverified():
    verification = _runner_verification("http://endpoint", "0", None)
    assert verification == {"status": "unverified_client_claim", "verified": False, "runner": "v1"}


def test_remote_runner_evidence_is_hashed_and_verified(tmp_path: Path):
    evidence = tmp_path / "startup.log"
    evidence.write_text("VLLM_USE_V2_MODEL_RUNNER=True is set; using Model Runner V2\n", encoding="utf-8")
    verification = _runner_verification("http://endpoint", "1", str(evidence))
    assert verification["status"] == "verified_server_startup_log"
    assert verification["verified"] is True
    assert verification["runner"] == "v2"
    assert len(verification["evidence_sha256"]) == 64


def test_remote_runner_evidence_rejects_wrong_runner(tmp_path: Path):
    evidence = tmp_path / "startup.log"
    evidence.write_text("using Model Runner V1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="observed.*V1"):
        _runner_verification("http://endpoint", "1", str(evidence))


def test_remote_runner_evidence_rejects_mixed_runner_logs(tmp_path: Path):
    evidence = tmp_path / "startup.log"
    evidence.write_text("using Model Runner V1\nusing Model Runner V2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="observed.*V1.*V2"):
        _runner_verification("http://endpoint", "1", str(evidence))


_RANK0_STARTUP_LOG = (
    "VLLM_USE_V2_MODEL_RUNNER=False is set; using Model Runner V1.\n"
    "non-default args: {'model': 'm', 'speculative_config': {'method': 'dspark', "
    "'num_speculative_tokens': 7, 'draft_sample_method': 'greedy', 'enforce_eager': True}, 'x': 1}\n"
)


def test_remote_runner_evidence_records_and_checks_server_config(tmp_path: Path):
    evidence = tmp_path / "startup.log"
    evidence.write_text(_RANK0_STARTUP_LOG, encoding="utf-8")
    claimed = {"num_speculative_tokens": 7, "draft_sample_method": "greedy", "enforce_eager": True}
    verification = _runner_verification("http://endpoint", "0", str(evidence), claimed)
    assert verification["server_config"] == claimed


def test_server_config_parser_handles_nested_values():
    log = (
        "non-default args: {'speculative_config': {'num_speculative_tokens': 7, "
        "'draft_sample_method': 'greedy', 'enforce_eager': True, "
        "'extra': {'nested': 1}}, 'other': {'nested': 2}}\n"
    )
    assert _server_speculative_config(log) == {
        "num_speculative_tokens": 7,
        "draft_sample_method": "greedy",
        "enforce_eager": True,
    }


def test_remote_runner_evidence_rejects_client_default_masquerading_as_server_config(tmp_path: Path):
    # A client-side default must never override the configuration recorded by the server.
    evidence = tmp_path / "startup.log"
    evidence.write_text(_RANK0_STARTUP_LOG, encoding="utf-8")
    claimed = {"num_speculative_tokens": 7, "draft_sample_method": "greedy", "enforce_eager": False}
    with pytest.raises(SystemExit, match="enforce_eager"):
        _runner_verification("http://endpoint", "0", str(evidence), claimed)


def test_remote_runner_evidence_without_server_config_line_is_still_verified(tmp_path: Path):
    # A worker (non rank-0) log has the runner marker but no 'non-default args' line.
    evidence = tmp_path / "startup.log"
    evidence.write_text("VLLM_USE_V2_MODEL_RUNNER=False is set; using Model Runner V1.\n", encoding="utf-8")
    claimed = {"num_speculative_tokens": 7, "draft_sample_method": "greedy", "enforce_eager": False}
    verification = _runner_verification("http://endpoint", "0", str(evidence), claimed)
    assert verification["verified"] is True
    assert verification["server_config"] is None


def _dump_with_text(runner: str, text: str) -> dict:
    dump = _minimal_dump()
    dump["runner"] = runner
    dump["outputs"][0]["text"] = text
    return dump


def test_degenerate_detector_thresholds():
    # Regression for the V1 defect: a short correct prefix followed by a run of token 0 ('!').
    assert _degenerate_prompts(_dump_with_text("v1", " Then" + "!" * 255)) == ["p0"]
    assert _degenerate_prompts(_dump_with_text("v1", "a" * 60 + "b" * 40)) == ["p0"]  # exactly 60% one char
    assert _degenerate_prompts(_dump_with_text("v1", "a" * 59 + "b" * 41)) == []  # 59% is not
    assert _degenerate_prompts(_dump_with_text("v1", "!" * 15)) == []  # too short to judge
    assert _degenerate_prompts(_dump_with_text("v1", " " * 200 + "\n" * 50)) == []  # whitespace is never counted
    assert _degenerate_prompts(_dump_with_text("v1", "    " + "x = 1\n    " * 3)) == []  # indentation-heavy code
    assert _degenerate_prompts(_dump_with_text("v2", "Ordinary, varied sentence text goes here.")) == []


def test_compare_marks_degenerate_baseline_invalid_instead_of_regression():
    v1 = _dump_with_text("v1", " Then" + "!" * 255)
    v2 = _dump_with_text("v2", "A perfectly normal and sufficiently long English sentence.")
    report, verdict_ok = _markdown(v1, v2, _compare_outputs(v1, v2))
    assert verdict_ok is False
    assert "Invalid comparison" in report
    assert "accuracy regression signal" not in report


def test_compare_still_reports_regression_for_ordinary_mismatch():
    v1 = _dump_with_text("v1", "A perfectly normal and sufficiently long English sentence.")
    v2 = _dump_with_text("v2", "Another perfectly normal and sufficiently long reply instead.")
    report, verdict_ok = _markdown(v1, v2, _compare_outputs(v1, v2))
    assert verdict_ok is False
    assert "accuracy regression signal" in report
    assert "Invalid comparison" not in report


def test_compare_invalid_even_when_both_sides_degenerate_identically():
    v1 = _dump_with_text("v1", "!" * 64)
    v2 = _dump_with_text("v2", "!" * 64)
    report, verdict_ok = _markdown(v1, v2, _compare_outputs(v1, v2))
    assert verdict_ok is False
    assert "Invalid comparison" in report


def test_degenerate_detector_prefers_token_ids_when_present():
    dump = _dump_with_text("v1", "")
    dump["outputs"][0]["token_ids"] = [17374] + [0] * 39  # renders as nothing/anything, but is one id repeated
    assert _degenerate_prompts(dump) == ["p0"]
    dump["outputs"][0]["token_ids"] = list(range(40))
    assert _degenerate_prompts(dump) == []


def test_compare_degenerate_candidate_next_to_healthy_baseline_is_a_real_regression():
    v1 = _dump_with_text("v1", "A perfectly normal and sufficiently long English sentence.")
    v2 = _dump_with_text("v2", " Then" + "!" * 255)
    report, verdict_ok = _markdown(v1, v2, _compare_outputs(v1, v2))
    assert verdict_ok is False
    assert "Candidate-side V2 degeneration" in report
    assert "Invalid comparison" not in report
    assert "accuracy regression signal" in report
