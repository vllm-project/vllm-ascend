from pathlib import Path

import pytest

from tools.bisect.verdict import RunOutcome, evaluate


def test_evaluate_zero_exit_checks_benchmark_json(tmp_path: Path):
    (tmp_path / "case_ok.json").write_text('{"pass_fail": "pass"}', encoding="utf-8")
    (tmp_path / "case_failed.json").write_text('{"pass_fail": "fail"}', encoding="utf-8")

    verdict, note = evaluate(RunOutcome(exit_code=0, results_dir=tmp_path))

    assert verdict == "FAIL"
    assert "benchmark pass_fail=fail" in note


def test_evaluate_zero_exit_without_benchmark_failures_is_pass(tmp_path: Path):
    (tmp_path / "broken.json").write_text("{not-json", encoding="utf-8")

    verdict, note = evaluate(RunOutcome(exit_code=0, results_dir=tmp_path))

    assert verdict == "PASS"
    assert note == "pytest ok (no benchmark regression)"


@pytest.mark.parametrize(
    ("exit_code", "expected_note"),
    [
        (2, "could not collect/run"),
        (3, "could not collect/run"),
        (4, "could not collect/run"),
        (5, "could not collect/run"),
        (124, "timed out"),
    ],
)
def test_evaluate_infra_exit_codes_are_skip(exit_code: int, expected_note: str):
    verdict, note = evaluate(RunOutcome(exit_code=exit_code))

    assert verdict == "SKIP"
    assert expected_note in note


def test_evaluate_nonzero_runtime_failure_is_fail():
    verdict, note = evaluate(RunOutcome(exit_code=1))

    assert verdict == "FAIL"
    assert note == "pytest exited non-zero (rc=1)"


def test_evaluate_explicit_infra_error_uses_skip_reason():
    verdict, note = evaluate(RunOutcome(exit_code=0, infra_error=True, skip_reason="vllm mismatch"))

    assert verdict == "SKIP"
    assert note == "vllm mismatch"
