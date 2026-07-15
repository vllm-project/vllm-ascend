from __future__ import annotations

import importlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
assemble_coverage = importlib.import_module("assemble_coverage")


TEST_GROUPS = [
    {
        "npu_type": "a2",
        "tests": "tests/e2e/test_a.py tests/e2e/test_b.py::test_one",
    },
    {
        "npu_type": "cpu",
        "tests": "tests/ut/test_c.py tests/ut/test_d.py",
    },
]


def _write_coverage(root: Path, key: str, content: str = "coverage") -> Path:
    path = root / key / "covdata" / "coverage.data"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    return path


def _mark_failed(root: Path, key: str) -> Path:
    path = root / key / "covdata" / assemble_coverage.FAILED_SENTINEL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1", encoding="utf-8")
    return path


def test_expected_coverage_keys_match_runner_output_names():
    assert assemble_coverage.coverage_key("tests/e2e/test_a.py") == "tests__e2e__test_a"
    assert assemble_coverage.coverage_key("tests/e2e/test_b.py::test_one") == "tests__e2e__test_b.py--test_one"
    assert assemble_coverage.expected_coverage_keys(TEST_GROUPS) == {
        "tests__e2e__test_a",
        "tests__e2e__test_b.py--test_one",
        "cpu-ut",
    }


def test_missing_coverage_keys_support_nested_artifact_directories(tmp_path):
    _write_coverage(tmp_path / "artifact-a", "tests__e2e__test_a")
    _write_coverage(tmp_path / "artifact-cpu", "cpu-ut")

    assert assemble_coverage.missing_coverage_keys(TEST_GROUPS, tmp_path) == [
        "tests__e2e__test_b.py--test_one"
    ]


def test_backfill_copies_only_missing_coverage(tmp_path):
    current = tmp_path / "current"
    previous = tmp_path / "previous" / "vllm-ascend" / "VLLM-ASCEND@task_2026071401"
    current_file = _write_coverage(current, "tests__e2e__test_a", "current")
    _write_coverage(previous, "tests__e2e__test_a", "old")
    _write_coverage(previous, "tests__e2e__test_b.py--test_one", "backfilled")
    _write_coverage(previous, "cpu-ut", "cpu")

    unresolved = assemble_coverage.backfill_missing_coverage(TEST_GROUPS, current, previous.parent.parent)

    assert unresolved == []
    assert current_file.read_text(encoding="utf-8") == "current"
    assert (
        current / "tests__e2e__test_b.py--test_one" / "covdata" / "coverage.data"
    ).read_text(encoding="utf-8") == "backfilled"
    assert (current / "cpu-ut" / "covdata" / "coverage.data").is_file()


def test_backfill_reports_tests_missing_from_both_runs(tmp_path):
    _write_coverage(tmp_path / "current", "tests__e2e__test_a")

    assert assemble_coverage.backfill_missing_coverage(
        TEST_GROUPS,
        tmp_path / "current",
        tmp_path / "previous",
    ) == ["cpu-ut", "tests__e2e__test_b.py--test_one"]


def test_backfill_command_warns_without_failing_for_unresolved_tests(tmp_path, monkeypatch, capsys):
    missing_file = tmp_path / "missing.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assemble_coverage.py",
            "backfill",
            "--test-groups-json",
            '[{"npu_type":"a2","tests":"tests/e2e/test_a.py"}]',
            "--current-root",
            str(tmp_path / "current"),
            "--previous-root",
            str(tmp_path / "previous"),
            "--missing-file",
            str(missing_file),
        ],
    )

    assemble_coverage.main()

    output = capsys.readouterr().out
    assert "::warning::Coverage data is still missing for 1 test target(s)" in output
    assert missing_file.read_text(encoding="utf-8") == "tests__e2e__test_a\n"


def test_failed_sentinel_marks_target_as_missing(tmp_path):
    """A target whose covdata carries a FAILED sentinel counts as missing."""
    current = tmp_path / "current"
    _write_coverage(current, "tests__e2e__test_a", "partial")
    _mark_failed(current, "tests__e2e__test_a")
    _write_coverage(current, "cpu-ut")

    assert assemble_coverage.missing_coverage_keys(TEST_GROUPS, current) == [
        "tests__e2e__test_a",
        "tests__e2e__test_b.py--test_one",
    ]


def test_backfill_replaces_failed_coverage_with_history(tmp_path):
    """Backfill discards the failed run's partial covdata (and sentinel) and
    copies the historical coverage in its place."""
    current = tmp_path / "current"
    previous = tmp_path / "previous" / "vllm-ascend" / "VLLM-ASCEND@task_2026071401"

    failed_file = _write_coverage(current, "tests__e2e__test_a", "partial-failed")
    _mark_failed(current, "tests__e2e__test_a")
    _write_coverage(previous, "tests__e2e__test_a", "historical")

    unresolved = assemble_coverage.backfill_missing_coverage(TEST_GROUPS, current, previous.parent.parent)

    assert unresolved == []
    covdata = current / "tests__e2e__test_a" / "covdata"
    assert not (covdata / assemble_coverage.FAILED_SENTINEL).exists()
    assert not (covdata / "coverage.data").read_text(encoding="utf-8") == "partial-failed"
    assert (covdata / "coverage.data").read_text(encoding="utf-8") == "historical"
    # The failed partial file was wiped, not renamed alongside the historical one.
    assert failed_file.read_text(encoding="utf-8") == "historical"
