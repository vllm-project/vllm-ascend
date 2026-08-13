from __future__ import annotations

import importlib
import json
import textwrap
from pathlib import Path

update_estimated_times = importlib.import_module("update_estimated_times")


CONFIG_TEMPLATE = textwrap.dedent(
    """\
    modules:
      - name: default_cpu_ut
        optional: false

    ---
    # === Estimated Times ===
    estimated_times:
      tests/e2e/one_card/test_whole_file.py: 1110
      tests/e2e/two_card/test_other.py: 300

    partition:
      a2_x1: 2
    """
)


def _write_timings(tmp_path: Path, records: list[dict]) -> Path:
    timing_dir = tmp_path / "timing"
    timing_dir.mkdir()
    (timing_dir / "timing.json").write_text(json.dumps({"tests": records}))
    return timing_dir


def _run(tmp_path: Path, records: list[dict]) -> dict[str, int]:
    config = tmp_path / "test_config.yaml"
    config.write_text(CONFIG_TEMPLATE)
    timings = update_estimated_times.collect_timings(_write_timings(tmp_path, records))
    update_estimated_times.update_config(config, timings)

    import yaml

    docs = list(yaml.safe_load_all(config.read_text()))
    for doc in docs:
        if isinstance(doc, dict) and "estimated_times" in doc:
            return doc["estimated_times"]
    raise AssertionError("estimated_times section missing after update")


def test_nodeid_timing_does_not_clobber_file_level_estimate(tmp_path):
    # A single method timed at 100s must not replace the 1110s whole-file
    # estimate used when other modules run the file in full.
    result = _run(
        tmp_path,
        [{"name": "tests/e2e/one_card/test_whole_file.py::test_small_case", "passed": True, "elapsed": 100}],
    )

    assert result["tests/e2e/one_card/test_whole_file.py"] == 1110
    assert result["tests/e2e/one_card/test_whole_file.py::test_small_case"] == 110


def test_sibling_nodeids_keep_independent_estimates(tmp_path):
    result = _run(
        tmp_path,
        [
            {"name": "tests/e2e/one_card/test_multi.py::test_fast", "passed": True, "elapsed": 30},
            {"name": "tests/e2e/one_card/test_multi.py::test_slow", "passed": True, "elapsed": 600},
        ],
    )

    assert result["tests/e2e/one_card/test_multi.py::test_fast"] == 30
    assert result["tests/e2e/one_card/test_multi.py::test_slow"] == 660


def test_file_level_timing_still_updates_file_key(tmp_path):
    result = _run(
        tmp_path,
        [{"name": "tests/e2e/two_card/test_other.py", "passed": True, "elapsed": 500}],
    )

    assert result["tests/e2e/two_card/test_other.py"] == 550


def test_non_test_entries_are_skipped(tmp_path):
    result = _run(
        tmp_path,
        [{"name": "cpu-ut (115 targets)", "passed": True, "elapsed": 900}],
    )

    assert "cpu-ut (115 targets)" not in result
