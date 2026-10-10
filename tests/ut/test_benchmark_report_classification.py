"""Regression tests for benchmark report classification."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "scripts" / "convert_json_to_markdown.py"


def test_classification_uses_filename_not_parent_directory(tmp_path):
    cases = {
        "latency_fixture.json": {
            "avg_latency": 0.01,
            "percentiles": {str(p): 0.02 for p in (10, 25, 50, 75, 90, 99)},
        },
        "throughput_fixture.json": {
            "num_requests": 3,
            "total_num_tokens": 9,
            "elapsed_time": 1.5,
            "requests_per_second": 2.0,
            "tokens_per_second": 6.0,
        },
        "serving_fixture.json": {
            "request_rate": 1.0,
            "request_throughput": 2.0,
            "output_throughput": 3.0,
            "median_ttft_ms": 4.0,
            "median_tpot_ms": 5.0,
            "median_itl_ms": 6.0,
        },
        "unrecognized.json": {"ignored": True},
    }
    results = tmp_path / "serving_latency_throughput_reports"
    results.mkdir()
    for name, payload in cases.items():
        (results / name).write_text(json.dumps(payload), encoding="utf-8")
    output = results / "output"
    output.mkdir()
    template = tmp_path / "template.md"
    template.write_text(
        "{latency_tests_markdown_table}\n{throughput_tests_markdown_table}\n{serving_tests_markdown_table}",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--results_folder", str(results),
            "--output_folder", str(output),
            "--markdown_template", str(template),
        ],
        capture_output=True, text=True, check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = (output / "benchmark_results.md").read_text(encoding="utf-8")
    assert "latency_fixture" in report
    assert "throughput_fixture" in report
    assert "serving_fixture" in report
    assert "unrecognized" not in report
