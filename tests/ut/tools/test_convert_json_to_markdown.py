# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestConvertJsonToMarkdown(unittest.TestCase):
    def test_parent_directory_does_not_change_result_type(self):
        script = Path(__file__).resolve().parents[3] / "benchmarks/scripts/convert_json_to_markdown.py"
        fixtures = {
            "latency_fixture.json": {
                "avg_latency": 0.01,
                "percentiles": {str(percentile): 0.02 for percentile in (10, 25, 50, 75, 90, 99)},
            },
            "throughput_fixture.json": {
                "num_requests": 2,
                "total_num_tokens": 4,
                "elapsed_time": 1,
                "requests_per_second": 2,
                "tokens_per_second": 4,
            },
            "serving_fixture.json": {
                "request_rate": 1,
                "request_throughput": 1,
                "output_throughput": 2,
                "median_ttft_ms": 3,
                "median_tpot_ms": 4,
                "median_itl_ms": 5,
            },
            "unrecognized_fixture.json": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.md"
            template.write_text("{benchmarking_results_in_json_string}", encoding="utf-8")
            for parent in ("reports", "serving_reports", "latency_reports", "throughput_reports"):
                with self.subTest(parent=parent):
                    results = root / parent
                    results.mkdir()
                    output = results / "output"
                    output.mkdir()
                    for name, data in fixtures.items():
                        (results / name).write_text(json.dumps(data), encoding="utf-8")
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(script),
                            "--results_folder",
                            str(results),
                            "--output_folder",
                            str(output),
                            "--markdown_template",
                            str(template),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    report = json.loads((output / "benchmark_results.md").read_text(encoding="utf-8"))
                    for kind in ("latency", "throughput", "serving"):
                        self.assertEqual(report[kind]["Test name"], {"0": f"{kind}_fixture"})
                    self.assertEqual(report["latency"]["Mean latency (ms)"], {"0": 10.0})
                    self.assertEqual(report["latency"]["Median latency (ms)"], {"0": 20.0})
                    self.assertIn("Skipping", completed.stdout)
                    self.assertIn("unrecognized_fixture.json", completed.stdout)


if __name__ == "__main__":
    unittest.main()
