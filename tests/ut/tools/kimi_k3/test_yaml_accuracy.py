# SPDX-License-Identifier: Apache-2.0
import ast
import logging
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from tools.accuracy_runner import run_custom_accuracy


class YamlAccuracyTests(unittest.TestCase):
    def test_custom_return_contract(self):
        module = types.SimpleNamespace(run=lambda **kwargs: 62.4)
        config = {"case_type": "accuracy", "runner": "example.run"}
        with patch("tools.accuracy_runner.importlib.import_module", return_value=module):
            self.assertEqual(run_custom_accuracy("model", 1234, "localhost", config), 62.4)
            for value in [None, True, float("nan"), float("inf"), -1, 101]:
                module.run = lambda _value=value, **kwargs: _value
                with self.assertRaises(ValueError):
                    run_custom_accuracy("model", 1234, "localhost", config)
        with self.assertRaises(ValueError):
            run_custom_accuracy("model", 1, "localhost", {"case_type": "performance"})
        with self.assertRaises(ValueError):
            run_custom_accuracy("model", 1, "localhost", {"case_type": "accuracy", "runner": "bad"})

    def test_shared_dispatch_preserves_existing_runner(self):
        # Load only this orchestration function; the full AISBench module needs runtime dependencies.
        tree = ast.parse(Path("tools/aisbench.py").read_text())
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "run_aisbench_cases")
        calls = []

        class ExistingRunner:
            result = 75

            def __init__(self, **kwargs):
                calls.append(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        namespace = {"logging": logging, "AisbenchRunner": ExistingRunner}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<shared-dispatch>", "exec"), namespace)
        run = namespace["run_aisbench_cases"]
        self.assertEqual(run("model", 1234, [{"case_type": "accuracy"}]), [75])
        self.assertEqual(len(calls), 1)
        with patch("tools.accuracy_runner.run_custom_accuracy", return_value=62.4) as custom:
            self.assertEqual(run("model", 1234, [{"case_type": "accuracy", "runner": "example.run"}]), [62.4])
            custom.assert_called_once()
        self.assertEqual(len(calls), 1)
        with (
            patch("tools.accuracy_runner.run_custom_accuracy", side_effect=ValueError("bad results")),
            self.assertRaises(AssertionError),
        ):
            run("model", 1234, [{"case_type": "accuracy", "runner": "example.run"}])

    def test_yaml_registration_and_service_parity(self):
        root = Path("tests/e2e/cases/models/configs/Kimi")
        case = yaml.safe_load((root / "Kimi-K3-W4A8-EP256-A5.yaml").read_text())["test_cases"][0]
        from tools.kimi_k3.accuracy_case import service_arguments

        flags = list(case["server_cmd"])
        port = flags.index("--port")
        del flags[port : port + 2]
        self.assertEqual(flags, service_arguments(Path("tools/kimi_k3/serve_ep256.sh").read_text()))
        self.assertEqual(case["model"], "${KIMI_EP256_MODEL}")
        self.assertEqual(case["test_content"], [])
        self.assertEqual(case["benchmarks"]["accuracy"]["runner"], "tools.kimi_k3.yaml_accuracy.run_accuracy")
        entries = yaml.safe_load(Path(".github/workflows/configs/nightly_config.yaml").read_text())["a5"][
            "single_node"
        ]["test_config"]
        entry = next(x for x in entries if x["name"] == case["name"])
        self.assertNotIn("tests", entry)
        self.assertTrue((Path(entry["config_base_path"]) / entry["config_file_path"]).is_file())

    def test_resolved_thresholds_reach_shared_report(self):
        path = Path("tests/e2e/common/single_node/test_single_node.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "_run_benchmarks")
        config = types.SimpleNamespace(
            name="example",
            model="model",
            test_content=[],
            benchmarks={"accuracy": {"runner": "example.run", "baseline": "${BASELINE}"}},
        )

        def run(**kwargs):
            kwargs["aisbench_cases"][0].update(baseline=60.9, threshold=5)
            return [62.4]

        reports = []
        namespace = {
            "SingleNodeConfig": object,
            "run_aisbench_cases": run,
            "_save_benchmark_results_json": lambda cfg, keys, result: reports.append(cfg.benchmarks),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<report-dispatch>", "exec"), namespace)
        namespace["_run_benchmarks"](config, 1234)
        self.assertEqual(reports[0]["accuracy"]["baseline"], 60.9)
        self.assertEqual(reports[0]["accuracy"]["threshold"], 5)
