# SPDX-License-Identifier: Apache-2.0
"""Exercise the public CLI without importing NPU or benchmark runtimes."""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/e2e/nightly/single_node/models/configs/Qwen3-30B-A3B-W8A8.yaml"
CASE = "Qwen3-30B-A3B-W8A8-TP1"


class NightlyCLITest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.benchmark = self.root / "benchmark"
        configs = self.benchmark / "ais_bench/benchmark/configs"
        model = configs / "models/vllm_api/vllm_api_stream_chat.py"
        model.parent.mkdir(parents=True)
        model.write_text(
            """models = [dict(
    model='', path='', host_ip='localhost', host_port=8080,
    max_out_len=512, batch_size=1, request_rate=0,
    trust_remote_code=False,
    generation_kwargs=dict(temperature=0.01, ignore_eos=False,
    ),
)]
""",
            encoding="utf-8",
        )
        # Use the layout of upstream templates: one field on each line.
        model.write_text(model.read_text().replace(", ", ",\n    "), encoding="utf-8")
        dataset = configs / "datasets/gsm8k/gsm8k_gen_0_shot_cot_str_perf.py"
        dataset.parent.mkdir(parents=True)
        dataset.write_text("gsm8k_datasets = [dict(\n    path='original',\n)]\n", encoding="utf-8")
        self.templates = {path: path.read_bytes() for path in configs.rglob("*.py")}
        self.guard = self.root / "guard"
        self.guard.mkdir()
        (self.guard / "sitecustomize.py").write_text(
            """import sys
def prohibit(event, args):
    if event in {'subprocess.Popen', 'os.system', 'socket.connect'}:
        raise RuntimeError('CLI must not start processes or connect to services')
    if event == 'import' and args[0] in {'tools.aisbench', 'pytest', 'vllm', 'modelscope', 'huggingface_hub'}:
        raise RuntimeError('CLI must not import a runtime or lifecycle wrapper')
sys.addaudithook(prohibit)
""",
            encoding="utf-8",
        )

    def cli(self, action, *arguments, optimized=False):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(self.guard) + os.pathsep + environment.get("PYTHONPATH", "")
        return subprocess.run(
            [
                sys.executable,
                *(["-O"] if optimized else []),
                "-m",
                "tools.nightly_cli",
                action,
                "--config",
                str(CONFIG),
                "--case",
                CASE,
                "--benchmark",
                "perf",
                *map(str, arguments),
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_prepare_generates_private_direct_cli_config_without_running_or_modifying_templates(self):
        output = self.root / "private job"
        result = self.cli(
            "prepare",
            "--benchmark-home",
            self.benchmark,
            "--output-dir",
            output,
            "--model-path",
            "/models/local",
            "--dataset-path",
            "/datasets/local",
            "--host",
            "192.0.2.8",
            "--port",
            "18123",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(result.stdout)
        self.assertEqual(
            manifest["argv"],
            [
                "ais_bench",
                str(output / "benchmark.py"),
                "--mode",
                "perf",
                "--num-prompts",
                "180",
                "--work-dir",
                str(output / "results"),
                "--debug",
            ],
        )
        tree = ast.parse((output / "benchmark.py").read_text(encoding="utf-8"))
        self.assertIn(
            "datasets",
            [
                target.id
                for node in tree.body
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
            ],
        )
        namespace = {}
        exec(compile(tree, "benchmark.py", "exec"), namespace)
        model = namespace["models"][0]
        self.assertEqual((model["max_out_len"], model["batch_size"], model["request_rate"]), (1500, 45, 0))
        self.assertEqual((model["host_ip"], model["host_port"], model["path"]), ("192.0.2.8", 18123, "/models/local"))
        self.assertEqual(model["generation_kwargs"], {"temperature": 0, "ignore_eos": True})
        self.assertEqual(namespace["datasets"][0]["path"], "/datasets/local")
        self.assertTrue((output / "benchmark.sh").read_text().splitlines()[-1].startswith("exec ais_bench "))
        self.assertEqual(self.templates, {path: path.read_bytes() for path in self.templates})

    def test_server_command_preserves_yaml_parameters_and_emits_only_quoted_exec(self):
        output = self.root / "server"
        result = self.cli(
            "server-command",
            "--output-dir",
            output,
            "--model-path",
            "/models/model with 'quotes'",
            "--host",
            "192.0.2.9",
            "--port",
            "18124",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(result.stdout)
        self.assertEqual(manifest["argv"][:3], ["vllm", "serve", "/models/model with 'quotes'"])
        self.assertEqual(manifest["argv"][manifest["argv"].index("--tensor-parallel-size") + 1], "1")
        self.assertEqual(manifest["argv"][manifest["argv"].index("--max-model-len") + 1], "5600")
        self.assertEqual(manifest["argv"][manifest["argv"].index("--port") + 1], "18124")
        self.assertEqual(manifest["environment"]["OMP_NUM_THREADS"], "10")
        self.assertEqual(manifest["environment"]["SERVER_PORT"], "18124")
        script = (output / "server.sh").read_text()
        self.assertTrue(script.splitlines()[-1].startswith("exec vllm serve "))
        self.assertNotIn("$SERVER_PORT", script)

    def test_verify_checks_existing_results_without_restarting_benchmark(self):
        result_json, result_csv = self.root / "gsm8k.json", self.root / "gsm8k.csv"
        result_csv.write_text(",Average\nTPOT,10ms\n", encoding="utf-8")
        for throughput, code, verdict in (("800token/s", 0, "passed"), ("780token/s", 1, "failed")):
            with self.subTest(throughput=throughput):
                result_json.write_text(json.dumps({"Output Token Throughput": {"total": throughput}}), encoding="utf-8")
                output = self.root / (verdict + ".json")
                result = self.cli(
                    "verify", "--result-json", result_json, "--result-csv", result_csv, "--output-file", output
                )
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(json.loads(result.stdout)["verdict"], verdict)
                self.assertEqual(json.loads(output.read_text())["result_json"], str(result_json))
        result_json.unlink()
        missing = self.cli("verify", "--result-json", result_json, "--result-csv", result_csv)
        self.assertEqual(missing.returncode, 2)

    def test_verify_preserves_optional_input_throughput_and_tpot_thresholds(self):
        result_json, result_csv = self.root / "gsm8k.json", self.root / "gsm8k.csv"
        result_json.write_text(
            json.dumps(
                {"Output Token Throughput": {"total": "900token/s"}, "Input Token Throughput": {"total": "90token/s"}}
            ),
            encoding="utf-8",
        )
        result_csv.write_text(",Average\nTPOT,10ms\n", encoding="utf-8")
        for option, threshold, message in (
            ("input_throughput_threshold", 100, "Input Token Throughput"),
            ("tpot_threshold", 5, "TPOT"),
        ):
            document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
            document["test_cases"][0]["benchmarks"]["perf"][option] = threshold
            configuration = self.root / (option + ".yaml")
            configuration.write_text(yaml.safe_dump(document), encoding="utf-8")
            result = self.cli(
                "verify", "--config", configuration, "--result-json", result_json, "--result-csv", result_csv
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(message, json.loads(result.stdout)["reason"])

    def test_prepare_escapes_paths_and_does_not_overwrite_existing_outputs(self):
        output = self.root / "escaped"
        path = "/models/a'quoted\\name"
        arguments = (
            "--benchmark-home",
            self.benchmark,
            "--output-dir",
            output,
            "--model-path",
            path,
            "--dataset-path",
            path,
            "--host",
            "localhost",
            "--port",
            "18123",
        )
        result = self.cli("prepare", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        namespace = {}
        exec((output / "benchmark.py").read_text(), namespace)
        self.assertEqual(namespace["models"][0]["path"], path)
        self.assertEqual(namespace["datasets"][0]["path"], path)
        before = (output / "benchmark.py").read_bytes()
        self.assertEqual(self.cli("prepare", *arguments).returncode, 2)
        self.assertEqual((output / "benchmark.py").read_bytes(), before)

    def test_prepare_maps_single_gsm8k_file_to_private_train_test_layout(self):
        source = self.root / "official.jsonl"
        raw = b'{"question":"unchanged question","answer":"unchanged answer"}\n'
        source.write_bytes(raw)
        output = self.root / "file-case"
        result = self.cli(
            "prepare",
            "--benchmark-home",
            self.benchmark,
            "--output-dir",
            output,
            "--model-path",
            "/models/local",
            "--dataset-path",
            source,
            "--host",
            "localhost",
            "--port",
            "18123",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        namespace = {}
        exec((output / "benchmark.py").read_text(), namespace)
        directory = Path(namespace["datasets"][0]["path"])
        self.assertTrue(directory.is_relative_to(output))
        self.assertEqual((directory / "train.jsonl").read_bytes(), raw)
        self.assertEqual((directory / "test.jsonl").read_bytes(), raw)
        self.assertEqual(source.read_bytes(), raw)
        manifest = json.loads(result.stdout)
        self.assertEqual(len(manifest["dataset_input_sha256"]), 64)
        self.assertEqual(len(manifest["source_sha256"]), 64)

    def test_optimized_python_still_rejects_below_threshold_results(self):
        result_json, result_csv = self.root / "gsm8k.json", self.root / "gsm8k.csv"
        result_json.write_text('{"Output Token Throughput":{"total":"1token/s"}}', encoding="utf-8")
        result_csv.write_text(",Average\nTPOT,10ms\n", encoding="utf-8")
        result = self.cli("verify", "--result-json", result_json, "--result-csv", result_csv, optimized=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(len(report["source_sha256"]), 64)
        self.assertEqual(len(report["benchmark_sha256"]), 64)

    def test_locate_uses_current_log_and_copies_exact_results_for_explicit_verification(self):
        output = self.root / "located"
        prepared = self.cli(
            "prepare",
            "--benchmark-home",
            self.benchmark,
            "--output-dir",
            output,
            "--model-path",
            "/models/local",
            "--dataset-path",
            "/datasets/local",
            "--host",
            "localhost",
            "--port",
            "18123",
        )
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        results = output / "results/2026-09-14/performances/vllm-api-stream-chat"
        results.mkdir(parents=True)
        raw = b'{"Output Token Throughput":{"total":"800token/s"}}\n'
        (results / "gsm8k.json").write_bytes(raw)
        (results / "gsm8k.csv").write_bytes(b",Average\nTPOT,10ms\n")
        log = output / "benchmark.log"
        log.write_text("INFO Performance Result files located in " + str(results) + ".\n", encoding="utf-8")
        archive = output / "collected"
        located = self.cli(
            "locate-results", "--manifest", output / "manifest.json", "--log", log, "--output-dir", archive
        )
        self.assertEqual(located.returncode, 0, located.stderr)
        evidence = json.loads(located.stdout)
        self.assertEqual(Path(evidence["result_json"]).read_bytes(), raw)
        self.assertEqual(evidence["source_result_json"], str(results / "gsm8k.json"))
        self.assertEqual(len(evidence["result_json_sha256"]), 64)
        self.assertIn(str(archive / "result.json"), (archive / "verify.sh").read_text())
        checked = self.cli("verify", "--result-json", evidence["result_json"], "--result-csv", evidence["result_csv"])
        self.assertEqual(checked.returncode, 0, checked.stderr)
        for text in (
            "No completion marker",
            "Performance Result files located in " + str(self.root) + ".\n",
            log.read_text() + "Performance Result files located in " + str(results.parent) + ".\n",
        ):
            log.write_text(text, encoding="utf-8")
            rejected = self.cli(
                "locate-results",
                "--manifest",
                output / "manifest.json",
                "--log",
                log,
                "--output-dir",
                output / "rejected",
            )
            self.assertEqual(rejected.returncode, 2, rejected.stderr)
        self.assertFalse((output / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
