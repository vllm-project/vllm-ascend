# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for aggregation and mode isolation; no NPU imports."""

import csv
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_SCRIPT = Path(__file__).resolve().parents[4] / "benchmarks/benchmark_all_reduce_modes.py"
_SPEC = importlib.util.spec_from_file_location("all_reduce_modes_benchmark", _SCRIPT)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


class TestAllReduceModesBenchmark(unittest.TestCase):
    def test_slowest_rank_is_selected_before_statistics(self):
        # max(rank medians) would be 10; median(per-call maximum) must be 100.
        samples = bench.slowest_rank_samples([[1, 100, 2], [100, 2, 10]])
        self.assertEqual(samples, [100, 100, 10])
        summary = bench.summarize(samples)
        self.assertEqual(summary["median_us"], 100)
        self.assertEqual(summary["mean_us"], 70)
        self.assertEqual(summary["p95_us"], 100)
        for invalid in ([], [0], [-1], [math.inf], [math.nan]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bench.summarize(invalid)
        with self.assertRaises(ValueError):
            bench.slowest_rank_samples([[1], [1, 2]])

    def test_comparison_direction_and_workload_consistency(self):
        config = dict(world_size=4, dtype="bfloat16", timing="event", warmup=20, iterations=100, rounds=3)
        reports = {
            "AI_CPU": {
                **config,
                "results": [{"payload_bytes": 4096, "numel": 2048, "mean_us": 25, "median_us": 20, "p95_us": 30}],
            },
            "AIV": {
                **config,
                "results": [{"payload_bytes": 4096, "numel": 2048, "mean_us": 12, "median_us": 10, "p95_us": 15}],
            },
        }
        row = bench.comparison_rows(reports)[0]
        self.assertEqual(row["ai_cpu_mean_us"], 25)
        self.assertEqual(row["aiv_mean_us"], 12)
        self.assertEqual(row["speedup_ai_cpu_over_aiv"], 2)
        self.assertEqual(row["aiv_latency_reduction_pct"], 50)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)

            def run_mode(command, **kwargs):
                mode = command[command.index("--mode") + 1]
                (output_dir / f"{mode}.json").write_text(json.dumps(reports[mode]), encoding="utf-8")

            output = io.StringIO()
            args = bench.parse_args(["--sizes", "4K", "--output-dir", directory])
            with redirect_stdout(output), patch.object(bench.subprocess, "run", side_effect=run_mode):
                bench._launch(args)
            self.assertIn("AI_CPU avg", output.getvalue())
            self.assertIn("AIV avg", output.getvalue())
            self.assertIn("Speedup(med)", output.getvalue())
            self.assertIn("25.000", output.getvalue())
            self.assertIn("12.000", output.getvalue())
            with (output_dir / "comparison.csv").open(newline="", encoding="utf-8") as stream:
                csv_row = next(csv.DictReader(stream))
            self.assertEqual(float(csv_row["ai_cpu_mean_us"]), 25)
            self.assertEqual(float(csv_row["aiv_mean_us"]), 12)
            json_row = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))["comparison"][0]
            self.assertEqual(json_row["ai_cpu_mean_us"], 25)
            self.assertEqual(json_row["aiv_mean_us"], 12)
        reports["AIV"]["timing"] = "wall"
        with self.assertRaisesRegex(ValueError, "timing"):
            bench.comparison_rows(reports)

    def test_fresh_jobs_receive_distinct_modes_and_identical_inputs(self):
        args = bench.parse_args(
            [
                "--nproc-per-node",
                "4",
                "--sizes",
                "4KiB",
                "1M",
                "--hccl-buffsize-mib",
                "512",
                "--profile",
                "--profile-steps",
                "3",
                "--profile-with-stack",
                "--profile-memory",
            ]
        )
        self.assertEqual(args.sizes, [4096, 1048576])
        with patch.dict(os.environ, {"HCCL_OP_EXPANSION_MODE": "CCU_SCHED", "HCCL_ALGO": "unchanged"}):
            commands = []
            for mode in ("AI_CPU", "AIV"):
                command, env = bench._launch_command(args, mode, Path("result"))
                commands.append(command)
                self.assertEqual(env["HCCL_OP_EXPANSION_MODE"], mode)
                self.assertEqual(env["HCCL_BUFFSIZE"], "512")
                self.assertEqual(env["HCCL_ALGO"], "unchanged")
                self.assertEqual(command[1:3], ["-m", "torch.distributed.run"])
                self.assertEqual(command[command.index("--mode") + 1], mode)
                self.assertIn("--profile", command)
                self.assertIn("--profile-with-stack", command)
                self.assertIn("--profile-memory", command)
                self.assertEqual(command[command.index("--profile-steps") + 1], "3")
            self.assertEqual(os.environ["HCCL_OP_EXPANSION_MODE"], "CCU_SCHED")
            changed = [(a, b) for a, b in zip(*commands, strict=True) if a != b]
            self.assertEqual(changed, [("AI_CPU", "AIV")])

    def test_dry_run_uses_only_stdlib_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not_created"
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(_SCRIPT), "--dry-run", "--profile", "--output-dir", str(output)],
                capture_output=True,
                text=True,
                check=True,
                env={key: value for key, value in os.environ.items() if key not in ("RANK", "WORLD_SIZE")},
            )
            self.assertIn("HCCL_OP_EXPANSION_MODE=AI_CPU", result.stdout)
            self.assertIn("HCCL_OP_EXPANSION_MODE=AIV", result.stdout)
            self.assertFalse(output.exists())

    def test_profiler_exports_vllm_compatible_traces(self):
        profiler = MagicMock()
        profiler.ExportType.Text = "text"
        profiler.ProfilerLevel.Level1 = "level1"
        profiler.AiCMetrics.PipeUtilization = "pipe_utilization"
        profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU = "cpu", "npu"
        torch_npu = SimpleNamespace(profiler=profiler)
        args = bench.parse_args(["--profile", "--profile-with-stack", "--profile-memory"])
        with patch.dict(sys.modules, {"torch_npu": torch_npu}):
            self.assertIs(bench._create_profiler(Path("traces"), "host_rank3", args), profiler.profile.return_value)
        config = profiler._ExperimentalConfig.call_args.kwargs
        self.assertEqual(config["export_type"], "text")
        self.assertEqual(config["profiler_level"], "level1")
        self.assertEqual(config["aic_metrics"], "pipe_utilization")
        profiler.tensorboard_trace_handler.assert_called_once_with(
            "traces", worker_name="host_rank3", analyse_flag=True, async_mode=False
        )
        options = profiler.profile.call_args.kwargs
        self.assertEqual(options["activities"], ["cpu", "npu"])
        self.assertTrue(options["with_modules"])
        self.assertTrue(options["profile_memory"])
        self.assertFalse(options["with_stack"])
        self.assertIs(options["on_trace_ready"], profiler.tensorboard_trace_handler.return_value)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(["--profile-steps", "0"])

    def test_profile_capture_uses_own_calls_and_rank_size_directories(self):
        class Tensor:
            value = 0

            def copy_(self, source):
                self.value = source.value

        inputs = []
        lifecycle = []
        paths = []

        def all_reduce(tensor, **kwargs):
            inputs.append(tensor.value)
            tensor.value = 3  # SUM for ranks 0 and 1.

        class Profiler:
            def __enter__(self):
                lifecycle.append("start")
                return self

            def step(self):
                lifecycle.append("step")

            def __exit__(profiler_self, *args):
                # Pending NPU work must complete before trace export.
                self.assertEqual(lifecycle[-1], "sync")
                lifecycle.append("stop")

        def create_profiler(path, worker_name, args):
            paths.append(path)
            self.assertIn("rank1", worker_name)
            return Profiler()

        def full(shape, value, **kwargs):
            tensor = Tensor()
            tensor.value = value
            return tensor

        dist = SimpleNamespace(all_reduce=all_reduce, barrier=lambda: None, ReduceOp=SimpleNamespace(SUM="sum"))
        torch = SimpleNamespace(
            distributed=dist,
            full=full,
            empty_like=lambda source: Tensor(),
            bfloat16="bf16",
            npu=SimpleNamespace(current_device=lambda: 1, synchronize=lambda: lifecycle.append("sync")),
            profiler=SimpleNamespace(record_function=lambda name: nullcontext()),
        )
        args = bench.parse_args(
            [
                "--mode",
                "AIV",
                "--sizes",
                "4K",
                "16K",
                "--warmup",
                "1",
                "--profile",
                "--profile-steps",
                "3",
            ]
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}),
            patch.object(bench, "_create_profiler", side_effect=create_profiler),
            patch.object(bench, "_check_result") as check,
        ):
            root = Path(bench._capture_profiles(args, object(), 1, 2, Path(directory)))
            self.assertEqual(root, Path(directory) / "profiles" / "AIV" / "rank_1")
            self.assertEqual(paths, [root / "4096B", root / "16384B"])
            self.assertEqual(inputs, [2] * 8)  # One warmup + three profile calls for each size.
            self.assertEqual(lifecycle.count("step"), 6)
            self.assertEqual(lifecycle.count("stop"), 2)
            self.assertEqual(check.call_count, 2)
            for call in check.call_args_list:
                self.assertEqual(call.args[1:], (3.0, 2, "bfloat16"))

    def test_timed_in_place_sum_receives_reset_input_every_iteration(self):
        class Tensor:
            def __init__(self, value):
                self.value = value

            def copy_(self, source):
                self.value = source.value

        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self):
                pass

            def elapsed_time(self, other):
                return 0.02

        observed_inputs = []

        def all_reduce(tensor, **kwargs):
            observed_inputs.append(tensor.value)
            tensor.value *= 8

        dist = SimpleNamespace(all_reduce=all_reduce, barrier=lambda: None, ReduceOp=SimpleNamespace(SUM="sum"))
        torch = SimpleNamespace(distributed=dist, npu=SimpleNamespace(Event=Event, synchronize=lambda: None))
        for timing in ("event", "wall"):
            with self.subTest(timing=timing), patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
                observed_inputs.clear()
                source, working = Tensor(3), Tensor(-1)
                args = SimpleNamespace(iterations=10, timing=timing)
                samples = bench._measure(source, working, object(), args)
                self.assertEqual(observed_inputs, [3] * 10)
                self.assertEqual(source.value, 3)
                self.assertEqual(working.value, 24)
                self.assertEqual(len(samples), 10)
                self.assertTrue(all(value > 0 for value in samples))


if __name__ == "__main__":
    unittest.main()
