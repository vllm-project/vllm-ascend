# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the combined MXFP8 trace launcher and capture boundaries."""

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

_BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _BENCHMARKS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("trace_base_fixture", "benchmark_mxfp8_all_reduce.py")
profiling = _load("trace_profiler_fixture", "benchmark_all_reduce_modes.py")
with patch.dict(sys.modules, {"benchmark_mxfp8_all_reduce": base}):
    pipeline = _load("trace_pipeline_fixture", "benchmark_mxfp8_pipeline.py")
with patch.dict(
    sys.modules,
    {
        "benchmark_mxfp8_all_reduce": base,
        "benchmark_mxfp8_pipeline": pipeline,
        "benchmark_all_reduce_modes": profiling,
    },
):
    bench = _load("trace_capture_fixture", "profile_mxfp8_pipeline.py")


class TestMxfp8Profile(unittest.TestCase):
    def test_arguments_validate_capture_matrix(self):
        args = bench._parse_args([])
        self.assertEqual(args.chunks, [2])
        self.assertEqual(args.steps, 3)
        for argv in (["--steps", "0"], ["--chunks", "1"], ["--tp", "4", "4"], ["--hidden-sizes", "63"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench._parse_args(argv)
        args = bench._parse_args(["--tp", "2", "8", "--tokens", "8"])
        with patch.object(bench.subprocess, "run") as run, self.assertRaises(ValueError):
            bench._launch(args)
        run.assert_not_called()

    def test_launcher_preserves_parameters_and_uses_sequential_torchrun_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace directory"
            args = bench._parse_args(
                [
                    "--tp",
                    "2",
                    "4",
                    "--tokens",
                    "128",
                    "256",
                    "--chunks",
                    "2",
                    "4",
                    "--steps",
                    "5",
                    "--output-dir",
                    str(output),
                    "--profile-memory",
                ]
            )
            with patch.object(bench.subprocess, "run") as run, redirect_stdout(io.StringIO()):
                bench._launch(args)
        self.assertEqual(run.call_count, 2)
        for call, tp in zip(run.call_args_list, (2, 4)):
            command = call.args[0]
            self.assertEqual(command[:4], [sys.executable, "-m", "torch.distributed.run", "--standalone"])
            self.assertIn(f"--nproc-per-node={tp}", command)
            self.assertEqual(command[command.index("--output-dir") + 1], str(output.resolve()))
            self.assertEqual(command[command.index("--steps") + 1], "5")
            self.assertEqual(command[command.index("--chunks") + 1 : command.index("--chunks") + 3], ["2", "4"])
            self.assertIn("--profile-memory", command)
            self.assertTrue(call.kwargs["check"])

    def test_combined_capture_resets_every_input_and_keeps_synchronization_between_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            args = bench._parse_args(["--output-dir", directory, "--warmup", "2", "--steps", "3"])
            source = torch.ones(8, 64, dtype=torch.bfloat16)
            state = {"active": False, "scope": None}
            calls, labels, sync_scopes, barriers = [], [], [], []
            capture = MagicMock()

            @contextmanager
            def profiler(*args):
                state["active"] = True
                yield capture
                state["active"] = False

            @contextmanager
            def record(label):
                previous = state["scope"]
                state["scope"] = label
                labels.append(label)
                yield
                state["scope"] = previous

            def operator(name):
                def run(working):
                    torch.testing.assert_close(working, source, rtol=0, atol=0)
                    calls.append((name, state["active"]))
                    working.mul_(7)

                return run

            names = ("bf16_all_reduce", "mxfp8_full", "mxfp8_pipeline_chunks_2")
            variants = {name: operator(name) for name in names}
            with (
                patch.object(bench.torch, "npu", create=True) as npu,
                patch.object(bench.dist, "get_world_size", return_value=4),
                patch.object(bench.dist, "barrier", side_effect=lambda: barriers.append(state["active"])),
                patch.object(bench.profiling, "_create_profiler", side_effect=profiler),
                patch.object(bench.torch.profiler, "record_function", side_effect=record),
            ):
                npu.synchronize.side_effect = lambda: sync_scopes.append(state["scope"])
                result = bench._capture(variants, source, args, rank=2)
                with self.assertRaises(FileExistsError):
                    bench._capture(variants, source, args, rank=2)
            self.assertEqual(result, Path(directory) / "tp4/8x64/rank2")
            self.assertEqual(
                calls,
                [(name, False) for _ in range(2) for name in names]
                + [(name, True) for _ in range(3) for name in names],
            )
            self.assertEqual(barriers, [False, False])
            self.assertEqual(capture.step.call_count, 3)
            self.assertEqual(sync_scopes[:2], [None, None])
            self.assertEqual(
                sync_scopes[2:],
                [scope for _ in range(3) for name in names for scope in (f"input_reset/{name}", f"variant/{name}")],
            )
            self.assertIn("comparison/step_2", labels)
            torch.testing.assert_close(source, torch.ones_like(source), rtol=0, atol=0)

    def test_stage_annotations_preserve_full_and_pipeline_algorithms(self):
        labels = []

        @contextmanager
        def record(label):
            labels.append(label)
            yield

        def quantize(source, **kwargs):
            return source.to(torch.float8_e4m3fn), torch.full((source.shape[0], 2), 127, dtype=torch.uint8)

        def dequantize(values, scales, **kwargs):
            self.assertEqual(kwargs["dst_type"], 15)
            return values.to(torch.bfloat16)

        def exchange(target, source, group=None, async_op=False):
            target.copy_(source.chunk(2)[0].repeat(2, 1))
            return MagicMock() if async_op else None

        def gather(target, source, group=None, async_op=False):
            target.copy_(source.repeat(2, 1))
            return MagicMock() if async_op else None

        with (
            patch.object(bench.dist, "all_to_all_single", side_effect=exchange),
            patch.object(bench.dist, "all_gather_into_tensor", side_effect=gather),
            patch.object(bench.torch.profiler, "record_function", side_effect=record),
        ):
            variants = bench._variants(2, quantize, dequantize, [2, 4])
            self.assertEqual(
                list(variants), ["bf16_all_reduce", "mxfp8_full", "mxfp8_pipeline_chunks_2", "mxfp8_pipeline_chunks_4"]
            )
            for name, op in variants.items():
                if name != "bf16_all_reduce":
                    source = torch.ones(8, 64, dtype=torch.bfloat16)
                    self.assertIs(op(source), source)
                    torch.testing.assert_close(source, torch.full_like(source, 2), rtol=0, atol=0)
        self.assertIn("mxfp8/quantize/rows_4_8", labels)
        self.assertIn("mxfp8/quantize/rows_6_8", labels)
        self.assertIn("mxfp8/all_to_all", labels)
        self.assertIn("mxfp8/dequantize_and_sum", labels)

    def test_worker_writes_rank_manifest_and_always_destroys_group(self):
        with tempfile.TemporaryDirectory() as directory:
            args = bench._parse_args(
                ["--tp", "4", "--tokens", "16", "32", "--hidden-sizes", "64", "--output-dir", directory]
            )
            variants = {name: MagicMock() for name in ("bf16_all_reduce", "mxfp8_full", "mxfp8_pipeline_chunks_2")}
            cpu = torch.device("cpu")

            def capture(ops, source, options, rank):
                path = Path(directory) / "tp4" / f"{source.shape[0]}x64" / "rank0"
                path.mkdir(parents=True)
                return path

            with (
                patch.dict(bench.os.environ, {"WORLD_SIZE": "4", "LOCAL_RANK": "0"}),
                patch.object(bench.importlib, "import_module", return_value=SimpleNamespace(__version__="fixture")),
                patch.object(bench.torch, "npu", create=True) as npu,
                patch.object(bench.torch, "manual_seed"),
                patch.object(bench.torch, "device", return_value=cpu),
                patch.object(bench, "_variants", return_value=variants),
                patch.object(bench.dist, "init_process_group"),
                patch.object(bench.dist, "get_rank", return_value=0),
                patch.object(bench.dist, "destroy_process_group") as destroy,
                patch.object(bench.base, "_check", return_value={"pass": True}) as check,
                patch.object(bench, "_capture", side_effect=capture),
                redirect_stdout(io.StringIO()),
            ):
                npu.get_device_name.return_value = "CPU fixture"
                bench._worker(args)
                destroy.assert_called_once()
                manifest = bench.json.loads((Path(directory) / "tp4/manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(len(manifest["cases"]), 2)
                self.assertEqual(len(manifest["cases"][0]["rank_directories"]), 4)
                self.assertEqual(check.call_count, 4)
                destroy.reset_mock()
                check.side_effect = AssertionError("accuracy failed")
                with self.assertRaisesRegex(AssertionError, "accuracy failed"):
                    bench._worker(args)
                destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
