# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the native MXFP8 benchmark; no CANN/vLLM imports required."""

import importlib.util
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

_SCRIPT = Path(__file__).resolve().parents[4] / "benchmarks/benchmark_native_mxfp8_all_reduce.py"
_SPEC = importlib.util.spec_from_file_location("native_mxfp8_benchmark", _SCRIPT)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


class TestNativeMxfp8Benchmark(unittest.TestCase):
    def test_arguments_and_supported_world_sizes(self):
        args = bench._parse_args([])
        self.assertEqual(args.hidden_sizes, [2048, 5120])
        self.assertFalse(args.check_only)
        for argv in (
            ["--tokens", "0"],
            ["--hidden-sizes", "128"],
            ["--hidden-sizes", "8193"],
            ["--hidden-sizes", "1030"],
            ["--iterations", "0"],
            ["--rounds", "0"],
            ["--warmup", "-1"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                bench._parse_args(argv)
        for size in (1, 3, 16):
            with self.assertRaises(ValueError):
                bench.NativeMxfp8AllReduce(size, "quant-domain", MagicMock(), MagicMock())
        with self.assertRaises(RuntimeError):
            bench.NativeMxfp8AllReduce(2, "quant-domain", None, MagicMock())

    def test_official_signatures_and_scale_bytes(self):
        source = torch.ones(2, 1024, dtype=torch.bfloat16)
        q = source.to(torch.float8_e4m3fn)
        # Viewing another dtype here deliberately exercises storage semantics.
        scale_bytes = torch.arange(120, 184, dtype=torch.uint8).reshape(2, 16, 2)
        quantize = MagicMock(return_value=(q, scale_bytes.view(torch.float8_e8m0fnu)))
        native = MagicMock()
        pipeline = bench.NativeMxfp8AllReduce(2, "quant-only", quantize, native)
        values, scales = pipeline.quantize(source)
        quantize.assert_called_once_with(source, dst_type=torch.float8_e4m3fn)
        torch.testing.assert_close(scales, scale_bytes, rtol=0, atol=0)
        pipeline.reduce_quantized(values, scales)
        native.assert_called_once_with(
            values, scales, "quant-only", 2, reduce_op="sum", output_dtype=15, x_dtype=292, scales_dtype=293
        )
        pipeline.reduce_quantized(values, scales, bench._DTYPE_FP32)
        self.assertEqual(native.call_args.kwargs["output_dtype"], 6)

    def test_all_fp8_encodings_match_independent_torch_conversion(self):
        raw = torch.arange(256, dtype=torch.uint8).reshape(1, 256)
        scales = torch.full((1, 4, 2), 127, dtype=torch.uint8)
        actual = bench.decode_mxfp8_cpu(raw.view(torch.float8_e4m3fn), scales)
        expected = raw.view(torch.float8_e4m3fn).float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

    def test_e8m0_edge_encodings_and_block_layout(self):
        values = torch.ones(1, 128).to(torch.float8_e4m3fn)
        scales = torch.tensor([0, 126, 127, 254], dtype=torch.uint8).reshape(1, 2, 2)
        decoded = bench.decode_mxfp8_cpu(values, scales)
        expected = torch.tensor([2.0**-127, 0.5, 1.0, 2.0**127]).repeat_interleave(32).reshape(1, 128)
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
        scales.fill_(255)
        self.assertTrue(torch.isnan(bench.decode_mxfp8_cpu(values, scales)).all())
        with (
            patch.object(bench.dist, "all_reduce", side_effect=self._identical_rank_sum),
            self.assertRaisesRegex(AssertionError, "NaN/Inf"),
        ):
            bench._reference_sum(values, scales)

    @staticmethod
    def _quantize_unit_scale(source, dst_type):
        scales = torch.full((source.shape[0], source.shape[1] // 64, 2), 127, dtype=torch.uint8)
        return source.to(dst_type), scales

    @staticmethod
    def _identical_rank_sum(tensor, op=None, group=None, world_size=2):
        if op != bench.dist.ReduceOp.MAX:
            tensor.mul_(world_size)

    def test_zero_and_constant_inputs_on_two_four_eight_ranks(self):
        for size in (2, 4, 8):
            def collective(tensor, op=None, group=None, world_size=size):
                self._identical_rank_sum(tensor, op, group, world_size=world_size)

            def native(values, scales, hcom, world_size, **kwargs):
                self.assertEqual(hcom, "quant-only")
                dtype = torch.bfloat16 if kwargs["output_dtype"] == 15 else torch.float32
                return (bench.decode_mxfp8_cpu(values, scales) * world_size).to(dtype)

            pipeline = bench.NativeMxfp8AllReduce(size, "quant-only", self._quantize_unit_scale, native)
            for value in (0.0, 1.0):
                with (
                    self.subTest(size=size, value=value),
                    patch.object(bench.dist, "all_reduce", side_effect=collective),
                ):
                    accuracy = bench._check(pipeline, torch.full((2, 1024), value, dtype=torch.bfloat16), exact=True)
                self.assertEqual(accuracy["mxfp8_relative_l2_vs_fp32"], 0)
                self.assertTrue(accuracy["comm_bf16"]["pass"])
                self.assertTrue(accuracy["comm_fp32"]["pass"])

    def test_noop_and_wrong_output_dtype_are_rejected(self):
        source = torch.ones(2, 1024, dtype=torch.bfloat16)
        for output in (
            torch.zeros_like(source),
            torch.ones_like(source, dtype=torch.float32),
            torch.full_like(source, float("nan")),
        ):
            pipeline = bench.NativeMxfp8AllReduce(
                2, "quant-only", self._quantize_unit_scale, MagicMock(return_value=output)
            )
            with (
                self.subTest(dtype=output.dtype),
                patch.object(bench.dist, "all_reduce", side_effect=self._identical_rank_sum),
                self.assertRaises(AssertionError),
            ):
                bench._check(pipeline, source, exact=True)

    def test_distinct_rank_and_block_scales_use_separate_collective_domains(self):
        for size in (2, 4, 8):
            with self.subTest(world_size=size):
                self._run_ranks(size)

    def _run_ranks(self, size):
        raw_inputs = [torch.full((2, 1024), 0x38 + rank, dtype=torch.uint8) for rank in range(size)]
        scale_inputs = [
            ((torch.arange(64).reshape(2, 16, 2) + rank) % 7 + 124).to(torch.uint8) for rank in range(size)
        ]
        decoded = [
            (
                raw.view(torch.float8_e4m3fn).float().reshape(2, 32, 32)
                * (2.0 ** (scales.float().reshape(2, 32, 1) - 127))
            ).reshape(2, 1024)
            for raw, scales in zip(raw_inputs, scale_inputs)
        ]
        unrounded = torch.stack(decoded).sum(0)
        expected = unrounded.to(torch.bfloat16)
        state = threading.local()
        ordinary_slots = [None] * size
        quant_slots = [None] * size

        def ordinary_complete():
            stack = torch.stack([slot[0].clone() for slot in ordinary_slots])
            result = stack.amax(0) if ordinary_slots[0][1] == bench.dist.ReduceOp.MAX else stack.sum(0)
            for target, _ in ordinary_slots:
                target.copy_(result)

        def quant_complete():
            for target in quant_slots:
                target.copy_(unrounded.to(target.dtype))

        ordinary_barrier = threading.Barrier(size, action=ordinary_complete, timeout=20)
        quant_barrier = threading.Barrier(size, action=quant_complete, timeout=20)

        def ordinary(tensor, op=None, group=None):
            self.assertIsNone(group, "Ordinary collective touched the quantized group")
            ordinary_slots[state.rank] = (tensor, op)
            ordinary_barrier.wait()

        def quantize(source, dst_type):
            return raw_inputs[state.rank].view(dst_type), scale_inputs[state.rank]

        def native(values, scales, hcom, world_size, **kwargs):
            self.assertEqual(hcom, "dedicated-native-domain")
            self.assertEqual(world_size, size)
            # Communication-only fixture uses an independently computed sum.
            dtype = torch.float32 if kwargs["output_dtype"] == bench._DTYPE_FP32 else torch.bfloat16
            output = torch.empty(expected.shape, dtype=dtype)
            quant_slots[state.rank] = output
            quant_barrier.wait()
            return output

        def worker(rank):
            state.rank = rank
            pipeline = bench.NativeMxfp8AllReduce(size, "dedicated-native-domain", quantize, native)
            source = decoded[rank].to(torch.bfloat16)
            accuracy = bench._check(pipeline, source)
            self.assertTrue(accuracy["full_path"]["pass"])
            working = source.clone()
            self.assertIs(pipeline(working), working)
            torch.testing.assert_close(working, expected, rtol=0, atol=0)

        with (
            patch.object(bench.dist, "all_reduce", side_effect=ordinary),
            ThreadPoolExecutor(max_workers=size) as pool,
        ):
            list(pool.map(worker, range(size)))

    def test_measurement_resets_input_outside_event_interval(self):
        clock = SimpleNamespace(value=0)
        trace = []

        class Event:
            def __init__(self, enable_timing):
                self.timestamp = None

            def record(self):
                self.timestamp = clock.value

            def elapsed_time(self, other):
                return other.timestamp - self.timestamp

        def reset(source):
            trace.append("reset")
            clock.value += 5

        def operation(working):
            self.assertEqual(trace[-1], "reset")
            trace.append("operation")
            clock.value += 3

        fake_npu = SimpleNamespace(
            Event=Event,
            synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None,
            memory_allocated=lambda: 100,
            max_memory_allocated=lambda: 100 + 1024 * 1024,
        )
        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(torch, "empty_like", return_value=SimpleNamespace(copy_=reset)),
            patch.object(bench.dist, "barrier"),
            patch.object(bench, "_max_across_ranks", side_effect=lambda values, device: values),
        ):
            result = bench._measure(operation, torch.ones(2, 1024), iterations=3)
        self.assertEqual(trace, ["reset", "operation"] * 3)
        self.assertEqual(result["samples_ms"], [3, 3, 3])
        self.assertEqual(result["extra_peak_allocated_mib"], 1)

    def test_alternating_order_full_path_speedup_and_check_only(self):
        calls = []
        pipeline = MagicMock()

        def measure(op, source, iterations):
            kind = "mxfp8" if op is pipeline else "bf16"
            calls.append(kind)
            return {
                "samples_ms": [2 if kind == "mxfp8" else 4] * iterations,
                "wall_ms_including_reset": 5,
                "extra_peak_allocated_mib": 1,
            }

        with (
            patch.object(bench, "_check", return_value={}),
            patch.object(bench, "_measure", side_effect=measure),
            patch.object(torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True),
        ):
            args = bench._parse_args(["--warmup", "0", "--rounds", "2", "--iterations", "3"])
            result = bench._benchmark_case(pipeline, torch.ones(2, 1024), args)
            self.assertEqual(calls, ["bf16", "mxfp8", "mxfp8", "bf16"])
            self.assertEqual(result["speedup"], 2)
            self.assertEqual(result["latency_reduction_pct"], 50)
            calls.clear()
            args.check_only = True
            checked = bench._benchmark_case(pipeline, torch.ones(2, 1024), args)
            self.assertEqual(calls, [])
            self.assertNotIn("speedup", checked)

    def test_percentiles(self):
        summary = bench.summarize(list(range(1, 21)))
        self.assertEqual(summary["median_ms"], 10.5)
        self.assertEqual(summary["p95_ms"], 19)
        with self.assertRaises(ValueError):
            bench.summarize([])
