# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for shared-scale INT8 pipelines and ordered asynchronous SUMs."""

import importlib.util
import io
import sys
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

_BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"
_BASE_SPEC = importlib.util.spec_from_file_location("int8_base_fixture", _BENCHMARKS / "benchmark_int8_all_reduce.py")
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)
_SPEC = importlib.util.spec_from_file_location("int8_pipeline_fixture", _BENCHMARKS / "benchmark_int8_pipeline.py")
bench = importlib.util.module_from_spec(_SPEC)
with patch.dict(sys.modules, {"benchmark_int8_all_reduce": base}):
    _SPEC.loader.exec_module(bench)


class _Work:
    def __init__(self, engine, index, rank):
        self.engine = engine
        self.index = index
        self.rank = rank
        self.waited = False

    def wait(self):
        with self.engine.condition:
            self.engine.events[self.rank].append(("wait", self.index))
            ready = self.engine.condition.wait_for(
                lambda: self.engine.completed > self.index or self.engine.error is not None, timeout=20
            )
            if self.engine.error is not None:
                raise self.engine.error
            if not ready:
                raise AssertionError(f"Collective {self.index}, rank {self.rank} did not complete")
            self.waited = True


class _OrderedAllReduce:
    """One FIFO communication stream; weak references detect early buffer release."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.condition = threading.Condition()
        self.local = threading.local()
        self.sequences = [0] * world_size
        self.events = [[] for _ in range(world_size)]
        self.calls = {}
        self.completed = 0
        self.error = None

    def abort(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def all_reduce(self, tensor, op=bench.dist.ReduceOp.SUM, group=None, async_op=False):
        rank = self.local.rank
        with self.condition:
            index = self.sequences[rank]
            self.sequences[rank] += 1
            kind = "MAX" if op == bench.dist.ReduceOp.MAX else "SUM"
            self.events[rank].append((kind, index))
            work = _Work(self, index, rank)
            self.calls.setdefault(index, {})[rank] = (weakref.ref(tensor), kind, work)
            while len(self.calls.get(self.completed, {})) == self.world_size:
                batch = [self.calls[self.completed][owner] for owner in range(self.world_size)]
                tensors = [entry[0]() for entry in batch]
                if any(value is None for value in tensors):
                    raise AssertionError("Async SUM buffer was released before completion")
                metadata = {(entry[1], value.shape, value.dtype) for entry, value in zip(batch, tensors)}
                if len(metadata) != 1:
                    raise AssertionError("Collective shape/dtype/order differs across ranks")
                stacked = torch.stack(tensors)
                if batch[0][1] == "MAX":
                    result = stacked.amax(dim=0)
                else:
                    sum_dtype = torch.int32 if stacked.dtype == torch.int8 else stacked.dtype
                    result = stacked.sum(dim=0, dtype=sum_dtype).to(stacked.dtype)
                for value in tensors:
                    value.copy_(result)
                self.completed += 1
                self.condition.notify_all()
        if not async_op:
            work.wait()
            return None
        return work


class TestInt8PipelineBenchmark(unittest.TestCase):
    def test_uneven_rows_do_not_require_tp_alignment(self):
        self.assertEqual(bench.chunk_ranges(15, 4), [(0, 4), (4, 8), (8, 12), (12, 15)])
        self.assertEqual(bench.chunk_ranges(3, 3), [(0, 1), (1, 2), (2, 3)])
        for rows, chunks in ((0, 2), (2, 4), (8, 0)):
            with self.subTest(rows=rows, chunks=chunks), self.assertRaises(ValueError):
                bench.chunk_ranges(rows, chunks)

    def test_arguments_and_integer_bound(self):
        args = bench._parse_args([])
        self.assertEqual((args.chunks, args.quant_backend, args.dequant_backend), ([2, 4], "npu", "npu"))
        invalid = (
            ["--chunks", "1"],
            ["--chunks", "2", "2"],
            ["--tokens", "3"],
            ["--iterations", "0"],
            ["--hidden-sizes", "0"],
            ["--check-only", "--profile-dir", "trace"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench._parse_args(argv)
        for world_size, maximum in ((2, 63), (4, 31), (8, 15)):
            pipeline = bench.ChunkedInt8AllReduce(
                world_size, base.QuantizationOps("torch", "torch"), chunks=2, pipelined=True
            )
            self.assertEqual(pipeline.max_q, maximum)
        with self.assertRaises(ValueError):
            bench.ChunkedInt8AllReduce(2, MagicMock(), chunks=1, pipelined=True)

    @staticmethod
    def _sources(world_size):
        indices = torch.arange(15 * 64).reshape(15, 64)
        sources = []
        for rank in range(world_size):
            source = (((indices + rank * 3) % 31 - 15).float() / 8).to(torch.bfloat16)
            source[:, 0] = 0
            source[:, 1] = 2**-30
            source[-1, 2] = 32 * (rank + 1)
            source[0, 3] = -64 * (world_size - rank)
            sources.append(source)
        return sources

    def _run_threads(self, engine, worker):
        def guarded_worker(rank):
            engine.local.rank = rank
            try:
                worker(rank)
            except BaseException as error:
                engine.abort(error)
                raise

        with (
            patch.object(bench.dist, "all_reduce", side_effect=engine.all_reduce),
            ThreadPoolExecutor(max_workers=engine.world_size) as executor,
        ):
            futures = [executor.submit(guarded_worker, rank) for rank in range(engine.world_size)]
            for future in futures:
                future.result(timeout=30)

    def _simulate_pipeline(self, world_size, chunks, pipelined):
        sources = self._sources(world_size)
        max_q = 127 // world_size
        scale = (torch.stack(sources).float().abs().amax(dim=(0, 1)) / max_q).clamp_min(1e-8)
        encoded = [(source.float() / scale).round().to(torch.int8) for source in sources]
        expected_sum = torch.stack(encoded).sum(dim=0, dtype=torch.int32)
        expected = (expected_sum.float() * scale).to(torch.bfloat16)
        engine = _OrderedAllReduce(world_size)

        def worker(rank):
            pipeline = bench.ChunkedInt8AllReduce(
                world_size, base.QuantizationOps("torch", "torch"), chunks=chunks, pipelined=pipelined
            )
            working = torch.empty_like(sources[rank])
            for _ in range(3):
                working.copy_(sources[rank])
                self.assertIs(pipeline(working), working)
                torch.testing.assert_close(working, expected, rtol=0, atol=0)
            actual_sum = pipeline.reduce_quantized(encoded[rank])
            torch.testing.assert_close(actual_sum.to(torch.int32), expected_sum, rtol=0, atol=0)
            # Raw SUM must not mutate the reference payload.
            torch.testing.assert_close(encoded[rank], (sources[rank].float() / scale).round().to(torch.int8))
            with engine.condition:
                for batch in engine.calls.values():
                    if rank in batch:
                        self.assertTrue(batch[rank][2].waited)

        self._run_threads(engine, worker)
        self.assertEqual(engine.completed, 3 * (1 + chunks) + chunks)
        for events in engine.events:
            prefix = [("MAX", 0), ("wait", 0), ("SUM", 1)]
            prefix += [("SUM", 2), ("wait", 1)] if pipelined else [("wait", 1), ("SUM", 2)]
            self.assertEqual(events[:5], prefix)
        for index in range(3 * (1 + chunks)):
            expected_kind = "MAX" if index % (chunks + 1) == 0 else "SUM"
            self.assertEqual(engine.calls[index][0][1], expected_kind)

    def test_multirank_shared_scale_tail_lifetime_and_repeated_calls(self):
        for world_size in (2, 4, 8):
            for chunks in (2, 4):
                for pipelined in (False, True):
                    with self.subTest(tp=world_size, chunks=chunks, pipelined=pipelined):
                        self._simulate_pipeline(world_size, chunks, pipelined)

    def test_accuracy_reference_with_distinct_rank_data(self):
        for world_size in (2, 4, 8):
            with self.subTest(tp=world_size):
                sources = self._sources(world_size)
                engine = _OrderedAllReduce(world_size)

                def worker(rank, world_size=world_size, sources=sources):
                    ops = base.QuantizationOps("torch", "torch")
                    variants = {"int8_full": base.Int8AllReduce(world_size, ops)}
                    for chunks in (2, 4):
                        for pipelined, name in ((False, "serial"), (True, "pipeline")):
                            variants[f"{name}_{chunks}"] = bench.ChunkedInt8AllReduce(
                                world_size, ops, chunks=chunks, pipelined=pipelined
                            )
                    accuracy = bench._check_variants(variants, sources[rank])
                    for result in accuracy.values():
                        self.assertTrue(result["pass"])
                        self.assertTrue(result["same_quantized_bytes"])
                        self.assertEqual(result["same_payload_max_abs_vs_int32_sum"], 0)
                        self.assertEqual(result["max_abs_vs_full"], 0)

                self._run_threads(engine, worker)

    @staticmethod
    def _identical_rank_all_reduce(tensor, op=None, group=None, async_op=False):
        if op != bench.dist.ReduceOp.MAX:
            tensor.mul_(2)
        return MagicMock() if async_op else None

    def test_same_sign_boundaries_zero_and_cancellation(self):
        ops = base.QuantizationOps("torch", "torch")
        variants = {
            "int8_full": base.Int8AllReduce(2, ops),
            "pipeline_chunks_2": bench.ChunkedInt8AllReduce(2, ops, chunks=2, pipelined=True),
        }
        for value in (0.0, 1.0, -1.0):
            with (
                self.subTest(value=value),
                patch.object(bench.dist, "all_reduce", side_effect=self._identical_rank_all_reduce),
            ):
                accuracy = bench._check_variants(variants, torch.full((9, 64), value, dtype=torch.bfloat16))
                self.assertTrue(accuracy["pipeline_chunks_2"]["pass"])
        engine = _OrderedAllReduce(2)

        def worker(rank):
            accuracy = bench._check_variants(variants, torch.full((9, 64), (-1.0) ** rank, dtype=torch.bfloat16))
            self.assertEqual(accuracy["pipeline_chunks_2"]["int8_relative_l2_vs_fp32"], 0)

        self._run_threads(engine, worker)

    def test_accuracy_rejects_unsafe_integer_range_and_corrupt_sum(self):
        source = torch.ones(8, 64, dtype=torch.bfloat16)
        ops = base.QuantizationOps("torch", "torch")
        full = base.Int8AllReduce(2, ops)
        chunked = bench.ChunkedInt8AllReduce(2, ops, chunks=2, pipelined=True)
        with patch.object(bench.dist, "all_reduce", side_effect=self._identical_rank_all_reduce):
            with (
                patch.object(ops, "quantize", return_value=torch.full_like(source, 127, dtype=torch.int8)),
                self.assertRaisesRegex(AssertionError, "integer-safe"),
            ):
                bench._check_variants({"int8_full": full}, source)
            with (
                patch.object(chunked, "reduce_quantized", return_value=torch.zeros_like(source, dtype=torch.int8)),
                self.assertRaisesRegex(AssertionError, "INT8 SUM differs"),
            ):
                bench._check_variants({"int8_full": full, "pipeline_chunks_2": chunked}, source)
            original_quantize = ops.quantize

            def different_chunk_quantization(x, scale):
                return original_quantize(x, scale) if x.shape[0] == 8 else torch.zeros_like(x, dtype=torch.int8)

            with (
                patch.object(ops, "quantize", side_effect=different_chunk_quantization),
                self.assertRaisesRegex(AssertionError, "quantization bytes differ"),
            ):
                bench._check_variants({"int8_full": full, "pipeline_chunks_2": chunked}, source)

    def test_measurements_keep_distinct_baselines_and_rotated_order(self):
        args = bench._parse_args(["--chunks", "2", "--warmup", "0", "--rounds", "2"])
        variants = {name: MagicMock() for name in ("bf16", "int8_full", "serial_chunks_2", "pipeline_chunks_2")}
        durations = dict(zip(variants, (10.0, 8.0, 12.0, 6.0)))
        calls = []

        def measure(op, source, iterations):
            name = next(name for name, candidate in variants.items() if op is candidate)
            calls.append(name)
            duration = durations[name]
            return {
                "samples_ms": [duration, duration],
                "wall_ms_including_reset": duration + 1,
                "extra_peak_allocated_mib": 2.0,
            }

        with (
            patch.object(bench, "_check_variants", return_value={}),
            patch.object(bench.base, "_measure", side_effect=measure),
            patch.object(bench.torch, "npu", create=True),
        ):
            case = bench._benchmark_case(variants, torch.ones(8, 64, dtype=torch.bfloat16), args)
        self.assertEqual(calls[:4], list(variants))
        self.assertNotEqual(calls[:4], calls[4:])
        result = case["measurements"]["pipeline_chunks_2"]
        self.assertEqual(result["speedup_vs_bf16"], 10 / 6)
        self.assertEqual(result["speedup_vs_full"], 8 / 6)
        self.assertEqual(result["speedup_vs_serial_chunks"], 2.0)
        self.assertEqual(result["samples_ms"], [6.0] * 4)
        self.assertEqual(result["round_medians_ms"], [6.0, 6.0])
        self.assertEqual(result["p95_ms"], 6.0)
        args.check_only = True
        with (
            patch.object(bench, "_check_variants", return_value={}),
            patch.object(bench.base, "_measure", side_effect=AssertionError("Check-only must not time")),
        ):
            self.assertNotIn("measurements", bench._benchmark_case(variants, torch.ones(8, 64), args))

    def test_offline_report_does_not_initialize_npu(self):
        report = {
            "world_size": 2,
            "device": "CPU fixture",
            "mode": "benchmark",
            "cases": [
                {
                    "shape": [8, 64],
                    "bf16_payload_mib": 0.001,
                    "chunk_rows": {"2": [4, 4]},
                    "accuracy": {
                        "pipeline_chunks_2": {
                            "int8_relative_l2_vs_fp32": 0.01,
                            "int8_max_abs_vs_fp32": 0.5,
                            "same_payload_max_abs_vs_int32_sum": 0,
                            "max_abs_vs_full": 0,
                        }
                    },
                    "measurements": {
                        "pipeline_chunks_2": {
                            "median_ms": 6.0,
                            "p95_ms": 7.0,
                            "speedup_vs_bf16": 10 / 6,
                            "speedup_vs_full": 8 / 6,
                            "speedup_vs_serial_chunks": 2.0,
                            "extra_peak_allocated_mib": 2.0,
                            "median_wall_ms_including_reset": 6.5,
                        }
                    },
                }
            ],
        }
        args = bench._parse_args([])
        args.report_json = MagicMock()
        args.report_json.read_text.return_value = bench.json.dumps(report)
        output = io.StringIO()
        with (
            patch.object(bench, "_parse_args", return_value=args),
            patch.object(bench.importlib, "import_module", side_effect=AssertionError("Must not import torch_npu")),
            redirect_stdout(output),
        ):
            bench.main()
        for label in ("relative L2", "max abs", "INT8 SUM vs INT32", "p95 ms", "2.000x"):
            self.assertIn(label, output.getvalue())


if __name__ == "__main__":
    unittest.main()
