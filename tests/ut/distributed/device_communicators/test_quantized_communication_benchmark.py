# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the same-run runner and shared FIFO collective ordering."""

import importlib.util
import io
import math
import sys
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

_BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"
_SPEC = importlib.util.spec_from_file_location(
    "same_run_communication_fixture", _BENCHMARKS / "benchmark_quantized_communication.py"
)
bench = importlib.util.module_from_spec(_SPEC)
with patch.object(sys, "path", [str(_BENCHMARKS), *sys.path]), patch.dict(sys.modules, {_SPEC.name: bench}):
    _SPEC.loader.exec_module(bench)


class _Work:
    def __init__(self, engine, sequence, rank):
        self.engine, self.sequence, self.rank = engine, sequence, rank
        self.waited = False

    def wait(self):
        with self.engine.condition:
            self.engine.events[self.rank].append(("wait", self.sequence))
            ready = self.engine.condition.wait_for(
                lambda: self.engine.completed > self.sequence or self.engine.error is not None, timeout=20
            )
            if self.engine.error is not None:
                raise self.engine.error
            if not ready:
                raise AssertionError("Collective order diverged between ranks")
            self.waited = True


class _MixedCollectives:
    """One FIFO for SUM, MAX, AllToAll and AllGather, across every variant."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.local = threading.local()
        self.condition = threading.Condition()
        self.calls, self.error, self.completed = {}, None, 0
        self.sequences = [0] * world_size
        self.events = [[] for _ in range(world_size)]

    def abort(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def _call(self, kind, output, source, group=None, async_op=False):
        if group is not None:
            raise AssertionError("Every variant must use the same default group")
        rank = self.local.rank
        with self.condition:
            sequence = self.sequences[rank]
            self.sequences[rank] += 1
            work = _Work(self, sequence, rank)
            self.calls.setdefault(sequence, {})[rank] = (kind, weakref.ref(source), weakref.ref(output), work)
            self.events[rank].append((kind, sequence))
            while len(self.calls.get(self.completed, {})) == self.world_size:
                entries = [self.calls[self.completed][owner] for owner in range(self.world_size)]
                sources = [entry[1]() for entry in entries]
                outputs = [entry[2]() for entry in entries]
                if any(value is None for value in (*sources, *outputs)):
                    raise AssertionError("Async storage was released too early")
                metadata = {(entry[0], value.shape, value.dtype) for entry, value in zip(entries, sources)}
                if len(metadata) != 1:
                    raise AssertionError("Collective kind/shape/dtype diverged between ranks")
                operation = entries[0][0]
                if operation == "a2a":
                    pieces = [source.chunk(self.world_size) for source in sources]
                    results = [
                        torch.cat([pieces[owner][target] for owner in range(self.world_size)])
                        for target in range(self.world_size)
                    ]
                elif operation == "gather":
                    results = [torch.cat(sources)] * self.world_size
                else:
                    stacked = torch.stack(sources)
                    if operation == "max":
                        result = stacked.amax(0)
                    else:
                        dtype = torch.int32 if stacked.dtype == torch.int8 else stacked.dtype
                        result = stacked.sum(0, dtype=dtype).to(stacked.dtype)
                    results = [result] * self.world_size
                for output, result in zip(outputs, results):
                    output.copy_(result)
                self.completed += 1
                self.condition.notify_all()
        if async_op:
            return work
        work.wait()
        return None

    def all_reduce(self, source, op=bench.dist.ReduceOp.SUM, **kwargs):
        return self._call("max" if op == bench.dist.ReduceOp.MAX else "sum", source, source, **kwargs)

    def all_to_all_single(self, output, source, **kwargs):
        return self._call("a2a", output, source, **kwargs)

    def all_gather_into_tensor(self, output, source, **kwargs):
        return self._call("gather", output, source, **kwargs)


def _ops():
    def quantize(x, scale, offset, dtype, axis, div_mode):
        return (x.float() / scale).round().to(dtype)

    def dynamic_mx(x, **kwargs):
        # CPU FP8 conversion plus neutral E8M0 scales; the independent byte
        # decoder must agree. Encoded sanity checks additionally vary scales.
        return x.to(torch.float8_e4m3fn), torch.full((x.shape[0], x.shape[1] // 64, 2), 127, dtype=torch.uint8)

    def anti_mx(values, scales, **kwargs):
        rows, hidden = values.shape
        factors = torch.exp2(scales.float().reshape(rows, hidden // 32, 1) - 127)
        return (values.float().reshape(rows, hidden // 32, 32) * factors).reshape(rows, hidden).to(torch.bfloat16)

    return SimpleNamespace(
        npu_quantize=quantize,
        npu_anti_quant=lambda q, scale, dst_dtype: (q.float() * scale).to(dst_dtype),
        npu_dynamic_mx_quant=dynamic_mx,
        npu_anti_mx_quant=anti_mx,
    )


class TestQuantizedCommunicationBenchmark(unittest.TestCase):
    def test_builds_all_variants_with_native_bf16_output_and_one_group(self):
        variants = bench._build_variants(2, ["int8", "mxfp8"], [2, 4], _ops())
        self.assertEqual(len(variants), 11)
        self.assertEqual(variants[0].name, "bf16")
        self.assertIs(variants[0].op, bench.dist.all_reduce)
        self.assertEqual(len({variant.name for variant in variants}), 11)
        for variant in variants[1:]:
            self.assertIsNone(variant.op.group)
            if variant.backend == "int8":
                self.assertEqual(variant.op.max_q, 63)
                self.assertEqual(variant.op.ops.quant_backend, "npu")
            else:
                self.assertEqual(variant.op.dequant_dtype, torch.bfloat16)
            self.assertEqual(set(variant.metadata()), {"name", "backend", "mode", "chunks"})

    def test_order_balances_every_position_and_reverses(self):
        names = [f"variant{index}" for index in range(11)]
        for start in (0, len(names)):
            orders = [bench._order(names, cycle) for cycle in range(start, start + len(names))]
            for position in range(len(names)):
                self.assertEqual({order[position] for order in orders}, set(names))
        self.assertEqual(bench._order(names, 11), names[::-1])
        self.assertEqual(bench._order(names, 22), names)

    def test_arguments_and_shape_constraints(self):
        args = bench._parse_args([])
        self.assertEqual((args.backends, args.chunks, args.block_iterations), (["int8", "mxfp8"], [2, 4], 5))
        for argv in (
            ["--rounds", "0"],
            ["--block-iterations", "0"],
            ["--tokens", "0"],
            ["--hidden-sizes", "32"],
            ["--chunks", "1"],
            ["--chunks", "2", "2"],
            ["--backends", "int8", "int8"],
        ):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench._parse_args(argv)
        odd = bench._parse_args(["--tokens", "129"])
        with self.assertRaises(ValueError):
            bench._validate_shapes(odd, 2)
        odd.backends = ["int8"]
        bench._validate_shapes(odd, 2)
        for size in (2, 4, 8):
            bench._validate_shapes(args, size)
        with self.assertRaises(ValueError):
            bench._validate_shapes(args, 1)

    def test_summary_mean_percentiles_variation_and_invalid_timings(self):
        result = bench._summary([1.0] * 99 + [101.0])
        self.assertEqual((result["mean_ms"], result["median_ms"], result["p95_ms"], result["p99_ms"]), (2, 1, 1, 1))
        self.assertEqual(result["sample_count"], 100)
        self.assertGreater(result["stddev_ms"], 0)
        self.assertEqual(bench._summary([2.0])["stddev_ms"], 0)
        for samples in ([], [0.0], [-1.0], [math.nan], [math.inf]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                bench._summary(samples)

    def _measured_case(self):
        args = bench._parse_args(["--iterations", "7", "--block-iterations", "3", "--rounds", "2", "--warmup", "0"])
        variants = bench._build_variants(2, ["int8", "mxfp8"], [2, 4], _ops())
        durations = {variant.name: 2.0 + index for index, variant in enumerate(variants)}
        durations.update(bf16=20.0, int8_full=10.0, mxfp8_full=8.0)
        names = {id(variant.op): variant.name for variant in variants}
        calls = []
        source = torch.ones(16, 64, dtype=torch.bfloat16)

        def measure(op, actual_source, count):
            self.assertIs(actual_source, source)
            calls.append((names[id(op)], count))
            value = durations[names[id(op)]]
            return {
                "samples_ms": [value] * count,
                "wall_ms_including_reset": float(count),
                "extra_peak_allocated_mib": 3.0,
            }

        with (
            patch.object(bench, "_check_variants", return_value={}),
            patch.object(bench.int8_base, "_measure", side_effect=measure),
            patch.object(torch, "npu", MagicMock(), create=True),
        ):
            case = bench._benchmark_case(variants, source, args)
        return case, calls, variants, args

    def test_interleaved_samples_counts_weighted_wall_and_comparisons(self):
        case, calls, variants, args = self._measured_case()
        expected = [(name, cycle["iterations"]) for cycle in case["schedule"] for name in cycle["order"]]
        self.assertEqual(calls, expected)
        self.assertEqual(len(case["schedule"]), 6)
        self.assertNotEqual(case["schedule"][0]["order"], case["schedule"][1]["order"])
        for item in case["measurements"].values():
            self.assertEqual(item["sample_count"], 14)
            self.assertEqual([len(block["samples_ms"]) for block in item["blocks"]], [3, 3, 1, 3, 3, 1])
            self.assertAlmostEqual(item["wall_ms_including_reset"], 19 / 7)
        full = next(pair for pair in case["cross_backend_comparisons"] if pair["comparison"] == "full_chunks_1")
        self.assertAlmostEqual(full["mxfp8_latency_reduction_pct"], 20.0)
        self.assertEqual(len(full["paired_round_mean_reduction_pct"]), 2)
        self.assertEqual(len(case["cross_backend_comparisons"]), 6)
        args.check_only = True
        with (
            patch.object(bench, "_check_variants", return_value={}),
            patch.object(bench.int8_base, "_measure", side_effect=AssertionError("No timing in check-only")),
        ):
            self.assertNotIn("measurements", bench._benchmark_case(variants, torch.ones(16, 64), args))

    def test_offline_report_needs_neither_npu_import_nor_distributed_initialization(self):
        case, _, variants, _ = self._measured_case()
        report = {"world_size": 2, "device": "CPU fixture", "mode": "benchmark", "cases": [case]}
        args = bench._parse_args([])
        args.report_json = MagicMock()
        args.report_json.read_text.return_value = bench.json.dumps(report)
        output = io.StringIO()
        with (
            patch.object(bench, "_parse_args", return_value=args),
            patch.object(bench.importlib, "import_module", side_effect=AssertionError("Offline mode")),
            patch.object(bench.dist, "init_process_group", side_effect=AssertionError("Offline mode")),
            redirect_stdout(output),
        ):
            bench.main()
        for label in ("mean ms", "p50 ms", "p95 ms", "p99 ms", "CV %", "paired round", "best_observed_p50"):
            self.assertIn(label, output.getvalue())
        for variant in variants:
            self.assertIn(variant.name, output.getvalue())

    def test_multirank_both_algorithms_share_collective_order_and_independent_references(self):
        for size in (2, 4, 8):
            with self.subTest(world_size=size):
                engine = _MixedCollectives(size)

                def worker(rank, engine=engine, size=size):
                    engine.local.rank = rank
                    try:
                        variants = bench._build_variants(size, ["int8", "mxfp8"], [2, 4], _ops())
                        rows, hidden = size * 9, 64
                        source = (
                            ((torch.arange(rows * hidden).reshape(rows, hidden) + rank * 3) % 31 - 15).float() / 8
                        ).to(torch.bfloat16)
                        source[-1, 0] = 8 * (rank + 1)
                        source[:, 1] = 0
                        saved = source.clone()
                        accuracy = bench._check_variants(variants, source)
                        self.assertEqual(set(accuracy), {variant.name for variant in variants})
                        for name, item in accuracy.items():
                            self.assertTrue(item["pass"])
                            self.assertTrue(math.isfinite(item["relative_l2_vs_fp32"]))
                            if name != "bf16":
                                self.assertEqual(item["same_payload_max_abs"], 0)
                                self.assertEqual(item["max_abs_vs_full"], 0)
                        for variant in variants:
                            if variant.backend == "mxfp8":
                                bench.mx_pipeline._check_encoded(variant.op, rank, rows, hidden, source.device)
                        torch.testing.assert_close(source, saved, rtol=0, atol=0)
                    except BaseException as error:
                        engine.abort(error)
                        raise

                with (
                    patch.object(bench.dist, "all_reduce", side_effect=engine.all_reduce),
                    patch.object(bench.dist, "all_to_all_single", side_effect=engine.all_to_all_single),
                    patch.object(bench.dist, "all_gather_into_tensor", side_effect=engine.all_gather_into_tensor),
                    patch.object(bench.dist, "new_group", side_effect=AssertionError("No extra groups")),
                    ThreadPoolExecutor(max_workers=size) as executor,
                ):
                    futures = [executor.submit(worker, rank) for rank in range(size)]
                    for future in futures:
                        future.result(timeout=30)
                self.assertEqual(len(set(engine.sequences)), 1)
                self.assertEqual(engine.completed, engine.sequences[0])
                for entries in engine.calls.values():
                    self.assertTrue(all(entry[3].waited for entry in entries.values()))


if __name__ == "__main__":
    unittest.main()
