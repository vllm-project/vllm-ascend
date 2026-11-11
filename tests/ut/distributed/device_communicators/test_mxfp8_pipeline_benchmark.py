# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU simulation of ordered asynchronous collectives with distinct rank data."""

import importlib.util
import io
import sys
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

_BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"
_BASE_SPEC = importlib.util.spec_from_file_location(
    "pipeline_base_fixture", _BENCHMARKS / "benchmark_mxfp8_all_reduce.py"
)
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)
_SPEC = importlib.util.spec_from_file_location("pipeline_fixture", _BENCHMARKS / "benchmark_mxfp8_pipeline.py")
bench = importlib.util.module_from_spec(_SPEC)
with patch.dict(sys.modules, {"benchmark_mxfp8_all_reduce": base}):
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


class _OrderedCollectives:
    """One FIFO HCCL stream. Weak references expose prematurely freed buffers."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.condition = threading.Condition()
        self.local = threading.local()
        self.calls = {}
        self.sequences = [0] * world_size
        self.events = [[] for _ in range(world_size)]
        self.completed = 0
        self.error = None

    def abort(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def _collective(self, kind, output, source, group=None, async_op=False):
        rank = self.local.rank
        with self.condition:
            index = self.sequences[rank]
            self.sequences[rank] += 1
            self.events[rank].append((kind, index))
            work = _Work(self, index, rank)
            calls = self.calls.setdefault(index, {})
            calls[rank] = (kind, weakref.ref(output), weakref.ref(source), work)
            while len(self.calls.get(self.completed, {})) == self.world_size:
                batch = self.calls[self.completed]
                ordered = [batch[owner] for owner in range(self.world_size)]
                if len({entry[0] for entry in ordered}) != 1:
                    raise AssertionError("Collective ordering differs across ranks")
                outputs = [entry[1]() for entry in ordered]
                sources = [entry[2]() for entry in ordered]
                if any(tensor is None for tensor in outputs + sources):
                    raise AssertionError("Async collective buffer was released before completion")
                for owner, target in enumerate(outputs):
                    if ordered[0][0] == "a2a":
                        target.copy_(torch.cat([tensor.chunk(self.world_size)[owner] for tensor in sources]))
                    else:
                        target.copy_(torch.cat(sources))
                self.completed += 1
                self.condition.notify_all()
        if not async_op:
            work.wait()
            return None
        return work

    def all_to_all(self, output, source, **kwargs):
        return self._collective("a2a", output, source, **kwargs)

    def all_gather(self, output, source, **kwargs):
        return self._collective("gather", output, source, **kwargs)


class TestMxfp8PipelineBenchmark(unittest.TestCase):
    def test_chunk_ranges_keep_equal_collective_splits_and_cover_tail(self):
        self.assertEqual(bench.chunk_ranges(72, 8, 4), [(0, 24), (24, 40), (40, 56), (56, 72)])
        for rows, tp, chunks in ((71, 8, 4), (16, 8, 4), (0, 2, 2), (16, 2, 0)):
            with self.subTest(rows=rows, tp=tp, chunks=chunks), self.assertRaises(ValueError):
                bench.chunk_ranges(rows, tp, chunks)

    def test_arguments_and_bf16_contract(self):
        self.assertEqual(bench._parse_args([]).chunks, [2, 4])
        for argv in (["--chunks", "1"], ["--chunks", "2", "2"], ["--hidden-sizes", "63"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                bench._parse_args(argv)
        with self.assertRaisesRegex(ValueError, "BF16"):
            bench.ChunkedMxfp8AllToAllAllReduce(
                2, MagicMock(), MagicMock(), chunks=2, pipelined=True, dequant_dtype=torch.float32
            )

    @staticmethod
    def _decode(values, scales, *, axis, dst_type, src_type):
        if dst_type != 15 or src_type != 292:
            raise AssertionError("Wrong MXFP8/BF16 operator dtype tags")
        rows, hidden = values.shape
        factors = torch.exp2(scales.float().reshape(rows, -1, 1) - 127)
        return (values.float().reshape(rows, -1, 32) * factors).reshape(rows, hidden).to(torch.bfloat16)

    def _simulate(self, world_size, chunks, pipelined):
        rows, hidden = world_size * 9, 128
        engine = _OrderedCollectives(world_size)
        fixtures = []
        for rank in range(world_size):
            indices = torch.arange(rows * hidden).reshape(rows, hidden)
            raw = ((indices + rank) % 8 + 0x30 + (0x80 if rank % 2 else 0)).to(torch.uint8)
            values = raw.view(torch.float8_e4m3fn)
            scales = ((torch.arange(rows * hidden // 32).reshape(rows, -1) + rank) % 4 + 126).to(torch.uint8)
            decoded = self._decode(values, scales, axis=-1, dst_type=15, src_type=292)
            fixtures.append((values, scales, decoded))
        expected = torch.stack([fixture[2].float() for fixture in fixtures]).sum(0).to(torch.bfloat16)

        def worker(rank):
            engine.local.rank = rank
            values, scales, original = fixtures[rank]

            def quantize(source, **kwargs):
                start = source.storage_offset() // hidden
                end = start + source.shape[0]
                torch.testing.assert_close(source, original[start:end], rtol=0, atol=0)
                return values[start:end], scales[start:end]

            pipeline = bench.ChunkedMxfp8AllToAllAllReduce(
                world_size, quantize, self._decode, chunks=chunks, pipelined=pipelined
            )
            try:
                # Repeated calls test final gather joins before input/output reuse.
                for _ in range(3):
                    output = original.clone()
                    self.assertIs(pipeline(output), output)
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    encoded_output = torch.empty_like(output)
                    pipeline.reduce_quantized(values, scales, encoded_output)
                    torch.testing.assert_close(encoded_output, expected, rtol=0, atol=0)
                    with engine.condition:
                        for batch in engine.calls.values():
                            if rank in batch:
                                self.assertTrue(batch[rank][3].waited)
            except BaseException as error:
                engine.abort(error)
                raise

        with (
            patch.object(bench.dist, "all_to_all_single", side_effect=engine.all_to_all),
            patch.object(bench.dist, "all_gather_into_tensor", side_effect=engine.all_gather),
            patch.object(bench.dist, "all_reduce", side_effect=AssertionError("Candidate must not call all_reduce")),
            ThreadPoolExecutor(max_workers=world_size) as executor,
        ):
            futures = [executor.submit(worker, rank) for rank in range(world_size)]
            for future in futures:
                future.result(timeout=30)
        self.assertEqual(engine.completed, 3 * 2 * chunks * 2)
        kinds = [engine.calls[index][0][0] for index in range(2 * chunks)]
        if pipelined:
            for events in engine.events:
                self.assertEqual(events[:3], [("a2a", 0), ("a2a", 1), ("wait", 0)])
            expected_order = ["a2a", "a2a"]
            for _ in range(chunks - 2):
                expected_order.extend(("gather", "a2a"))
            expected_order.extend(("gather", "gather"))
        else:
            expected_order = ["a2a", "gather"] * chunks
        self.assertEqual(kinds, expected_order)

    def test_multirank_tail_routing_lifetime_and_repeated_calls(self):
        for world_size in (2, 4, 8):
            for chunks in (2, 4):
                for pipelined in (False, True):
                    with self.subTest(tp=world_size, chunks=chunks, pipelined=pipelined):
                        self._simulate(world_size, chunks, pipelined)

    def test_measurements_compare_paired_serial_and_full_controls(self):
        args = bench._parse_args(["--chunks", "2", "--warmup", "0", "--rounds", "2"])
        variants = {name: MagicMock() for name in ("bf16", "mxfp8_full", "serial_chunks_2", "pipeline_chunks_2")}
        durations = {"bf16": 10.0, "mxfp8_full": 8.0, "serial_chunks_2": 12.0, "pipeline_chunks_2": 6.0}
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
            patch.object(bench.base, "_measure", side_effect=measure),
            patch.object(bench.base, "_check", return_value={}),
            patch.object(bench, "_check_chunk_quantization", return_value={"pass": True}),
            patch.object(bench.dist, "get_world_size", return_value=2),
            patch.object(bench.torch, "npu", create=True) as npu,
        ):
            case = bench._benchmark_case(variants, torch.ones(8, 64, dtype=torch.bfloat16), args)
        npu.synchronize.assert_called_once()
        self.assertEqual(calls[:4], list(variants))
        self.assertNotEqual(calls[:4], calls[4:])
        result = case["measurements"]["pipeline_chunks_2"]
        self.assertEqual(result["speedup_vs_serial_chunks"], 2.0)
        self.assertEqual(result["speedup_vs_full"], 8 / 6)
        self.assertEqual(result["speedup_vs_bf16"], 10 / 6)
        self.assertEqual(result["round_medians_ms"], [6.0, 6.0])
        self.assertEqual(result["samples_ms"], [6.0] * 4)
        self.assertEqual(result["p95_ms"], 6.0)

    def test_offline_report_prints_accuracy_and_latency_without_npu(self):
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
                            "mxfp8_relative_l2_vs_fp32": 0.01,
                            "mxfp8_max_abs_vs_fp32": 0.5,
                            "same_payload_collective": {"max_abs_vs_dequantized_sum": 0.0},
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
        path = MagicMock()
        path.read_text.return_value = bench.json.dumps(report)
        args = bench._parse_args([])
        args.report_json = path
        output = io.StringIO()
        with (
            patch.object(bench, "_parse_args", return_value=args),
            patch.object(bench.importlib, "import_module", side_effect=AssertionError("Must not initialize NPU")),
            redirect_stdout(output),
        ):
            bench.main()
        self.assertIn("relative L2", output.getvalue())
        self.assertIn("max abs", output.getvalue())
        self.assertIn("p95 ms", output.getvalue())
        self.assertIn("2.000x", output.getvalue())


if __name__ == "__main__":
    unittest.main()
