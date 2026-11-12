# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered multi-rank simulation and graph-routing checks for INT8 pipelining."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import torch

from tests.ut.distributed.device_communicators import test_int8_pipeline_benchmark as benchmark_tests
from tests.ut.distributed.device_communicators.test_quantized_all_reduce import _make
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.distributed.device_communicators import quantized_all_reduce as qar


class TestInt8PipelineRuntime(unittest.TestCase):
    def test_pipeline_settings_and_row_partition(self):
        self.assertEqual(_make(min_numel=512).pipeline_min_numel, 512)
        self.assertEqual(qar._chunk_ranges(15, 4), [(0, 4), (4, 8), (8, 12), (12, 15)])
        for rows, chunks in ((0, 2), (2, 4), (8, 0)):
            with self.subTest(rows=rows, chunks=chunks), self.assertRaises(ValueError):
                qar._chunk_ranges(rows, chunks)
        for chunks in (0, -1, 3, 8):
            with self.subTest(chunks=chunks), self.assertRaisesRegex(ValueError, "pipeline_chunks"):
                _make(pipeline_chunks=chunks)
        with self.assertRaisesRegex(ValueError, "pipeline_min_numel"):
            _make(pipeline_min_numel=0)

    def test_pipeline_threshold_rows_and_graph_capture(self):
        source = torch.ones(8, 64, dtype=torch.bfloat16)
        for device in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5):
            for kwargs, capturing, expected in (
                ({}, False, "disabled"),
                ({"pipeline_chunks": 2, "pipeline_min_numel": 513}, False, "below_pipeline_threshold"),
                ({"pipeline_chunks": 2, "pipeline_min_numel": 512}, False, None),
                ({"pipeline_chunks": 4}, False, None),
                ({"pipeline_chunks": 2}, True, "graph_capture"),
            ):
                with self.subTest(device=device, kwargs=kwargs, capturing=capturing):
                    wrapper = _make(device_type=device, **kwargs)
                    with patch.object(torch.npu, "is_current_stream_capturing", return_value=capturing):
                        self.assertEqual(wrapper._pipeline_skip_reason(source), expected)
        wrapper = _make(pipeline_chunks=4)
        self.assertEqual(wrapper._pipeline_skip_reason(torch.ones(3, 64, dtype=torch.bfloat16)), "insufficient_rows")

    def test_selected_route_and_debug_reason(self):
        source = torch.ones(8, 64, dtype=torch.bfloat16)
        for threshold, capturing, mode, reason in (
            (512, False, "pipeline", "eligible"),
            (513, False, "full", "below_pipeline_threshold"),
            (512, True, "full", "graph_capture"),
        ):
            with self.subTest(threshold=threshold, capturing=capturing):
                wrapper = _make(pipeline_chunks=2, pipeline_min_numel=threshold)
                with (
                    patch.object(torch.npu, "is_current_stream_capturing", return_value=capturing),
                    patch.object(wrapper, "_all_reduce_full", return_value=MagicMock()) as full,
                    patch.object(wrapper, "_all_reduce_pipelined", return_value=MagicMock()) as pipeline,
                    patch.object(qar, "logger") as logger,
                    patch("vllm.forward_context.get_forward_context", side_effect=AssertionError("No phase gate")),
                ):
                    wrapper.all_reduce(source)
                selected, other = (pipeline, full) if mode == "pipeline" else (full, pipeline)
                selected.assert_called_once_with(source)
                other.assert_not_called()
                args = logger.debug.call_args.args
                message = args[0] % args[1:]
                self.assertIn(f"mode={mode}", message)
                self.assertIn(f"pipeline_reason={reason}", message)

    def test_capture_retains_synchronous_full_int8_calls(self):
        wrapper = _make(world_size=2, pipeline_chunks=4, min_numel=1)
        source = torch.ones(9, 64, dtype=torch.bfloat16)

        def reduce(tensor, op=None, group=None, **kwargs):
            self.assertNotIn("async_op", kwargs)
            if op != qar.dist.ReduceOp.MAX:
                tensor.mul_(2)

        def quantize(x, scale, offset, dtype, axis, div_mode):
            return (x.float() / scale).round().to(dtype)

        with (
            patch.object(torch.npu, "is_current_stream_capturing", return_value=True),
            patch.object(qar.dist, "all_reduce", side_effect=reduce) as communication,
            patch.object(qar.torch_npu, "npu_quantize", side_effect=quantize, create=True),
            patch.object(
                qar.torch_npu,
                "npu_anti_quant",
                side_effect=lambda q, s, dst_dtype: (q.float() * s).to(dst_dtype),
                create=True,
            ),
        ):
            output = wrapper.all_reduce(source)
        self.assertEqual(communication.call_count, 2)
        torch.testing.assert_close(output, torch.full_like(source, 2), rtol=0, atol=0)
        torch.testing.assert_close(source, torch.ones_like(source), rtol=0, atol=0)

    def test_ineligible_payloads_skip_pipeline_and_capture_query(self):
        for source, size, threshold in (
            (torch.ones(8, 64, dtype=torch.bfloat16), 16, 1),
            (torch.ones(8, 64, dtype=torch.float16), 2, 1),
            (torch.ones(8, 64, dtype=torch.bfloat16), 2, 513),
            (torch.ones(8, 32, dtype=torch.bfloat16), 2, 1),
        ):
            with self.subTest(size=size, shape=source.shape, dtype=source.dtype):
                wrapper = _make(world_size=size, min_numel=threshold, pipeline_chunks=2)
                with (
                    patch.object(qar.dist, "all_reduce") as plain,
                    patch.object(wrapper, "_global_scale", side_effect=AssertionError("Must not quantize")),
                    patch.object(torch.npu, "is_current_stream_capturing", side_effect=AssertionError("No query")),
                ):
                    self.assertIs(wrapper.all_reduce(source), source)
                plain.assert_called_once_with(source, group=wrapper.group)

    def test_multirank_scale_repeated_calls_and_noncontiguous_input(self):
        for size in (2, 4, 8):
            for chunks in (2, 4):
                for noncontiguous in (False, True):
                    with self.subTest(size=size, chunks=chunks, noncontiguous=noncontiguous):
                        self._run_ranks(size, chunks, noncontiguous)

    def test_same_sign_boundaries_and_zeros(self):
        for size in (2, 4, 8):
            for chunks in (2, 4):
                for value in (0.0, 1.0, -1.0):
                    with self.subTest(size=size, chunks=chunks, value=value):
                        self._run_ranks(size, chunks, False, value)

    def test_wrong_dequantization_metadata_is_rejected(self):
        source = torch.ones(4, 64, dtype=torch.bfloat16)
        for wrong in (torch.ones(2, 64), torch.ones(1, 64, dtype=torch.bfloat16)):
            with self.subTest(dtype=wrong.dtype, shape=wrong.shape):
                wrapper = _make(pipeline_chunks=2, min_numel=1)
                with (
                    patch.object(torch.npu, "is_current_stream_capturing", return_value=False),
                    patch.object(qar.dist, "all_reduce", return_value=MagicMock()),
                    patch.object(
                        qar.torch_npu, "npu_quantize", return_value=torch.ones(2, 64, dtype=torch.int8), create=True
                    ),
                    patch.object(qar.torch_npu, "npu_anti_quant", return_value=wrong, create=True),
                    self.assertRaisesRegex(RuntimeError, "BF16"),
                ):
                    wrapper.all_reduce(source)

    def _run_ranks(self, size, chunks, noncontiguous, value=None):
        sources = benchmark_tests.TestInt8PipelineBenchmark._sources(size)
        if value is not None:
            sources = [torch.full_like(source, value) for source in sources]
        scale = (torch.stack(sources).float().abs().amax(dim=(0, 1)) / (127 // size)).clamp_min(1e-8)
        encoded = [(source.float() / scale).round().to(torch.int8) for source in sources]
        expected = (torch.stack(encoded).sum(dim=0, dtype=torch.int32).float() * scale).to(torch.bfloat16)
        wrappers = [_make(world_size=size, device_type=AscendDeviceType.A5, pipeline_chunks=chunks) for _ in sources]
        engine = benchmark_tests._OrderedAllReduce(size)

        def reduce(tensor, **kwargs):
            self.assertIs(kwargs["group"], wrappers[engine.local.rank].group)
            if tensor.dtype == torch.int8:
                self.assertTrue(kwargs["async_op"])
            else:
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertEqual(kwargs["op"], qar.dist.ReduceOp.MAX)
            return engine.all_reduce(tensor, **kwargs)

        def quantize(source, shared_scale, offset, dtype, axis, div_mode):
            torch.testing.assert_close(shared_scale, scale, rtol=0, atol=0)
            self.assertEqual((dtype, axis, div_mode, offset), (torch.int8, 1, True, None))
            return (source.float() / shared_scale).round().to(dtype)

        def dequantize(quantized, shared_scale, dst_dtype):
            self.assertEqual(dst_dtype, torch.bfloat16)
            return (quantized.float() * shared_scale).to(dst_dtype)

        def worker(rank):
            engine.local.rank = rank
            source = sources[rank].clone().reshape(3, 5, 64)
            if noncontiguous:
                source = source.transpose(0, 1).contiguous().transpose(0, 1)
                self.assertFalse(source.is_contiguous())
            try:
                for _ in range(3):
                    output = wrappers[rank].all_reduce(source)
                    self.assertNotEqual(output.data_ptr(), source.data_ptr())
                    self.assertEqual(output.shape, source.shape)
                    self.assertTrue(output.is_contiguous())
                    torch.testing.assert_close(output.reshape(15, 64), expected, rtol=0, atol=0)
                    torch.testing.assert_close(source.reshape(15, 64), sources[rank], rtol=0, atol=0)
                    with engine.condition:
                        for batch in engine.calls.values():
                            if rank in batch:
                                self.assertTrue(batch[rank][2].waited)
            except BaseException as error:
                engine.abort(error)
                raise

        with (
            patch.object(torch.npu, "is_current_stream_capturing", return_value=False),
            patch.object(torch.npu, "synchronize", side_effect=AssertionError("No device synchronization")),
            patch.object(qar.dist, "new_group", side_effect=AssertionError("Must reuse the existing group")),
            patch.object(qar.dist, "all_reduce", side_effect=reduce),
            patch.object(qar.torch_npu, "npu_quantize", side_effect=quantize, create=True),
            patch.object(qar.torch_npu, "npu_anti_quant", side_effect=dequantize, create=True),
            ThreadPoolExecutor(max_workers=size) as pool,
        ):
            futures = [pool.submit(worker, rank) for rank in range(size)]
            for future in futures:
                future.result(timeout=30)
        self.assertEqual(engine.completed, 3 * (1 + chunks))
        for events in engine.events:
            self.assertEqual(events[:5], [("MAX", 0), ("wait", 0), ("SUM", 1), ("SUM", 2), ("wait", 1)])


if __name__ == "__main__":
    unittest.main()
