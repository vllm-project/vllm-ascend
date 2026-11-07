# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mocked CPU routing and multi-rank checks for the A5 MXFP8 wrapper."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch_npu

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.distributed.device_communicators import mxfp8_all_reduce as mx
from vllm_ascend.distributed.device_communicators import npu_communicator as comm_module


def _make(world_size=2, min_numel=1, device_type=AscendDeviceType.A5, quantize=None, dequantize=None):
    with (
        patch.object(mx, "get_ascend_device_type", return_value=device_type),
        patch.object(mx.dist, "get_world_size", return_value=world_size),
        patch.object(torch_npu, "npu_dynamic_mx_quant", quantize or MagicMock(), create=True),
        patch.object(torch_npu, "npu_anti_mx_quant", dequantize or MagicMock(), create=True),
    ):
        return mx.Mxfp8AllReduce(MagicMock(), min_numel=min_numel, group_name="tp:0")


def _decode(values, scales):
    rows, hidden = values.shape
    mantissas = values.float().reshape(rows, hidden // 32, 32)
    powers = torch.exp2(scales.float().reshape(rows, hidden // 32, 1) - 127)
    return (mantissas * powers).reshape(rows, hidden)


class TestMxfp8AllReduce(unittest.TestCase):
    def test_requires_a5_and_available_operators(self):
        for device in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType._310P):
            with self.subTest(device=device), self.assertRaisesRegex(RuntimeError, "requires Atlas A5"):
                _make(device_type=device)
        with (
            patch.object(mx, "get_ascend_device_type", return_value=AscendDeviceType.A5),
            patch.object(mx.dist, "get_world_size", return_value=2),
            patch.object(torch_npu, "npu_dynamic_mx_quant", None, create=True),
            patch.object(torch_npu, "npu_anti_mx_quant", None, create=True),
            patch.object(torch.ops.npu, "npu_anti_mx_quant", None, create=True),
            self.assertRaisesRegex(RuntimeError, "requires npu_dynamic_mx_quant and npu_anti_mx_quant"),
        ):
            mx.Mxfp8AllReduce(MagicMock(), min_numel=1)

    def test_registered_anti_quant_entry_is_supported(self):
        registered_op = MagicMock()
        with (
            patch.object(mx, "get_ascend_device_type", return_value=AscendDeviceType.A5),
            patch.object(mx.dist, "get_world_size", return_value=2),
            patch.object(torch_npu, "npu_dynamic_mx_quant", MagicMock(), create=True),
            patch.object(torch_npu, "npu_anti_mx_quant", None, create=True),
            patch.object(torch.ops.npu, "npu_anti_mx_quant", registered_op, create=True),
        ):
            wrapper = mx.Mxfp8AllReduce(MagicMock(), min_numel=1)
        self.assertIs(wrapper.dequantize_op, registered_op)

    def test_fallbacks_use_only_ordinary_all_reduce(self):
        cases = (
            (2, 513, torch.ones(4, 128, dtype=torch.bfloat16), "below_threshold"),
            (2, 1, torch.ones(4, 128, dtype=torch.float16), "dtype"),
            (2, 1, torch.ones(512, dtype=torch.bfloat16), "degenerate_shape"),
            (2, 1, torch.ones(16, 32, dtype=torch.bfloat16), "degenerate_shape"),
            (2, 1, torch.ones(4, 96, dtype=torch.bfloat16), "hidden_alignment"),
            (2, 1, torch.ones(3, 128, dtype=torch.bfloat16), "row_sharding"),
            (2, 1, torch.empty(0, 128, dtype=torch.bfloat16), "below_threshold"),
            (16, 1, torch.ones(16, 128, dtype=torch.bfloat16), "unsupported_group"),
        )
        for size, min_numel, source, reason in cases:
            with self.subTest(size=size, shape=tuple(source.shape), reason=reason):
                wrapper = _make(world_size=size, min_numel=min_numel)
                self.assertEqual(wrapper._skip_reason(source), reason)
                with (
                    patch.object(mx.dist, "all_reduce") as plain,
                    patch.object(mx.dist, "all_to_all_single") as exchange,
                    patch.object(mx.dist, "all_gather_into_tensor") as gather,
                ):
                    self.assertIs(wrapper.all_reduce(source), source)
                plain.assert_called_once_with(source, group=wrapper.group)
                exchange.assert_not_called()
                gather.assert_not_called()
                wrapper.quantize_op.assert_not_called()

    def test_threshold_boundary_and_non_contiguous_layout(self):
        wrapper = _make(min_numel=512)
        source = torch.ones(128, 4, dtype=torch.bfloat16).t()
        self.assertFalse(source.is_contiguous())
        self.assertIsNone(wrapper._skip_reason(source))

    def test_incompatible_quantization_result_fails_before_payload_exchange(self):
        source = torch.ones(4, 128, dtype=torch.bfloat16)
        quantize = MagicMock(return_value=(source, torch.zeros(4, 2, 2, dtype=torch.uint8)))
        wrapper = _make(quantize=quantize)
        with (
            patch.object(mx.dist, "all_to_all_single") as exchange,
            self.assertRaisesRegex(RuntimeError, "Unexpected MXFP8 quantization"),
        ):
            wrapper.all_reduce(source)
        exchange.assert_not_called()

    def test_distinct_rank_scales_and_out_of_place_result_on_tp2_tp4_tp8(self):
        for size in (2, 4, 8):
            with self.subTest(world_size=size):
                self._run_ranks(size)

    def _run_ranks(self, size):
        rows, hidden = 2 * size, 128
        raw_values = [torch.full((rows, hidden), 0x38 + rank, dtype=torch.uint8) for rank in range(size)]
        scale_indices = torch.arange(rows * hidden // 32).reshape(rows, hidden // 64, 2)
        raw_scales = [((scale_indices + rank) % 7 + 124).to(torch.uint8) for rank in range(size)]
        decoded = [_decode(values.view(torch.float8_e4m3fn), scales) for values, scales in zip(raw_values, raw_scales)]
        expected = torch.stack(decoded).sum(0).to(torch.bfloat16).reshape(2, size, hidden)
        state = threading.local()
        exchange_slots = [None] * size
        gather_slots = [None] * size

        def complete_exchange():
            sends = [send.clone() for _, send in exchange_slots]
            for owner, (received, _) in enumerate(exchange_slots):
                received.copy_(torch.cat([send.chunk(size, dim=0)[owner] for send in sends]))

        def complete_gather():
            result = torch.cat([partial.clone() for _, partial in gather_slots])
            for output, _ in gather_slots:
                output.copy_(result)

        exchange_barrier = threading.Barrier(size, action=complete_exchange, timeout=20)
        gather_barrier = threading.Barrier(size, action=complete_gather, timeout=20)

        def quantize(source, **kwargs):
            self.assertEqual(source.dtype, torch.bfloat16)
            self.assertEqual(tuple(source.shape), (rows, hidden))
            self.assertEqual(kwargs["dst_type"], torch.float8_e4m3fn)
            self.assertEqual(kwargs["block_size"], 32)
            return raw_values[state.rank].view(torch.float8_e4m3fn), raw_scales[state.rank]

        def dequantize(values, scales, **kwargs):
            self.assertEqual(kwargs["dst_type"], 15)
            self.assertEqual(kwargs["src_type"], 292)
            return _decode(values, scales).to(torch.bfloat16)

        wrappers = [_make(world_size=size, quantize=quantize, dequantize=dequantize) for _ in range(size)]

        def exchange(received, send, group=None):
            self.assertIs(group, wrappers[state.rank].group)
            self.assertEqual(send.dtype, torch.uint8)
            exchange_slots[state.rank] = (received, send)
            exchange_barrier.wait()

        def gather(output, partial, group=None):
            self.assertIs(group, wrappers[state.rank].group)
            self.assertEqual(partial.dtype, torch.bfloat16)
            gather_slots[state.rank] = (output, partial)
            gather_barrier.wait()

        def worker(rank):
            state.rank = rank
            source = decoded[rank].to(torch.bfloat16).reshape(2, size, hidden)
            original = source.clone()
            output = wrappers[rank].all_reduce(source)
            self.assertIsNot(output, source)
            torch.testing.assert_close(source, original, rtol=0, atol=0)
            torch.testing.assert_close(output, expected, rtol=2**-7, atol=1e-6)

        with (
            patch.object(mx.dist, "all_reduce", side_effect=AssertionError("Unexpected ordinary all-reduce")),
            patch.object(mx.dist, "all_to_all_single", side_effect=exchange),
            patch.object(mx.dist, "all_gather_into_tensor", side_effect=gather),
            ThreadPoolExecutor(max_workers=size) as pool,
        ):
            list(pool.map(worker, range(size)))


class TestQuantizedBackendSelection(unittest.TestCase):
    def test_int8_and_mxfp8_select_separate_backends(self):
        def initialize(comm, cpu_group, device, device_group, unique_name, **kwargs):
            comm.device_group = device_group
            comm.unique_name = unique_name
            comm.world_size = 2

        for dtype in ("int8", "mxfp8"):
            with self.subTest(quant_dtype=dtype):
                config = SimpleNamespace(enable=True, min_numel=512, quant_dtype=dtype)
                wrapper = SimpleNamespace(supported=True, min_numel=512, max_q=63, all_reduce=MagicMock())
                group = MagicMock()
                with (
                    patch.object(comm_module.DeviceCommunicatorBase, "__init__", initialize),
                    patch.object(
                        comm_module,
                        "get_ascend_config",
                        return_value=SimpleNamespace(quantized_all_reduce_config=config),
                    ),
                    patch.object(torch.npu, "current_device", return_value=0, create=True),
                    patch.object(comm_module, "QuantizedAllReduce", return_value=wrapper) as int8,
                    patch.object(comm_module, "Mxfp8AllReduce", return_value=wrapper) as mxfp8,
                ):
                    comm = comm_module.NPUCommunicator(MagicMock(), device_group=group, unique_name="tp:0")
                    source = torch.ones(4, 128, dtype=torch.bfloat16)
                    comm.all_reduce(source)
                selected, other = (int8, mxfp8) if dtype == "int8" else (mxfp8, int8)
                selected.assert_called_once_with(group, min_numel=512, group_name="tp:0")
                other.assert_not_called()
                wrapper.all_reduce.assert_called_once_with(source)


if __name__ == "__main__":
    unittest.main()
