# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for MXFP8 staging dtypes; no CANN/vLLM imports required."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

_SCRIPT = Path(__file__).resolve().parents[4] / "benchmarks/benchmark_mxfp8_all_reduce.py"
_SPEC = importlib.util.spec_from_file_location("mxfp8_alltoall_benchmark", _SCRIPT)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


class TestMxfp8AllToAllBenchmark(unittest.TestCase):
    def test_default_bf16_and_explicit_fp32_arguments(self):
        self.assertEqual(bench._parse_args([]).dequant_dtype, "bf16")
        self.assertEqual(bench._parse_args(["--dequant-dtype", "fp32"]).dequant_dtype, "fp32")
        with self.assertRaises(SystemExit):
            bench._parse_args(["--dequant-dtype", "fp16"])
        with self.assertRaisesRegex(ValueError, "Dequantization dtype"):
            bench.Mxfp8AllToAllAllReduce(2, MagicMock(), MagicMock(), dequant_dtype=torch.float16)

    @staticmethod
    def _quantize_unit_scale(source, *, axis, dst_type, block_size, round_mode):
        scales = torch.full((source.shape[0], source.shape[1] // 64, 2), 127, dtype=torch.uint8)
        return source.to(dst_type), scales

    @staticmethod
    def _dequantize(values, scales, *, axis, dst_type, src_type):
        # Independent torch FP8 conversion and power-of-two scales on CPU.
        rows, hidden = values.shape
        groups = values.float().reshape(rows, hidden // 32, 32)
        factors = torch.exp2(scales.float().reshape(rows, hidden // 32, 1) - 127)
        dtype = torch.bfloat16 if dst_type == 15 else torch.float32
        return (groups * factors).reshape(rows, hidden).to(dtype)

    def test_dequantization_passes_selected_operator_dtype(self):
        values = torch.ones(4, 64).to(torch.float8_e4m3fn)
        scales = torch.full((4, 2), 128, dtype=torch.uint8)
        for dtype, tag in ((torch.bfloat16, 15), (torch.float32, 6)):
            with self.subTest(dtype=dtype):
                dequantize = MagicMock(side_effect=self._dequantize)
                pipeline = bench.Mxfp8AllToAllAllReduce(2, MagicMock(), dequantize, dequant_dtype=dtype)
                decoded = pipeline.dequantize(values, scales)
                self.assertEqual(dequantize.call_args.kwargs["dst_type"], tag)
                self.assertEqual(dequantize.call_args.kwargs["src_type"], 292)
                self.assertEqual(dequantize.call_args.args[1].shape, (4, 1, 2))
                torch.testing.assert_close(decoded, torch.full((4, 64), 2.0, dtype=dtype), rtol=0, atol=0)

    def test_finish_sums_distinct_rank_scales_and_gathers_bf16(self):
        # The owner receives two rows from each source rank. Independent
        # expected sums: 1*1 + (-1)*4 = -3; 0.5*2 + 1.5*0.5 = 1.75.
        values = torch.tensor([1.0, 0.5, -1.0, 1.5]).unsqueeze(1).expand(4, 64).to(torch.float8_e4m3fn)
        scales = torch.tensor([127, 128, 129, 126], dtype=torch.uint8).unsqueeze(1).expand(4, 2)
        expected_partial = torch.tensor([-3.0, 1.75], dtype=torch.bfloat16).unsqueeze(1).expand(2, 64)
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                pipeline = bench.Mxfp8AllToAllAllReduce(2, MagicMock(), self._dequantize, dequant_dtype=dtype)
                received, scale_dtype = pipeline.pack_payload(values, scales)
                output = torch.empty(4, 64, dtype=torch.bfloat16)

                def gather(target, partial, group=None):
                    torch.testing.assert_close(partial, expected_partial, rtol=0, atol=0)
                    target.copy_(partial.repeat(2, 1))

                with patch.object(bench.dist, "all_gather_into_tensor", side_effect=gather):
                    self.assertIs(pipeline.finish_received(received, scale_dtype, output), output)
                torch.testing.assert_close(output, expected_partial.repeat(2, 1), rtol=0, atol=0)

    @staticmethod
    def _identical_rank_all_reduce(tensor, op=None, group=None):
        if op != bench.dist.ReduceOp.MAX:
            tensor.mul_(2)

    @staticmethod
    def _identical_rank_exchange(received, send, group=None):
        received.copy_(send[: send.shape[0] // 2].repeat(2, 1))

    @staticmethod
    def _identical_rank_gather(output, partial, group=None):
        output.copy_(partial.repeat(2, 1))

    def test_correctness_uses_independent_fp32_reference_for_both_modes(self):
        for dtype in (torch.bfloat16, torch.float32):
            for value in (0.0, 1.0):
                with self.subTest(dtype=dtype, value=value):
                    pipeline = bench.Mxfp8AllToAllAllReduce(
                        2, self._quantize_unit_scale, self._dequantize, dequant_dtype=dtype
                    )
                    source = torch.full((4, 64), value, dtype=torch.bfloat16)
                    with (
                        patch.object(bench.dist, "all_reduce", side_effect=self._identical_rank_all_reduce),
                        patch.object(bench.dist, "all_to_all_single", side_effect=self._identical_rank_exchange),
                        patch.object(bench.dist, "all_gather_into_tensor", side_effect=self._identical_rank_gather),
                    ):
                        accuracy = bench._check(pipeline, source, exact=True)
                    self.assertTrue(accuracy["dequantization"]["pass"])
                    self.assertTrue(accuracy["same_payload_collective"]["pass"])
                    self.assertTrue(accuracy["full_path"]["pass"])
                    self.assertEqual(accuracy["mxfp8_relative_l2_vs_fp32"], 0)

    def test_wrong_dequantization_dtype_is_rejected(self):
        def wrong_dtype(*args, **kwargs):
            return self._dequantize(*args, **kwargs).float()

        pipeline = bench.Mxfp8AllToAllAllReduce(2, self._quantize_unit_scale, wrong_dtype)
        with (
            patch.object(bench.dist, "all_reduce", side_effect=self._identical_rank_all_reduce),
            self.assertRaisesRegex(AssertionError, "Unexpected npu_anti_mx_quant"),
        ):
            bench._check(pipeline, torch.ones(4, 64, dtype=torch.bfloat16))

    def test_new_and_legacy_correctness_reports(self):
        check = {"pass": True, "max_abs_vs_dequantized_sum": 0.0}
        accuracy = {
            "dequantization": check,
            "same_payload_collective": check,
            "full_path": check,
            "mxfp8_relative_l2_vs_fp32": 0.01,
            "quantization_relative_l2_vs_fp32": 0.01,
            "bf16_relative_l2_vs_fp32": 0.001,
            "mxfp8_max_abs_vs_fp32": 0.1,
        }
        report = {
            "world_size": 2,
            "device": "CPU fixture",
            "mode": "correctness_only",
            "dequant_dtype": "bf16",
            "sum_output_dtype": "bf16",
            "cases": [{"shape": [4, 64], "accuracy": accuracy}],
        }
        self.assertIn("反量化类型=bf16", bench.format_report(report))
        report.pop("dequant_dtype")
        report.pop("sum_output_dtype")
        accuracy["dequant_fp32"] = accuracy.pop("dequantization")
        self.assertIn("反量化类型=fp32", bench.format_report(report))


if __name__ == "__main__":
    unittest.main()
