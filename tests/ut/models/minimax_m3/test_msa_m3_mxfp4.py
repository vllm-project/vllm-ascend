# SPDX-License-Identifier: Apache-2.0
"""Focused CPU tests for the actual packing adapter; device math is tested separately."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ADAPTER_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/models/minimax_m3/ops/msa_m3_mxfp4.py"


def quantize_stub(matrix, **kwargs):
    n, d = matrix.shape
    return (
        torch.arange(n * (d // 2), dtype=torch.int64).to(torch.uint8).reshape(n, d // 2),
        torch.arange(n * (d // 32), dtype=torch.int64).to(torch.uint8).reshape(n, d // 64, 2),
    )


class PackingTests(unittest.TestCase):
    def setUp(self):
        cache_module = types.ModuleType("vllm_ascend.ops.triton.msa.msa_m3_mxfp4_cache")
        cache_module.scatter_mxfp4_cache = mock.Mock()
        spec = importlib.util.spec_from_file_location("test_mxfp4_adapter", ADAPTER_PATH)
        self.adapter = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {cache_module.__name__: cache_module}):
            spec.loader.exec_module(self.adapter)

    def test_shapes_and_group_order(self):
        for width in [64, 128]:
            with (
                self.subTest(width=width),
                mock.patch.object(
                    self.adapter.torch_npu, "npu_dynamic_mx_quant", side_effect=quantize_stub, create=True
                ) as quant,
            ):
                source = torch.ones((7, 3, width), dtype=torch.bfloat16)
                packed, scales = self.adapter.pack_mxfp4(source)
                self.assertEqual(tuple(packed.shape), (7, 3, width // 2))
                self.assertEqual(tuple(scales.shape), (7, 3, width // 32))
                self.assertEqual(scales.flatten().tolist(), torch.arange(scales.numel()).to(torch.uint8).tolist())
                self.assertEqual(quant.call_args.kwargs["block_size"], 32)
                self.assertEqual(quant.call_args.kwargs["dst_type"], self.adapter.torch_npu.float4_e2m1fn_x2)
                self.assertEqual(quant.call_args.kwargs["scale_alg"], 0)

    def test_strided_projection_is_made_contiguous(self):
        with mock.patch.object(
            self.adapter.torch_npu, "npu_dynamic_mx_quant", side_effect=quantize_stub, create=True
        ) as quant:
            self.adapter.pack_mxfp4(torch.ones((3, 256), dtype=torch.bfloat16)[:, ::2])
            self.assertTrue(quant.call_args.args[0].is_contiguous())

    def test_empty_batch_does_not_launch_quantization(self):
        with mock.patch.object(self.adapter.torch_npu, "npu_dynamic_mx_quant", create=True) as quant:
            packed, scales = self.adapter.pack_mxfp4(torch.empty((0, 1, 128), dtype=torch.bfloat16))
            self.assertEqual(tuple(packed.shape), (0, 1, 64))
            self.assertEqual(tuple(scales.shape), (0, 1, 4))
            quant.assert_not_called()

    def test_rejects_unsupported_logical_width(self):
        with self.assertRaises(ValueError):
            self.adapter.pack_mxfp4(torch.zeros((1, 96), dtype=torch.bfloat16))

    def test_rejects_fp8_before_native_quantization(self):
        with self.assertRaises(TypeError):
            self.adapter.pack_mxfp4(torch.zeros((1, 128), dtype=torch.float32))

    def test_cache_update_uses_both_managed_buffers(self):
        key = torch.zeros((2, 128, 64), dtype=torch.uint8)
        scale = torch.zeros((2, 128, 2, 2), dtype=torch.uint8)
        slots = torch.tensor([0, 129, -1])
        with (
            mock.patch.object(self.adapter.torch_npu, "npu_dynamic_mx_quant", side_effect=quantize_stub, create=True),
            mock.patch.object(self.adapter, "scatter_mxfp4_cache") as scatter,
        ):
            self.adapter.insert_mxfp4_index_cache((key, scale), torch.ones((3, 128), dtype=torch.bfloat16), slots)
            self.assertIs(scatter.call_args.args[2], slots)
            self.assertIs(scatter.call_args.args[3], key)
            self.assertIs(scatter.call_args.args[4], scale)

    def test_joint_quantization_selects_query_and_new_key_rows(self):
        for width in (64, 128):
            for heads in (1, 3):
                with self.subTest(width=width, heads=heads):
                    tokens = 4
                    source = torch.ones((tokens * (heads + 1), width), dtype=torch.bfloat16)
                    slots = torch.tensor([0, 129, -1])
                    cache = (torch.empty(2, 128, width // 2), torch.empty(2, 128, width // 32))
                    with (
                        mock.patch.object(
                            self.adapter.torch_npu, "npu_dynamic_mx_quant", side_effect=quantize_stub, create=True
                        ) as quant,
                        mock.patch.object(self.adapter, "scatter_mxfp4_cache") as scatter,
                    ):
                        query, scales = self.adapter.pack_mxfp4_query_and_insert_cache(cache, source, heads, slots)
                    quant.assert_called_once()
                    self.assertEqual(tuple(query.shape), (3, heads, width // 2))
                    self.assertEqual(tuple(scales.shape), (3, heads, width // 32))
                    expected, expected_scales = quantize_stub(source)
                    expected_scales = expected_scales.reshape(source.shape[0], width // 32)
                    self.assertTrue(
                        torch.equal(
                            query.flatten(), expected[: tokens * heads].reshape(tokens, heads, -1)[:3].flatten()
                        )
                    )
                    self.assertTrue(
                        torch.equal(scatter.call_args.args[0], expected[tokens * heads : tokens * heads + 3])
                    )
                    self.assertTrue(
                        torch.equal(scatter.call_args.args[1], expected_scales[tokens * heads : tokens * heads + 3])
                    )
                    self.assertIs(scatter.call_args.args[2], slots)

    def test_joint_empty_input_does_not_quantize_or_insert(self):
        with (
            mock.patch.object(self.adapter.torch_npu, "npu_dynamic_mx_quant", create=True) as quant,
            mock.patch.object(self.adapter, "scatter_mxfp4_cache") as scatter,
        ):
            query, scales = self.adapter.pack_mxfp4_query_and_insert_cache(
                (torch.empty(2, 128, 64), torch.empty(2, 128, 4)),
                torch.empty(8, 128, dtype=torch.bfloat16),
                1,
                torch.empty(0, dtype=torch.int64),
            )
        self.assertEqual(query.shape, (0, 1, 64))
        self.assertEqual(scales.shape, (0, 1, 4))
        quant.assert_not_called()
        scatter.assert_not_called()

    def test_joint_invalid_rows_rejected_before_quantization(self):
        for rows, heads, slots in ((3, 1, 1), (8, 0, 1), (8, 1, 5)):
            with (
                self.subTest(rows=rows, heads=heads, slots=slots),
                mock.patch.object(self.adapter.torch_npu, "npu_dynamic_mx_quant", create=True) as quant,
                self.assertRaises(ValueError),
            ):
                self.adapter.pack_mxfp4_query_and_insert_cache(
                    (None, None), torch.empty(rows, 128, dtype=torch.bfloat16), heads, torch.arange(slots)
                )
            quant.assert_not_called()

    def test_prequantized_query_does_not_run_quantization_again(self):
        ops = types.ModuleType("cann_ops_transformer_a4c4.ops")
        ops.msa_index_score = mock.Mock(return_value=torch.empty((1, 3, 16)))
        packed = torch.zeros(3, 1, 64, dtype=torch.uint8)
        scales = torch.zeros(3, 1, 4, dtype=torch.uint8)
        cache = (torch.zeros(2, 128, 64, dtype=torch.uint8), torch.zeros(2, 128, 2, 2, dtype=torch.uint8))
        with (
            mock.patch.dict(sys.modules, {ops.__name__: ops}),
            mock.patch.object(self.adapter, "pack_mxfp4", side_effect=AssertionError("duplicate quantization")),
        ):
            self.adapter.mxfp4_index_score((packed, scales), cache, torch.zeros(1, dtype=torch.int32))
        self.assertEqual(ops.msa_index_score.call_args.args[0].data_ptr(), packed.data_ptr())

    def test_actual_fp4_and_e8m0_operands_reach_score_api(self):
        parent = types.ModuleType("cann_ops_transformer_a4c4")
        parent.__path__ = []
        ops = types.ModuleType("cann_ops_transformer_a4c4.ops")
        ops.msa_index_score = mock.Mock(return_value=torch.empty((1, 3, 16)))
        key = torch.zeros((2, 128, 64), dtype=torch.uint8)
        scales = torch.zeros((2, 128, 2, 2), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        with (
            mock.patch.dict(sys.modules, {parent.__name__: parent, ops.__name__: ops}),
            mock.patch.object(self.adapter.torch_npu, "npu_dynamic_mx_quant", side_effect=quantize_stub, create=True),
        ):
            self.adapter.mxfp4_index_score(
                torch.ones((3, 1, 128), dtype=torch.bfloat16),
                (key, scales),
                torch.zeros(1, dtype=torch.int32),
                layout_key="BBND",
            )
        args = ops.msa_index_score.call_args
        self.assertEqual(args.args[0].dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(args.args[1].dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(tuple(args.args[1].shape), (2, 128, 1, 64))
        self.assertEqual(args.kwargs["query_scale"].dtype, torch.float8_e8m0fnu)
        self.assertEqual(args.kwargs["key_scale"].dtype, torch.float8_e8m0fnu)
        self.assertEqual(tuple(args.kwargs["key_scale"].shape), (2, 128, 1, 4))


if __name__ == "__main__":
    unittest.main()
