import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.quantization.methods.base import PreparedLinearInput
from vllm_ascend.quantization.methods.w4a8.w4a8_linear import AscendW4A8DynamicLinearMethod

MODULE = "vllm_ascend.quantization.methods.w4a8.w4a8_linear"


class TestAscendW4A8DynamicLinearMethod(unittest.TestCase):
    def setUp(self):
        self.quant_config = SimpleNamespace(quant_description={}, get_name=lambda: "ascend")
        self.config_patch = patch(
            MODULE + ".get_current_vllm_config", return_value=SimpleNamespace(quant_config=self.quant_config)
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.method = AscendW4A8DynamicLinearMethod()

    def test_reject_unsupported_checkpoint_formats(self):
        self.quant_config.quant_description = {"group_size": 32}
        with self.assertRaisesRegex(ValueError, "per-channel"):
            AscendW4A8DynamicLinearMethod()
        self.quant_config.quant_description = {}
        self.quant_config.get_name = lambda: "compressed-tensors"
        with self.assertRaisesRegex(ValueError, "ModelSlim"):
            AscendW4A8DynamicLinearMethod()

    def test_packed_weight_shape_and_alignment(self):
        params = self.method.get_weight(16, 80, torch.bfloat16)
        self.assertEqual(params["weight"].shape, (40, 16))
        self.assertEqual(params["weight"].dtype, torch.int8)
        self.assertEqual((params["_packed_dim"], params["_packed_factor"]), (0, 2))
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            self.method.get_weight(16, 79, torch.bfloat16)

    def test_row_parallel_scale_bias_partitions(self):
        with patch(MODULE + ".get_tensor_model_parallel_world_size", return_value=8):
            params = self.method.get_pergroup_param(16, 80, torch.bfloat16, "row")
        self.assertEqual(params["scale_bias"].shape, (80, 2))
        self.assertEqual(params["_input_dim"], 1)
        with (
            patch(MODULE + ".get_tensor_model_parallel_world_size", return_value=3),
            self.assertRaisesRegex(ValueError, "divide 16"),
        ):
            self.method.get_pergroup_param(16, 80, torch.bfloat16, "row")

    def test_nz_conversion_alignment_and_scale_bias_reduction(self):
        for input_size, output_size, use_nz in [(16, 64, True), (16, 80, False), (8, 64, False)]:
            with self.subTest(input_size=input_size, output_size=output_size):
                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(
                    torch.zeros(output_size // 2, input_size, dtype=torch.int8), requires_grad=False
                )
                for name, value in {
                    "weight_scale": torch.ones(output_size, 1),
                    "weight_offset": torch.zeros(output_size, 1),
                    "scale_bias": torch.ones(output_size, 2),
                }.items():
                    layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
                with patch(MODULE + ".maybe_trans_nz", side_effect=lambda x: x) as convert:
                    self.method.process_weights_after_loading(layer)
                self.assertEqual(convert.call_count, int(use_nz))
                if use_nz:
                    self.assertEqual(convert.call_args.args[0].shape, (1, input_size, output_size // 2))
                    self.assertIsNone(convert.call_args.args[0]._base)
                self.assertEqual(layer.weight.shape, (1, input_size, output_size // 8))
                self.assertEqual(layer.weight.dtype, torch.int32)
                expected_scale = torch.ones(output_size).view(torch.int32).to(torch.int64)
                torch.testing.assert_close(layer.weight_scale, expected_scale)
                torch.testing.assert_close(layer.scale_bias, torch.full((output_size,), 2.0))

    def test_reject_asymmetric_weight_offsets(self):
        layer = SimpleNamespace(weight_offset=torch.ones(80, 1))
        with self.assertRaisesRegex(ValueError, "symmetric"):
            self.method.process_weights_after_loading(layer)

    def test_empty_input_does_not_call_quantization_or_matmul(self):
        layer = SimpleNamespace(weight_scale=torch.ones(80))
        with (
            patch(MODULE + ".torch_npu.npu_dynamic_quant", create=True) as quant,
            patch(MODULE + ".torch_npu.npu_grouped_matmul", create=True) as matmul,
        ):
            output = self.method.apply(layer, torch.empty(2, 0, 16, dtype=torch.bfloat16))
        self.assertEqual(output.shape, (2, 0, 80))
        self.assertEqual(output.dtype, torch.bfloat16)
        quant.assert_not_called()
        matmul.assert_not_called()

    def test_prepared_input_survives_mutating_matmul(self):
        original = torch.ones(2, 16, dtype=torch.int8)
        prepared = PreparedLinearInput(original.clone(), torch.ones(2), torch.bfloat16)
        layer = SimpleNamespace(weight=torch.zeros(1, 16, 8), weight_scale=torch.ones(64), scale_bias=torch.zeros(64))

        def mutating_matmul(**kwargs):
            kwargs["x"][0].zero_()
            return [torch.ones(2, 64, dtype=torch.bfloat16)]

        with patch(MODULE + ".torch_npu.npu_grouped_matmul", side_effect=mutating_matmul, create=True):
            for _ in range(2):
                output = self.method.apply(layer, prepared, bias=torch.ones(64, dtype=torch.bfloat16))
                torch.testing.assert_close(output, torch.full((2, 64), 2.0, dtype=torch.bfloat16))
                torch.testing.assert_close(prepared.quantized, original)
