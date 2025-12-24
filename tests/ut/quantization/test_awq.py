from types import MappingProxyType
from unittest.mock import ANY, MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearBase

from tests.ut.base import TestBase
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.awq.awq import (AWQLinearAscendMethod,
                                              AWQMoEAscendMethod,
                                              AWQQuantConfig)
from vllm_ascend.utils import AWQ_QUANTIZATION_METHOD


class TestAWQQuantization(TestBase):

    def setUp(self):
        super().setUp()
        self.sample_config = {
            "quant_method": AWQ_QUANTIZATION_METHOD,
            "group_size": 128,
            "bits": 4,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": ["visual"],
        }

        self.awq_quant_config = AWQQuantConfig.from_config(self.sample_config)
        self.awq_quant_config.packed_modules_mapping = MappingProxyType({})

    def test_init(self):
        self.assertEqual(self.awq_quant_config.group_size, 128)
        self.assertEqual(self.awq_quant_config.weight_bits, 4)
        self.assertTrue(self.awq_quant_config.zero_point)
        self.assertEqual(self.awq_quant_config.modules_to_not_convert,
                         ["visual"])

    def test_init_with_invalid_bits(self):
        invalid_config = self.sample_config.copy()
        invalid_config["bits"] = 8
        with self.assertRaises(ValueError):
            AWQQuantConfig.from_config(invalid_config)

    def test_get_name(self):
        self.assertEqual(self.awq_quant_config.get_name(),
                         AWQ_QUANTIZATION_METHOD)

    def test_get_supported_act_dtypes(self):
        supported_dtypes = self.awq_quant_config.get_supported_act_dtypes()
        self.assertIn(torch.float16, supported_dtypes)
        self.assertIn(torch.bfloat16, supported_dtypes)
        self.assertEqual(len(supported_dtypes), 2)

    def test_get_min_capability(self):
        with self.assertRaises(NotImplementedError):
            AWQQuantConfig.get_min_capability()

    def test_get_config_filenames(self):
        filenames = AWQQuantConfig.get_config_filenames()
        self.assertIn("quant_config.json", filenames)
        self.assertIn("quantize_config.json", filenames)
        self.assertEqual(len(filenames), 2)

    def test_from_config(self):
        config = AWQQuantConfig.from_config(self.sample_config)
        self.assertIsInstance(config, AWQQuantConfig)

    def test_get_quant_method_for_linear(self):
        linear_layer = MagicMock(spec=LinearBase)
        # Test skipped layer
        quant_method = self.awq_quant_config.get_quant_method(
            linear_layer, "visual")
        self.assertIsInstance(quant_method, AscendUnquantizedLinearMethod)

        # Test quantized layer
        quant_method = self.awq_quant_config.get_quant_method(
            linear_layer, "attn")
        self.assertIsInstance(quant_method, AWQLinearAscendMethod)

    def test_get_quant_method_for_fused_moe(self):
        fused_moe_layer = MagicMock(spec=FusedMoE)
        fused_moe_config = MagicMock(spec=FusedMoEConfig)
        fused_moe_layer.moe_config = fused_moe_config

        # Test skipped layer
        with patch(
                'vllm_ascend.quantization.awq.awq.AscendUnquantizedFusedMoEMethod',
                return_value=MagicMock()) as mock_ascend_moe:
            quant_method = self.awq_quant_config.get_quant_method(
                fused_moe_layer, "visual")
            self.assertIs(quant_method, mock_ascend_moe.return_value)

        # Test quantized layer
        with patch('vllm_ascend.quantization.awq.awq.AWQMoEAscendMethod',
                   return_value=MagicMock()) as mock_ascend_moe:
            quant_method = self.awq_quant_config.get_quant_method(
                fused_moe_layer, "attn")
            self.assertIs(quant_method, mock_ascend_moe.return_value)


class TestAWQLinearAscendMethod(TestBase):

    def setUp(self):
        super().setUp()
        self.sample_config = {
            "quant_method": AWQ_QUANTIZATION_METHOD,
            "group_size": 128,
            "bits": 4,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": ["visual"],
        }

        self.awq_quant_config = AWQQuantConfig.from_config(self.sample_config)
        self.method = AWQLinearAscendMethod(self.awq_quant_config)

    def test_create_weights(self):
        with patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0), \
            patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1):

            layer = MagicMock(spec=LinearBase)
            self.method.create_weights(
                layer=layer,
                input_size_per_partition=128,
                output_partition_sizes=[64],
                input_size=128,
                output_size=64,
                params_dtype=torch.float16,
            )
            layer.register_parameter.assert_any_call("qweight", ANY)
            layer.register_parameter.assert_any_call("qzeros", ANY)
            layer.register_parameter.assert_any_call("scales", ANY)

    def test_process_weights_after_loading(self):
        layer = MagicMock(spec=LinearBase)
        layer.qweight = torch.randint(10, (64, 128), dtype=torch.int32)
        # AWQ pack order [0 2 4 6 1 3 5 7]
        layer.qweight[0][0] = 0x75316420
        layer.qzeros = torch.randint(
            10, (1, 128 // self.awq_quant_config.group_size),
            dtype=torch.int32)
        # AWQ pack order [0 2 4 6 1 3 5 7]
        layer.qzeros[0][0] = 0x75316420
        layer.scales = torch.randn(1,
                                   128 // self.awq_quant_config.group_size,
                                   dtype=torch.float16)

        self.method.process_weights_after_loading(layer)
        # unpacked and signed number. eg: 0 -> 1000b(-8 in int4) -> 0x8 in uint32
        self.assertEqual(layer.qweight[0][0].to(torch.uint32), 0xFEDCBA98)
        self.assertTrue(
            torch.equal(
                layer.qzeros[0],
                torch.Tensor([8., 7., 6., 5., 4., 3., 2.,
                              1.]).to(torch.float16)))

    def test_apply(self):
        with patch("torch_npu.npu_weight_quant_batchmatmul") as mock_func:
            layer = MagicMock(spec=LinearBase)
            layer.qweight = torch.randint(10, (64, 128), dtype=torch.int32)
            layer.qzeros = torch.randint(
                -8, 8,
                (8, 128 // self.awq_quant_config.group_size)).to(torch.float16)
            layer.scales = torch.randn(1,
                                       128 // self.awq_quant_config.group_size,
                                       dtype=torch.float16)

            x = torch.randn(2, 16, 128, dtype=torch.float16)
            self.method.apply(layer, x)
            mock_func.assert_called_once()


class TestAWQMoEAscendMethod(TestBase):

    def setUp(self):
        super().setUp()
        self.sample_config = {
            "quant_method": AWQ_QUANTIZATION_METHOD,
            "group_size": 128,
            "bits": 4,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": ["visual"],
        }

        self.awq_quant_config = AWQQuantConfig.from_config(self.sample_config)
        self.method = AWQMoEAscendMethod(self.awq_quant_config)

    def test_create_weights(self):
        layer = MagicMock(spec=FusedMoE)
        self.method.create_weights(
            layer,
            num_experts=4,
            hidden_size=256,
            intermediate_size_per_partition=128,
            params_dtype=torch.float16,
        )

        layer.register_parameter.assert_any_call("w13_qweight", ANY)
        layer.register_parameter.assert_any_call("w2_qweight", ANY)
        layer.register_parameter.assert_any_call("w13_scales", ANY)
        layer.register_parameter.assert_any_call("w2_scales", ANY)
        layer.register_parameter.assert_any_call("w13_qzeros", ANY)
        layer.register_parameter.assert_any_call("w2_qzeros", ANY)

    def test_process_weights_after_loading(self):
        layer = MagicMock(spec=FusedMoE)
        layer.register_parameter = lambda name, param: setattr(
            layer, name, param)
        layer.w13_qweight = torch.randint(10, (4, 128, 256), dtype=torch.int32)
        # AWQ pack order [0 2 4 6 1 3 5 7]
        layer.w13_qweight[0][0][0] = 0x75316420
        layer.w13_qzeros = torch.randint(10, (4, 2), dtype=torch.int32)
        # AWQ pack order [0 2 4 6 1 3 5 7]
        layer.w13_qzeros[0][0] = 0x75316420
        layer.w13_scales = torch.randn(4, 2, dtype=torch.float16)

        layer.w2_qweight = torch.randint(10, (4, 256, 128), dtype=torch.int32)
        # AWQ pack order [0 2 4 6 1 3 5 7]
        layer.w2_qweight[0][0][0] = 0x75316420
        layer.w2_qzeros = torch.randint(10, (4, 2), dtype=torch.int32)
        # AWQ pack order [0 2 4 6 1 3 5 7]
        layer.w2_qzeros[0][0] = 0x75316420
        layer.w2_scales = torch.randn(4, 2, dtype=torch.float16)

        self.method.process_weights_after_loading(layer)

        # unpacked and signed number. eg: 0 -> 1000b(-8 in int4) -> 0x8 in uint32
        self.assertEqual(layer.w13_qweight[0][0][0].to(torch.uint32),
                         0xFEDCBA98)
        print(layer.w13_qzeros[0])
        self.assertTrue(
            torch.equal(
                layer.w13_qzeros[0][0],
                torch.Tensor([8., 7., 6., 5., 4., 3., 2.,
                              1.]).to(torch.float16)))

        # unpacked and signed number. eg: 0 -> 1000b(-8 in int4) -> 0x8 in uint32
        self.assertEqual(layer.w2_qweight[0][0][0].to(torch.uint32),
                         0xFEDCBA98)
        self.assertTrue(
            torch.equal(
                layer.w2_qzeros[0][0],
                torch.Tensor([8., 7., 6., 5., 4., 3., 2.,
                              1.]).to(torch.float16)))
