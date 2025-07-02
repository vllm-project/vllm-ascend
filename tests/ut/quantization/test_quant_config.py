import unittest
from unittest.mock import patch, MagicMock

import torch
from vllm.attention.layer import Attention
from vllm_ascend.quantization.quant_config import AscendQuantConfig, AscendLinearMethod, AscendKVCacheMethod, \
    AscendFusedMoEMethod
from vllm_ascend.utils import ASCEND_QUATIZATION_METHOD
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm_ascend.ops.fused_moe import AscendUnquantizedFusedMoEMethod


class TestAscendQuantConfig(unittest.TestCase):
    def setUp(self):
        self.quant_config = AscendQuantConfig(
            {"linear.weight": "FLOAT", "attention.fa_quant_type": "C8", "fused_moe.weight": "W8A8"})

    def test_get_name(self):
        self.assertEqual(self.quant_config.get_name(), ASCEND_QUATIZATION_METHOD)

    def test_get_supported_act_dtypes(self):
        self.assertEqual(self.quant_config.get_supported_act_dtypes(), [torch.int8, torch.float16, torch.bfloat16])

    def test_get_config_filenames(self):
        self.assertEqual(self.quant_config.get_config_filenames(), ["quant_model_description.json"])

    def test_from_config(self):
        config = AscendQuantConfig.from_config({"linear.weight": "FLOAT"})
        self.assertIsInstance(config, AscendQuantConfig)
        self.assertEqual(config.quant_description, {"linear.weight": "FLOAT"})

    def test_override_quantization_method(self):
        with patch('torch.npu.is_available', return_value=True):
            self.assertEqual(self.quant_config.override_quantization_method({}, {}), ASCEND_QUATIZATION_METHOD)
        with patch('torch.npu.is_available', return_value=False):
            self.assertIsNone(self.quant_config.override_quantization_method({}, {}))

    def test_get_quant_method_linear(self):
        layer = MagicMock(spec=LinearBase)
        self.assertIsInstance(self.quant_config.get_quant_method(layer, "linear"), UnquantizedLinearMethod)

    def test_get_quant_method_attention_fa_quant_type(self):
        layer = MagicMock(spec=Attention)
        self.quant_config.quant_description["fa_quant_type"] = "W8A8"
        self.quant_config.quant_description["attention.weight"] = "W8A8"
        self.assertIsInstance(self.quant_config.get_quant_method(layer, "attention"), AscendKVCacheMethod)

    def test_get_quant_method_attention_kv_quant_type(self):
        self.quant_config.quant_description["attention.weight"] = "W8A8"
        self.quant_config.quant_description["kv_quant_type"] = "C8"
        layer = MagicMock(spec=Attention)
        print(self.quant_config.quant_description)
        self.assertIsInstance(self.quant_config.get_quant_method(layer, "attention"), AscendKVCacheMethod)

    def test_get_quant_method_fused_moe(self):
        layer = MagicMock(spec=FusedMoE)
        self.assertIsInstance(self.quant_config.get_quant_method(layer, "fused_moe"), AscendFusedMoEMethod)

    # Edge case
    def test_get_quant_method_unsupported_layer(self):
        layer = MagicMock()
        self.assertIsNone(self.quant_config.get_quant_method(layer, "unsupported"))

    def test_is_layer_skipped_ascend_linear(self):
        self.assertTrue(self.quant_config.is_layer_skipped_ascend("linear"))

    def test_is_layer_skipped_ascend_attention(self):
        self.quant_config.quant_description = {"attention.weight": "FLOAT"}
        self.assertTrue(self.quant_config.is_layer_skipped_ascend("attention"))

    def test_is_layer_skipped_ascend_fused_moe(self):
        self.quant_config.quant_description = {"fused_moe.weight": "FLOAT"}
        self.assertTrue(self.quant_config.is_layer_skipped_ascend("fused_moe"))

    def test_is_layer_skipped_ascend_fused_mapping(self):
        self.quant_config.quant_description = {"fused_moe_1.weight1.weight": "FLOAT",
                                               "fused_moe_1.weight2.weight": "FLOAT"}
        self.assertTrue(
            self.quant_config.is_layer_skipped_ascend("fused_moe_1.weight", {"weight": ["weight1", "weight2"]}))

    def test_is_layer_skipped_ascend_mixed_precision(self):
        self.quant_config.quant_description = {"fused_moe_1.weight1.weight": "FLOAT",
                                               "fused_moe_1.weight2.weight": "INT8"}
        with self.assertRaises(ValueError):
            self.quant_config.is_layer_skipped_ascend("fused_moe_1.weight", {"weight": ["weight1", "weight2"]})


class TestAscendLinearMethod(unittest.TestCase):
    def setUp(self):
        quant_config = AscendQuantConfig({"linear.weight": "W8A8"})
        self.linear_method = AscendLinearMethod(quant_config, "linear", {})

    def test_create_weights_ascend_linear_method(self):
        layer = MagicMock(spec=LinearBase)
        self.linear_method.create_weights(layer, 10, [10], 10, 10, torch.float32)
        layer.register_parameter.assert_called()

    def test_process_weights_after_loading_ascend_linear_method(self):
        layer = MagicMock(spec=LinearBase)
        self.linear_method.quant_method.process_weights_after_loading = MagicMock()
        self.linear_method.process_weights_after_loading(layer)
        self.linear_method.quant_method.process_weights_after_loading.assert_called_once_with(layer)


class TestAscendKVCacheMethod(unittest.TestCase):
    def setUp(self):
        quant_config = AscendQuantConfig({"attention.weight": "C8"})
        self.kv_cache_method = AscendKVCacheMethod(quant_config, "attention")

    def test_create_weights_ascend_kv_cache_method(self):
        layer = MagicMock(spec=Attention)
        layer.num_kv_heads = 16
        layer.head_size = 10
        self.kv_cache_method.create_weights(layer)
        layer.register_parameter.assert_called()

    def test_process_weights_after_loading_ascend_kv_cache_method(self):
        layer = MagicMock(spec=Attention)
        self.kv_cache_method.quant_method.process_weights_after_loading = MagicMock()
        self.kv_cache_method.process_weights_after_loading(layer)
        self.kv_cache_method.quant_method.process_weights_after_loading.assert_called_once_with(layer)


class TestAscendFusedMoEMethod(unittest.TestCase):
    def setUp(self):
        quant_config = AscendQuantConfig({"fused_moe.weight": "W8A8"})
        self.fused_moe_method = AscendFusedMoEMethod(quant_config, "fused_moe", {})

    def test_create_weights_ascend_fused_moe_method(self):
        layer = MagicMock(spec=FusedMoE)
        self.fused_moe_method.create_weights(layer, 10, 10, 10, torch.float32)
        layer.register_parameter.assert_called()

    def test_process_weights_after_loading_ascend_fused_moe_method(self):
        layer = MagicMock(spec=FusedMoE)
        self.fused_moe_method.quant_method.process_weights_after_loading = MagicMock()
        self.fused_moe_method.process_weights_after_loading(layer)
        self.fused_moe_method.quant_method.process_weights_after_loading.assert_called_once_with(layer)
