from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import COMPRESSED_TENSORS_W8A8_CONFIG
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod
from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.method_adapters import AscendLinearMethod, AscendFusedMoEMethod
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod, AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD


class TestAscendCompressedTensorsConfigBasic(TestBase):

    def test_get_name(self):
        config = AscendCompressedTensorsConfig.from_config(COMPRESSED_TENSORS_W8A8_CONFIG)
        self.assertEqual(config.get_name(), "compressed-tensors")

    def test_get_supported_act_dtypes(self):
        dtypes = AscendCompressedTensorsConfig.get_supported_act_dtypes()
        self.assertIn(torch.int8, dtypes)
        self.assertIn(torch.float16, dtypes)
        self.assertIn(torch.bfloat16, dtypes)
        self.assertEqual(len(dtypes), 3)

    def test_get_min_capability_raises(self):
        with self.assertRaises(NotImplementedError):
            AscendCompressedTensorsConfig.get_min_capability()

    def test_get_config_filenames(self):
        filenames = AscendCompressedTensorsConfig.get_config_filenames()
        self.assertEqual(filenames, [])

    def test_init(self):
        config = AscendCompressedTensorsConfig.from_config(COMPRESSED_TENSORS_W8A8_CONFIG)
        self.assertEqual(config.ignore, ["lm_head"])
        self.assertEqual(config.quant_format, "int-quantized")
        self.assertEqual(list(config.target_scheme_map.keys()), ["Linear"])
        self.assertEqual(config.target_scheme_map["Linear"]["format"], "int-quantized")
        self.assertIsNotNone(config.target_scheme_map["Linear"]["input_activations"])

    def test_apply_vllm_mapper(self):
        hf_to_vllm_mapper = MagicMock()
        config = AscendCompressedTensorsConfig(
            target_scheme_map={"Linear": {}},
            ignore=["lm_head"],
            quant_format="",
        )
        config.apply_vllm_mapper(hf_to_vllm_mapper)
        hf_to_vllm_mapper.apply_dict.assert_called_once()
        hf_to_vllm_mapper.apply_list.assert_called_once()


class TestAscendCompressedTensorsQuanType(TestBase):

    def setUp(self):
        self.config = AscendCompressedTensorsConfig(
            target_scheme_map={"Linear": {}},
            ignore=["lm_head"],
            quant_format="",
            config={},
        )

    def _make_weight_quant(self, num_bits=8, strategy="channel", dynamic=False, symmetric=True, group_size=None):
        mock = MagicMock()
        mock.num_bits = num_bits
        mock.strategy = strategy
        mock.dynamic = dynamic
        mock.symmetric = symmetric
        mock.group_size = group_size
        return mock

    def _make_input_quant(self, num_bits=8, strategy="tensor", dynamic=False, symmetric=True):
        mock = MagicMock()
        mock.num_bits = num_bits
        mock.strategy = strategy
        mock.dynamic = dynamic
        mock.symmetric = symmetric
        return mock

    def test_detect_w8a8_static(self):
        weight = self._make_weight_quant(num_bits=8, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=8, strategy="tensor", dynamic=False, symmetric=True)
        result = self.config._detect_quant_type(weight, input_q, "int-quantized")
        self.assertEqual(result, "W8A8")

    def test_detect_w8a8_dynamic(self):
        weight = self._make_weight_quant(num_bits=8, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=8, strategy="token", dynamic=True, symmetric=True)
        result = self.config._detect_quant_type(weight, input_q, "int-quantized")
        self.assertEqual(result, "W8A8_DYNAMIC")

    def test_detect_w4a8_dynamic(self):
        weight = self._make_weight_quant(num_bits=4, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=8, strategy="token", dynamic=True, symmetric=True)
        result = self.config._detect_quant_type(weight, input_q, "int-quantized")
        self.assertEqual(result, "W4A8_DYNAMIC")

    def test_detect_w4a16(self):
        from compressed_tensors.quantization import QuantizationType
        weight = MagicMock()
        weight.num_bits = 4
        weight.strategy = "group"
        weight.dynamic = False
        weight.type = QuantizationType.INT
        result = self.config._detect_quant_type(weight, None, None)
        self.assertEqual(result, "W4A16")

    def test_detect_unsupported_raises(self):
        weight = self._make_weight_quant(num_bits=2, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=2, strategy="tensor", dynamic=False, symmetric=True)
        with self.assertRaises(NotImplementedError):
            self.config._detect_quant_type(weight, input_q, "int_quantized")


class TestAscendCompressedTensorsConfigGetQuantMethod(TestBase):

    def setUp(self):
        self.config = AscendCompressedTensorsConfig.from_config(COMPRESSED_TENSORS_W8A8_CONFIG)

    def test_get_linear_quant_method(self):
        layer = MagicMock(spec=RowParallelLinear)
        result = self.config.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        self.assertEqual(layer.ascend_quant_method, COMPRESSED_TENSORS_METHOD)
        self.assertTrue(isinstance(result, AscendLinearMethod))
        self.assertTrue(isinstance(layer.scheme, AscendW8A8DynamicLinearMethod))

    def test_get_linear_unquantized_method(self):
        layer = MagicMock(spec=RowParallelLinear)
        result = self.config.get_quant_method(layer, "lm_head")
        self.assertEqual(layer.ascend_quant_method, COMPRESSED_TENSORS_METHOD)
        self.assertTrue(isinstance(result, UnquantizedLinearMethod))

    from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod, AscendW8A8DynamicFusedMoEMethod
    @patch("vllm_ascend.quantization.methods.AscendW8A8DynamicFusedMoEMethod.__init__")
    def test_get_moe_quant_method(self, mock_method):
        mock_method.return_value = None
        mock_method.return_value = MagicMock(spec=AscendW8A8DynamicFusedMoEMethod)
        layer = MagicMock(spec=FusedMoE)
        layer.moe_config = {}
        result = self.config.get_quant_method(layer, "model.layers.0.mlp.experts")
        self.assertEqual(layer.ascend_quant_method, COMPRESSED_TENSORS_METHOD)
        self.assertTrue(isinstance(result, AscendFusedMoEMethod))
        self.assertTrue(isinstance(layer.scheme, AscendW8A8DynamicFusedMoEMethod))

    @patch("vllm_ascend.ops.fused_moe.fused_moe.AscendUnquantizedFusedMoEMethod.__init__")
    @patch("vllm_ascend.quantization.compressed_tensors_config.should_ignore_layer")
    def test_get_moe_unquantized_method(self, mock_ignore_layer, mock_method):
        mock_method.return_value = None
        mock_ignore_layer.return_value = True
        layer = MagicMock(spec=FusedMoE)
        layer.moe_config = {}
        result = self.config.get_quant_method(layer, "model.layers.0.mlp.experts")
        self.assertEqual(layer.ascend_quant_method, COMPRESSED_TENSORS_METHOD)
        self.assertTrue(isinstance(result, AscendUnquantizedFusedMoEMethod))


    def test_no_quant_method(self):
        layer = MagicMock(spec=Attention)
        result = self.config.get_quant_method(layer, "attn")
        self.assertIsNone(result)

