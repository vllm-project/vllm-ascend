from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import COMPRESSED_TENSORS_W8A8_CONFIG
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod


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

    def _make_mx_quant(self, num_bits=4, dynamic=False):
        from compressed_tensors.quantization import QuantizationType

        mock = MagicMock()
        mock.num_bits = num_bits
        mock.type = QuantizationType.FLOAT
        mock.strategy = "group"
        mock.dynamic = dynamic
        mock.symmetric = True
        mock.group_size = 32
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

    def test_detect_mxfp4_dynamic(self):
        weight = self._make_mx_quant(num_bits=4, dynamic=False)
        input_q = self._make_mx_quant(num_bits=4, dynamic=True)
        result = self.config._detect_quant_type(weight, input_q, "mxfp4-pack-quantized")
        self.assertEqual(result, "W4A4_MXFP4")

    def test_detect_mxfp4_mxfp8_dynamic(self):
        weight = self._make_mx_quant(num_bits=4, dynamic=False)
        input_q = self._make_mx_quant(num_bits=8, dynamic=True)
        result = self.config._detect_quant_type(weight, input_q, "mxfp4-pack-quantized")
        self.assertEqual(result, "W4A8_MXFP")
        self.assertEqual(self.config.quant_description["group_size"], 32)

    def test_mxfp4_mxfp8_dynamic_from_runtime_override(self):
        config = {
            "format": "mixed-precision",
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "targets": ["Linear"],
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "strategy": "group",
                        "group_size": 32,
                        "dynamic": True,
                        "symmetric": True,
                    },
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "group",
                        "group_size": 32,
                        "dynamic": False,
                        "symmetric": True,
                    },
                }
            },
        }
        parsed = AscendCompressedTensorsConfig.from_config(config)
        scheme = parsed.target_scheme_map["Linear"]
        self.assertEqual(scheme["input_activations"].num_bits, 8)
        result = parsed._detect_quant_type(scheme["weights"], scheme["input_activations"], scheme["format"])
        self.assertEqual(result, "W4A8_MXFP")

    def test_mxfp4_mxfp8_dynamic_rejects_incompatible_formats(self):
        for quant_format in ("int-quantized", "mxfp8-quantized"):
            with self.subTest(format=quant_format), self.assertRaises(NotImplementedError):
                self.config._detect_quant_type(self._make_mx_quant(), self._make_mx_quant(8, True), quant_format)

    def test_mxfp4_mxfp8_dynamic_rejects_invalid_weight_geometry(self):
        for attribute, value in (("num_bits", 8), ("group_size", 16), ("dynamic", True), ("symmetric", False)):
            weight = self._make_mx_quant()
            setattr(weight, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(NotImplementedError):
                self.config._detect_quant_type(weight, self._make_mx_quant(8, True), "mxfp4-pack-quantized")

    def test_mxfp4_weight_only_does_not_implicitly_select_dynamic_w4a8(self):
        with self.assertRaises(NotImplementedError):
            self.config._detect_quant_type(self._make_mx_quant(), None, "mxfp4-pack-quantized")

    def test_detect_mxfp8_dynamic(self):
        weight = self._make_mx_quant(num_bits=8, dynamic=False)
        input_q = self._make_mx_quant(num_bits=8, dynamic=True)
        result = self.config._detect_quant_type(weight, input_q, "mxfp8-quantized")
        self.assertEqual(result, "W8A8_MXFP8")

    def test_mixed_precision_preserves_input_activations(self):
        from compressed_tensors.quantization.quant_scheme import MXFP4

        config = {
            "format": "mixed-precision",
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "targets": ["Linear"],
                    **MXFP4,
                }
            },
        }
        scheme_map = self.config._quantization_scheme_map_from_config(config)
        self.assertIsNotNone(scheme_map["Linear"]["input_activations"])

    def test_detect_unsupported_raises(self):
        weight = self._make_weight_quant(num_bits=2, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=2, strategy="tensor", dynamic=False, symmetric=True)
        with self.assertRaises(NotImplementedError):
            self.config._detect_quant_type(weight, input_q, "int_quantized")


class TestAscendCompressedTensorsConfigGetQuantMethod(TestBase):
    def setUp(self):
        self.config = AscendCompressedTensorsConfig.from_config(COMPRESSED_TENSORS_W8A8_CONFIG)

    @patch("vllm_ascend.quantization.method_adapters.AscendLinearMethod.__init__")
    def test_get_linear_quant_method(self, mock_method):
        mock_method.return_value = None
        layer = MagicMock(spec=RowParallelLinear)
        result = self.config.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        self.assertTrue(isinstance(result, AscendLinearMethod))
        self.assertTrue(isinstance(layer.scheme, AscendW8A8DynamicLinearMethod))

    def test_get_linear_unquantized_method(self):
        layer = MagicMock(spec=RowParallelLinear)
        result = self.config.get_quant_method(layer, "lm_head")
        self.assertTrue(isinstance(result, UnquantizedLinearMethod))

    def test_adds_routed_experts_target_for_linear_scheme(self):
        linear_scheme = self.config.target_scheme_map["Linear"]

        self.config._add_fused_moe_to_target_scheme_map()

        self.assertIs(self.config.target_scheme_map["RoutedExperts"], linear_scheme)

    @patch("vllm_ascend.quantization.configs.compressed_tensors_config.find_matched_target", return_value=None)
    def test_get_scheme_dict_returns_none_for_unmatched_target(self, _mock_find_target):
        layer = MagicMock(spec=Attention)

        result = self.config.get_scheme_dict(layer, "model.layers.0.self_attn.attn")

        self.assertIsNone(result)

    @patch(
        "vllm_ascend.quantization.configs.compressed_tensors_config.find_matched_target",
        return_value="Linear",
    )
    def test_get_scheme_dict_returns_none_for_none_scheme(self, _mock_find_target):
        self.config.target_scheme_map["Linear"] = None
        layer = MagicMock(spec=Attention)

        result = self.config.get_scheme_dict(layer, "model.layers.0.self_attn.attn")

        self.assertIsNone(result)

    @patch("vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod")
    def test_get_routed_experts_quant_method(self, mock_method):
        layer = RoutedExperts.__new__(RoutedExperts)
        torch.nn.Module.__init__(layer)
        layer.moe_config = MagicMock()
        moe_scheme = MagicMock()

        with patch.object(self.config, "_get_moe_scheme", return_value=moe_scheme):
            result = self.config.get_quant_method(layer, "model.layers.0.mlp.experts")

        self.assertIs(result, mock_method.return_value)
        self.assertIs(layer.scheme, moe_scheme)
        mock_method.assert_called_once_with(moe_scheme, layer.moe_config)

    def test_no_quant_method(self):
        layer = MagicMock(spec=Attention)
        result = self.config.get_quant_method(layer, "attn")
        self.assertIsNone(result)
