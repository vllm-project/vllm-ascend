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
    MXFP4_FORMATS = (
        ("mxfp4-pack-quantized", None),
        ("pack-quantized", "mxfp4-pack-quantized"),
        ("mixed-precision", "mxfp4-pack-quantized"),
    )

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

    def _make_mxfp4_config(self, quant_format, group_format=None):
        group = {
            "targets": ["Linear"],
            "input_activations": None,
            "weights": {
                "num_bits": 4,
                "type": "float",
                "strategy": "group",
                "group_size": 32,
                "dynamic": False,
                "symmetric": True,
            },
        }
        if group_format is not None:
            group["format"] = group_format
        return {
            "quant_method": "compressed-tensors",
            "format": quant_format,
            "ignore": ["lm_head"],
            "config_groups": {"group_0": group},
        }

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

    def test_native_mxfp4_selects_w4a8(self):
        result = self.config._detect_quant_type(self._make_mx_quant(), None, "mxfp4-pack-quantized")
        self.assertEqual(result, "W4A8_MXFP")
        self.assertEqual(self.config.quant_description["group_size"], 32)

    def test_native_mxfp4_uses_global_format(self):
        self.config.quant_format = "mxfp4-pack-quantized"
        result = self.config._detect_quant_type(self._make_mx_quant(), None, None)
        self.assertEqual(result, "W4A8_MXFP")

    def test_native_mxfp4_from_checkpoint_config(self):
        for quant_format, group_format in self.MXFP4_FORMATS:
            for omit_activations in (False, True):
                with self.subTest(format=quant_format, omit_activations=omit_activations):
                    config = self._make_mxfp4_config(quant_format, group_format)
                    if omit_activations:
                        del config["config_groups"]["group_0"]["input_activations"]
                    parsed = AscendCompressedTensorsConfig.from_config(config)
                    scheme = parsed.target_scheme_map["Linear"]
                    self.assertIsNone(scheme["input_activations"])
                    with patch("vllm_ascend.quantization.methods.get_scheme_class") as get_scheme_class:
                        result = parsed._create_scheme_for_layer_type(scheme["weights"], None, scheme["format"], "moe")
                    get_scheme_class.assert_called_once_with("W4A8_MXFP", "moe")
                    self.assertIs(result, get_scheme_class.return_value.return_value)
                    self.assertEqual(parsed.quant_description["group_size"], 32)

    def test_mxfp4_explicit_activations_from_config(self):
        for quant_format, group_format in self.MXFP4_FORMATS:
            for layer_type in ("linear", "moe"):
                with self.subTest(format=quant_format, layer_type=layer_type):
                    config = self._make_mxfp4_config(quant_format, group_format)
                    group = config["config_groups"]["group_0"]
                    input_activations = {**group["weights"], "dynamic": True}
                    group["input_activations"] = input_activations
                    parsed = AscendCompressedTensorsConfig.from_config(config)
                    scheme = parsed.target_scheme_map["Linear"]
                    self.assertIsNotNone(scheme["input_activations"])
                    self.assertEqual(
                        scheme["input_activations"].model_dump(include=set(input_activations), mode="json"),
                        input_activations,
                    )
                    with patch("vllm_ascend.quantization.methods.get_scheme_class") as get_scheme_class:
                        result = parsed._create_scheme_for_layer_type(
                            scheme["weights"], scheme["input_activations"], scheme["format"], layer_type
                        )
                    get_scheme_class.assert_called_once_with("W4A4_MXFP4", layer_type)
                    self.assertIs(result, get_scheme_class.return_value.return_value)
                    self.assertEqual(parsed.quant_description["group_size"], 32)

    def test_mxfp4_incompatible_activations_from_config(self):
        for quant_format, group_format in self.MXFP4_FORMATS:
            for overrides in (
                {},  # This release does not support explicitly configured MXFP8 activations with MXFP4 weights.
                {"type": "int"},
                {"group_size": 16},
                {"group_size": 64},
                {"dynamic": False},
                {"symmetric": False},
                {"strategy": "token", "group_size": None},
            ):
                with self.subTest(format=quant_format, overrides=overrides):
                    config = self._make_mxfp4_config(quant_format, group_format)
                    group = config["config_groups"]["group_0"]
                    group["input_activations"] = {**group["weights"], "num_bits": 8, "dynamic": True, **overrides}
                    parsed = AscendCompressedTensorsConfig.from_config(config)
                    scheme = parsed.target_scheme_map["Linear"]
                    self.assertIsNotNone(scheme["input_activations"])
                    with (
                        patch("vllm_ascend.quantization.methods.get_scheme_class") as get_scheme_class,
                        self.assertRaises(NotImplementedError),
                    ):
                        parsed._create_scheme_for_layer_type(
                            scheme["weights"], scheme["input_activations"], scheme["format"], "moe"
                        )
                    get_scheme_class.assert_not_called()

    def test_per_group_format_overrides_global_mxfp4(self):
        config = {**COMPRESSED_TENSORS_W8A8_CONFIG, "format": "mxfp4-pack-quantized"}
        parsed = AscendCompressedTensorsConfig.from_config(config)
        scheme = parsed.target_scheme_map["Linear"]
        self.assertEqual(scheme["format"], "int-quantized")
        self.assertIsNotNone(scheme["input_activations"])
        with patch("vllm_ascend.quantization.methods.get_scheme_class") as get_scheme_class:
            parsed._create_scheme_for_layer_type(
                scheme["weights"], scheme["input_activations"], scheme["format"], "linear"
            )
        get_scheme_class.assert_called_once_with("W8A8_DYNAMIC", "linear")

    def test_native_mxfp4_rejects_incompatible_formats(self):
        for quant_format in ("int-quantized", "pack-quantized", "mxfp8-quantized"):
            with self.subTest(format=quant_format), self.assertRaises(NotImplementedError):
                self.config._detect_quant_type(self._make_mx_quant(), None, quant_format)

    def test_native_mxfp4_rejects_invalid_weight_geometry(self):
        for attribute, value in (
            ("num_bits", 8),
            ("group_size", 16),
            ("group_size", 64),
            ("strategy", "channel"),
            ("dynamic", True),
            ("symmetric", False),
        ):
            weight = self._make_mx_quant()
            setattr(weight, attribute, value)
            with self.subTest(attribute=attribute, value=value), self.assertRaises(NotImplementedError):
                self.config._detect_quant_type(weight, None, "mxfp4-pack-quantized")

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
        mock_method.assert_called_once_with(moe_scheme, layer.moe_config, None)

    def test_no_quant_method(self):
        layer = MagicMock(spec=Attention)
        result = self.config.get_quant_method(layer, "attn")
        self.assertIsNone(result)
