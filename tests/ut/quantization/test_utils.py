import types

from tests.ut.base import TestBase
from vllm_ascend.quantization.utils import (ASCEND_QUANTIZATION_METHOD_MAP,
                                            get_quant_method)


class TestGetQuantMethod(TestBase):

    def setUp(self):
        self.original_quantization_method_map = ASCEND_QUANTIZATION_METHOD_MAP.copy(
        )
        for quant_type, layer_map in ASCEND_QUANTIZATION_METHOD_MAP.items():
            for layer_type in layer_map.keys():
                ASCEND_QUANTIZATION_METHOD_MAP[quant_type][
                    layer_type] = types.new_class(f"{quant_type}_{layer_type}")

    def tearDown(self):
        # Restore original map
        ASCEND_QUANTIZATION_METHOD_MAP.clear()
        ASCEND_QUANTIZATION_METHOD_MAP.update(
            self.original_quantization_method_map)

    def test_linear_quant_methods(self):
        for quant_type, layer_map in ASCEND_QUANTIZATION_METHOD_MAP.items():
            if "linear" in layer_map.keys():
                prefix = "linear_layer"
                cls = layer_map["linear"]
                method = get_quant_method({"linear_layer.weight": quant_type},
                                          prefix, "linear")
                self.assertIsInstance(method, cls)

    def test_moe_quant_methods(self):
        for quant_type, layer_map in ASCEND_QUANTIZATION_METHOD_MAP.items():
            if "moe" in layer_map.keys():
                prefix = "layer"
                cls = layer_map["moe"]
                method = get_quant_method({"layer.weight": quant_type}, prefix,
                                          "moe")
                self.assertIsInstance(method, cls)

    def test_with_fa_quant_type(self):
        quant_description = {"fa_quant_type": "C8"}
        method = get_quant_method(quant_description, ".attn", "attention")
        self.assertIsInstance(
            method, ASCEND_QUANTIZATION_METHOD_MAP["C8"]["attention"])

    def test_with_kv_quant_type(self):
        quant_description = {"kv_quant_type": "C8"}
        method = get_quant_method(quant_description, ".attn", "attention")
        self.assertIsInstance(
            method, ASCEND_QUANTIZATION_METHOD_MAP["C8"]["attention"])

    def test_invalid_layer_type(self):
        quant_description = {"linear_layer.weight": "W8A8"}
        with self.assertRaises(NotImplementedError):
            get_quant_method(quant_description, "linear_layer", "unsupported")

    def test_invalid_quant_type(self):
        quant_description = {"linear_layer.weight": "UNKNOWN"}
        with self.assertRaises(NotImplementedError):
            get_quant_method(quant_description, "linear_layer", "linear")
