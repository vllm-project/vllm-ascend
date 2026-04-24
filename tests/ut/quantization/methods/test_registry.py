from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.base import (
    AscendLinearScheme,
    AscendMoEScheme,
)
from vllm_ascend.quantization.methods.registry import (
    _SCHEME_REGISTRY,
    get_scheme_class,
    register_scheme,
)


class TestRegisterScheme(TestBase):
    def test_register_scheme(self):
        @register_scheme("TEST_QUANT_TYPE", "linear")
        class TestLinearScheme(AscendLinearScheme):
            def get_weight(self, input_size, output_size, params_dtype):
                return {}

            def apply(self, layer, x, bias=None, tp_rank=0):
                return x

        scheme_class = get_scheme_class("TEST_QUANT_TYPE", "linear")
        self.assertIs(scheme_class, TestLinearScheme)


class TestGetSchemeClass(TestBase):
    def test_get_scheme_class_existing_linear(self):
        cls = get_scheme_class("W8A8_DYNAMIC", "linear")
        self.assertIsNotNone(cls)
        self.assertTrue(issubclass(cls, AscendLinearScheme))

    def test_get_scheme_class_existing_moe(self):
        cls = get_scheme_class("W8A8_DYNAMIC", "moe")
        self.assertIsNotNone(cls)
        self.assertTrue(issubclass(cls, AscendMoEScheme))

    def test_get_scheme_class_existing_attention(self):
        cls = get_scheme_class("FAKQuant", "attention")
        self.assertIsNotNone(cls)

    def test_get_scheme_class_nonexistent(self):
        cls = get_scheme_class("NONEXISTENT", "linear")
        self.assertIsNone(cls)

    def test_get_scheme_class_nonexistent_layer_type(self):
        cls = get_scheme_class("W8A8_DYNAMIC", "nonexistent")
        self.assertIsNone(cls)

    def test_register_scheme_duplicate_raises(self):
        with self.assertRaises(ValueError):

            @register_scheme("W8A8_DYNAMIC", "linear")
            class Duplicate:
                pass

    def test_all_linear_schemes_subclass_ascend_linear_scheme(self):
        for (quant_type, layer_type), scheme_cls in _SCHEME_REGISTRY.items():
            if layer_type == "linear":
                self.assertTrue(
                    issubclass(scheme_cls, AscendLinearScheme),
                    f"{scheme_cls.__name__} for {quant_type}/{layer_type} should be subclass of AscendLinearScheme",
                )

    def test_all_moe_schemes_subclass_ascend_moe_scheme(self):
        for (quant_type, layer_type), scheme_cls in _SCHEME_REGISTRY.items():
            if layer_type == "moe":
                self.assertTrue(
                    issubclass(scheme_cls, AscendMoEScheme),
                    f"{scheme_cls.__name__} for {quant_type}/{layer_type} should be subclass of AscendMoEScheme",
                )

    def test_registry_not_empty(self):
        self.assertGreater(len(_SCHEME_REGISTRY), 0)

    def test_registry_key_format(self):
        for key in _SCHEME_REGISTRY.keys():
            self.assertEqual(len(key), 2)
            self.assertIsInstance(key[0], str)
            self.assertIsInstance(key[1], str)
