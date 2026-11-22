from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig

from tests.ut.base import TestBase
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.configs.mxfp4_config import AscendMxfp4Config


class TestAscendMxfp4Config(TestBase):
    @staticmethod
    def checkpoint_config():
        return {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
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
            },
        }

    def test_supported_and_registered_without_replacing_compressed_tensors(self):
        self.assertIn("mxfp4", NPUPlatform.supported_quantization)
        self.assertIs(get_quantization_config("mxfp4"), AscendMxfp4Config)
        self.assertIs(get_quantization_config("compressed-tensors"), AscendCompressedTensorsConfig)

    def test_unmodified_upstream_kimi_rewrite_selects_ascend_config(self):
        model_config = SimpleNamespace(
            **{
                name: SimpleNamespace(quantization_config=self.checkpoint_config())
                for name in ("hf_config", "hf_text_config", "model_arch_config")
            }
        )
        KimiK3ForConditionalGenerationConfig.verify_and_update_model_config(model_config)
        for name in ("hf_config", "hf_text_config", "model_arch_config"):
            checkpoint = getattr(model_config, name).quantization_config
            self.assertEqual(checkpoint["quant_method"], "mxfp4")
            parsed = get_quantization_config(checkpoint["quant_method"]).from_config(checkpoint)
            self.assertIsInstance(parsed, AscendMxfp4Config)
            scheme = parsed.target_scheme_map["Linear"]
            self.assertEqual(parsed._detect_quant_type(scheme["weights"], None, scheme["format"]), "W4A8_MXFP")

    def test_parsing_preserves_ignore_metadata_and_activation_absence(self):
        checkpoint = self.checkpoint_config()
        parsed = AscendMxfp4Config.from_config(checkpoint)
        self.assertEqual(parsed.get_name(), "mxfp4")
        self.assertEqual(parsed.ignore, ["lm_head"])
        self.assertIs(parsed.quant_description, checkpoint)
        self.assertIsNone(parsed.target_scheme_map["Linear"]["input_activations"])
        self.assertEqual(checkpoint, self.checkpoint_config())

    def test_linear_and_moe_reuse_existing_w4a8_scheme(self):
        parsed = AscendMxfp4Config.from_config(self.checkpoint_config())
        scheme = parsed.target_scheme_map["Linear"]
        for layer_type in ("linear", "moe"):
            with (
                self.subTest(layer_type=layer_type),
                patch("vllm_ascend.quantization.methods.get_scheme_class") as get_scheme_class,
            ):
                result = parsed._create_scheme_for_layer_type(scheme["weights"], None, None, layer_type)
                get_scheme_class.assert_called_once_with("W4A8_MXFP", layer_type)
                self.assertIs(result, get_scheme_class.return_value.return_value)
        self.assertEqual(parsed.quant_description["group_size"], 32)

    def test_mapper_still_updates_targets_and_ignored_layers(self):
        parsed = AscendMxfp4Config.from_config(self.checkpoint_config())
        mapper = MagicMock()
        mapper.apply_dict.return_value = {"model.layers.0.mlp.gate_proj": parsed.target_scheme_map["Linear"]}
        mapper.apply_list.return_value = ["language_model.lm_head"]
        parsed.apply_vllm_mapper(mapper)
        self.assertEqual(parsed.target_scheme_map, mapper.apply_dict.return_value)
        self.assertEqual(parsed.ignore, mapper.apply_list.return_value)

    def test_group_format_takes_precedence_and_missing_format_uses_global(self):
        checkpoint = self.checkpoint_config()
        checkpoint["format"] = "mixed-precision"
        checkpoint["config_groups"]["group_0"]["format"] = "mxfp4-pack-quantized"
        parsed = AscendMxfp4Config.from_config(checkpoint)
        self.assertEqual(parsed.target_scheme_map["Linear"]["format"], "mxfp4-pack-quantized")

        checkpoint["config_groups"]["group_0"]["format"] = "pack-quantized"
        checkpoint["format"] = "mxfp4-pack-quantized"
        with self.assertRaisesRegex(ValueError, "Unsupported Ascend mxfp4 recipe"):
            AscendMxfp4Config.from_config(checkpoint)

    def test_explicit_activation_recipe_is_rejected_before_shared_parser(self):
        for num_bits in (4, 8):
            checkpoint = self.checkpoint_config()
            group = checkpoint["config_groups"]["group_0"]
            group["input_activations"] = {**group["weights"], "num_bits": num_bits, "dynamic": True}
            with (
                self.subTest(num_bits=num_bits),
                patch.object(AscendCompressedTensorsConfig, "from_config") as parse,
                self.assertRaisesRegex(ValueError, "input activation recipe"),
            ):
                AscendMxfp4Config.from_config(checkpoint)
            parse.assert_not_called()

    def test_missing_activation_key_is_also_native(self):
        checkpoint = self.checkpoint_config()
        del checkpoint["config_groups"]["group_0"]["input_activations"]
        parsed = AscendMxfp4Config.from_config(checkpoint)
        self.assertIsNone(parsed.target_scheme_map["Linear"]["input_activations"])

    def test_missing_groups_or_targets_fail_clearly(self):
        for checkpoint in ({}, {"config_groups": {}}):
            with self.subTest(checkpoint=checkpoint), self.assertRaisesRegex(ValueError, "config_groups"):
                AscendMxfp4Config.from_config(checkpoint)
        checkpoint = self.checkpoint_config()
        checkpoint["config_groups"]["group_0"]["targets"] = []
        with self.assertRaisesRegex(ValueError, "quantized target"):
            AscendMxfp4Config.from_config(checkpoint)

    def test_incompatible_weight_recipes_are_rejected(self):
        for update in (
            {"num_bits": 8},
            {"group_size": 64},
            {"dynamic": True},
            {"symmetric": False},
            {"type": "int"},
        ):
            checkpoint = deepcopy(self.checkpoint_config())
            checkpoint["config_groups"]["group_0"]["weights"].update(update)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "Unsupported Ascend mxfp4 recipe"):
                AscendMxfp4Config.from_config(checkpoint)
