import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import MoERunner, RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from tests.ut.base import TestBase
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.fp8_config import AscendDeepseekV4FP8Config, AscendFp8Config


class TestAscendDeepseekV4FP8Config(TestBase):
    def setUp(self):
        self.config = AscendDeepseekV4FP8Config()
        self.config.weight_block_size = (128, 128)

    def test_get_quant_method_for_linear_base(self):
        """LinearBase layer returns AscendLinearMethod with ds_linear scheme."""
        linear_layer = MagicMock(spec=LinearBase)
        mock_scheme_instance = MagicMock()
        mock_scheme_class = MagicMock(return_value=mock_scheme_instance)
        mock_ascend_linear = MagicMock()

        with (
            patch(
                "vllm_ascend.quantization.fp8_config.get_scheme_class",
                return_value=mock_scheme_class,
            ),
            patch(
                "vllm_ascend.quantization.method_adapters.AscendLinearMethod",
                return_value=mock_ascend_linear,
            ),
        ):
            method = self.config.get_quant_method(linear_layer, "model.layers.0.mlp.gate_proj")

        mock_scheme_class.assert_called_once_with((128, 128))
        self.assertIs(method, mock_ascend_linear)

    def test_get_quant_method_for_moe_runner_fp4(self):
        """MoERunner layer with fp4 expert_dtype returns AscendFusedMoEMethod."""
        moe_config = MagicMock()
        moe_layer = MagicMock(spec=MoERunner)
        moe_layer.moe_config = moe_config

        mock_scheme_instance = MagicMock()
        mock_scheme_class = MagicMock(return_value=mock_scheme_instance)
        mock_ascend_moe = MagicMock()

        self.config._resolved_expert_dtype = "fp4"

        with (
            patch(
                "vllm_ascend.quantization.fp8_config.get_scheme_class",
                return_value=mock_scheme_class,
            ),
            patch(
                "vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod",
                return_value=mock_ascend_moe,
            ),
        ):
            method = self.config.get_quant_method(moe_layer, "model.layers.0.mlp")

        mock_scheme_class.assert_called_once_with()
        self.assertIs(method, mock_ascend_moe)

    def test_get_quant_method_for_routed_experts_fp4(self):
        """RoutedExperts layer with fp4 expert_dtype returns AscendFusedMoEMethod."""
        moe_config = MagicMock()
        moe_layer = MagicMock(spec=RoutedExperts)
        moe_layer.moe_config = moe_config

        mock_scheme_instance = MagicMock()
        mock_scheme_class = MagicMock(return_value=mock_scheme_instance)
        mock_ascend_moe = MagicMock()

        self.config._resolved_expert_dtype = "fp4"

        with (
            patch(
                "vllm_ascend.quantization.fp8_config.get_scheme_class",
                return_value=mock_scheme_class,
            ),
            patch(
                "vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod",
                return_value=mock_ascend_moe,
            ),
        ):
            method = self.config.get_quant_method(moe_layer, "model.layers.0.mlp.experts")

        mock_scheme_class.assert_called_once_with()
        self.assertIs(method, mock_ascend_moe)

    def test_get_quant_method_for_moe_non_fp4_raises(self):
        """MoE layer with non-fp4 expert_dtype raises NotImplementedError."""
        moe_config = MagicMock()
        moe_layer = MagicMock(spec=MoERunner)
        moe_layer.moe_config = moe_config

        self.config._resolved_expert_dtype = "fp8"

        with self.assertRaises(NotImplementedError):
            self.config.get_quant_method(moe_layer, "model.layers.0.mlp")

    def test_get_quant_method_for_other_layer_returns_none(self):
        """Non-LinearBase and non-MoE layers return None."""
        other_layer = torch.nn.Linear(10, 10)

        method = self.config.get_quant_method(other_layer, "some.prefix")
        self.assertIsNone(method)


class TestAscendFp8Config(TestBase):
    """Native block-wise FP8 checkpoints, e.g. Qwen/Qwen3.8-27B-FP8."""

    def build_config(self, ignored_layers=None, weight_block_size=(128, 128), activation_scheme="dynamic"):
        config = AscendFp8Config()
        config.is_checkpoint_fp8_serialized = True
        config.activation_scheme = activation_scheme
        config.weight_block_size = weight_block_size
        config.ignored_layers = ignored_layers or []
        return config

    def test_ignored_linear_falls_back_to_unquantized(self):
        # visual.merger.linear_fc1 is the layer that used to crash: the
        # checkpoint keeps it in bfloat16 and lists it as not converted.
        config = self.build_config(ignored_layers=["visual.merger.linear_fc1"])
        method = config.get_quant_method(MagicMock(spec=LinearBase), "visual.merger.linear_fc1")
        self.assertIsInstance(method, AscendUnquantizedLinearMethod)

    def test_official_visual_layer_name_is_skipped(self):
        # Qwen/Qwen3.8-27B-FP8 stores exact names, not globs such as visual.merger.*.
        official_name = "visual.blocks.0.mlp.linear_fc1"
        config = self.build_config(ignored_layers=[official_name])
        method = config.get_quant_method(MagicMock(spec=LinearBase), official_name)
        self.assertIsInstance(method, AscendUnquantizedLinearMethod)
        quantized_prefix = "model.layers.0.mlp.gate_proj"
        with (
            patch(
                "vllm_ascend.quantization.fp8_config.get_scheme_class",
                return_value=MagicMock(),
            ),
            patch(
                "vllm_ascend.quantization.method_adapters.AscendLinearMethod",
                return_value="block-fp8",
            ),
        ):
            self.assertEqual(
                config.get_quant_method(MagicMock(spec=LinearBase), quantized_prefix),
                "block-fp8",
            )

    def test_glm53_flash_kda_layer_stays_unquantized(self):
        # Official zai-org/GLM-5.3-Flash lists KDA projections in
        # modules_to_not_convert; they have no weight_scale_inv.
        kda_name = "model.layers.0.self_attn.q_proj"
        config = self.build_config(ignored_layers=[kda_name])
        method = config.get_quant_method(MagicMock(spec=LinearBase), kda_name)
        self.assertIsInstance(method, AscendUnquantizedLinearMethod)

    def test_ignored_moe_falls_back_to_unquantized(self):
        config = self.build_config(ignored_layers=["model.layers.0.mlp.experts"])
        moe_layer = MagicMock(spec=RoutedExperts)
        moe_layer.moe_config = MagicMock()
        unquantized_moe = MagicMock()

        with patch(
            "vllm_ascend.ops.fused_moe.routed_experts.AscendUnquantizedFusedMoEMethod",
            return_value=unquantized_moe,
        ):
            method = config.get_quant_method(moe_layer, "model.layers.0.mlp.experts")

        self.assertIs(method, unquantized_moe)

    def test_quantized_linear_uses_the_block_scheme(self):
        config = self.build_config()
        mock_scheme_instance = MagicMock()
        mock_scheme_class = MagicMock(return_value=mock_scheme_instance)
        mock_linear_method = MagicMock()

        with (
            patch(
                "vllm_ascend.quantization.fp8_config.get_scheme_class",
                return_value=mock_scheme_class,
            ) as mock_get_scheme,
            patch(
                "vllm_ascend.quantization.method_adapters.AscendLinearMethod",
                return_value=mock_linear_method,
            ),
        ):
            method = config.get_quant_method(MagicMock(spec=LinearBase), "model.layers.0.mlp.gate_proj")

        mock_get_scheme.assert_called_once_with("fp8", "linear")
        mock_scheme_class.assert_called_once_with((128, 128))
        self.assertIs(method, mock_linear_method)

    def test_quantized_moe_uses_the_block_scheme(self):
        config = self.build_config()
        moe_config = MagicMock()
        moe_layer = MagicMock(spec=MoERunner)
        moe_layer.moe_config = moe_config
        mock_scheme_class = MagicMock()
        mock_moe_method = MagicMock()

        with (
            patch(
                "vllm_ascend.quantization.fp8_config.get_scheme_class",
                return_value=mock_scheme_class,
            ) as mock_get_scheme,
            patch(
                "vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod",
                return_value=mock_moe_method,
            ),
        ):
            method = config.get_quant_method(moe_layer, "model.layers.0.mlp.experts")

        mock_get_scheme.assert_called_once_with("fp8", "moe")
        mock_scheme_class.assert_called_once_with((128, 128), moe_config)
        self.assertIs(method, mock_moe_method)

    def test_embedding_is_left_to_the_default_method(self):
        # Returning None makes VocabParallelEmbedding fall back to
        # UnquantizedEmbeddingMethod on its own.
        config = self.build_config()
        method = config.get_quant_method(MagicMock(spec=VocabParallelEmbedding), "model.embed_tokens")
        self.assertIsNone(method)

    def test_per_tensor_checkpoint_reports_an_actionable_error(self):
        config = self.build_config(weight_block_size=None)
        with self.assertRaisesRegex(NotImplementedError, "weight_block_size"):
            config.get_quant_method(MagicMock(spec=LinearBase), "model.layers.0.mlp.gate_proj")

    def test_static_activation_scheme_reports_an_actionable_error(self):
        config = self.build_config(activation_scheme="static")
        with self.assertRaisesRegex(NotImplementedError, "activation_scheme"):
            config.get_quant_method(MagicMock(spec=LinearBase), "model.layers.0.mlp.gate_proj")

    def test_unrelated_layer_returns_none(self):
        config = self.build_config()
        self.assertIsNone(config.get_quant_method(torch.nn.Linear(8, 8), "some.prefix"))

    def test_get_quant_method_when_is_layer_skipped_has_no_match_mode(self):
        # vLLM 0.27.1's is_layer_skipped has no match_mode keyword.
        def old_is_layer_skipped(prefix, ignored_layers, packed_modules_mapping):
            return False

        config = self.build_config()
        with (
            patch("vllm_ascend.quantization.fp8_config._IS_LAYER_SKIPPED_SUPPORTS_MATCH_MODE", False),
            patch("vllm_ascend.quantization.fp8_config.is_layer_skipped", side_effect=old_is_layer_skipped),
            patch("vllm_ascend.quantization.fp8_config.get_scheme_class", return_value=MagicMock()),
            patch(
                "vllm_ascend.quantization.method_adapters.AscendLinearMethod",
                return_value="block-fp8",
            ),
        ):
            method = config.get_quant_method(MagicMock(spec=LinearBase), "model.layers.0.mlp.gate_proj")
        self.assertEqual(method, "block-fp8")


if __name__ == "__main__":
    unittest.main()
