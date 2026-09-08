from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from tests.ut.base import TestBase
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.configs.modelopt_mxfp8_config import AscendModelOptMxFp8Config


class TestAscendModelOptMxFp8Config(TestBase):
    @staticmethod
    def _from_minimax_config(ignored_layers=None):
        return AscendModelOptMxFp8Config.from_config(
            {
                "quant_method": "mxfp8",
                "ignored_layers": ignored_layers or [],
                "weight_block_size": [1, 32],
            }
        )

    def test_registered_for_both_modelopt_mxfp8_names(self):
        self.assertIs(get_quantization_config("mxfp8"), AscendModelOptMxFp8Config)
        self.assertIs(get_quantization_config("modelopt_mxfp8"), AscendModelOptMxFp8Config)

    def test_from_minimax_config_reuses_modelopt_parser(self):
        ignored_layers = ["language_model.model.embed_tokens", "language_model.lm_head"]

        config = self._from_minimax_config(ignored_layers)

        self.assertIsInstance(config, AscendModelOptMxFp8Config)
        self.assertTrue(config.is_checkpoint_mxfp8_serialized)
        self.assertEqual(config.exclude_modules, ignored_layers)
        self.assertFalse(hasattr(config, "quant_description"))

    def test_vllm_mapper_updates_excluded_layers(self):
        config = self._from_minimax_config(["language_model.model.layers.3.block_sparse_moe.gate"])
        mapper = MagicMock()
        mapper.apply_list.return_value = ["model.layers.3.block_sparse_moe.gate"]

        config.apply_vllm_mapper(mapper)

        self.assertTrue(config.is_layer_excluded("model.layers.3.block_sparse_moe.gate"))
        self.assertFalse(config.is_layer_excluded("model.layers.3.block_sparse_moe.experts"))

    def test_excluded_embedding_and_lm_head_are_unquantized(self):
        config = self._from_minimax_config(["model.embed_tokens", "lm_head"])

        embedding_method = config.get_quant_method(
            MagicMock(spec=VocabParallelEmbedding),
            "model.embed_tokens",
        )
        lm_head_method = config.get_quant_method(
            MagicMock(spec=ParallelLMHead),
            "lm_head",
        )

        self.assertIsInstance(embedding_method, UnquantizedEmbeddingMethod)
        self.assertIsInstance(lm_head_method, UnquantizedEmbeddingMethod)

    def test_excluded_linear_uses_ascend_unquantized_method(self):
        config = self._from_minimax_config(["model.layers.0.block_sparse_moe.gate"])

        method = config.get_quant_method(
            MagicMock(spec=LinearBase),
            "model.layers.0.block_sparse_moe.gate",
        )

        self.assertIsInstance(method, AscendUnquantizedLinearMethod)

    def test_linear_uses_ascend_mxfp8_method(self):
        config = self._from_minimax_config()
        scheme = MagicMock()
        expected_method = MagicMock()

        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.AscendW8A8MXFP8DynamicLinearMethod",
                return_value=scheme,
            ),
            patch(
                "vllm_ascend.quantization.method_adapters.AscendLinearMethod",
                return_value=expected_method,
            ) as mock_linear_method,
        ):
            method = config.get_quant_method(
                MagicMock(spec=LinearBase),
                "model.layers.0.self_attn.qkv_proj",
            )

        self.assertIs(method, expected_method)
        mock_linear_method.assert_called_once_with(scheme)

    def test_moe_uses_ascend_mxfp8_method(self):
        config = self._from_minimax_config()
        layer = torch.nn.Module()
        layer.moe_config = MagicMock()
        scheme = MagicMock()
        expected_method = MagicMock()
        tid2eid = MagicMock()

        with (
            patch(
                "vllm_ascend.quantization.configs.modelopt_mxfp8_config.is_fused_moe_layer",
                return_value=True,
            ),
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.AscendW8A8MXFP8DynamicFusedMoEMethod",
                return_value=scheme,
            ),
            patch(
                "vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod",
                return_value=expected_method,
            ) as mock_fused_method,
        ):
            method = config.get_quant_method(
                layer,
                "model.layers.0.block_sparse_moe",
                tid2eid,
            )

        self.assertIs(method, expected_method)
        mock_fused_method.assert_called_once_with(scheme, layer.moe_config, tid2eid)

    def test_bf16_kv_cache_and_embedding_keep_native_methods(self):
        config = self._from_minimax_config()

        self.assertIsNone(config.get_quant_method(torch.nn.Module(), "model.layers.0.self_attn"))
        self.assertIsNone(
            config.get_quant_method(
                MagicMock(spec=VocabParallelEmbedding),
                "model.embed_tokens",
            )
        )

    def test_get_min_capability_is_not_available_on_ascend(self):
        with self.assertRaises(NotImplementedError):
            AscendModelOptMxFp8Config.get_min_capability()
