import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import torch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearBase

from tests.ut.base import TestBase
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.modelslim_config import (
    MODELSLIM_CONFIG_FILENAME,
    AscendModelSlimConfig,
    create_scheme_for_layer,
    get_linear_quant_type,
    get_packed_modules_mapping,
    get_quant_type_for_layer,
)
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase


class TestAscendModelSlimConfig(TestBase):

    def setUp(self):
        self.sample_config = {
            "weight": "INT8",
            "fa_quant_type": "C8",
            "layers.1.fa_k.scale": "C8",
            "layer1.weight": "INT8",
            "layer2.weight": "FLOAT",
            "fused_layer.weight": "FLOAT",
            "fused_layer.shard1.weight": "FLOAT",
            "fused_layer.shard2.weight": "FLOAT",
            "shard1.weight": "FLOAT",
            "shard2.weight": "FLOAT",
        }
        self.ascend_config = AscendModelSlimConfig(self.sample_config)
        self.ascend_config.packed_modules_mapping = None

    def test_init(self):
        self.assertEqual(self.ascend_config.quant_description,
                         self.sample_config)

    def test_repr(self):
        repr_str = repr(self.ascend_config)
        self.assertTrue(repr_str.startswith("AscendModelSlimConfig:\n"))

    def test_get_name(self):
        self.assertEqual(AscendModelSlimConfig.get_name(),
                         ASCEND_QUANTIZATION_METHOD)

    def test_get_supported_act_dtypes(self):
        supported_dtypes = AscendModelSlimConfig.get_supported_act_dtypes()
        self.assertEqual(len(supported_dtypes), 3)

    def test_get_min_capability(self):
        with self.assertRaises(NotImplementedError):
            AscendModelSlimConfig.get_min_capability()

    def test_get_config_filenames(self):
        filenames = AscendModelSlimConfig.get_config_filenames()
        self.assertEqual(filenames, [])

    def test_from_config(self):
        config = AscendModelSlimConfig.from_config(self.sample_config)
        self.assertIsInstance(config, AscendModelSlimConfig)
        self.assertEqual(config.quant_description, self.sample_config)

    @patch('torch.npu.is_available')
    def test_override_quantization_method(self, mock_is_available):
        # Test when NPU is available
        mock_is_available.return_value = True
        result = AscendModelSlimConfig.override_quantization_method(None, None)
        self.assertIsNone(result)
        hf_quant_cfg = {"quant_method": ""}
        result = AscendModelSlimConfig.override_quantization_method(
            hf_quant_cfg, None)
        self.assertEqual(result, "ascend")

        # Test when NPU is not available
        mock_is_available.return_value = False
        result = AscendModelSlimConfig.override_quantization_method(None, None)
        self.assertIsNone(result)
        hf_quant_cfg = {"quant_method": ""}
        result = AscendModelSlimConfig.override_quantization_method(
            hf_quant_cfg, None)
        self.assertIsNone(result)

    def test_get_quant_method_for_linear(self):
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None
        linear_layer = MagicMock(spec=LinearBase)
        # Test skipped layer
        with patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch.object(self.ascend_config, \
                          'is_layer_skipped_ascend',
                          return_value=True):
            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIsInstance(method, AscendUnquantizedLinearMethod)

        # Test quantized layer
        mock_scheme = MagicMock()
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=False), \
            patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch("vllm_ascend.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme), \
            patch('vllm_ascend.quantization.method_adapters.AscendLinearMethod', return_value=MagicMock()) as mock_ascend_linear:

            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIs(method, mock_ascend_linear.return_value)
            mock_ascend_linear.assert_called_once_with(mock_scheme)

    def test_get_quant_method_for_attention(self):
        attention_layer = MagicMock(spec=Attention)
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None
        mock_scheme = MagicMock()
        with patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch("vllm_ascend.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme), \
            patch('vllm_ascend.quantization.method_adapters.AscendKVCacheMethod', \
                   return_value=MagicMock()) as mock_ascend_kvcache:
            # Test with fa_quant_type
            method = self.ascend_config.get_quant_method(
                attention_layer, ".attn")
            self.assertIs(method, None)
            method = self.ascend_config.get_quant_method(
                attention_layer, "layers.1.attn")
            self.assertIs(method, mock_ascend_kvcache.return_value)

    def test_get_quant_method_for_c8_kv_cache_attention(self):
        c8_config = AscendModelSlimConfig({"kv_cache_type": "C8"})
        attention_layer = MagicMock(spec=AttentionLayerBase)
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.hf_config.model_type = None
        with patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_vllm_config), \
            patch("vllm_ascend.quantization.method_adapters.AscendKVCacheMethod", return_value=MagicMock()) as mock_kvcache:
            method = c8_config.get_quant_method(attention_layer, "model.layers.0.self_attn.attn")
            self.assertIs(method, mock_kvcache.return_value)
            args, _ = mock_kvcache.call_args
            from vllm_ascend.quantization.methods.kv_c8 import AscendC8KVCacheAttentionMethod
            self.assertIsInstance(args[0], AscendC8KVCacheAttentionMethod)

    def test_get_quant_method_for_fused_moe(self):
        fused_moe_layer = MagicMock(spec=FusedMoE)
        fused_moe_layer.moe = MagicMock(spec=FusedMoEConfig)
        fused_moe_layer.moe_config = MagicMock(spec=FusedMoEConfig)
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None

        # Test skipped layer
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=True), \
            patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch('vllm_ascend.ops.fused_moe.fused_moe.AscendUnquantizedFusedMoEMethod', return_value=MagicMock()) as mock_ascend_moe:
            method = self.ascend_config.get_quant_method(
                fused_moe_layer, "moe_layer")
            self.assertIs(method, mock_ascend_moe.return_value)

        # Test quantized layer
        mock_scheme = MagicMock()
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=False), \
            patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch("vllm_ascend.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme), \
            patch('vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod', return_value=MagicMock()) as mock_ascend_moe:
            method = self.ascend_config.get_quant_method(
                fused_moe_layer, "moe_layer")
            self.assertIs(method, mock_ascend_moe.return_value)

    def test_is_layer_skipped_ascend(self):
        # Test non-fused layer that should be quantized
        self.assertFalse(self.ascend_config.is_layer_skipped_ascend("layer1"))

        # Test non-fused layer that should be skipped
        self.assertTrue(self.ascend_config.is_layer_skipped_ascend("layer2"))

        # Test fused layer
        fused_mapping = {"fused_layer": ["shard1", "shard2"]}
        self.assertTrue(
            self.ascend_config.is_layer_skipped_ascend("fused_layer",
                                                       fused_mapping))

        # Test inconsistent fused layer shards
        bad_config = {"shard1.weight": "FLOAT", "shard2.weight": "INT8"}
        config = AscendModelSlimConfig(bad_config)
        with self.assertRaises(ValueError):
            config.is_layer_skipped_ascend("fused_layer", fused_mapping)

    def test_init_with_none_config(self):
        config = AscendModelSlimConfig(None)
        self.assertEqual(config.quant_description, {})

    def test_init_with_default_config(self):
        config = AscendModelSlimConfig()
        self.assertEqual(config.quant_description, {})

    def test_maybe_update_config_already_populated(self):
        # When quant_description is already populated, should be a no-op
        self.assertTrue(len(self.ascend_config.quant_description) > 0)
        self.ascend_config.maybe_update_config("/some/model/path")
        # quant_description should remain unchanged
        self.assertEqual(self.ascend_config.quant_description,
                         self.sample_config)

    def test_maybe_update_config_loads_from_file(self):
        config = AscendModelSlimConfig()
        self.assertEqual(config.quant_description, {})

        quant_data = {"layer1.weight": "INT8", "layer2.weight": "FLOAT"}
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, MODELSLIM_CONFIG_FILENAME)
            with open(config_path, "w") as f:
                json.dump(quant_data, f)

            config.maybe_update_config(tmpdir)

        self.assertEqual(config.quant_description, quant_data)

    def test_maybe_update_config_raises_when_file_missing(self):
        config = AscendModelSlimConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                config.maybe_update_config(tmpdir)

            error_msg = str(ctx.exception)
            self.assertIn("ModelSlim Quantization Config Not Found", error_msg)
            self.assertIn(MODELSLIM_CONFIG_FILENAME, error_msg)

    def test_maybe_update_config_raises_with_json_files_listed(self):
        config = AscendModelSlimConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy json file that is NOT the config file
            dummy_path = os.path.join(tmpdir, "config.json")
            with open(dummy_path, "w") as f:
                json.dump({"dummy": True}, f)

            with self.assertRaises(ValueError) as ctx:
                config.maybe_update_config(tmpdir)

            error_msg = str(ctx.exception)
            self.assertIn("config.json", error_msg)

    def test_maybe_update_config_non_directory_raises(self):
        config = AscendModelSlimConfig()

        with self.assertRaises(ValueError) as ctx:
            config.maybe_update_config("not_a_real_directory_path")

        error_msg = str(ctx.exception)
        self.assertIn("ModelSlim Quantization Config Not Found", error_msg)

    def test_apply_extra_quant_adaptations_shared_head(self):
        config = AscendModelSlimConfig()
        config.quant_description = {
            "model.layers.0.shared_head.weight": "INT8",
        }
        config._apply_extra_quant_adaptations()
        self.assertIn("model.layers.0.weight", config.quant_description)
        self.assertEqual(config.quant_description["model.layers.0.weight"],
                         "INT8")

    def test_apply_extra_quant_adaptations_weight_packed(self):
        config = AscendModelSlimConfig()
        config.quant_description = {
            "model.layers.0.weight_packed": "INT8",
        }
        config._apply_extra_quant_adaptations()
        self.assertIn("model.layers.0.weight", config.quant_description)
        self.assertEqual(config.quant_description["model.layers.0.weight"],
                         "INT8")


class TestGetPackedModulesMapping(TestBase):

    def test_known_model_type(self):
        result = get_packed_modules_mapping("deepseek_v3")
        self.assertIsInstance(result, dict)
        self.assertIn("gate_up_proj", result)
        self.assertIn("experts", result)

    def test_unknown_model_type(self):
        result = get_packed_modules_mapping("unknown_model")
        self.assertEqual(result, {})


class TestGetLinearQuantType(TestBase):

    def test_simple_non_fused_layer(self):
        quant_description = {"layer1.weight": "W8A8_DYNAMIC"}
        packed_modules_mapping = {}
        result = get_linear_quant_type(quant_description, "layer1", packed_modules_mapping)
        self.assertEqual(result, "W8A8_DYNAMIC")

    def test_fused_layer_consistent_quant_types(self):
        quant_description = {
            "shard1.weight": "W8A8_DYNAMIC",
            "shard2.weight": "W8A8_DYNAMIC",
        }
        packed_modules_mapping = {"fused_layer": ["shard1", "shard2"]}
        result = get_linear_quant_type(quant_description, "fused_layer", packed_modules_mapping)
        self.assertEqual(result, "W8A8_DYNAMIC")

    def test_fused_layer_inconsistent_quant_types(self):
        quant_description = {
            "shard1.weight": "W8A8_DYNAMIC",
            "shard2.weight": "FLOAT",
        }
        packed_modules_mapping = {"fused_layer": ["shard1", "shard2"]}
        with self.assertRaises(ValueError) as ctx:
            get_linear_quant_type(quant_description, "fused_layer", packed_modules_mapping)
        self.assertIn("Not all shards", str(ctx.exception))


class TestGetQuantTypeForLayer(TestBase):

    def test_attention_layer_with_fa_quant_type(self):
        quant_description = {
            "fa_quant_type": "C8",
            "layer1.weight": "W8A8_DYNAMIC",
        }
        result = get_quant_type_for_layer(quant_description, "layer1", "attention")
        self.assertEqual(result, "C8")

    def test_attention_layer_with_indexer_quant_type(self):
        quant_description = {
            "indexer_quant_type": "INT8",
            "layer1.weight": "W8A8_DYNAMIC",
        }
        result = get_quant_type_for_layer(quant_description, "layer1", "attention")
        self.assertEqual(result, "INT8")

    def test_attention_layer_with_layer_indexer_quant_type(self):
        quant_description = {
            "layer1.indexer.quant_type": "INT8",
            "layer1.weight": "W8A8_DYNAMIC",
        }
        result = get_quant_type_for_layer(quant_description, "layer1", "attention")
        self.assertEqual(result, "INT8")

    def test_linear_layer(self):
        quant_description = {
            "layer1.weight": "W8A8_DYNAMIC",
        }
        result = get_quant_type_for_layer(quant_description, "layer1", "linear")
        self.assertEqual(result, "W8A8_DYNAMIC")

    def test_moe_layer(self):
        quant_description = {
            "moe_layer.weight": "W8A8_DYNAMIC",
        }
        result = get_quant_type_for_layer(quant_description, "moe_layer", "moe")
        self.assertEqual(result, "W8A8_DYNAMIC")

    def test_with_none_packed_modules_mapping(self):
        quant_description = {
            "layer1.weight": "W8A8_DYNAMIC",
        }
        result = get_quant_type_for_layer(quant_description, "layer1", "linear", None)
        self.assertEqual(result, "W8A8_DYNAMIC")


class TestCreateSchemeForLayer(TestBase):

    @patch("vllm_ascend.quantization.modelslim_config.get_scheme_class")
    def test_successful_scheme_creation(self, mock_get_scheme_class):
        mock_scheme_class = MagicMock()
        mock_scheme_instance = MagicMock()
        mock_scheme_class.return_value = mock_scheme_instance
        mock_get_scheme_class.return_value = mock_scheme_class

        quant_description = {"layer1.weight": "W8A8_DYNAMIC"}
        result = create_scheme_for_layer(quant_description, "layer1", "linear")

        self.assertIs(result, mock_scheme_instance)
        mock_get_scheme_class.assert_called_once_with("W8A8_DYNAMIC", "linear")

    def test_quant_type_none_raises_error(self):
        quant_description = {}
        with self.assertRaises(KeyError) as ctx:
            create_scheme_for_layer(quant_description, "layer1", "linear")

    @patch("vllm_ascend.quantization.modelslim_config.get_scheme_class")
    def test_unsupported_quant_type_raises_error(self, mock_get_scheme_class):
        mock_get_scheme_class.return_value = None

        quant_description = {"layer1.weight": "UNSUPPORTED_TYPE"}
        with self.assertRaises(NotImplementedError) as ctx:
            create_scheme_for_layer(quant_description, "layer1", "linear")
        self.assertIn("doesn't support UNSUPPORTED_TYPE", str(ctx.exception))


class TestApplyVllmMapper(TestBase):

    def test_apply_mapper_with_populated_quant_description(self):
        config = AscendModelSlimConfig({"old_key.weight": "INT8"})
        mock_mapper = MagicMock()
        mock_mapper.apply_dict.return_value = {"new_key.weight": "INT8"}

        config.apply_vllm_mapper(mock_mapper)

        self.assertEqual(config.quant_description, {"new_key.weight": "INT8"})
        mock_mapper.apply_dict.assert_called_once_with({"old_key.weight": "INT8"})

    def test_idempotency_when_mapper_already_applied(self):
        config = AscendModelSlimConfig({"key.weight": "INT8"})
        mock_mapper = MagicMock()
        mock_mapper.apply_dict.return_value = {"key.weight": "INT8"}

        config.apply_vllm_mapper(mock_mapper)
        first_result = config.quant_description

        config.apply_vllm_mapper(mock_mapper)
        second_result = config.quant_description

        self.assertEqual(first_result, second_result)
        mock_mapper.apply_dict.assert_called_once()

    def test_empty_quant_description(self):
        config = AscendModelSlimConfig({})
        mock_mapper = MagicMock()

        config.apply_vllm_mapper(mock_mapper)

        mock_mapper.apply_dict.assert_not_called()


class TestGetCacheScale(TestBase):

    def test_c8_kv_cache_type_k_proj_scale(self):
        config = AscendModelSlimConfig({"kv_cache_type": "C8"})
        result = config.get_cache_scale("model.layers.0.k_proj.kv_cache_scale")
        self.assertEqual(result, "model.layers.0.attn.k_cache_scale")

    def test_c8_kv_cache_type_k_proj_offset(self):
        config = AscendModelSlimConfig({"kv_cache_type": "C8"})
        result = config.get_cache_scale("model.layers.0.k_proj.kv_cache_offset")
        self.assertEqual(result, "model.layers.0.attn.k_cache_offset")

    def test_c8_kv_cache_type_v_proj_scale(self):
        config = AscendModelSlimConfig({"kv_cache_type": "C8"})
        result = config.get_cache_scale("model.layers.0.v_proj.kv_cache_scale")
        self.assertEqual(result, "model.layers.0.attn.v_cache_scale")

    def test_c8_kv_cache_type_v_proj_offset(self):
        config = AscendModelSlimConfig({"kv_cache_type": "C8"})
        result = config.get_cache_scale("model.layers.0.v_proj.kv_cache_offset")
        self.assertEqual(result, "model.layers.0.attn.v_cache_offset")

    def test_non_c8_kv_cache_type(self):
        config = AscendModelSlimConfig({"kv_cache_type": "FLOAT"})
        result = config.get_cache_scale("model.layers.0.k_proj.kv_cache_scale")
        self.assertIsNone(result)

    def test_no_matching_suffix(self):
        config = AscendModelSlimConfig({"kv_cache_type": "C8"})
        result = config.get_cache_scale("model.layers.0.other_key")
        self.assertIsNone(result)


class TestQuantPrefixMapper(TestBase):

    def test_sets_model_type_and_returns_prefix(self):
        config = AscendModelSlimConfig({})
        result = config.quant_prefix_mapper("deepseek_v3", "model.layers.0")
        self.assertEqual(result, "model.layers.0")
        self.assertEqual(config.model_type, "deepseek_v3")


class TestIsFaQuantLayer(TestBase):

    def test_enable_fa_quant_true_matching_layer(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "C8",
            "layers.1.fa_k.scale": "C8",
        })
        result = config.is_fa_quant_layer("layers.1.attn")
        self.assertTrue(result)

    def test_enable_fa_quant_false(self):
        config = AscendModelSlimConfig({})
        result = config.is_fa_quant_layer("layers.1.attn")
        self.assertFalse(result)

    def test_non_matching_layer_index(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "C8",
            "layers.1.fa_k.scale": "C8",
        })
        result = config.is_fa_quant_layer("layers.2.attn")
        self.assertFalse(result)

    def test_no_layer_index(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "C8",
            "layers.1.fa_k.scale": "C8",
        })
        result = config.is_fa_quant_layer("model.attn")
        self.assertFalse(result)


class TestEnablingFaQuant(TestBase):

    def test_decode_instance_with_fa_quant_layer(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "C8",
            "layers.1.fa_k.scale": "C8",
        })
        mock_vllm_config = MagicMock()
        mock_vllm_config.kv_transfer_config.is_kv_consumer = True
        mock_vllm_config.kv_transfer_config.is_kv_producer = False

        result = config.enabling_fa_quant(mock_vllm_config, "layers.1.attn")
        self.assertTrue(result)

    def test_non_decode_instance(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "C8",
            "layers.1.fa_k.scale": "C8",
        })
        mock_vllm_config = MagicMock()
        mock_vllm_config.kv_transfer_config.is_kv_consumer = False
        mock_vllm_config.kv_transfer_config.is_kv_producer = True

        result = config.enabling_fa_quant(mock_vllm_config, "layers.1.attn")
        self.assertFalse(result)

    def test_is_fa_quant_layer_false(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "C8",
            "layers.1.fa_k.scale": "C8",
        })
        mock_vllm_config = MagicMock()
        mock_vllm_config.kv_transfer_config.is_kv_consumer = True
        mock_vllm_config.kv_transfer_config.is_kv_producer = False

        result = config.enabling_fa_quant(mock_vllm_config, "layers.2.attn")
        self.assertFalse(result)


class TestIsIndexerQuantLayer(TestBase):

    def test_enable_indexer_quant_true_matching_layer(self):
        config = AscendModelSlimConfig({
            "indexer_quant_type": "INT8",
            "layers.1.indexer.quant_type": "INT8",
        })
        result = config.is_indexer_quant_layer("layers.1.attn")
        self.assertTrue(result)

    def test_enable_indexer_quant_false(self):
        config = AscendModelSlimConfig({})
        result = config.is_indexer_quant_layer("layers.1.attn")
        self.assertFalse(result)

    def test_non_matching_layer_index(self):
        config = AscendModelSlimConfig({
            "indexer_quant_type": "INT8",
            "layers.1.indexer.quant_type": "INT8",
        })
        result = config.is_indexer_quant_layer("layers.2.attn")
        self.assertFalse(result)

    def test_no_layer_index(self):
        config = AscendModelSlimConfig({
            "indexer_quant_type": "INT8",
            "layers.1.indexer.quant_type": "INT8",
        })
        result = config.is_indexer_quant_layer("model.attn")
        self.assertFalse(result)


class TestGetKvQuantDtype(TestBase):

    def test_enable_fa_quant_true_with_mla(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "C8",
            "layers.1.fa_k.scale": "C8",
        })
        mock_model_config = MagicMock()
        mock_model_config.dtype = torch.float16
        mock_model_config.use_mla = True

        k_dtype, v_dtype = config.get_kv_quant_dtype("layers.1.attn", torch.float16, mock_model_config)
        self.assertEqual(k_dtype, torch.int8)
        self.assertEqual(v_dtype, torch.float16)

    def test_enable_fa_quant_true_without_mla(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "C8",
            "layers.1.fa_k.scale": "C8",
        })
        mock_model_config = MagicMock()
        mock_model_config.dtype = torch.float16
        mock_model_config.use_mla = False

        k_dtype, v_dtype = config.get_kv_quant_dtype("layers.1.attn", torch.float16, mock_model_config)
        self.assertEqual(k_dtype, torch.int8)
        self.assertEqual(v_dtype, torch.int8)

    def test_enable_fa_quant_false(self):
        config = AscendModelSlimConfig({})
        mock_model_config = MagicMock()
        mock_model_config.dtype = torch.float16
        mock_model_config.use_mla = True

        k_dtype, v_dtype = config.get_kv_quant_dtype("layers.1.attn", torch.float16, mock_model_config)
        self.assertEqual(k_dtype, torch.float16)
        self.assertEqual(v_dtype, torch.float16)


class TestGetKvQuantSplitFactor(TestBase):

    @patch("vllm_ascend.quantization.modelslim_config.calc_split_factor")
    def test_enable_fa_quant_true(self, mock_calc_split_factor):
        mock_calc_split_factor.return_value = 2.0
        config = AscendModelSlimConfig({
            "fa_quant_type": "C8",
            "layers.1.fa_k.scale": "C8",
        })
        kv_head_dim_list = [64, 64]

        result = config.get_kv_quant_split_factor("layers.1.attn", kv_head_dim_list)
        self.assertEqual(result, 2.0)
        mock_calc_split_factor.assert_called_once_with([64, 128])

    @patch("vllm_ascend.quantization.modelslim_config.calc_split_factor")
    def test_enable_fa_quant_false(self, mock_calc_split_factor):
        mock_calc_split_factor.return_value = 1.0
        config = AscendModelSlimConfig({})
        kv_head_dim_list = [64, 64]

        result = config.get_kv_quant_split_factor("layers.1.attn", kv_head_dim_list)
        self.assertEqual(result, 1.0)
        mock_calc_split_factor.assert_called_once_with([64, 64])


class TestAddKvcacheQuantMetadata(TestBase):

    def test_with_fa_quant_type(self):
        config = AscendModelSlimConfig({
            "fa_quant_type": "C8",
            "layers.1.fa_k.scale": "C8",
            "layers.2.fa_k.scale": "C8",
        })
        config._add_kvcache_quant_metadata()

        self.assertTrue(config.enable_fa_quant)
        self.assertIn(1, config.kvcache_quant_layers)
        self.assertIn(2, config.kvcache_quant_layers)
        self.assertFalse(config.enable_indexer_quant)
        self.assertEqual(config.indexer_quant_layers, [])

    def test_with_indexer_quant_type(self):
        config = AscendModelSlimConfig({
            "indexer_quant_type": "INT8",
            "layers.1.indexer.quant_type": "INT8",
            "layers.3.indexer.quant_type": "INT8",
        })
        config._add_kvcache_quant_metadata()

        self.assertFalse(config.enable_fa_quant)
        self.assertEqual(config.kvcache_quant_layers, [])
        self.assertTrue(config.enable_indexer_quant)
        self.assertIn(1, config.indexer_quant_layers)
        self.assertIn(3, config.indexer_quant_layers)

    def test_with_neither_quant_type(self):
        config = AscendModelSlimConfig({})
        config._add_kvcache_quant_metadata()

        self.assertFalse(config.enable_fa_quant)
        self.assertEqual(config.kvcache_quant_layers, [])
        self.assertFalse(config.enable_indexer_quant)
        self.assertEqual(config.indexer_quant_layers, [])
