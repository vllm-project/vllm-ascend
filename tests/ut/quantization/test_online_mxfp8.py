# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, get_quant_config

from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod, AscendLinearMethod
from vllm_ascend.quantization.methods import w8a8_mxfp8 as methods
from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig
from vllm_ascend.utils import AscendDeviceType


def fake_quantize(rows, dst_type):
    groups = rows.shape[-1] // 32
    scales = torch.arange(rows.shape[0] * groups, dtype=torch.int64).reshape(rows.shape[0], groups)
    return rows.to(dst_type), scales.to(torch.uint8)


def storage_snapshot(layer):
    return {
        name: (id(param), param.data_ptr(), tuple(param.shape), tuple(param.stride()), param.dtype)
        for name, param in layer.named_parameters()
    }


class TestOnlineMXFP8(unittest.TestCase):
    def setUp(self):
        config = SimpleNamespace(
            compilation_config=SimpleNamespace(mode=None),
            model_config=SimpleNamespace(enforce_eager=True),
            quant_config=SimpleNamespace(quant_description={"group_size": 32}),
        )
        ascend = SimpleNamespace(
            eplb_config=SimpleNamespace(dynamic_eplb=False),
            multistream_overlap_gate=False,
        )
        for target, value in (
            ("ensure_mxfp8_linear_available", None),
            ("ensure_mxfp8_moe_available", None),
            ("get_current_vllm_config", config),
            ("get_ascend_config", ascend),
        ):
            mock = patch.object(methods, target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch.object(methods.torch_npu, "npu_dynamic_mx_quant", side_effect=fake_quantize, create=True)
        self.quantize = mock.start()
        self.addCleanup(mock.stop)
        for target, value in (
            ("vllm_ascend.quantization.method_adapters.enable_dsa_cp_with_layer_shard", False),
            ("vllm_ascend.quantization.method_adapters.FusedMoEMethodBase.__init__", None),
        ):
            mock = patch(target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def linear_layer(self, dtype, input_size=96):
        layer = torch.nn.Module()
        adapter = AscendLinearMethod(methods.AscendMXFP8OnlineLinearMethod())
        layer.quant_method = adapter
        adapter.create_weights(layer, input_size, [8], input_size, 8, dtype, weight_loader=default_weight_loader)
        return layer

    def moe_layer(self, dtype):
        layer = torch.nn.Module()
        # Avoid unrelated fused-kernel configuration; create_weights and loading
        # still run through the real Ascend adapter implementation.
        adapter = object.__new__(AscendFusedMoEMethod)
        adapter.quant_method = methods.AscendMXFP8OnlineMoEMethod()
        layer.quant_method = adapter
        adapter.create_weights(layer, 2, 96, 96, dtype, weight_loader=default_weight_loader)
        return layer

    def test_linear_and_moe_native_reload_preserves_graph_storage(self):
        for dtype in (torch.float16, torch.bfloat16):
            for make_layer, names in (
                (self.linear_layer, ("weight",)),
                (self.moe_layer, ("w13_weight", "w2_weight")),
            ):
                with self.subTest(dtype=dtype, names=names):
                    layer = make_layer(dtype)
                    checkpoint_shapes = {name: tuple(getattr(layer, name).shape) for name in names}
                    loader = layer.quant_method
                    record_metadata_for_reloading(layer)
                    for name in names:
                        param = getattr(layer, name)
                        param.data.fill_(1)
                    loader.process_weights_after_loading(layer)
                    before = storage_snapshot(layer)
                    for value in (2, 3):
                        initialize_layerwise_reload(layer)
                        for name in names:
                            param = getattr(layer, name)
                            self.assertEqual(param.dtype, dtype)
                            self.assertEqual(tuple(param.shape), checkpoint_shapes[name])
                            loaded = torch.full(checkpoint_shapes[name], value, dtype=dtype)
                            param.weight_loader(param, loaded)
                        finalize_layerwise_reload(layer, SimpleNamespace(dtype=dtype))
                        self.assertEqual(before, storage_snapshot(layer))
                        for name in names:
                            runtime = getattr(layer, name)
                            self.assertTrue(runtime.is_contiguous())
                            torch.testing.assert_close(runtime.float(), torch.full(runtime.shape, float(value)))
                    calls = self.quantize.call_count
                    loader.process_weights_after_loading(layer)
                    self.assertEqual(calls, self.quantize.call_count)

    def test_odd_groups_and_padded_scale_storage(self):
        weight = torch.ones(2, 96, dtype=torch.bfloat16)
        with patch.object(methods.torch_npu, "npu_dynamic_mx_quant") as quant:
            quant.return_value = (
                weight.to(torch.float8_e4m3fn),
                torch.tensor([[1, 2, 3, 99], [4, 5, 6, 99]], dtype=torch.uint8),
            )
            _, scales = methods._quantize_online_weight(weight)
        expected = torch.tensor([[[1, 2], [4, 5]], [[3, 0], [6, 0]]], dtype=torch.uint8)
        torch.testing.assert_close(scales, expected)

    def test_rejects_invalid_dtype_and_reduction_dimension(self):
        for dtype, dimension in ((torch.float32, 64), (torch.bfloat16, 33), (torch.float16, 0)):
            with self.subTest(dtype=dtype, dimension=dimension), self.assertRaises(ValueError):
                methods.AscendMXFP8OnlineLinearMethod().get_weight(dimension, 8, dtype)
        self.quantize.assert_not_called()

    def test_invalid_scale_does_not_publish_partial_moe_weights(self):
        layer = self.moe_layer(torch.bfloat16)
        before = storage_snapshot(layer)
        self.quantize.side_effect = (
            fake_quantize(layer.w13_weight.reshape(-1, 96), torch.float8_e4m3fn),
            (layer.w2_weight.reshape(-1, 96).to(torch.float8_e4m3fn), torch.ones(1, dtype=torch.float32)),
        )
        with self.assertRaisesRegex(ValueError, "one-byte"):
            layer.quant_method.process_weights_after_loading(layer)
        self.assertEqual(before, storage_snapshot(layer))

    def test_offline_allocations_are_still_fp8(self):
        linear = methods.AscendW8A8MXFP8DynamicLinearMethod()
        moe = methods.AscendW8A8MXFP8DynamicFusedMoEMethod()
        self.assertEqual(linear.get_weight(64, 8, torch.bfloat16)["weight"].dtype, torch.float8_e4m3fn)
        for weight in moe.get_weight(2, 64, 64, torch.bfloat16).values():
            self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertFalse(getattr(linear, "online_quantization", False))
        self.assertFalse(getattr(moe, "online_quantization", False))
        self.quantize.assert_not_called()

    def online_config(self, **kwargs):
        description = {
            "online_quantization": True,
            "model_quant_type": "W8A8_MXFP8",
            "group_size": 32,
            **kwargs,
        }
        with patch(
            "vllm_ascend.quantization.modelslim_config.get_ascend_device_type", return_value=AscendDeviceType.A5
        ):
            return AscendModelSlimConfig.from_config_dict_json(description)

    def test_existing_ascend_config_hook_accepts_dict_and_json(self):
        import json

        description = self.online_config(ignore=["lm_head"]).quant_description
        with patch(
            "vllm_ascend.quantization.modelslim_config.get_ascend_device_type", return_value=AscendDeviceType.A5
        ):
            parsed = AscendModelSlimConfig.from_config_dict_json(json.dumps(description))
        self.assertTrue(parsed.online_quantization)
        self.assertEqual(parsed._online_ignored_layers, ["lm_head"])
        with self.assertRaisesRegex(ValueError, "JSON object"):
            AscendModelSlimConfig.from_config_dict_json("[]")

    def test_native_vllm_config_loader_uses_existing_hook(self):
        model_config = SimpleNamespace(
            quantization="ascend",
            hf_config=SimpleNamespace(),
            hf_text_config=SimpleNamespace(),
            hf_overrides={"quantization_config_dict_json": self.online_config().quant_description},
        )
        with (
            patch(
                "vllm.model_executor.model_loader.weight_utils.get_quantization_config",
                return_value=AscendModelSlimConfig,
            ),
            patch("vllm_ascend.quantization.modelslim_config.get_ascend_device_type", return_value=AscendDeviceType.A5),
        ):
            selected = get_quant_config(model_config, SimpleNamespace())
        self.assertIsInstance(selected, AscendModelSlimConfig)
        self.assertTrue(selected.online_quantization)

    def test_selects_online_schemes_and_preserves_ignored_methods(self):
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        linear = object.__new__(LinearBase)
        torch.nn.Module.__init__(linear)
        moe = object.__new__(RoutedExperts)
        torch.nn.Module.__init__(moe)
        moe.moe_config = SimpleNamespace()
        config = self.online_config()
        self.assertIsInstance(config.get_quant_method(linear, "linear"), AscendLinearMethod)
        selected = config.get_quant_method(moe, "moe", tid2eid="sentinel")
        self.assertIsInstance(selected, AscendFusedMoEMethod)
        self.assertEqual(selected.tid2eid, "sentinel")
        config = self.online_config(ignore=["linear", "moe"])
        self.assertIsInstance(config.get_quant_method(linear, "linear"), AscendUnquantizedLinearMethod)
        ignored_moe = object()
        with patch(
            "vllm_ascend.ops.fused_moe.fused_moe.AscendUnquantizedFusedMoEMethod",
            return_value=ignored_moe,
        ) as unquantized_method:
            self.assertIs(
                config.get_quant_method(moe, "moe", tid2eid="ignored"),
                ignored_moe,
            )
        unquantized_method.assert_called_once_with(moe.moe_config, tid2eid="ignored")
        self.assertIsNone(config.get_quant_method(torch.nn.Module(), "other"))

    def test_ignore_handles_regex_and_consistent_fused_shards(self):
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        layer = object.__new__(LinearBase)
        torch.nn.Module.__init__(layer)
        prefix = "model.layers.0.self_attn.qkv_proj"
        config = self.online_config(ignore=["re:.*[qkv]_proj"])
        config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        self.assertIsInstance(config.get_quant_method(layer, prefix), AscendUnquantizedLinearMethod)
        config._online_ignored_layers = ["model.layers.0.self_attn.q_proj"]
        with self.assertRaisesRegex(ValueError, "different quantization schemes"):
            config.get_quant_method(layer, prefix)

    def test_attention_and_embedding_keep_unquantized_behavior(self):
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
            VocabParallelEmbedding,
        )

        class ConcreteAttentionLayer(AttentionLayerBase):
            def get_attn_backend(self):
                return None

            def get_kv_cache_spec(self):
                return None

        config = self.online_config()
        attention = object.__new__(ConcreteAttentionLayer)
        embedding = object.__new__(VocabParallelEmbedding)
        torch.nn.Module.__init__(embedding)
        self.assertIsNone(config.get_quant_method(attention, "attention"))
        self.assertIsInstance(config.get_quant_method(embedding, "embedding"), UnquantizedEmbeddingMethod)

    def test_invalid_online_options_are_rejected(self):
        for override in (
            {"online_quantization": "true"},
            {"model_quant_type": "W8A8"},
            {"group_size": 64},
            {"group_size": True},
            {"ignore": "lm_head"},
            {"ignore": [1]},
            {"layer.weight": "FLOAT"},
            {"kv_cache_type": "C8"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.online_config(**override)

    def test_a5_gate_is_only_applied_to_online_quantization(self):
        with patch("vllm_ascend.quantization.modelslim_config.get_ascend_device_type", return_value=object()):
            with self.assertRaisesRegex(ValueError, "A5"):
                AscendModelSlimConfig.from_config_dict_json(
                    {"online_quantization": True, "model_quant_type": "W8A8_MXFP8"}
                )
            config = AscendModelSlimConfig.from_config({})
            self.assertFalse(config.online_quantization)

    def test_mapper_applies_to_online_ignore_once(self):
        config = self.online_config(ignore=["model.lm_head"])
        mapper = SimpleNamespace(
            apply_list=lambda names: [name.replace("model.", "") for name in names],
            apply_dict=lambda values: values,
        )
        with patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config_or_none", return_value=None):
            config.apply_vllm_mapper(mapper)
            config.apply_vllm_mapper(mapper)
        self.assertEqual(config._online_ignored_layers, ["lm_head"])


if __name__ == "__main__":
    unittest.main()
