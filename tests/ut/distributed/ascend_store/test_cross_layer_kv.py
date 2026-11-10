# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cross_layer_kv_hooks import (
    dispatch_cross_layer_kv_hook,
    supports_cross_layer_kv_hooks,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.transfer_mode import (
    KVPoolTransferMode,
    kv_source_layer_ids,
    resolve_kv_pool_transfer_mode,
)

V41_SOURCES = (2, 8, 14, 20)


class TestCrossLayerKVHookDispatch(unittest.TestCase):
    def test_nested_multi_connector_reaches_all_children(self):
        wait = MagicMock()
        other_wait = MagicMock()
        connector = SimpleNamespace(
            sub_connectors=[
                SimpleNamespace(),
                SimpleNamespace(sub_connectors=[SimpleNamespace(wait_for_cross_layer_kv_load=wait)]),
                SimpleNamespace(wait_for_cross_layer_kv_load=other_wait),
            ]
        )
        dispatch_cross_layer_kv_hook(connector, "wait_for_cross_layer_kv_load", 8)
        wait.assert_called_once_with(8)
        other_wait.assert_called_once_with(8)

    def test_direct_hook_owns_forwarding(self):
        save = MagicMock()
        child_save = MagicMock()
        connector = SimpleNamespace(save_kv_group=save, sub_connectors=[SimpleNamespace(save_kv_group=child_save)])
        dispatch_cross_layer_kv_hook(connector, "save_kv_group", 14)
        save.assert_called_once_with(14)
        child_save.assert_not_called()


def _vllm_config(extra, architectures=("DeepseekV41ForCausalLM",), text_config=None):
    extra = {"backend": "memcache", **extra}
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        model_config=SimpleNamespace(
            architectures=list(architectures),
            hf_text_config=text_config or SimpleNamespace(num_hidden_layers=40, kv_source_layer_ids=list(V41_SOURCES)),
        ),
    )


class TestResolveKVPoolTransferMode(unittest.TestCase):
    def test_disabled_by_default_even_for_shared_models(self):
        self.assertIs(resolve_kv_pool_transfer_mode(_vllm_config({})), KVPoolTransferMode.BULK)
        self.assertIs(resolve_kv_pool_transfer_mode(_vllm_config({"use_layerwise": False})), KVPoolTransferMode.BULK)

    def test_ds4_compression_keeps_physical_layerwise(self):
        config = _vllm_config(
            {"use_layerwise": True},
            architectures=("DeepseekV4ForCausalLM",),
            text_config=SimpleNamespace(num_hidden_layers=40, compress_ratios=[0, 4, 128] * 14),
        )
        self.assertIs(resolve_kv_pool_transfer_mode(config), KVPoolTransferMode.PHYSICAL_LAYER)

    def test_shared_topology_uses_existing_switch_regardless_of_architecture(self):
        for architecture in (
            "DeepseekV41ForCausalLM",
            "DeepseekV41ForConditionalGeneration",
            "FutureSharedKVForCausalLM",
        ):
            config = _vllm_config(
                {"use_layerwise": True},
                architectures=(architecture,),
                text_config=SimpleNamespace(num_hidden_layers=16, kv_source_layer_ids=[1, 5, 9]),
            )
            self.assertIs(resolve_kv_pool_transfer_mode(config), KVPoolTransferMode.CROSS_LAYER_KV)

    def test_all_layers_own_kv_keeps_physical_layerwise(self):
        config = _vllm_config(
            {"use_layerwise": True}, text_config={"num_hidden_layers": 4, "kv_source_layer_ids": [0, 1, 2, 3]}
        )
        self.assertIs(resolve_kv_pool_transfer_mode(config), KVPoolTransferMode.PHYSICAL_LAYER)

    def test_nested_raw_text_config_and_explicit_text_config_priority(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"backend": "memcache"}),
            model_config=SimpleNamespace(
                hf_config={"text_config": {"num_hidden_layers": 16, "kv_source_layer_ids": [1, 9]}}
            ),
        )
        self.assertIs(resolve_kv_pool_transfer_mode(config, True), KVPoolTransferMode.CROSS_LAYER_KV)
        config.model_config.hf_text_config = {"num_hidden_layers": 16, "kv_source_layer_ids": []}
        self.assertIs(resolve_kv_pool_transfer_mode(config, True), KVPoolTransferMode.PHYSICAL_LAYER)

    def test_invalid_source_declarations_do_not_silently_enter_old_path(self):
        for sources in ([1, 1], [5, 1], [-1, 5], [1, 16], [True, 5], ["1", 5], "1,5"):
            with self.subTest(sources=sources):
                text = {"num_hidden_layers": 16, "kv_source_layer_ids": sources}
                self.assertFalse(kv_source_layer_ids(text))
                with self.assertRaisesRegex(ValueError, "topology"):
                    resolve_kv_pool_transfer_mode(_vllm_config({"use_layerwise": True}, text_config=text))
                self.assertIs(
                    resolve_kv_pool_transfer_mode(_vllm_config({}, text_config=text)), KVPoolTransferMode.BULK
                )

    def test_every_backbone_consumer_requires_hooks(self):
        context = {
            str(layer): SimpleNamespace(layer_idx=layer, supports_cross_layer_kv_transfer_hooks=True)
            for layer in range(16)
        }
        self.assertTrue(supports_cross_layer_kv_hooks(context, 16))
        context["8"].supports_cross_layer_kv_transfer_hooks = False
        self.assertFalse(supports_cross_layer_kv_hooks(context, 16))
        context["8"].supports_cross_layer_kv_transfer_hooks = True
        del context["9"]
        self.assertFalse(supports_cross_layer_kv_hooks(context, 16))

    def test_mooncake_block_key_never_selects_cross_layer(self):
        for enabled in (False, True):
            config = _vllm_config({"backend": "mooncake", "use_layerwise": enabled})
            expected = KVPoolTransferMode.PHYSICAL_LAYER if enabled else KVPoolTransferMode.BULK
            self.assertIs(resolve_kv_pool_transfer_mode(config), expected)
