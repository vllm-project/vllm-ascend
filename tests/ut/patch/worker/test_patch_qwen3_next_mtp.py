# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.patch.worker import patch_qwen3_next_mtp


class TestQwen3NextMTPBindKvCache(unittest.TestCase):
    def test_bind_kv_cache_appends_all_layers_with_same_layer_index(self):
        layer0 = "model.layers.0.self_attn"
        mtp_layer0 = "model.mtp.layers.0.self_attn"
        layer1 = "model.layers.1.self_attn"

        cache0 = object()
        mtp_cache0 = object()
        cache1 = object()

        kv_caches = {
            layer0: cache0,
            mtp_layer0: mtp_cache0,
            layer1: cache1,
        }
        forward_context = {layer_name: SimpleNamespace(kv_cache=None) for layer_name in kv_caches}
        runner_kv_caches: list[object] = []

        layer_index_map = {
            layer0: 0,
            mtp_layer0: 0,
            layer1: 1,
        }

        def fake_extract_layer_index(layer_name, num_attn_module):
            return layer_index_map[layer_name]

        with patch.object(
            patch_qwen3_next_mtp,
            "extract_layer_index",
            side_effect=fake_extract_layer_index,
        ):
            patch_qwen3_next_mtp.bind_kv_cache(
                kv_caches,
                forward_context,
                runner_kv_caches,
            )

        self.assertEqual(runner_kv_caches, [cache0, mtp_cache0, cache1])

        for layer_name, kv_cache in kv_caches.items():
            self.assertIs(forward_context[layer_name].kv_cache, kv_cache)


if __name__ == "__main__":
    unittest.main()
