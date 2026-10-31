# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lock wire formats across the bound scheduler/worker key interface."""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import memcache_backend, mooncake_layerwise


class TestBoundLayerwiseKeys(unittest.TestCase):
    @staticmethod
    def make_config(pp_size=1):
        return SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size, tensor_parallel_size=2, rank=0),
            model_config=SimpleNamespace(
                get_layers_start_end_indices=lambda p: ((0, 3), (3, 7))[p.rank // 2],
                get_total_num_hidden_layers=lambda: 7,
                compute_hash=lambda: "model-config",
            ),
            cache_config=SimpleNamespace(block_size=16, cache_dtype="auto", compute_hash=lambda: "cache-config"),
            speculative_config=None,
        )

    @staticmethod
    def make_cache_config(num_groups):
        return SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(
                    [f"model.layers.{group}.kv"],
                    FullAttentionSpec(block_size=16 * (group + 1), num_kv_heads=1, head_size=1, dtype="uint8"),
                )
                for group in range(num_groups)
            ]
        )

    def test_wire_compatibility_and_all_stage_writer_keys(self):
        for protocol in (mooncake_layerwise, memcache_backend):
            for pp_size in (1, 2):
                for num_groups in (1, 2):
                    with self.subTest(protocol=protocol.__name__, pp_size=pp_size, num_groups=num_groups):
                        config = self.make_config(pp_size)
                        cache = self.make_cache_config(num_groups)
                        sizes = [16 * (group + 1) for group in range(num_groups)]
                        builder = protocol.bind_layerwise_keys(
                            vllm_config=config,
                            kv_cache_config=cache,
                            model_name="model",
                            use_hybrid=num_groups > 1,
                            grouped_block_size=sizes,
                        )
                        namespace = mooncake_layerwise.layerwise_topology_namespace(config, cache)
                        layout = mooncake_layerwise.hybrid_layout_id(cache, 2, namespace=namespace)
                        for group in range(num_groups):
                            expected = []
                            for stage in range(pp_size):
                                for head in range(2):
                                    if protocol is memcache_backend:
                                        group_tag = f"@{group}" if num_groups > 1 else ""
                                        stage_tag = f"@pp{stage}" if pp_size > 1 else ""
                                        key = f"model{group_tag}{stage_tag}@hash@{head}"
                                    elif num_groups > 1:
                                        stage_tag = f"@pp_rank:{stage}" if pp_size > 1 else ""
                                        key = (
                                            f"model@mooncake_hybrid_v1:{layout}{stage_tag}"
                                            f"@group:{group}@block:{sizes[group]}@hash@{head}"
                                        )
                                    else:
                                        stage_tag = f"@{namespace}@pp_rank:{stage}" if pp_size > 1 else ""
                                        key = f"model{stage_tag}@hash@{head}"
                                    expected.append(key)
                                    self.assertEqual(builder.make_full_key(group, "hash", head, stage), key)
                            self.assertEqual(builder.make_hit_check_keys(group, "hash", 2), expected)
                            # MLA replicas require only the saving head on each stage.
                            self.assertEqual(builder.make_hit_check_keys(group, "hash", 1), expected[::2])
                            self.assertEqual(builder.make_hit_check_keys(group, "hash", 0), [])

    def test_binding_snapshots_identity_and_does_not_rehash_on_lookup(self):
        for protocol in (mooncake_layerwise, memcache_backend):
            with self.subTest(protocol=protocol.__name__):
                config = self.make_config(2)
                cache = self.make_cache_config(2)
                sizes = [16, 32]
                builder = protocol.bind_layerwise_keys(
                    vllm_config=config,
                    kv_cache_config=cache,
                    model_name="model",
                    use_hybrid=True,
                    grouped_block_size=sizes,
                )
                expected = builder.make_hit_check_keys(1, "hash", 2)
                # Neither caller-owned lists nor config objects remain live inputs.
                config.parallel_config.pipeline_parallel_size = 4
                sizes[1] = 64
                cache.kv_cache_groups[1].layer_names.append("model.layers.9.kv")
                with (
                    patch.object(mooncake_layerwise, "hybrid_layout_id", side_effect=AssertionError("rehash")),
                    patch.object(
                        mooncake_layerwise, "layerwise_topology_namespace", side_effect=AssertionError("rehash")
                    ),
                ):
                    self.assertEqual(builder.make_hit_check_keys(1, "hash", 2), expected)

    def test_mooncake_rejects_empty_pp_groups_at_binding(self):
        cache = self.make_cache_config(2)
        cache.kv_cache_groups[1].layer_names.clear()
        with self.assertRaisesRegex(ValueError, "empty groups"):
            mooncake_layerwise.bind_layerwise_keys(
                vllm_config=self.make_config(2),
                kv_cache_config=cache,
                model_name="model",
                use_hybrid=True,
                grouped_block_size=[16, 32],
            )

    def test_mooncake_distinguishes_layouts_and_groups(self):
        config = self.make_config()
        cache = self.make_cache_config(2)
        changed_cache = deepcopy(cache)
        changed_cache.kv_cache_groups[1].layer_names.append("model.layers.2.kv")
        builders = [
            mooncake_layerwise.bind_layerwise_keys(
                vllm_config=config,
                kv_cache_config=c,
                model_name="model",
                use_hybrid=True,
                grouped_block_size=[16, 32],
            )
            for c in (cache, changed_cache)
        ]
        self.assertNotEqual(builders[0].make_full_key(0, "hash", 0, 0), builders[0].make_full_key(1, "hash", 0, 0))
        self.assertNotEqual(builders[0].make_full_key(1, "hash", 0, 0), builders[1].make_full_key(1, "hash", 0, 0))
