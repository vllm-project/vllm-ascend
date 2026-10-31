# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from dataclasses import replace
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_layout import CacheComponentBinding
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.key_codec import PoolKeyIdentity
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.prefix_object import (
    component_prefix_id,
    full_prefix_objects,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import CacheSlice, TargetCacheSlice


class TestPrefixObject(unittest.TestCase):
    def inputs(self):
        bindings, pages = {}, []
        # Deliberately register layers backwards; ordinary pool order is by
        # physical layer and component, not cache-dict insertion order.
        for index in (1, 0):
            name, address = f"model.layers.{index}", 100 + 100 * index
            bindings[name] = [
                CacheComponentBinding(name, 0, 0, "FullAttentionSpec", "bf16", address, (4, 2, 1), (4, 2, 2), 2, 4)
            ]
            pages.append(TargetCacheSlice(CacheSlice(name, "0", "kv", 0, 2, 0, 1, 2, 2, 4), address + 4))
        request = SimpleNamespace(block_ids=([1],), group_hashes=(("prefix-hash",),), token_start=0, token_end=2)
        profile = SimpleNamespace(profile_id="source", key_identity=PoolKeyIdentity("memcache-v1", "model", 1, 1))
        return bindings, pages, request, profile

    def build(self, bindings, pages, request, profile):
        return full_prefix_objects(
            bindings, pages, request, profile, rank=0, stage=0, group_block_sizes=[2], base_layers=2
        )

    def test_full_block_uses_ordinary_key_and_multilayer_component_offsets(self):
        objects, layers, refs, covered = self.build(*self.inputs())
        self.assertEqual(objects, {"model@prefix-hash@0": 8})
        self.assertEqual(
            [(item.source, item.object_offset, item.size) for item in layers["model.layers.0"]], [(104, 0, 4)]
        )
        self.assertEqual(
            [(item.source, item.object_offset, item.size) for item in layers["model.layers.1"]], [(204, 4, 4)]
        )
        self.assertEqual(
            [(part.cache.layer, part.offset) for part in refs[0].slices], [("model.layers.0", 0), ("model.layers.1", 4)]
        )
        self.assertEqual(covered, {0, 1})

    def test_missing_layer_or_partial_token_range_never_publishes_full_key(self):
        for missing in (True, False):
            bindings, pages, request, profile = self.inputs()
            if missing:
                pages.pop()
            else:
                request.token_end = 1
            objects, layers, refs, covered = self.build(bindings, pages, request, profile)
            self.assertFalse(objects or layers or refs or covered)

    def test_attempt_identity_does_not_change_shared_key(self):
        bindings, pages, request, profile = self.inputs()
        request.transfer_id, request.attempt_id = "first", "first"
        first = self.build(bindings, pages, request, profile)
        request.transfer_id, request.attempt_id = "retry", "retry"
        second = self.build(bindings, pages, request, profile)
        self.assertEqual(first, second)

    def test_qualified_full_keys_isolate_equal_byte_sizes_with_different_tensor_orders(self):
        inputs = self.inputs()
        bindings, pages, request, profile = inputs

        def build(parts):
            return full_prefix_objects(
                bindings,
                parts,
                request,
                profile,
                rank=0,
                stage=0,
                group_block_sizes=[2],
                base_layers=2,
                schema_qualified=True,
            )

        first = build(pages)
        # Equal span and vector bytes, but a different semantic byte format.
        changed = [replace(page, cache=replace(page.cache, format_id="other-format")) for page in pages]
        second = build(changed)
        self.assertTrue(first[0] and second[0])
        self.assertNotEqual(set(first[0]), set(second[0]))
        request.attempt_id = "retry"
        self.assertEqual(first, build(pages))

    def test_component_identity_is_stable_across_attempts_and_physical_allocations(self):
        bindings, pages, request, _ = self.inputs()
        binding = bindings["model.layers.1"][0]
        cache = pages[0].cache
        spec = SimpleNamespace(tokens_per_state=1)
        first = component_prefix_id(cache, binding, spec, request, group_block_size=2)
        request.attempt_id = "retry"
        changed = replace(binding, address=binding.address + 4096)
        assert component_prefix_id(cache, changed, spec, request, group_block_size=2) == first
        assert (
            component_prefix_id(replace(cache, head_start=1, head_end=2), changed, spec, request, group_block_size=2)
            != first
        )

    def test_partial_component_identity_includes_full_input_digest(self):
        bindings, pages, request, _ = self.inputs()
        request.token_end, request.request_digest = 1, "input-a"
        binding = bindings["model.layers.1"][0]
        spec = SimpleNamespace(tokens_per_state=1)
        first = component_prefix_id(pages[0].cache, binding, spec, request, group_block_size=2)
        request.request_digest = "input-b"
        assert component_prefix_id(pages[0].cache, binding, spec, request, group_block_size=2) != first

    def test_checkpoint_for_a_later_prefix_never_uses_an_earlier_block_hash(self):
        bindings, pages, request, profile = self.inputs()
        pages[0] = replace(
            pages[0],
            cache=replace(
                pages[0].cache,
                format_id="mamba-v2:{}",
                token_start=3,
                token_end=4,
                vector_bytes=4,
                token_stride=4,
                head_stride=4,
            ),
        )
        self.assertFalse(self.build(bindings, pages, request, profile)[0])

    def test_pp_stage_uses_ordinary_global_origin_and_reserved_size(self):
        bindings, pages, request, _ = self.inputs()
        profile = SimpleNamespace(profile_id="source", key_identity=PoolKeyIdentity("memcache-v1", "model", 1, 2))
        objects, layers, _, _ = full_prefix_objects(
            bindings,
            pages,
            request,
            profile,
            rank=0,
            stage=1,
            group_block_sizes=[2],
            base_layers=4,
            layer_offset=2,
        )
        self.assertEqual(objects, {"model@pp1@prefix-hash@0": 16})
        self.assertEqual(layers["model.layers.0"][0].object_offset, 8)
        self.assertEqual(layers["model.layers.1"][0].object_offset, 12)

    def test_aligned_state_keeps_tp_rank_even_when_mla_rank_is_replicated(self):
        bindings, pages, request, profile = self.inputs()
        bindings = {
            name: [replace(item, spec_type="vllm.v1.kv_cache_interface.MambaSpec") for item in parts]
            for name, parts in bindings.items()
        }
        pages = [
            replace(
                page,
                cache=replace(
                    page.cache, format_id="mamba-v2:{}", token_start=1, vector_bytes=4, token_stride=4, head_stride=4
                ),
            )
            for page in pages
        ]
        objects, _, refs, _ = full_prefix_objects(
            bindings,
            pages,
            request,
            profile,
            rank=0,
            state_rank=1,
            stage=0,
            group_block_sizes=[2],
            base_layers=2,
        )
        self.assertEqual(objects, {"model@prefix-hash@1": 8})
        self.assertEqual(refs[0].head_or_tp_rank, 1)
