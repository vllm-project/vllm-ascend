# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.executor import PoolReadRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import (
    CacheSlice,
    RestorePlanner,
    SourceCacheSlice,
    TargetCacheSlice,
)


class TestRestorePlanner(unittest.TestCase):
    def test_disjoint_pages_do_not_rescan_all_earlier_source_pages(self):
        pages = 256
        sources = [
            SourceCacheSlice(self.cache(tokens=(page * 4, page * 4 + 4), heads=(0, 1)), str(page), 8, 0)
            for page in range(pages)
        ]
        targets = [TargetCacheSlice(source.cache, 1000 + page * 8) for page, source in enumerate(sources)]
        inspections = 0

        def counted_getattribute(cache, name):
            nonlocal inspections
            if name in ("format_id", "token_start", "token_end"):
                inspections += 1
            return object.__getattribute__(cache, name)

        # Count candidate work rather than timing a machine-dependent benchmark.
        # The same cache descriptions are used for the source and target pages.
        with patch.object(CacheSlice, "__getattribute__", counted_getattribute):
            plan = RestorePlanner().build(sources, targets)
        self.assertEqual(plan, [PoolReadRange(str(page), 8, 0, 1000 + page * 8, 8) for page in range(pages)])
        self.assertLess(inspections, pages * 32)

    def test_interval_lookup_preserves_input_replica_priority_and_straddling_sources(self):
        short = SourceCacheSlice(self.cache(tokens=(8, 10), heads=(0, 1)), "short-first", 4, 0)
        long = SourceCacheSlice(self.cache(tokens=(0, 20), heads=(0, 1)), "long-second", 40, 0)
        target = TargetCacheSlice(self.cache(tokens=(7, 12), heads=(0, 1)), 100)
        self.assertEqual(
            RestorePlanner().build([short, long], [target]),
            [
                PoolReadRange("short-first", 4, 0, 102, 4),
                PoolReadRange("long-second", 40, 14, 100, 2),
                PoolReadRange("long-second", 40, 20, 106, 4),
            ],
        )
        self.assertEqual(
            RestorePlanner().build([long, short], [target]),
            [PoolReadRange("long-second", 40, 14, 100, 10)],
        )

    def test_long_context_contiguous_restore_does_not_materialize_per_token_ranges(self):
        cache = self.cache(tokens=(0, 131072), heads=(0, 1))
        source = SourceCacheSlice(cache, "long-context", 262144, 0)
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner.PoolReadRange",
            wraps=PoolReadRange,
        ) as constructor:
            plan = RestorePlanner().build([source], [TargetCacheSlice(cache, 100)])
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].size, 262144)
        self.assertLess(constructor.call_count, 4)

    def cache(self, tokens=(0, 4), heads=(0, 4), layout="hnd", component="k", layer="layer.0", format_id="fp16"):
        width = 2
        return CacheSlice(
            layer,
            component,
            format_id,
            *tokens,
            *heads,
            width,
            width if layout == "hnd" else (heads[1] - heads[0]) * width,
            (tokens[1] - tokens[0]) * width if layout == "hnd" else width,
        )

    def materialize(self, sources, targets):
        # Encode each logical vector independently; do not mirror planner byte
        # arithmetic in the expected output. The target is checked coordinate-wise.
        objects = {}
        for source in sources:
            blob = objects.setdefault(source.key, bytearray(source.object_size))
            cache = source.cache
            for token in range(cache.token_start, cache.token_end):
                for head in range(cache.head_start, cache.head_end):
                    offset = source.object_offset + cache.offset(token, head)
                    blob[offset : offset + 2] = bytes((token + 1, head + 1))
        result = bytearray(10000)
        plan = RestorePlanner().build(sources, targets)
        for item in plan:
            result[item.destination : item.destination + item.size] = objects[item.key][
                item.source_offset : item.source_offset + item.size
            ]
        for target in targets:
            cache = target.cache
            for token in range(cache.token_start, cache.token_end):
                for head in range(cache.head_start, cache.head_end):
                    offset = target.address + cache.offset(token, head)
                    self.assertEqual(result[offset : offset + 2], bytes((token + 1, head + 1)))
        return plan

    def test_tp_gather_and_hnd_to_nhd_repack(self):
        sources = [SourceCacheSlice(self.cache(heads=(head, head + 2)), f"rank{head}", 16, 0) for head in (0, 2)]
        plan = self.materialize(sources, [TargetCacheSlice(self.cache(layout="nhd"), 100)])
        self.assertEqual({item.key for item in plan}, {"rank0", "rank2"})

    def test_tp_scatter_and_nhd_to_hnd_repack(self):
        sources = [SourceCacheSlice(self.cache(layout="nhd"), "source", 32, 0)]
        self.materialize(sources, [TargetCacheSlice(self.cache(heads=(2, 4)), 100)])

    def test_different_block_sizes_and_tail_token_ranges(self):
        sources = [
            SourceCacheSlice(self.cache(tokens=(start, start + 2)), f"block{start}", 16, 0) for start in (0, 2, 4)
        ]
        self.materialize(sources, [TargetCacheSlice(self.cache(tokens=(1, 5)), 100)])

    def test_interleaved_cp_pages_are_explicit_token_slices(self):
        sources = [
            SourceCacheSlice(self.cache(tokens=(token, token + 1)), f"cp{token % 2}-block{token}", 8, 0)
            for token in (0, 2, 1, 3)
        ]
        self.materialize(sources, [TargetCacheSlice(self.cache(), 100)])

    def test_replica_selection_does_not_write_target_twice(self):
        cache = self.cache()
        sources = [SourceCacheSlice(cache, key, 32, 0) for key in ("first", "replica")]
        plan = self.materialize(sources, [TargetCacheSlice(cache, 100)])
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].key, "first")
        self.assertEqual(plan[0].size, 32)

    def test_pp_and_components_match_by_identity_not_list_order(self):
        components = [
            self.cache(heads=(0, 1), component=name, layer=layer)
            for layer in ("layer.7", "draft.0")
            for name in ("latent", "indexer", "scale")
        ]
        sources = [SourceCacheSlice(cache, f"object{i}", 16, 4) for i, cache in enumerate(components)]
        targets = [TargetCacheSlice(cache, 100 + i * 100) for i, cache in enumerate(reversed(components))]
        self.materialize(sources, targets)

    def test_missing_draft_component_or_incompatible_format_fails(self):
        source = SourceCacheSlice(self.cache(), "source", 32, 0)
        for cache in (self.cache(layer="draft.0"), self.cache(component="scale"), self.cache(format_id="int8")):
            with self.subTest(cache=cache), self.assertRaisesRegex(ValueError, "Incomplete pool cache coverage"):
                RestorePlanner().build([source], [TargetCacheSlice(cache, 100)])

    def test_holes_fail_before_returning_partial_plan(self):
        source = SourceCacheSlice(self.cache(tokens=(0, 2)), "source", 16, 0)
        with self.assertRaisesRegex(ValueError, "coverage"):
            RestorePlanner().build([source], [TargetCacheSlice(self.cache(), 100)])

    def test_overlapping_physical_vectors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            CacheSlice("layer", "k", "fp16", 0, 4, 0, 4, 2, 2, 2)

    def test_source_object_bounds_are_checked(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            SourceCacheSlice(self.cache(), "source", 32, 1)
