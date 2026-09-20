# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hybrid layerwise layout under pipeline parallelism.

Under PP the scheduler and each worker see different ``layer_names`` for the
same cache group, because upstream projects the global groups onto each stage.
The layout digest must still agree everywhere — otherwise every stage gets its
own key space and the pool silently never reports a hit — while still changing
when the layout or the topology changes.
"""

import unittest
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm.v1.core.kv_cache_utils import _project_kv_cache_groups_to_worker
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_layerwise
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_layerwise import (
    hybrid_layout_id,
    validate_hybrid_pp_coverage,
)


def make_groups() -> list[KVCacheGroupSpec]:
    """Two groups whose layers interleave, as a hybrid model produces."""
    return [
        KVCacheGroupSpec(
            ["model.layers.0.kv", "model.layers.2.kv", "model.layers.4.kv"],
            FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype="uint8"),
        ),
        KVCacheGroupSpec(
            ["model.layers.1.c4", "model.layers.3.c4", "model.layers.5.c4"],
            FullAttentionSpec(block_size=32, num_kv_heads=1, head_size=1, dtype="uint8"),
        ),
    ]


def as_config(groups) -> SimpleNamespace:
    return SimpleNamespace(kv_cache_groups=groups)


def project(groups, stage_layer_names) -> list[KVCacheGroupSpec]:
    """Project the global groups onto one PP stage, using upstream's own rule."""
    specs = {}
    for group in groups:
        for name in group.layer_names:
            specs[name] = group.kv_cache_spec
    worker_spec = {name: specs[name] for name in stage_layer_names}
    return _project_kv_cache_groups_to_worker(groups, worker_spec)


def pp_config(pp_size=2, tp_size=1, rank=0, dcp_size=1, interleave=1) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_parallel_size=pp_size,
        tensor_parallel_size=tp_size,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=interleave,
        rank=rank,
    )


def stage_model(partitions, tp_size=1) -> SimpleNamespace:
    """``get_layers_start_end_indices`` for each PP stage, in rank order."""
    ranges = []
    start = 0
    for count in partitions:
        ranges.append((start, start + count))
        start += count
    config = SimpleNamespace()
    config.get_layers_start_end_indices = lambda parallel_config: ranges[parallel_config.rank // tp_size]
    return config


STAGE0 = ["model.layers.0.kv", "model.layers.1.c4", "model.layers.2.kv"]
STAGE1 = ["model.layers.3.c4", "model.layers.4.kv", "model.layers.5.c4"]


class TestHybridLayoutUnderPipelineParallelism(unittest.TestCase):
    def test_digest_agrees_between_scheduler_and_every_stage(self):
        groups = make_groups()
        parallel = pp_config()
        model = stage_model([3, 3])

        scheduler = hybrid_layout_id(as_config(groups), parallel, model)
        self.assertEqual(scheduler, hybrid_layout_id(as_config(project(groups, STAGE0)), parallel, model))
        self.assertEqual(scheduler, hybrid_layout_id(as_config(project(groups, STAGE1)), parallel, model))

    def test_stage_local_membership_does_not_move_the_digest(self):
        groups = make_groups()
        parallel = pp_config()
        model = stage_model([3, 3])
        base = hybrid_layout_id(as_config(groups), parallel, model)

        # Same global layout, re-projected for one stage: membership is stage
        # local, so it must not participate.
        self.assertEqual(base, hybrid_layout_id(as_config(project(groups, STAGE1)), parallel, model))

    def test_topology_and_page_geometry_do_move_the_digest(self):
        groups = make_groups()
        parallel = pp_config()
        model = stage_model([3, 3])
        base = hybrid_layout_id(as_config(groups), parallel, model)

        self.assertNotEqual(base, hybrid_layout_id(as_config(groups), parallel, stage_model([2, 4])))
        self.assertNotEqual(base, hybrid_layout_id(as_config(groups), pp_config(dcp_size=2), model))
        self.assertNotEqual(base, hybrid_layout_id(as_config(groups), pp_config(interleave=16), model))

        reordered = [groups[1], groups[0]]
        self.assertNotEqual(base, hybrid_layout_id(as_config(reordered), parallel, model))

        repaged = make_groups()
        repaged[0].kv_cache_spec = FullAttentionSpec(block_size=64, num_kv_heads=1, head_size=1, dtype="uint8")
        self.assertNotEqual(base, hybrid_layout_id(as_config(repaged), parallel, model))

    def test_pipeline_parallelism_needs_the_model_config(self):
        groups = make_groups()
        with self.assertRaisesRegex(ValueError, "model config"):
            hybrid_layout_id(as_config(groups), pp_config())


class TestHybridPpCoverage(unittest.TestCase):
    def test_empty_pooled_group_is_rejected(self):
        # Stage 0 owns no layer of group 1, so it would never save group 1's
        # keys while the hit check still demands them from every stage.
        stage = project(make_groups(), STAGE0)
        self.assertEqual([len(group.layer_names) for group in stage], [2, 1])

        stage[1].layer_names.clear()
        with self.assertRaisesRegex(ValueError, "holds no layers"):
            validate_hybrid_pp_coverage(as_config(stage), pp_config())

    def test_drafter_group_is_reported_not_rejected(self):
        # A drafter group is empty away from the last stage, and the projection
        # clears its flag, so it cannot be classified here: report, don't raise.
        stage = project(make_groups(), STAGE0)
        stage[1].layer_names.clear()
        with self.assertLogs("vllm", level="WARNING") as captured:
            validate_hybrid_pp_coverage(as_config(stage), pp_config(), use_spec_decode=True)
        self.assertIn("holds no layers", "\n".join(captured.output))

    def test_single_stage_needs_no_coverage_check(self):
        stage = project(make_groups(), STAGE0)
        stage[1].layer_names.clear()
        validate_hybrid_pp_coverage(as_config(stage), pp_config(pp_size=1))


class TestHybridKeyCoordinates(unittest.TestCase):
    def test_group_key_carries_stage_and_shard(self):
        key = mooncake_layerwise.hybrid_block_key("model", "layout", 1, 32, "6831", 0, pp_rank=2, dcp_rank=3)
        self.assertEqual(
            key,
            "model@mooncake_hybrid_v1:layout@pp_rank:2@dcp_rank:3@group:1@block:32@6831@0",
        )

    def test_hit_check_requires_every_stage_and_shard(self):
        # Sharded and replicated groups are treated the same way: every DCP
        # rank's key is required, so a replicated group is redundant but never
        # ambiguous.
        keys = mooncake_layerwise.make_hit_check_keys(
            "model", 0, "6831", 2, 1, namespace="ns", pp_size=2, dcp_size=2
        )
        self.assertEqual(len(keys), 8)
        self.assertEqual(len(set(keys)), 8)
        self.assertEqual(sum("@pp_rank:1" in key for key in keys), 4)
        self.assertEqual(sum("@dcp_rank:1" in key for key in keys), 4)
