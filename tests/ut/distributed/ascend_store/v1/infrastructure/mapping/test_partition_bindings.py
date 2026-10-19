"""Production GVA formulas and initialization-only layout binding."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tests.ut.distributed.ascend_store.v1.infrastructure.mapping.fixtures import make_case
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache import layout as memcache_layout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.layout import (
    build_memcache_object_layouts,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.naming import BoundLayerwiseKeys
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration.topology import (
    CPPartitionSpec,
    PPPartitionSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping.object_layout import (
    context_partitioning,
    layer_partitioning,
)


@pytest.mark.parametrize("pp_rank", (0, 1))
@pytest.mark.parametrize("dcp_size,put_step,dcp_rank", ((1, 1, 0), (2, 1, 1), (2, 2, 0), (2, 2, 1)))
def test_bound_layout_matches_production_global_allocation_and_offsets(
    monkeypatch, pp_rank, dcp_size, put_step, dcp_rank
):
    production = SimpleNamespace(
        num_layers=4,
        num_kv_cache_groups=1,
        group_num_layers={0: 2},
        group_block_len={0: [16, 4, 16, 4]},
        use_block_key_layerwise=False,
        total_layers=7,
        pp_size=pp_rank + 1,
        layerwise_key_layer_offset=3 if pp_rank else 0,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        put_step=put_step,
        token_database=None,
    )
    production._global_group_alloc_size = lambda group: pool_worker.KVPoolWorker._global_group_alloc_size(
        production, group
    )
    monkeypatch.setattr(
        pool_worker, "LayerBatchBuilder", lambda _db, size, _layers, **kw: SimpleNamespace(size=size, **kw)
    )
    expected = pool_worker.KVPoolWorker._build_group_layer_builders(production)[0]
    shared = dcp_size > 1 and put_step > 1
    span, origin = layer_partitioning.bind_layer_object_span(
        PPPartitionSpec(global_layer_count=7),
        40,
        2,
        stage_layer_start=3 if pp_rank else 0,
        stage_layer_count=4,
    )
    cp = CPPartitionSpec(1, dcp_size, dcp_size if shared else 1, 2 * 1024 * 1024 if shared else 1)
    assert context_partitioning.bind_context_object_span(cp, span, origin, shard_index=dcp_rank if shared else 0) == (
        expected.size,
        expected.layer_byte_offset,
    )


def test_runtime_reuses_layout_and_does_not_mutate_bulk():
    case = make_case(layers=(3, 4), heads=1, head_slices=1)
    bulk = case.bind()
    keys = bulk.keys(["hash"])
    before = bulk.arguments(keys, [1])
    mapping = case.bind_layers(
        object_layout_factory=partial(
            build_memcache_object_layouts,
            pp_partition=PPPartitionSpec(global_layer_count=7),
            cp_partition=CPPartitionSpec(1, 2, 2, 2 * 1024 * 1024),
            stage_layer_start=3,
            stage_layer_count=4,
            object_shard_index=1,
        ),
    )
    selection = mapping.prepare(keys, [1])
    assert tuple(obj.size_bytes for obj in mapping.object_layouts) == (4 * 1024 * 1024,)
    with (
        patch.object(memcache_layout, "bind_layer_object_span", side_effect=AssertionError("rebind PP")),
        patch.object(memcache_layout, "bind_context_object_span", side_effect=AssertionError("rebind CP")),
    ):
        first = mapping.arguments(selection, layer_id=3)
        mapping.arguments(selection, layer_id=4)
        assert mapping.arguments(selection, layer_id=3) == first
    assert bulk.arguments(keys, [1]) == before
    assert all(
        not layout.object_offsets_bytes.flags.writeable
        for layers in mapping.layer_layouts.values()
        for layout in layers
    )


def test_registered_nonuniform_layer_payloads_cannot_use_uniform_global_offsets():
    case = make_case(layers=(3, 4), head_slices=1, stored_tokens=(2, 4))
    with pytest.raises(ValueError, match="uniform"):
        case.bind_layers(
            object_layout_factory=partial(
                build_memcache_object_layouts,
                pp_partition=PPPartitionSpec(global_layer_count=7),
                cp_partition=CPPartitionSpec(),
                stage_layer_start=3,
                stage_layer_count=4,
            ),
        )


@pytest.mark.parametrize("layout", ("NHD", "HND", "LBHNC"))
@pytest.mark.parametrize("head_slices", (1, 2, 4))
@pytest.mark.parametrize("shard_index", (0, 1))
def test_head_then_stage_then_context_changes_only_remote_coordinates(layout, head_slices, shard_index):
    case = make_case(layers=(3, 4), layout=layout, head_slices=head_slices, padding=1, kernels=2)
    bulk = case.bind()
    local = case.bind_layers()
    partitioned = case.bind_layers(
        object_layout_factory=partial(
            build_memcache_object_layouts,
            pp_partition=PPPartitionSpec(global_layer_count=7),
            cp_partition=CPPartitionSpec(1, 2, 2, 1024),
            stage_layer_start=3,
            stage_layer_count=4,
            object_shard_index=shard_index,
        ),
    )
    keys = partitioned.keys((b"a", b"b"))
    assert keys == local.keys((b"a", b"b"))
    assert partitioned.lookup_keys((b"a",)) == bulk.lookup_keys((b"a",))
    before = local.prepare(keys, (2, 0))
    after = partitioned.prepare(keys, (2, 0))
    layer_bytes = tuple(obj.payload_bytes for obj in bulk.local.objects)[0] // 2
    stride = ((layer_bytes * 7 + 1023) // 1024) * 1024
    origin = shard_index * stride + 3 * layer_bytes
    assert tuple(obj.size_bytes for obj in partitioned.object_layouts) == (stride * 2,) * head_slices
    for layer_id in (3, 4):
        old_keys, old_addresses, old_lengths, old_offsets = local.arguments(before, layer_id=layer_id)
        new_keys, new_addresses, new_lengths, new_offsets = partitioned.arguments(after, layer_id=layer_id)
        assert (new_keys, new_addresses, new_lengths) == (old_keys, old_addresses, old_lengths)
        assert new_offsets == [[offset + origin for offset in row] for row in old_offsets]
        for layer in partitioned.layer_layouts[layer_id]:
            original = tuple(obj.memory_segments for obj in partitioned.local.objects)[layer.object_index]
            assert np.shares_memory(layer.local_memory_segments.block_zero_addresses, original.block_zero_addresses)


def test_unequal_target_pp_spans_keep_sizes_on_their_own_head_axes():
    case = make_case(head_slices=2, pp_layer_counts=(1, 3))
    bulk = case.bind()
    mapping = case.bind_layers(
        object_layout_factory=partial(
            build_memcache_object_layouts,
            pp_partition=PPPartitionSpec(),
            cp_partition=CPPartitionSpec(1, 2, 2),
            object_shard_index=1,
        ),
    )
    assert tuple(obj.size_bytes for obj in mapping.object_layouts) == tuple(
        size * 2 for size in tuple(obj.payload_bytes for obj in bulk.local.objects)
    )
    for layouts in mapping.layer_layouts.values():
        for layout in layouts:
            assert (
                layout.object_size_bytes == tuple(obj.size_bytes for obj in mapping.object_layouts)[layout.object_index]
            )


def test_bound_names_share_the_geometry_axis():
    case = make_case(head_slices=2)
    parts = (("head0@", "@stage1"), ("head1@", "@stage1"))
    bound = BoundLayerwiseKeys(parts, parts + (("head2@", "@stage0"),))
    mapping = case.bind_layers(key_builder=SimpleNamespace(bind=lambda *_args, **_kwargs: bound))
    assert mapping.keys((b"a",)) == (("head0@61@stage1",), ("head1@61@stage1",))
    assert len(mapping.lookup_keys((b"a",))) == 3


def test_extra_registered_layer_cannot_overrun_the_global_layer_span():
    # A final PP stage can also register a speculative layer beyond the model's base layer count.
    case = make_case(layers=(3, 4, 5, 6, 7), head_slices=1)
    with pytest.raises(ValueError, match="exceeds"):
        case.bind_layers(
            object_layout_factory=partial(
                build_memcache_object_layouts,
                pp_partition=PPPartitionSpec(global_layer_count=7),
                cp_partition=CPPartitionSpec(),
                stage_layer_start=3,
                stage_layer_count=5,
            ),
        )
