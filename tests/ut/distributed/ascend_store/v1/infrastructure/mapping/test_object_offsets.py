"""Common offset arithmetic is reusable without constructing a default object first."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tests.ut.distributed.ascend_store.v1.infrastructure.mapping.fixtures import make_case
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.layout import (
    build_memcache_object_layouts,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration.topology import (
    CPPartitionSpec,
    PPPartitionSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping.object_layout import (
    build_default_object_layouts,
    default,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping.object_layout.offsets import (
    build_segment_offsets,
)


def test_equal_head_slices_share_compiled_offsets_and_local_sources_are_unchanged():
    local = make_case(head_slices=2, padding=1).bind().local_layout
    sources = tuple(source.local_segments.block_zero_segment_addresses.copy() for source in local.partitions)
    with patch.object(np, "cumsum", wraps=np.cumsum) as scan:
        offsets = build_segment_offsets(local.partitions)
    scan.assert_called_once()
    assert offsets[0] is offsets[1]
    for source, original, positions in zip(local.partitions, sources, offsets, strict=True):
        assert not positions.flags.writeable
        assert int(positions[0]) == 0 and int(positions[-1]) == source.payload_bytes
        assert len(positions) == len(source.local_segments.block_zero_segment_addresses) + 1
        np.testing.assert_array_equal(source.local_segments.block_zero_segment_addresses, original)
        np.testing.assert_array_equal(
            np.diff(positions),
            np.broadcast_to(source.local_segments.segment_lengths_bytes, (len(positions) - 1,)),
        )


def test_offsets_reject_incomplete_local_payload():
    source = make_case().bind().local_layout.partitions[0]
    with pytest.raises(ValueError, match="complete payload"):
        build_segment_offsets((replace(source, payload_bytes=source.payload_bytes + 1),))


@pytest.mark.parametrize("shared", (False, True))
def test_memcache_builds_final_objects_directly_and_aligns_only_context_slots(shared):
    local = make_case(layers=(2, 3), heads=1).bind().local_layout
    ordinary = build_default_object_layouts(local)[0]
    layer_bytes = local.partitions[0].payload_bytes // 2
    alignment = 2 * 1024 * 1024 if shared else 1
    with patch.object(default, "RemoteObjectLayout", side_effect=AssertionError("unused intermediate default object")):
        remote = build_memcache_object_layouts(
            local,
            pp_partition=PPPartitionSpec(global_layer_count=4),
            cp_partition=CPPartitionSpec(1, 2, 2 if shared else 1, alignment),
            stage_layer_start=2,
            stage_layer_count=2,
            object_shard_index=1 if shared else 0,
        )[0]
    assert remote.size_bytes == (4 * 1024 * 1024 if shared else 4 * layer_bytes)
    origin = (2 * 1024 * 1024 if shared else 0) + 2 * layer_bytes
    np.testing.assert_array_equal(
        remote.segment_boundary_offsets_bytes, ordinary.segment_boundary_offsets_bytes + origin
    )
    assert not remote.segment_boundary_offsets_bytes.flags.writeable
