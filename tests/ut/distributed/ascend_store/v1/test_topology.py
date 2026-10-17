"""Transfer coordinates and layer topology preserve valid boundaries and physical identity."""

from __future__ import annotations

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import (
    KVPoolLayerTopology,
    resolve_group_layers,
)


@pytest.mark.parametrize("start,end", ((-1, 0), (4, 3)))
def test_token_ranges_reject_negative_or_reversed_boundaries(start, end) -> None:
    with pytest.raises(ValueError):
        TokenRange(start, end)


def test_layer_topology_groups_shared_entries_and_offsets_mtp_layers() -> None:
    layers = resolve_group_layers(["model.layers.1.v", "mtp.layers.0.attn", "model.layers.1.k"], base_layer_count=4)
    assert layers == (
        KVPoolLayerTopology(1, ("model.layers.1.k", "model.layers.1.v")),
        KVPoolLayerTopology(4, ("mtp.layers.0.attn",)),
    )
