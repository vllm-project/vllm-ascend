"""Production GVA formulas and initialization-only layout binding."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.ut.distributed.ascend_store.v1.infrastructure.mapping.fixtures import make_case
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import LayerwiseMapping
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping.gva_layout import (
    GVAObjectLayout,
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
    layout = GVAObjectLayout(7, 4, 3 if pp_rank else 0, dcp_size if shared else 1, dcp_rank if shared else 0)
    assert layout.bind(40, 2) == (expected.size, expected.layer_byte_offset)


def test_runtime_reuses_layout_and_does_not_mutate_bulk():
    bulk = make_case(layers=(3, 4), heads=1, head_slices=1).bind()
    keys = bulk.keys(["hash"])
    before = bulk.arguments(keys, [1])
    mapping = LayerwiseMapping(bulk, GVAObjectLayout(7, 4, 3, 2, 1))
    selection = mapping.prepare(keys, [1])
    assert mapping.object_size_bytes == 4 * 1024 * 1024
    with patch.object(GVAObjectLayout, "bind", side_effect=AssertionError("rebind")):
        first = mapping.arguments(selection, layer_id=3)
        mapping.arguments(selection, layer_id=4)
        assert mapping.arguments(selection, layer_id=3) == first
    assert bulk.arguments(keys, [1]) == before
    assert all(
        not layout.object_offsets_bytes.flags.writeable
        for layers in mapping.layer_layouts.values()
        for layout in layers
    )


@pytest.mark.parametrize("size,count", ((0, 2), (40, 0), (41, 2)))
def test_invalid_group_geometry_is_rejected_at_binding(size, count):
    with pytest.raises(ValueError, match="uniform"):
        GVAObjectLayout(7, 4, 3).bind(size, count)
