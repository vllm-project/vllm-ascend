"""Compare GVA reservations with the existing production allocation formula."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration.topology import (
    CPPartitionSpec,
    PPPartitionSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import (
    build_gva_remote,
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
        use_layerwise_transfer=True,
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
    production._group_pp_byte_offset = lambda group: pool_worker.KVPoolWorker._group_pp_byte_offset(production, group)
    monkeypatch.setattr(
        pool_worker, "LayerBatchBuilder", lambda _db, size, _layers, **kw: SimpleNamespace(size=size, **kw)
    )
    expected = pool_worker.KVPoolWorker._build_group_layer_builders(production)[0]
    shared = dcp_size > 1 and put_step > 1
    topology = SimpleNamespace(
        pp_partition=PPPartitionSpec(global_layer_count=7),
        cp_partition=CPPartitionSpec(1, dcp_size, dcp_size if shared else 1, 2 * 1024 * 1024 if shared else 1),
        stage_layer_start=3 if pp_rank else 0,
        stage_layer_count=4,
        object_shard_index=dcp_rank if shared else 0,
    )
    local = {
        (0, 0, 0): {
            "bases": np.array((100, 200, 300, 400), dtype=np.uint64),
            "block_strides": np.array((20,), dtype=np.uint64),
            "lengths": np.array((16, 4, 16, 4), dtype=np.uint64),
        }
    }
    remote = build_gva_remote(topology, local, {0: {0: slice(0, 2), 1: slice(2, 4)}})[0]
    assert remote["size_bytes"] == expected.size
    assert remote["offsets"].tolist() == [expected.layer_byte_offset + offset for offset in (0, 16, 20, 36)]
