# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tests.ut.kvpp_utils import indexer_name, layer_name, make_cache_config, make_kvpp_config, make_kvpp_specs
from vllm_ascend.worker import kvpp_cache


@pytest.mark.parametrize("num_blocks,total_bytes", [(3, 1176), (2, 784)])
def test_physical_allocations_and_scratch_aliases(monkeypatch, num_blocks, total_bytes):
    monkeypatch.setattr(kvpp_cache, "get_kvpp_group", lambda: SimpleNamespace(rank_in_group=1))
    specs = make_kvpp_specs()
    caches = kvpp_cache.allocate_kvpp_cache(
        make_kvpp_config(), make_cache_config(specs, num_blocks), torch.device("cpu")
    )
    assert set(caches) == set(specs)
    storages = {
        part.untyped_storage().data_ptr(): part.untyped_storage() for parts in caches.values() for part in parts
    }
    assert sum(storage.nbytes() for storage in storages.values()) == total_bytes
    assert all(torch.count_nonzero(part).item() == 0 for parts in caches.values() for part in parts)

    def storage_id(index):
        return caches[layer_name(index)][0].untyped_storage().data_ptr()

    assert storage_id(9) == storage_id(11) == storage_id(15)
    assert storage_id(10) == storage_id(16)
    assert len({storage_id(i) for i in (9, 10, 12, 13, 14, 17)}) == 6
    for index, size in ((9, 76), (10, 76), (12, 32), (13, 48), (14, 64), (17, 96)):
        assert caches[layer_name(index)][0].untyped_storage().nbytes() == size * num_blocks

    parts = (*caches[layer_name(11)], *caches[indexer_name(11)])
    assert [part.numel() for part in parts] == [64 * num_blocks, 8 * num_blocks, 4 * num_blocks]
    assert [part.storage_offset() for part in parts] == [0, 64 * num_blocks, 72 * num_blocks]
    assert all(part.untyped_storage().data_ptr() == storage_id(11) for part in parts)
    caches[layer_name(9)][0][0] = 7
    assert all(caches[layer_name(i)][0][0].item() == 7 for i in (11, 15))
    assert all(caches[layer_name(i)][0][0].item() == 0 for i in (10, 12, 13, 14, 16, 17))
    for value, part in enumerate(parts, 1):
        part.fill_(value)
    for value, part in enumerate(parts, 1):
        assert torch.all(part == value)
