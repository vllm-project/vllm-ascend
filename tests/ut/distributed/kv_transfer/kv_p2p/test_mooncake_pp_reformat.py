# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from itertools import permutations
from threading import Lock, Thread
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import KVCacheRecvingThread


def _receiver():
    obj = KVCacheRecvingThread.__new__(KVCacheRecvingThread)
    obj.pending_reformat = defaultdict(dict)
    obj.pending_reformat_lock = Lock()
    obj._apply_kv_cache_reformat = MagicMock()
    return obj


@pytest.mark.parametrize("order", list(permutations(range(4))))
def test_pp_reformat_retains_every_segment(order):
    obj = _receiver()
    for pp_rank in order:
        for shard in (0, 1):
            obj._stash_pending_reformat("r", shard, [(0, [[shard]], 2, list(range(pp_rank * 15, (pp_rank + 1) * 15)))])
    for shard in (0, 1):
        groups = obj.pending_reformat["r"][shard]
        assert sorted(layer for _, _, _, layers in groups for layer in layers) == list(range(60))
    obj._reformat_pending_kv_caches("r")
    assert "r" not in obj.pending_reformat
    assert obj._apply_kv_cache_reformat.call_count == 2
    assert all(len(call.args[0]) == 4 for call in obj._apply_kv_cache_reformat.call_args_list)
    obj._reformat_pending_kv_caches("r")
    assert obj._apply_kv_cache_reformat.call_count == 2


def test_pp_reformat_concurrent_requests_and_shards():
    obj = _receiver()
    threads = [
        Thread(target=obj._stash_pending_reformat, args=(req, shard, [(0, [[shard]], 2, [pp])]))
        for req in ("a", "b")
        for shard in (0, 1)
        for pp in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for req in ("a", "b"):
        for shard in (0, 1):
            assert sorted(group[3][0] for group in obj.pending_reformat[req][shard]) == list(range(4))
    obj._reformat_pending_kv_caches("a")
    assert "b" in obj.pending_reformat
    obj._reformat_pending_kv_caches("b")
    assert not obj.pending_reformat
