# SPDX-License-Identifier: Apache-2.0
"""Only populated Host blocks may survive as reusable D2RH cache entries."""

import threading
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_d2rh_connector as d2rh


def make_transfer(*, ratio=1, hbm_blocks=1, pp_size=1, sliding=False, existing_hit=False, fail_port=None):
    manager = d2rh.D2RHCPUCacheManager(8)
    hashes = ([b"prefix-first", b"prefix-second"],)
    pulls = [[dict(group_id=0, remote_tp_offset=0, num_group_pulls=1, prefill_pp_rank=rank)] for rank in range(pp_size)]
    if existing_hit:
        mapping, _, misses = manager.alloc_sharded_block_map(([10],), pulls, ([hashes[0][0]],))
        manager.commit_block_map(mapping, misses)
        manager.free_block_map(mapping)
    block_map, hits, misses = manager.alloc_sharded_block_map(([10, 11],), pulls, hashes)
    worker = object.__new__(d2rh.D2RHThread)
    worker.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda *args: {"pp_size": pp_size})
    )
    worker.remote_local_block_map = {"local-a": block_map, "remote-a": block_map}
    worker.cpu_kvcache_manager = manager
    worker.log_full_block_map = False
    worker.remote_metadata_lock = threading.Lock()
    names = [f"layer.{rank}" for rank in range(pp_size)]
    spec = {
        "layer_names": names,
        "kv_cache_spec_type": "AscendSlidingWindowMLASpec" if sliding else "AttentionSpec",
    }
    worker.kv_group2layeridx = {0: (spec, list(range(pp_size)))}
    worker.cpu_kv_caches_base_addr = [[1000 + rank * 1000] for rank in range(pp_size)]
    worker.cpu_block_len_per_addr = [[16] for _ in names]
    worker.cpu_block_stride_per_addr = [[16] for _ in names]
    worker.cpu_block_size_scale = [[1] for _ in names]
    worker.kv_caches_base_addr = {"p": {10 + rank: [[10000 + rank * 1000]] for rank in range(pp_size)}}
    worker.remote_kv_group2layeridx = {
        "p": {10 + rank: {0: ({"layer_names": [names[rank]]}, [0])} for rank in range(pp_size)}
    }
    worker.remote_block_size_scale = {"p": {10 + rank: [[1]] for rank in range(pp_size)}}
    worker.remote_block_stride_per_addr = {"p": {10 + rank: [[16]] for rank in range(pp_size)}}
    worker.remote_te_port = {"p": {10 + rank: 1010 + rank for rank in range(pp_size)}}
    worker.block_size = 32
    worker.group_compress_ratios = {0: ratio}
    written: set[int] = set()
    calls = []

    def transfer(session, local_addrs, remote_addrs, lengths):
        calls.append((session, local_addrs, remote_addrs, lengths))
        if session == fail_port:
            return -1
        for address, length in zip(local_addrs, lengths):
            written.update(range(address, address + length, 16))
        return 0

    worker.engine = SimpleNamespace(batch_transfer_sync_read=transfer)
    worker._send_done_recv_signal = lambda *args: None
    worker.send_pull_done = lambda *args: None
    request = dict(
        request_id="local-a",
        remote_request_id="remote-a",
        remote_engine_id="p",
        remote_host="p-host",
        remote_port=10,
        remote_block_ids=([10, 11],),
        remote_handshake_ports=list(range(10, 10 + pp_size)),
        group_pulls_by_port=pulls,
        num_computed_tokens=hbm_blocks * 32 * ratio,
        cache_hits=hits,
        cacheable_misses=misses,
    )
    return worker, request, manager, block_map, hashes, pulls, written, calls


@pytest.mark.parametrize("ratio", [1, 4, 128])
@pytest.mark.parametrize("hbm_blocks", [0, 1, 2])
@pytest.mark.parametrize("pp_size", [1, 2])
def test_only_written_host_blocks_are_reusable(ratio, hbm_blocks, pp_size):
    worker, request, manager, block_map, hashes, pulls, written, _ = make_transfer(
        ratio=ratio, hbm_blocks=hbm_blocks, pp_size=pp_size
    )
    worker._handle_request(request)
    for index, remote_id in enumerate([10, 11]):
        copied = index >= hbm_blocks
        assert ((0, 0, hashes[0][index]) in manager.cache_by_key) == copied
        for rank in range(pp_size):
            address = 1000 + rank * 1000 + block_map[(0, remote_id, 0)] * 16
            assert (address in written) == copied
    manager.free_block_map(block_map)
    assert not manager.pending_by_key
    assert not manager.pin_count
    # Simulate eviction of the D-HBM prefix: the next request needs Host again.
    next_map, next_hits, _ = manager.alloc_sharded_block_map(([20, 21],), pulls, hashes)
    assert next_hits == {(0, 20 + index, 0) for index in range(hbm_blocks, 2)}
    manager.free_block_map(next_map)


def test_hbm_hit_does_not_invalidate_existing_host_hit():
    worker, request, manager, block_map, hashes, _, written, _ = make_transfer(existing_hit=True)
    prefix_block = block_map[(0, 10, 0)]
    worker._handle_request(request)
    assert 1000 + prefix_block * 16 not in written
    assert manager.cache_by_key[(0, 0, hashes[0][0])] == prefix_block
    manager.free_block_map(block_map)
    assert len(manager.cache_by_key) == 2


def test_partial_block_is_still_copied_and_cached():
    worker, request, manager, _, _, _, _, _ = make_transfer(ratio=4)
    request["num_computed_tokens"] -= 1
    worker._handle_request(request)
    assert len(manager.cache_by_key) == 2


def test_sliding_window_tail_is_copied_despite_hbm_prefix():
    worker, request, manager, _, _, _, _, _ = make_transfer(sliding=True, hbm_blocks=2)
    worker._handle_request(request)
    assert len(manager.cache_by_key) == 2


def test_later_pp_failure_cannot_publish_partial_host_state():
    worker, request, manager, _, _, _, _, calls = make_transfer(pp_size=2, fail_port="p-host:1011")
    with pytest.raises(RuntimeError, match="hop1 transfer failed"):
        worker._handle_request(request)
    assert len(calls) == 2
    assert not manager.cache_by_key
    assert not manager.pending_by_key
    assert not manager.pin_count
    assert not manager.used_set
    assert len(manager.free_queue) == manager.num_blocks
    assert not worker.remote_local_block_map
