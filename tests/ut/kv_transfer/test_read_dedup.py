# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the KV pool read-path deduplication (task #14140).

No NPU and no collective communication is involved. The helpers under test are
host-side logic over plain CPU tensors, and ``all_gather_into_tensor`` is
simulated by handing every rank the buffer the collective would have produced,
so the full pack -> gather -> write-back round trip is covered byte for byte.

The fake KV cache is laid out as ``[block][segment][bytes]``, which makes the
blocks of one shard exactly ``tp_size`` blocks apart - the arithmetic address
pattern the strided copy plan relies on.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

SEG_BYTES = 64  # bytes per segment
SEGS_PER_BLOCK = 3  # e.g. MLA: latent + rope, per layer


def make_worker(tp_rank: int, tp_size: int, num_blocks: int, seed: int = 0) -> KVPoolWorker:
    """Build a ``KVPoolWorker`` skeleton with one fake KV cache tensor."""
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.tp_rank = tp_rank
    worker.tp_size = tp_size
    worker.put_step = tp_size
    worker.dcp_size = 1
    worker.use_kvpp = False
    worker.use_layerwise = False
    worker.load_async = False
    worker.current_layer = 0
    worker.m_store = None
    torch.manual_seed(seed)
    cache = torch.randint(0, 256, (num_blocks, SEGS_PER_BLOCK, SEG_BYTES), dtype=torch.uint8)
    base = cache.data_ptr()
    worker._kv_cache_addr_segments = [(base, cache.numel(), cache)]
    stride = SEGS_PER_BLOCK * SEG_BYTES
    worker.test_addr_list = [
        [base + b * stride + s * SEG_BYTES for s in range(SEGS_PER_BLOCK)] for b in range(num_blocks)
    ]
    worker.test_size_list = [[SEG_BYTES] * SEGS_PER_BLOCK for _ in range(num_blocks)]
    worker.test_keys = [f"blk-{b}" for b in range(num_blocks)]
    return worker


def gather_sim(workers: list[KVPoolWorker], num_blocks: int, rank: int) -> tuple[torch.Tensor, list[int], int]:
    """Return the buffer ``all_gather_into_tensor`` would give ``rank``."""
    shards = []
    for r, w in enumerate(workers):
        my_idx = list(range(r, num_blocks, w.put_step))
        shards.append(w._pack_blocks(my_idx, w.test_addr_list, w.test_size_list, num_blocks))
    lens = [int(s.numel()) for s in shards]
    max_len = max(lens)
    recv = torch.zeros(max_len * len(workers), dtype=torch.uint8)
    for k, shard in enumerate(shards):
        recv[k * max_len : k * max_len + shard.numel()] = shard
    return recv, lens, max_len


def test_seg_plan_emits_one_strided_entry_per_segment():
    """A shard is one arithmetic run per segment, not a per-block copy list."""
    num_blocks, tp_size = 17, 4
    worker = make_worker(0, tp_size, num_blocks)
    plan = worker._seg_plan(
        list(range(0, num_blocks, tp_size)), worker.test_addr_list, worker.test_size_list, num_blocks
    )
    assert len(plan) == SEGS_PER_BLOCK
    for kind, _addr, block_bytes, n, stride in plan:
        assert kind == "stride"
        assert block_bytes == SEG_BYTES
        assert n == len(range(0, num_blocks, tp_size))
        assert stride == tp_size * SEGS_PER_BLOCK * SEG_BYTES


def test_seg_plan_falls_back_to_blocks_when_not_arithmetic():
    """A non arithmetic address pattern must degrade to per-segment copies."""
    num_blocks = 6
    worker = make_worker(0, 2, num_blocks)
    base = worker._kv_cache_addr_segments[0][0]
    # Deliberately irregular: block i starts at i*i segments.
    irregular = [[base + (i * i + s) * SEG_BYTES for s in range(SEGS_PER_BLOCK)] for i in range(num_blocks)]
    plan = worker._seg_plan([0, 2, 4], irregular, worker.test_size_list, num_blocks)
    assert all(item[0] == "blocks" for item in plan)
    assert sum(len(item[1]) for item in plan) == 3 * SEGS_PER_BLOCK


def test_pack_is_segment_major():
    """The packed buffer must follow the segment-major order the receiver expects."""
    num_blocks, tp_size = 9, 3
    worker = make_worker(1, tp_size, num_blocks)
    my_idx = list(range(1, num_blocks, tp_size))
    packed = worker._pack_blocks(my_idx, worker.test_addr_list, worker.test_size_list, num_blocks)
    cache = worker._kv_cache_addr_segments[0][2]
    # cat/stack 出来的形状是 (blocks*segs, SEG_BYTES)，必须展平才能和被
    # 测的扁平 uint8 buffer 比 —— 少了 reshape 这条断言恒 False（CI 抓到过）。
    expected = torch.cat([torch.stack([cache[i][s] for i in my_idx]).reshape(-1) for s in range(SEGS_PER_BLOCK)])
    assert torch.equal(packed, expected)


def test_strided_view_is_not_rebased_to_the_storage_start():
    """Regression: as_strided's storage_offset is absolute.

    Passing 0 re-bases the view on the storage start, so every rank but rank 0
    silently reads block 0. The packed bytes of rank 3 must be its own blocks.
    """
    num_blocks, tp_size = 12, 4
    rank = 3
    worker = make_worker(rank, tp_size, num_blocks)
    cache = worker._kv_cache_addr_segments[0][2]
    view = worker._resolve_stride_tensor(
        worker.test_addr_list[rank][0],
        SEG_BYTES,
        len(range(rank, num_blocks, tp_size)),
        tp_size * SEGS_PER_BLOCK * SEG_BYTES,
    )
    assert view is not None
    my_idx = list(range(rank, num_blocks, tp_size))
    for j, block in enumerate(my_idx):
        assert torch.equal(view[j], cache[block][0]), f"strided view row {j} is not block {block}"


def test_local_shard_lens_matches_the_real_pack():
    """The locally derived shard lengths must equal the actual packed byte counts."""
    num_blocks, tp_size = 17, 4
    workers = [make_worker(r, tp_size, num_blocks, seed=r) for r in range(tp_size)]
    local = workers[0]._local_shard_lens(num_blocks, workers[0].test_size_list)
    actual = [
        int(w._pack_blocks(list(range(r, num_blocks, tp_size)), w.test_addr_list, w.test_size_list, num_blocks).numel())
        for r, w in enumerate(workers)
    ]
    assert local == actual


@pytest.mark.parametrize("tp_size,num_blocks", [(2, 7), (4, 17), (8, 63), (4, 511)])
def test_roundtrip_reproduces_the_full_kv_cache(tp_size: int, num_blocks: int):
    """Every rank ends up with the complete KV cache after pack + gather + write-back."""
    torch.manual_seed(tp_size * 1000 + num_blocks)
    master = torch.randint(0, 256, (num_blocks, SEGS_PER_BLOCK, SEG_BYTES), dtype=torch.uint8)
    workers = [make_worker(r, tp_size, num_blocks, seed=r) for r in range(tp_size)]
    for w in workers:
        w._kv_cache_addr_segments[0][2].zero_()
    # 1) each rank reads only its own shard out of the pool
    for r, w in enumerate(workers):
        cache = w._kv_cache_addr_segments[0][2]
        for i in range(r, num_blocks, tp_size):
            cache[i].copy_(master[i])
    # 2) pack + all_gather + write-back, rank by rank
    for r, w in enumerate(workers):
        recv, lens, max_len = gather_sim(workers, num_blocks, r)
        w._writeback_shards(recv, lens, max_len, r, w.test_addr_list, w.test_size_list, num_blocks)
    # 3) every rank must hold the full KV cache byte for byte
    for r, w in enumerate(workers):
        assert torch.equal(w._kv_cache_addr_segments[0][2], master), f"rank {r} did not rebuild the KV cache"


def test_writeback_reports_a_single_strided_copy_per_segment():
    """The strided plan must collapse the write-back into one copy per segment."""
    num_blocks, tp_size = 65, 8
    workers = [make_worker(r, tp_size, num_blocks, seed=r) for r in range(tp_size)]
    recv, lens, max_len = gather_sim(workers, num_blocks, 0)
    runs, wb_ops, stride_ops = workers[0]._writeback_shards(
        recv, lens, max_len, 0, workers[0].test_addr_list, workers[0].test_size_list, num_blocks
    )
    peer_segments = SEGS_PER_BLOCK * (tp_size - 1)
    assert stride_ops == peer_segments
    assert runs == peer_segments
    assert wb_ops == peer_segments


def test_falls_back_when_the_backend_reports_missing_blocks():
    """A partial shard read must hand the request back to the full read path."""
    num_blocks, tp_size = 8, 4
    worker = make_worker(0, tp_size, num_blocks)
    worker.m_store = MagicMock()
    worker.m_store.get.return_value = [1, 0]
    assert worker._load_with_read_dedup(worker.test_keys, worker.test_addr_list, worker.test_size_list, []) is False


def test_empty_request_is_not_handled(monkeypatch):
    worker = make_worker(0, 4, 8)
    assert worker._load_with_read_dedup([], [], [], []) is False


def test_guards(monkeypatch):
    """Dedup must stay off for every configuration it was not verified on."""
    worker = make_worker(0, 4, 8)
    monkeypatch.delenv("VLLM_ASCEND_KV_POOL_READ_DEDUP", raising=False)
    assert worker._read_dedup_enabled()

    monkeypatch.setenv("VLLM_ASCEND_KV_POOL_READ_DEDUP", "0")
    assert not worker._read_dedup_enabled()
    monkeypatch.setenv("VLLM_ASCEND_KV_POOL_READ_DEDUP", "1")

    # put_step < tp_size: several ranks hold the same shard
    worker.put_step = 2
    assert not worker._read_dedup_enabled()
    worker.put_step = 4

    # DCP replication
    worker.dcp_size = 2
    assert not worker._read_dedup_enabled()
    worker.dcp_size = 1

    # KVPP owners hold layers, not shards
    worker.use_kvpp = True
    assert not worker._read_dedup_enabled()
    worker.use_kvpp = False

    # no address table means the exchange cannot write back
    worker._kv_cache_addr_segments = []
    assert not worker._read_dedup_enabled()


def test_lens_cache_is_bounded():
    """The shard length cache must not grow with the number of block counts."""
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import LENS_CACHE_MAX

    worker = make_worker(0, 2, 4)
    worker._lens_cache = {}
    worker._lens_ok = set()
    for total in range(LENS_CACHE_MAX * 2):
        if len(worker._lens_cache) >= LENS_CACHE_MAX:
            worker._lens_cache.clear()
            worker._lens_ok.clear()
        worker._lens_cache[total] = [total]
        worker._lens_ok.add(total)
    assert len(worker._lens_cache) <= LENS_CACHE_MAX
