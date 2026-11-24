# SPDX-License-Identifier: Apache-2.0
"""Real Gloo/cache contracts for reusable protocol storage and owner telemetry."""

import io
import logging
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.deepseek_v41.engram.ssd.cache import SSDRowCache
from vllm_ascend.models.deepseek_v41.engram.ssd.distributed import EngramSSDProtocol, LeaderCacheHooks


def _worker(rank, init_file, plane, merged, telemetry):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank,
                            world_size=3, timeout=timedelta(seconds=40))
    group = SimpleNamespace(cpu_group=dist.group.WORLD, world_size=3, rank_in_group=rank)
    cache = SSDRowCache(50000, row_limit=100000) if rank == 0 else None

    class Hooks(LeaderCacheHooks):
        def fill(self, rows, slots):
            plane.numpy()[slots] = rows

    hooks = Hooks(cache, None, None, None) if rank == 0 else None
    protocol = EngramSSDProtocol(group, epoch_fn=lambda: 1, leader_hooks=hooks,
                                 merge_p23=merged, timeout_s=40)
    retained = []
    log_output = io.StringIO()
    handler = logging.StreamHandler(log_output)
    logger = logging.getLogger(EngramSSDProtocol.__module__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        if telemetry:
            cases = [([7, 7], [7, 8], []), ([7], [9, 9], [8]), ([], [], []), ([], [9], [8])]
            expected = [(4, 0, 4, 2, 0.0), (4, 2, 2, 1, 0.5), (0, 0, 0, 0, None), (2, 2, 0, 0, 1.0)]
        else:
            # Grow, shrink, cross merged/split threshold, and include idle peers.
            counts = [(4096, 4000, 0), (2, 0, 1), (5000, 5000, 5000),
                      (0, 0, 0), (1, 3, 2), (4096, 4096, 4096)]
            cases = [tuple(np.arange(n, dtype=np.uint64) + 10000 * r for r, n in enumerate(c))
                     for c in counts]
        for index, case in enumerate(cases):
            rows = np.array(case[rank], dtype=np.uint64)
            # Rank1 can own a large padded prefill while the cache owner is idle.
            tokens = 600 if telemetry and index == 3 and rank == 1 else len(rows)
            slots = protocol.step(rows, num_tokens=tokens)
            assert slots.dtype == np.int32
            np.testing.assert_array_equal(plane.numpy()[slots], rows)
            for previous, saved in retained:
                np.testing.assert_array_equal(previous, saved)
            retained.append((slots, slots.copy()))
            if telemetry:
                observed = protocol.last_cache_stats
                if rank == 0:
                    got = tuple(observed[k] for k in
                                ('rows', 'hit_rows', 'miss_rows', 'unique_misses', 'hit_rate'))
                    assert got == expected[index]
                    assert observed['fill_wall_ms'] >= 0
                    if index == 2:
                        assert observed['fill_wall_ms'] == 0
                else:
                    assert observed is None  # Do not report fake zeros on clients.
        if telemetry and rank == 0:
            assert protocol.stats.query_rows_total == 10
            assert protocol.stats.hit_rows_total == 4
            assert protocol.stats.miss_queries_total == 6
            assert protocol.stats.miss_rows_total == 3  # Unique fill rows, not query misses.
            assert 'scope=owner_group round=3 ' in log_output.getvalue()
    finally:
        logger.removeHandler(handler)
        if cache is not None:
            cache.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('merged', [False, True])
def test_owner_cache_counts_duplicates_partial_hits_and_empty_rounds(tmp_path, merged):
    plane = torch.zeros(50000, dtype=torch.int64).share_memory_()
    mp.spawn(_worker, args=(str(tmp_path / 'init'), plane, merged, True), nprocs=3)


@pytest.mark.parametrize('merged', [False, True])
def test_variable_batches_preserve_payload_and_previous_slot_results(tmp_path, merged):
    plane = torch.zeros(50000, dtype=torch.int64).share_memory_()
    mp.spawn(_worker, args=(str(tmp_path / 'init'), plane, merged, False), nprocs=3)


def test_runtime_reports_unknown_cache_on_clients_instead_of_zero_misses():
    from vllm_ascend.models.deepseek_v41.engram.ssd.runtime import EngramSSDRuntime

    rt = EngramSSDRuntime.__new__(EngramSSDRuntime)
    rt.protocol = SimpleNamespace(last_cache_stats=None)
    observed = rt._cache_observation(None)
    assert observed['hits'] is None and observed['misses'] is None
    assert observed['cache_scope'] == 'owner_group_unavailable'

    rt.protocol.last_cache_stats = dict(rows=10, hit_rows=6, miss_rows=4,
                                        unique_misses=2, fill_wall_ms=1.25)
    observed = rt._cache_observation(None)
    assert observed['hits'] == 6
    assert observed['misses'] == 2
    assert observed['query_misses'] == 4
    assert observed['cache_scope'] == 'owner_group'
    lease = SimpleNamespace(stats=dict(hit_rows=3, miss_rows=2), miss_rows=[1])
    observed = rt._cache_observation(lease)
    assert observed['hits'] == 3 and observed['misses'] == 1
    assert observed['cache_scope'] == 'local'
