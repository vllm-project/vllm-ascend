# SPDX-License-Identifier: Apache-2.0
"""PCP configuration and real multi-process shared-cache window contracts."""

import json
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.ascend_config import AscendConfig, EngramSSDConfig
from vllm_ascend.models.deepseek_v41.engram.ssd.cache import SSDRowCache
from vllm_ascend.models.deepseek_v41.engram.ssd.distributed import EngramSSDProtocol, LeaderCacheHooks, SSDProtocolError


@pytest.mark.parametrize('shared', [True, False])
def test_pcp_legacy_config_requires_shared_cache(shared):
    vc = SimpleNamespace(parallel_config=SimpleNamespace(
        pipeline_parallel_size=1, prefill_context_parallel_size=8,
        data_parallel_size=1, enable_eplb=False))
    config = SimpleNamespace(engram_ssd=EngramSSDConfig())
    engram = SimpleNamespace(cpu_offload=True, embedding_across_dp=False, dp_shared_memory=shared)
    errors = AscendConfig._validate_engram_ssd_cross_config(config, vc, engram)
    if shared:
        assert errors == []
    else:
        assert any('dp_shared_memory' in e for e in errors)


def _windows_worker(rank, init_file, directory, plane, merged):
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank,
                            world_size=3, timeout=timedelta(seconds=30))
    group = SimpleNamespace(cpu_group=dist.group.WORLD, world_size=3, rank_in_group=rank)
    cache = SSDRowCache(4, row_limit=10000) if rank == 0 else None

    class Hooks:
        def acquire(self, rows):
            assert len(np.unique(rows)) <= 4
            return cache.acquire(rows)

        def fill(self, rows, slots):
            for row, slot in zip(rows, slots):
                plane[int(slot)] = int(row)

        def publish(self, lease):
            cache.publish(lease)

        def release(self, lease):
            cache.release(lease)

    protocol = EngramSSDProtocol(group, epoch_fn=lambda: 1,
                                 leader_hooks=Hooks() if rank == 0 else None,
                                 merge_p23=merged, timeout_s=30)
    counts = [(2, 1, 0), (1, 3, 2), (0, 0, 0), (1, 0, 0)]
    expected_rounds = [2, 3, 1, 1]
    results = []
    try:
        for case, sizes in enumerate(counts):
            n = sizes[rank]
            windows = protocol.token_windows(n, rows_per_token=2, capacity=4)
            assert len(windows) == expected_rounds[case]
            seen = []
            for lo, hi in windows:
                rows = np.arange(2 * lo, 2 * hi, dtype=np.uint64) + 100 * case + 10 * rank
                slots = protocol.step(rows, num_tokens=hi - lo)
                # A faster peer may reach the next round, but cannot overwrite
                # the payload until this consumer has finished reading it.
                if rank == 1:
                    time.sleep(0.01)
                assert [int(plane[int(s)]) for s in slots] == rows.tolist()
                seen.extend(range(lo, hi))
            assert seen == list(range(n))
            results.append(windows)
        Path(directory, f'{rank}.json').write_text(json.dumps(results))
    finally:
        if cache is not None:
            cache.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('merged', [False, True])
def test_group_windows_bound_union_and_keep_uneven_idle_ranks_in_order(tmp_path, merged):
    plane = torch.full((4,), -1, dtype=torch.int64).share_memory_()
    mp.spawn(_windows_worker, args=(str(tmp_path / 'init'), str(tmp_path), plane, merged), nprocs=3)
    # Independent, hand-derived first batch: two tokens on rank0, one on rank1.
    assert json.loads((tmp_path / '0.json').read_text())[0] == [[0, 2], [2, 2]]
    assert json.loads((tmp_path / '1.json').read_text())[0] == [[0, 0], [0, 1]]
    assert json.loads((tmp_path / '2.json').read_text())[0] == [[0, 0], [0, 0]]


def test_shared_runtime_windows_preserve_layers_padding_and_empty_round(monkeypatch):
    from vllm_ascend.models.deepseek_v41.engram.ssd import runtime as module

    rt = module.EngramSSDRuntime.__new__(module.EngramSSDRuntime)
    rt.capacity, rt.num_layers, rt.col_count = 4, 2, 1
    rt.layer_ids, rt.layer_bases = [1, 14], [0, 100]
    rt._output = torch.full((2, 2, 1, 256), -99.0)
    rt._slot_host = torch.empty((2, 2, 1), dtype=torch.int32)
    rt._slot_dev = torch.empty_like(rt._slot_host)
    rt.planes = SimpleNamespace(codes=None, scales=None)
    rt._step = 0
    events, requests = [], []
    rt._gather_event = SimpleNamespace(record=lambda: events.append('record'),
                                      synchronize=lambda: events.append('done'))

    class Protocol:
        def token_windows(self, n, **kwargs):
            assert n == 2
            return [(0, 1), (1, 2), (2, 2)]

        def step(self, rows, **kwargs):
            if requests:
                assert events[-1] == 'done'
            requests.append(rows.tolist())
            return np.arange(len(rows), dtype=np.int32)

    def gather(codes, scales, ids, *, output, **kwargs):
        for i, slot in enumerate(ids.tolist()):
            output[i].fill_(0 if slot < 0 else requests[-1][slot])

    rt.protocol = Protocol()
    monkeypatch.setattr(module, 'gather_dequantize_host_uva', gather)
    host = np.array([[[3], [5]], [[7], [-1]]], dtype=np.int32)
    valid = [np.array([[True], [True]]), np.array([[True], [False]])]
    prepared = rt._prepare_shared_windowed(np.ones(2, bool), host, valid, 2, step_id=9)
    assert requests == [[3, 105], [7], []]
    assert prepared.layer_outputs[1][:, 0, 0].tolist() == [3.0, 7.0]
    assert prepared.layer_outputs[14][:, 0, 0].tolist() == [105.0, 0.0]
    assert prepared.step_id == 9
    assert prepared.stats['windows'] == 3


def _failure_worker(rank, init_file, n):
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    group = SimpleNamespace(cpu_group=dist.group.WORLD, world_size=2, rank_in_group=rank)
    cache = SSDRowCache(16384, row_limit=100000) if rank == 0 else None

    class Hooks(LeaderCacheHooks):
        fail_once = True

        def fill(self, rows, slots):
            if self.fail_once:
                self.fail_once = False
                raise OSError('injected SSD read failure')

    hooks = Hooks(cache, None, None, None) if rank == 0 else None
    protocol = EngramSSDProtocol(group, epoch_fn=lambda: 1, leader_hooks=hooks, timeout_s=30)
    rows = np.arange(n, dtype=np.uint64) + rank * n
    try:
        with pytest.raises(SSDProtocolError, match='leader reported failure'):
            protocol.step(rows)
        slots = protocol.step(rows)
        assert len(slots) == n and (slots >= 0).all()
        assert protocol.stats.rounds_failed == 1
        # 4096 rows per rank exceeds the merged-message threshold; 2 does not.
        assert ('p3_scatter' in protocol.stats.phase_ms) is (n == 4096)
    finally:
        if cache is not None:
            cache.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('rows_per_rank', [2, 4096])
def test_fill_failure_aborts_reserved_lease_and_next_round_recovers(tmp_path, rows_per_rank):
    mp.spawn(_failure_worker, args=(str(tmp_path / 'init'), rows_per_rank), nprocs=2)
