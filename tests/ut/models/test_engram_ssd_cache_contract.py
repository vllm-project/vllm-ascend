# SPDX-License-Identifier: Apache-2.0
"""T2 缓存契约测试:SSDRowCache 状态机与淘汰(04 §7)。

以可注入 fill 回调模拟引擎填充,oracle 用独立 dict 参考实现校验数据正确性。
不依赖 NPU/真实表。
"""

from __future__ import annotations

import numpy as np
import pytest

import vllm_ascend.ops  # noqa: F401,E402  # 既有导入环规避, 见 docs/02
from vllm_ascend.models.deepseek_v41.engram.ssd.cache import (  # noqa: E402
    CacheLease,
    SSDCacheError,
    SSDInputError,
    SSDRowCache,
)


class RefCache:
    """独立有限容量参考实现(仅未淘汰阶段可比; 淘汰后以'当前在缓存中的行'为准)。"""

    def __init__(self, cap: int):
        self.cap = cap
        self.data: dict[int, bytes] = {}

    def fill(self, rows, slots, row_bytes):
        for r, s in zip(rows, slots):
            self.data[int(s)] = row_bytes(int(r))


def row_bytes(gid: int) -> bytes:
    codes = ((np.arange(256, dtype=np.uint64) * 2654435761 + gid * 40503) % 256).astype(np.uint8)
    scales = (np.arange(8, dtype=np.float64) * 0.5 + gid % 977 + 1.0).astype(np.float32)
    return codes.tobytes() + scales.tobytes()


def do_lookup(cache: SSDRowCache, ref: RefCache, codes, scales, rows, *, fail=False):
    """acquire→(fill)→publish→read→release, 返回本批每行数据。"""
    lease = cache.acquire(rows)
    if len(lease.miss_rows):
        if fail:
            cache.abort(lease)
            return None
        cache._lib  # noqa: B018 - ensure bound
        # 模拟引擎 fill: miss 行写入平面
        for r, s in zip(lease.miss_rows, lease.miss_slots):
            b = row_bytes(int(r))
            codes[s] = np.frombuffer(b[:256], np.uint8)
            scales[s] = np.frombuffer(b[256:], np.uint8)
        ref.fill(lease.miss_rows, lease.miss_slots, row_bytes)
    cache.publish(lease)
    out = [codes[lease.slots[i]].tobytes() + scales[lease.slots[i]].tobytes() for i in range(len(rows))]
    cache.release(lease)
    return out


def test_hits_pinned_before_eviction():
    """本批命中的行不得被同批 miss 淘汰(单批唯一行≤capacity, 超容量批次由 runtime 分块)。"""
    cap = 64
    cache = SSDRowCache(cap, row_limit=100000)
    ref = RefCache(cap)
    codes = np.zeros((cap, 256), np.uint8)
    scales = np.zeros((cap, 32), np.uint8)
    rows = np.arange(50, dtype=np.uint64)
    got = do_lookup(cache, ref, codes, scales, rows)
    assert all(got[i] == row_bytes(int(rows[i])) for i in range(len(rows)))
    # 第二批: 50 个旧行全部命中 + 10 个新行 → 命中被 pin, 不得被同批 miss 挤掉
    rows2 = np.concatenate([rows, np.arange(50, 60, dtype=np.uint64)])
    got2 = do_lookup(cache, ref, codes, scales, rows2)
    assert all(got2[i] == row_bytes(int(rows2[i])) for i in range(len(rows2))), "命中行被同批淘汰或数据错误"
    # 第三批: 只查旧行, 仍应全部命中且数据不变
    l3 = cache.acquire(rows)
    assert l3.stats["hit_rows"] == 50
    cache.abort(l3)
    cache.close()


def test_duplicate_miss_filled_once():
    cap = 32
    cache = SSDRowCache(cap, row_limit=100000)
    rows = np.array([5, 5, 5, 7, 7, 9], np.uint64)
    lease = cache.acquire(rows)
    assert len(lease.miss_rows) == 3, f"重复 miss 应去重: {lease.miss_rows}"
    assert sorted(lease.miss_rows.tolist()) == [5, 7, 9]
    # 请求顺序还原: 相同行获得相同槽位
    assert lease.slots[0] == lease.slots[1] == lease.slots[2]
    assert lease.slots[3] == lease.slots[4]
    cache.abort(lease)
    cache.close()


def test_abort_removes_pending():
    cap = 32
    cache = SSDRowCache(cap, row_limit=100000)
    ref = RefCache(cap)
    codes = np.zeros((cap, 256), np.uint8)
    scales = np.zeros((cap, 32), np.uint8)
    # 批1 正常
    do_lookup(cache, ref, codes, scales, np.array([1, 2, 3], np.uint64))
    # 批2 失败 abort → 新行不可见
    assert do_lookup(cache, ref, codes, scales, np.array([10, 11], np.uint64), fail=True) is None
    lease = cache.acquire(np.array([10], np.uint64))
    assert len(lease.miss_rows) == 1, "abort 后的行必须重新 miss"
    cache.abort(lease)
    # 老行仍命中
    lease = cache.acquire(np.array([1, 2, 3], np.uint64))
    assert lease.stats["hit_rows"] == 3
    cache.release(lease) if lease.published else cache.abort(lease)
    cache.close()


def test_second_active_lease_rejected():
    cap = 16
    cache = SSDRowCache(cap, row_limit=100000)
    l1 = cache.acquire(np.array([1], np.uint64))
    with pytest.raises(SSDCacheError):
        cache.acquire(np.array([2], np.uint64))
    # 过期租约操作被拒
    with pytest.raises(SSDCacheError):
        cache.publish(
            CacheLease(
                lease_id=999,
                rows=np.zeros(0, np.uint64),
                slots=np.zeros(0, np.int32),
                miss_rows=np.zeros(0, np.uint64),
                miss_slots=np.zeros(0, np.int32),
            )
        )
    # release 前不允许直接 release(publish 先行)
    with pytest.raises(SSDCacheError):
        cache.release(l1)
    cache.publish(l1)
    cache.release(l1)
    l2 = cache.acquire(np.array([2], np.uint64))
    assert l2.lease_id > l1.lease_id
    cache.abort(l2)
    cache.close()


def test_batch_exceeds_capacity_rejected():
    cap = 8
    cache = SSDRowCache(cap, row_limit=100000)
    with pytest.raises(SSDCacheError):
        cache.acquire(np.arange(9, dtype=np.uint64))
    cache.close()


def test_row_range_and_input_validation():
    cache = SSDRowCache(16, row_limit=100)
    with pytest.raises(SSDInputError):
        cache.acquire(np.array([100], np.uint64))  # 越界
    with pytest.raises(SSDInputError):
        cache.acquire(np.array([[1]], np.uint64))  # 非一维
    with pytest.raises(SSDInputError):
        cache.acquire(np.array([1], np.int64))  # dtype
    cache.close()


def test_small_cache_long_churn():
    """≥100 倍容量的不同 keys:数据始终正确;记录 tombstone/探测/维护退化。

    重建阈值 cap/8: 探测长度上界 ≈ used(≤cap) + 阈值, 断言 < cap+cap/4。
    """
    cap = 512
    cache = SSDRowCache(cap, row_limit=10_000_000, rebuild_threshold=cap // 8)
    RefCache(cap)
    codes = np.zeros((cap, 256), np.uint8)
    scales = np.zeros((cap, 32), np.uint8)
    rng = np.random.default_rng(2026)
    n_steps = 300
    batch = 128
    probe_max_hist, tomb_hist, maintain_hist = [], [], []
    rebuilds = evictions = 0
    for step in range(n_steps):
        rows = rng.integers(0, 100 * cap, batch).astype(np.uint64)
        lease = cache.acquire(rows)
        for r, s in zip(lease.miss_rows, lease.miss_slots):
            b = row_bytes(int(r))
            codes[s] = np.frombuffer(b[:256], np.uint8)
            scales[s] = np.frombuffer(b[256:], np.uint8)
        cache.publish(lease)
        bad = 0
        for i, r in enumerate(rows):
            s = lease.slots[i]
            if codes[s].tobytes() + scales[s].tobytes() != row_bytes(int(r)):
                bad += 1
        cache.release(lease)
        assert bad == 0, f"step{step}: {bad}/{batch} 行数据错误"
        st = lease.stats
        probe_max_hist.append(st["probe_max"])
        tomb_hist.append(st["tombstones"])
        maintain_hist.append(st["maintain_ms"])
        rebuilds += st["rebuilt"]
        evictions += st["evictions"]
    assert evictions > 0, "churn 未触发淘汰"
    tomb = np.array(tomb_hist)
    probe = np.array(probe_max_hist)
    maint = np.array(maintain_hist)
    print(
        f"churn: evictions={evictions} rebuilds={rebuilds} "
        f"tomb p50/max={np.percentile(tomb, 50):.0f}/{tomb.max()} "
        f"probe p50/p90/max={np.percentile(probe, 50):.0f}/{np.percentile(probe, 90):.0f}/{probe.max()} "
        f"maintain max={maint.max():.3f}ms"
    )
    assert probe.max() < cap + cap // 4, f"探测长度失控: {probe.max()}"
    if rebuilds:
        assert tomb.min() <= cap // 8, "重建后 tombstone 未回落"
    cache.close()


def test_poison_freezes_cache():
    cache = SSDRowCache(16, row_limit=100000)
    lease = cache.acquire(np.array([1], np.uint64))
    cache.poison("consumer state unknown")
    cache.abort(lease)
    with pytest.raises(SSDCacheError):
        cache.acquire(np.array([2], np.uint64))
    assert cache.poisoned
    cache.close()
