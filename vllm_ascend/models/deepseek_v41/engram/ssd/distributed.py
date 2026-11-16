# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Engram SSD DP owner/client protocol (docs/04 §10, R2 amendment).

One payload plane and one row cache per node-local same-TP-position engram
DP group. The group leader (rank 0 of the engram DP CPU group) owns the
cache and the IO engine; every rank — busy or engine-idle — joins the same
five-phase per-step protocol on the CPU (gloo) group:

    P0  header all_gather   [epoch, round, valid_count, status, tag]
    P1  padded rows gather  uint64 packed ids, leader-side dedup+acquire
    P2  status broadcast    leader publishes acquire/fill outcome
    P3  slots scatter       int32 slots per rank (empty for idle ranks)
    P4  ack all_gather      every consumer confirms receipt

Activation is NOT self-timed: a rank joins only at ``epoch >= 1``, and the
epoch flip is commanded through the ordered engine-side channel (the parent
process activates workers after all cores finish warmup, before READY), never
by a collective racing engine traffic — startup windows are not lockstepped
and ``monitored_barrier`` timeouts poison the shared gloo group (T0
evidence, runs/20261109T2200Z-t0-dp-probe-tp4dp2). Once all replicas are in
the serving loop, the engine's own wave machinery guarantees a peer step
arrives within one step, so P0 is a plain blocking all_gather whose wait is
bounded by construction; a watchdog still bounds the whole step.

Failure rule: any P2 failure status means no slots follow (P3 skipped in
the same order everywhere); every rank raises, the runtime aborts its local
view, and the group retries the round at the next step.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

_HEADER_LEN = 5  # [epoch, round, valid_count, status, tag]

STATUS_OK = 0
STATUS_FAIL = 1

PHASE_P0 = "p0_header"
PHASE_P1 = "p1_gather"
PHASE_P2 = "p2_status"
PHASE_P3 = "p3_scatter"
PHASE_P4 = "p4_ack"


class SSDProtocolError(RuntimeError):
    """Protocol violation (order mismatch, failed round, dead peer)."""


@dataclass
class ProtocolStats:
    steps: int = 0
    rows_total: int = 0
    miss_rows_total: int = 0
    rounds_failed: int = 0
    bytes_gathered: int = 0
    phase_ms: dict[str, float] = field(default_factory=dict)

    def add_phase(self, phase: str, ms: float) -> None:
        self.phase_ms[phase] = self.phase_ms.get(phase, 0.0) + ms


class EngramSSDProtocol:
    """Per-step CPU protocol over the engram DP group's gloo cpu_group.

    One instance per rank. The leader additionally drives the cache and IO
    engine through ``leader_hooks`` so the protocol and the payload plane
    stay in one transaction.

    Args:
        group: GroupCoordinator of the node-local engram DP group.
        epoch_fn: callable returning the current protocol epoch (>=1 means
            active). Sourced from the ordered engine-side activation
            channel, not from a self-timed collective.
        timeout_s: hard bound for one protocol step (watchdog exit 87).
        leader_hooks: leader-side transaction hooks — ``acquire(rows) ->
            lease``, ``fill(miss_rows, miss_slots)``, ``publish(lease)``,
            ``release(lease)`` — provided by SSDRowCache + SSDFillEngine;
            the protocol drives them between P1 and P4.
    """

    def __init__(self, group, *, epoch_fn, timeout_s: float = 120.0,
                 leader_hooks: Any | None = None, log_fn=None,
                 merge_p23: bool = True):
        self.group = group
        self.pg = group.cpu_group
        self.size = group.world_size
        self.rank = group.rank_in_group
        self.leader_global = dist.get_global_rank(self.pg, 0)
        self.epoch_fn = epoch_fn
        self.timeout_s = timeout_s
        self.hooks = leader_hooks
        self.merge_p23 = merge_p23
        self.round = 0
        self.stats = ProtocolStats()
        self._log_fn = log_fn
        self._watchdog: threading.Thread | None = None
        self._deadline = float("inf")
        self._last_phase = ""
        if self.pg is not None:
            self._start_watchdog()

    # ---------- logging / watchdog ----------

    def _log(self, evt: str, **kw) -> None:
        if self._log_fn is not None:
            self._log_fn(evt=evt, round=self.round, rank=self.rank, **kw)

    def _start_watchdog(self) -> None:
        if self._watchdog is not None:
            return

        def _run():
            import os

            while True:
                time.sleep(1.0)
                d = self._deadline
                if d != float("inf") and time.monotonic() > d:
                    self._log("timeout", deadline_s=self.timeout_s,
                              phase=self._last_phase)
                    os._exit(87)

        self._watchdog = threading.Thread(target=_run, daemon=True,
                                          name="engram-ssd-proto-wd")
        self._watchdog.start()

    # ---------- per-step entry ----------

    def step(self, rows: np.ndarray, *, num_tokens: int = 0) -> np.ndarray:
        """Join one protocol round; returns slots aligned with ``rows``.

        Idle ranks pass an empty ``rows`` array (valid_count 0) and still
        join every phase — P0 headers decide the common padded length.
        Raises SSDProtocolError on a failed round (no partial slot use).
        """
        if self.pg is None:
            raise SSDProtocolError("protocol group unavailable")
        epoch = int(self.epoch_fn())
        if epoch < 1:
            raise SSDProtocolError("step() called with protocol epoch 0")
        rows = np.ascontiguousarray(rows, dtype=np.uint64)
        n = len(rows)
        slots = np.full(n, -1, np.int32)

        t0 = time.monotonic()
        self._deadline = time.monotonic() + self.timeout_s
        try:
            if self.merge_p23:
                self._step_merged(epoch, rows, slots, num_tokens)
            else:
                headers, pad = self._p0(epoch, n, num_tokens)
                gathered = self._p1(rows, pad)
                status = self._p2(headers, gathered, pad)
                if status == STATUS_FAIL:
                    # 全组同序跳过 P3; 本轮作废, 由 runtime 在下一步重试
                    self.stats.rounds_failed += 1
                    self._p4(STATUS_FAIL)
                    raise SSDProtocolError(
                        f"round {self.round}: leader reported failure")
                self._p3(gathered, headers, slots, pad)
                self._p4(STATUS_OK)
        finally:
            self._deadline = float("inf")
        self.stats.steps += 1
        self.stats.rows_total += n
        self._log("step-done", tokens=num_tokens, rows=n, ms=(time.monotonic() - t0) * 1e3)
        self.round += 1
        return slots

    def _step_merged(self, epoch: int, rows: np.ndarray, slots: np.ndarray,
                     num_tokens: int) -> None:
        """P2+P3 合并(04 §10): leader 单广播 status+全组槽位, 少一轮往返。

        decode pad 小(数百行×4B), 广播全组槽位传输增量可忽略; prefill 大 pad
        时传输为 size*pad*8B(可后续按需回退分相模式)。
        """
        import time as _time

        g = self.pg
        size, rank = self.size, self.rank
        leader_global = dist.get_global_rank(g, 0)
        n = len(rows)
        # P0
        header = torch.tensor([epoch, self.round, n, STATUS_OK, num_tokens],
                              dtype=torch.int64)
        out = [torch.zeros(_HEADER_LEN, dtype=torch.int64) for _ in range(size)]
        self._last_phase = PHASE_P0
        _t = _time.monotonic()
        dist.all_gather(out, header, group=g)
        self.stats.add_phase(PHASE_P0, (_time.monotonic() - _t) * 1e3)
        for t in out:
            if int(t[0]) != epoch or int(t[1]) != self.round:
                raise SSDProtocolError("P0 order mismatch (merged)")
        pad = max(max(int(t[2]) for t in out), 1)
        # P1
        self._last_phase = PHASE_P1
        payload = torch.zeros(pad, dtype=torch.int64)
        if n:
            payload[:n] = torch.from_numpy(rows[:n].astype(np.int64))
        _t = _time.monotonic()
        if rank == 0:
            buf = [torch.zeros(pad, dtype=torch.int64) for _ in range(size)]
            dist.gather(payload, buf, dst=leader_global, group=g)
            gathered = torch.stack(buf)
        else:
            dist.gather(payload, None, dst=leader_global, group=g)
            gathered = torch.empty(0)
        self.stats.add_phase(PHASE_P1, (_time.monotonic() - _t) * 1e3)
        # P2' 合并广播: [status, rank0 slots[pad], rank1 slots[pad], ...]
        self._last_phase = PHASE_P2
        bmsg = torch.zeros(1 + size * pad, dtype=torch.int64)
        if rank == 0:
            counts = [int(t[2]) for t in out]
            if self.hooks is not None:
                try:
                    parts = [gathered[r, :counts[r]].numpy().astype(np.uint64)
                             for r, c in enumerate(counts) if c > 0]
                    allrows = np.concatenate(parts) if parts else np.zeros(0, np.uint64)
                    lease = self.hooks.acquire(allrows)
                    if len(lease.miss_rows):
                        self.hooks.fill(lease.miss_rows, lease.miss_slots)
                    self.hooks.publish(lease)
                    self._leader_lease = lease
                    base = 0
                    for r, c in enumerate(counts):
                        if c > 0:
                            bmsg[1 + r * pad:1 + r * pad + c] = torch.from_numpy(
                                lease.slots[base:base + c].astype(np.int64))
                            base += c
                    self.stats.miss_rows_total += len(lease.miss_rows)
                except Exception as exc:  # noqa: BLE001 - 经 P2' 状态上报
                    self._log("leader-round-failed", err=str(exc)[:120])
                    bmsg[0] = STATUS_FAIL
                    lease = getattr(self, "_leader_lease", None)
                    if lease is not None:
                        try:
                            self.hooks.release(lease)
                        except Exception:
                            pass
                        self._leader_lease = None
        _t = _time.monotonic()
        dist.broadcast(bmsg, src=leader_global, group=g)
        self.stats.add_phase(PHASE_P2, (_time.monotonic() - _t) * 1e3)
        if int(bmsg[0]) == STATUS_FAIL:
            self.stats.rounds_failed += 1
            self._p4(STATUS_FAIL)
            raise SSDProtocolError(f"round {self.round}: leader reported failure")
        if n:
            got = bmsg[1 + rank * pad:1 + rank * pad + n].to(torch.int32).numpy()
            if (got < 0).any():
                raise SSDProtocolError("P2' returned unfilled slots")
            slots[:] = got
        self._p4(STATUS_OK)

    # ---------- phases ----------

    def _p0(self, epoch: int, n: int, num_tokens: int):
        self._last_phase = PHASE_P0
        header = torch.tensor([epoch, self.round, n, STATUS_OK, num_tokens],
                              dtype=torch.int64)
        out = [torch.zeros(_HEADER_LEN, dtype=torch.int64) for _ in range(self.size)]
        import time as _time
        _t0 = _time.monotonic()
        dist.all_gather(out, header, group=self.pg)
        self.stats.add_phase(PHASE_P0, (_time.monotonic() - _t0) * 1e3)
        for t in out:
            if int(t[0]) != epoch or int(t[1]) != self.round:
                raise SSDProtocolError(
                    f"P0 order mismatch: headers={[int(t[0:2].tolist()) for t in out]}")
        pad = max(max(int(t[2]) for t in out), 1)
        return out, pad

    def _p1(self, rows: np.ndarray, pad: int) -> torch.Tensor:
        self._last_phase = PHASE_P1
        payload = torch.zeros(pad, dtype=torch.int64)
        n = min(len(rows), pad)
        if n:
            payload[:n] = torch.from_numpy(rows[:n].astype(np.int64))
        counts = torch.zeros(1, dtype=torch.int64)
        if self.rank == 0:
            buf = [torch.zeros(pad, dtype=torch.int64) for _ in range(self.size)]
            t0 = time.monotonic()
            dist.gather(payload, buf, dst=self.leader_global, group=self.pg)
            self.stats.add_phase(PHASE_P1, (time.monotonic() - t0) * 1e3)
            self.stats.bytes_gathered += pad * 8 * self.size
            # payload[:counts[r]] 为有效行, 其后为 pad 填充; leader 只读有效段
            return torch.stack(buf)          # [ranks, pad]
        t0 = time.monotonic()
        dist.gather(payload, None, dst=self.leader_global, group=self.pg)
        self.stats.add_phase(PHASE_P1, (time.monotonic() - t0) * 1e3)
        return torch.empty(0)

    def _p2(self, headers, gathered, pad: int) -> int:
        self._last_phase = PHASE_P2
        status = torch.zeros(1, dtype=torch.int64)
        if self.rank == 0:
            counts = [int(t[2]) for t in headers]
            st = STATUS_OK
            if self.hooks is not None:
                try:
                    flat = self._leader_acquire(gathered, counts, pad)
                    self._leader_fill()
                    self.hooks.publish(self._leader_lease)
                    self._leader_slots = flat
                except Exception as exc:  # noqa: BLE001 - reported via P2
                    self._log("leader-round-failed", err=str(exc)[:120])
                    st = STATUS_FAIL
                    lease = getattr(self, "_leader_lease", None)
                    if lease is not None:
                        try:
                            self.hooks.release(lease)
                        except Exception:
                            pass
                        self._leader_lease = None
            status[0] = st
        dist.broadcast(status, src=self.leader_global, group=self.pg)
        return int(status[0])

    def _p3(self, gathered, headers, slots: np.ndarray, pad: int) -> None:
        self._last_phase = PHASE_P3
        chunk = torch.zeros(pad, dtype=torch.int32)
        if self.rank == 0:
            counts = [int(t[2]) for t in headers]
            chunks = []
            for r in range(self.size):
                c = torch.full((pad,), -1, dtype=torch.int32)
                if self.hooks is not None and counts[r] > 0:
                    base = r * pad
                    c[:counts[r]] = self._leader_slots[base:base + counts[r]]
                chunks.append(c)
            t0 = time.monotonic()
            dist.scatter(chunk, chunks, src=self.leader_global, group=self.pg)
            self.stats.add_phase(PHASE_P3, (time.monotonic() - t0) * 1e3)
        else:
            t0 = time.monotonic()
            dist.scatter(chunk, None, src=self.leader_global, group=self.pg)
            self.stats.add_phase(PHASE_P3, (time.monotonic() - t0) * 1e3)
        n = len(slots)
        if n:
            got = chunk[:n].to(torch.int32)
            if int((got < 0).sum()):
                raise SSDProtocolError("P3 returned unfilled slots for valid rows")
            slots[:] = got.numpy()

    def _p4(self, status: int) -> None:
        self._last_phase = PHASE_P4
        ack = torch.tensor([self.round, status], dtype=torch.int64)
        acks = [torch.zeros(2, dtype=torch.int64) for _ in range(self.size)]
        dist.all_gather(acks, ack, group=self.pg)
        for a in acks:
            if int(a[0]) != self.round:
                raise SSDProtocolError(
                    f"P4 ack mismatch: {[[int(x) for x in a.tolist()] for a in acks]}")
        # 全组消费确认后, leader 释放租约(payload 保留供后续命中)
        if self.rank == 0 and status == STATUS_OK and self.hooks is not None:
            lease = getattr(self, "_leader_lease", None)
            if lease is not None:
                self.hooks.release(lease)
                self._leader_lease = None

    # ---------- leader-side transaction ----------

    _leader_slots: torch.Tensor | None = None

    def _leader_acquire(self, gathered: torch.Tensor, counts: list[int],
                        pad: int) -> torch.Tensor:
        """Concatenate per-rank valid rows (rank-major), dedup via cache.

        Returns slots laid out rank-major so P3 slices ``[r*pad, r*pad+n)``.
        """
        assert self.hooks is not None
        parts = []
        for r, c in enumerate(counts):
            if c > 0:
                parts.append(gathered[r, :c].numpy().astype(np.uint64))
        rows = np.concatenate(parts) if parts else np.zeros(0, np.uint64)
        lease = self.hooks.acquire(rows)
        self._leader_lease = lease
        self.stats.miss_rows_total += len(lease.miss_rows)
        # P3 按 r*pad 步长切片; lease.slots 与紧致拼接序对齐 → 按 rank 基址重排
        flat = torch.full((self.size * pad,), -1, dtype=torch.int32)
        base = 0
        for r, c in enumerate(counts):
            if c > 0:
                seg = lease.slots[base:base + c]
                flat[r * pad:r * pad + c] = torch.from_numpy(
                    seg.astype(np.int64)).to(torch.int32)
                base += c
        self._log("leader-acquire", rows=len(rows), misses=len(lease.miss_rows))
        return flat

    def _leader_fill(self) -> None:
        assert self.hooks is not None
        lease = getattr(self, "_leader_lease", None)
        if lease is not None and len(lease.miss_rows):
            self.hooks.fill(lease.miss_rows, lease.miss_slots)

    def close(self) -> None:
        self._log("close", steps=self.stats.steps,
                  failed=self.stats.rounds_failed)


class ProtocolEpoch:
    """Process-local epoch, activated by the ordered engine startup RPC.

    One worker owns one epoch shared by all Engram layers. Worker construction
    resets it before warmup; engine parents activate it only after every DP
    engine has finished warmup. Inference never performs filesystem I/O here.
    """

    _value = 0
    _lock = threading.Lock()

    @classmethod
    def reset(cls) -> None:
        """Start a fresh worker lifetime, before model loading/capture."""
        with cls._lock:
            cls._value = 0

    @classmethod
    def activate(cls, epoch: int = 1) -> int:
        if type(epoch) is not int or epoch < 1:
            raise ValueError("epoch must be an integer >= 1")
        with cls._lock:
            cls._value = max(cls._value, epoch)
            return cls._value

    @classmethod
    def current(cls) -> int:
        with cls._lock:
            return cls._value


class LeaderCacheHooks:
    """Bridge the group protocol to the leader-owned cache and IO engine.

    ``acquire``/``publish``/``release`` map straight onto SSDRowCache; ``fill``
    routes large miss batches through extent-merged reads and small ones
    through per-row reads.
    """

    def __init__(self, cache, engine, codes_np, scales_np, *, base1: int = 0,
                 extents_threshold: int = 64):
        self.cache = cache
        self.engine = engine
        self.codes_np = codes_np
        self.scales_np = scales_np
        self.base1 = base1
        self.extents_threshold = extents_threshold

    def acquire(self, rows: np.ndarray):
        return self.cache.acquire(rows)

    def fill(self, miss_rows: np.ndarray, miss_slots: np.ndarray) -> None:
        if len(miss_rows) == 0:
            return
        if len(miss_rows) >= self.extents_threshold:
            from vllm_ascend.models.deepseek_v41.engram.ssd.engine import plan_row_extents

            ext, sct, _ = plan_row_extents(miss_rows, miss_slots,
                                           base1=self.base1,
                                           dual=getattr(self.engine, "dual", False))
            self.engine.read_extents(ext, sct, self.codes_np, self.scales_np)
        else:
            self.engine.fill(miss_rows, miss_slots, self.codes_np, self.scales_np)

    def publish(self, lease) -> None:
        self.cache.publish(lease)

    def release(self, lease) -> None:
        self.cache.release(lease)
