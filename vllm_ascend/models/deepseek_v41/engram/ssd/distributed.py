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
"""Engram SSD row-cache protocol for DP and PCP ranks.

One payload plane and one row cache per node-local same-TP-position engram
DP group. The group leader (rank 0 of the engram DP CPU group) owns the
cache and the IO engine; every rank — busy or engine-idle — joins the same
five-phase per-step protocol on the CPU (gloo) group:

    P0  header all_gather   [epoch, round, valid_count, first_layer_count, tag]
    P1  padded rows gather  uint64 packed ids, leader-side dedup+acquire
    P2  status broadcast    leader publishes acquire/fill outcome
    P3  slots scatter       int32 slots per rank (empty for idle ranks)
    P4  ack all_gather      every consumer confirms receipt

Engine parents activate workers after warmup and before READY. Once
the replicas are serving, the engine's DP waves include idle ranks, so
P0 can use a blocking all_gather. A watchdog bounds the full round.

Failure rule: any P2 failure status means no slots follow (P3 skipped in
the same order everywhere); every rank raises, the runtime aborts its local
view, and the group retries the round at the next step.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

# Keep Decode packets small. Batches exceeding this per-layer limit retry with variable sizes.
LAYERED_DECODE_INLINE_ROWS = 768
_HEADER_LEN = 5  # [epoch, round, valid_count, first_layer_count, tag]
_MERGED_SLOT_MAX_BYTES = 64 * 1024
_CACHE_LOG_MIN_TOKENS = 512
_CACHE_LOG_INTERVAL = 200

STATUS_OK = 0
STATUS_FAIL = 1
STATUS_UNSAFE = 2
_STATUS_FALLBACK = 3
_STATUS_INVALID_REQUEST = 4
_STATUS_STAGING_FAILED = 5

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
    hit_rows_total: int = 0
    miss_queries_total: int = 0
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
        layered_inline_rows: fixed per-layer request capacity for packed layered
            rounds. Zero uses the variable-size protocol for all rounds.
        timeout_s: hard bound for one protocol step (watchdog exit 87).
        leader_hooks: leader-side transaction hooks — ``acquire(rows) ->
            lease``, ``fill(miss_rows, miss_slots)``, ``publish(lease)``,
            ``release(lease)`` — provided by SSDRowCache + SSDFillEngine;
            the protocol drives them between P1 and P4.
    """

    def __init__(self, group, *, epoch_fn, timeout_s: float = 120.0,
                 leader_hooks: Any | None = None, log_fn=None,
                 merge_p23: bool = True, cpu_group=None,
                 layered_inline_rows: int = 0):
        self.group = group
        self.pg = group.cpu_group if cpu_group is None else cpu_group
        self.size = group.world_size
        self.rank = group.rank_in_group
        self.leader_global = dist.get_global_rank(self.pg, 0)
        self.epoch_fn = epoch_fn
        self.timeout_s = timeout_s
        self.hooks = leader_hooks
        self.merge_p23 = merge_p23
        self.layered_inline_rows = layered_inline_rows
        self.round = 0
        self.stats = ProtocolStats()
        self._log_fn = log_fn
        self._watchdog: threading.Thread | None = None
        self._deadline = float("inf")
        self._last_phase = ""
        self._buffers: dict[str, torch.Tensor] = {}
        self._leader_slots: torch.Tensor | None = None
        self._leader_lease = None
        self._group_max_tokens = 0
        self._layered_poisoned = False
        # Only the owner observes the shared cache. None on clients is not 0%.
        self.last_cache_stats: dict[str, Any] | None = None
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

    def _cpu_buffer(self, name: str, count: int, dtype) -> torch.Tensor:
        """Reuse one high-water allocation per role; all collectives are blocking.

        NumPy backing also keeps these buffers on CPU under a default NPU
        device context. Host-only fills/copies use its views rather than
        launching Torch's CPU thread pool for small metadata arrays.
        """
        buffer = self._buffers.get(name)
        if buffer is None or buffer.numel() < count:
            capacity = 1 << max(count - 1, 0).bit_length()
            buffer = torch.from_numpy(np.empty(capacity, dtype=dtype))
            self._buffers[name] = buffer
        return buffer[:count]

    def token_windows(self, num_tokens: int, *, rows_per_token: int,
                      capacity: int) -> list[tuple[int, int]]:
        """Bound the union of all ranks' rows, with identical window counts.

        Tokens are ordered rank-major solely for capacity planning. Each rank
        keeps its original token order. Even an empty rank joins every window.
        Call only when the configured worst-case group batch can exceed capacity;
        the ordinary large-cache path does not pay for this planning collective.
        """
        if num_tokens < 0 or rows_per_token < 1 or capacity < rows_per_token:
            raise SSDProtocolError("cache capacity must hold at least one token's rows")
        epoch = int(self.epoch_fn())
        if epoch < 1:
            raise SSDProtocolError("window planning before protocol activation")
        self._deadline = time.monotonic() + self.timeout_s
        try:
            headers, _ = self._p0(epoch, 0, num_tokens)
        finally:
            self._deadline = float("inf")
        counts = [int(header[4]) for header in headers]
        start = sum(counts[:self.rank])
        total = sum(counts)
        width = capacity // rows_per_token
        return [
            (min(max(offset - start, 0), num_tokens),
             min(max(offset + width - start, 0), num_tokens))
            for offset in range(0, max(total, 1), width)
        ]

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
        self.last_cache_stats = None

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
        self._finish_round(n, num_tokens, t0)
        return slots

    def step_layered(self, rows: np.ndarray, first_count: int, split_row_id: int,
                     ready, retired, *, num_tokens: int = 0, submit_layer=None,
                     synchronize_gathers=None) -> None:
        """Run on the CPU producer thread, holding one lease across both fills.

        A ready future gets the rank-local slots or the result of submitting
        that layer on the producer stream. The caller signals retirement after
        all submissions finish and UVA no longer reads the lease. The prefill
        lookahead may instead check that dependency on the producer itself.
        A negative first_count propagates a local D2H/input failure in P0.
        """
        t0 = time.monotonic()
        self._deadline = t0 + self.timeout_s
        self.last_cache_stats = None
        failure = None
        try:
            epoch = int(self.epoch_fn())
            if self.pg is None or epoch < 1 or self._layered_poisoned:
                raise SSDProtocolError("layered protocol unavailable")
            inline = self.layered_inline_rows
            if inline:
                request_status, gathered, counts, first, pad = self._gather_layered_inline(
                    epoch, rows, first_count, num_tokens, inline)
                failure = self._deliver_layers(
                    gathered, counts, first, pad, split_row_id, ready, submit_layer,
                    inline=inline, request_status=request_status,
                    local_widths=(first_count, len(rows) - first_count))
            if not inline or failure == _STATUS_FALLBACK:
                gathered, counts, first, pad = self._gather_layered_variable(
                    epoch, rows, first_count, num_tokens)
                failure = self._deliver_layers(
                    gathered, counts, first, pad, split_row_id, ready, submit_layer)
            ready_ms = (time.monotonic() - t0) * 1e3
            self._last_phase = "retire_uva"
            safe = retired.result() if synchronize_gathers is None else synchronize_gathers()
            status_code = STATUS_FAIL if failure is not None else STATUS_OK
            self._p4(status_code if safe else STATUS_UNSAFE, layered=True)
            if failure is not None:
                self._discard_leader_lease()
                self.round += 1
            else:
                self._finish_round(len(rows), num_tokens, t0, lookup_ms=ready_ms)
        except BaseException as exc:
            # A broken collective or unsafe UVA retirement leaves ownership unknown.
            # No later round may reuse its slots.
            self._layered_poisoned = True
            for result in ready:
                if not result.done():
                    result.set_exception(exc)
            raise
        finally:
            self._deadline = float("inf")
        if failure is not None:
            raise failure

    def _gather_layered_variable(self, epoch: int, rows: np.ndarray,
                                 first_count: int, num_tokens: int):
        headers, pad = self._p0(epoch, len(rows), num_tokens, first_count=first_count)
        counts = [int(header[2]) for header in headers]
        first = [int(header[3]) for header in headers]
        if any(value < 0 for value in first):
            self.stats.rounds_failed += 1
            raise SSDProtocolError(f"round {self.round}: a rank failed to stage input rows")
        return self._p1(rows, pad), counts, first, pad

    def _gather_layered_inline(self, epoch: int, rows: np.ndarray,
                               first_count: int, num_tokens: int, capacity: int):
        self._last_phase = PHASE_P1
        pad = 2 * capacity
        packet_width = _HEADER_LEN + pad
        payload = self._cpu_buffer("layer_request", packet_width, np.int64)
        packet = payload.numpy()
        packet[:_HEADER_LEN] = (epoch, self.round, len(rows), first_count, num_tokens)
        valid = min(len(rows), pad)
        packet[_HEADER_LEN:_HEADER_LEN + valid].view(np.uint64)[:] = rows[:valid]
        packet[_HEADER_LEN + valid:].fill(0)
        gathered = None
        inputs = None
        if self.rank == 0:
            gathered = self._cpu_buffer("layer_requests", self.size * packet_width,
                                        np.int64).view(self.size, packet_width)
            inputs = list(gathered.unbind())
        start = time.monotonic()
        dist.gather(payload, inputs, dst=self.leader_global, group=self.pg)
        self.stats.add_phase(PHASE_P1, (time.monotonic() - start) * 1e3)
        if self.rank != 0:
            return STATUS_OK, None, [], [], pad

        self.stats.bytes_gathered += self.size * packet_width * 8
        headers = gathered.numpy()[:, :_HEADER_LEN]
        if (headers[:, 0] != epoch).any() or (headers[:, 1] != self.round).any():
            self._log("packed-order-mismatch", headers=headers[:, :2].tolist())
            return _STATUS_INVALID_REQUEST, None, [], [], pad
        counts = headers[:, 2].tolist()
        first = headers[:, 3].tolist()
        if any(value < 0 for value in first):
            return _STATUS_STAGING_FAILED, None, [], [], pad
        if any(total < split for total, split in zip(counts, first)):
            return _STATUS_INVALID_REQUEST, None, [], [], pad
        if any(max(split, total - split) > capacity for total, split in zip(counts, first)):
            return _STATUS_FALLBACK, None, [], [], pad
        self._group_max_tokens = int(headers[:, 4].max())
        return STATUS_OK, gathered[:, _HEADER_LEN:], counts, first, pad

    def _scatter_layer_result(self, status: int, packed, starts: list[int],
                              ends: list[int], width: int, capacity: int):
        self._last_phase = PHASE_P3
        packet_width = 1 + capacity
        chunk = self._cpu_buffer("layer_result", packet_width, np.int32)
        chunks = None
        if self.rank == 0:
            send = self._cpu_buffer("layer_results", self.size * packet_width,
                                    np.int32).view(self.size, packet_width)
            matrix = send.numpy()
            matrix[:, 0] = status
            if status == STATUS_OK:
                for rank, (begin, end) in enumerate(zip(starts, ends)):
                    limit = 1 + end - begin
                    matrix[rank, 1:limit] = packed[rank, begin:end]
                    matrix[rank, limit:].fill(-1)
            else:
                matrix[:, 1:].fill(-1)
            chunks = list(send.unbind())
        start = time.monotonic()
        dist.scatter(chunk, chunks, src=self.leader_global, group=self.pg)
        self.stats.add_phase(PHASE_P3, (time.monotonic() - start) * 1e3)
        result = chunk.numpy()
        status = int(result[0])
        if status != STATUS_OK:
            return status, None
        slots = result[1:1 + width].copy()
        if (slots < 0).any():
            raise SSDProtocolError("P3 returned unfilled layer slots")
        return status, slots

    def _deliver_layers(self, gathered, counts: list[int], first: list[int],
                        pad: int, split_row_id: int, ready, submit_layer,
                        *, inline: int = 0, request_status: int = STATUS_OK,
                        local_widths: tuple[int, int] = (0, 0)):
        packed = None
        first_misses = None
        starts, ends = [0] * self.size, first
        for stage, local_width in enumerate(local_widths):
            status = request_status if stage == 0 else STATUS_OK
            if self.rank == 0 and status == STATUS_OK:
                self._last_phase = PHASE_P2
                try:
                    if stage == 0:
                        packed = self._leader_acquire(gathered, counts, pad).numpy().reshape(self.size, pad)
                        first_misses = self._leader_lease.miss_rows < np.uint64(split_row_id)
                        self.last_cache_stats["layer_fill_ms"] = [0.0, 0.0]
                    selection = first_misses if stage == 0 else ~first_misses
                    self._leader_fill(selection, stage=stage)
                    if stage == 1:
                        self.hooks.publish(self._leader_lease)
                except Exception as exc:
                    self._log("leader-round-failed", stage=stage + 1, err=str(exc)[:120])
                    logging.getLogger(__name__).exception("Engram SSD layer stage %d failed", stage + 1)
                    status = STATUS_FAIL
            if inline:
                status, slots = self._scatter_layer_result(
                    status, packed, starts, ends, local_width, inline)
                if status == _STATUS_FALLBACK:
                    # All ranks retry the variable-size request before acquiring slots.
                    return _STATUS_FALLBACK
                if status == _STATUS_INVALID_REQUEST:
                    raise SSDProtocolError(f"round {self.round}: invalid packed layer request")
                if status == _STATUS_STAGING_FAILED:
                    self.stats.rounds_failed += 1
                    raise SSDProtocolError(f"round {self.round}: a rank failed to stage input rows")
            else:
                self._last_phase = PHASE_P2
                message = self._cpu_buffer("status", 1, np.int64)
                message.numpy()[0] = status
                start = time.monotonic()
                dist.broadcast(message, src=self.leader_global, group=self.pg)
                self.stats.add_phase(PHASE_P2, (time.monotonic() - start) * 1e3)
                status = int(message.numpy()[0])
                if status == STATUS_OK:
                    slots = self._scatter_layer(packed, starts, ends)
            if status == STATUS_FAIL:
                failure = SSDProtocolError(f"round {self.round}: layer stage {stage + 1} failed")
                self.stats.rounds_failed += 1
                for result in ready[stage:]:
                    result.set_exception(failure)
                return failure
            try:
                result = slots if submit_layer is None else submit_layer(stage, slots)
            except Exception as exc:
                # Keep joining collectives when a local launch fails. Retirement
                # checks any partially submitted UVA reads before slot reuse.
                ready[stage].set_exception(exc)
            else:
                ready[stage].set_result(result)
            starts, ends = first, counts
        return None

    def _finish_round(self, n: int, num_tokens: int, t0: float, *, lookup_ms: float | None = None) -> None:
        self.stats.steps += 1
        self.stats.rows_total += n
        cache = self.last_cache_stats
        if cache is not None and (self._group_max_tokens >= _CACHE_LOG_MIN_TOKENS or self.stats.steps <= 3
                                  or self.stats.steps % _CACHE_LOG_INTERVAL == 0):
            hit_rate = cache['hit_rate']
            logging.getLogger(__name__).info(
                "engram-ssd-cache scope=owner_group round=%d rows=%d hits=%d "
                "miss_queries=%d unique_misses=%d hit_rate=%s evictions=%d "
                "cache_ms=%.3f fill_ms=%.3f protocol_ms=%.3f",
                self.round, cache['rows'], cache['hit_rows'], cache['miss_rows'],
                cache['unique_misses'], 'n/a' if hit_rate is None else f'{hit_rate:.6f}',
                cache.get('evictions', 0), cache.get('wall_ms', 0),
                cache['fill_wall_ms'], lookup_ms if lookup_ms is not None else (time.monotonic() - t0) * 1e3)
            if lookup_ms is not None:
                logging.getLogger(__name__).info(
                    "engram-ssd-layered round=%d first_fill_ms=%.3f second_fill_ms=%.3f "
                    "ready_ms=%.3f retire_ms=%.3f",
                    self.round, *cache["layer_fill_ms"], lookup_ms,
                    (time.monotonic() - t0) * 1e3 - lookup_ms)
        self._log("step-done", tokens=num_tokens, rows=n, ms=(time.monotonic() - t0) * 1e3)
        self.round += 1

    def _step_merged(self, epoch: int, rows: np.ndarray, slots: np.ndarray,
                     num_tokens: int) -> None:
        """Merge status/slots for small batches; scatter large prefill batches."""
        headers, pad = self._p0(epoch, len(rows), num_tokens)
        gathered = self._p1(rows, pad)
        if (1 + self.size * pad) * 8 > _MERGED_SLOT_MAX_BYTES:
            status = self._p2(headers, gathered, pad)
            if status == STATUS_FAIL:
                self.stats.rounds_failed += 1
                self._p4(STATUS_FAIL)
                raise SSDProtocolError(f"round {self.round}: leader reported failure")
            self._p3(gathered, headers, slots, pad)
            self._p4(STATUS_OK)
            return

        self._last_phase = PHASE_P2
        bmsg = self._cpu_buffer('merged_slots', 1 + self.size * pad, np.int32)
        if self.rank == 0:
            bmsg.numpy().fill(-1)
            bmsg.numpy()[0] = STATUS_OK
            if self.hooks is not None:
                try:
                    counts = [int(t[2]) for t in headers]
                    self._leader_acquire(gathered, counts, pad, output=bmsg[1:])
                    self._leader_fill()
                    self.hooks.publish(self._leader_lease)
                except Exception as exc:
                    self._log("leader-round-failed", err=str(exc)[:120])
                    bmsg.numpy()[0] = STATUS_FAIL
                    self._discard_leader_lease()
        t0 = time.monotonic()
        dist.broadcast(bmsg, src=self.leader_global, group=self.pg)
        self.stats.add_phase(PHASE_P2, (time.monotonic() - t0) * 1e3)
        if int(bmsg[0]) == STATUS_FAIL:
            self.stats.rounds_failed += 1
            self._p4(STATUS_FAIL)
            raise SSDProtocolError(f"round {self.round}: leader reported failure")
        n = len(slots)
        if n:
            got = bmsg.numpy()[1 + self.rank * pad:1 + self.rank * pad + n]
            if (got < 0).any():
                raise SSDProtocolError("P2' returned unfilled slots")
            slots[:] = got
        self._p4(STATUS_OK)

    # ---------- phases ----------

    def _p0(self, epoch: int, n: int, num_tokens: int, *, first_count: int = 0):
        self._last_phase = PHASE_P0
        header = self._cpu_buffer('header', _HEADER_LEN, np.int64)
        header.numpy()[:] = (epoch, self.round, n, first_count, num_tokens)
        out = list(self._cpu_buffer('headers', self.size * _HEADER_LEN,
                                    np.int64).view(self.size, _HEADER_LEN).unbind())
        t0 = time.monotonic()
        dist.all_gather(out, header, group=self.pg)
        self.stats.add_phase(PHASE_P0, (time.monotonic() - t0) * 1e3)
        for t in out:
            if int(t[0]) != epoch or int(t[1]) != self.round:
                raise SSDProtocolError(
                    f"P0 order mismatch: headers={[t[0:2].tolist() for t in out]}")
        pad = max(max(int(t[2]) for t in out), 1)
        self._group_max_tokens = max(int(t[4]) for t in out)
        return out, pad

    def _p1(self, rows: np.ndarray, pad: int) -> torch.Tensor | None:
        self._last_phase = PHASE_P1
        payload = self._cpu_buffer('rows_send', pad, np.int64)
        n = len(rows)
        payload_np = payload.numpy().view(np.uint64)
        payload_np[:n] = rows
        payload_np[n:].fill(0)
        gathered = None
        buf = None
        if self.rank == 0:
            gathered = self._cpu_buffer('rows_receive', self.size * pad,
                                        np.int64).view(self.size, pad)
            # Each receive view is contiguous; Gloo overwrites it completely.
            buf = list(gathered.unbind())
        t0 = time.monotonic()
        dist.gather(payload, buf, dst=self.leader_global, group=self.pg)
        self.stats.add_phase(PHASE_P1, (time.monotonic() - t0) * 1e3)
        if self.rank == 0:
            self.stats.bytes_gathered += pad * 8 * self.size
        return gathered

    def _p2(self, headers, gathered, pad: int) -> int:
        self._last_phase = PHASE_P2
        status = self._cpu_buffer('status', 1, np.int64)
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
                    self._discard_leader_lease()
            status[0] = st
        dist.broadcast(status, src=self.leader_global, group=self.pg)
        return int(status[0])

    def _p3(self, gathered, headers, slots: np.ndarray, pad: int) -> None:
        self._last_phase = PHASE_P3
        chunk = self._cpu_buffer('slots_receive', pad, np.int32)
        chunks = None
        if self.rank == 0:
            flat = self._leader_slots
            if self.hooks is None:
                flat = self._cpu_buffer('slots_send', self.size * pad, np.int32)
                flat.numpy().fill(-1)
            # _leader_acquire already packed slots in rank-major order.
            chunks = list(flat.view(self.size, pad).unbind())
        t0 = time.monotonic()
        dist.scatter(chunk, chunks, src=self.leader_global, group=self.pg)
        self.stats.add_phase(PHASE_P3, (time.monotonic() - t0) * 1e3)
        n = len(slots)
        if n:
            got = chunk.numpy()[:n]
            if (got < 0).any():
                raise SSDProtocolError("P3 returned unfilled slots for valid rows")
            slots[:] = got

    def _scatter_layer(self, packed, starts: list[int], ends: list[int]) -> np.ndarray:
        self._last_phase = PHASE_P3
        widths = [end - start for start, end in zip(starts, ends)]
        pad = max(1, *widths)
        chunk = self._cpu_buffer("slots_receive", pad, np.int32)
        chunks = None
        if self.rank == 0:
            send = self._cpu_buffer("layer_slots", self.size * pad, np.int32).view(self.size, pad)
            send_np = send.numpy()
            for index, (start, end, width) in enumerate(zip(starts, ends, widths)):
                send_np[index, :width] = packed[index, start:end]
                send_np[index, width:].fill(-1)
            chunks = list(send.unbind())
        t0 = time.monotonic()
        dist.scatter(chunk, chunks, src=self.leader_global, group=self.pg)
        self.stats.add_phase(PHASE_P3, (time.monotonic() - t0) * 1e3)
        slots = chunk.numpy()[:widths[self.rank]].copy()
        if (slots < 0).any():
            raise SSDProtocolError("P3 returned unfilled layer slots")
        return slots

    def _p4(self, status: int, *, layered: bool = False) -> None:
        self._last_phase = PHASE_P4
        ack = self._cpu_buffer('ack', 2, np.int64)
        ack.numpy()[:] = (self.round, status)
        acks = list(self._cpu_buffer('acks', self.size * 2, np.int64).view(self.size, 2).unbind())
        t0 = time.monotonic()
        dist.all_gather(acks, ack, group=self.pg)
        self.stats.add_phase(PHASE_P4, (time.monotonic() - t0) * 1e3)
        for a in acks:
            if int(a[0]) != self.round:
                raise SSDProtocolError(
                    f"P4 ack mismatch: {[[int(x) for x in a.tolist()] for a in acks]}")
        if layered and any(int(a[1]) != status or status == STATUS_UNSAFE for a in acks):
            raise SSDProtocolError("cannot recycle layered lease: unsafe or inconsistent retirement")
        # 全组消费确认后, leader 释放租约(payload 保留供后续命中)
        if self.rank == 0 and status == STATUS_OK and self.hooks is not None:
            lease = getattr(self, "_leader_lease", None)
            if lease is not None:
                self.hooks.release(lease)
                self._leader_lease = None

    # ---------- leader-side transaction ----------

    def _discard_leader_lease(self) -> None:
        """Undo a failed fill before broadcasting failure to the group."""
        lease = getattr(self, "_leader_lease", None)
        if lease is not None:
            if lease.published:
                self.hooks.release(lease)
            else:
                self.hooks.abort(lease)
            self._leader_lease = None

    def _leader_acquire(self, gathered: torch.Tensor, counts: list[int],
                        pad: int, *, output: torch.Tensor | None = None) -> torch.Tensor:
        """Acquire valid rank-major rows and pack slots directly for delivery."""
        assert self.hooks is not None
        matrix = gathered.numpy().view(np.uint64)
        if all(c == pad for c in counts):
            rows = matrix.reshape(-1)
        else:
            parts = [matrix[r, :c] for r, c in enumerate(counts) if c]
            rows = np.concatenate(parts) if parts else np.empty(0, np.uint64)
        lease = self.hooks.acquire(rows)
        self._leader_lease = lease
        cache = dict(lease.stats)
        cache.update(rows=len(rows), unique_misses=len(lease.miss_rows),
                     hit_rate=cache['hit_rows'] / len(rows) if len(rows) else None,
                     fill_wall_ms=0.0)
        self.last_cache_stats = cache
        self.stats.hit_rows_total += cache['hit_rows']
        self.stats.miss_queries_total += cache['miss_rows']
        self.stats.miss_rows_total += len(lease.miss_rows)
        flat = (output if output is not None else
                self._cpu_buffer('slots_send', self.size * pad, np.int32))
        packed = flat.numpy().reshape(self.size, pad)
        base = 0
        for r, c in enumerate(counts):
            packed[r, :c] = lease.slots[base:base + c]
            packed[r, c:].fill(-1)
            base += c
        self._log("leader-acquire", rows=len(rows), misses=len(lease.miss_rows))
        return flat

    def _leader_fill(self, selection=None, *, stage: int | None = None) -> None:
        assert self.hooks is not None
        lease = self._leader_lease
        if lease is None:
            return
        rows, slots = lease.miss_rows, lease.miss_slots
        if selection is not None:
            rows, slots = rows[selection], slots[selection]
        if len(rows) == 0:
            return
        t0 = time.monotonic()
        try:
            self.hooks.fill(rows, slots)
        finally:
            elapsed = (time.monotonic() - t0) * 1e3
            self.last_cache_stats["fill_wall_ms"] += elapsed
            if stage is not None:
                self.last_cache_stats["layer_fill_ms"][stage] = elapsed

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

    def abort(self, lease) -> None:
        self.cache.abort(lease)
