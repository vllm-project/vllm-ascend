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
"""Engram SSD DP participation probe (phase-3 T0, dev/diagnostic only).

Answers one architectural question before the SSD DP protocol is built:
does every rank of the node-local engram DP group reach the engram prepare
point once per engine step, in the same order, with bounded completion —
on the real scheduler, including idle ranks, mixed prefill/decode, bucket
switches and graph capture?

Mechanics: an env-gated probe hooks the exact places where the SSD runtime
will later run its per-step protocol (``EngramModelState.prepare_engram``
entry, which fires once per executed step on both the eager pre-hook and
the FULL-replay pre-hook) and executes a tiny fixed-length CPU collective
sequence that mirrors the planned P0..P4 shape (header all_gather, padded
gather to leader, status broadcast, scatter, ack all_gather) over the
engram DP group's gloo CPU group. Every step is validated for identical
round numbers across ranks; a mismatch fails fast; a step that cannot
finish inside the watchdog bound terminates the process after flushing a
TIMEOUT record, so absent/disordered ranks can never hang the engine.

Capture/profile steps are logged but never join the protocol.

Enabled with VLLM_ASCEND_ENGRAM_SSD_DP_PROBE=1. Never enable in
production: the probe adds per-step CPU collectives on purpose.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

_PHASES = ("p0_header", "p1_gather", "p2_status", "p3_scatter", "p4_ack")

# Which step categories join the protocol. "auto" matches the SSD design:
# real engine steps and engine-idle dummy waves join; capture/profile steps
# are recorded only (the SSD runtime does not exist during startup).
# "auto_gated" additionally requires a one-time group barrier before the
# first join (mirrors the SSD runtime's all-group-ready startup barrier;
# without it a replica that starts its idle waves earlier than its peers
# deadlocks the per-step protocol during startup — observed on real
# scheduling 2026-11-09, runs/20261109T2200Z-t0-dp-probe-tp4dp2).
_JOIN_POLICIES = {
    "auto": {"real", "idle"},
    "auto_gated": {"real", "idle"},
    "real_only": {"real"},
    "none": set(),
}

# Header layout for P0 (int64): [round, phase_idx, valid_count, status, tag]
_HEADER_LEN = 5


def _parse_faults(spec: str) -> dict[str, Any]:
    faults: dict[str, Any] = {"skip_step": None, "skip_phase": None,
                              "delay_phase": None, "extra_step": None}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        kv = dict(p.split("=", 1) for p in part.split(":") if "=" in p)
        kind = part.split(":", 1)[0].split("=", 1)[0]
        # 主值既可写 kind=<v>(如 skip_step=10), 也可写 phase=<v>/<step=<v>。
        primary = kv.get(kind, kv.get("phase", kv.get("step", "")))
        if kind == "skip_step":
            faults["skip_step"] = (int(primary), int(kv["rank"]))
        elif kind == "skip_phase":
            faults["skip_phase"] = (primary, int(kv["rank"]))
        elif kind == "delay_phase":
            faults["delay_phase"] = (primary, int(kv["rank"]), float(kv["ms"]))
        elif kind == "extra_step":
            faults["extra_step"] = int(kv.get("rank", -1))
    return faults


@dataclass
class _ProbeConfig:
    enabled: bool = False
    log_dir: Path = field(default_factory=lambda: Path("/tmp/engram_ssd_dp_probe"))
    timeout_s: float = 60.0
    join_policy: str = "auto"
    gate_timeout_s: float = 1800.0
    faults: dict[str, Any] = field(default_factory=lambda: _parse_faults(""))


class EngramDPProbe:
    """Per-process probe state. One instance per model state."""

    def __init__(self):
        # Lazy import keeps this module importable standalone (CPU tests);
        # values stay centralized in vllm_ascend.envs per AGENTS.md.
        from vllm_ascend import envs

        self.cfg = _ProbeConfig(
            enabled=bool(int(envs.VLLM_ASCEND_ENGRAM_SSD_DP_PROBE)),
            log_dir=Path(envs.VLLM_ASCEND_ENGRAM_SSD_DP_PROBE_DIR),
            timeout_s=float(envs.VLLM_ASCEND_ENGRAM_SSD_DP_PROBE_TIMEOUT_S),
            join_policy=envs.VLLM_ASCEND_ENGRAM_SSD_DP_PROBE_JOIN_POLICY,
            gate_timeout_s=float(envs.VLLM_ASCEND_ENGRAM_SSD_DP_PROBE_GATE_TIMEOUT_S),
            faults=_parse_faults(envs.VLLM_ASCEND_ENGRAM_SSD_DP_PROBE_FAULT),
        )
        self.group = None            # GroupCoordinator of the engram DP group
        self.pg = None               # underlying gloo ProcessGroup
        self.group_size = 1
        self.group_rank = 0
        self.group_ranks: list[int] = []
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        self.round = 0
        self._skip_consumed = False
        self._gate_done = False
        self._log_file = None
        self._watchdog: threading.Thread | None = None
        self._deadline = float("inf")
        self._records = 0
        self._inited = False
        self._disabled_reason: str | None = None

    # ---------- logging ----------

    def _log(self, evt: str, **kw) -> None:
        rec = {"ts": time.time(), "pid": os.getpid(), "rank": self.global_rank,
               "evt": evt, "round": self.round, **kw}
        if self._log_file is not None:
            self._log_file.write(json.dumps(rec, default=str) + "\n")
            self._log_file.flush()
        self._records += 1

    # ---------- lifecycle ----------

    def bind(self, pg, size: int, rank: int, ranks: list[int]) -> None:
        """Test-only direct binding (bypasses the process-group lookup)."""
        self._inited = True
        self.pg = pg
        self.group_size = size
        self.group_rank = rank
        self.group_ranks = list(ranks)
        self.global_rank = ranks[rank] if ranks and rank < len(ranks) else rank
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(
            self.cfg.log_dir / f"probe_rank{self.global_rank}.jsonl", "a", encoding="utf-8")
        self._start_watchdog()
        self._log("init", group_ranks=self.group_ranks, group_size=self.group_size,
                  timeout_s=self.cfg.timeout_s, faults=self.cfg.faults, bound=True)

    def try_init(self) -> bool:
        """Bind to the engram DP group's CPU (gloo) group once."""
        if not self.cfg.enabled or self._inited:
            return self._inited
        self._inited = True
        try:
            from vllm.distributed.parallel_state import get_engram_dp_group
            group = get_engram_dp_group()
        except Exception as exc:  # pragma: no cover - defensive
            self._disabled_reason = f"get_engram_dp_group failed: {exc}"
            self._log("disabled", reason=self._disabled_reason)
            return False
        if group is None:
            self._disabled_reason = "engram dp group not created"
            self._log("disabled", reason=self._disabled_reason)
            return False
        self.group = group
        self.group_size = group.world_size
        self.group_rank = group.rank_in_group
        try:
            self.group_ranks = list(group.ranks)
        except Exception:
            self.group_ranks = []
        if self.group_size < 2:
            self._disabled_reason = f"engram dp group size {self.group_size} < 2"
            self._log("disabled", reason=self._disabled_reason)
            return False
        self.pg = self.group.cpu_group
        if self.pg is None:
            self._disabled_reason = "cpu_group is None"
            self._log("disabled", reason=self._disabled_reason)
            return False
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(
            self.cfg.log_dir / f"probe_rank{self.global_rank}.jsonl", "a", encoding="utf-8")
        self._start_watchdog()
        self._log("init", group_ranks=self.group_ranks, group_size=self.group_size,
                  timeout_s=self.cfg.timeout_s, faults=self.cfg.faults)
        return True

    def _start_watchdog(self) -> None:
        if self._watchdog is not None:
            return

        def _run():
            while True:
                time.sleep(1.0)
                deadline = self._deadline
                if deadline != float("inf") and time.monotonic() > deadline:
                    self._log("timeout", deadline_s=self.cfg.timeout_s,
                              phase=self._last_phase, round=self.round)
                    if self._log_file is not None:
                        self._log_file.flush()
                    # A wedged gloo collective cannot be unwound from Python;
                    # the only honest bound is terminating this process.
                    os._exit(86)

        self._watchdog = threading.Thread(target=_run, daemon=True, name="engram-dp-probe-wd")
        self._watchdog.start()

    _last_phase: str = ""

    # ---------- protocol ----------

    def _fault_rank_matches(self, fault: tuple | int | None) -> bool:
        if fault is None:
            return False
        if isinstance(fault, bool):
            return fault
        rank = fault[-1] if isinstance(fault, tuple) else fault
        try:
            return rank == self.group_rank
        except Exception:
            return False

    def step(self, *, category: str, num_tokens: int = 0, num_reqs: int = 0,
             cg_mode: str = "", bucket: int = 0, extra: dict | None = None) -> None:
        """One probe protocol step at the engram prepare point.

        ``category`` is one of: real / idle / capture / profile / stage-dummy.
        Only ``real`` and ``idle`` join the collectives; others are logged.
        """
        if not self.cfg.enabled:
            return
        if not self._inited:
            self.try_init()
        if self.pg is None:
            # Group unavailable (e.g. DP1): still log the step for the matrix.
            if self._disabled_reason is not None and self._records < 2:
                self._log("step-no-group", category=category, num_tokens=num_tokens)
            return

        rec = {"category": category, "num_tokens": num_tokens, "num_reqs": num_reqs,
               "cg_mode": str(cg_mode), "bucket": bucket,
               "group_rank": self.group_rank, "group_size": self.group_size}
        if extra:
            rec.update(extra)

        join = _JOIN_POLICIES.get(self.cfg.join_policy, _JOIN_POLICIES["auto"])
        if category not in join:
            self._log("step-nonpart", **rec)
            return

        # Gated policies wait for all group peers before the first join.
        # Startup windows (graph capture / one-sided idle waves / staggered
        # warmup requests) are not lockstepped; one-time gating is not
        # enough (observed: 5 paired rounds then one-sided join). P0 instead
        # uses a bounded-wait barrier with retry: transient divergence heals
        # itself; a peer absent beyond the bound is fatal (bounded exit).
        if self.cfg.join_policy == "auto_gated" and not self._gate_done:
            self._gate_done = True
            self._log("gate-wait", **rec)
            deadline = time.monotonic() + self.cfg.gate_timeout_s
            while True:
                try:
                    dist.monitored_barrier(group=self.pg, timeout=timedelta(seconds=30),
                                           wait_all_ranks=True)
                    break
                except Exception as exc:
                    if time.monotonic() > deadline:
                        self._log("gate-timeout", err=str(exc)[:120], **rec)
                        raise RuntimeError(
                            "engram dp probe: startup gate barrier timed out; "
                            "group never became ready together") from exc
                    time.sleep(1.0)
            self._log("gate-open", **rec)

        f = self.cfg.faults
        # extra_step: this rank runs an extra lone round to force divergence.
        if f["extra_step"] is not None and f["extra_step"] == self.group_rank:
            self.round += 1
            self._log("fault-extra-step", **rec)

        # skip_step: rank goes absent at its N-th protocol round (consumed
        # once). Its round counter still advances, so the next join is one
        # round behind the group and P0 fails fast instead of hanging.
        skip_at = f["skip_step"]
        if (skip_at is not None and skip_at[1] == self.group_rank
                and self.round == skip_at[0] and not self._skip_consumed):
            self._skip_consumed = True
            self.round += 1
            self._log("fault-skip-step", **rec)
            return

        self._last_phase = "entry"
        self._deadline = time.monotonic() + self.cfg.timeout_s
        self._p0_deadline = time.monotonic() + self.cfg.timeout_s
        try:
            self._run_protocol(rec)
        finally:
            self._deadline = float("inf")
        self._log("step-done", **rec)
        self.round += 1

    def _maybe_delay(self, phase: str) -> None:
        d = self.cfg.faults["delay_phase"]
        if d is not None and d[0] == phase and d[1] == self.group_rank:
            self._log("fault-delay-phase", phase=phase, ms=d[2])
            time.sleep(d[2] / 1000.0)

    def _maybe_skip(self, phase: str) -> bool:
        s = self.cfg.faults["skip_phase"]
        if s is not None and s[0] == phase and s[1] == self.group_rank:
            self._log("fault-skip-phase", phase=phase)
            return True
        return False

    def _run_protocol(self, rec: dict) -> None:
        g = self.pg
        size, rank = self.group_size, self.group_rank
        # 注意: split_group 派生的 cpu_group 上, src/dst 必须用全局 rank
        # (与 engram/npu.py SharedUvaBuffer 的用法一致)。
        leader_global = dist.get_global_rank(g, 0)
        header = torch.tensor([self.round, 0, int(rec["num_tokens"]), 0, 0x5A5A],
                              dtype=torch.int64)

        # P0: bounded-wait alignment then fixed-length header all_gather.
        # The barrier retry heals transient divergence (startup windows,
        # staggered warmup requests); the watchdog bounds the whole step.
        self._last_phase = _PHASES[0]
        self._maybe_delay(_PHASES[0])
        out = [header.clone()]
        if not self._maybe_skip(_PHASES[0]):
            attempt = 0
            while True:
                try:
                    dist.monitored_barrier(group=g, timeout=timedelta(seconds=5),
                                           wait_all_ranks=False)
                    break
                except Exception as exc:
                    attempt += 1
                    self._log("p0-peer-wait", attempt=attempt, err=str(exc)[:80], **rec)
                    if time.monotonic() > self._p0_deadline:
                        raise RuntimeError(
                            "engram dp probe: P0 peer did not arrive within bound") from exc
                    time.sleep(1.0)
            out = [torch.zeros(_HEADER_LEN, dtype=torch.int64) for _ in range(size)]
            dist.all_gather(out, header, group=g)
            rounds = {int(t[0]) for t in out}
            if len(rounds) != 1 or rounds != {self.round}:
                detail = [int(t[0]) for t in out]
                self._log("round-mismatch", phase=_PHASES[0], headers=detail, **rec)
                raise RuntimeError(
                    f"engram dp probe: rank {self.global_rank} round {self.round} "
                    f"mismatch across group {self.group_ranks}: {detail}")

        # P1: pad to the group max count and gather uint64 rows to the leader.
        self._last_phase = _PHASES[1]
        max_count = max(int(t[2]) for t in out)
        pad = max(max_count, 1)
        rows = torch.arange(rank * pad, rank * pad + min(int(rec["num_tokens"]), pad),
                            dtype=torch.int64)
        payload = torch.zeros(pad, dtype=torch.int64)
        payload[: rows.numel()] = rows
        self._maybe_delay(_PHASES[1])
        if not self._maybe_skip(_PHASES[1]):
            if rank == 0:
                buf = [torch.zeros(pad, dtype=torch.int64) for _ in range(size)]
                dist.gather(payload, buf, dst=leader_global, group=g)
            else:
                dist.gather(payload, None, dst=leader_global, group=g)

        # P2: fixed-length status broadcast from the leader.
        self._last_phase = _PHASES[2]
        status = torch.tensor([0 if rank != 0 else 1], dtype=torch.int64)
        self._maybe_delay(_PHASES[2])
        if not self._maybe_skip(_PHASES[2]):
            dist.broadcast(status, src=leader_global, group=g)
            if int(status[0]) != 1:
                self._log("status-not-ok", phase=_PHASES[2], status=int(status[0]), **rec)
                raise RuntimeError("engram dp probe: leader reported failure")

        # P3: scatter int32 slots back (padded equal chunks).
        self._last_phase = _PHASES[3]
        self._maybe_delay(_PHASES[3])
        if not self._maybe_skip(_PHASES[3]):
            chunk = torch.zeros(pad, dtype=torch.int32)
            if rank == 0:
                chunks = [torch.full((pad,), i + 1, dtype=torch.int32) for i in range(size)]
                dist.scatter(chunk, chunks, src=leader_global, group=g)
            else:
                dist.scatter(chunk, None, src=leader_global, group=g)
            if int(chunk[0]) != rank + 1:
                self._log("scatter-mismatch", got=int(chunk[0]), **rec)
                raise RuntimeError("engram dp probe: scatter order mismatch")

        # P4: completion ack all_gather.
        self._last_phase = _PHASES[4]
        self._maybe_delay(_PHASES[4])
        if not self._maybe_skip(_PHASES[4]):
            ack = torch.tensor([self.round, 1], dtype=torch.int64)
            acks = [torch.zeros(2, dtype=torch.int64) for _ in range(size)]
            dist.all_gather(acks, ack, group=g)
            if any(int(a[0]) != self.round or int(a[1]) != 1 for a in acks):
                self._log("ack-mismatch", acks=[[int(a[0]), int(a[1])] for a in acks], **rec)
                raise RuntimeError("engram dp probe: completion ack mismatch")

    def close(self) -> None:
        self._log("close", records=self._records)
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None
