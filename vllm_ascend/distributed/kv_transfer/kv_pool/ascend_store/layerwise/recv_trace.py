# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded host timeline and conservative, sampled receive-queue HOL evidence.

The observer never waits on device events. Readiness callbacks must only query
already-recorded events and must return None for dependencies they cannot prove.
Unknown readiness is never counted as ready. Sampling provides readiness observations, not exact device completion
timestamps or proof of uninterrupted readiness between samples.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

Readiness = Mapping[str, bool | None]
TRACE_FLUSH_INTERVAL_S = 5.0


def trace_session_path(prefix: str, tp_rank: int) -> str:
    """Keep worker snapshots distinct even with shared paths and PID namespaces."""
    return f"{prefix}.tp{tp_rank}.pid{os.getpid()}.{uuid4().hex}.json"


@dataclass
class PendingTask:
    task_id: int
    probe: Callable[[], Readiness]
    metadata: dict[str, Any]
    published: bool = False


class RecvTimeline:
    """One bounded trace per receive thread; independent of FIFO scheduling."""

    def __init__(self, capacity: int = 100000, clock: Callable[[], int] = time.monotonic_ns):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("layerwise_trace_capacity must be a positive integer")
        self.capacity = capacity
        self.clock_anchor = {"monotonic_ns": clock(), "unix_ns": time.time_ns()}
        self.clock = clock
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._pending: dict[int, PendingTask] = {}
        self._next_id = 0
        self._active: int | None = None
        self._epoch = 0
        self.dropped = 0

    def _record(self, task_id: int, event: str, **fields: Any) -> None:
        if len(self._events) >= self.capacity:
            self.dropped += 1
            return
        self._events.append(dict(ts_ns=self.clock(), task_id=task_id, event=event, **fields))

    def enqueue(self, probe: Callable[[], Readiness], **metadata: Any) -> int | None:
        with self._lock:
            # Bound retained callbacks and request metadata as well as events.
            if len(self._events) >= self.capacity or len(self._pending) >= self.capacity:
                self.dropped += 1
                return None
            task_id = self._next_id
            self._next_id += 1
            self._pending[task_id] = PendingTask(task_id, probe, metadata)
            self._record(task_id, "enqueue", **metadata)
            return task_id

    def publish(self, task_id: int | None) -> None:
        with self._lock:
            if task_id is not None and task_id in self._pending:
                self._pending[task_id].published = True

    def mark(self, task_id: int | None, event: str, **fields: Any) -> None:
        if task_id is None:
            return
        with self._lock:
            self._record(task_id, event, **fields)

    def boundary(self, task_id: int | None, event: str) -> None:
        with self._lock:
            self._epoch += 1
            if task_id is not None:
                self._record(task_id, event)

    def dequeue(self, task_id: int | None) -> None:
        with self._lock:
            self._active = task_id
            self._epoch += 1
            if task_id is not None:
                self._record(task_id, "dequeue")

    def finish(self, task_id: int | None, outcome: str = "complete") -> None:
        with self._lock:
            if task_id is not None:
                self._record(task_id, outcome)
                self._pending.pop(task_id, None)
            self._active = None
            self._epoch += 1

    def sample(self) -> None:
        """Record all dependency observations, not just a derived HOL boolean.

        Recheck the head after probing followers: it must still be blocked at
        the end of the observation. Epoch validation discards samples crossing
        a consumer dispatch boundary. Callbacks run outside the trace lock.
        """
        with self._lock:
            head_id, epoch = self._active, self._epoch
            if head_id is None or head_id not in self._pending or len(self._events) >= self.capacity:
                return
            tasks = [task for task in self._pending.values() if task.published or task.task_id == head_id]
        head = next(task for task in tasks if task.task_id == head_id)
        observations = []
        for task in tasks:
            if task.task_id < head_id:
                continue
            try:
                dependencies = dict(task.probe())
            except Exception as exc:
                dependencies = {"probe_error": None}
                self.mark(task.task_id, "probe_error", error=type(exc).__name__)
            observations.append((task.task_id, dependencies))
        try:
            head_dependencies = dict(head.probe())
        except Exception:
            head_dependencies = {"probe_error": None}
        blocked = any(
            value is False
            for name, value in head_dependencies.items()
            if name not in ("has_payload", "transfer_enabled")
        )
        ready = [
            task_id
            for task_id, deps in observations
            if task_id > head_id and deps and all(value is True for value in deps.values())
        ]
        with self._lock:
            if epoch != self._epoch or head_id != self._active:
                return
            self._record(
                head_id,
                "readiness_sample",
                dependencies=observations,
                head_recheck=head_dependencies,
                ready_followers=ready if blocked else [],
                pending_tasks=len(tasks),
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "clock": "monotonic_ns",
                "clock_anchor": self.clock_anchor,
                "capacity": self.capacity,
                "dropped": self.dropped,
                "events": list(self._events),
            }

    def dump(self, path: str | Path) -> None:
        """Call after quiescing the observer; never called on the disabled path."""
        Path(path).write_text(json.dumps(self.snapshot(), ensure_ascii=False), encoding="utf-8")


class RecvTraceSession:
    """Optional receive-thread observer. Never changes queue selection."""

    def __init__(
        self,
        worker: Any,
        path: str,
        capacity: int,
        sample_interval_ms: float,
        metrics_probe: Callable[[], Mapping[str, int | None]] | None = None,
    ):
        if not 1 <= sample_interval_ms <= 1000:
            raise ValueError("layerwise_trace_interval_ms must be between 1 and 1000")
        self.worker = worker
        self.metrics_probe = metrics_probe
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.timeline = RecvTimeline(capacity)
        self.interval = sample_interval_ms / 1000
        self._stop = threading.Event()
        self._active_id: int | None = None
        self._waiting = False
        self._satisfied: dict[int, set[str]] = {}
        self._observer = threading.Thread(target=self._observe, name="KVRecvObserver", daemon=True)
        self._observer.start()

    def enqueue(self, task: Any) -> None:
        dependencies: set[str] = set()
        payload_groups = set()
        for item in task.transfer_tasks:
            shared = item.shared_block_data
            if shared is not None:
                has_payload = len(shared.block_ids_arr) > 0
            else:
                has_payload = any(
                    min(br.end_block, len(br.request.block_hashes)) > br.start_block for br in item.block_ranges
                )
            if has_payload:
                payload_groups.add(item.group_id)

        def probe() -> Readiness:
            result: dict[str, bool | None] = {}
            owner = task.wait_for_save_layer
            if owner is not None:
                cpu_ready = "save_cpu" in dependencies or self.worker.layer_save_finished_events[owner].is_set()
                result["save_cpu"] = cpu_ready
                events = getattr(self.worker, "sync_save_events", None)
                if events is not None:
                    result["save_npu"] = "save_npu" in dependencies or (cpu_ready and bool(events[owner].query()))
            gate = task.attention_start_gate
            if gate is not None:
                recorded, complete = gate.readiness()
                result.update(gate_record=recorded, gate_npu=complete)
            # A blocking external callback has no readiness contract. Never
            # infer its state, invoke it early, or claim such a follower ready.
            if getattr(self.worker, "external_slot_release_waiter", None) is not None:
                result["external_slot"] = True if "external_slot" in dependencies else None
            abort = getattr(self.worker, "_load_abort_event", None)
            enabled = not self.worker.load_disabled.is_set() and not (abort is not None and abort.is_set())
            result["transfer_enabled"] = enabled
            active_groups = getattr(self.worker, "_group_load_indices", {})
            has_payload = any(active_groups.get(group) is None or active_groups[group] for group in payload_groups)
            result["has_payload"] = has_payload
            return result

        task_id = self.timeline.enqueue(
            probe,
            layer=task.layer_id,
            groups=sorted({item.group_id for item in task.transfer_tasks}),
            requests=sorted({br.request.req_id for item in task.transfer_tasks for br in item.block_ranges}),
            save_owner=task.wait_for_save_layer,
            path="block_key" if any(t.use_key_major_ranges for t in task.transfer_tasks) else self.worker.name,
        )
        task.recv_trace_id = task_id
        if task_id is not None:
            self._satisfied[task_id] = dependencies

    def publish(self, task: Any) -> None:
        self.timeline.publish(task.recv_trace_id)

    def dequeue(self, task: Any) -> None:
        self._active_id = task.recv_trace_id
        self._waiting = False
        self.timeline.dequeue(self._active_id)

    def mark(self, event: str) -> None:
        self._waiting = event in {
            "save_cpu_wait_begin",
            "save_npu_wait_begin",
            "gate_record_wait_begin",
            "gate_npu_wait_begin",
            "external_slot_wait_begin",
        }
        self.timeline.boundary(self._active_id, event)
        dependency = {
            "save_cpu_wait_end": "save_cpu",
            "save_npu_wait_end": "save_npu",
            "external_slot_wait_end": "external_slot",
        }.get(event)
        if dependency is not None and self._active_id is not None and self._active_id in self._satisfied:
            self._satisfied[self._active_id].add(dependency)

    def finish(self, outcome: str = "complete") -> None:
        self._waiting = False
        self.timeline.finish(self._active_id, outcome)
        if self._active_id is not None:
            self._satisfied.pop(self._active_id, None)
        self._active_id = None

    def _observe(self) -> None:
        try:
            self.worker.m_store.set_device()
            next_metrics = 0.0
            next_flush = time.monotonic() + TRACE_FLUSH_INTERVAL_S
            while not self._stop.wait(self.interval):
                if self._waiting:
                    self.timeline.sample()
                if self.metrics_probe is not None and time.monotonic() >= next_metrics:
                    try:
                        fields = dict(self.metrics_probe())
                        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        fields["host_peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
                        self.timeline.mark(-1, "resources", **fields)
                    except Exception as exc:
                        self.timeline.mark(-1, "metrics_error", error=type(exc).__name__)
                    next_metrics = time.monotonic() + 0.1
                if time.monotonic() >= next_flush:
                    self._flush()
                    next_flush = time.monotonic() + TRACE_FLUSH_INTERVAL_S
        except Exception as exc:
            self.timeline.mark(-1, "observer_error", error=type(exc).__name__)

    def close(self) -> None:
        self._stop.set()
        self._observer.join()
        self._flush()

    def _flush(self) -> None:
        # Engine subprocess termination does not always call connector shutdown.
        # Replace a bounded snapshot atomically so completed captures survive it.
        temporary = Path(self.path + ".tmp")
        self.timeline.dump(temporary)
        temporary.replace(self.path)
