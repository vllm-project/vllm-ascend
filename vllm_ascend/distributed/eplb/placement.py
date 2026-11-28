# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Committed EPLB placement snapshot schema, atomic writer, and PP coordinator."""

from __future__ import annotations

import hmac
import json
import os
import socket
import struct
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from vllm.logger import logger

PLACEMENT_SNAPSHOT_VERSION = 1
_REQUIRED_KEYS = frozenset({"version", "ep_size", "target"})
_MODEL_KEYS = frozenset({"num_logical_experts", "num_redundant_experts", "physical_to_logical_map"})
_HEADER_STRUCT = struct.Struct("!I32s")
_ACCEPT_POLL_SECONDS = 0.2
_SOCKET_TIMEOUT_SECONDS = 5.0
_MAX_REPORT_BYTES = 16 * 1024 * 1024


def _require_int(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_model_entry(entry: Any, role: str, ep_size: int) -> None:
    if not isinstance(entry, dict) or set(entry) != _MODEL_KEYS:
        raise ValueError(f"EPLB {role} placement must contain exactly {sorted(_MODEL_KEYS)}")
    num_logical = _require_int(entry["num_logical_experts"], f"{role}.num_logical_experts", 1)
    num_redundant = _require_int(entry["num_redundant_experts"], f"{role}.num_redundant_experts", 0)
    num_physical = num_logical + num_redundant
    if num_physical % ep_size != 0:
        raise ValueError(f"EPLB {role} physical expert count ({num_physical}) must be divisible by ep_size")
    mapping = entry["physical_to_logical_map"]
    if not isinstance(mapping, list) or not mapping or any(not isinstance(row, list) for row in mapping):
        raise ValueError(f"EPLB {role}.physical_to_logical_map must be a non-empty list of layer lists")

    slots_per_rank = num_physical // ep_size
    expected = set(range(num_logical))
    for layer_idx, row in enumerate(mapping):
        if len(row) != num_physical:
            raise ValueError(f"EPLB {role} layer {layer_idx} has an invalid physical slot count")
        if any(type(e) is not int or not 0 <= e < num_logical for e in row):
            raise ValueError(f"EPLB {role} layer {layer_idx} contains an invalid logical expert ID")
        if set(row) != expected:
            raise ValueError(f"EPLB {role} layer {layer_idx} does not cover every logical expert")
        for rank in range(ep_size):
            slot_slice = row[rank * slots_per_rank : (rank + 1) * slots_per_rank]
            if len(set(slot_slice)) != slots_per_rank:
                raise ValueError(f"EPLB {role} layer {layer_idx} contains duplicate logical experts on rank {rank}")


def validate_placement_snapshot(snapshot: Any) -> None:
    if not isinstance(snapshot, dict):
        raise ValueError("EPLB placement snapshot must be a JSON object")
    if type(snapshot.get("version")) is not int or snapshot["version"] != PLACEMENT_SNAPSHOT_VERSION:
        raise ValueError(f"Unsupported EPLB placement version: {snapshot.get('version')!r}")
    keys = set(snapshot)
    if _REQUIRED_KEYS - keys or keys - _REQUIRED_KEYS - {"draft"}:
        raise ValueError(f"EPLB placement snapshot must contain {sorted(_REQUIRED_KEYS)} and optional 'draft'")
    ep_size = _require_int(snapshot["ep_size"], "ep_size", 1)
    _validate_model_entry(snapshot["target"], "target", ep_size)
    if "draft" in snapshot:
        _validate_model_entry(snapshot["draft"], "draft", ep_size)


def write_placement_snapshot(path: str | os.PathLike[str], snapshot: dict[str, Any]) -> None:
    """Validate and atomically replace the destination file on the same filesystem."""
    validate_placement_snapshot(snapshot)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(snapshot, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise EOFError("PP placement socket closed unexpectedly")
        buf.extend(chunk)
    return bytes(buf)


def _send_authenticated_json(sock: socket.socket, authkey: bytes, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    sock.sendall(_HEADER_STRUCT.pack(len(body), hmac.digest(authkey, body, "sha256")) + body)


def _recv_authenticated_json(sock: socket.socket, authkey: bytes) -> dict[str, Any]:
    length, digest = _HEADER_STRUCT.unpack(_recv_exact(sock, _HEADER_STRUCT.size))
    if not 0 < length <= _MAX_REPORT_BYTES:
        raise ValueError(f"Invalid PP placement report length: {length}")
    body = _recv_exact(sock, length)
    if not hmac.compare_digest(digest, hmac.digest(authkey, body, "sha256")):
        raise ValueError("Invalid PP placement authentication digest")
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("PP placement report must be a JSON object")
    return data


def _build_stitched_snapshot(
    ep_size: int, stage_targets: list[dict[str, Any] | None], draft: dict[str, Any] | None
) -> dict[str, Any] | None:
    layers: list[list[int]] = []
    head: dict[str, Any] | None = None
    for stage in stage_targets:
        if stage is None:
            continue
        if head is None:
            head = stage
        elif (
            stage["num_logical_experts"] != head["num_logical_experts"]
            or stage["num_redundant_experts"] != head["num_redundant_experts"]
        ):
            raise ValueError("PP stages reported inconsistent MoE expert counts")
        layers.extend(stage["physical_to_logical_map"])
    if head is None or not layers:
        return None
    target = {
        "num_logical_experts": head["num_logical_experts"],
        "num_redundant_experts": head["num_redundant_experts"],
        "physical_to_logical_map": layers,
    }
    snapshot: dict[str, Any] = {"version": PLACEMENT_SNAPSHOT_VERSION, "ep_size": ep_size, "target": target}
    if draft is not None:
        snapshot["draft"] = draft
    return snapshot


@dataclass
class _CoordinatorState:
    pp_configured: bool = False
    active_target_stages: set[int] = field(default_factory=set)
    exported_targets: list[dict[str, Any] | None] = field(default_factory=list)
    stage_latest_target: dict[int, tuple[int, float, int, dict[str, Any]]] = field(default_factory=dict)
    stage_evaluated_seq: dict[int, int] = field(default_factory=dict)
    stage_pending_commit: dict[int, bool] = field(default_factory=dict)
    has_exported_target: bool = False
    min_target_ratio: float | None = None
    min_draft_ratio: float | None = None
    best_draft: dict[str, Any] | None = None
    draft_pending_commit: bool = False
    draft_needs_write: bool = False

    def clone(self) -> _CoordinatorState:
        return _CoordinatorState(
            pp_configured=self.pp_configured,
            active_target_stages=set(self.active_target_stages),
            exported_targets=list(self.exported_targets),
            stage_latest_target=dict(self.stage_latest_target),
            stage_evaluated_seq=dict(self.stage_evaluated_seq),
            stage_pending_commit=dict(self.stage_pending_commit),
            has_exported_target=self.has_exported_target,
            min_target_ratio=self.min_target_ratio,
            min_draft_ratio=self.min_draft_ratio,
            best_draft=self.best_draft,
            draft_pending_commit=self.draft_pending_commit,
            draft_needs_write=self.draft_needs_write,
        )


class PlacementExporter:
    """Track whole-model ratio-gated placements and export snapshots on global rank 0."""

    def __init__(self, ep_size: int, pp_rank: int, pp_size: int, is_stage_leader: bool, path: str | None) -> None:
        self.ep_size = ep_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.is_stage_leader = is_stage_leader
        self.path = path if (is_stage_leader and pp_rank == 0) else None
        self._state = _CoordinatorState(exported_targets=[None] * pp_size)
        self._committed_roles: set[str] = set()
        self._next_seq = 0
        self._lock = threading.Lock()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._endpoint: tuple[str, int] | None = None
        self._authkey: bytes | None = None

    @property
    def min_peak_to_mean_ratio(self) -> float | None:
        return self._state.min_target_ratio

    @min_peak_to_mean_ratio.setter
    def min_peak_to_mean_ratio(self, value: float | None) -> None:
        self._state.min_target_ratio = value

    @property
    def has_unexported_commit(self) -> bool:
        return bool(self._committed_roles)

    def mark_committed(self, role: str = "target") -> None:
        self._committed_roles.add(role)

    def start_pp_listener(self, host: str) -> tuple[str, int, bytes]:
        infos = socket.getaddrinfo(host or None, 0, type=socket.SOCK_STREAM)
        family = socket.AF_INET6 if any(info[0] == socket.AF_INET6 for info in infos) else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
                with suppress(OSError):
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            sock.bind(("::", 0, 0, 0) if family == socket.AF_INET6 else ("0.0.0.0", 0))
            sock.listen(max(8, self.pp_size * 2))
            sock.settimeout(_ACCEPT_POLL_SECONDS)
            port = int(sock.getsockname()[1])
        except Exception:
            sock.close()
            raise
        self._listener, self._authkey, self._closed = sock, os.urandom(32), False
        self._thread = threading.Thread(target=self._accept_loop, name="eplb-pp-export", daemon=True)
        self._thread.start()
        return host, port, self._authkey

    def configure_pp_stages(self, gathered: list[dict[str, Any]], same_node_as_writer: bool) -> None:
        if self.pp_rank == 0:
            with self._lock:
                st = self._state
                st.pp_configured, st.active_target_stages = True, set()
                for item in gathered:
                    r = int(item["pp_rank"])
                    st.exported_targets[r] = item["target"]
                    if item["target"] is not None:
                        st.active_target_stages.add(r)
                    if item["draft"] is not None:
                        st.best_draft = item["draft"]
        else:
            host, port, authkey = gathered[0]["endpoint"]
            loopback = "::1" if ":" in host else "127.0.0.1"
            self._endpoint = (loopback if same_node_as_writer else host, int(port))
            self._authkey = authkey

    def _accept_loop(self) -> None:
        sock, authkey = self._listener, self._authkey
        if sock is None or authkey is None:
            return
        while not self._closed:
            try:
                conn, _ = sock.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._closed:
                    break
                continue
            with conn:
                ok = False
                try:
                    conn.settimeout(_SOCKET_TIMEOUT_SECONDS)
                    ok = self.record_stage_report(**_recv_authenticated_json(conn, authkey))
                except Exception:
                    logger.exception("Could not process PP EPLB placement report on rank 0")
                with suppress(OSError):
                    conn.sendall(b"\x01" if ok else b"\x00")

    def close(self) -> None:
        self._closed = True
        if self._listener is not None:
            with suppress(OSError):
                self._listener.close()
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def record_stage_report(
        self,
        *,
        pp_rank: int,
        seq: int,
        target_stats: tuple[float, int] | list[float | int] | None,
        target_committed: bool,
        target_entry: dict[str, Any] | None,
        draft_stats: tuple[float, int] | list[float | int] | None,
        draft_committed: bool,
        draft_entry: dict[str, Any] | None,
    ) -> bool:
        if self.path is None:
            return False
        with self._lock:
            st = self._state.clone()
            if not st.pp_configured:
                st.pp_configured = True
                if target_entry is not None:
                    st.active_target_stages = {pp_rank}
                    st.exported_targets[pp_rank] = target_entry
            if draft_entry is not None and st.best_draft is None:
                st.best_draft = draft_entry
            if pp_rank in st.active_target_stages and target_entry is not None:
                st.stage_pending_commit[pp_rank] = st.stage_pending_commit.get(pp_rank, False) or bool(target_committed)
                if target_stats is not None:
                    st.stage_latest_target[pp_rank] = (
                        int(seq),
                        float(target_stats[0]),
                        int(target_stats[1]),
                        target_entry,
                    )
            if draft_entry is not None:
                st.draft_pending_commit = st.draft_pending_commit or bool(draft_committed)

            baseline_log: float | None = None
            export_log: tuple[str, float, float] | None = None
            if st.active_target_stages and all(
                s in st.stage_latest_target and st.stage_latest_target[s][0] > st.stage_evaluated_seq.get(s, -1)
                for s in st.active_target_stages
            ):
                for s in st.active_target_stages:
                    st.stage_evaluated_seq[s] = st.stage_latest_target[s][0]
                ratio = sum(st.stage_latest_target[s][1] for s in st.active_target_stages) / sum(
                    st.stage_latest_target[s][2] for s in st.active_target_stages
                )
                if st.min_target_ratio is None:
                    st.min_target_ratio = baseline_log = ratio
                elif ratio < st.min_target_ratio:
                    prev, st.min_target_ratio = st.min_target_ratio, ratio
                    if any(st.stage_pending_commit.get(s, False) for s in st.active_target_stages):
                        for s in st.active_target_stages:
                            st.exported_targets[s] = st.stage_latest_target[s][3]
                            st.stage_pending_commit[s] = False
                        st.has_exported_target = True
                        export_log = ("target", prev, ratio)

            if draft_stats is not None and draft_entry is not None:
                d_ratio = float(draft_stats[0]) / int(draft_stats[1])
                if st.min_draft_ratio is None:
                    st.min_draft_ratio = d_ratio
                elif d_ratio < st.min_draft_ratio:
                    d_prev, st.min_draft_ratio = st.min_draft_ratio, d_ratio
                    if st.draft_pending_commit:
                        st.best_draft, st.draft_pending_commit, st.draft_needs_write = draft_entry, False, True
                        if export_log is None:
                            export_log = ("draft", d_prev, d_ratio)

            should_write = (export_log is not None and export_log[0] == "target") or (
                st.draft_needs_write and (st.has_exported_target or not st.active_target_stages)
            )
            if should_write:
                snapshot = _build_stitched_snapshot(self.ep_size, st.exported_targets, st.best_draft)
                if snapshot is not None:
                    write_placement_snapshot(self.path, snapshot)
                    st.draft_needs_write = False
            self._state = st
            if baseline_log is not None:
                logger.info("Recorded initial EPLB baseline peak_to_mean_ratio=%.4f", baseline_log)
            if export_log is not None and should_write:
                logger.info(
                    "Exported committed EPLB %s placement (peak_to_mean_ratio %.4f -> %.4f)",
                    export_log[0],
                    export_log[1],
                    export_log[2],
                )
        return True

    def publish_evaluation(
        self,
        *,
        target_stats: tuple[float, int] | None,
        target_entry: dict[str, Any] | None,
        draft_stats: tuple[float, int] | None,
        draft_entry: dict[str, Any] | None,
    ) -> bool:
        t_comm, d_comm = "target" in self._committed_roles, "draft" in self._committed_roles
        seq = self._next_seq
        self._next_seq += 1
        payload: dict[str, Any] = {
            "pp_rank": self.pp_rank,
            "seq": seq,
            "target_stats": list(target_stats) if target_stats is not None else None,
            "target_committed": t_comm,
            "target_entry": target_entry,
            "draft_stats": list(draft_stats) if draft_stats is not None else None,
            "draft_committed": d_comm,
            "draft_entry": draft_entry,
        }
        if self.pp_rank == 0:
            ok = self.record_stage_report(**payload)
        elif self._endpoint is None or self._authkey is None:
            return False
        else:
            with socket.create_connection(self._endpoint, timeout=_SOCKET_TIMEOUT_SECONDS) as conn:
                conn.settimeout(_SOCKET_TIMEOUT_SECONDS)
                _send_authenticated_json(conn, self._authkey, payload)
                ok = _recv_exact(conn, 1) == b"\x01"
        if ok:
            self._committed_roles -= {r for r, c in (("target", t_comm), ("draft", d_comm)) if c}
        return ok
