# SPDX-License-Identifier: Apache-2.0
"""Node-local rendezvous for replicated MLA KV caches.

The shared directory must be on a local filesystem supporting flock (also
when mounted into multiple containers). A shared lock is held throughout
TE reads. Closing publication takes the exclusive lock, so an ACK deadline
can never release blocks underneath an in-flight reader.
"""

import contextlib
import fcntl
import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
POLL_INTERVAL = 0.01
INITIAL_POLL_INTERVAL = 0.001
PIN_WARNING_INTERVAL = 5.0
LOCK_STRIPES = 4096


class _LockTimeout(TimeoutError):
    """Distinguish rendezvous lock timeouts from TE read timeouts."""


def transferred_blocks(req_meta: dict[str, Any]) -> list[list[int]]:
    """Match the suffix trimming used by the hybrid transfer descriptors."""
    return [
        remote[-len(local) :] if local else []
        for remote, local in zip(req_meta["remote_block_ids"], req_meta["local_block_ids"], strict=True)
    ]


def reader_blocks(req_meta: dict[str, Any], publication: dict[str, Any]) -> list[list[int]]:
    """Translate P logical blocks to their actual locations in the leader."""
    result = []
    for required, source, received in zip(
        transferred_blocks(req_meta), publication["source_blocks"], publication["local_blocks"], strict=True
    ):
        if len(source) != len(received):
            raise ValueError("leader block table has inconsistent lengths")
        mapping = dict(zip(source, received, strict=True))
        # Missing coverage (e.g. a divergent prefix hit) must fall back to P.
        result.append([mapping[block] for block in required])
    return result


class MLAFanout:
    def __init__(self, config: dict[str, Any], rank: int, size: int):
        self.rank = rank
        self.size = size
        self.leader = config.get("leader_rank", 0)
        self.ready_timeout = float(config.get("ready_timeout", 30.0))
        self.ack_timeout = float(config.get("ack_timeout", 60.0))
        if not isinstance(self.leader, int) or not 0 <= self.leader < size:
            raise ValueError("leader_fanout.leader_rank must be within the decode TP group")
        if any(not math.isfinite(value) or value <= 0 for value in (self.ready_timeout, self.ack_timeout)):
            raise ValueError("leader_fanout timeouts must be finite and positive")
        directory = config.get("shared_dir")
        session = config.get("session_id")
        if not isinstance(directory, str) or not directory or not isinstance(session, str) or not session:
            raise ValueError("leader_fanout requires shared_dir and a fresh session_id for each decode startup")
        self.directory = Path(directory) / hashlib.sha256(session.encode()).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stats: Counter[str] = Counter()
        self._stats_lock = threading.Lock()

    def stats(self) -> dict[str, int]:
        """Return per-worker cumulative counters for diagnostics."""
        with self._stats_lock:
            return dict(self._stats)

    def _count(self, event: str, req_meta: dict[str, Any]) -> None:
        with self._stats_lock:
            self._stats[event] += 1
            if event.startswith("fallback_"):
                self._stats["fallback"] += 1
            count = self._stats[event]
        logger.debug(
            "MLA fanout event=%s count=%d rank=%d request=%s attempt=%s",
            event,
            count,
            self.rank,
            req_meta["request_id"],
            req_meta.get("transfer_attempt"),
        )

    def _key(self, req_meta: dict[str, Any]) -> str:
        # TP ranks may read different P replicas of the same MLA cache. Do not
        # include the source rank/port, or their rendezvous keys would differ.
        identity = (
            req_meta["request_id"],
            req_meta["remote_request_id"],
            req_meta["offset"],
            req_meta["transfer_attempt"],
        )
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    @contextlib.contextmanager
    def _lock(self, key: str, exclusive: bool = False, timeout: float | None = None) -> Iterator[None]:
        # Bounded lock files are never unlinked: unlinking an in-use flock inode
        # permits late readers to lock a different inode and bypass the pin.
        path = self.directory / f"lock-{int(key, 16) % LOCK_STRIPES}"
        with path.open("a+b") as lock:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            started = time.monotonic()
            deadline = None if timeout is None else started + max(0, timeout)
            warning_at = started + PIN_WARNING_INTERVAL
            interval = INITIAL_POLL_INTERVAL
            while True:
                try:
                    fcntl.flock(lock, operation | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    now = time.monotonic()
                    if deadline is not None and now >= deadline:
                        raise _LockTimeout(f"MLA fanout lock acquisition timed out: key={key} rank={self.rank}")
                    if now >= warning_at:
                        logger.warning(
                            "MLA fanout pin blocked key=%s rank=%d exclusive=%s waited=%.1fs; "
                            "KV remains pinned until active TE reads return; inspect/recover the worker",
                            key,
                            self.rank,
                            exclusive,
                            now - started,
                        )
                        with self._stats_lock:
                            self._stats["pin_wait_warnings"] += 1
                        warning_at = now + PIN_WARNING_INTERVAL
                    time.sleep(interval if deadline is None else min(interval, max(0, deadline - now)))
                    interval = min(interval * 2, POLL_INTERVAL)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _publish(self, path: Path, record: dict[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(dir=self.directory)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(record, stream)
                os.fchmod(stream.fileno(), 0o644)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def transfer(
        self,
        req_meta: dict[str, Any],
        pull: Callable[[], None],
        read: Callable[[dict[str, Any]], None],
        metadata: dict[str, Any],
    ) -> None:
        # Old/unversioned metadata must not participate in a rendezvous.
        if not isinstance(req_meta.get("transfer_attempt"), str) or not req_meta["transfer_attempt"]:
            self._count("fallback_missing_attempt", req_meta)
            logger.warning("MLA fanout missing scheduler attempt; falling back to P request=%s", req_meta["request_id"])
            pull()
            return
        key = self._key(req_meta)
        ready = self.directory / f"{key}.json"
        if self.rank == self.leader:
            # Publish only after the synchronous TE transfer succeeds.
            pull()
            self._count("leader_pull", req_meta)
            record = dict(metadata)
            record.update(
                source_blocks=transferred_blocks(req_meta),
                local_blocks=req_meta["local_block_ids"],
                deadline=time.monotonic() + self.ack_timeout,
                transfer_attempt=req_meta["transfer_attempt"],
            )
            try:
                with self._lock(key, exclusive=True):
                    self._publish(ready, record)
                interval = INITIAL_POLL_INTERVAL
                while time.monotonic() < record["deadline"]:
                    if all(
                        (self.directory / f"{key}.ack-{rank}").exists()
                        for rank in range(self.size)
                        if rank != self.leader
                    ):
                        break
                    time.sleep(min(interval, max(0, record["deadline"] - time.monotonic())))
                    interval = min(interval * 2, POLL_INTERVAL)
                else:
                    self._count("ack_timeout", req_meta)
                    logger.warning("MLA leader fanout ACK deadline reached for %s", req_meta["request_id"])
            finally:
                # A reader already inside TE holds the shared lock until TE
                # returns, even past the deadline. Late readers see no ready
                # record and fall back to P. Do not impose a timeout on this pin.
                with self._lock(key, exclusive=True):
                    ready.unlink(missing_ok=True)
                    for rank in range(self.size):
                        (self.directory / f"{key}.ack-{rank}").unlink(missing_ok=True)
            return

        ack_sent = False
        deadline = time.monotonic() + self.ready_timeout
        interval = INITIAL_POLL_INTERVAL
        reason = "ready_timeout"
        while time.monotonic() < deadline:
            try:
                with self._lock(key, timeout=max(0, deadline - time.monotonic())):
                    if ready.exists():
                        try:
                            record = json.loads(ready.read_text())
                            if record["transfer_attempt"] != req_meta["transfer_attempt"]:
                                raise ValueError("MLA fanout attempt mismatch")
                            if time.monotonic() >= record["deadline"]:
                                reason = "expired_publication"
                                break
                            read(record)
                            self._count("local_read", req_meta)
                            return
                        finally:
                            # Includes incompatible layouts, insufficient leader
                            # coverage and TE errors. ACK before fallback.
                            (self.directory / f"{key}.ack-{self.rank}").touch()
                            ack_sent = True
            except _LockTimeout:
                reason = "lock_timeout"
                logger.warning("MLA reader lock timed out; falling back to P request=%s", req_meta["request_id"])
                break
            except Exception:
                reason = "local_read_error"
                logger.warning("MLA local read failed; falling back to P", exc_info=True)
                break
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
            interval = min(interval * 2, POLL_INTERVAL)
        # A ready timeout may precede the leader's P read completion. Publish
        # an early ACK so the leader does not wait for this fallback reader.
        try:
            if not ack_sent:
                with self._lock(key, timeout=max(0, deadline - time.monotonic())):
                    (self.directory / f"{key}.ack-{self.rank}").touch()
        except OSError:
            # The leader's deadline handles a lost ACK. Failure of the local
            # rendezvous must not prevent a valid fallback read from P.
            logger.warning("Unable to ACK MLA leader fanout", exc_info=True)
        self._count(f"fallback_{reason}", req_meta)
        logger.info(
            "MLA reader falling back to P request=%s attempt=%s rank=%d reason=%s stats=%s",
            req_meta["request_id"],
            req_meta["transfer_attempt"],
            self.rank,
            reason,
            self.stats(),
        )
        pull()
