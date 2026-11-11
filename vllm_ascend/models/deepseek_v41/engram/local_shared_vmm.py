# SPDX-License-Identifier: Apache-2.0
"""Node-local DP sharing for one Engram VMM buffer."""

import os
import tempfile
from datetime import timedelta

import torch.distributed as dist
from vllm.logger import logger

from .shared_host import MappedHostBuffer, retry_pending_rollbacks

_failed_initializations: list["LocalSharedVmmBuffer"] = []


def retain_failed_initialization(buffer):
    """Retain a live buffer whose caller cannot safely complete cleanup."""
    if not any(existing is buffer for existing in _failed_initializations):
        _failed_initializations.append(buffer)


def pending_failed_initializations():
    """Return failed constructors retained for explicit collective recovery.

    All members of the original sharing group must coordinate any retry.
    Process exit or peer loss is not made safe by retaining Python objects.
    """
    return tuple(_failed_initializations)


class EngramSharedInitError(RuntimeError):
    """A VMM initialization error was reported to the entire sharing group."""


class EngramSharedCleanupError(RuntimeError):
    """VMM rollback failed and further collectives cannot be assumed safe."""


class LocalSharedVmmBuffer:
    """One creator and per-rank imports; the creator alone publishes rows.

    Construction is collective. Callers must stop all graph users and close
    consumers before closing the creator.
    """

    def __init__(self, shape, dtype, device, group):
        self.group = group
        self.mapping = None
        self.view = None
        self.tensor = None
        self.ptrs = None
        self.directory = None
        self.owner = group.rank_in_group == 0
        cpu_group = group.cpu_group
        leader = dist.get_global_rank(cpu_group, 0)
        payload: list[str | None] = [None]
        if self.owner:
            try:
                self.directory = tempfile.mkdtemp(prefix="engram-vmm-", dir="/dev/shm")
                owner_path = os.path.join(self.directory, "buffer")
                self.mapping = MappedHostBuffer(owner_path, shape[0], shape[1], dtype, owner=True, device=device)
                payload[0] = owner_path
            except Exception as exc:  # noqa: BLE001 - propagate to all ranks
                payload[0] = f"ERROR: {type(exc).__name__}: {exc}"
        try:
            dist.broadcast_object_list(payload, src=leader, group=cpu_group)
        except Exception:
            # Peers may not be synchronized; retain locally without cleanup.
            retain_failed_initialization(self)
            raise
        path = payload[0]
        error: str | None = None
        if not path or path.startswith("ERROR:"):
            error = path or "owner did not publish a VMM descriptor"
        else:
            try:
                if not self.owner:
                    self.mapping = MappedHostBuffer(path, shape[0], shape[1], dtype, owner=False, device=device)
                assert self.mapping is not None
                self.view = self.mapping.view()
                self.tensor, self.ptrs = self.view.tensor, self.view.ptrs
            except Exception as exc:  # noqa: BLE001 - propagate to all ranks
                error = f"{type(exc).__name__}: {exc}"
        errors = [None] * group.world_size
        try:
            dist.all_gather_object(errors, error, group=cpu_group)
        except Exception:
            retain_failed_initialization(self)
            raise
        failures = "; ".join(f"rank {rank}: {message}" for rank, message in enumerate(errors) if message)
        if failures:
            # Native construction may leave pending rollback state that owns
            # descriptor files. Retry it even if other cleanup steps fail.
            try:
                self.close_collective(retry_rollbacks=True)
            except Exception as exc:  # noqa: BLE001 - retain original failure
                # __init__ will raise and its caller receives no buffer object.
                # Keep every rank's allocation metadata for coordinated retry.
                retain_failed_initialization(self)
                raise EngramSharedCleanupError(
                    f"Engram local VMM initialization failed: {failures}; cleanup: {type(exc).__name__}: {exc}"
                ) from exc
            raise EngramSharedInitError(f"Engram local VMM initialization failed: {failures}")

    def publish(self, start, source):
        if not self.owner or self.mapping is None:
            raise RuntimeError("Only the VMM owner may publish checkpoint rows")
        self.mapping.publish(start, source)

    def close_collective(self, retry_rollbacks=False):
        """Release consumers first; retain the owner if peers cannot confirm."""
        cpu_group = self.group.cpu_group
        timeout = timedelta(seconds=20)
        dist.monitored_barrier(group=cpu_group, timeout=timeout, wait_all_ranks=True)
        error = None
        if not self.owner:
            try:
                self._close_with_rollbacks(retry_rollbacks)
            except Exception as exc:  # noqa: BLE001 - report to all ranks
                error = f"{type(exc).__name__}: {exc}"
        dist.monitored_barrier(group=cpu_group, timeout=timeout, wait_all_ranks=True)
        errors = [None] * self.group.world_size
        dist.all_gather_object(errors, error, group=cpu_group)
        failures = "; ".join(f"rank {rank}: {message}" for rank, message in enumerate(errors) if message)
        if failures:
            raise RuntimeError(f"Engram VMM consumer release failed: {failures}")
        if self.owner:
            try:
                self._close_with_rollbacks(retry_rollbacks)
            except Exception as exc:  # noqa: BLE001 - report to all ranks
                error = f"{type(exc).__name__}: {exc}"
        dist.monitored_barrier(group=cpu_group, timeout=timeout, wait_all_ranks=True)
        dist.all_gather_object(errors, error, group=cpu_group)
        failures = "; ".join(f"rank {rank}: {message}" for rank, message in enumerate(errors) if message)
        if failures:
            raise RuntimeError(f"Engram VMM owner release failed: {failures}")
        if self in _failed_initializations:
            _failed_initializations.remove(self)

    def _close_with_rollbacks(self, retry_rollbacks):
        errors = []
        for cleanup in (retry_pending_rollbacks, self.close) if retry_rollbacks else (self.close,):
            try:
                cleanup()
            except Exception as exc:  # noqa: BLE001 - attempt both cleanup steps
                errors.append(f"{type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def close(self):
        if self.view is not None:
            self.view.close()
            self.view = None
            self.tensor = self.ptrs = None
        if self.mapping is not None:
            self.mapping.close()
            logger.info("Local Engram VMM mapping released: rank=%s owner=%s", self.group.rank_in_group, self.owner)
            self.mapping = None
        if self.directory is not None:
            os.rmdir(self.directory)
            self.directory = None
