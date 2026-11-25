# SPDX-License-Identifier: Apache-2.0
"""Node-local DP sharing for one Engram VMM buffer."""

import ctypes
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from vllm.logger import logger

GIB = 1 << 30
CHUNK_ROWS = 1 << 22


class EngramReleaseError(RuntimeError):
    """A VMM release failed and remains eligible for retry."""


class _Allocation(ctypes.Structure):
    _fields_ = [
        ("host_ptr", ctypes.c_void_p),
        ("device_ptr", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("token", ctypes.c_uint64),
    ]


def _library(path: Path):
    lib = ctypes.CDLL(str(path))
    if lib.host_vmm_abi_version() != 3:
        raise RuntimeError("Engram host-VMM ABI mismatch")
    lib.host_shared_registered_alloc.argtypes = [
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(_Allocation),
    ]
    lib.host_shared_registered_free.argtypes = [ctypes.POINTER(_Allocation)]
    lib.host_shared_last_error.restype = ctypes.c_char_p
    return lib


def retry_pending_rollbacks(*, lib):
    if lib.host_vmm_retry_rollbacks():
        raise EngramReleaseError(f"Engram VMM rollback: {lib.host_shared_last_error().decode()}")


class MappedHostBuffer:
    """One physical range or its import; no CPU aliases escape this object.

    The creator must outlive consumers until the pool's lease manager releases
    them. Stop all launches/graphs before close. Failed release stays retryable.
    """

    def __init__(self, path, rows, width, dtype, *, owner, device, lib):
        if rows <= 0 or width <= 0 or dtype not in (torch.int8, torch.float32):
            raise ValueError("Host VMM requires positive INT8/FP32 dimensions")
        path = Path(path)
        parent = path.parent.stat()
        if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
            raise ValueError("Capability directory must be owned by this uid and mode0700")
        self.rows, self.width, self.dtype = rows, width, dtype
        self.owner = owner
        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("An explicit NPU device index is required")
        self.row_bytes = width * torch.empty((), dtype=dtype).element_size()
        self.size = ((rows * self.row_bytes + GIB - 1) // GIB) * GIB
        self.lib = lib
        self.allocation = _Allocation()
        self.closed = False
        with torch.npu.device(self.device):
            rc = self.lib.host_shared_registered_alloc(
                self.size, self.device.index, os.fsencode(path), int(owner), ctypes.byref(self.allocation)
            )
        if rc:
            raise RuntimeError(f"Engram host VMM: {self.lib.host_shared_last_error().decode()}")

    def publish(self, start, source):
        """Copy one CPU checkpoint chunk into creator-owned host memory."""
        if self.closed or not self.owner:
            raise RuntimeError("Only a live owner may publish table rows")
        if source.device.type != "cpu" or source.dtype != self.dtype:
            raise ValueError("Publication requires a matching CPU dtype")
        if source.ndim != 2 or source.shape[1] != self.width or start < 0 or start + len(source) > self.rows:
            raise ValueError("Publication is outside the table shape")
        source = source.contiguous()
        ctypes.memmove(
            self.allocation.host_ptr + start * self.row_bytes, source.data_ptr(), source.numel() * source.element_size()
        )

    def close(self):
        if self.closed:
            return
        with torch.npu.device(self.device):
            torch.npu.synchronize()
            rc = self.lib.host_shared_registered_free(ctypes.byref(self.allocation))
        if rc:
            raise EngramReleaseError(f"Engram VMM release: {self.lib.host_shared_last_error().decode()}")
        self.closed = True


_failed_initializations: list["LocalSharedVmmBuffer"] = []


def local_vmm_available(group):
    """Select the same backend on every rank before allocating any table."""
    cached = getattr(group, "_engram_local_vmm_available", None)
    if cached is not None:
        return cached
    try:
        probe = LocalSharedVmmBuffer((1, 1), torch.int8, torch.device("npu", torch.npu.current_device()), group)
    except EngramSharedInitError as exc:
        logger.warning("Engram Host VMM runtime probe failed; using HostRegister: %s", exc)
        group._engram_local_vmm_available = False
        return False
    probe.close_collective()
    group._engram_local_vmm_available = True
    return True


def retain_failed_initialization(buffer):
    """Retain a live buffer whose caller cannot safely complete cleanup."""
    if not any(existing is buffer for existing in _failed_initializations):
        _failed_initializations.append(buffer)


class EngramSharedInitError(RuntimeError):
    """A VMM initialization error was reported to the entire sharing group."""


class EngramSharedCleanupError(RuntimeError):
    """VMM rollback failed and further collectives cannot be assumed safe."""


class EngramSharedReleaseError(RuntimeError):
    """Every rank observed a release failure and can continue in order."""


class LocalSharedVmmScaleView:
    """A scale view into the same allocation as the codes."""

    def __init__(self, owner, shape, dtype, offset, device):
        self.owner = owner
        self.shape, self.dtype, self.offset = shape, dtype, offset
        self.row_bytes = shape[1] * torch.empty((), dtype=dtype).element_size()
        self.tensor = torch.empty(shape, dtype=dtype, device="meta")
        self.ptrs = torch.tensor(
            [
                owner.mapping.allocation.device_ptr + offset + row * self.row_bytes
                for row in range(0, shape[0], CHUNK_ROWS)
            ],
            dtype=torch.int64,
            device=device,
        )

    def publish(self, start, source):
        if not self.owner.owner or self.owner.mapping is None:
            raise RuntimeError("Only the VMM owner may publish checkpoint rows")
        if source.device.type != "cpu" or source.dtype != self.dtype or source.ndim != 2:
            raise ValueError("Publication requires a matching CPU tensor")
        if source.shape[1] != self.shape[1] or start < 0 or start + len(source) > self.shape[0]:
            raise ValueError("Publication is outside the scale shape")
        source = source.contiguous()
        ctypes.memmove(
            self.owner.mapping.allocation.host_ptr + self.offset + start * self.row_bytes,
            source.data_ptr(),
            source.numel() * source.element_size(),
        )


def _codes_view(mapping, shape, dtype, device):
    tensor = torch.empty(shape, dtype=dtype, device="meta")
    ptrs = torch.tensor(
        [mapping.allocation.device_ptr + row * mapping.row_bytes for row in range(0, shape[0], CHUNK_ROWS)],
        dtype=torch.int64,
        device=device,
    )
    return tensor, ptrs


class LocalSharedVmmBuffer:
    """One creator and per-rank imports; the creator alone publishes rows.

    Construction is collective. Callers must stop all graph users and close
    consumers before closing the creator.
    """

    def __init__(self, shape, dtype, device, group, *, scales_shape=None, scale_dtype=None):
        self.group = group
        self.mapping = None
        self.tensor = None
        self.ptrs = None
        self.scales: LocalSharedVmmScaleView | None = None
        self.directory = None
        self.owner = group.rank_in_group == 0
        self.lib = None
        library_error = None
        try:
            library_path = Path(__file__).resolve().parents[3] / "libengram_host_vmm.so"
            self.lib = _library(library_path)
        except Exception as exc:  # noqa: BLE001 - synchronize library failures
            library_error = f"{type(exc).__name__}: {exc}"
        allocation_shape = shape
        scale_offset = shape[0] * shape[1] * torch.empty((), dtype=dtype).element_size()
        if scales_shape is not None:
            assert dtype == torch.int8 and scale_dtype == torch.float32
            scale_bytes = scales_shape[0] * scales_shape[1] * torch.empty((), dtype=scale_dtype).element_size()
            allocation_shape = ((scale_offset + scale_bytes + shape[1] - 1) // shape[1], shape[1])
        cpu_group = group.cpu_group
        leader = dist.get_global_rank(cpu_group, 0)
        payload: list[str | None] = [None]
        if self.owner:
            try:
                if library_error is not None:
                    raise RuntimeError(library_error)
                self.directory = tempfile.mkdtemp(prefix="engram-vmm-", dir="/dev/shm")
                owner_path = os.path.join(self.directory, "buffer")
                self.mapping = MappedHostBuffer(
                    owner_path,
                    allocation_shape[0],
                    allocation_shape[1],
                    dtype,
                    owner=True,
                    device=device,
                    lib=self.lib,
                )
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
        elif library_error is not None:
            error = library_error
        else:
            try:
                if not self.owner:
                    self.mapping = MappedHostBuffer(
                        path,
                        allocation_shape[0],
                        allocation_shape[1],
                        dtype,
                        owner=False,
                        device=device,
                        lib=self.lib,
                    )
                assert self.mapping is not None
                self.tensor, self.ptrs = _codes_view(self.mapping, shape, dtype, device)
                if scales_shape is not None:
                    self.scales = LocalSharedVmmScaleView(self, scales_shape, scale_dtype, scale_offset, device)
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
            raise EngramSharedReleaseError(f"Engram VMM consumer release failed: {failures}")
        if self.owner:
            try:
                self._close_with_rollbacks(retry_rollbacks)
            except Exception as exc:  # noqa: BLE001 - report to all ranks
                error = f"{type(exc).__name__}: {exc}"
        dist.monitored_barrier(group=cpu_group, timeout=timeout, wait_all_ranks=True)
        dist.all_gather_object(errors, error, group=cpu_group)
        failures = "; ".join(f"rank {rank}: {message}" for rank, message in enumerate(errors) if message)
        if failures:
            raise EngramSharedReleaseError(f"Engram VMM owner release failed: {failures}")
        if self in _failed_initializations:
            _failed_initializations.remove(self)

    def _close_with_rollbacks(self, retry_rollbacks):
        errors = []
        for cleanup in (
            (lambda: retry_pending_rollbacks(lib=self.lib), self.close)
            if retry_rollbacks and getattr(self, "lib", None) is not None
            else (self.close,)
        ):
            try:
                cleanup()
            except Exception as exc:  # noqa: BLE001 - attempt both cleanup steps
                errors.append(f"{type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def close(self):
        if self.scales is not None:
            self.scales.tensor = self.scales.ptrs = None
            self.scales = None
        self.tensor = self.ptrs = None
        if self.mapping is not None:
            self.mapping.close()
            logger.info("Local Engram VMM mapping released: rank=%s owner=%s", self.group.rank_in_group, self.owner)
            self.mapping = None
        if self.directory is not None:
            os.rmdir(self.directory)
            self.directory = None
