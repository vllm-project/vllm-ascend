# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Native 3FS USRBIO (RDMA zero-copy) backend for the filesystem KV tier.

The buffered secondary tier (:mod:`vllm_ascend.kv_offload.fs_tier`) reads and
writes KV blocks through ordinary ``os.read`` / ``os.write`` against the 3FS
FUSE mount. Every byte therefore crosses the FUSE boundary and the kernel page
cache before the 3FS client forwards it to the storage nodes. 3FS ships a
user-space block I/O API (**USRBIO**, ``hf3fs_usrbio``) that bypasses FUSE
entirely: data is DMA'd directly between a NIC-registered shared-memory buffer
and the storage nodes over **RDMA**. This module wraps that API and exposes two
drop-in block primitives (:func:`store_block` / :func:`load_block`) with exactly
the same semantics (atomic temp-file write, short-read/short-write detection,
corrupt-entry removal) as the buffered tier, so functionality and bit-exact KV
fidelity are preserved.

Design notes
------------
* **Bounce buffer, not full zero-copy.** True end-to-end zero-copy would require
  the primary CPU KV mmap region itself to be allocated from a registered
  ``iovec``. That is owned by upstream ``SharedOffloadRegion`` and out of scope
  here, so each worker thread keeps a small per-thread registered ``iovec`` and
  performs one CPU ``memcpy`` between it and the primary buffer. The expensive
  hop -- crossing FUSE and the kernel for every block -- is what USRBIO removes;
  the residual in-DRAM copy is cheap by comparison.
* **Per-thread rings.** A USRBIO I/O ring (``ioring``) is not thread-safe, so
  each pool thread lazily creates its own ring + ``iovec`` via thread-local
  state. All contexts are tracked for deterministic cleanup on shutdown.
* **Safe by construction.** :func:`probe` returns ``None`` -- and the caller
  transparently falls back to the buffered path -- whenever the binding is
  absent, ``root_dir`` is not on a 3FS mount, or ring/iovec setup fails. USRBIO
  is therefore strictly opt-in (``"use_usrbio": true``) and can never break a
  working deployment.

.. warning::
   The symbol names used to talk to the binding are centralised in
   :class:`_Hf3fsApi`. They follow the upstream ``hf3fs_fuse.io`` Python wrapper
   shipped with DeepSeek 3FS. If your 3FS build exposes the bindings under
   different names, adjust :class:`_Hf3fsApi` only -- the rest of the module is
   binding-agnostic. This data path must be validated on a real 3FS + RDMA node;
   it cannot be exercised without the native library.
"""

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Reuse the buffered tier's thread-local temp suffix so concurrent writers to
# the same destination never collide on the temp file name.
from vllm_ascend.kv_offload.fs_tier import _ensure_dir, _get_tmp_suffix

# Flags for the underlying file objects. USRBIO manages its own data path and
# does not require ``O_DIRECT`` (and the 3FS FUSE layer rejects it anyway), so
# plain buffered-open flags are used purely to create/open the file the ring
# then operates on by fd.
_STORE_OPEN_FLAGS = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC
_LOAD_OPEN_FLAGS = os.O_RDONLY

# I/O ring depth. One block is in flight per submit, so a tiny ring suffices.
_RING_ENTRIES = 1


class _Hf3fsApi:
    """Thin, centralised adapter over the 3FS USRBIO Python binding.

    This is the *only* place that references binding symbols. Keeping it
    isolated means adapting to a different 3FS binding version touches one
    class. Construction raises ``ImportError`` if the binding is unavailable.
    """

    def __init__(self) -> None:
        # The upstream wrapper module; raises ImportError when 3FS is not
        # installed, which probe() treats as "USRBIO unavailable".
        import hf3fs_fuse.io as _io  # type: ignore

        self._io = _io
        # Resolve the handful of callables we need, tolerating the common
        # naming variants seen across binding versions.
        self._iovec = self._resolve("iovec", "make_iovec", "Iov")
        self._ioring = self._resolve("ioring", "make_ioring", "Ior")
        self._extract_mount_point = self._resolve(
            "extract_mount_point", "extractMountPoint"
        )
        self._register_fd = self._resolve("register_fd", "registerFd", "reg_fd")
        self._deregister_fd = self._resolve(
            "deregister_fd", "deregisterFd", "dereg_fd"
        )

    def _resolve(self, *names: str):
        for name in names:
            fn = getattr(self._io, name, None)
            if fn is not None:
                return fn
        raise ImportError(
            f"hf3fs_fuse.io exposes none of {names!r}; the installed 3FS "
            "USRBIO binding is incompatible with this adapter."
        )

    def extract_mount_point(self, path: str) -> Optional[str]:
        try:
            return self._extract_mount_point(path)
        except Exception:
            return None

    def make_iovec(self, mount_point: str, size: int):
        return self._iovec(mount_point, size)

    def make_ioring(self, mount_point: str, iov, for_read: bool):
        return self._ioring(mount_point, iov, for_read, _RING_ENTRIES)

    def register_fd(self, fd: int) -> None:
        self._register_fd(fd)

    def deregister_fd(self, fd: int) -> None:
        self._deregister_fd(fd)

    @staticmethod
    def iov_memoryview(iov) -> memoryview:
        """Return a writable ``memoryview`` over a registered ``iovec``."""
        try:
            return memoryview(iov).cast("B")
        except TypeError:
            for attr in ("data", "buffer", "buf"):
                buf = getattr(iov, attr, None)
                if buf is not None:
                    return memoryview(buf).cast("B")
        raise TypeError("Cannot obtain a memoryview over the iovec buffer")

    @staticmethod
    def ring_transfer(ring, iov_slice, for_read: bool, fd: int,
                      file_offset: int, length: int) -> int:
        """Submit one I/O on ``ring`` and return the bytes transferred."""
        ring.prepare(iov_slice, for_read, fd, file_offset)
        ring.submit()
        results = ring.wait(1)
        # ``wait`` returns one completion per in-flight op. Each completion's
        # ``result`` is the byte count (>=0) or a negative errno.
        completion = results[0]
        n = getattr(completion, "result", completion)
        if n is None:
            n = length
        if n < 0:
            raise OSError(-n, f"USRBIO I/O failed with errno {-n}")
        return int(n)


class _ThreadCtx:
    """Per-thread registered buffer plus read and write rings."""

    def __init__(self, api: _Hf3fsApi, mount_point: str, capacity: int):
        self._api = api
        self.iov = api.make_iovec(mount_point, capacity)
        self.mv = api.iov_memoryview(self.iov)
        self.read_ring = api.make_ioring(mount_point, self.iov, True)
        self.write_ring = api.make_ioring(mount_point, self.iov, False)

    def close(self) -> None:
        # Drop references; the binding objects release their RDMA/shm
        # resources in their destructors. Guard each in case it is partial.
        for attr in ("write_ring", "read_ring", "mv", "iov"):
            try:
                setattr(self, attr, None)
            except Exception:  # pragma: no cover - defensive
                pass


class Hf3fsUsrbioBackend:
    """Owns the USRBIO resources shared by the tier's worker threads."""

    def __init__(self, api: _Hf3fsApi, mount_point: str, capacity: int):
        self._api = api
        self._mount_point = mount_point
        self._capacity = capacity
        self._tls = threading.local()
        self._contexts: list[_ThreadCtx] = []
        self._lock = threading.Lock()

    def _ctx(self) -> _ThreadCtx:
        ctx = getattr(self._tls, "ctx", None)
        if ctx is None:
            ctx = _ThreadCtx(self._api, self._mount_point, self._capacity)
            self._tls.ctx = ctx
            with self._lock:
                self._contexts.append(ctx)
        return ctx

    def write(self, fd: int, view_slice: memoryview, length: int) -> int:
        ctx = self._ctx()
        ctx.mv[:length] = view_slice
        self._api.register_fd(fd)
        try:
            return self._api.ring_transfer(
                ctx.write_ring, ctx.mv[:length], False, fd, 0, length
            )
        finally:
            self._api.deregister_fd(fd)

    def read(self, fd: int, out_slice: memoryview, length: int) -> int:
        ctx = self._ctx()
        self._api.register_fd(fd)
        try:
            n = self._api.ring_transfer(
                ctx.read_ring, ctx.mv[:length], True, fd, 0, length
            )
        finally:
            self._api.deregister_fd(fd)
        if n >= length:
            out_slice[:length] = ctx.mv[:length]
        return n

    def close(self) -> None:
        with self._lock:
            contexts, self._contexts = self._contexts, []
        for ctx in contexts:
            ctx.close()


def probe(root_dir: str, block_size: int) -> Optional[Hf3fsUsrbioBackend]:
    """Return a USRBIO backend for ``root_dir`` or ``None`` to fall back.

    ``None`` is returned (with a logged reason) whenever USRBIO cannot be used,
    so the caller can transparently use the buffered path instead.
    """
    try:
        api = _Hf3fsApi()
    except ImportError as exc:
        logger.warning(
            "use_usrbio=True but the 3FS USRBIO binding is unavailable (%s); "
            "falling back to buffered I/O.", exc
        )
        return None

    mount_point = api.extract_mount_point(root_dir)
    if not mount_point:
        logger.warning(
            "use_usrbio=True but '%s' is not on a 3FS mount; falling back to "
            "buffered I/O.", root_dir
        )
        return None

    try:
        backend = Hf3fsUsrbioBackend(api, mount_point, block_size)
        # Eagerly validate that a ring + iovec can actually be created on this
        # thread; a failure here means we must not advertise USRBIO.
        backend._ctx()
    except Exception as exc:
        logger.warning(
            "use_usrbio=True but USRBIO initialisation failed (%s); falling "
            "back to buffered I/O.", exc
        )
        return None

    logger.info(
        "USRBIO (RDMA) enabled for KV tier on 3FS mount '%s' (block_size=%d).",
        mount_point, block_size,
    )
    return backend


def store_block(
    backend: Hf3fsUsrbioBackend,
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """Write one KV block to ``dest_path`` via USRBIO (atomic temp + replace)."""
    if os.path.exists(dest_path):
        return

    dir_path = os.path.dirname(dest_path)
    tmp_path = dest_path + _get_tmp_suffix()
    _ensure_dir(dir_path)

    view_slice = buffer.cast("B")[offset : offset + block_size]
    try:
        try:
            fd = os.open(tmp_path, _STORE_OPEN_FLAGS, 0o644)
        except FileNotFoundError:
            os.makedirs(dir_path, exist_ok=True)
            fd = os.open(tmp_path, _STORE_OPEN_FLAGS, 0o644)
        try:
            written = backend.write(fd, view_slice, block_size)
            if written != block_size:
                raise OSError(
                    f"USRBIO short write: expected {block_size} bytes, "
                    f"wrote {written}"
                )
        finally:
            os.close(fd)
        os.replace(tmp_path, dest_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove temp file %s: %s", tmp_path, cleanup_exc)
        raise


def load_block(
    backend: Hf3fsUsrbioBackend,
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """Read one KV block from ``source_path`` via USRBIO into ``view``."""
    view_slice = view.cast("B")[offset : offset + block_size]
    fd: Optional[int] = None
    try:
        fd = os.open(source_path, _LOAD_OPEN_FLAGS)
        read_total = backend.read(fd, view_slice, block_size)
        if read_total < block_size:
            raise OSError(
                f"USRBIO short read: expected {block_size} bytes, "
                f"read {read_total}"
            )
    except Exception:
        try:
            os.remove(source_path)
        except OSError as cleanup_exc:
            logger.warning(
                "Failed to remove unreadable file %s: %s", source_path, cleanup_exc
            )
        raise
    finally:
        if fd is not None:
            os.close(fd)
