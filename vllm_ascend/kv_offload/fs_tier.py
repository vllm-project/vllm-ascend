# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend filesystem secondary tier for multi-tier KV cache offloading.

The upstream ``fs_python`` tier (``vllm.v1.kv_offload.tiering.fs``) opens every
block file with ``O_DIRECT``. ``O_DIRECT`` imposes strict alignment
requirements on the buffer address, the file offset and the transfer length
(typically the filesystem logical block size, 512 B or 4 KiB). The primary-tier
CPU buffer is an mmap region whose per-block stride (``block_size``) is derived
from the model KV layout and is frequently *not* a multiple of that alignment.
On filesystems such as 3FS (mounted via FUSE) this surfaces as
``[Errno 22] Invalid argument`` (``EINVAL``) on the first write, e.g.::

    FileSystemTierManagerPython: job N block I/O failed:
        [Errno 22] Invalid argument: '/3fs/.../<hash>.bin_<suffix>.tmp'

This tier reuses the entire upstream ``FileSystemTierManager`` flow (file
mapping, thread pool, lookup, job bookkeeping) and only replaces the per-block
read/write so that ``O_DIRECT`` is disabled by default. KV blocks are therefore
written/read through buffered I/O, which works on any filesystem including 3FS.

``O_DIRECT`` can be re-enabled per tier for filesystems that support it by
setting ``"use_direct_io": true`` in the secondary tier configuration.
"""

import functools
import logging
import os
import random
import threading

from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

logger = logging.getLogger(__name__)

# Thread-local unique suffix for temp files (mirrors upstream io.py).
_thread_local = threading.local()

# Process-wide cache of directories already known to exist. On FUSE-backed
# filesystems such as 3FS, ``os.makedirs`` walks and stats every path
# component, so issuing it per block (the upstream behaviour) dominates the
# offload cost when thousands of small block files are written. KV block paths
# share a small, fixed set of parent directories, so caching the ones we have
# created collapses that syscall storm to (almost) one ``makedirs`` per
# directory. Membership tests run lock-free on the fast path; only insertions
# take the lock. A spurious miss merely re-issues an idempotent
# ``makedirs(exist_ok=True)``, so the unlocked read is safe.
_created_dirs: set[str] = set()
_created_dirs_lock = threading.Lock()


def _ensure_dir(dir_path: str) -> None:
    """Create ``dir_path`` (and parents) once, caching the result."""
    if dir_path in _created_dirs:
        return
    os.makedirs(dir_path, exist_ok=True)
    with _created_dirs_lock:
        _created_dirs.add(dir_path)


def _get_tmp_suffix() -> str:
    try:
        return _thread_local.tmp_suffix
    except AttributeError:
        _thread_local.tmp_suffix = f"_{random.randint(0, 2**63 - 1)}.tmp"
        return _thread_local.tmp_suffix


def store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
    open_flags: int,
) -> None:
    """Write one KV block to ``dest_path`` atomically (temp file + replace).

    Unlike the upstream implementation, the open flags are injected by the
    caller so ``O_DIRECT`` can be omitted. A short-write loop makes the write
    robust for large KV blocks where a single ``os.write`` may be partial.
    """
    if os.path.exists(dest_path):
        return

    dir_path = os.path.dirname(dest_path)
    tmp_path = dest_path + _get_tmp_suffix()
    _ensure_dir(dir_path)

    # Flat byte view; the raw memoryview may be multi-dimensional.
    view_slice = buffer.cast("B")[offset : offset + block_size]
    try:
        try:
            fd = os.open(tmp_path, open_flags, 0o644)
        except FileNotFoundError:
            # The parent directory was reclaimed (e.g. evicted by another
            # process) after we cached it; recreate it and retry once.
            _created_dirs.discard(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            with _created_dirs_lock:
                _created_dirs.add(dir_path)
            fd = os.open(tmp_path, open_flags, 0o644)
        try:
            total = len(view_slice)
            written = 0
            while written < total:
                n = os.write(fd, view_slice[written:])
                if n <= 0:
                    raise OSError(f"Short write: expected {total} bytes, wrote {written}")
                written += n
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
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
    open_flags: int,
) -> None:
    """Read one KV block from ``source_path`` into ``view``.

    Removes the file on failure so a corrupt/short entry is not retried.
    """
    fd: int | None = None
    view_slice = view.cast("B")[offset : offset + block_size]
    try:
        fd = os.open(source_path, open_flags)
        read_total = 0
        while read_total < block_size:
            n = os.readv(fd, [view_slice[read_total:]])
            if n == 0:
                break  # EOF
            read_total += n
        if read_total < block_size:
            raise OSError(f"Short read: expected {block_size} bytes, read {read_total}")
    except Exception:
        try:
            os.remove(source_path)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove unreadable file %s: %s", source_path, cleanup_exc)
        raise
    finally:
        if fd is not None:
            os.close(fd)


class AscendFileSystemTierManager(FileSystemTierManager):
    """``fs_python`` secondary tier without the ``O_DIRECT`` requirement.

    Accepts every upstream ``FileSystemTierManager`` argument plus one Ascend
    knob:

    * ``use_direct_io`` (default ``False``) -- when ``True`` and the platform
      supports it, ``O_DIRECT`` is used exactly like upstream.
    """

    def __init__(
        self,
        offloading_spec,
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        use_direct_io: bool = False,
    ):
        direct = getattr(os, "O_DIRECT", 0) if use_direct_io else 0
        if use_direct_io and direct == 0:
            logger.warning(
                "use_direct_io=True but O_DIRECT is unavailable on this platform; falling back to buffered I/O."
            )
        self._store_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | direct
        self._load_flags = os.O_RDONLY | direct

        super().__init__(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
            root_dir=root_dir,
            n_read_threads=n_read_threads,
            n_write_threads=n_write_threads,
        )

    def submit_store(self, job_metadata: JobMetadata) -> None:
        tasks = (
            functools.partial(
                store_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
                self._store_flags,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    def submit_load(self, job_metadata: JobMetadata) -> None:
        tasks = (
            functools.partial(
                load_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
                self._load_flags,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)
