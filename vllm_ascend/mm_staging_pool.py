#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Fixed pinned staging pool for the multimodal input path (issue #21094).

Why:
    torch_npu 2.10 pinned allocations go through ``aclrtMallocHostWithCfg``
    (VA flag), binding every allocation to 2MiB base-heap physical pages
    with no small-page fallback. The multimodal input path allocates
    per-request staging temporaries and cross-request merge outputs via
    ``pin_memory=True``, producing a high-frequency stream of variable-size
    driver grabs (tiny scheduling metadata plus image staging/merges,
    observed 85MB-1GB). Under host memory pressure the 2MiB buddy
    inventory drains and these grabs fail with error 207001/EL0018,
    crashing serving.

How:
    One fixed pinned pool is allocated once per process at first use
    (VA-backed 2MiB pages, full DMA speed) and callers route temporaries
    through pool slices: multi-slot contiguous slices serve oversize
    merges, and a slice is recycled only after the H2D copy enqueued from
    it completes (event-gated; the event is recorded AFTER the copy is
    enqueued). Size and slot granularity come from
    ``additional_config.mm_staging_config``; ``pinned_pool_mb=0`` keeps
    the original allocation behavior. Pool slices are reused across
    requests, so every handed-out slice must be fully overwritten by the
    caller before it is published.

    Module-level state and its lifetime: ``_POOL`` holds the lazily
    created per-process pool (a process-resident singleton by design;
    processes that never touch the multimodal path never pay the pinned
    allocation), ``_POOL_FAILED`` pins the disabled/failed decision, and
    ``_LOCK`` serializes pool creation and slot bookkeeping.

Related PR (if no, explain why):
    No upstream vllm PR: the failure comes from an NPU-driver-specific
    allocation behavior (ACL_RT_MEM_ATTR_VA_FLAG base-heap pinning) that
    does not exist on CUDA, so the pool lives in the vllm-ascend platform
    layer. The allocator-side alternative is tracked separately in
    torch_npu MR 49638 (``pinned_mem_register`` gate with plain
    small-page fallback).

Future Plan:
    Keep while torch_npu lacks a failure fallback for the VA pinned path;
    revisit when the allocator provides small-page fallback natively.
"""

import contextlib
import threading

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Defaults for additional_config.mm_staging_config; mirrored by
# vllm_ascend.ascend_config.MMStagingConfig.
DEFAULT_POOL_MB = 2048
DEFAULT_SLOT_MB = 512

_LOCK = threading.Lock()
_POOL: "MMPinnedStagingPool | None" = None
_POOL_FAILED = False
_WARNED_NO_EVENT = False


def _new_pinned_event():
    """Create an event on the current stream of the current accelerator.

    torch has no unified event API across accelerator backends; try the
    common ones and return None when none is available (the caller keeps
    the slice active instead of recycling it, which is safe but bounded).
    """
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.npu.Event()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return torch.cuda.Event()
    except Exception:
        pass
    return None


class MMPinnedStagingPool:
    """Fixed pinned staging pool: startup allocation, runtime slice reuse.

    Slots are consecutive windows of one contiguous pinned buffer, so a
    merge larger than one slot is still served from the pool by taking
    several adjacent slots (single contiguous view, zero driver grabs).
    """

    def __init__(self, pool_bytes: int, slot_bytes: int):
        if pool_bytes <= 0 or slot_bytes <= 0 or pool_bytes < slot_bytes:
            raise ValueError("staging pool disabled by configuration")
        self.slot_bytes = slot_bytes
        self.num_slots = pool_bytes // slot_bytes
        self._buf = self._allocate_backing(self.num_slots * slot_bytes)
        self._free = list(range(self.num_slots))
        self._pending = []  # [(slots_tuple, event)]
        self._active = {}  # data_ptr -> slots_tuple

    def _allocate_backing(self, nbytes: int) -> torch.Tensor:
        """Allocate the pinned backing buffer.

        Kept as a method so tests can substitute a plain CPU buffer (real
        pin_memory is unavailable on CPU-only builds).
        """
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

    def _reclaim(self) -> None:
        remaining = []
        for slots, event in self._pending:
            try:
                done = event.query()
            except Exception:
                done = False
            if done:
                self._free.extend(slots)
                self._free.sort()
            else:
                remaining.append((slots, event))
        self._pending = remaining

    def _find_run(self, k: int) -> int | None:
        free = self._free
        for i in range(len(free) - k + 1):
            if free[i + k - 1] - free[i] == k - 1:
                return free[i]
        return None

    def alloc(self, nbytes: int, dtype: torch.dtype) -> tuple[torch.Tensor, int] | None:
        """Allocate a slice of ``nbytes`` bytes viewed as ``dtype``, or None.

        Returns ``(view, base_ptr)``; ``base_ptr`` identifies the slice for
        :func:`release_after_enqueue`. Returns None when the request is
        empty, exceeds the pool, is not element-aligned, or no contiguous
        slot run fits it (the caller falls back to direct allocation).
        """
        if nbytes <= 0 or nbytes > self.num_slots * self.slot_bytes:
            return None
        esz = torch.empty((), dtype=dtype).element_size()
        if nbytes % esz != 0:
            return None
        with _LOCK:
            self._reclaim()
            k = (nbytes + self.slot_bytes - 1) // self.slot_bytes
            start = self._find_run(k)
            if start is None:
                return None
            slots = tuple(range(start, start + k))
            for s in slots:
                self._free.remove(s)
            offset = start * self.slot_bytes
            ptr = self._buf.data_ptr() + offset
            flat = self._buf[offset : offset + nbytes].view(dtype)
            self._active[ptr] = slots
        return flat, ptr

    def flush_if_active(self, ptr: int) -> None:
        """Move the slice backing ``ptr`` into event-gated recycling.

        The caller records the event AFTER its H2D copy has been enqueued,
        so a slot is never reused while its copy is still in flight.
        """
        global _WARNED_NO_EVENT
        with _LOCK:
            slots = self._active.pop(ptr, None)
            if slots is None:
                return
            event = _new_pinned_event()
            if event is None:
                if not _WARNED_NO_EVENT:
                    logger.warning(
                        "no accelerator event API available; the mm staging slice stays active (safe, bounded)"
                    )
                    _WARNED_NO_EVENT = True
                self._active[ptr] = slots
                return
            event.record()
            self._pending.append((slots, event))


def _resolve_pool_mb() -> tuple[int, int]:
    """Resolve ``(pinned_pool_mb, slot_mb)`` from additional_config.

    Falls back to the documented defaults when the Ascend config is not
    initialized in this process: config-unavailability must not disable a
    crash fix. An explicit ``pinned_pool_mb=0`` is honored whenever the
    config is readable.
    """
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        cfg = get_ascend_config().mm_staging_config
        return cfg.pinned_pool_mb, cfg.slot_mb
    except Exception:
        return DEFAULT_POOL_MB, DEFAULT_SLOT_MB


def get_pool() -> MMPinnedStagingPool | None:
    """Return the process-wide staging pool, or None when unavailable.

    The pool is created lazily on first use; a disabled config or a failed
    pinned allocation permanently degrades to the direct-allocation path.
    """
    global _POOL, _POOL_FAILED
    if _POOL is not None or _POOL_FAILED:
        return _POOL
    with _LOCK:
        if _POOL is not None or _POOL_FAILED:
            return _POOL
        pool_mb, slot_mb = _resolve_pool_mb()
        if pool_mb <= 0:
            logger.info("mm staging pool disabled (pinned_pool_mb=0)")
            _POOL_FAILED = True
            return None
        try:
            _POOL = MMPinnedStagingPool(pool_mb << 20, slot_mb << 20)
        except Exception as e:
            logger.warning("mm staging pool init failed, falling back to direct allocation: %s", e)
            _POOL_FAILED = True
            return None
        logger.info("mm staging pool initialized: %d slots x %d MB", _POOL.num_slots, slot_mb)
        return _POOL


def alloc_view(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor | None:
    """Allocate an uninitialized pinned pool slice with the given shape.

    Returns None (caller falls back to the original allocation) when the
    pool is disabled, not yet creatable, or no contiguous slot run fits
    the request. The slice is uninitialized pool memory: the caller must
    fully overwrite it (``copy_``, concat/stack with ``out=``, or an
    explicit ``zero_()``) before publishing it.
    """
    pool = get_pool()
    if pool is None:
        return None
    try:
        n = 1
        for d in shape:
            n *= d
        if n <= 0:
            return None
        nbytes = n * torch.empty((), dtype=dtype).element_size()
        alloced = pool.alloc(nbytes, dtype)
        if alloced is None:
            return None
        flat, _ptr = alloced
        return flat.view(tuple(shape))
    except Exception:
        return None


def release_after_enqueue(tensor: torch.Tensor) -> None:
    """Recycle the pool slice backing ``tensor`` (if any) after its H2D
    copy has been enqueued; no-op for non-pool tensors.

    Reads ``_POOL`` directly: the release path must never trigger pool
    creation.
    """
    pool = _POOL
    if pool is None:
        return
    with contextlib.suppress(Exception):
        pool.flush_if_active(tensor.data_ptr())
