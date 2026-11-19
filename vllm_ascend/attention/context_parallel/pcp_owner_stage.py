# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Stage a prefill's long-KV blocks from the owner slab into local scratch.

Direct peer reads are correct but expensive at the access granularity of
sparse attention (one 1152-byte MLA row per selected position): measured peer
row-gather bandwidth is one third to one quarter of local HBM, while a 540 KB
block copy still reaches half.  Staging replaces the kernel's scattered peer
reads with one bulk copy per layer and keeps the attention kernel entirely on
local memory.

The mapping is dense per request: block ``i`` of request ``r`` lands in
``scratch[offset[r] + i]`` and ``block_table[r, i]`` is rewritten to
``offset[r] + i``.  Positions stay unchanged because they are resolved through
the block table by the kernel, so ``cmp_sparse_indices`` needs no remap.  The
copy is a plain ``index_select`` over the slab; with a full-VA owner mapping
the kernel reads local and peer pages through one logical gather.

Overflow is fail-open: when the selected blocks do not fit the bounded
scratch, the caller keeps the original table and the kernel falls back to
direct peer reads.  Results are identical either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.logger import logger

from vllm_ascend import envs

DEFAULT_STAGE_BUDGET_BYTES = 1024 * 1024 * 1024
DEFAULT_SWA_CHUNK_STAGE_BUDGET_BYTES = 256 * 1024 * 1024
DEFAULT_INDEX_STAGE_BUDGET_BYTES = 128 * 1024 * 1024


def _env_flag(value: object) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def owner_stage_enabled() -> bool:
    return _env_flag(envs.VLLM_ASCEND_PCP_OWNER_STAGE)


def owner_cpu_plan_enabled() -> bool:
    """Use the host-side block counts (sync-free) instead of mask indexing.

    Off by default: a 1000-request run with this path hung one worker at the
    tail (``shm_broadcast`` 60 s stall), so it needs root-causing before it is
    trusted.  The mask path has completed multiple 1000/1000 runs.
    """
    return _env_flag(envs.VLLM_ASCEND_PCP_OWNER_STAGE_CPUPLAN)


def owner_stage_swa_enabled() -> bool:
    """Stage the sliding-window cache as well (off by default).

    Measured slower than staging the long-KV cache alone: the extra gather
    fixes the kernel's peer reads but the per-call plan build costs more than
    it saves until the plan build is sync-free.
    """
    return _env_flag(envs.VLLM_ASCEND_PCP_OWNER_STAGE_SWA)


def owner_stage_swa_chunk_enabled() -> bool:
    """Stage only each local chunk's visible SWA window (off by default).

    Each prefill request reaches this rank as two virtual segments (one per
    DualChunkSwap chunk), so the kernel only ever reads
    ``[chunk_start - sliding_window, chunk_end)`` per segment. Copying that
    bounded range instead of the whole live prefix keeps the fabric movement
    proportional to ``local_tokens + 2 * window``.
    """
    return _env_flag(envs.VLLM_ASCEND_PCP_OWNER_STAGE_SWA_CHUNK)


def swa_chunk_debug_enabled() -> bool:
    """Print the rank-local chunk layout once, for assumption checks."""
    return _env_flag(envs.VLLM_ASCEND_PCP_OWNER_STAGE_SWA_CHUNK_DEBUG)


def owner_stage_index_enabled() -> bool:
    """Stage a pure prefill's index pages instead of reading peer pages.

    The indexer K plane shares the owner-sharded page with long KV, so the
    QLI's per-block gather pays the same one-byte-grain peer latency the long
    KV kernel did. Copying the live pages once per step and remapping the
    block table keeps the whole ``select`` on local HBM without the 1.2 GiB
    replicated copy that ``VLLM_ASCEND_PCP_LOCAL_INDEX`` needs.
    """
    return _env_flag(envs.VLLM_ASCEND_PCP_OWNER_STAGE_INDEX)


def _index_stage_budget_bytes() -> int:
    raw = envs.VLLM_ASCEND_PCP_OWNER_STAGE_INDEX_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_INDEX_STAGE_BUDGET_BYTES
    return value if value > 0 else DEFAULT_INDEX_STAGE_BUDGET_BYTES


def _swa_chunk_stage_budget_bytes() -> int:
    raw = envs.VLLM_ASCEND_PCP_OWNER_STAGE_SWA_CHUNK_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_SWA_CHUNK_STAGE_BUDGET_BYTES
    return value if value > 0 else DEFAULT_SWA_CHUNK_STAGE_BUDGET_BYTES


def owner_stage_reserve_bytes() -> int:
    """Scratch the enabled staging paths may allocate after the KV pool is sized.

    The staging caches allocate their budget lazily on the first staged step,
    so the KV planner has to keep this much HBM out of the block count when it
    turns owner-sharded savings into capacity.
    """
    total = 0
    if owner_stage_enabled() or owner_stage_swa_enabled():
        total += _stage_budget_bytes()
    if owner_stage_swa_chunk_enabled():
        total += _swa_chunk_stage_budget_bytes()
    if owner_stage_index_enabled():
        total += _index_stage_budget_bytes()
    return total


@dataclass(frozen=True)
class SwaChunkSegment:
    """Visible SWA rows for one local virtual segment (one PCP chunk).

    ``replay_start`` is block aligned and ``replay_start + replay_seq_len`` is
    the segment's exclusive global end position, so the operator's rebased
    positions keep the sliding-window mask unchanged.
    """

    replay_start: int
    replay_seq_len: int
    first_block: int
    num_blocks: int


def plan_swa_chunk_segments(
    query_start_loc_cpu,
    seq_lens_cpu,
    num_reqs: int,
    block_size: int,
    window: int,
) -> list[SwaChunkSegment] | None:
    """Derive one bounded segment per rank-local chunk row (pure CPU).

    The PCP manager already turns each prefill into rank-local chunk rows, so
    a row's query length is its chunk length and its ``seq_len`` is the global
    position of the chunk's exclusive end. This function has no torch
    dependency beyond accepting tensors/arrays with ``tolist``.

    Returns ``None`` (caller keeps the direct peer-read path) on any input the
    bounded view cannot represent exactly.
    """
    if num_reqs <= 0 or block_size <= 0 or window < 0:
        return None
    if query_start_loc_cpu is None or seq_lens_cpu is None:
        return None
    try:
        starts = query_start_loc_cpu.tolist() if hasattr(query_start_loc_cpu, "tolist") else query_start_loc_cpu
        seqs = seq_lens_cpu.tolist() if hasattr(seq_lens_cpu, "tolist") else seq_lens_cpu
        starts = [int(value) for value in starts[: num_reqs + 1]]
        seqs = [int(value) for value in seqs[:num_reqs]]
    except (TypeError, ValueError):
        return None
    if len(starts) < num_reqs + 1 or len(seqs) < num_reqs:
        return None

    segments: list[SwaChunkSegment] = []
    for row in range(num_reqs):
        chunk_len = starts[row + 1] - starts[row]
        chunk_end = seqs[row]
        if chunk_len <= 0 or chunk_end < chunk_len:
            # Padded graph rows (or an inconsistent CPU view) are not worth
            # guessing about: keep the unbounded path for this step.
            return None
        chunk_start = chunk_end - chunk_len
        replay_start = max(0, chunk_start - window)
        replay_start = (replay_start // block_size) * block_size
        replay_seq_len = chunk_end - replay_start
        first_block = replay_start // block_size
        num_blocks = (replay_seq_len + block_size - 1) // block_size
        if num_blocks <= 0:
            return None
        segments.append(
            SwaChunkSegment(
                replay_start=replay_start,
                replay_seq_len=replay_seq_len,
                first_block=first_block,
                num_blocks=num_blocks,
            )
        )
    return segments


def _stage_budget_bytes() -> int:
    raw = envs.VLLM_ASCEND_PCP_OWNER_STAGE_BYTES
    if not raw:
        return DEFAULT_STAGE_BUDGET_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_STAGE_BUDGET_BYTES
    return value if value > 0 else DEFAULT_STAGE_BUDGET_BYTES


def build_stage_tables_fixed(
    block_table: torch.Tensor, counts: list[int], num_blocks: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sync-free variant: per-request live-block counts are known on the host.

    Uses ``cat`` over fixed-length slices and ``torch.where`` instead of
    boolean-mask indexing, so no device-to-host synchronization happens.  The
    counts may over-approximate the live prefix (padding slots are copied from
    block 0 and are never referenced by the kernel, which bounds positions by
    the sequence length).
    """
    rows, max_blocks = block_table.shape
    device = block_table.device
    counts_t = torch.tensor(counts[:rows], dtype=torch.int64, device=device)
    positions = torch.arange(max_blocks, dtype=torch.int64, device=device)
    valid = positions[None, :] < counts_t[:, None]
    base = torch.cumsum(counts_t, dim=0) - counts_t
    slots = base[:, None] + positions[None, :]
    staged = torch.where(valid, slots, torch.full_like(slots, -1)).to(block_table.dtype)
    # Padding slots may hold stale ids from an earlier step.  Clamp into the
    # block range so an over-approximated count can never issue an
    # out-of-bounds gather (a suspected cause of the 998/1000 worker hang).
    upper = int(num_blocks) - 1 if num_blocks else None
    flat = torch.cat(
        [
            (
                block_table[row, : int(counts[row])].to(torch.int64).clamp_min(0)
                if upper is None
                else block_table[row, : int(counts[row])].to(torch.int64).clamp(0, upper)
            )
            for row in range(rows)
        ]
    )
    return staged, flat, int(flat.numel())


def build_stage_tables(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor | None = None,
    block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dense stage tables for one batch.

    ``block_table`` is ``[num_reqs, max_blocks]``.  The scheduler zero-fills
    the unused tail instead of writing ``-1``, so ``seq_lens`` (already in
    compressed positions) bounds the live prefix; without it only negative
    entries are treated as padding.  Returns ``(staged_table,
    flat_source_blocks, per_request_slots)`` where ``staged_table`` has the
    same shape and maps each live entry to its slot in the scratch cache,
    ``flat_source_blocks`` lists the slab blocks to copy in slot order, and
    ``per_request_slots`` gives each request's slot count.
    """
    valid = block_table >= 0
    if seq_lens is not None and block_size:
        live = (seq_lens + block_size - 1) // block_size
        positions = torch.arange(block_table.shape[1], device=block_table.device)
        valid = valid & (positions[None, :] < live[:, None])
    counts = valid.sum(dim=1)
    offsets = torch.cumsum(counts, dim=0) - counts
    slot_in_request = torch.cumsum(valid.to(torch.int64), dim=1) - 1
    slots = slot_in_request + offsets[:, None]
    staged = torch.full_like(block_table, -1)
    staged[valid] = slots[valid].to(block_table.dtype)
    flat = block_table[valid].to(torch.int64)
    return staged, flat, counts


class PcpOwnerStageCache:
    """One bounded local scratch reused by every owner-sharded layer."""

    def __init__(self, device: torch.device, budget_bytes: int | None = None) -> None:
        self.device = device
        self._budget_bytes = budget_bytes
        self._scratch: torch.Tensor | None = None
        self.capacity_blocks = 0
        self.stage_calls = 0
        self.staged_blocks = 0
        self.overflow_fallbacks = 0

    def _ensure(self, source_cache: torch.Tensor) -> torch.Tensor:
        if self._scratch is not None:
            return self._scratch
        # The shared-slot KV cache is a strided view: a block occupies the
        # first half of a slot and the alias tensor the second half.  Copying
        # whole slots keeps the gather on a contiguous 2-D source, which is
        # two orders of magnitude faster than index_select over the strided
        # view.
        slot_elements = int(source_cache.stride(0))
        self.block_elements = int(source_cache[0].numel())
        self.slot_elements = slot_elements
        slot_bytes = slot_elements * source_cache.element_size()
        budget = self._budget_bytes if self._budget_bytes is not None else _stage_budget_bytes()
        capacity = max(1, budget // max(1, slot_bytes))
        self._scratch = torch.empty((capacity, slot_elements), dtype=source_cache.dtype, device=self.device)
        self.capacity_blocks = capacity
        logger.info_once(
            "PCP owner KV stage scratch: %.2f GiB (%d slots x %d B) on %s; source shape=%s stride=%s contiguous=%s",
            capacity * slot_bytes / (1024**3),
            capacity,
            slot_bytes,
            self.device,
            tuple(source_cache.shape),
            source_cache.stride(),
            source_cache.is_contiguous(),
        )
        return self._scratch

    def stage(
        self,
        source_cache: torch.Tensor,
        block_table: torch.Tensor,
        *,
        seq_lens: torch.Tensor | None = None,
        block_size: int | None = None,
        counts: list[int] | None = None,
        prepared: tuple[torch.Tensor, torch.Tensor, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return ``(local_cache, staged_table)`` or ``None`` when it overflows.

        ``prepared`` carries a ``(staged_table, flat_source_blocks, total)``
        tuple that the caller already built once for a whole forward (the
        per-chunk SWA plan).  Layers in one SWA group share the same rows, so
        rebuilding the dense table per layer would be pure overhead.
        """
        from .pcp_owner_timing import enter as timing_enter
        from .pcp_owner_timing import exit as timing_exit

        timing = timing_enter("stage_tables")
        if prepared is not None:
            staged_table, flat, total = prepared
        elif counts is not None:
            staged_table, flat, total = build_stage_tables_fixed(
                block_table, counts, num_blocks=int(source_cache.shape[0])
            )
        else:
            staged_table, flat, _ = build_stage_tables(block_table, seq_lens=seq_lens, block_size=block_size)
            total = int(flat.numel())
        timing_exit(timing)
        if total == 0:
            return None
        scratch = self._ensure(source_cache)
        if total > self.capacity_blocks:
            self.overflow_fallbacks += 1
            if self.overflow_fallbacks == 1:
                logger.warning_once(
                    "PCP owner KV stage overflow: %d blocks > %d capacity; "
                    "falling back to direct peer reads for this step.",
                    total,
                    self.capacity_blocks,
                )
            return None
        slot_rows = scratch[:total]
        timing = timing_enter("stage_copy")
        source_slots = torch.as_strided(
            source_cache,
            (source_cache.shape[0], self.slot_elements),
            (self.slot_elements, 1),
            source_cache.storage_offset(),
        )
        torch.index_select(source_slots, 0, flat, out=slot_rows)
        timing_exit(timing)
        # Present the staged blocks with the same strided layout as the slab.
        view = torch.as_strided(
            slot_rows,
            (total,) + tuple(source_cache.shape[1:]),
            (self.slot_elements,) + tuple(source_cache.stride()[1:]),
            0,
        )
        self.stage_calls += 1
        self.staged_blocks += total
        if self.stage_calls % 200 == 1:
            logger.info(
                "PCP owner KV stage active: calls=%d last=%d blocks (%.1f MiB), avg=%.1f blocks, capacity=%d",
                self.stage_calls,
                total,
                total * source_cache[0].numel() * source_cache.element_size() / (1024**2),
                self.staged_blocks / self.stage_calls,
                self.capacity_blocks,
            )
        return view, staged_table


class PcpPlaneStageCache:
    """Bounded scratch for one plane of an owner-sharded page.

    The indexer K plane shares the physical page with long KV, compressor state
    and SWA aliases. The attention stager copies whole pages because the kernel
    reads every long-KV row; the QLI reads only its own 8 KiB K plane (and a
    0.5 KiB scale plane), so copying the enclosing 135 KiB page would move more
    than an order of magnitude more peer data than the operator consumes. This
    view copies only ``shape[1:]`` of each selected block, which is dense in the
    layout produced by ``_adjust_dsv4_kv_layout``.
    """

    def __init__(self, device: torch.device, budget_bytes: int | None = None) -> None:
        self.device = device
        self._budget_bytes = budget_bytes
        self._scratch: torch.Tensor | None = None
        self.capacity_blocks = 0
        self.stage_calls = 0
        self.staged_blocks = 0
        self.overflow_fallbacks = 0

    def _ensure(self, source_cache: torch.Tensor) -> torch.Tensor:
        if self._scratch is not None:
            return self._scratch
        self.block_elements = int(source_cache[0].numel())
        block_bytes = self.block_elements * source_cache.element_size()
        budget = self._budget_bytes if self._budget_bytes is not None else _index_stage_budget_bytes()
        capacity = max(1, budget // max(1, block_bytes))
        self._scratch = torch.empty((capacity, self.block_elements), dtype=source_cache.dtype, device=self.device)
        self.capacity_blocks = capacity
        logger.info_once(
            "PCP index plane stage scratch: %.2f MiB (%d blocks x %d B) on %s; source shape=%s stride=%s",
            capacity * block_bytes / (1024**2),
            capacity,
            block_bytes,
            self.device,
            tuple(source_cache.shape),
            source_cache.stride(),
        )
        return self._scratch

    def stage(
        self, source_cache: torch.Tensor, prepared: tuple[torch.Tensor, torch.Tensor, int]
    ) -> torch.Tensor | None:
        """Return the staged plane view, or ``None`` when the batch overflows."""
        from .pcp_owner_timing import enter as timing_enter
        from .pcp_owner_timing import exit as timing_exit

        staged_table, flat, total = prepared
        if total == 0:
            return None
        scratch = self._ensure(source_cache)
        if total > self.capacity_blocks:
            self.overflow_fallbacks += 1
            if self.overflow_fallbacks == 1:
                logger.warning_once(
                    "PCP index stage overflow: %d blocks > %d capacity; "
                    "falling back to direct peer reads for this step.",
                    total,
                    self.capacity_blocks,
                )
            return None
        timing = timing_enter("stage_index_copy")
        source_rows = torch.as_strided(
            source_cache,
            (int(source_cache.shape[0]), self.block_elements),
            (int(source_cache.stride(0)), 1),
            source_cache.storage_offset(),
        )
        torch.index_select(source_rows, 0, flat, out=scratch[:total])
        timing_exit(timing)
        view = torch.as_strided(
            scratch[:total],
            (total,) + tuple(source_cache.shape[1:]),
            (self.block_elements,) + tuple(source_cache.stride()[1:]),
            0,
        )
        self.stage_calls += 1
        self.staged_blocks += total
        if self.stage_calls % 200 == 1:
            logger.info(
                "PCP index plane stage active: calls=%d last=%d blocks (%.2f MiB), avg=%.1f blocks, capacity=%d",
                self.stage_calls,
                total,
                total * self.block_elements * source_cache.element_size() / (1024**2),
                self.staged_blocks / self.stage_calls,
                self.capacity_blocks,
            )
        return view


class PcpPageStageCache:
    """Bounded scratch holding whole owner pages, sliced locally per plane.

    The indexer K, its scale twin and the folded candidate view live at
    different offsets of one owner page. Staging each plane separately issues
    nine small strided gathers per forward; staging the page once issues one
    contiguous gather, after which every plane is a zero-copy view at its
    original intra-page offset and block stride. The copy moves the whole page
    (3.2x more bytes than the index planes alone) but removes eight kernel
    launches and turns the source rows into full-page copies.
    """

    def __init__(self, device: torch.device, budget_bytes: int | None = None) -> None:
        self.device = device
        self._budget_bytes = budget_bytes
        self._scratch: torch.Tensor | None = None
        self._page_elements = 0
        self.capacity_blocks = 0
        self.stage_calls = 0
        self.staged_blocks = 0
        self.overflow_fallbacks = 0

    def _ensure(self, base: torch.Tensor) -> torch.Tensor:
        page_elements = int(base.stride(0))
        if self._scratch is not None and self._page_elements == page_elements:
            return self._scratch
        page_bytes = page_elements * base.element_size()
        budget = self._budget_bytes if self._budget_bytes is not None else _index_stage_budget_bytes()
        capacity = max(1, budget // max(1, page_bytes))
        self._scratch = torch.empty((capacity, page_elements), dtype=base.dtype, device=self.device)
        self._page_elements = page_elements
        self.capacity_blocks = capacity
        logger.info_once(
            "PCP index page stage scratch: %.2f MiB (%d pages x %d B) on %s",
            capacity * page_bytes / (1024**2),
            capacity,
            page_bytes,
            self.device,
        )
        return self._scratch

    def stage(
        self, base: torch.Tensor, views: tuple[torch.Tensor, ...], prepared: tuple[torch.Tensor, torch.Tensor, int]
    ) -> tuple[torch.Tensor, ...] | None:
        """Copy whole pages once and return each plane as a local view."""
        from .pcp_owner_timing import enter as timing_enter
        from .pcp_owner_timing import exit as timing_exit

        _, flat, total = prepared
        if total == 0:
            return None
        scratch = self._ensure(base)
        if total > self.capacity_blocks:
            self.overflow_fallbacks += 1
            if self.overflow_fallbacks == 1:
                logger.warning_once(
                    "PCP index page stage overflow: %d pages > %d capacity; "
                    "falling back to direct peer reads for this step.",
                    total,
                    self.capacity_blocks,
                )
            return None
        page_elements = self._page_elements
        page_bytes = page_elements * base.element_size()
        timing = timing_enter("stage_index_copy")
        # The slab starts at page 0 of its own storage; block b occupies
        # ``[b * page_bytes, (b + 1) * page_bytes)``.
        source_pages = torch.as_strided(base, (int(base.shape[0]), page_elements), (page_elements, 1), 0)
        torch.index_select(source_pages, 0, flat, out=scratch[:total])
        timing_exit(timing)
        staged = []
        for view in views:
            offset_bytes = int(view.storage_offset()) * view.element_size()
            block_bytes = int(view[0].numel()) * view.element_size()
            if offset_bytes < 0 or offset_bytes >= page_bytes:
                return None
            if offset_bytes + block_bytes > page_bytes:
                return None
            if view.element_size() and offset_bytes % view.element_size():
                return None
            if view.dtype == base.dtype:
                rows = scratch[:total]
                block_stride = page_elements
            else:
                if page_bytes % view.element_size():
                    return None
                rows = scratch[:total].view(view.dtype)
                block_stride = page_bytes // view.element_size()
            staged.append(
                torch.as_strided(
                    rows,
                    (total,) + tuple(view.shape[1:]),
                    (block_stride,) + tuple(view.stride()[1:]),
                    offset_bytes // view.element_size(),
                )
            )
        self.stage_calls += 1
        self.staged_blocks += total
        if self.stage_calls % 200 == 1:
            logger.info(
                "PCP index page stage active: calls=%d last=%d pages (%.2f MiB), avg=%.1f pages, capacity=%d",
                self.stage_calls,
                total,
                total * page_bytes / (1024**2),
                self.staged_blocks / self.stage_calls,
                self.capacity_blocks,
            )
        return tuple(staged)


_stage_caches: dict[tuple, PcpOwnerStageCache | PcpPlaneStageCache | PcpPageStageCache] = {}


def get_stage_cache(device: torch.device, source_cache: torch.Tensor) -> PcpOwnerStageCache:
    """One scratch per (device, dtype, per-block shape, page stride).

    Layers with different compression ratios expose different block shapes;
    sharing one buffer across them would incorrectly size the gather.  The page stride
    is part of the key because tuple aliases pad some planes to a larger page
    (147712 vs 131072 B); a scratch sized for the smaller stride cannot index
    the padded view.
    """
    key = (
        str(device),
        str(source_cache.dtype),
        tuple(source_cache.shape[1:]),
        int(source_cache.stride(0)),
    )
    cache = _stage_caches.get(key)
    if cache is None:
        cache = PcpOwnerStageCache(device)
        _stage_caches[key] = cache
    return cache


def get_index_page_stage_cache(device: torch.device, page_bytes: int) -> PcpPageStageCache:
    """Scratch for whole owner pages; one per page layout (128 / 144 KiB)."""
    key = ("index_page", str(device), int(page_bytes))
    cache = _stage_caches.get(key)
    if cache is None:
        cache = PcpPageStageCache(device)
        _stage_caches[key] = cache
    return cache


def get_index_plane_stage_cache(device: torch.device, source_cache: torch.Tensor) -> PcpPlaneStageCache:
    """Scratch for one indexer plane; K and scale have distinct shapes."""
    key = (
        "index_plane",
        str(device),
        str(source_cache.dtype),
        tuple(source_cache.shape[1:]),
        int(source_cache.stride(0)),
        tuple(source_cache.stride()[1:]),
    )
    cache = _stage_caches.get(key)
    if cache is None:
        cache = PcpPlaneStageCache(device)
        _stage_caches[key] = cache
    return cache


def get_swa_chunk_stage_cache(device: torch.device, source_cache: torch.Tensor) -> PcpOwnerStageCache:
    """Bounded scratch dedicated to per-chunk SWA staging.

    The chunk view only needs ``local_tokens + 2 * window`` rows per layer, so
    it must not inherit the 1 GiB whole-prefix budget.  Kept separate from
    ``get_stage_cache`` so tuning one path cannot starve the other.
    """
    key = (
        "swa_chunk",
        str(device),
        str(source_cache.dtype),
        tuple(source_cache.shape[1:]),
        int(source_cache.stride(0)),
    )
    cache = _stage_caches.get(key)
    if cache is None:
        cache = PcpOwnerStageCache(device, budget_bytes=_swa_chunk_stage_budget_bytes())
        _stage_caches[key] = cache
    return cache
