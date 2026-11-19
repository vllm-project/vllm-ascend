# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Owner-sharded PCP KV placement and bounded consumer staging.

M2 stores each KV byte once per PCP group.  The device side lives in
:mod:`pcp_owner_vmm`: every rank keeps a complete virtual address range over a
slab and only ``1 / pcp_size`` of its physical two-megabyte chunks, so a stock
attention kernel reads and writes the full logical cache without knowing which
rank owns a page.  Physical ownership is therefore decoupled from the compute
owner.

:class:`PcpOwnerLayout` describes the compute/write owner of one request's
pages.  MRV2 PCP cuts a prefill into ``2 * pcp_size`` chunks (rank ``r`` owns
chunks ``r`` and ``2P-1-r``); short prefills and decodes stay replicated.  A
batch can mix both, so the layout is per request.  The stage-cache utilities
below are the transport-independent consumer contract: they materialize only
the pages sparse attention selects (top-k long-KV pages plus the current
step's pages), never the whole block table, and remap the sparse indices and
block table into that bounded view.  When the slab is a full-VA mapping,
staging reads local and peer pages with one logical gather because the VA is
complete on every rank.

No code in this module performs a collective, a device synchronization, or a
request-time allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from vllm.logger import logger

PCP_KV_SHARING_REPLICATED = "replicated"
PCP_KV_SHARING_OWNER = "owner"
PCP_KV_SHARING_MODES = (PCP_KV_SHARING_REPLICATED, PCP_KV_SHARING_OWNER)

OWNER_PLACEMENT_MODULO = "modulo"
OWNER_PLACEMENT_CONTIGUOUS = "contiguous"
# Match the MRV2 PCP token split: query_len is cut into 2 * pcp_size chunks and
# rank r owns chunk r and chunk 2 * pcp_size - 1 - r.
OWNER_PLACEMENT_CHUNK = "chunk"
OWNER_PLACEMENTS = (OWNER_PLACEMENT_MODULO, OWNER_PLACEMENT_CONTIGUOUS, OWNER_PLACEMENT_CHUNK)
DEFAULT_PCP_CHUNKS_PER_RANK = 2


def get_pcp_kv_sharing_mode() -> str:
    """Return the requested PCP KV sharing mode (default: replicated)."""
    from vllm_ascend import envs

    return normalize_pcp_kv_sharing_mode(envs.VLLM_ASCEND_PCP_KV_SHARING)


def normalize_pcp_kv_sharing_mode(raw: str | None) -> str:
    mode = (raw or PCP_KV_SHARING_REPLICATED).strip().lower()
    if mode not in PCP_KV_SHARING_MODES:
        raise ValueError(f"pcp_kv_sharing must be one of {PCP_KV_SHARING_MODES}, got {raw!r}")
    return mode


def pcp_kv_sharing_mode_from_config(vllm_config) -> str:
    """Read ``additional_config.pcp_kv_sharing``, then the env override."""
    additional = getattr(vllm_config, "additional_config", None) or {}
    if "pcp_kv_sharing" in additional:
        return normalize_pcp_kv_sharing_mode(additional["pcp_kv_sharing"])
    return get_pcp_kv_sharing_mode()


def owner_sharding_enabled(vllm_config=None) -> bool:
    if vllm_config is None:
        return get_pcp_kv_sharing_mode() == PCP_KV_SHARING_OWNER
    return pcp_kv_sharing_mode_from_config(vllm_config) == PCP_KV_SHARING_OWNER


def pcp_request_is_replicated(
    *,
    num_query_tokens: int,
    pcp_size: int,
    dcp_size: int,
    is_prefilling: bool,
    chunks_per_rank: int = DEFAULT_PCP_CHUNKS_PER_RANK,
) -> bool:
    """Mirror ``PcpManager.replicated_requests`` for one request.

    Decodes are replicated.  A prefill that cannot fill all
    ``chunks_per_rank * pcp_size`` chunks is also replicated, but only when
    ``dcp_size > 1``; the DCP=1 lane still chunks short prefills, leaving some
    ranks with an empty chunk.
    """
    if not is_prefilling:
        return True
    if dcp_size <= 1:
        return False
    num_chunks = chunks_per_rank * pcp_size
    chunk_size = (num_query_tokens + num_chunks - 1) // num_chunks
    return (num_chunks - 1) * chunk_size >= num_query_tokens


@dataclass(frozen=True)
class PcpOwnerLayout:
    """Owner of every KV page of **one request**.

    MRV2 PCP is not a single global page placement.  ``pcp_manager`` cuts a
    chunked prefill into ``2 * pcp_size`` parts and rank ``r`` owns chunks
    ``r`` and ``2 * pcp_size - 1 - r``; a short prefill that cannot fill its
    chunks, and every decode request, stays replicated.  A batch can contain
    both kinds, so this layout is constructed per request and applied to that
    request's page ids (its block-table slice).

    ``OWNER_PLACEMENT_CHUNK`` models the MRV2 split.  Chunk boundaries must be
    page aligned; use :meth:`for_pcp_request`, which rounds the chunk size up
    to a whole storage page, and make the scheduler use that rounded size.
    ``modulo`` and ``contiguous`` remain available for local replay/tests.
    """

    num_pages: int
    pcp_size: int
    placement: str = OWNER_PLACEMENT_MODULO
    chunk_pages: int | None = None
    replicated: bool = False
    chunks_per_rank: int = DEFAULT_PCP_CHUNKS_PER_RANK

    def __post_init__(self) -> None:
        if self.num_pages < 0:
            raise ValueError(f"num_pages must be non-negative, got {self.num_pages}")
        if self.pcp_size <= 0:
            raise ValueError(f"pcp_size must be positive, got {self.pcp_size}")
        if self.chunks_per_rank <= 0:
            raise ValueError(f"chunks_per_rank must be positive, got {self.chunks_per_rank}")
        if self.placement not in OWNER_PLACEMENTS:
            raise ValueError(f"placement must be one of {OWNER_PLACEMENTS}, got {self.placement!r}")
        if self.chunk_pages is not None:
            if self.placement != OWNER_PLACEMENT_CHUNK:
                raise ValueError("chunk_pages requires OWNER_PLACEMENT_CHUNK")
            if self.chunk_pages < 0:
                raise ValueError(f"chunk_pages must be non-negative, got {self.chunk_pages}")

    @classmethod
    def for_pcp_request(
        cls,
        *,
        num_query_tokens: int,
        tokens_per_page: int,
        pcp_size: int,
        replicated: bool = False,
        chunks_per_rank: int = DEFAULT_PCP_CHUNKS_PER_RANK,
    ) -> PcpOwnerLayout:
        """Build the owner layout of one request from its query length.

        ``tokens_per_page`` is ``block_size * tokens_per_state`` of the cache
        plane.  The chunk size is ``ceil(query_len / (chunks_per_rank *
        pcp_size))`` rounded up to a whole page; callers that split the
        prefill must use that rounded chunk size so no page straddles two
        owners.  ``replicated`` covers short prefills and decode.
        """
        if num_query_tokens < 0:
            raise ValueError(f"num_query_tokens must be non-negative, got {num_query_tokens}")
        if tokens_per_page <= 0:
            raise ValueError(f"tokens_per_page must be positive, got {tokens_per_page}")
        if chunks_per_rank <= 0:
            raise ValueError(f"chunks_per_rank must be positive, got {chunks_per_rank}")
        num_pages = (num_query_tokens + tokens_per_page - 1) // tokens_per_page
        if replicated:
            return cls(
                num_pages=num_pages,
                pcp_size=pcp_size,
                placement=OWNER_PLACEMENT_CHUNK,
                replicated=True,
                chunks_per_rank=chunks_per_rank,
            )
        chunk_count = chunks_per_rank * pcp_size
        chunk_tokens = (num_query_tokens + chunk_count - 1) // chunk_count
        chunk_pages = (chunk_tokens + tokens_per_page - 1) // tokens_per_page
        return cls(
            num_pages=num_pages,
            pcp_size=pcp_size,
            placement=OWNER_PLACEMENT_CHUNK,
            chunk_pages=chunk_pages,
            chunks_per_rank=chunks_per_rank,
        )

    @property
    def chunk_count(self) -> int:
        return self.chunks_per_rank * self.pcp_size

    @property
    def resolved_chunk_pages(self) -> int:
        if self.chunk_pages is not None:
            return self.chunk_pages
        if self.num_pages == 0:
            return 0
        return (self.num_pages + self.chunk_count - 1) // self.chunk_count

    def owner_of_chunk(self, chunk_index: int) -> int:
        """MRV2 mapping: rank r owns chunk r and chunk ``2P - 1 - r``."""
        if not 0 <= chunk_index < self.chunk_count:
            raise ValueError(f"chunk_index={chunk_index} is outside {self.chunk_count} chunks")
        if chunk_index < self.pcp_size:
            return chunk_index
        return 2 * self.pcp_size - 1 - chunk_index

    def chunks_of_rank(self, rank: int) -> tuple[int, int]:
        if not 0 <= rank < self.pcp_size:
            raise ValueError(f"rank={rank} is outside PCP size {self.pcp_size}")
        return rank, 2 * self.pcp_size - 1 - rank

    def pages_in_chunk(self, chunk_index: int) -> int:
        chunk_pages = self.resolved_chunk_pages
        if chunk_pages == 0:
            return 0
        return max(0, min(chunk_pages, self.num_pages - chunk_index * chunk_pages))

    def local_page_count(self, rank: int) -> int:
        if self.replicated:
            return self.num_pages
        first, second = self.chunks_of_rank(rank)
        return self.pages_in_chunk(first) + self.pages_in_chunk(second)

    @property
    def num_local_pages(self) -> int:
        if self.replicated:
            return self.num_pages
        if self.placement != OWNER_PLACEMENT_CHUNK:
            return (self.num_pages + self.pcp_size - 1) // self.pcp_size
        return max(self.local_page_count(rank) for rank in range(self.pcp_size))

    def _chunked_owner_of(self, page_ids: torch.Tensor) -> torch.Tensor:
        chunk_pages = self.resolved_chunk_pages
        if chunk_pages == 0:
            return torch.zeros_like(page_ids)
        chunk_index = torch.clamp(
            torch.div(page_ids, chunk_pages, rounding_mode="floor"),
            max=self.chunk_count - 1,
        )
        first_half = chunk_index < self.pcp_size
        return torch.where(first_half, chunk_index, 2 * self.pcp_size - 1 - chunk_index)

    def owner_of(self, page_ids: torch.Tensor) -> torch.Tensor:
        page_ids = page_ids.to(torch.int64)
        if self.replicated:
            # -1 means "this request is not owner-sharded"; every rank keeps
            # a complete copy and callers must not filter writes by owner.
            return torch.full_like(page_ids, -1)
        if self.placement == OWNER_PLACEMENT_MODULO:
            return torch.remainder(page_ids, self.pcp_size)
        if self.placement == OWNER_PLACEMENT_CHUNK:
            return self._chunked_owner_of(page_ids)
        # Contiguous: the first ceil(N/P) pages belong to rank 0, and so on.
        if self.num_local_pages == 0:
            return torch.zeros_like(page_ids)
        return torch.clamp(
            torch.div(page_ids, self.num_local_pages, rounding_mode="floor"),
            max=self.pcp_size - 1,
        )

    def local_page(self, page_ids: torch.Tensor) -> torch.Tensor:
        page_ids = page_ids.to(torch.int64)
        if self.replicated:
            raise ValueError("a replicated request has no owner-local page index")
        if self.placement == OWNER_PLACEMENT_MODULO:
            return torch.div(page_ids, self.pcp_size, rounding_mode="floor")
        if self.placement == OWNER_PLACEMENT_CHUNK:
            chunk_pages = self.resolved_chunk_pages
            if chunk_pages == 0:
                return torch.zeros_like(page_ids)
            chunk_index = torch.div(page_ids, chunk_pages, rounding_mode="floor")
            offset = page_ids - chunk_index * chunk_pages
            # Chunk c < P is the first chunk of its owner, so its local index
            # starts at zero.  Chunk c >= P is the owner's second chunk and
            # starts after the owner's first chunk.
            first_size = torch.tensor(
                [self.pages_in_chunk(index) for index in range(self.pcp_size)],
                dtype=torch.int64,
                device=page_ids.device,
            )
            owner = self.owner_of(page_ids)
            base = torch.where(
                chunk_index < self.pcp_size,
                torch.zeros_like(offset),
                first_size[owner.clamp_min(0)],
            )
            return base + offset
        if self.num_local_pages == 0:
            return torch.zeros_like(page_ids)
        return torch.remainder(page_ids, self.num_local_pages)

    def logical_page(self, owner_rank: int, local_page_ids: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`local_page` for one owner rank."""
        if self.replicated:
            raise ValueError("a replicated request has no owner-local page index")
        local = local_page_ids.to(torch.int64)
        if not 0 <= owner_rank < self.pcp_size:
            raise ValueError(f"owner_rank={owner_rank} is outside PCP size {self.pcp_size}")
        if self.placement == OWNER_PLACEMENT_MODULO:
            return local * self.pcp_size + owner_rank
        if self.placement == OWNER_PLACEMENT_CONTIGUOUS:
            return local + owner_rank * self.num_local_pages
        chunk_pages = self.resolved_chunk_pages
        if chunk_pages == 0:
            return local
        first_chunk, second_chunk = self.chunks_of_rank(owner_rank)
        first_size = self.pages_in_chunk(first_chunk)
        in_first = local < first_size
        return torch.where(
            in_first,
            first_chunk * chunk_pages + local,
            second_chunk * chunk_pages + (local - first_size),
        )

    def owned_mask(self, page_ids: torch.Tensor, rank: int) -> torch.Tensor:
        if not 0 <= rank < self.pcp_size:
            raise ValueError(f"rank={rank} is outside PCP size {self.pcp_size}")
        if self.replicated:
            return torch.zeros_like(page_ids, dtype=torch.bool)
        valid = page_ids >= 0
        return valid & (self.owner_of(page_ids.clamp_min(0)) == rank)


def page_ids_from_slot_mapping(slot_mapping: torch.Tensor, storage_block_size: int) -> torch.Tensor:
    """Return logical page ids from either a ``[rows, 2]`` or flat mapping.

    ``[rows, 2]`` mappings carry ``(page, offset)`` coordinates.  Flat mappings
    address a physical slot; their page is ``slot // storage_block_size``.
    Negative entries are padding and stay negative so callers can mask them.
    """
    if storage_block_size <= 0:
        raise ValueError(f"storage_block_size must be positive, got {storage_block_size}")
    slots = torch.as_tensor(slot_mapping)
    if slots.ndim == 2:
        if slots.shape[-1] != 2:
            raise ValueError(f"2-D slot mapping must have two columns, got {tuple(slots.shape)}")
        return slots[:, 0]
    if slots.ndim != 1:
        raise ValueError(f"slot mapping must be 1-D or 2-D, got {tuple(slots.shape)}")
    return torch.where(slots >= 0, torch.div(slots, storage_block_size, rounding_mode="floor"), slots)


def remap_block_table_to_stage(block_table: torch.Tensor, staged_pages: torch.Tensor) -> torch.Tensor:
    """Map logical pages in ``block_table`` to their index in ``staged_pages``.

    ``staged_pages`` must be sorted and unique.  Padding (negative) entries stay
    padding.  A valid page absent from the staged set is an error: silently
    feeding sparse attention another request's page is a correctness bug, not a
    capacity fallback.
    """
    if staged_pages.ndim != 1:
        raise ValueError("staged_pages must be one-dimensional")
    staged_pages = staged_pages.to(torch.int64)
    if staged_pages.numel() and not bool(torch.all(staged_pages[1:] > staged_pages[:-1])):
        raise ValueError("staged_pages must be sorted and unique")

    result = torch.full_like(block_table, -1)
    valid = block_table >= 0
    if not bool(valid.any()):
        return result
    if staged_pages.numel() == 0:
        raise ValueError("block table references pages but the staged set is empty")

    selected = block_table[valid].to(torch.int64)
    indices = torch.searchsorted(staged_pages, selected)
    in_range = indices < staged_pages.numel()
    matched = torch.zeros_like(in_range, dtype=torch.bool)
    if bool(in_range.any()):
        matched[in_range] = staged_pages[indices[in_range]] == selected[in_range]
    if not bool(matched.all()):
        missing = selected[~matched][:8].tolist()
        raise ValueError(f"block table references pages absent from the staged set: {missing}")
    result[valid] = indices.to(result.dtype)
    return result


def remap_indices_to_stage(indices: torch.Tensor, staged_pages: torch.Tensor) -> torch.Tensor:
    """Map top-k logical page indices into the staged view.

    Negative entries remain -1.  Unlike :func:`remap_block_table_to_stage` a
    missing page is allowed to become -1, because sparse top-k selections may
    contain padding/unselected sentinels that the kernel skips.
    """
    if staged_pages.ndim != 1:
        raise ValueError("staged_pages must be one-dimensional")
    staged_pages = staged_pages.to(torch.int64)
    result = torch.full_like(indices, -1)
    valid = indices >= 0
    if not bool(valid.any()):
        return result
    if staged_pages.numel() == 0:
        return result
    selected = indices[valid].to(torch.int64)
    positions = torch.searchsorted(staged_pages, selected)
    in_range = positions < staged_pages.numel()
    matched = torch.zeros_like(in_range, dtype=torch.bool)
    if bool(in_range.any()):
        matched[in_range] = staged_pages[positions[in_range]] == selected[in_range]
    result[valid] = torch.where(matched, positions, torch.full_like(positions, -1)).to(result.dtype)
    return result


def stage_pages_from_sparse_selection(
    selected_pages: torch.Tensor,
    current_pages: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the sorted unique pages one attention call may dereference.

    ``selected_pages`` is the sparse selection (top-k long-KV pages), not the
    full block table.  A 1M-token context at block size 128 has 8192 pages,
    while the bounded scratch holds tens; feeding every block-table page would
    overflow it on the first long request.  ``current_pages`` adds pages that
    the running step writes so they can be overlaid into the same view.
    Negative entries are padding and are dropped.
    """
    if selected_pages.ndim > 2:
        raise ValueError(f"selected_pages must be 1-D or 2-D, got {tuple(selected_pages.shape)}")
    pages = selected_pages.reshape(-1).to(torch.int64)
    if current_pages is not None:
        current = current_pages.to(torch.int64)
        if current.ndim == 2:
            if current.shape[-1] != 2:
                raise ValueError("current_pages must be page ids or a [rows, 2] slot mapping")
            current = current[:, 0]
        elif current.ndim != 1:
            raise ValueError("current_pages must be page ids or a [rows, 2] slot mapping")
        pages = torch.cat((pages, current.reshape(-1)))
    return torch.unique(pages[pages >= 0], sorted=True)


@runtime_checkable
class PeerPageReader(Protocol):
    """Transport that copies owner pages into a local destination tensor."""

    def copy_pages(
        self,
        owner_rank: int,
        owner_page_ids: torch.Tensor,
        destination: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> None: ...


class VmmPeerPageReader:
    """Page reader over a non-owning VMM alias of the complete slab.

    ``page_view`` is a ``(num_pages, *page_shape)`` view into the PCP-wide
    owner slab.  Peer pages are already mapped into this rank's virtual
    address space at their logical offsets, so reading them is an ordinary
    indexed device copy: no HCCL, no device-to-host transfer, and no import or
    map call on the request path.  The reader only translates the compact
    per-owner page index used by :class:`BoundedStageCache` back to the
    logical page id.
    """

    def __init__(self, page_view: torch.Tensor, layout: PcpOwnerLayout) -> None:
        if page_view.ndim < 2:
            # A 2-D view is (pages, bytes); higher ranks carry page structure.
            raise ValueError(f"page_view must be at least 2-D, got {tuple(page_view.shape)}")
        if page_view.shape[0] < layout.num_pages:
            raise ValueError(f"page_view has {page_view.shape[0]} pages, layout needs {layout.num_pages}")
        self._page_view = page_view
        self._layout = layout

    @property
    def layout(self) -> PcpOwnerLayout:
        return self._layout

    def _logical_pages(self, owner_rank: int, owner_page_ids: torch.Tensor) -> torch.Tensor:
        if not 0 <= owner_rank < self._layout.pcp_size:
            raise ValueError(f"owner_rank={owner_rank} is outside PCP size {self._layout.pcp_size}")
        if owner_page_ids.ndim != 1:
            raise ValueError("owner_page_ids must be one-dimensional")
        ids = owner_page_ids.to(torch.int64)
        if ids.numel() == 0:
            return ids
        if bool((ids < 0).any()):
            raise ValueError("owner_page_ids must be non-negative compact owner indices")
        if self._layout.replicated:
            raise ValueError("a replicated request has no owner-local pages to read")
        if int(ids.max()) >= self._layout.local_page_count(owner_rank):
            raise ValueError(
                f"owner page {int(ids.max())} is outside rank {owner_rank}'s "
                f"{self._layout.local_page_count(owner_rank)} local pages"
            )
        logical = self._layout.logical_page(owner_rank, ids)
        if logical.numel() and int(logical.max()) >= self._layout.num_pages:
            raise ValueError(
                f"owner page {int(ids.max())} of rank {owner_rank} maps past the last logical page "
                f"{self._layout.num_pages - 1}"
            )
        return logical

    def copy_pages(
        self,
        owner_rank: int,
        owner_page_ids: torch.Tensor,
        destination: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> None:
        """Copy the owner's compact pages into ``destination``."""
        del non_blocking  # both endpoints are device tensors on the current stream
        logical = self._logical_pages(owner_rank, owner_page_ids)
        if destination.shape[0] != logical.numel():
            raise ValueError(f"page count mismatch: ids={logical.numel()}, destination={destination.shape[0]}")
        if tuple(destination.shape[1:]) != tuple(self._page_view.shape[1:]):
            raise ValueError(
                f"destination page shape {tuple(destination.shape[1:])} != "
                f"page_view shape {tuple(self._page_view.shape[1:])}"
            )
        torch.index_select(self._page_view, 0, logical, out=destination)

    def write_pages(
        self,
        owner_rank: int,
        owner_page_ids: torch.Tensor,
        source: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> None:
        """Persist the owner's compact pages from ``source`` (inverse copy)."""
        del non_blocking
        logical = self._logical_pages(owner_rank, owner_page_ids)
        if source.shape[0] != logical.numel():
            raise ValueError(f"page count mismatch: ids={logical.numel()}, source={source.shape[0]}")
        if tuple(source.shape[1:]) != tuple(self._page_view.shape[1:]):
            raise ValueError(
                f"source page shape {tuple(source.shape[1:])} != page_view shape {tuple(self._page_view.shape[1:])}"
            )
        self._page_view.index_copy_(0, logical, source)


class TensorPeerPageReader:
    """In-process reader over per-owner shards (tests and local replay)."""

    def __init__(self, owner_shards: list[torch.Tensor]) -> None:
        if not owner_shards:
            raise ValueError("owner_shards must not be empty")
        self._shards = owner_shards

    @property
    def pcp_size(self) -> int:
        return len(self._shards)

    def copy_pages(
        self,
        owner_rank: int,
        owner_page_ids: torch.Tensor,
        destination: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> None:
        if not 0 <= owner_rank < len(self._shards):
            raise ValueError(f"owner_rank={owner_rank} is outside {len(self._shards)} shards")
        if owner_page_ids.ndim != 1:
            raise ValueError("owner_page_ids must be one-dimensional")
        if owner_page_ids.numel() != destination.shape[0]:
            raise ValueError(f"page count mismatch: ids={owner_page_ids.numel()}, destination={destination.shape[0]}")
        torch.index_select(self._shards[owner_rank], 0, owner_page_ids.long(), out=destination)


@dataclass
class StageCounters:
    """Request-time staging counters used by capacity and no-copy gates."""

    materializations: int = 0
    pages_materialized: int = 0
    peer_pages_materialized: int = 0
    local_pages_materialized: int = 0
    current_rows_overlaid: int = 0
    peak_pages: int = 0

    def reset(self) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, 0)


@dataclass
class StagedKvView:
    """The local attention view of one consumer call."""

    cache: torch.Tensor
    block_table: torch.Tensor
    staged_pages: torch.Tensor
    peer_pages: int
    local_pages: int
    counters: StageCounters = field(repr=False, default_factory=StageCounters)


class BoundedStageCache:
    """Bounded local scratch holding only the pages referenced this step."""

    def __init__(self, page_shape: tuple[int, ...], capacity_pages: int, dtype: torch.dtype, device=None) -> None:
        if capacity_pages <= 0:
            raise ValueError(f"capacity_pages must be positive, got {capacity_pages}")
        if any(dim <= 0 for dim in page_shape):
            raise ValueError(f"page_shape must be positive, got {page_shape}")
        self.capacity_pages = int(capacity_pages)
        self.cache = torch.empty((capacity_pages, *page_shape), dtype=dtype, device=device)
        self.counters = StageCounters()
        self._local_shard: torch.Tensor | None = None
        self._logical_source: torch.Tensor | None = None

    def materialize(
        self,
        required_pages: torch.Tensor,
        *,
        layout: PcpOwnerLayout,
        rank: int,
        reader: PeerPageReader,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        """Copy the sorted-unique ``required_pages`` into the local scratch.

        Returns the logical pages in stage order.  Copies are issued per owner
        so a VMM reader can batch one contiguous peer range at a time.
        """
        if not 0 <= rank < layout.pcp_size:
            raise ValueError(f"rank={rank} is outside PCP size {layout.pcp_size}")
        pages = torch.unique(required_pages[required_pages >= 0], sorted=True).to(torch.int64)
        if pages.numel() > self.capacity_pages:
            raise RuntimeError(
                f"PCP owner staging needs {pages.numel()} pages but the bounded scratch holds "
                f"{self.capacity_pages}. Raise the scratch budget or reduce the selected history."
            )
        if pages.numel() == 0:
            return pages

        self.counters.materializations += 1
        self.counters.pages_materialized += int(pages.numel())
        self.counters.peak_pages = max(self.counters.peak_pages, int(pages.numel()))

        # A full-VA source already maps every peer page at its logical offset,
        # so local and peer pages are both ordinary device reads.  The owner
        # layout is only used for counters here; it must not decide the read
        # path (indexing a full VA with compact owner pages reads wrong data).
        if self._logical_source is not None:
            torch.index_select(self._logical_source, 0, pages, out=self.cache[: pages.numel()])
            owners = layout.owner_of(pages)
            peer = int((owners != rank).sum())
            self.counters.peer_pages_materialized += peer
            self.counters.local_pages_materialized += int(pages.numel()) - peer
            return pages

        owners = layout.owner_of(pages)
        for owner in range(layout.pcp_size):
            owner_mask = owners == owner
            if not bool(owner_mask.any()):
                continue
            stage_positions = torch.nonzero(owner_mask, as_tuple=False).flatten()
            owner_pages = layout.local_page(pages[owner_mask])
            # Advanced indexing yields a copy.  Materialize into a contiguous
            # temporary and publish it back with one index_put on the same
            # stream so a VMM reader can still write a single dense range.
            destination = torch.empty(
                (owner_pages.numel(), *self.cache.shape[1:]),
                dtype=self.cache.dtype,
                device=self.cache.device,
            )
            if owner == rank:
                # Compact owner shards are indexed with owner-local page ids.
                # Do not bind a full VA here: that layout is handled above.
                self._copy_local(owner_pages, destination)
                self.counters.local_pages_materialized += int(owner_pages.numel())
            else:
                reader.copy_pages(owner, owner_pages.long(), destination, non_blocking=non_blocking)
                self.counters.peer_pages_materialized += int(owner_pages.numel())
            self.cache.index_copy_(0, stage_positions.long(), destination)
        return pages

    def _copy_local(self, local_page_ids: torch.Tensor, destination: torch.Tensor) -> None:
        # Subclasses/owner-cache install the compact local shard through
        # ``bind_local``.  This path must never receive a full-VA tensor.
        local = getattr(self, "_local_shard", None)
        if local is None:
            raise RuntimeError("local owner shard is not bound to the stage cache")
        torch.index_select(local, 0, local_page_ids.long(), out=destination)

    def bind_local(self, local_shard: torch.Tensor) -> None:
        """Bind this rank's compact owner shard (owner-local page order)."""
        if local_shard.ndim != self.cache.ndim:
            raise ValueError("local shard rank does not match the stage cache")
        if tuple(local_shard.shape[1:]) != tuple(self.cache.shape[1:]):
            raise ValueError(
                f"local shard page shape {tuple(local_shard.shape[1:])} != stage {tuple(self.cache.shape[1:])}"
            )
        self._local_shard = local_shard
        self._logical_source = None

    def bind_logical_source(self, logical_cache: torch.Tensor, num_pages: int) -> None:
        """Bind a full-VA tensor; staging then reads by logical page id only."""
        if logical_cache.ndim != self.cache.ndim:
            raise ValueError("logical cache rank does not match the stage cache")
        if tuple(logical_cache.shape[1:]) != tuple(self.cache.shape[1:]):
            raise ValueError(
                f"logical cache page shape {tuple(logical_cache.shape[1:])} != stage {tuple(self.cache.shape[1:])}"
            )
        if logical_cache.shape[0] < num_pages:
            raise ValueError(f"logical cache has {logical_cache.shape[0]} pages, layout needs {num_pages}")
        self._logical_source = logical_cache
        self._local_shard = None

    def overlay_current_rows(
        self,
        staged_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        values: torch.Tensor,
    ) -> int:
        """Overlay rows produced by the current step onto the staged pages.

        ``slot_mapping`` is ``(page, row)``.  Rows whose page is not staged are
        skipped, matching a prefill chunk that only touches a subset of the
        pages visible to this attention call.  The caller must invoke this on
        the same stream as :meth:`materialize` and the kernel.
        """
        if slot_mapping.ndim != 2 or slot_mapping.shape[-1] != 2:
            raise ValueError("current-row slot_mapping must have shape [rows, 2]")
        if slot_mapping.shape[0] < values.shape[0]:
            raise ValueError("slot_mapping has fewer rows than values")
        slot_mapping = slot_mapping[: values.shape[0]]
        valid = slot_mapping[:, 0] >= 0
        if not bool(valid.any()):
            return 0
        pages = slot_mapping[:, 0]
        stage_positions = torch.full_like(pages, -1)
        selected = pages[valid].to(torch.int64)
        staged_pages = staged_pages.to(torch.int64)
        if staged_pages.numel():
            positions = torch.searchsorted(staged_pages, selected)
            in_range = positions < staged_pages.numel()
            matched = torch.zeros_like(in_range, dtype=torch.bool)
            if bool(in_range.any()):
                matched[in_range] = staged_pages[positions[in_range]] == selected[in_range]
            stage_positions[valid] = torch.where(matched, positions, torch.full_like(positions, -1)).to(
                stage_positions.dtype
            )
        keep = stage_positions >= 0
        if not bool(keep.any()):
            return 0
        rows = slot_mapping[keep, 1].long()
        self.cache[stage_positions[keep].long(), rows, :] = values[: values.shape[0]][keep].to(self.cache.dtype)
        self.counters.current_rows_overlaid += int(keep.sum())
        return int(keep.sum())


class PcpOwnerShardCache:
    """Owner-sharded KV pages plus a bounded consumer stage view.

    ``full_va=True`` wraps the PCP-wide owner slab: ``persistent_cache`` is a
    non-owning tensor over the complete logical range and every peer page is
    already mapped.  Reads then use logical page ids for both local and peer
    pages; owner-local compact ids would read the wrong page.  ``full_va=False``
    keeps the phase-A compact owner shard used by the in-process transport.
    """

    def __init__(
        self,
        persistent_cache: torch.Tensor,
        stage_cache: BoundedStageCache,
        layout: PcpOwnerLayout,
        rank: int,
        *,
        full_va: bool = False,
    ) -> None:
        required_pages = layout.num_pages if full_va else layout.num_local_pages
        if persistent_cache.shape[0] < required_pages:
            raise ValueError(
                f"persistent cache has {persistent_cache.shape[0]} pages, layout requires at least "
                f"{required_pages} {'logical' if full_va else 'owner'} pages"
            )
        if not 0 <= rank < layout.pcp_size:
            raise ValueError(f"rank={rank} is outside PCP size {layout.pcp_size}")
        self.persistent_cache = persistent_cache
        self.stage_cache = stage_cache
        self.layout = layout
        self.rank = rank
        self.full_va = full_va
        self._bind_storage()

    def _bind_storage(self) -> None:
        if self.full_va:
            self.stage_cache.bind_logical_source(self.persistent_cache, self.layout.num_pages)
        else:
            self.stage_cache.bind_local(self.persistent_cache)

    @property
    def persistent_bytes(self) -> int:
        return self.persistent_cache.numel() * self.persistent_cache.element_size()

    def scatter_owned(
        self,
        slot_mapping: torch.Tensor,
        values: torch.Tensor | None,
        *,
        page_ids: torch.Tensor | None = None,
    ) -> int:
        """Persist only this rank's owned rows.

        ``slot_mapping`` may be flat (physical slot) or ``(page, row)``.  The
        caller still computes the globally ordered KV; placement filtering is
        the only difference from the replicated baseline.
        """
        if values is None or values.shape[0] == 0:
            return 0
        if page_ids is None:
            if slot_mapping.ndim != 2 or slot_mapping.shape[-1] != 2:
                raise ValueError("scatter_owned requires a [rows, 2] (page, row) slot mapping")
            if slot_mapping.shape[0] < values.shape[0]:
                raise ValueError("slot_mapping has fewer rows than values")
            mapping = slot_mapping[: values.shape[0]]
            page_ids = mapping[:, 0]
            rows = mapping[:, 1]
        elif slot_mapping.ndim == 2:
            if slot_mapping.shape[-1] != 2 or slot_mapping.shape[0] < values.shape[0]:
                raise ValueError("explicit page_ids require a [rows, 2] (page, row) slot mapping")
            rows = slot_mapping[: values.shape[0], 1]
        else:
            raise ValueError("explicit page_ids require (page, row) coordinates for the owned rows")
        owned = self.layout.owned_mask(page_ids, self.rank)
        if not bool(owned.any()):
            return 0
        if self.full_va:
            logical_pages = page_ids[owned].clamp_min(0).long()
            self.persistent_cache[logical_pages, rows[owned].long(), :] = values[owned].to(self.persistent_cache.dtype)
            return int(owned.sum())
        local_pages = self.layout.local_page(page_ids[owned].clamp_min(0)).long()
        self.persistent_cache[local_pages, rows[owned].long(), :] = values[owned].to(self.persistent_cache.dtype)
        return int(owned.sum())

    def materialize_for_attention(
        self,
        required_pages: torch.Tensor,
        *,
        block_table: torch.Tensor | None = None,
        reader: PeerPageReader,
        non_blocking: bool = False,
    ) -> StagedKvView:
        """Stage the pages this call needs, not the whole block table.

        ``required_pages`` must be the sparse attention selection plus the
        current step's pages, for example the output of
        :func:`stage_pages_from_sparse_selection`.  Passing every page
        referenced by a 1M-context block table (8192 pages at block size 128)
        would overflow any practical scratch and is not a capacity-tuning
        problem: sparse attention only dereferences top-k pages.
        """
        # One scratch view may be shared by sequential layers; bind this
        # layer's storage before staging.
        self._bind_storage()
        staged = self.stage_cache.materialize(
            required_pages,
            layout=self.layout,
            rank=self.rank,
            reader=reader,
            non_blocking=non_blocking,
        )
        remapped = None if block_table is None else remap_block_table_to_stage(block_table, staged)
        peer_pages = 0
        if staged.numel():
            peer_pages = int((self.layout.owner_of(staged) != self.rank).sum())
        return StagedKvView(
            cache=self.stage_cache.cache[: staged.numel()],
            block_table=remapped if remapped is not None else block_table,
            staged_pages=staged,
            peer_pages=peer_pages,
            local_pages=int(staged.numel()) - peer_pages,
            counters=self.stage_cache.counters,
        )


def validate_owner_sharing(vllm_config) -> tuple[bool, str]:
    """Check whether owner sharing can be enabled for this configuration."""
    parallel = vllm_config.parallel_config
    if parallel.prefill_context_parallel_size <= 1:
        return False, "owner sharing requires prefill_context_parallel_size > 1"
    if parallel.decode_context_parallel_size != 1:
        return False, "owner sharing requires decode_context_parallel_size == 1"
    if parallel.tensor_parallel_size != 1:
        return False, "owner sharing currently requires tensor_parallel_size == 1 (PCP replicated-cache contract)"
    if getattr(vllm_config, "kv_transfer_config", None) is not None and not _owner_kv_transfer_allowed():
        return False, "owner sharing does not support an external KV-transfer connector"
    return True, ""


def _owner_kv_transfer_allowed() -> bool:
    """Experimental opt-in: let an external KV connector register owner slabs.

    Default off.  The owner slab is a VMM range whose physical pages belong to
    this rank but which peers also map; an external connector (KV pool) can
    only work if the driver accepts registering that range.
    """
    from vllm_ascend import envs

    return str(envs.VLLM_ASCEND_PCP_OWNER_ALLOW_KV_TRANSFER).strip().lower() in ("1", "true", "yes", "on")


# The runner installs the owner-slab allocator and peer transport before the
# first attention metadata build.  Stage A ships the placement/staging contract
# and leaves this seam explicit: enabling ``owner`` without a runtime must fail
# closed instead of silently running a replicated cache.
_owner_kv_runtime: object | None = None


def register_owner_kv_runtime(runtime: object | None) -> None:
    global _owner_kv_runtime
    _owner_kv_runtime = runtime


def owner_kv_runtime_available() -> bool:
    return _owner_kv_runtime is not None


def resolve_owner_sharing(vllm_config) -> tuple[bool, str]:
    """Whether this process may run the owner-sharded PCP prefill path."""
    supported, reason = validate_owner_sharing(vllm_config)
    if not supported:
        return False, reason
    if owner_kv_runtime_available():
        return True, ""
    # The device backend is probed without touching a device: it only checks
    # that the CANN V2 symbols and the torch_npu external-storage constructors
    # exist.  The actual export/import/map happens once during KV allocation
    # and falls back collectively if any step fails.
    from .pcp_owner_vmm import owner_slab_support

    available, reason = owner_slab_support()
    if not available:
        return False, "owner-slab allocator and VMM peer reader are not installed in this build: " + reason
    return True, ""


def log_owner_sharing_fallback(reason: str) -> None:
    logger.warning_once("PCP owner-sharded KV is unavailable (%s); using the replicated cache path.", reason)
