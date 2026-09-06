# SPDX-License-Identifier: Apache-2.0
"""MemFabric Hybrid MTE backend for KV layer parallelism.

MemFabric Hybrid 1.2 cannot export an existing KV tensor as a remote MTE GVA.
This backend therefore allocates one bounded symmetric active-page staging
segment per rank. Layer owners copy selected persistent pages directly to each
consumer's segment with batched AscendC GM->UB->remote-GM launches. Consumers
unpack the same device-resident page descriptors into their existing scratch
cache before attention. No full layer cache is copied or staged.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.core.kv_cache_placement import kvpp_staging_size_bytes


def _build_group_store_url(store_url: str, kvpp_group: GroupCoordinator) -> str:
    """Give each KVPP process group its own MemFabric config store.

    A PP deployment has one KVPP group per pipeline stage. MemFabric's SHM
    initializer identifies participants only by ``(store_url, world_size,
    rank_id)``; reusing one URL would therefore merge stage-local ranks from
    different PP stages. KVPP groups are contiguous slices of the global rank
    grid, so their ordinal can safely select a stage-local TCP port.
    """
    group_index = min(kvpp_group.ranks) // kvpp_group.world_size
    if group_index == 0:
        return store_url

    parsed = urlsplit(store_url)
    port = cast(int, parsed.port) + group_index
    host = cast(str, parsed.hostname)
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


@dataclass(frozen=True)
class KVPPActivePages:
    """Fixed-shape device page IDs and their compact staging slots."""

    physical_page_ids: torch.Tensor
    valid_page_mask: torch.Tensor
    staging_page_indices: torch.Tensor


@dataclass(frozen=True)
class MTEStagingRegion:
    """One rank's remotely addressable staging region."""

    base_address: int
    capacity_bytes: int
    kvpp_group_rank: int


@dataclass(frozen=True)
class _MTECompletion:
    """Completion event that keeps asynchronous descriptor tensors alive."""

    event: Any
    retained_resources: tuple[torch.Tensor, ...]

    def synchronize(self) -> None:
        self.event.synchronize()


@dataclass(frozen=True)
class _MTETransferRegion:
    """One base address and its logical-page layout."""

    base_tensor: torch.Tensor
    page_stride_bytes: int
    page_length_bytes: int


@dataclass(frozen=True)
class _MTEDeviceTransferPlan:
    """Static device metadata shared by every transfer of one cache bundle."""

    anchor: torch.Tensor
    local_base_offsets: torch.Tensor
    page_strides: torch.Tensor
    page_lengths: torch.Tensor
    staging_region_offsets: torch.Tensor


class MemFabricMTEKVPPTransport:
    """Move active physical pages through bounded symmetric MTE staging."""

    def __init__(
        self,
        kvpp_group: GroupCoordinator,
        num_physical_pages: int,
        *,
        shared_memory_backend: Any | None = None,
        copy_pages_op: Callable[..., None] | None = None,
        staging_capacity_bytes: int | None = None,
    ) -> None:
        self._kvpp_group = kvpp_group
        self._num_physical_pages = num_physical_pages
        self._shared_memory_backend = shared_memory_backend
        self._copy_pages_op = copy_pages_op
        self._planned_staging_capacity_bytes = staging_capacity_bytes
        self._staging_memory: Any | None = None
        self._local_staging_region: MTEStagingRegion | None = None
        self._staging_regions_by_rank: list[MTEStagingRegion] = []
        self._transfer_regions_by_cache: dict[str, tuple[_MTETransferRegion, ...]] = {}
        self._device_transfer_plans_by_cache_bundle: dict[tuple[str, ...], _MTEDeviceTransferPlan] = {}
        self._staging_shm_id: int

    def initialize_transport(
        self,
        kv_caches_by_name: dict[str, Any],
        cache_bundles: tuple[tuple[str, ...], ...],
        max_active_pages: int,
    ) -> None:
        memfabric_backend = None
        if self._shared_memory_backend is None:
            import memfabric_hybrid  # type: ignore

            memfabric_backend = memfabric_hybrid
            self._shared_memory_backend = memfabric_hybrid.shm
        shared_memory_backend = cast(Any, self._shared_memory_backend)
        if self._copy_pages_op is None:
            # Load the extension lazily to avoid early RTS initialization.
            import vllm_ascend.vllm_ascend_C  # type: ignore # noqa: F401

            self._copy_pages_op = torch.ops._C_ascend.kvpp_mte_copy

        store_url = os.getenv("MF_CONFIG_STORE_URL") or os.environ["ASCEND_MF_STORE_URL"]
        store_url = _build_group_store_url(store_url, self._kvpp_group)
        self._transfer_regions_by_cache = self._build_transfer_regions(kv_caches_by_name)
        self._device_transfer_plans_by_cache_bundle, required_bytes = self._build_device_transfer_plans(
            cache_bundles, max_active_pages
        )
        # Standalone transport callers have no worker budget. Production uses
        # the capacity already included in the worker's memory plan.
        staging_capacity_bytes = self._planned_staging_capacity_bytes
        if staging_capacity_bytes is None:
            staging_capacity_bytes = required_bytes
        self._check_staging_layout(required_bytes, staging_capacity_bytes, max_active_pages)
        self._staging_shm_id = envs.ASCEND_KVPP_MTE_SHM_ID
        staging_memory = self._create_staging_memory(
            shared_memory_backend,
            memfabric_backend,
            store_url,
            staging_capacity_bytes,
        )
        local_staging_region = self._gather_staging_regions(staging_memory, staging_capacity_bytes)

        logger.info(
            "KVPP MemFabric MTE initialized: rank=%d, gva=%#x, staging_capacity_bytes=%d, shm_id=%d, store_url=%s",
            self._kvpp_group.rank_in_group,
            local_staging_region.base_address,
            staging_capacity_bytes,
            self._staging_shm_id,
            store_url,
        )

    def _check_staging_layout(self, required_bytes: int, capacity_bytes: int, max_active_pages: int) -> None:
        """Check the wire layout, not equality of rank-local scratch allocations."""
        # Equal total sizes alone do not ensure the owner and receiver agree
        # on segment offsets or data types. Local base addresses and strides
        # may differ and are deliberately excluded from the wire signature.
        bundles = tuple(
            (
                names,
                tuple(
                    (r.page_length_bytes, r.base_tensor.dtype)
                    for name in names
                    for r in self._transfer_regions_by_cache[name]
                ),
            )
            for names in sorted(self._device_transfer_plans_by_cache_bundle)
        )
        local = (capacity_bytes, required_bytes, self._num_physical_pages, max_active_pages, bundles)
        layouts = [local] * self._kvpp_group.world_size
        dist.all_gather_object(layouts, local, group=self._kvpp_group.cpu_group)
        if any(
            capacity != capacity_bytes
            or required > capacity
            or not 0 < slots <= pages
            or (pages, slots, peer_bundles) != local[2:]
            for capacity, required, pages, slots, peer_bundles in layouts
        ):
            raise ValueError("KVPP staging capacity or wire layout does not match across ranks; SHM was not allocated.")

    def _create_staging_memory(
        self,
        shared_memory_backend: Any,
        memfabric_backend: Any | None,
        store_url: str,
        staging_capacity_bytes: int,
    ) -> Any:
        shm_config = shared_memory_backend.ShmConfig()
        shm_config.start_store = self._kvpp_group.rank_in_group == 0
        timeout_seconds = envs.ASCEND_KVPP_MTE_TIMEOUT_SECONDS
        shm_config.init_timeout = timeout_seconds
        shm_config.create_timeout = timeout_seconds
        shm_config.operation_timeout = timeout_seconds
        device_id = torch.npu.current_device()
        if memfabric_backend is not None:
            return_code = memfabric_backend.initialize(0)
            if return_code != 0:
                raise RuntimeError(f"KVPP MemFabric global initialization failed: error={return_code}.")
        return_code = shared_memory_backend.initialize(
            store_url,
            self._kvpp_group.world_size,
            self._kvpp_group.rank_in_group,
            device_id,
            shm_config,
        )
        if return_code != 0:
            raise RuntimeError(f"KVPP MemFabric SHM initialization failed: error={return_code}.")
        staging_memory = shared_memory_backend.create(
            self._staging_shm_id,
            self._kvpp_group.world_size,
            self._kvpp_group.rank_in_group,
            staging_capacity_bytes,
            shared_memory_backend.ShmDataOpType.MTE,
        )
        if staging_memory is None:
            raise RuntimeError("KVPP MemFabric SHM creation returned no memory.")
        # ``create`` publishes the symmetric address before every rank's
        # device-side mapping is necessarily ready. The SHM barrier completes
        # that setup; a process-group barrier is not an equivalent substitute.
        staging_memory.barrier()
        self._staging_memory = staging_memory
        return staging_memory

    def _build_transfer_regions(
        self,
        kv_caches_by_name: dict[str, Any],
    ) -> dict[str, tuple[_MTETransferRegion, ...]]:
        transfer_regions_by_cache: dict[str, tuple[_MTETransferRegion, ...]] = {}
        for cache_name, cache in kv_caches_by_name.items():
            cache_tensors = (cache,) if isinstance(cache, torch.Tensor) else tuple(cache)
            if not cache_tensors:
                raise ValueError(f"KVPP cache group {cache_name} cannot be empty.")

            transfer_regions: list[_MTETransferRegion] = []
            for cache_tensor in cache_tensors:
                if cache_tensor.ndim == 0 or cache_tensor.shape[0] % self._num_physical_pages != 0:
                    raise RuntimeError(
                        f"KVPP cache {cache_name} shape {tuple(cache_tensor.shape)} "
                        f"cannot be divided into {self._num_physical_pages} pages."
                    )
                rows_per_page = cache_tensor.shape[0] // self._num_physical_pages
                page = cache_tensor[0:rows_per_page]
                if not page.is_contiguous():
                    raise RuntimeError(f"KVPP cache {cache_name} page is not contiguous.")

                page_stride_bytes = cache_tensor.stride(0) * cache_tensor.element_size() * rows_per_page
                page_length_bytes = page.numel() * cache_tensor.element_size()
                if page_length_bytes > page_stride_bytes:
                    raise RuntimeError(
                        f"KVPP cache {cache_name} pages overlap: "
                        f"length={page_length_bytes}, stride={page_stride_bytes}."
                    )
                transfer_regions.append(
                    _MTETransferRegion(
                        base_tensor=cache_tensor,
                        page_stride_bytes=page_stride_bytes,
                        page_length_bytes=page_length_bytes,
                    )
                )
            transfer_regions_by_cache[cache_name] = tuple(transfer_regions)
        return transfer_regions_by_cache

    def _gather_staging_regions(
        self,
        staging_memory: Any,
        staging_capacity_bytes: int,
    ) -> MTEStagingRegion:
        # ``gva`` is the common symmetric base. MemFabric may align each
        # rank's segment to an internal symmetric size larger than the local
        # contribution. That size is intentionally queried inside the
        # AscendC kernel; the Python binding does not expose it.
        self._local_staging_region = MTEStagingRegion(
            base_address=int(staging_memory.gva),
            capacity_bytes=staging_capacity_bytes,
            kvpp_group_rank=self._kvpp_group.rank_in_group,
        )
        peer_staging_regions = [self._local_staging_region] * self._kvpp_group.world_size
        dist.all_gather_object(
            peer_staging_regions,
            self._local_staging_region,
            group=self._kvpp_group.cpu_group,
        )
        self._staging_regions_by_rank = peer_staging_regions
        return self._local_staging_region

    def _build_device_transfer_plans(
        self,
        cache_bundles: tuple[tuple[str, ...], ...],
        max_active_pages: int,
    ) -> tuple[dict[tuple[str, ...], _MTEDeviceTransferPlan], int]:
        """Build static descriptors and their staging requirement in one pass."""
        plans: dict[tuple[str, ...], _MTEDeviceTransferPlan] = {}
        required_page_bytes = 0
        for cache_names in cache_bundles:
            if cache_names in plans:
                continue

            regions: list[_MTETransferRegion] = []
            staging_offsets: list[int] = []
            page_offset = 0
            for cache_name in cache_names:
                for region in self._transfer_regions_by_cache[cache_name]:
                    regions.append(region)
                    staging_offsets.append(page_offset * max_active_pages)
                    page_offset += region.page_length_bytes
            required_page_bytes = max(required_page_bytes, page_offset)
            if not regions:
                raise ValueError("KVPP MTE cache bundle cannot be empty.")

            anchor = regions[0].base_tensor
            if any(region.base_tensor.device != anchor.device for region in regions):
                raise RuntimeError("KVPP MTE cache bundle tensors must share one device.")
            anchor_address = anchor.data_ptr()
            plans[cache_names] = _MTEDeviceTransferPlan(
                anchor=anchor,
                local_base_offsets=torch.tensor(
                    [region.base_tensor.data_ptr() - anchor_address for region in regions],
                    dtype=torch.int64,
                    device=anchor.device,
                ),
                page_strides=torch.tensor(
                    [region.page_stride_bytes for region in regions],
                    dtype=torch.int64,
                    device=anchor.device,
                ),
                page_lengths=torch.tensor(
                    [region.page_length_bytes for region in regions],
                    dtype=torch.int64,
                    device=anchor.device,
                ),
                staging_region_offsets=torch.tensor(
                    staging_offsets,
                    dtype=torch.int64,
                    device=anchor.device,
                ),
            )
        return plans, kvpp_staging_size_bytes(required_page_bytes, self._num_physical_pages)

    def _build_active_page_descriptors(
        self,
        cache_names: tuple[str, ...],
        active_pages: KVPPActivePages,
    ) -> tuple[_MTEDeviceTransferPlan, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine cached layout metadata with this forward's active pages."""
        plan = self._device_transfer_plans_by_cache_bundle[cache_names]
        page_ids = active_pages.physical_page_ids.to(dtype=torch.int64)
        staging_page_indices = active_pages.staging_page_indices.to(dtype=torch.int64)
        valid_page_mask = active_pages.valid_page_mask
        if (
            page_ids.device != plan.anchor.device
            or staging_page_indices.device != plan.anchor.device
            or valid_page_mask.device != plan.anchor.device
        ):
            raise RuntimeError("KVPP MTE active pages and transfer plan must share one device.")

        local_offsets = (plan.local_base_offsets[:, None] + page_ids[None, :] * plan.page_strides[:, None]).flatten()
        staging_offsets = (
            plan.staging_region_offsets[:, None] + staging_page_indices[None, :] * plan.page_lengths[:, None]
        ).flatten()
        lengths = torch.where(
            valid_page_mask[None, :],
            plan.page_lengths[:, None],
            torch.zeros((), dtype=torch.int64, device=plan.anchor.device),
        ).flatten()
        return plan, local_offsets, staging_offsets, lengths

    def copy_active_pages_to_staging(
        self,
        cache_names: tuple[str, ...],
        active_pages: KVPPActivePages,
        stream: Any,
    ) -> Any:
        if self._local_staging_region is None or not self._staging_regions_by_rank:
            raise RuntimeError("KVPP MTE transport was not initialized.")

        copy_pages_op = cast(Callable[..., None], self._copy_pages_op)
        plan, local_offsets, staging_offsets, lengths = self._build_active_page_descriptors(
            cache_names,
            active_pages,
        )
        for peer_staging_region in self._staging_regions_by_rank:
            if peer_staging_region.kvpp_group_rank == self._kvpp_group.rank_in_group:
                continue
            copy_pages_op(
                plan.anchor,
                local_offsets,
                staging_offsets,
                lengths,
                peer_staging_region.base_address,
                -1,
                peer_staging_region.kvpp_group_rank,
                self._staging_shm_id,
            )
        completion_event = torch.npu.Event()
        completion_event.record(stream)
        return _MTECompletion(completion_event, (local_offsets, staging_offsets, lengths))

    def copy_active_pages_from_staging(
        self,
        cache_names: tuple[str, ...],
        active_pages: KVPPActivePages,
        stream: Any,
    ) -> Any:
        if self._local_staging_region is None:
            raise RuntimeError("KVPP MTE transport was not initialized.")

        copy_pages_op = cast(Callable[..., None], self._copy_pages_op)
        plan, local_offsets, staging_offsets, lengths = self._build_active_page_descriptors(
            cache_names,
            active_pages,
        )
        copy_pages_op(
            plan.anchor,
            local_offsets,
            staging_offsets,
            lengths,
            self._local_staging_region.base_address,
            self._local_staging_region.kvpp_group_rank,
            -1,
            self._staging_shm_id,
        )
        completion_event = torch.npu.Event()
        completion_event.record(stream)
        return _MTECompletion(completion_event, (local_offsets, staging_offsets, lengths))
