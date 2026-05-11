"""Worker-side handler for the Ascend ``SimpleCPUOffloadConnector``.

Mirrors :class:`vllm.v1.simple_kv_offload.worker.SimpleCPUOffloadWorker`
but uses ``torch.npu`` streams/events and the NPU-flavored DMA backend.
The scheduler-side metadata protocol is identical and reused as-is.
"""

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

from vllm_ascend.simple_kv_offload.copy_backend import NPUDmaCopyBackend

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


def _flatten_kv_value(
    value: torch.Tensor | tuple | list,
) -> list[torch.Tensor]:
    """Yield every constituent tensor of a per-layer KV-cache entry.

    On Ascend, attention layers register ``kv_caches[name]`` as a tuple
    of independently-allocated tensors (e.g. ``(k_cache, v_cache)``);
    Mamba layers register a list. Each tensor has its own backing
    storage and shape ``[num_blocks, ...]``.
    """
    if isinstance(value, torch.Tensor):
        return [value]
    assert isinstance(value, (tuple, list)), f"unexpected kv_caches value type: {type(value)}"
    return [t for t in value if isinstance(t, torch.Tensor)]


class SimpleCPUOffloadNPUWorker:
    """Worker-side handler for CPU offloading transfers on Ascend NPU."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
    ) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes

        self.npu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        self.load_stream: torch.npu.Stream | None = None
        self.store_stream: torch.npu.Stream | None = None

        self._backend = NPUDmaCopyBackend()

        # FIFO of (event_idx, Event), monotonic per direction.
        self._load_events: list[tuple[int, torch.npu.Event]] = []
        self._store_events: list[tuple[int, torch.npu.Event]] = []
        # High-water marks: highest event_idx completed per stream.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        self._connector_metadata: SimpleCPUOffloadMetadata | None = None
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        self._completed_store_events: dict[int, int] = {}

    # ------------------------------------------------------------------
    # KV cache registration
    # ------------------------------------------------------------------
    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | tuple | list],
    ) -> None:
        """Register NPU KV caches and allocate pinned CPU mirrors.

        For every unique storage backing ``kv_caches`` we expose a
        contiguous ``[num_blocks, block_bytes]`` int8 view. The batch
        memcpy backend then strides blocks uniformly across all such
        sub-tensors in a single ``aclrtMemcpyBatchAsync`` call.
        """
        if not kv_caches:
            logger.warning("No NPU KV caches to offload.")
            return

        first_tensor = _flatten_kv_value(next(iter(kv_caches.values())))[0]
        self.device = first_tensor.device

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks

        # Deduplicate by untyped_storage().data_ptr(): a single NPU
        # allocation may back multiple layers (e.g. shared KV across
        # tied weights or via aliasing). On Ascend, K and V live in
        # *separate* allocations, so we must iterate every sub-tensor
        # — taking only ``value[0]`` would silently drop the V cache.
        unique_caches: dict[str, torch.Tensor] = {}
        seen_ptrs: set[int] = set()
        for layer_name, value in kv_caches.items():
            for sub_idx, tensor in enumerate(_flatten_kv_value(value)):
                storage = tensor.untyped_storage()
                ptr = storage.data_ptr()
                if ptr in seen_ptrs:
                    continue
                seen_ptrs.add(ptr)

                key = layer_name if sub_idx == 0 else f"{layer_name}.{sub_idx}"
                unique_caches.update(self._build_block_views(key, tensor, num_blocks))

        per_tensor_bpb = [t.stride(0) * t.element_size() for t in unique_caches.values()]
        total_bytes_per_block = sum(per_tensor_bpb)
        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)
        logger.info(
            "SimpleCPUOffloadNPUWorker: %d unique NPU KV tensors, allocating %d CPU blocks (%.2f GB)",
            len(unique_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

        pin_memory = is_pin_memory_available()
        if not pin_memory:
            logger.warning("Pinned memory not available; CPU offload throughput may be degraded on this host.")

        self.npu_kv_caches = unique_caches
        self.cpu_kv_caches = {
            name: torch.zeros(
                (self.num_cpu_blocks,) + tuple(t.shape[1:]),
                dtype=t.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            for name, t in unique_caches.items()
        }

        # Mirror upstream: lowest-priority transfer streams so KV I/O
        # yields to compute on the default stream.
        low_pri, _ = torch.npu.Stream.priority_range()
        self.load_stream = torch.npu.Stream(priority=low_pri)
        self.store_stream = torch.npu.Stream(priority=low_pri)
        self._backend.init(
            self.npu_kv_caches,
            self.cpu_kv_caches,
            self.device,
            self.load_stream,
            self.store_stream,
        )

    @staticmethod
    def _build_block_views(
        key: str,
        tensor: torch.Tensor,
        num_blocks: int,
    ) -> dict[str, torch.Tensor]:
        """Return ``{name: [num_blocks, block_bytes] int8 view}`` for one tensor.

        Sizes views from the tensor's own metadata, NOT
        ``storage.nbytes()``. When offload is enabled,
        ``NPUModelRunner._allocate_kv_cache_tensors`` over-allocates
        each KV tensor by ``+alignment`` (2 MiB) and slices back with
        ``_align_memory(...)[:size]``; ``storage.nbytes()`` then
        includes alignment-driven leading offset *and* trailing
        padding that are not part of the block grid (the total is in
        general not a multiple of ``num_blocks``).

        Most Ascend layers register K and V as separate blocks-outermost
        tensors (single segment). The ``cache_only_layers`` path with
        ``AscendAttentionBackend`` produces ``(N, num_blocks, ...)`` —
        N segments stacked in one allocation; we split it into N keyed
        views. The runner's actual blocks-dim size may exceed
        ``kv_cache_config.num_blocks``; we only view the leading
        ``num_blocks`` blocks the connector knows about.
        """
        el = tensor.element_size()
        storage = tensor.untyped_storage()
        storage_offset_bytes = tensor.storage_offset() * el

        if tensor.ndim >= 1 and tensor.shape[0] >= num_blocks:
            # Single-segment, blocks-outermost.
            page_size_bytes = tensor.stride(0) * el
            data_bytes = num_blocks * page_size_bytes
            raw = torch.empty(0, dtype=torch.int8, device=tensor.device).set_(
                storage, storage_offset_bytes, (data_bytes,)
            )
            return {key: raw.view(num_blocks, page_size_bytes)}

        # Multi-segment: ``(N, num_blocks, ...)`` is the only NPU layout
        # observed (N=2 for K|V stacked). We assume a single outer
        # partition dim before the blocks dim.
        if tensor.ndim < 2 or tensor.shape[1] < num_blocks:
            raise RuntimeError(
                f"_build_block_views[{key}]: cannot locate blocks dim "
                f"(expected shape[0] or shape[1] >= {num_blocks}) in "
                f"shape {tuple(tensor.shape)}"
            )
        page_size_bytes = tensor.stride(1) * el
        seg_data_bytes = num_blocks * page_size_bytes
        seg_stride_bytes = tensor.stride(0) * el
        n_segments = tensor.shape[0]
        total_bytes = (n_segments - 1) * seg_stride_bytes + seg_data_bytes

        raw = torch.empty(0, dtype=torch.int8, device=tensor.device).set_(storage, storage_offset_bytes, (total_bytes,))
        segs: dict[str, torch.Tensor] = {}
        for idx in range(n_segments):
            start = idx * seg_stride_bytes
            chunk = raw[start : start + seg_data_bytes]
            segs[f"{key}.{idx}"] = chunk.view(num_blocks, page_size_bytes)
        return segs

    # ------------------------------------------------------------------
    # Per-step metadata plumbing
    # ------------------------------------------------------------------
    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        # Defer launching load/store until after model execution so the
        # Python-side block-list build overlaps with NPU compute.
        pass

    def wait_for_save(self) -> None:
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Submit transfers and report completed events to the scheduler."""
        metadata = self._connector_metadata
        if metadata is not None:
            if metadata.load_cpu_blocks:
                self._backend.launch_copy(
                    metadata.load_cpu_blocks,
                    metadata.load_gpu_blocks,
                    is_store=False,
                    event_idx=metadata.load_event,
                    events_list=self._load_events,
                )
            if metadata.store_gpu_blocks:
                self._backend.launch_copy(
                    metadata.store_gpu_blocks,
                    metadata.store_cpu_blocks,
                    is_store=True,
                    event_idx=metadata.store_event,
                    events_list=self._store_events,
                )

        finished_recving: set[str] = set()

        if self._pending_load_event_indices:
            load_wm = self._poll_stream_events(is_store=False)
            for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
                self._pending_load_event_indices.discard(j)
                req_ids = metadata.load_event_to_reqs.get(j) if metadata is not None else None
                if req_ids:
                    finished_recving.update(req_ids)

        if self._pending_store_event_indices:
            store_wm = self._poll_stream_events(is_store=True)
            for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
                self._pending_store_event_indices.discard(j)
                self._completed_store_events[j] = 1

        return None, finished_recving or None

    def build_connector_worker_meta(
        self,
    ) -> SimpleCPUOffloadWorkerMetadata | None:
        if not self._completed_store_events:
            return None
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
        )
        self._completed_store_events = {}
        return meta

    def handle_preemptions(self, kv_connector_metadata: SimpleCPUOffloadMetadata) -> None:
        if not kv_connector_metadata.need_flush:
            return
        self._flush_and_sync_all()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _flush_and_sync_all(self) -> None:
        for event_idx, event in self._load_events:
            event.synchronize()
            self._load_hwm = event_idx
        self._load_events.clear()

        for event_idx, event in self._store_events:
            event.synchronize()
            self._store_hwm = event_idx
        self._store_events.clear()

    def _poll_stream_events(self, is_store: bool) -> int:
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.pop(0)
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm
