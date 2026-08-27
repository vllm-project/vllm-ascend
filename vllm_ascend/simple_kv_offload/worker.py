"""Worker-side handler for the Ascend ``SimpleCPUOffloadConnector``.

Subclasses :class:`vllm.v1.simple_kv_offload.worker.SimpleCPUOffloadWorker`
and only overrides what differs on NPU:

* ``__init__`` swaps the CUDA copy backend for the NPU one. Step-time
  state (event lists, hwm cursors, pending sets, completed-store map)
  is fully inherited.
* ``register_kv_caches`` rebuilds block views around two NPU-specific
  realities: K/V live in *separate* allocations (not stacked under one
  outer dim) and the runner over-allocates each tensor for 2 MiB
  alignment, so view sizing must come from tensor shape/stride rather
  than ``storage.nbytes()``. The CPU mirrors are pinned via plain
  ``torch.zeros(pin_memory=True)`` since ``cudaHostRegister`` is
  CUDA-only, and transfer streams drop the lowest-priority hint that
  ``torch.npu.Stream`` does not yet expose.

All other handler entry points — ``bind_connector_metadata``,
``clear_connector_metadata``, ``start_load_kv``, ``wait_for_save``,
``get_finished``, ``build_connector_worker_meta``, ``handle_preemptions``,
``_flush_and_sync_all``, ``_poll_stream_events`` — are inherited
verbatim.
"""

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

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


class SimpleCPUOffloadNPUWorker(SimpleCPUOffloadWorker):
    """NPU-flavored ``SimpleCPUOffloadWorker``.

    The inherited ``gpu_kv_caches`` field holds NPU caches on this
    platform — kept as-is for parent-class compatibility.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config, cpu_capacity_bytes)
        # Replace the CUDA backend created by ``super().__init__``.
        # ``DmaCopyBackend.__init__`` only assigns None defaults — no
        # CUDA resource was allocated, so the transient instance is
        # just GC'd.
        self._backend = NPUDmaCopyBackend()

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

        self.gpu_kv_caches = unique_caches
        self.cpu_kv_caches = {
            name: torch.zeros(
                (self.num_cpu_blocks,) + tuple(t.shape[1:]),
                dtype=t.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            for name, t in unique_caches.items()
        }

        # Upstream creates these with the lowest CUDA priority so KV I/O
        # yields to compute on the default stream. ``torch.npu`` does
        # NOT expose ``Stream.priority_range()`` / a ``priority=`` kwarg
        # (``RuntimeError: NPU does not support Stream.priority_range()
        # currently``) and there is no equivalent torch_npu API today.
        # Use plain transfer streams — matches every other
        # ``torch.npu.Stream`` site in this repo. The transfers still
        # run off the default compute stream, so they overlap with the
        # forward pass; we only lose the explicit "always yield" hint,
        # which is a soft scheduling preference and not a correctness
        # requirement.
        self.load_stream = torch.npu.Stream()
        self.store_stream = torch.npu.Stream()
        self._backend.init(
            self.gpu_kv_caches,
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
        # NOTE: ``seg_page_size_bytes`` is per-segment (e.g. just K or
        # just V), NOT the full KVCacheSpec page size — for this stacked
        # layout the full page would be ``n_segments * seg_page_size_bytes``.
        # Naming aligns with the ``seg_data_bytes`` / ``seg_stride_bytes``
        # prefix convention used below.
        if tensor.ndim < 2 or tensor.shape[1] < num_blocks:
            raise RuntimeError(
                f"_build_block_views[{key}]: cannot locate blocks dim "
                f"(expected shape[0] or shape[1] >= {num_blocks}) in "
                f"shape {tuple(tensor.shape)}"
            )
        seg_page_size_bytes = tensor.stride(1) * el
        seg_data_bytes = num_blocks * seg_page_size_bytes
        seg_stride_bytes = tensor.stride(0) * el
        n_segments = tensor.shape[0]
        total_bytes = (n_segments - 1) * seg_stride_bytes + seg_data_bytes

        raw = torch.empty(0, dtype=torch.int8, device=tensor.device).set_(storage, storage_offset_bytes, (total_bytes,))
        segs: dict[str, torch.Tensor] = {}
        for idx in range(n_segments):
            start = idx * seg_stride_bytes
            chunk = raw[start : start + seg_data_bytes]
            segs[f"{key}.{idx}"] = chunk.view(num_blocks, seg_page_size_bytes)
        return segs
