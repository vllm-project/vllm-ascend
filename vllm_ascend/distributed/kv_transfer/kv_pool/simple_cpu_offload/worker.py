# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for Ascend SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_ascend.distributed.kv_transfer.kv_pool.simple_cpu_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class SimpleCPUOffloadWorker:
    """Worker-side handler for simple CPU/NPU KV cache transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        self.load_stream: torch.npu.Stream | None = None
        self.store_stream: torch.npu.Stream | None = None

        self._load_events: list[tuple[int, torch.npu.Event]] = []
        self._load_hwm: int = -1

        self._connector_metadata: SimpleCPUOffloadMetadata | None = None
        self._pending_load_event_indices: set[int] = set()
        self._submitted_load_event_indices: set[int] = set()
        self._completed_store_events: dict[int, int] = {}
        self._load_stream_waited = False

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register KV caches and initialize CPU/NPU transfer resources."""
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        def _repr_tensor(value: Any) -> torch.Tensor:
            if isinstance(value, torch.Tensor):
                return value
            assert isinstance(value, list)
            assert value
            return value[0]

        any_tensor = _repr_tensor(next(iter(kv_caches.values())))
        self.device = any_tensor.device

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks

        seen_ptrs: dict[int, tuple[str, torch.Tensor]] = {}
        for name, value in kv_caches.items():
            tensor = _repr_tensor(value)
            ptr = tensor.untyped_storage().data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs[ptr] = (name, tensor)

        unique_gpu_caches: dict[str, torch.Tensor] = {}
        for name, tensor in seen_ptrs.values():
            storage = tensor.untyped_storage()
            raw = torch.empty(0, dtype=torch.int8, device=self.device).set_(
                storage,
                0,
                (storage.nbytes(),),
            )
            element_size = tensor.element_size()
            page_size_bytes = storage.nbytes() // num_blocks
            outer_dims = [
                dim
                for dim in range(tensor.ndim)
                if tensor.stride(dim) * element_size > page_size_bytes
            ]
            if not outer_dims:
                unique_gpu_caches[name] = raw.view(num_blocks, -1)
            else:
                segment_stride = tensor.stride(outer_dims[0]) * element_size
                for idx in range(tensor.shape[outer_dims[0]]):
                    offset = idx * segment_stride
                    chunk = raw[offset : offset + segment_stride]
                    unique_gpu_caches[f"{name}.{idx}"] = chunk.view(num_blocks, -1)

        per_tensor_bytes_per_block = [
            tensor.stride(0) * tensor.element_size()
            for tensor in unique_gpu_caches.values()
        ]
        total_bytes_per_block = sum(per_tensor_bytes_per_block)
        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)

        self.gpu_kv_caches = unique_gpu_caches
        self.cpu_kv_caches = {}
        for name, gpu_tensor in unique_gpu_caches.items():
            cpu_shape = (self.num_cpu_blocks,) + gpu_tensor.shape[1:]
            self.cpu_kv_caches[name] = torch.zeros(
                cpu_shape,
                dtype=gpu_tensor.dtype,
                device="cpu",
            )

        self.load_stream = torch.npu.Stream()
        self.store_stream = torch.npu.Stream()

        logger.info(
            "SimpleCPUOffloadWorker scaffold registered %d unique KV tensors, "
            "allocating %d CPU blocks (%.2f GB).",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        self._load_stream_waited = False
        if metadata.preempt_load_event >= 0:
            self._pending_load_event_indices.add(metadata.preempt_load_event)

    def clear_connector_metadata(self) -> None:
        """Clear metadata after the model runner finishes the current step."""
        self._connector_metadata = None

    def handle_preemptions(
        self,
        kv_connector_metadata: SimpleCPUOffloadMetadata,
    ) -> None:
        """Save preempted blocks before input preparation can overwrite them."""
        if kv_connector_metadata.need_flush:
            self._flush_and_sync_all()

        # The scheduler may immediately reuse preempted block IDs in this same
        # step. This blocking D2H must therefore run before _update_states()
        # processes new_block_ids_to_zero and before model forward writes KV.
        self._submit_transfer(
            kv_connector_metadata.preempt_store_gpu_blocks,
            kv_connector_metadata.preempt_store_cpu_blocks,
            kv_connector_metadata.preempt_store_event,
            is_store=True,
            sync=True,
        )

    def start_load_kv(self) -> None:
        """Submit pre-forward recompute H2D transfers."""
        metadata = self._connector_metadata
        if metadata is None:
            return

        self._submit_transfer(
            metadata.preempt_load_cpu_blocks,
            metadata.preempt_load_gpu_blocks,
            metadata.preempt_load_event,
            is_store=False,
        )

    def wait_for_layer_load(self) -> None:
        """Make the current forward stream wait for the recompute H2D copy."""
        if self._load_stream_waited or self.load_stream is None:
            return
        metadata = self._connector_metadata
        if metadata is None or metadata.preempt_load_event < 0:
            return
        torch.npu.current_stream().wait_stream(self.load_stream)
        self._load_stream_waited = True

    def _flush_and_sync_all(self) -> None:
        """Synchronize all in-flight transfer events."""
        for event_idx, event in self._load_events:
            event.synchronize()
            self._load_hwm = event_idx
        self._load_events.clear()
        self._submitted_load_event_indices.clear()

    def _poll_load_events(self) -> int:
        """Return the highest completed H2D event index."""
        events = self._load_events
        hwm = self._load_hwm

        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.pop(0)

        self._load_hwm = hwm
        return hwm

    def _submit_transfer(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        event_idx: int,
        is_store: bool,
        sync: bool = False,
    ) -> None:
        """Submit a CPU<->NPU block copy and record a completion event."""
        if event_idx < 0:
            return
        if not is_store and event_idx in self._submitted_load_event_indices:
            return
        if not is_store:
            self._submitted_load_event_indices.add(event_idx)

        if not src_block_ids:
            if is_store:
                self._completed_store_events[event_idx] = 1
            else:
                self._load_hwm = max(self._load_hwm, event_idx)
            return

        assert len(src_block_ids) == len(dst_block_ids)
        assert self.gpu_kv_caches is not None
        assert self.cpu_kv_caches is not None

        stream = self.store_stream if is_store else self.load_stream
        assert stream is not None

        if is_store:
            stream.wait_stream(torch.npu.current_stream())

        with torch.npu.stream(stream):
            for src_block_id, dst_block_id in zip(src_block_ids, dst_block_ids):
                for name, gpu_tensor in self.gpu_kv_caches.items():
                    cpu_tensor = self.cpu_kv_caches[name]
                    if is_store:
                        # TODO: Replace this D2H torch copy with the NPU copy
                        # backend dedicated kernel.
                        cpu_tensor[dst_block_id].copy_(
                            gpu_tensor[src_block_id],
                            non_blocking=True,
                        )
                    else:
                        # TODO: Replace this H2D torch copy with the NPU copy
                        # backend dedicated kernel.
                        gpu_tensor[dst_block_id].copy_(
                            cpu_tensor[src_block_id],
                            non_blocking=True,
                        )
            event = torch.npu.Event()
            event.record(stream)

        if sync:
            event.synchronize()
            if is_store:
                self._completed_store_events[event_idx] = 1
            else:
                self._load_hwm = max(self._load_hwm, event_idx)
                self._pending_load_event_indices.discard(event_idx)
                self._submitted_load_event_indices.discard(event_idx)
            return

        assert not is_store
        self._load_events.append((event_idx, event))

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Poll recompute transfers and report completed request restores."""
        metadata = self._connector_metadata
        if metadata is None:
            return None, None

        finished_recving: set[str] = set()
        if self._pending_load_event_indices:
            load_hwm = self._poll_load_events()
            completed_loads = [
                event_idx
                for event_idx in self._pending_load_event_indices
                if event_idx <= load_hwm
            ]
            for event_idx in completed_loads:
                self._pending_load_event_indices.discard(event_idx)
                self._submitted_load_event_indices.discard(event_idx)
                finished_recving.update(
                    metadata.preempt_load_event_to_reqs.get(event_idx, [])
                )

        return None, finished_recving or None

    def build_connector_worker_meta(self) -> SimpleCPUOffloadWorkerMetadata | None:
        """Return completed store events since the previous call.

        The scheduler aggregates this metadata across workers/ranks. A store
        event becomes available to recompute requests only after all expected
        workers have reported completion.
        """
        if not self._completed_store_events:
            return None
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
        )
        self._completed_store_events = {}
        return meta
