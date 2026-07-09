# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for Ascend RecomputeCPUOffloadConnector."""

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.metadata import (
    RecomputeCPUOffloadMetadata,
    RecomputeCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class RecomputeCPUOffloadWorker:
    """Worker-side handler for recompute CPU/NPU KV cache transfers."""

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

        self._connector_metadata: RecomputeCPUOffloadMetadata | None = None
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

        any_tensor = next(iter(kv_caches.values()))
        if isinstance(any_tensor, (tuple, list)):
            any_tensor = any_tensor[0]
        self.device = any_tensor.device

        assert self.kv_cache_config is not None
        self.num_gpu_blocks = self.kv_cache_config.num_blocks
        self.block_size_scale = {}

        scheduler_gpu_kv_cache_tensors = []
        for t in self.kv_cache_config.kv_cache_tensors:
            if t.shared_by:
                scheduler_gpu_kv_cache_tensors.append(t)
        scheduler_gpu_total_bytes = sum(t.size for t in scheduler_gpu_kv_cache_tensors)
        scheduler_num_cpu_blocks = max(1, self.num_gpu_blocks * self.cpu_capacity_bytes // scheduler_gpu_total_bytes)

        unique_gpu_caches: dict[str, torch.Tensor] = {}
        register_cache_ptrs = []
        for layer_name, layer_tensor in kv_caches.items():
            if isinstance(layer_tensor, (tuple, list)):
                for idx, single_tensor in enumerate(layer_tensor):
                    if single_tensor.data_ptr() not in register_cache_ptrs:
                        unique_gpu_caches[f"{layer_name}.{idx}"] = single_tensor.view(single_tensor.shape[0], -1)
                        register_cache_ptrs.append(single_tensor.data_ptr())
                        self.block_size_scale[f"{layer_name}.{idx}"] = single_tensor.shape[0] // self.num_gpu_blocks
            else:
                if layer_tensor.data_ptr() not in register_cache_ptrs:
                    unique_gpu_caches[layer_name] = layer_tensor.view(layer_tensor.shape[0], -1)
                    register_cache_ptrs.append(layer_tensor.data_ptr())
                    self.block_size_scale[layer_name] = layer_tensor.shape[0] // self.num_gpu_blocks

        per_tensor_bytes_per_block = [tensor.shape[-1] * tensor.element_size() for tensor in unique_gpu_caches.values()]
        total_bytes_per_block = sum(per_tensor_bytes_per_block)
        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)
        if self.num_cpu_blocks != scheduler_num_cpu_blocks:
            self.num_cpu_blocks = scheduler_num_cpu_blocks
            logger.warning(
                "RecomputeCPUOffloadScheduler has different num_blocks: %d,"
                "worker-side num_block is set to %d to align with scheduler.",
                scheduler_num_cpu_blocks,
                scheduler_num_cpu_blocks,
            )

        self.gpu_kv_caches = unique_gpu_caches
        self.cpu_kv_caches = {}
        for name, gpu_tensor in unique_gpu_caches.items():
            tensor_block_size_scale = self.block_size_scale[name]
            cpu_shape = (self.num_cpu_blocks * tensor_block_size_scale,) + gpu_tensor.shape[1:]
            self.cpu_kv_caches[name] = torch.zeros(
                cpu_shape,
                dtype=gpu_tensor.dtype,
                pin_memory=True,
                device="cpu",
            )

        self.load_stream = torch.npu.Stream()
        self.store_stream = torch.npu.Stream()

        logger.info(
            "RecomputeCPUOffloadWorker scaffold registered %d unique KV tensors, allocating %d CPU blocks (%.2f GB).",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

    def bind_connector_metadata(self, metadata: RecomputeCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        self._load_stream_waited = False
        if metadata.preempt_load_event >= 0:
            self._pending_load_event_indices.add(metadata.preempt_load_event)

    def clear_connector_metadata(self) -> None:
        """Clear metadata after the model runner finishes the current step."""
        self._connector_metadata = None

    def handle_preemptions(
        self,
        kv_connector_metadata: RecomputeCPUOffloadMetadata,
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
            sync=True,
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
        torch.npu.synchronize()

        with torch.npu.stream(stream):
            for src_block_id, dst_block_id in zip(src_block_ids, dst_block_ids):
                for name, gpu_tensor in self.gpu_kv_caches.items():
                    cpu_tensor = self.cpu_kv_caches[name]
                    tensor_block_size_scale = self.block_size_scale[name]
                    if is_store:
                        # TODO: Replace this D2H torch copy with the NPU copy
                        # backend dedicated kernel.
                        if tensor_block_size_scale > 1:
                            cpu_tensor[
                                dst_block_id * tensor_block_size_scale : (dst_block_id + 1) * tensor_block_size_scale
                            ].copy_(
                                gpu_tensor[
                                    src_block_id * tensor_block_size_scale : (src_block_id + 1)
                                    * tensor_block_size_scale
                                ],
                                non_blocking=True,
                            )
                        else:
                            cpu_tensor[dst_block_id].copy_(
                                gpu_tensor[src_block_id],
                                non_blocking=True,
                            )
                    else:
                        # TODO: Replace this H2D torch copy with the NPU copy
                        # backend dedicated kernel.
                        if tensor_block_size_scale > 1:
                            gpu_tensor[
                                dst_block_id * tensor_block_size_scale : (dst_block_id + 1) * tensor_block_size_scale
                            ].copy_(
                                cpu_tensor[
                                    src_block_id * tensor_block_size_scale : (src_block_id + 1)
                                    * tensor_block_size_scale
                                ],
                                non_blocking=True,
                            )
                        else:
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
            completed_loads = [event_idx for event_idx in self._pending_load_event_indices if event_idx <= load_hwm]
            for event_idx in completed_loads:
                self._pending_load_event_indices.discard(event_idx)
                self._submitted_load_event_indices.discard(event_idx)
                finished_recving.update(metadata.preempt_load_event_to_reqs.get(event_idx, []))

        return None, finished_recving or None

    def build_connector_worker_meta(self) -> RecomputeCPUOffloadWorkerMetadata | None:
        """Return completed store events since the previous call.

        The scheduler aggregates this metadata across workers/ranks. A store
        event becomes available to recompute requests only after all expected
        workers have reported completion.
        """
        if not self._completed_store_events:
            return None
        meta = RecomputeCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
        )
        self._completed_store_events = {}
        return meta
