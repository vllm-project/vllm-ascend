# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for Ascend PreemptOffloadConnector."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.preempt_offload.metadata import (
    MambaConvLoadMeta,
    PreemptOffloadMetadata,
    PreemptOffloadWorkerMetadata,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True)
class MambaConvCacheBinding:
    """One physical Conv-state tensor requiring resume-time correction."""

    state_shape: tuple[int, int]
    state_dtype: torch.dtype


class PreemptOffloadWorker:
    """Worker-side handler for recompute CPU/NPU KV cache transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int | None,
        offload_host_memory_ratio: float = 1,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes
        self.offload_host_memory_ratio = offload_host_memory_ratio
        self.num_spec_tokens = (
            vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
        )

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0
        self.mamba_conv_cache_bindings: dict[str, MambaConvCacheBinding] = {}

        self.load_stream: torch.npu.Stream | None = None
        self.store_stream: torch.npu.Stream | None = None

        self._load_events: list[tuple[int, torch.npu.Event]] = []
        self._load_hwm: int = -1

        self._connector_metadata: PreemptOffloadMetadata | None = None
        self._pending_load_event_indices: set[int] = set()
        self._submitted_load_event_indices: set[int] = set()
        self._submitted_store_event_indices: set[int] = set()
        self._completed_store_events: dict[int, int] = {}
        self._load_stream_waited = False

    def register_kv_caches(
        self,
        kv_caches: dict[
            str,
            torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
        ],
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
            if get_kv_cache_tensor_layers(t):
                scheduler_gpu_kv_cache_tensors.append(t)
        scheduler_gpu_total_bytes = sum(t.size for t in scheduler_gpu_kv_cache_tensors)
        if self.cpu_capacity_bytes is None:
            scheduler_num_cpu_blocks = max(1, int(self.offload_host_memory_ratio * self.num_gpu_blocks))
        else:
            scheduler_num_cpu_blocks = max(
                1,
                self.num_gpu_blocks * self.cpu_capacity_bytes // scheduler_gpu_total_bytes,
            )

        mamba_layers: dict[str, MambaSpec] = {}
        for group in self.kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[layer_name]
                if isinstance(spec, MambaSpec):
                    mamba_layers[layer_name] = spec

        unique_gpu_caches: dict[str, torch.Tensor] = {}
        cache_name_by_ptr: dict[int, str] = {}
        for layer_name, layer_tensor in kv_caches.items():
            states = list(enumerate(layer_tensor)) if isinstance(layer_tensor, (tuple, list)) else [(0, layer_tensor)]
            for state_idx, single_tensor in states:
                logical_name = f"{layer_name}.{state_idx}" if isinstance(layer_tensor, (tuple, list)) else layer_name
                ptr = single_tensor.data_ptr()
                cache_name = cache_name_by_ptr.get(ptr)
                if cache_name is None:
                    cache_name = logical_name
                    cache_name_by_ptr[ptr] = cache_name
                    unique_gpu_caches[cache_name] = single_tensor.view(single_tensor.shape[0], -1)
                    self.block_size_scale[cache_name] = single_tensor.shape[0] // self.num_gpu_blocks

                spec = mamba_layers.get(layer_name)
                if spec is None or not self._is_conv_state(spec, state_idx):
                    continue

                state_shape = tuple(single_tensor.shape[1:])
                expected_shape = tuple(spec.shapes[state_idx])
                if len(state_shape) != 2 or state_shape != expected_shape:
                    raise RuntimeError(
                        "Unexpected Ascend Mamba Conv cache shape: "
                        f"layer={layer_name}, actual={state_shape}, "
                        f"expected={expected_shape}."
                    )
                if self.block_size_scale[cache_name] != 1:
                    raise RuntimeError(
                        "Mamba Conv cache must have one physical row per cache block: "
                        f"tensor={cache_name}, scale={self.block_size_scale[cache_name]}."
                    )

                binding = MambaConvCacheBinding(
                    state_shape=(state_shape[0], state_shape[1]),
                    state_dtype=single_tensor.dtype,
                )
                existing = self.mamba_conv_cache_bindings.get(cache_name)
                if existing is None:
                    self.mamba_conv_cache_bindings[cache_name] = binding
                elif existing != binding:
                    raise RuntimeError(
                        "Aliased Mamba Conv views must have identical shape and dtype: "
                        f"tensor={cache_name}, first={existing}, alias={binding}."
                    )

        per_tensor_bytes_per_block = [tensor.shape[-1] * tensor.element_size() for tensor in unique_gpu_caches.values()]
        total_bytes_per_block = sum(per_tensor_bytes_per_block)
        if self.cpu_capacity_bytes is None:
            self.num_cpu_blocks = scheduler_num_cpu_blocks
        else:
            self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)
        if self.num_cpu_blocks != scheduler_num_cpu_blocks:
            self.num_cpu_blocks = scheduler_num_cpu_blocks
            logger.warning(
                "PreemptOffloadScheduler has different num_blocks: %d,"
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
            "PreemptOffloadWorker scaffold registered %d unique KV tensors "
            "(%d Mamba Conv tensors), allocating %d CPU blocks (%.2f GB).",
            len(unique_gpu_caches),
            len(self.mamba_conv_cache_bindings),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

    @staticmethod
    def _is_conv_state(spec: MambaSpec, state_idx: int) -> bool:
        if spec.mamba_type == MambaAttentionBackendEnum.LINEAR:
            return False
        if spec.mamba_type == MambaAttentionBackendEnum.SHORT_CONV:
            return True
        if spec.mamba_type in (
            MambaAttentionBackendEnum.MAMBA1,
            MambaAttentionBackendEnum.MAMBA2,
            MambaAttentionBackendEnum.GDN_ATTN,
        ):
            return state_idx == 0
        raise ValueError(f"Preempt offload cannot determine Mamba state semantics for backend {spec.mamba_type}.")

    def bind_connector_metadata(self, metadata: PreemptOffloadMetadata) -> None:
        self._connector_metadata = metadata
        self._load_stream_waited = False
        if metadata.preempt_load_event >= 0:
            self._pending_load_event_indices.add(metadata.preempt_load_event)

    def clear_connector_metadata(self) -> None:
        """Clear metadata after the model runner finishes the current step."""
        if self._connector_metadata is not None:
            self._submitted_store_event_indices.discard(self._connector_metadata.preempt_store_event)
        self._connector_metadata = None

    def handle_preemptions(
        self,
        kv_connector_metadata: PreemptOffloadMetadata,
    ) -> None:
        """Save preempted blocks before input preparation can overwrite them."""
        if kv_connector_metadata.need_flush:
            self._flush_and_sync_all()

        store_event = kv_connector_metadata.preempt_store_event
        if store_event in self._submitted_store_event_indices:
            return

        # The scheduler may immediately reuse preempted block IDs in this same
        # step. This blocking D2H must therefore run before _update_states()
        # processes new_block_ids_to_zero and before model forward writes KV.
        self._submit_transfer(
            kv_connector_metadata.preempt_store_gpu_blocks,
            kv_connector_metadata.preempt_store_cpu_blocks,
            store_event,
            is_store=True,
            sync=True,
        )
        if store_event >= 0:
            self._submitted_store_event_indices.add(store_event)

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
            mamba_conv_loads=metadata.preempt_load_mamba_conv,
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
        mamba_conv_loads: list[MambaConvLoadMeta] | None = None,
    ) -> None:
        """Submit a CPU<->NPU block copy and record a completion event."""
        if event_idx < 0:
            return
        if not is_store and event_idx in self._submitted_load_event_indices:
            return
        if not is_store:
            self._submitted_load_event_indices.add(event_idx)

        if not src_block_ids and not mamba_conv_loads:
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
            if not is_store:
                self._copy_mamba_conv_loads(mamba_conv_loads or [])
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

    def _copy_mamba_conv_loads(
        self,
        loads: list[MambaConvLoadMeta],
    ) -> None:
        """Correct Conv state without changing the normal SSM block restore."""
        assert self.gpu_kv_caches is not None
        assert self.cpu_kv_caches is not None

        if loads and not self.mamba_conv_cache_bindings:
            raise RuntimeError("No registered Mamba Conv cache tensors for preempt H2D.")

        for load in loads:
            for name, binding in self.mamba_conv_cache_bindings.items():
                gpu_state = (
                    self.gpu_kv_caches[name][load.gpu_block_id].view(binding.state_dtype).view(binding.state_shape)
                )
                cpu_state = (
                    self.cpu_kv_caches[name][load.cpu_block_id].view(binding.state_dtype).view(binding.state_shape)
                )

                conv_width = binding.state_shape[0]
                # The first ``conv_width - num_spec_tokens`` rows are the
                # causal-conv history consumed as the next initial state. The
                # remaining rows are speculative working slots. Restoring
                # those slots is both unnecessary and incorrect: with async
                # scheduling they can still contain rejected candidates from
                # the offloaded iteration.
                num_dst_tokens = conv_width - self.num_spec_tokens
                if load.source_offset < 0 or num_dst_tokens <= 0 or load.source_offset + num_dst_tokens > conv_width:
                    raise RuntimeError(
                        "Invalid Mamba Conv restore range: "
                        f"tensor={name}, source_offset={load.source_offset}, "
                        f"width={conv_width}."
                    )
                gpu_state[:num_dst_tokens].copy_(
                    cpu_state[load.source_offset : load.source_offset + num_dst_tokens],
                    non_blocking=True,
                )

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

    def build_connector_worker_meta(self) -> PreemptOffloadWorkerMetadata | None:
        """Return completed store events since the previous call.

        The scheduler aggregates this metadata across workers/ranks. A store
        event becomes available to recompute requests only after all expected
        workers have reported completion.
        """
        if not self._completed_store_events:
            return None
        meta = PreemptOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
        )
        self._completed_store_events = {}
        return meta
