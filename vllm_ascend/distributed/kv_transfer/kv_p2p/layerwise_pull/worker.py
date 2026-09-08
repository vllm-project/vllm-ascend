# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Worker side of the backend-independent layerwise pull connector."""

from __future__ import annotations

import math
import os
import threading
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.utils.network_utils import get_ip
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.protocol import (
    ComponentLayout,
    SendTask,
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.read_thread import (
    ConsumerReadState,
    LayerwisePullReadThread,
    PullBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.send_thread import (
    LayerwisePullSendingThread,
    ProducerSendState,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    get_layerwise_physical_layer_index,
)
from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import (
    BACKEND_MEMFABRIC,
    MEMFABRIC_ROLE_DECODE,
    MEMFABRIC_ROLE_PREFILL,
    global_memfabric_te,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    RegisterRegions,
    collect_storage_merged_register_regions,
    get_transfer_timeout_value,
    validate_register_region_count,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata

BACKEND_MOONCAKE = "mooncake"
SUPPORTED_PULL_BACKENDS = (BACKEND_MEMFABRIC, BACKEND_MOONCAKE)
CONNECTOR_THREAD_STARTUP_TIMEOUT_SECONDS = 10.0
PD_READ_WAIT_LOG_INTERVAL_SECONDS = 10.0
MIN_TCP_PORT = 1
MAX_TCP_PORT = 65535


def _resolve_kv_transfer_backend(vllm_config: VllmConfig) -> str:
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
    backend = extra.get("transfer_backend")
    if backend not in SUPPORTED_PULL_BACKENDS:
        raise ValueError(
            "LayerwisePullConnector requires "
            'kv_connector_extra_config["transfer_backend"] to be one of '
            f"{SUPPORTED_PULL_BACKENDS}, got {backend!r}"
        )
    return backend


def _validate_tcp_port(port: int, *, description: str) -> None:
    if not MIN_TCP_PORT <= port <= MAX_TCP_PORT:
        raise ValueError(f"{description} must be in [{MIN_TCP_PORT}, {MAX_TCP_PORT}], got {port}")


class LayerwisePullConsumerWorker:
    """Build local component layouts and pull Prefill buffers into them."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None,
    ) -> None:
        if kv_cache_config is None:
            raise ValueError("LayerwisePullConsumerWorker requires KVCacheConfig")
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.use_layerwise = use_layerwise
        self._backend_name = _resolve_kv_transfer_backend(vllm_config)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.total_base_layers = vllm_config.model_config.get_total_num_hidden_layers()
        self.side_channel_host = get_ip()
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank * self.pp_size * self.tp_size
            + self.pp_rank * self.tp_size
        )
        _validate_tcp_port(
            self.side_channel_port + self.tp_size - 1,
            description="Layerwise pull D-side highest TP control-plane port",
        )

        self.engine = None
        self._read_thread: LayerwisePullReadThread | None = None
        self.offload_manager = None
        self._invalid_block_ids: set[int] = set()
        self.request_map: dict[str, str] = {}
        self._dest_blocks_by_req: dict[str, list[list[int]]] = {}
        self._dest_blocks_condition = threading.Condition()
        self._terminal_ext_ids: set[str] = set()
        self._pending_recv_req_ids: set[str] = set()
        self._deferred_cleanup_req_ids: set[str] = set()
        self.layer_layouts: dict[int, tuple[ComponentLayout, ...]] = {}

    def _ensure_engine(self) -> tuple[Any, PullBackend]:
        if self.engine is None:
            device_id = torch.npu.current_device()
            if self._backend_name == BACKEND_MEMFABRIC:
                global_memfabric_te.configure(role=MEMFABRIC_ROLE_DECODE, device_id=device_id)
                self.engine = global_memfabric_te.get_transfer_engine(self.side_channel_host)
            else:
                device_name = str(device_id) if self.pp_size > 1 else None
                self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=device_name)
        backend = (
            PullBackend.memfabric(self.engine)
            if self._backend_name == BACKEND_MEMFABRIC
            else PullBackend.mooncake(self.engine)
        )
        return self.engine, backend

    @staticmethod
    def _build_hbm_layouts(
        kv_cache_config: KVCacheConfig,
        kv_caches: dict[str, torch.Tensor],
        total_base_layers: int,
    ) -> dict[int, list[ComponentLayout]]:
        layer_to_group = {
            layer_name: group_idx
            for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
            for layer_name in group.layer_names
        }
        num_blocks = kv_cache_config.num_blocks
        layouts: dict[int, list[ComponentLayout]] = {}
        for layer_name, cache_or_caches in kv_caches.items():
            if layer_name not in layer_to_group:
                raise RuntimeError(f"Layerwise pull cannot find a KV cache group for {layer_name}")
            tensors = cache_or_caches if isinstance(cache_or_caches, (list, tuple)) else (cache_or_caches,)
            bases: list[int] = []
            dtypes: list[str] = []
            strides: list[int] = []
            lengths: list[int] = []
            shapes: list[tuple[int, ...]] = []
            scales: list[int] = []
            for tensor in tensors:
                if tensor.shape[0] % num_blocks != 0:
                    raise ValueError(
                        f"Layerwise pull tensor {layer_name} has {tensor.shape[0]} rows, "
                        f"which is not divisible by num_blocks={num_blocks}"
                    )
                scale = tensor.shape[0] // num_blocks
                stride = tensor.stride(0) * tensor.element_size() * scale
                length = tensor.element_size() * math.prod(tensor.shape[1:]) * scale
                bases.append(tensor.data_ptr())
                dtypes.append(str(tensor.dtype))
                strides.append(stride)
                lengths.append(length)
                shapes.append(tuple(tensor.shape[1:]))
                scales.append(scale)
            group_idx = layer_to_group[layer_name]
            layouts.setdefault(get_layerwise_physical_layer_index(layer_name, total_base_layers), []).append(
                ComponentLayout(
                    name=layer_name,
                    group_index=group_idx,
                    block_size=kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec.block_size,
                    dtypes=tuple(dtypes),
                    base_addrs=tuple(bases),
                    block_strides=tuple(strides),
                    block_lengths=tuple(lengths),
                    block_shapes=tuple(shapes),
                    block_size_scales=tuple(scales),
                )
            )
        return layouts

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        tp_shared_components: set[str] = set()

        # Sparse main caches contain optional HBM/CPU tensors and top-k buffers,
        # not an ordinary HBM component. Only register their TP-shared CPU K/V.
        from vllm_ascend.ascend_config import get_ascend_config

        main_names: set[str] = set()
        hbm_destinations = kv_caches
        if get_ascend_config().sparse_kv_offload_config.enabled:
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
                get_sparse_kv_offload_manager,
            )

            self.offload_manager = get_sparse_kv_offload_manager()
            main_names = set(getattr(self.offload_manager, "offload_layer_names", ()))
            if not main_names:
                raise RuntimeError("SparseKVOffloadManager.register_kv_caches must run before LayerwisePullConnector")
            hbm_destinations = {name: value for name, value in kv_caches.items() if name not in main_names}

        layouts = self._build_hbm_layouts(self.kv_cache_config, hbm_destinations, self.total_base_layers)
        registration = collect_storage_merged_register_regions(hbm_destinations)
        if main_names:
            layer_to_group = {
                layer_name: group_idx
                for group_idx, group in enumerate(self.kv_cache_config.kv_cache_groups)
                for layer_name in group.layer_names
            }
            main_ptrs: list[int] = []
            main_lengths: list[int] = []
            for pool_idx, layer_name in enumerate(self.offload_manager.offload_layer_names):
                group_idx = layer_to_group[layer_name]
                k_base = self.offload_manager.gvas_k_bases[pool_idx]
                v_base = self.offload_manager.gvas_v_bases[pool_idx]
                k_len, v_len = self.offload_manager.cpu_block_lens[pool_idx]
                layer_idx = get_layerwise_physical_layer_index(layer_name, self.total_base_layers)
                # Top-k tensors exist on every TP rank. Their head dimensions
                # and dtype match the CPU cache, but their row size does not.
                topk_tensors = (
                    self.offload_manager.topk_buffers_k[pool_idx],
                    self.offload_manager.topk_buffers_v[pool_idx],
                )
                block_shapes = tuple((self.offload_manager.block_size, *tensor.shape[2:]) for tensor in topk_tensors)
                block_size_scales = tuple(
                    length // (tensor.element_size() * math.prod(shape))
                    for length, tensor, shape in zip((k_len, v_len), topk_tensors, block_shapes, strict=True)
                )
                component = ComponentLayout(
                    name=layer_name,
                    group_index=group_idx,
                    block_size=self.kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec.block_size,
                    dtypes=tuple(str(tensor.dtype) for tensor in topk_tensors),
                    base_addrs=(k_base, v_base),
                    block_strides=(k_len, v_len),
                    block_lengths=(k_len, v_len),
                    block_shapes=block_shapes,
                    block_size_scales=block_size_scales,
                )
                layouts.setdefault(layer_idx, []).insert(0, component)
                tp_shared_components.add(layer_name)

                num_blocks = self.kv_cache_config.num_blocks
                start = min(k_base, v_base)
                end = max(k_base + k_len * num_blocks, v_base + v_len * num_blocks)
                main_ptrs.append(start)
                main_lengths.append(end - start)

            registration = RegisterRegions(
                ptrs=main_ptrs + registration.ptrs,
                lengths=main_lengths + registration.lengths,
                logical_tensor_count=len(main_ptrs) + (registration.logical_tensor_count or 0),
                logical_total_bytes=sum(main_lengths) + (registration.logical_total_bytes or 0),
            )

        self.layer_layouts = {layer_idx: tuple(components) for layer_idx, components in layouts.items()}
        validate_register_region_count(registration)
        _, backend = self._ensure_engine()
        if self._backend_name == BACKEND_MEMFABRIC:
            global_memfabric_te.register_buffer(registration.ptrs, registration.lengths)
        else:
            global_te.register_buffer(registration.ptrs, registration.lengths)

        self._read_thread = LayerwisePullReadThread(
            tp_rank=self.tp_rank,
            side_channel_port=self.side_channel_port,
            backend=backend,
            state=ConsumerReadState(
                tp_size=self.tp_size,
                layer_layouts=self.layer_layouts,
                dest_blocks_by_req=self._dest_blocks_by_req,
                tp_shared_components=frozenset(tp_shared_components),
                dest_blocks_condition=self._dest_blocks_condition,
            ),
        )
        self._read_thread.start()
        if not self._read_thread.ready_event.wait(timeout=CONNECTOR_THREAD_STARTUP_TIMEOUT_SECONDS):
            self._read_thread.stop()
            raise RuntimeError("Timed out waiting for the layerwise pull D-side read thread")
        if self._read_thread.startup_error is not None:
            error = self._read_thread.startup_error
            self._read_thread.stop()
            raise RuntimeError("Layerwise pull D-side read thread failed during startup") from error
        logger.info(
            "Layerwise pull D registered %d layers through %s",
            len(self.layer_layouts),
            self._backend_name,
        )

    def start_load_kv(self, metadata: KVConnectorMetadata) -> None:
        requests = getattr(metadata, "requests", [])
        if not requests:
            return
        with self._dest_blocks_condition:
            for request in requests:
                req_id = getattr(request, "req_id", None)
                if req_id is None:
                    continue
                ext_id = get_external_request_id(req_id)
                block_ids = [list(group) for group in getattr(request, "block_ids_by_group", [])]
                self.request_map[ext_id] = req_id
                self._dest_blocks_by_req[ext_id] = block_ids
                self._pending_recv_req_ids.add(req_id)
            self._dest_blocks_condition.notify_all()

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def _cleanup_request_state(self, req_ids: set[str]) -> None:
        ext_ids = set()
        for req_id in req_ids:
            ext_id = get_external_request_id(req_id)
            ext_ids.add(ext_id)
            self.request_map.pop(ext_id, None)
            self._dest_blocks_by_req.pop(ext_id, None)
            self._terminal_ext_ids.discard(ext_id)
        if self._read_thread is not None:
            self._read_thread.discard_requests(ext_ids)

    def _gather_tp_read_status(
        self,
        local_terminal: set[str],
        local_failed: set[str],
    ) -> list[tuple[set[str], set[str]]]:
        if self.tp_size == 1:
            return [(local_terminal, local_failed)]
        tp_group = get_tp_group()
        gathered: list[tuple[set[str], set[str]] | None] = [None] * tp_group.world_size
        torch.distributed.all_gather_object(
            gathered,
            (local_terminal, local_failed),
            group=tp_group.cpu_group,
        )
        return [status for status in gathered if status is not None]

    def get_finished(self, finished_req_ids: set[str] | None = None) -> tuple[set[str], set[str]]:
        done_recving: set[str] = set()
        local_failed: set[str] = set()
        if self._read_thread is not None:
            local_done = self._read_thread.get_and_clear_done()
            local_failed = self._read_thread.get_and_clear_failed()
            self._terminal_ext_ids.update(local_done | local_failed)

        tp_status = self._gather_tp_read_status(set(self._terminal_ext_ids), local_failed)
        finished_on_all = set.intersection(*(terminal for terminal, _ in tp_status)) if tp_status else set()
        failed_on_any = set().union(*(failed for _, failed in tp_status))
        self._terminal_ext_ids.difference_update(finished_on_all)
        for ext_id in failed_on_any:
            for group in self._dest_blocks_by_req.get(ext_id, []):
                self._invalid_block_ids.update(group)

        for ext_id in finished_on_all:
            internal = self.request_map.get(ext_id)
            if internal is None:
                raise RuntimeError(
                    f"Layerwise pull completed request {ext_id!r} before its "
                    "D-side internal request mapping was registered"
                )
            done_recving.add(internal)
        # A scheduler finish notification can be a cancellation while READs
        # are still active. vLLM keeps those blocks until finished_recving;
        # preserve their lookup and PP completion counts for the same duration.
        self._pending_recv_req_ids.difference_update(done_recving)
        cleanup_req_ids = self._deferred_cleanup_req_ids & done_recving
        if finished_req_ids:
            cleanup_req_ids.update(finished_req_ids - self._pending_recv_req_ids)
            self._deferred_cleanup_req_ids.update(finished_req_ids & self._pending_recv_req_ids)
        if cleanup_req_ids:
            self._deferred_cleanup_req_ids.difference_update(cleanup_req_ids)
            self._cleanup_request_state(cleanup_req_ids)
        return set(), done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid = self._invalid_block_ids
        self._invalid_block_ids = set()
        return invalid

    def shutdown(self) -> None:
        if self._read_thread is not None:
            self._read_thread.stop()


class LayerwisePullProducerWorker:
    """Expose local component layouts and publish per-layer readiness."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None) -> None:
        if kv_cache_config is None:
            raise ValueError("LayerwisePullProducerWorker requires KVCacheConfig")
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self._backend_name = _resolve_kv_transfer_backend(vllm_config)
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.total_base_layers = vllm_config.model_config.get_total_num_hidden_layers()
        self.side_channel_host = get_ip()
        self.side_channel_port = vllm_config.kv_transfer_config.kv_port + self.dp_rank * self.pp_size * self.tp_size
        self.block_size = tuple(group.kv_cache_spec.block_size for group in self.kv_cache_config.kv_cache_groups)
        device_id = torch.npu.current_device()
        if self._backend_name == BACKEND_MEMFABRIC:
            global_memfabric_te.configure(role=MEMFABRIC_ROLE_PREFILL, device_id=device_id)
            self.engine = global_memfabric_te.get_transfer_engine(self.side_channel_host)
            self.session_id = global_memfabric_te.unique_id
        else:
            device_name = str(device_id) if self.pp_size > 1 else None
            self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=device_name)
            self.session_id = f"{self.side_channel_host}:{self.engine.get_rpc_port()}"
        self.layer_layouts: dict[int, tuple[ComponentLayout, ...]] = {}
        self.index_to_name: dict[int, str] = {}
        self._layer_order: tuple[int, ...] = ()
        self.layer_storage_slots: dict[int, tuple[int, ...]] = {}
        self.reused_storage_slots: frozenset[int] = frozenset()
        self.current_layer = 0
        self.last_layer_idx = -1
        self.kv_send_layer_thread: LayerwisePullSendingThread | None = None
        self._pd_dispatched_layers: set[int] = set()
        self._routes_by_topology: dict[str, tuple[dict[int, tuple[str, int]], frozenset[int]]] = {}

    def get_finished(self, finished_req_ids: set[str] | None = None) -> tuple[set[str], set[str]]:
        if self.kv_send_layer_thread is None:
            return set(), set()
        done_sending = self.kv_send_layer_thread.get_and_clear_finished_requests(finished_req_ids or set())
        return done_sending, set()

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def start_load_kv(self, metadata: KVConnectorMetadata) -> None:
        self.current_layer = 0
        self._pd_dispatched_layers = set()
        if not metadata.requests:
            return
        if len(metadata.producer_pp_layers) != self.pp_size:
            raise RuntimeError("Layerwise pull producer PP topology has not been initialized")
        if tuple(metadata.producer_pp_layers[self.pp_rank]) != self._layer_order:
            raise RuntimeError("Layerwise pull producer layer ownership differs from startup metadata")
        assert self.kv_send_layer_thread is not None
        self.kv_send_layer_thread._state.pp_layers = metadata.producer_pp_layers
        self.kv_send_layer_thread._state.tp_size = self.tp_size
        for req_id, req_meta in getattr(metadata, "requests", {}).items():
            remote_tp_size = req_meta.remote_tp_size or self.tp_size
            topology_id = req_meta.remote_topology_id
            if not isinstance(topology_id, str) or not topology_id:
                raise ValueError("Layerwise pull requires the D startup topology ID")
            remote_tp_rank = self._map_prefill_rank_to_decode_rank(
                prefill_tp_size=self.tp_size,
                decode_tp_size=remote_tp_size,
                prefill_tp_rank=self.tp_rank,
            )
            req_meta.tp_ratio = self.tp_size // remote_tp_size
            req_meta.group_member_idx = self.tp_rank % req_meta.tp_ratio
            route = self._routes_by_topology.get(topology_id)
            if route is None:
                endpoints = req_meta.remote_endpoints
                if not endpoints or len(endpoints) != req_meta.remote_pp_size:
                    raise ValueError("Layerwise pull requires the complete D worker endpoint table")
                if any(len(stage) != remote_tp_size for stage in endpoints):
                    raise ValueError("Layerwise pull D endpoint table does not match its TP size")
                destination_by_layer = {}
                for stage in endpoints:
                    endpoint = stage[remote_tp_rank]
                    host, port, layers = endpoint["host"], endpoint["port"], endpoint["layer_ids"]
                    if not host:
                        raise ValueError("Layerwise pull D endpoint has no host")
                    _validate_tcp_port(port, description="Layerwise pull remote D-side port")
                    for layer in layers:
                        if layer in destination_by_layer:
                            raise ValueError(f"Layerwise pull D has duplicate ownership of layer {layer}")
                        destination_by_layer[layer] = (host, port)
                missing = set(self._layer_order) - destination_by_layer.keys()
                if missing:
                    raise ValueError(f"Layerwise pull D is missing destination layers {sorted(missing)}")
                layer_endpoints = {layer: destination_by_layer[layer] for layer in self._layer_order}
                last_by_endpoint = {endpoint: layer for layer, endpoint in layer_endpoints.items()}
                route = (layer_endpoints, frozenset(last_by_endpoint.values()))
                self._routes_by_topology[topology_id] = route
            req_meta.layer_endpoints, req_meta.terminal_layers = route
            # Register every destination before the first terminal-layer ACK
            # can arrive. A P stage may feed more than one D stage.
            if req_meta.chunk_finish:
                self.kv_send_layer_thread.track_requests({req_id}, set(req_meta.layer_endpoints.values()))
            logger.debug(
                "Layerwise pull P prepared req %s: layer_endpoints=%s, blocks=%s",
                req_id,
                req_meta.layer_endpoints,
                req_meta.local_block_ids,
            )

    @staticmethod
    def _map_prefill_rank_to_decode_rank(
        *,
        prefill_tp_size: int,
        decode_tp_size: int,
        prefill_tp_rank: int,
    ) -> int:
        if prefill_tp_size < 1 or decode_tp_size < 1:
            raise ValueError("Layerwise pull P/D tensor parallel sizes must be positive")
        if prefill_tp_size < decode_tp_size or prefill_tp_size % decode_tp_size != 0:
            raise ValueError(
                "Layerwise pull requires P tensor parallel size to be greater than or equal to, "
                f"and divisible by, D tensor parallel size; got P={prefill_tp_size}, D={decode_tp_size}"
            )
        if not 0 <= prefill_tp_rank < prefill_tp_size:
            raise ValueError(
                f"Layerwise pull P tensor parallel rank {prefill_tp_rank} is outside [0, {prefill_tp_size})"
            )
        return prefill_tp_rank // (prefill_tp_size // decode_tp_size)

    def _build_send_state(self) -> ProducerSendState:
        return ProducerSendState(
            last_layer_idx=self.last_layer_idx,
            layer_layouts=self.layer_layouts,
            p_session=self.session_id,
            block_sizes=self.block_size,
            layer_storage_slots=self.layer_storage_slots,
            num_blocks=self.kv_cache_config.num_blocks,
            pp_rank=self.pp_rank,
            tp_rank=self.tp_rank,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # The producer and an ordinary-HBM consumer use the same layout builder.
        raw_layouts = LayerwisePullConsumerWorker._build_hbm_layouts(
            self.kv_cache_config,
            kv_caches,
            self.total_base_layers,
        )
        self.layer_layouts = {layer_idx: tuple(components) for layer_idx, components in raw_layouts.items()}
        if not self.layer_layouts:
            raise RuntimeError("Layerwise pull producer did not find any KV cache layers")
        self._layer_order = tuple(sorted(self.layer_layouts))
        self.last_layer_idx = self._layer_order[-1]
        for layer_idx, components in self.layer_layouts.items():
            self.index_to_name[layer_idx] = components[0].name

        slot_by_storage: dict[tuple[int, ...], int] = {}
        for layer_idx, components in self.layer_layouts.items():
            slots = []
            for component in components:
                storage_key = tuple(component.base_addrs)
                slots.append(slot_by_storage.setdefault(storage_key, len(slot_by_storage)))
            self.layer_storage_slots[layer_idx] = tuple(slots)
        slot_use_count: dict[int, int] = {}
        for slots in self.layer_storage_slots.values():
            for slot_id in set(slots):
                slot_use_count[slot_id] = slot_use_count.get(slot_id, 0) + 1
        self.reused_storage_slots = frozenset(slot for slot, count in slot_use_count.items() if count > 1)

        registration = collect_storage_merged_register_regions(kv_caches)
        validate_register_region_count(registration)
        if self._backend_name == BACKEND_MEMFABRIC:
            global_memfabric_te.register_buffer(registration.ptrs, registration.lengths)
        else:
            global_te.register_buffer(registration.ptrs, registration.lengths)

        ready_event = threading.Event()
        self.kv_send_layer_thread = LayerwisePullSendingThread(
            ready_event=ready_event,
            state=self._build_send_state(),
        )
        self.kv_send_layer_thread.start()
        if not ready_event.wait(timeout=CONNECTOR_THREAD_STARTUP_TIMEOUT_SECONDS):
            self.kv_send_layer_thread.stop()
            raise RuntimeError("Timed out waiting for the layerwise pull P-side send thread")
        if self.kv_send_layer_thread.startup_error is not None:
            error = self.kv_send_layer_thread.startup_error
            self.kv_send_layer_thread.stop()
            raise RuntimeError("Layerwise pull P-side send thread failed during startup") from error
        logger.info(
            "Layerwise pull P registered %d layers through %s",
            len(self.layer_layouts),
            self._backend_name,
        )

    def _has_pull_target(self, metadata: KVConnectorMetadata, layer_idx: int) -> bool:
        group_indices = {component.group_index for component in self.layer_layouts[layer_idx]}
        for req_meta in getattr(metadata, "requests", {}).values():
            if layer_idx not in req_meta.layer_endpoints:
                continue
            has_blocks = any(
                group_idx < len(req_meta.local_block_ids) and bool(req_meta.local_block_ids[group_idx])
                for group_idx in group_indices
            )
            if has_blocks or (layer_idx in req_meta.terminal_layers and req_meta.chunk_finish):
                return True
        return False

    def on_kv_cache_written(self, layer_name: str, connector_metadata: KVConnectorMetadata) -> None:
        if self.kv_send_layer_thread is None:
            return
        fallback_idx = self._layer_order[self.current_layer] if self.current_layer < len(self._layer_order) else None
        resolved_name = layer_name or self.index_to_name.get(fallback_idx)
        if resolved_name is None:
            return
        layer_idx = get_layerwise_physical_layer_index(resolved_name, self.total_base_layers)
        if layer_idx not in self.layer_layouts or layer_idx in self._pd_dispatched_layers:
            return
        if not getattr(connector_metadata, "requests", None):
            return
        self._pd_dispatched_layers.add(layer_idx)
        if self._has_pull_target(connector_metadata, layer_idx):
            self.kv_send_layer_thread.mark_layer_pending(layer_idx)
        try:
            self.kv_send_layer_thread.record_p_save_event(layer_idx)
            self._enqueue_layer_send(resolved_name, layer_idx, connector_metadata)
        except Exception as error:
            self.kv_send_layer_thread._fail_layer(layer_idx, str(error))
            self._pd_dispatched_layers.discard(layer_idx)
            raise

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
        connector_metadata: KVConnectorMetadata,
        **kwargs,
    ) -> None:
        send_thread = self.kv_send_layer_thread
        if send_thread is None:
            raise RuntimeError("register_kv_caches() must complete before save_kv_layer()")
        fallback_idx = self._layer_order[self.current_layer] if self.current_layer < len(self._layer_order) else None
        resolved_name = layer_name or self.index_to_name.get(fallback_idx)
        if resolved_name is None:
            return
        layer_idx = get_layerwise_physical_layer_index(resolved_name, self.total_base_layers)
        if layer_idx not in self.layer_layouts:
            self.current_layer += 1
            return
        if layer_idx in self._pd_dispatched_layers:
            self.current_layer += 1
            return
        if not getattr(connector_metadata, "requests", None):
            return
        self._pd_dispatched_layers.add(layer_idx)
        if self._has_pull_target(connector_metadata, layer_idx):
            send_thread.mark_layer_pending(layer_idx)
        wait_event = torch.npu.Event()
        wait_event.record()
        self._enqueue_layer_send(
            resolved_name,
            layer_idx,
            connector_metadata,
            wait_event=wait_event,
        )
        self.current_layer += 1

    def _enqueue_layer_send(
        self,
        layer_name: str,
        layer_idx: int,
        connector_metadata: KVConnectorMetadata,
        wait_event: torch.npu.Event | None = None,
    ) -> None:
        assert self.kv_send_layer_thread is not None
        send_requests = {}
        group_indices = {component.group_index for component in self.layer_layouts[layer_idx]}
        for req_id, req_meta in connector_metadata.requests.items():
            has_ready_group = any(
                group_idx < len(req_meta.local_block_ids) and bool(req_meta.local_block_ids[group_idx])
                for group_idx in group_indices
            )
            if has_ready_group or (layer_idx in req_meta.terminal_layers and req_meta.chunk_finish):
                send_requests[req_id] = req_meta
        if send_requests:
            task = SendTask(
                send_request=send_requests,
                wait_event=wait_event,
                layer_idx=layer_idx,
                layer_name=layer_name,
            )
            self.kv_send_layer_thread.enqueue(task)
        else:
            self.kv_send_layer_thread._signal_layer_done(layer_idx)

    def wait_for_slot_release(self, layer_idx: int) -> None:
        """Wait for all Decode readers before AscendStore overwrites a slot."""
        send_thread = self.kv_send_layer_thread
        if send_thread is None:
            return
        if layer_idx not in self.layer_storage_slots:
            raise RuntimeError(f"Layerwise pull storage mapping is missing layer {layer_idx}")
        for slot_id in self.layer_storage_slots[layer_idx]:
            if slot_id not in self.reused_storage_slots:
                continue
            event = send_thread.get_storage_send_event(slot_id)
            if event is None:
                continue
            description = f"physical KV storage slot {slot_id} for layer {layer_idx}"
            while not event.wait(timeout=PD_READ_WAIT_LOG_INTERVAL_SECONDS):
                error = send_thread.get_storage_error(slot_id)
                if error is not None:
                    raise RuntimeError(f"Decode failed to read {description}: {error}")
                if not send_thread.is_alive():
                    raise RuntimeError(f"Layerwise pull send thread stopped while waiting for {description}")
                logger.info("Waiting for Decode to read %s", description)
            error = send_thread.get_storage_error(slot_id)
            if error is not None:
                raise RuntimeError(f"Decode failed to read {description}: {error}")

    def shutdown(self) -> None:
        if self.kv_send_layer_thread is not None:
            self.kv_send_layer_thread.stop()
