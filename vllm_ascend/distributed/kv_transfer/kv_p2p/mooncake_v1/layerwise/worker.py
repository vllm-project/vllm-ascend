# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Layerwise mooncake V1 worker.

The class used to live inside mooncake_layerwise_connector. The shutdown and
startup-readiness scaffold moved to mooncake_v1/base/worker.py; the class
keeps every other method verbatim and imports the package modules directly
instead of going through the entry module.
"""

import copy
import math
import os
import threading
import time
from collections import (
    defaultdict,
    deque,
)
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
from mooncake.engine import TransferEngine
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
)
from vllm.logger import logger
from vllm.utils.network_utils import (
    get_ip,
    make_zmq_path,
    make_zmq_socket,
)
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import extract_layer_index

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.base.worker import (
    MooncakeConnectorWorkerBase,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.endpoints import (
    layerwise_handshake_port,
    layerwise_worker_base_port,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.protocol import (
    DONE_SENDING_MSG,
    FAILED_SENDING_MSG,
    GET_META_MSG,
    LayerMetadata,
    LayerwiseAgentMetadata,
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.transport import (
    SizedDict,
    ensure_zmq_recv_with_poller,
    ensure_zmq_send,
    zmq_ctx,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.protocol import (
    MooncakeLayerwiseConnectorMetadata,
    ReqMeta,
    SendTask,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.step import (
    LayerwiseStepController,
    LayerwiseTransferPlan,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.threads import (
    KVCacheRecvingLayerThread,
    KVCacheSendingLayerThread,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    PD_QOS_DEFAULT,
    RegisterRegions,
    align_memory,
    collect_storage_merged_register_regions,
    context_parallel_parameters_check,
    get_cp_group,
    get_local_remote_block_port_mappings,
    get_transfer_mappings,
    get_transfer_timeout_value,
    inject_qos,
    parallel_info,
    validate_register_region_count,
)
from vllm_ascend.distributed.utils import get_decode_context_model_parallel_rank
from vllm_ascend.utils import (
    get_kv_cache_tensor_layers,
    trans_nd_to_nz,
)

MooncakeAgentMetadata = LayerwiseAgentMetadata

LAYERWISE_REUSE_WAIT_LOG_INTERVAL_SECONDS = 10.0


class MooncakeLayerwiseConnectorWorker(MooncakeConnectorWorkerBase):
    _thread_label = "Layerwise"
    _shutdown_log_name = "MooncakeLayerwiseConnectorWorker"
    _ready_thread_failed_message = "KV Cache layer thread failed to start."
    _ready_timeout_message = "Timeout waiting for KV Cache layer thread to be ready."

    def _transfer_threads(self):
        return (self.kv_recv_layer_thread, self.kv_send_layer_thread)

    def _rollback_registration(self) -> None:
        self._rollback_registered_memory()

    def _close_shutdown_resources(self) -> None:
        try:
            self._close_remote_sockets()
        except Exception as error:
            logger.warning("Failed to close layerwise remote sockets. error=%s. ", error)

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())

        if TransferEngine is None:
            raise RuntimeError("mooncake is not available")
        logger.info("Initializing Mooncake work %s", engine_id)

        # Metadata.
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        self.kv_cache_specs: list[KVCacheSpec] = [spec.kv_cache_spec for spec in self.kv_cache_config.kv_cache_groups]
        self.local_engine_id: str = " "
        self.engine_id = engine_id
        self.dp_rank: int = vllm_config.parallel_config.data_parallel_rank
        self.tp_rank: int = get_tensor_model_parallel_rank()
        self.tp_size: int = vllm_config.parallel_config.tensor_parallel_size
        self.pcp_size: int = vllm_config.parallel_config.prefill_context_parallel_size
        self.pcp_rank: int = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size: int = vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_rank: int = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0
        self.tp_group = get_tp_group()
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_ip()
        self.total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.use_mla = self.vllm_config.model_config.use_mla
        self.request_map = dict[str, str]()
        self.use_attn_mamba_hybrid = False
        if self.use_mla:
            self.total_num_kv_heads = 1
        else:
            self.total_num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()

        # Handshake base port
        self.side_channel_port = layerwise_worker_base_port(
            kv_port=vllm_config.kv_transfer_config.kv_port,
            dp_rank=self.dp_rank,
            pcp_size=self.pcp_size,
            tp_size=self.tp_size,
            pcp_rank=self.pcp_rank,
        )
        self.handshake_port = layerwise_handshake_port(base_port=self.side_channel_port, tp_rank=self.tp_rank)
        self.sockets: dict = {}
        logger.info("Initializing Mooncake work %s", engine_id)
        inject_qos(vllm_config.kv_transfer_config.get_from_extra_config("qos_priority", PD_QOS_DEFAULT))
        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()

        # Background thread for sending or receiving KV caches.
        self.kv_recv_layer_thread: KVCacheRecvingLayerThread | None = None
        self.kv_send_layer_thread: KVCacheSendingLayerThread | None = None
        # Idempotent shutdown state; see shutdown().
        self._shutdown_complete = False
        self._shutdown_lock = threading.Lock()
        # Native registrations owned by this startup; rolled back on failure.
        # See _rollback_registered_memory().
        self._registered_kv_buffer = False
        self._registered_engine_buffers: list[int] = []
        self.layer_storage_slots: dict[int, tuple[int, ...]] = {}
        self.storage_send_done_events: list[threading.Event] = []
        self._storage_send_errors: dict[int, str] = {}
        self._pending_reuse_layers: set[int] = set()
        self._cache_write_events: list[torch.npu.Event | None] = [None] * self.total_layers
        self._layerwise_step_plan: LayerwiseTransferPlan | None = None

        self.block_size: list[int] = [spec.block_size for spec in self.kv_cache_specs]
        self.kernel_block_size_scale: list[int] = [1 for _ in range(self.num_kv_cache_groups)]
        self.layer_metadata: dict[str, LayerMetadata] = {}
        self.sparse_sfa_c8_layers: set[str] = set()
        self.attn_resharding_group_idx = set[int]()

        self.enable_kv_quant = (
            vllm_config.quant_config.enable_fa_quant if vllm_config.quant_config is not None else False
        )
        self.enable_c8_quant = (
            vllm_config.quant_config.enable_c8_quant if vllm_config.quant_config is not None else False
        )
        self.pd_head_ratio = get_ascend_config().pd_head_ratio
        self.num_head_replica = get_ascend_config().num_head_replica
        self.resharding_stream = None
        if self.pd_head_ratio > 1 or self.enable_kv_quant or self.enable_c8_quant:
            self.resharding_stream = torch.npu.Stream()

        self.remote_poller = zmq.Poller()  # type: ignore
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.encoder = msgspec.msgpack.Encoder()

        self.index_to_name = defaultdict(list)
        self.remote_layer_metadata: dict[str, dict[int, dict[str, LayerMetadata]]] = SizedDict()
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        # ZMQ contexts created by _get_remote_socket(); see _close_remote_sockets().
        self.remote_socket_contexts: dict[Any, Any] = {}
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds
        self.k_buffer: torch.Tensor | None = None
        self.v_buffer: torch.Tensor | None = None
        self.virtual_request: set[str] = set()
        self._invalid_block_ids: set[int] = set()
        self._recving_metadata: dict[str, ReqMeta] = {}

    def _rollback_registered_memory(self) -> None:
        """Release the native memory registrations made by this startup.

        Idempotent and best effort: the engine buffers registered by
        ``create_kv_buffer`` are unregistered in reverse order, then the
        process-wide transfer engine registration is released only if this
        startup made it. Normal shutdown does not unregister.
        """
        engine_buffers = list(getattr(self, "_registered_engine_buffers", None) or [])
        self._registered_engine_buffers = []
        for ptr in reversed(engine_buffers):
            try:
                ret_value = self.engine.unregister_memory(ptr)
                if ret_value != 0:
                    logger.warning(
                        "Mooncake engine buffer unregistration failed on rollback. ptr=%s. ret_value=%s. ",
                        ptr,
                        ret_value,
                    )
            except Exception as error:
                logger.warning(
                    "Failed to unregister mooncake engine buffer on rollback. ptr=%s. error=%s. ",
                    ptr,
                    error,
                )
        if not getattr(self, "_registered_kv_buffer", False):
            return
        self._registered_kv_buffer = False
        try:
            global_te.unregister_buffer()
        except Exception as error:
            logger.warning(
                "Failed to roll back mooncake KV cache registration. engine_id=%s. error=%s. ",
                getattr(self, "engine_id", None),
                error,
            )

    def _init_layerwise_reuse_state(self) -> None:
        """Thin delegate to the layerwise step controller."""
        controller = self._new_layerwise_step_controller()
        controller.init_reuse_state()
        self.layer_storage_slots = controller.layer_storage_slots
        self.storage_send_done_events = controller.storage_send_done_events

    def _mark_layer_reuse_pending(self, layer_idx: int) -> None:
        if layer_idx not in self.layer_storage_slots:
            raise RuntimeError(f"Mooncake layerwise reuse mapping is missing layer {layer_idx}")
        if layer_idx in self._pending_reuse_layers:
            return
        self._pending_reuse_layers.add(layer_idx)
        for slot_id in self.layer_storage_slots[layer_idx]:
            self._storage_send_errors.pop(slot_id, None)
            self.storage_send_done_events[slot_id].clear()

    def _complete_layer_reuse(self, layer_idx: int, error: str | None) -> None:
        if layer_idx not in self._pending_reuse_layers:
            return
        self._pending_reuse_layers.remove(layer_idx)
        for slot_id in self.layer_storage_slots[layer_idx]:
            if error is not None:
                self._storage_send_errors[slot_id] = error
            self.storage_send_done_events[slot_id].set()

    def wait_for_layer_reuse(self, layer_idx: int) -> None:
        send_thread = self.kv_send_layer_thread
        if send_thread is None:
            return
        if layer_idx not in self.layer_storage_slots:
            raise RuntimeError(f"Mooncake layerwise reuse mapping is missing layer {layer_idx}")
        for slot_id in self.layer_storage_slots[layer_idx]:
            event = self.storage_send_done_events[slot_id]
            while not event.wait(timeout=LAYERWISE_REUSE_WAIT_LOG_INTERVAL_SECONDS):
                if not send_thread.is_alive():
                    raise RuntimeError("Mooncake send thread stopped while waiting for layerwise buffer reuse")
                logger.info("Waiting for Mooncake transfer to release physical KV slot %d", slot_id)
            error = self._storage_send_errors.get(slot_id)
            if error is not None:
                raise RuntimeError(f"Mooncake failed to transfer physical KV slot {slot_id}: {error}")

    def create_kv_buffer(self, first_kv_cache_tuple):
        alignment = 2 * 1024 * 1024
        first_k_cache = first_kv_cache_tuple[0]
        first_v_cache = first_kv_cache_tuple[1]
        if self.pd_head_ratio > 1 or self.enable_kv_quant or self.enable_c8_quant:
            # regesit kv buffer
            if self.enable_kv_quant:
                k_dtype = torch.int8
                v_dtype = first_v_cache.dtype
            elif self.enable_c8_quant:
                k_dtype = torch.int8
                v_dtype = torch.int8
            else:
                k_dtype = first_k_cache.dtype
                v_dtype = first_v_cache.dtype
            self.k_buffer = torch.zeros(first_k_cache.numel() + alignment, dtype=k_dtype, device=first_k_cache.device)
            self.k_buffer = align_memory(self.k_buffer, alignment)[: first_k_cache.numel()].view(
                -1, first_k_cache.shape[-1]
            )
            self.v_buffer = torch.zeros(first_v_cache.numel() + alignment, dtype=v_dtype, device=first_k_cache.device)
            self.v_buffer = align_memory(self.v_buffer, alignment)[: first_v_cache.numel()].view(
                -1, first_v_cache.shape[-1]
            )
        for tensor in (self.k_buffer, self.v_buffer):
            assert tensor.data_ptr() % alignment == 0, "The address of the registered kv cache should be aligned to 2M"
            ret_value = self.engine.register_memory(tensor.data_ptr(), tensor.numel() * tensor.element_size())
            logger.info("Register memory buffer for transfer, buffer size:%s", tensor.numel() * tensor.element_size())
            if ret_value != 0:
                raise RuntimeError("Mooncake memory registration failed. ")
            # Registered by this startup; _rollback_registered_memory() undoes it
            # if a later step fails.
            self._registered_engine_buffers.append(tensor.data_ptr())

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""
        layer2group_ids: dict[str, int] = {}
        kv_cache_groups = self.kv_cache_config.kv_cache_groups
        for i, kv_cache_group_spec in enumerate(kv_cache_groups):
            for layer_name in kv_cache_group_spec.layer_names:
                layer2group_ids[layer_name] = i

        use_mamba, use_attn = False, False
        conv_total_padding_size = 0
        for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
            for layer_name in get_kv_cache_tensor_layers(kv_cache_tensor):
                layer_kv_cache_spec = self.kv_cache_specs[layer2group_ids[layer_name]]
                if isinstance(layer_kv_cache_spec, MambaSpec):
                    use_mamba = True
                    conv_shape, conv_dtype = layer_kv_cache_spec.shapes[0], layer_kv_cache_spec.dtypes[0]
                    conv_total_padding_size = (
                        self.kv_cache_config.num_blocks * math.prod(conv_shape) * get_dtype_size(conv_dtype)
                    )
                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    use_attn = True
            if use_mamba and use_attn:
                self.use_attn_mamba_hybrid = True
                break

        ptrs = []
        lengths = []
        kv_buffer = None
        group_block_size_scales: dict[int, int] = {}
        for layer_name, kv_cache_tuple in kv_caches.items():
            if isinstance(kv_cache_tuple, (list, tuple)) is False:
                kv_cache_tuple = [kv_cache_tuple]
            layer_kv_group_id = layer2group_ids[layer_name]
            layer_kv_cache_spec = kv_cache_groups[layer_kv_group_id].kv_cache_spec
            if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
            # Read the layout flags directly: some Ascend versions expose the
            # cache spec flags without the corresponding utils helpers.
            is_mla = isinstance(layer_kv_cache_spec, MLAAttentionSpec)
            sparse_sfa_c8 = is_mla and bool(getattr(layer_kv_cache_spec, "cache_sparse_sfa_c8", False))
            sparse_sfa_turboquant = is_mla and bool(getattr(layer_kv_cache_spec, "cache_sparse_sfa_turboquant", False))
            if sparse_sfa_turboquant:
                raise NotImplementedError(
                    "Mooncake layerwise connector does not support packed SFA main KV caches other than C8."
                )
            if sparse_sfa_c8:
                # Attention already packs quantized NoPE, BF16 RoPE and scales
                # into one byte tensor on both P and D. Transfer it unchanged.
                if len(kv_cache_tuple) != 1 or not kv_cache_tuple[0].is_contiguous():
                    raise NotImplementedError("SFA C8 layerwise transfer requires one contiguous packed cache tensor.")
                self.sparse_sfa_c8_layers.add(layer_name)
            if (
                not sparse_sfa_c8
                and "indexer" not in layer_name.lower()
                and (
                    (
                        self.pd_head_ratio > 1
                        and (isinstance(layer_kv_cache_spec, (FullAttentionSpec, SlidingWindowSpec)))
                    )
                    or self.enable_kv_quant
                    or self.enable_c8_quant
                )
            ):
                self.attn_resharding_group_idx.add(layer_kv_group_id)
                if kv_buffer is None:
                    kv_buffer = kv_cache_tuple
            single_layer_meta = LayerMetadata([], [], [], [])
            for single_kv_cache in kv_cache_tuple:
                block_start_rank = 1
                num_blocks = self.kv_cache_config.num_blocks
                tensor_num_blocks = single_kv_cache.shape[0]
                assert tensor_num_blocks % num_blocks == 0, (
                    "The external block size must be an integer multiple of the kernel block size."
                )
                block_size_scale = tensor_num_blocks // num_blocks
                group_block_size_scale = group_block_size_scales.setdefault(
                    layer_kv_group_id,
                    block_size_scale,
                )
                if group_block_size_scale != block_size_scale:
                    raise NotImplementedError(
                        "Mooncake layerwise connector does not support KV cache components "
                        "with different block size scales in one group, including DCP replicated indexer cache."
                    )
                block_shape = single_kv_cache.shape[block_start_rank:]
                single_layer_meta.tensor_group_idx.append(layer_kv_group_id)
                single_layer_meta.kv_caches_base_addr.append(single_kv_cache.data_ptr())
                single_layer_meta.block_len.append(single_kv_cache.element_size() * math.prod(block_shape))
                single_layer_meta.block_size_scale.append(block_size_scale)
                self.kernel_block_size_scale[layer2group_ids[layer_name]] = block_size_scale
                if single_kv_cache.data_ptr() not in ptrs and not self.use_attn_mamba_hybrid:
                    ptrs.append(single_kv_cache.data_ptr())
                    lengths.append(
                        num_blocks * single_kv_cache.element_size() * math.prod(block_shape) * block_size_scale
                    )
                logger.info("layer: %s, num_blocks: %s, block_shape: %s", layer_name, num_blocks, block_shape)
            self.layer_metadata[layer_name] = single_layer_meta

        if self.use_attn_mamba_hybrid:
            for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
                tensor_addrs = []
                for layer_name in get_kv_cache_tensor_layers(kv_cache_tensor):
                    tensor_addrs.extend(self.layer_metadata[layer_name].kv_caches_base_addr)
                    if "mtp" in layer_name:
                        tensor_addrs.append(min(tensor_addrs) - conv_total_padding_size)
                assert min(set(tensor_addrs)) % (2 * 1024 * 1024) == 0, "Tensor start addr is not align with 2M."
                ptrs.append(min(set(tensor_addrs)))
                lengths.append(kv_cache_tensor.size)

        if self.use_attn_mamba_hybrid:
            register_regions = RegisterRegions(ptrs=ptrs, lengths=lengths)
        else:
            # For normal attention / sparse-c8 KV cache, register merged memory
            # ranges while keeping layer metadata at logical tensor addresses.
            register_regions = collect_storage_merged_register_regions(kv_caches)

        validate_register_region_count(register_regions)
        # register_buffer returns False when the process-wide transfer
        # engine is already registered; claiming only what this call
        # registered keeps a rollback from unregistering memory owned elsewhere.
        try:
            self._claim_kv_buffer_registration(register_regions)
            if kv_buffer is not None:
                self.create_kv_buffer(kv_buffer)

            model_type = self.vllm_config.model_config.hf_text_config.model_type
            num_attn_module = 2 if model_type in ("longcat_flash", "longcat_flash_ngram") else 1
            mtp_layer_name = ""
            for layer_name in kv_caches:
                if "mtp" in layer_name:
                    mtp_layer_name = layer_name
                    continue
                layer_idx = extract_layer_index(layer_name, num_attn_module)
                self.index_to_name[layer_idx].append(layer_name)
            for layer_names in self.index_to_name.values():
                layer_names.sort(key=lambda name: "indexer" in name.lower())
                primary_group_idx = self.layer_metadata[layer_names[0]].tensor_group_idx[0]
                assert all(
                    self.layer_metadata[name].tensor_group_idx[0] == primary_group_idx for name in layer_names
                ), f"Mooncake layerwise merged components must share one KV cache group. layer_names={layer_names}"
            if mtp_layer_name != "":
                self.index_to_name[max(self.index_to_name.keys()) + 1].append(mtp_layer_name)
            if self.index_to_name:
                self.total_layers = max(self.total_layers, max(self.index_to_name) + 1)

            self._init_layerwise_reuse_state()

            # After KV Caches registered, start the sending or receiving thread.
            metadata = MooncakeAgentMetadata(
                te_rpc_port=self.te_rpc_port,
                layer_metadata=self.layer_metadata,
            )
            if self.vllm_config.kv_transfer_config.is_kv_producer:
                ready_event = threading.Event()
                self.kv_send_layer_thread = KVCacheSendingLayerThread(
                    engine=self.engine,
                    vllm_config=self.vllm_config,
                    kv_cache_config=self.kv_cache_config,
                    kv_cache_specs=self.kv_cache_specs,
                    attn_resharding_group_idx=self.attn_resharding_group_idx,
                    total_layers=self.total_layers,
                    ready_event=ready_event,
                    tp_size=self.tp_size,
                    tp_rank=self.tp_rank,
                    pd_head_ratio=self.pd_head_ratio,
                    num_head_replica=self.num_head_replica,
                    layer_metadata=self.layer_metadata,
                    use_mla=self.use_mla,
                    use_attn_mamba_hybrid=self.use_attn_mamba_hybrid,
                    k_buffer=self.k_buffer,
                    v_buffer=self.v_buffer,
                    enable_kv_quant=self.enable_kv_quant,
                    enable_c8_quant=self.enable_c8_quant,
                    resharding_stream=self.resharding_stream,
                    callback_func=self.send_done_send_signal,
                    reuse_completion_callback=self._complete_layer_reuse,
                )
                self.kv_send_layer_thread.start()
                self._wait_until_ready(self.kv_send_layer_thread, ready_event)

            if self.vllm_config.kv_transfer_config.is_kv_consumer:
                ready_event = threading.Event()
                self.kv_recv_layer_thread = KVCacheRecvingLayerThread(
                    self.tp_rank,
                    self.side_channel_port,
                    self.tp_size,
                    self.pd_head_ratio,
                    self.engine_id,
                    metadata,
                    ready_event,
                )
                self.kv_recv_layer_thread.start()
                self._wait_until_ready(self.kv_recv_layer_thread, ready_event)
        except Exception:
            # Stop the layer threads first: rollback unregisters native memory,
            # which disconnects peers and must not race a live transfer thread.
            self.shutdown()
            self._rollback_registered_memory()
            raise

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_recving = (
            self.kv_recv_layer_thread.get_and_clear_done_requests(  # type: ignore[union-attr]
            )
            if self.vllm_config.kv_transfer_config.is_kv_consumer
            else set()
        )
        done_recving = {self.request_map[s] for s in done_recving if s in self.request_map}
        done_recving.update(self.virtual_request)
        self.virtual_request = set()

        failed_recving = (
            self.kv_recv_layer_thread.get_and_clear_failed_requests()
            if self.vllm_config.kv_transfer_config.is_kv_consumer and self.kv_recv_layer_thread is not None
            else set()
        )
        failed_recving = {self.request_map[s] for s in failed_recving if s in self.request_map}
        for req_id in failed_recving:
            if meta := self._recving_metadata.get(req_id):
                self._invalid_block_ids.update(block_id for group in meta.local_block_ids for block_id in group)
        for req_id in done_recving.union(failed_recving):
            org_req_id = req_id[:-9]
            self.request_map.pop(org_req_id, None)
            self._recving_metadata.pop(req_id, None)
        if len(done_recving) > 0:
            logger.info(
                "Number of completed KV cache recv requests: %s, receive requests: %s", len(done_recving), done_recving
            )
        return set(), done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Return and clear the set of block IDs that failed to load.
        This is called by the scheduler to identify blocks that need
        to be retried after a transfer failure.
        """
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    # {(ip, port)]: {local_block_ids: [], remote_block_ids: {}}}
    def _get_kv_split_metadata(self, req_meta: ReqMeta, req_idx: int, req_id: str, group_idx: int):
        remote_pcp_size = req_meta.remote_pcp_size
        remote_dcp_size = req_meta.remote_dcp_size
        remote_tp_size = req_meta.remote_tp_size
        remote_hosts = [req_meta.remote_host]
        remote_port = req_meta.remote_port
        local_transed_tokens = max(req_meta.remote_cache_tokens, req_meta.local_transed_tokens)
        # local_transed_tokens tokens that have already been transmitted on the local side
        local_computed_tokens = req_meta.local_computed_tokens
        prompt_len = req_meta.prompt_len
        p_parallel_info = parallel_info(
            tp_size=self.tp_size,
            pcp_size=self.pcp_size,
            dcp_size=self.dcp_size,
            pd_head_ratio=self.pd_head_ratio,
            use_mla=self.use_mla,
        )
        d_parallel_info = parallel_info(
            tp_size=remote_tp_size,
            pcp_size=remote_pcp_size,
            dcp_size=remote_dcp_size,
            pd_head_ratio=self.pd_head_ratio,
            use_mla=self.use_mla,
        )
        cp_size = self.pcp_size * self.dcp_size
        # to_trans_idx all tokens that have been processed up to the current step
        if req_meta.chunk_finish:
            to_trans_idx = math.ceil(local_computed_tokens / self.block_size[group_idx])
        else:
            to_trans_idx = math.floor(local_computed_tokens / self.block_size[group_idx])
        prompt_block_size = math.ceil(prompt_len / self.block_size[group_idx])
        #
        num_local_blocks = prompt_block_size // cp_size + int(
            (prompt_block_size % cp_size) > (self.pcp_rank * self.dcp_size + self.dcp_rank)
        )
        already_send_blocks = to_trans_idx // cp_size + int(
            (to_trans_idx % cp_size) > (self.pcp_rank * self.dcp_size + self.dcp_rank)
        )
        if num_local_blocks == already_send_blocks:
            req_meta.chunk_finish = True
        transed_idx = math.floor(local_transed_tokens / self.block_size[group_idx])

        p_cp_group = get_cp_group(self.tp_size, self.total_num_kv_heads, self.dcp_size)
        d_cp_group = get_cp_group(remote_tp_size, self.total_num_kv_heads, remote_dcp_size)
        logger.debug("Compute cp group for P&D req_id=%r p_cp_group=%r d_cp_group=%r", req_id, p_cp_group, d_cp_group)

        cp_ratio = len(p_cp_group) // len(d_cp_group)
        if cp_ratio == 0:
            selected_p_cp_groups = p_cp_group
            selected_d_cp_groups = d_cp_group
        else:
            x = req_idx % cp_ratio
            start = x * len(d_cp_group)
            selected_p_cp_groups = p_cp_group[start : (start + len(d_cp_group))]
            selected_d_cp_groups = d_cp_group
        assert len(selected_p_cp_groups) == len(selected_d_cp_groups)

        p_head_group_rank = (self.tp_rank - self.dcp_rank) // self.dcp_size
        selected_p_cp_group = []
        selected_d_cp_group = []
        for idx, cp_group in enumerate(selected_p_cp_groups):
            if p_head_group_rank in cp_group:  # Check whether the rank is in selected_p_cp_groups
                selected_p_cp_group = cp_group
                selected_d_cp_group = selected_d_cp_groups[idx]
        if len(selected_p_cp_group) == 0:
            return {}

        logger.debug(
            "MooncakeLayerwiseConnector _get_kv_split_metadata req_id=%r "
            "P-side selected head_group cp group: %s, D-side selected head_group cp group: %s",
            req_id,
            selected_p_cp_group,
            selected_d_cp_group,
        )

        context_parallel_parameters_check(
            remote_pcp_size, remote_dcp_size, p_parallel_info, d_parallel_info, self.total_num_kv_heads
        )
        p_rank_block_mapping, d_block_rank_mapping, pd_head_mapping, d_trans_count_mapping = (
            get_local_remote_block_port_mappings(
                to_trans_idx,
                p_parallel_info,
                d_parallel_info,
                remote_hosts,
                remote_port,
                selected_p_cp_group,
                selected_d_cp_group,
                prompt_len,
                self.block_size[group_idx],
                req_meta,
                self.total_num_kv_heads,
                req_id,
            )
        )
        transfer_mappings = get_transfer_mappings(
            p_rank_block_mapping,
            d_block_rank_mapping,
            pd_head_mapping,
            d_trans_count_mapping,
            req_meta,
            group_idx,
            p_parallel_info,
            req_id,
            transed_idx,
            to_trans_idx,
            self.tp_rank,
            self.pcp_rank,
            self.dcp_rank,
        )
        return transfer_mappings

    def _get_kv_split_metadata_for_mamba(self, req_meta: ReqMeta, req_idx: int, req_id: str, group_idx: int):
        assert self.tp_size >= req_meta.remote_tp_size, (
            "Mamba group prefill TP_size must equal or larger than decode TP_size."
        )
        remote_tp_size = req_meta.remote_tp_size
        tp_raito = self.tp_size // remote_tp_size
        remote_host = req_meta.remote_host
        remote_port = req_meta.remote_port + self.tp_rank // tp_raito
        transfer_mappings: dict[tuple[str, int], dict[str, Any]] = {}
        if req_meta.chunk_finish:
            transfer_mappings[(remote_host, remote_port)] = {
                "local_block_ids": req_meta.local_block_ids[group_idx],
                "remote_block_ids": req_meta.remote_block_ids[group_idx],
                "trans_count": self.tp_size // remote_tp_size,
            }

        return transfer_mappings

    def _align_remote_block_ids(self, req_meta: ReqMeta):
        remote_block_size = req_meta.remote_block_size
        remote_block_ids = req_meta.remote_block_ids
        for i in range(self.num_kv_cache_groups):
            if isinstance(self.kv_cache_specs[i], MambaSpec):
                continue
            if remote_block_size[i] != self.block_size[i] and len(req_meta.remote_block_ids[i]) > 0:
                assert remote_block_size[i] > self.block_size[i] and remote_block_size[i] % self.block_size[i] == 0, (
                    "Remote block size must be divisible by local block size."
                )
                assert self.pcp_size * self.dcp_size * req_meta.remote_pcp_size * req_meta.remote_dcp_size == 1, (
                    "Context parallel does not support different P/D block size now."
                )
                pd_block_size_ratio = remote_block_size[i] // self.block_size[i]
                remtote_block_ids_with_scale = [
                    block_id * pd_block_size_ratio + j
                    for block_id in remote_block_ids[i]
                    for j in range(pd_block_size_ratio)
                ]
                req_meta.remote_block_ids[i] = remtote_block_ids_with_scale

    def _get_kernel_block_ids(self, block_ids):
        for i in range(self.num_kv_cache_groups):
            if isinstance(self.kv_cache_specs[i], MambaSpec):
                continue
            if len(block_ids[i]) > 0:
                block_ids[i] = [
                    block_id * self.kernel_block_size_scale[i] + j
                    for block_id in block_ids[i]
                    for j in range(self.kernel_block_size_scale[i])
                ]
        return block_ids

    def start_load_kv(self, metadata: MooncakeLayerwiseConnectorMetadata):
        """Track D-side receives without resetting forward/MTP send state."""
        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            plan = getattr(self, "_layerwise_step_plan", None)
            requests = plan.requests if plan is not None else metadata.requests
            for req_id, meta in requests.items():
                if meta.do_virtual:
                    self.virtual_request.add(req_id)
                    continue
                external_req_id = get_external_request_id(req_id)
                assert self.kv_recv_layer_thread is not None
                self.request_map[external_req_id] = req_id
                self._recving_metadata[req_id] = meta

    def _new_layerwise_step_controller(self) -> LayerwiseStepController:
        return LayerwiseStepController(
            vllm_config=self.vllm_config,
            kv_cache_specs=getattr(self, "kv_cache_specs", None),
            num_kv_cache_groups=getattr(self, "num_kv_cache_groups", 0),
            block_size=getattr(self, "block_size", None),
            kernel_block_size_scale=getattr(self, "kernel_block_size_scale", None),
            attn_resharding_group_idx=getattr(self, "attn_resharding_group_idx", None),
            total_layers=getattr(self, "total_layers", 0),
            index_to_name=getattr(self, "index_to_name", None),
            layer_metadata=getattr(self, "layer_metadata", None),
            sparse_sfa_c8_layers=getattr(self, "sparse_sfa_c8_layers", None),
            pd_head_ratio=getattr(self, "pd_head_ratio", None),
            enable_c8_quant=getattr(self, "enable_c8_quant", False),
            enable_kv_quant=getattr(self, "enable_kv_quant", False),
            k_buffer=getattr(self, "k_buffer", None),
            v_buffer=getattr(self, "v_buffer", None),
            kv_send_layer_thread=getattr(self, "kv_send_layer_thread", None),
            resharding_stream=getattr(self, "resharding_stream", None),
            current_layer=getattr(self, "current_layer", 0),
            cache_write_events=getattr(self, "_cache_write_events", None),
            layer_storage_slots=getattr(self, "layer_storage_slots", None),
            storage_send_done_events=getattr(self, "storage_send_done_events", None),
            pending_reuse_layers=getattr(self, "_pending_reuse_layers", None),
            storage_send_errors=getattr(self, "_storage_send_errors", None),
            align_remote_block_ids=self._align_remote_block_ids,
            get_kv_split_metadata=self._get_kv_split_metadata,
            get_kv_split_metadata_for_mamba=self._get_kv_split_metadata_for_mamba,
            get_kernel_block_ids=self._get_kernel_block_ids,
            update_decoder_info=self.update_decoder_info,
            get_nz_cache=self.get_nz_cache,
            mark_layer_reuse_pending=self._mark_layer_reuse_pending,
            complete_layer_reuse=self._complete_layer_reuse,
            send_task_factory=SendTask,
        )

    def prepare_layerwise_step(self, metadata: MooncakeLayerwiseConnectorMetadata) -> None:
        """Prepare layerwise hooks once, when metadata is bound before forward."""
        controller = self._new_layerwise_step_controller()
        plan = controller.prepare(metadata)
        self.current_layer = controller.current_layer
        self._cache_write_events = controller._cache_write_events
        self._layerwise_step_plan = plan

    def on_kv_cache_written(self, layer_name: str) -> None:
        """Thin delegate to the layerwise step controller."""
        controller = self._new_layerwise_step_controller()
        controller.on_written(layer_name)
        self.current_layer = controller.current_layer

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: list[torch.Tensor],
        attn_metadata: "AttentionMetadata",
        connector_metadata: MooncakeLayerwiseConnectorMetadata,
        **kwargs,
    ) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        plan = getattr(self, "_layerwise_step_plan", None)
        if plan is None:
            plan = self._compat_layerwise_plan(connector_metadata)
        controller = self._new_layerwise_step_controller()
        controller.save_layer(layer_name, kv_layer, attn_metadata, plan, **kwargs)
        self.current_layer = controller.current_layer

    def _compat_layerwise_plan(self, connector_metadata: MooncakeLayerwiseConnectorMetadata) -> LayerwiseTransferPlan:
        """Build a read-only plan for direct worker calls that skipped prepare."""
        send_task = connector_metadata.send_task
        return LayerwiseTransferPlan(
            producer=True,
            requests=copy.deepcopy(connector_metadata.requests),
            aligned_remote_block_ids=[],
            send_task_update=None,
            send_task=send_task,
        )

    # NOTE: Due to the FIA operator constraints, the expected kv cache is ND format, NZ shape,
    # while the npu_format_cast method only modifies the memory layout, we manually convert it to NZ shape here
    def get_nz_cache(self, cache_tensor: torch.Tensor, layer_group_idx: int):
        head_num, head_dim = cache_tensor.shape[-2], cache_tensor.shape[-1]
        cache_tensor = cache_tensor.view(-1, self.block_size[layer_group_idx], head_num * head_dim)
        cache_tensor = trans_nd_to_nz(cache_tensor)
        cache_tensor = cache_tensor.reshape(-1, head_num, head_dim)
        return cache_tensor

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        """Get a socket to the remote host."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()

            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(
                ctx=ctx,
                path=remote_path,
                socket_type=zmq.REQ,  # type: ignore
                bind=False,
            )
            sock.setsockopt(
                zmq.SNDTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            # recv() is gated by the shared remote_poller, which also wakes for
            # other peers' sockets; bound it so it can never block forever.
            sock.setsockopt(
                zmq.RCVTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
            self.remote_socket_contexts[sock] = ctx
            return sock

    def _close_remote_sockets(self) -> None:
        """Close the remote sockets and ZMQ contexts created by this worker.

        Best effort and idempotent. The worker never returns remote sockets to
        the pool in the normal path, so every tracked socket is released, not
        only the pooled ones; shutdown joins the layer threads first, so no
        owner is still using them. The recv thread's ROUTER socket is owned by
        ``zmq_ctx`` and is not touched here.
        """
        sockets = getattr(self, "remote_sockets", None)
        contexts = getattr(self, "remote_socket_contexts", None)
        if not sockets and not contexts:
            return
        pooled: dict[str, deque[Any]] = sockets or {}
        tracked: dict[Any, Any] = contexts or {}
        lock = getattr(self, "remote_sockets_lock", None)
        if lock is not None:
            with lock:
                self.remote_sockets = defaultdict(deque)
                self.remote_socket_contexts = {}
        else:
            self.remote_sockets = defaultdict(deque)
            self.remote_socket_contexts = {}
        owned: set[Any] = set(tracked)
        for sock_queue in pooled.values():
            owned.update(sock_queue)
        poller = getattr(self, "remote_poller", None)
        for sock in owned:
            if poller is not None:
                try:
                    poller.unregister(sock)
                except Exception as error:
                    logger.warning("Failed to unregister layerwise remote socket. error=%s. ", error)
            try:
                sock.close(linger=0)
            except Exception as error:
                logger.warning("Failed to close layerwise remote socket. error=%s. ", error)
            ctx = tracked.get(sock)
            if ctx is None:
                continue
            try:
                ctx.destroy(linger=0)
            except Exception as error:
                logger.warning("Failed to destroy layerwise remote socket context. error=%s. ", error)

    def _discard_remote_socket(self, sock: Any) -> None:
        """Release a remote socket that will never be returned to the pool.

        Best effort and idempotent. A failed metadata query leaves a REQ socket
        that is registered in the shared poller and owns a ZMQ context; neither
        is reachable through the socket pool, so keeping them would accumulate
        one poller entry and one native context on every retry.
        """
        if sock is None:
            return
        poller = getattr(self, "remote_poller", None)
        if poller is not None:
            try:
                poller.unregister(sock)
            except Exception as error:
                logger.warning("Failed to unregister layerwise remote socket. error=%s. ", error)
        contexts = getattr(self, "remote_socket_contexts", None)
        lock = getattr(self, "remote_sockets_lock", None)
        ctx = None
        if contexts is not None:
            if lock is not None:
                with lock:
                    ctx = contexts.pop(sock, None)
            else:
                ctx = contexts.pop(sock, None)
        try:
            sock.close(linger=0)
        except Exception as error:
            logger.warning("Failed to close layerwise remote socket. error=%s. ", error)
        if ctx is None:
            return
        try:
            ctx.destroy(linger=0)
        except Exception as error:
            logger.warning("Failed to destroy layerwise remote socket context. error=%s. ", error)

    def update_decoder_info(self, req_id, req_meta: ReqMeta):
        if (
            req_meta.remote_engine_id not in self.remote_layer_metadata
            or req_meta.remote_port not in self.remote_layer_metadata[req_meta.remote_engine_id]
        ):
            sock: Any = None
            try:
                encoded_data = self.encoder.encode((GET_META_MSG, req_id))
                sock = self._get_remote_socket(req_meta.remote_host, req_meta.remote_port)
                path = f"{req_meta.remote_host}:{req_meta.remote_port}"
                ensure_zmq_send(sock, encoded_data, path)
                metadata_bytes = ensure_zmq_recv_with_poller(sock, self.remote_poller, path)
                agent_meta: MooncakeAgentMetadata = self.decoder.decode(metadata_bytes)
            except Exception as e:
                logger.error(
                    "Query to port and kv base addr for request fail. req_id=%s, source=%s:%s, error=%s. ",
                    req_id,
                    req_meta.remote_host,
                    req_meta.remote_port,
                    e,
                )
                self._discard_remote_socket(sock)
                raise e
            assert req_meta.remote_engine_id != self.engine_id, (
                f"Conflict engine id {req_meta.remote_engine_id} with local engine id {self.local_engine_id}."
            )
            self.remote_layer_metadata[req_meta.remote_engine_id][req_meta.remote_port] = agent_meta.layer_metadata
            self.remote_te_port[req_meta.remote_engine_id][req_meta.remote_port] = agent_meta.te_rpc_port
            logger.debug(
                "Query to port and kv base addr for request %s from %s:%s success "
                "agent_meta.layer_metadata=%r agent_meta.te_rpc_port=%r",
                req_id,
                req_meta.remote_host,
                req_meta.remote_port,
                agent_meta.layer_metadata,
                agent_meta.te_rpc_port,
            )
            if self.pd_head_ratio > 1:
                # for tp inequal, pre-create link to prevent alltoall out of memory
                session_id = f"{req_meta.remote_host}:{agent_meta.te_rpc_port}"
                first_layer_name = next(iter(self.layer_metadata.keys()))
                ret = self.engine.batch_transfer_sync_write(
                    session_id,
                    [self.layer_metadata[first_layer_name].kv_caches_base_addr[0]],
                    [agent_meta.layer_metadata[first_layer_name].kv_caches_base_addr[0]],
                    [128],
                )
                if ret < 0:
                    logger.error("Mooncake transfer failed to create link. session_id=%s, ret=%d. ", session_id, ret)
        req_meta.remote_te_rpc_port = self.remote_te_port[req_meta.remote_engine_id][req_meta.remote_port]
        req_meta.remote_layer_metadata = self.remote_layer_metadata[req_meta.remote_engine_id][req_meta.remote_port]
        return req_meta

    def send_done_send_signal(self, req_id, req_meta, group_idx, trans_flag: bool = True):
        external_req_id = get_external_request_id(req_id)
        send_msg_type = DONE_SENDING_MSG if trans_flag else FAILED_SENDING_MSG
        logger.info(
            "Sending transmitting signal %s for request %s to %s:%d",
            send_msg_type,
            external_req_id,
            req_meta.remote_host,
            req_meta.remote_port,
        )
        try:
            path = make_zmq_path("tcp", req_meta.remote_host, req_meta.remote_port)
            msg_encoder = msgspec.msgpack.Encoder()
            side_channel_path = f"{self.side_channel_host}:{self.handshake_port}"
            encoded_data = msg_encoder.encode(
                (send_msg_type, external_req_id, req_meta.trans_count[group_idx], side_channel_path)
            )
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore
                        sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
                        ensure_zmq_send(sock, encoded_data, f"{req_meta.remote_host}:{req_meta.remote_port}")
                        if not sock.poll(int(self.timeout * 1000), zmq.POLLIN):  # type: ignore[attr-defined]
                            raise TimeoutError(
                                f"Timed out waiting for ACK from {req_meta.remote_host}:{req_meta.remote_port}"
                            )
                        ack = sock.recv()
                        if ack != b"ACK":
                            raise ValueError(f"Unexpected ACK response: {ack}")
                        return
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(
                            "Failed to send done sending signal. "
                            "request_id=%s, destination=%s:%d, attempt=%d/%d, error=%s. ",
                            external_req_id,
                            req_meta.remote_host,
                            req_meta.remote_port,
                            attempt,
                            max_retries,
                            e,
                        )
                        time.sleep(0.1)
                    else:
                        raise RuntimeError(f"Failed to receive ACK after {max_retries} attempts: {e}") from e
        except Exception as e:
            logger.error(
                "Sending signal fail. signal_type=%s, request_id=%s, destination=%s:%s, error=%s. ",
                send_msg_type,
                external_req_id,
                req_meta.remote_host,
                req_meta.remote_port,
                e,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass


if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
