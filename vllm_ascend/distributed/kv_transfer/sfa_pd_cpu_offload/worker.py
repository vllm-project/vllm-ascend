# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Worker side of the PD-disaggregated SFA connector (memfabric pull mode).

D (``kv_consumer``): binds to :class:`KVOffloadDecodeManager`'s TP-shared CPU
KV pool and receives indexer KV into rank-local HBM. Every TP rank pulls a
disjoint part of main MLA KV and its rank-local indexer KV. Decode KV continues
to be written directly to the same CPU pool by the decode-offload manager.

P (``kv_producer``): registers its HBM KV with memfabric and runs a pull-mode
sending thread that notifies D to read (no RDMA push). A per-layer
send-completion event gates P's KV buffer reuse.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.network_utils import get_ip
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.kv_offload_decode.kv_offload_decode_manager import (
    get_kv_offload_decode_manager,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    LayerMetadata,
    SendTask,
    get_external_request_id,
    infer_sfa_component_group_ids,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.read_thread import (
    ConsumerReadState,
    MembPullReadThread,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.send_thread import (
    MembPullSendingThread,
    ProducerSendState,
)
from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import (
    BACKEND_MEMFABRIC,
    MEMFABRIC_ROLE_DECODE,
    MEMFABRIC_ROLE_PREFILL,
    global_memfabric_te,
)
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    collect_storage_merged_register_regions,
    get_transfer_timeout_value,
    validate_register_region_count,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata

# Matches the transformer-layer index in a kv-cache layer name, e.g.
# "model.layers.5.self_attn" / "model.layers.5.self_attn.indexer" -> 5. Prefer
# this over extract_layer_index(), which asserts the name holds exactly one
# integer and would raise on names carrying an extra index/shard suffix.
_LAYER_IDX_RE = re.compile(r"layers\.(\d+)")

CONNECTOR_THREAD_STARTUP_TIMEOUT_SECONDS = 10.0
PD_READ_WAIT_LOG_INTERVAL_SECONDS = 10.0
MIN_TCP_PORT = 1
MAX_TCP_PORT = 65535


def _layer_idx(layer_name: str) -> int:
    match = _LAYER_IDX_RE.search(layer_name)
    assert match is not None, f"no transformer layer index in layer name {layer_name!r}"
    return int(match.group(1))


def _resolve_kv_transfer_backend(vllm_config: VllmConfig) -> str:
    """Pick the KV transfer backend.

    ``kv_connector_extra_config["transfer_backend"]`` overrides the
    ``VLLM_ASCEND_KV_TRANSFER_BACKEND`` env var.
    """
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
    return extra.get("transfer_backend") or envs.VLLM_ASCEND_KV_TRANSFER_BACKEND


def _validate_tcp_port(port: int, *, description: str) -> None:
    if not MIN_TCP_PORT <= port <= MAX_TCP_PORT:
        raise ValueError(f"{description} must be in [{MIN_TCP_PORT}, {MAX_TCP_PORT}], got {port}")


class SFAPDCpuOffloadConsumerWorker:
    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.use_layerwise = use_layerwise
        self.tp_rank = get_tensor_model_parallel_rank()  # TP-local rank for the per-rank ZMQ port
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.side_channel_host = get_ip()
        # D-side ZMQ control-plane base port; each TP rank listens on base + tp_rank.
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank * vllm_config.parallel_config.tensor_parallel_size
        )
        highest_side_channel_port = self.side_channel_port + vllm_config.parallel_config.tensor_parallel_size - 1
        _validate_tcp_port(
            highest_side_channel_port,
            description="SFAPD D-side highest TP control-plane port",
        )

        self.layer_metadata: dict[str, LayerMetadata] = {}
        self.engine = None

        self.decode_manager = None
        self._cpu_blocks_by_req: dict[str, int] = {}
        self._invalid_block_ids: set[int] = set()
        # external_req_id -> internal_req_id, so get_finished can map the recv
        # thread's done_recving (keyed by external id from P's DONE signal) back
        # to the vLLM-internal id that the scheduler expects.
        self.request_map: dict[str, str] = {}
        # external_req_id -> (main CPU block ids, indexer HBM block ids).
        self._dest_blocks_by_req: dict[str, tuple[list[int], list[int]]] = {}
        # External req ids whose DONE signal arrived before request_map
        # was seeded (see get_finished). Retried every step until mapped.
        self._pending_done: set[str] = set()
        # Keep rank-local terminal state (success or failure) until every TP
        # rank has finished the same request. This is scheduler readiness state,
        # not a per-layer barrier.
        self._terminal_ext_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Common
    # ------------------------------------------------------------------
    def _ensure_engine(self):
        if self.engine is None:
            backend = _resolve_kv_transfer_backend(self.vllm_config)
            if backend != BACKEND_MEMFABRIC:
                raise RuntimeError(
                    "SFAPDCpuOffloadConnector D side supports MemFabric pull only (set transfer_backend=memfabric)."
                )
            global_memfabric_te.configure(
                role=MEMFABRIC_ROLE_DECODE,
                device_id=torch.npu.current_device(),
            )
            self.engine = global_memfabric_te.get_transfer_engine(self.side_channel_host)
        return self.engine

    # ------------------------------------------------------------------
    # D side (kv_consumer) — this class is only instantiated for consumers;
    # Producers use :class:`SFAPDCpuOffloadProducerWorker`.
    # ------------------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Bind MemFabric destinations owned by KVOffloadDecodeManager."""
        assert _resolve_kv_transfer_backend(self.vllm_config) == BACKEND_MEMFABRIC, (
            "SFAPDCpuOffloadConnector D side supports memfabric pull only (set transfer_backend=memfabric)."
        )
        self.decode_manager = get_kv_offload_decode_manager()
        if not hasattr(self.decode_manager, "offload_layer_names"):
            raise RuntimeError(
                "KVOffloadDecodeManager.register_kv_caches must run before the PD connector is registered"
            )
        self._register_memfabric_pull(kv_caches)

    # -- D-side forwards to the composed SFA worker (LRU load path) --
    def start_load_kv(self, metadata: KVConnectorMetadata):
        for req in getattr(metadata, "requests", []):
            req_id = getattr(req, "req_id", None)
            if req_id is not None:
                ext_id = get_external_request_id(req_id)
                self.request_map[ext_id] = req_id
                main_ids = list(getattr(req, "main_block_ids", []) or [])
                indexer_ids = list(getattr(req, "indexer_block_ids", []) or [])
                self._dest_blocks_by_req[ext_id] = (main_ids, indexer_ids)
                self._cpu_blocks_by_req[req_id] = len(main_ids)
                if envs.VLLM_ASCEND_SFA_DEBUG:
                    logger.info(
                        "MembPull D stored dest blocks req %s: indexer_hbm_ids=%s, main_cpu_ids=%s",
                        ext_id,
                        indexer_ids,
                        main_ids,
                    )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        return

    def wait_for_save(self):
        return

    def _cleanup_request_state(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            ext_id = get_external_request_id(req_id)
            self._cpu_blocks_by_req.pop(req_id, None)
            self.request_map.pop(ext_id, None)
            self._dest_blocks_by_req.pop(ext_id, None)
            self._pending_done.discard(ext_id)
            self._terminal_ext_ids.discard(ext_id)

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

        # memfabric pull mode: done comes from MembPullReadThread
        if hasattr(self, "_mf_read_thread") and self._mf_read_thread is not None:
            local_done = self._mf_read_thread.get_and_clear_done()
            local_failed = self._mf_read_thread.get_and_clear_failed()
            self._terminal_ext_ids.update(local_done | local_failed)

        tp_status = self._gather_tp_read_status(
            self._terminal_ext_ids,
            local_failed,
        )
        finished_on_all = (
            set.intersection(*(terminal for terminal, _ in tp_status))
            if tp_status
            else set()
        )
        failed_on_any = set().union(*(failed for _, failed in tp_status))
        self._terminal_ext_ids.difference_update(finished_on_all)

        for ext_id in failed_on_any:
            dest = self._dest_blocks_by_req.get(ext_id)
            if dest is None:
                continue
            main_block_ids, indexer_block_ids = dest
            self._invalid_block_ids.update(main_block_ids)
            self._invalid_block_ids.update(indexer_block_ids)

        if finished_on_all or self._pending_done:
            still_pending: set[str] = set()
            for ext_id in finished_on_all | self._pending_done:
                internal = self.request_map.get(ext_id)
                if internal is not None:
                    done_recving.add(internal)
                else:
                    still_pending.add(ext_id)
            self._pending_done = still_pending

            if done_recving or self._pending_done:
                if envs.VLLM_ASCEND_SFA_DEBUG:
                    logger.info(
                        "MembPull D get_finished: finished_all_tp_ext=%s, "
                        "done_recving_internal=%s, pending_done_ext=%s",
                        finished_on_all,
                        done_recving,
                        self._pending_done,
                    )
        # else: read thread not up yet -> nothing finished (done_recving empty).

        # Purge scheduler-finished req state AFTER resolving this step's
        # done signals against request_map. Doing it at the top would pop
        # request_map[ext_id] and discard _pending_done[ext_id] before the
        # resolution loop above, leaking any finished req whose DONE arrives in
        # the same step (unmappable -> stuck in _pending_done forever).
        if finished_req_ids:
            self._cleanup_request_state(finished_req_ids)

        return set(), done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        """Per-req actual main-MLA CPU-block count for the solution-1 threshold."""
        if self.decode_manager is None:
            return None
        result = {rid: self._cpu_blocks_by_req[rid] for rid in req_ids if rid in self._cpu_blocks_by_req}
        return result or None

    def _build_consumer_read_state(self) -> ConsumerReadState:
        assert self.decode_manager is not None
        return ConsumerReadState(
            num_blocks=self.kv_cache_config.num_blocks,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            layer_metadata=self.layer_metadata,
            main_name_to_idx=self._main_name_to_idx,
            cpu_pools=self._cpu_pools,
            main_gva_bases=self._main_gva_bases,
            main_block_lens=self._main_block_lens,
            indexer_tensors=self._indexer_tensors,
            indexer_scale_tensors=self._indexer_scale_tensors,
            dest_blocks_by_req=self._dest_blocks_by_req,
            get_offload_layer_id=self.decode_manager._get_offload_layer_id,
        )

    def _register_memfabric_pull(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Start D pull thread with manager CPU KV and rank-local indexer HBM."""
        assert self.decode_manager is not None
        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks
        main_names = list(self.decode_manager.offload_layer_names)
        indexer_by_layer = {_layer_idx(name): name for name in kv_caches if "indexer" in name.lower()}

        # Store layer info for MembPullReadThread
        self._main_names = main_names
        self._main_name_to_idx = {n: i for i, n in enumerate(main_names)}
        k_caches_cpu = self.decode_manager.k_caches_cpu
        v_caches_cpu = self.decode_manager.v_caches_cpu
        gvas_k = self.decode_manager.gvas_k_bases
        gvas_v = self.decode_manager.gvas_v_bases
        if len(gvas_k) != len(main_names) or len(gvas_v) != len(main_names):
            raise RuntimeError("KVOffloadDecodeManager shared CPU GVA/layer count mismatch")
        self._main_gva_bases = list(zip(gvas_k, gvas_v))
        self._main_block_lens = self.decode_manager.cpu_block_lens
        if len(self._main_block_lens) != len(main_names):
            raise RuntimeError("KVOffloadDecodeManager shared CPU block-size/layer count mismatch")
        if self.tp_rank == 0:
            if len(k_caches_cpu) != len(main_names) or len(v_caches_cpu) != len(main_names):
                raise RuntimeError("KVOffloadDecodeManager CPU pool/layer count mismatch")
            self._cpu_pools = list(zip(k_caches_cpu, v_caches_cpu))
        else:
            self._cpu_pools = [None] * len(main_names)
        self._indexer_tensors = []
        self._indexer_scale_tensors: list[torch.Tensor | None] = []
        for main_name in main_names:
            indexer_name = indexer_by_layer.get(_layer_idx(main_name))
            if indexer_name is None:
                self._indexer_tensors.append(None)
                self._indexer_scale_tensors.append(None)
                continue
            indexer_tuple = kv_caches[indexer_name]
            if not isinstance(indexer_tuple, (list, tuple)):
                indexer_tuple = (indexer_tuple,)
            self._indexer_tensors.append(indexer_tuple[0])
            self._indexer_scale_tensors.append(indexer_tuple[1] if len(indexer_tuple) > 1 else None)

        main_group_idx, indexer_group_idx = infer_sfa_component_group_ids(self.kv_cache_config)
        for pool_idx, mname in enumerate(main_names):
            indexer_t = self._indexer_tensors[pool_idx]
            indexer_scale_t = self._indexer_scale_tensors[pool_idx]
            cpu_pool = self._cpu_pools[pool_idx]
            if cpu_pool is not None:
                k_cpu, v_cpu = cpu_pool
                addrs = [k_cpu.data_ptr(), v_cpu.data_ptr()]
                block_lens = [
                    k_cpu.element_size() * math.prod(k_cpu.shape[1:]),
                    v_cpu.element_size() * math.prod(v_cpu.shape[1:]),
                ]
                scales = [k_cpu.shape[0] // num_blocks, v_cpu.shape[0] // num_blocks]
            else:
                addrs, block_lens, scales = [], [], []
            groups = [main_group_idx] * len(addrs)
            if indexer_t is not None:
                addrs.append(indexer_t.data_ptr())
                block_lens.append(indexer_t.element_size() * math.prod(indexer_t.shape[1:]))
                scales.append(indexer_t.shape[0] // num_blocks)
                groups.append(indexer_group_idx)
            if indexer_scale_t is not None:
                addrs.append(indexer_scale_t.data_ptr())
                block_lens.append(indexer_scale_t.element_size() * math.prod(indexer_scale_t.shape[1:]))
                scales.append(indexer_scale_t.shape[0] // num_blocks)
                groups.append(indexer_group_idx)
            self.layer_metadata[mname] = LayerMetadata(
                tensor_group_idx=groups,
                kv_caches_base_addr=addrs,
                block_len=block_lens,
                block_size_scale=scales,
                main_tensor_count=2 if cpu_pool is not None else 0,
                has_indexer=indexer_t is not None,
            )

        # Create memfabric engine (no registration)
        self._ensure_engine()
        read_state = self._build_consumer_read_state()
        # Start MembPullReadThread (ZMQ ROUTER + memfabric read)
        self._mf_read_thread = MembPullReadThread(
            tp_rank=self.tp_rank,
            side_channel_port=self.side_channel_port,
            engine=self.engine,
            state=read_state,
        )
        self._mf_read_thread.start()
        if not self._mf_read_thread.ready_event.wait(timeout=CONNECTOR_THREAD_STARTUP_TIMEOUT_SECONDS):
            self._mf_read_thread.stop()
            raise RuntimeError("Timed out waiting for the SFAPD D-side read thread to start")
        if self._mf_read_thread.startup_error is not None:
            error = self._mf_read_thread.startup_error
            self._mf_read_thread.stop()
            raise RuntimeError("SFAPD D-side read thread failed during startup") from error
        logger.info(
            "SFAPDCpuOffload D-side registered (memfabric pull): "
            "%d indexer + %d TP-shared main layers",
            sum(t is not None for t in self._indexer_tensors),
            len(main_names),
        )

    def shutdown(self) -> None:
        read_thread = getattr(self, "_mf_read_thread", None)
        if read_thread is not None:
            read_thread.stop()


class SFAPDCpuOffloadProducerWorker:
    """P-side worker for memfabric pull mode.

    It registers P's local KV tensors with memfabric and runs a pull-mode
    sending thread. P never pushes KV; it sends READ_READY_BATCH messages so D
    can read P's source blocks and reply with READ_DONE / READ_FAILED.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        # Preserve the Mooncake worker's transfer-engine timeout setup. The
        # memfabric engine reads this during construction.
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
        self._backend = _resolve_kv_transfer_backend(vllm_config)
        if self._backend != BACKEND_MEMFABRIC:
            raise RuntimeError(
                "SFAPDCpuOffloadConnector P side supports MemFabric pull only (set transfer_backend=memfabric)."
            )
        global_memfabric_te.configure(
            role=MEMFABRIC_ROLE_PREFILL,
            device_id=torch.npu.current_device(),
        )
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.engine_id = engine_id
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.side_channel_host = get_ip()
        self.side_channel_port = vllm_config.kv_transfer_config.kv_port + self.dp_rank * self.tp_size
        self.total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.last_layer_idx = self.total_layers - 1
        self.engine = global_memfabric_te.get_transfer_engine(self.side_channel_host)
        self.te_rpc_port = self.engine.get_rpc_port()
        self.kv_cache_specs = [group_spec.kv_cache_spec for group_spec in self.kv_cache_config.kv_cache_groups]
        self.block_size = [spec.block_size for spec in self.kv_cache_specs]
        self.num_kv_cache_groups = len(self.kv_cache_specs)
        self.main_group_idx, self.indexer_group_idx = infer_sfa_component_group_ids(self.kv_cache_config)
        self.use_mla = self.vllm_config.model_config.use_mla
        self.layer_metadata: dict[str, LayerMetadata] = {}
        self.index_to_name: dict[int, str] = {}
        # A layer can touch a main storage slot and, optionally, a separate
        # indexer storage slot. The send thread owns one completion gate per
        # physical storage slot so reuse is safe across layer and step
        # boundaries, including the last -> first transition of a reuse ring.
        self.layer_storage_slots: dict[int, tuple[int, ...]] = {}
        self.current_layer = 0
        self.kv_send_layer_thread: MembPullSendingThread | None = None
        # Layers whose PD send was already dispatched at scatter time by
        # on_kv_cache_written; save_kv_layer skips these at layer end.
        self._pd_dispatched_layers: set[int] = set()

    def get_finished(self) -> tuple[set[str], set[str]]:
        return set(), set()

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        return None

    def update_decoder_info(self, req_id: str, req_meta: Any) -> Any:
        """Override: in memfabric pull mode, P does NOT need D's metadata
        (P is not pushing to D — D reads from P). Skip GET_META entirely
        to avoid flooding D's ROUTER with 61 unnecessary requests that
        delay MF_META / READ_READY_BATCH."""
        if self._backend == BACKEND_MEMFABRIC:
            return req_meta
        raise RuntimeError("SFAPDCpuOffloadConnector P side supports memfabric pull only.")

    def start_load_kv(self, metadata: KVConnectorMetadata) -> None:
        """Prepare P-side request metadata for memfabric pull mode.

        * reset ``self.current_layer`` — the per-step layer counter that
          ``save_kv_layer`` increments; without the reset it drifts to
          ``>= total_layers`` and every request after the first is skipped.
        * adjust ``remote_port`` by ``tp_rank`` — D's ROUTER binds
          ``side_channel_port + tp_rank`` (one per rank) but D advertises the
          base port, so each P rank must send to ``base + tp_rank``.

        ``remote_host`` / ``local_block_ids`` are already correct from
        ``build_connector_meta``; main and indexer group ids remain separate."""
        if self._backend == BACKEND_MEMFABRIC:
            self.current_layer = 0
            self._pd_dispatched_layers = set()
            for req_id, req_meta in getattr(metadata, "requests", {}).items():
                if req_meta.remote_port is None:
                    continue
                remote_tp_size = req_meta.remote_tp_size
                if remote_tp_size is None:
                    remote_tp_size = self.tp_size
                remote_tp_rank = self._map_prefill_rank_to_decode_rank(
                    prefill_tp_size=self.tp_size,
                    decode_tp_size=remote_tp_size,
                    prefill_tp_rank=self.tp_rank,
                )
                tp_ratio = self.tp_size // remote_tp_size
                old_remote_port = req_meta.remote_port
                req_meta.remote_port = req_meta.remote_port + remote_tp_rank
                _validate_tcp_port(
                    req_meta.remote_port,
                    description="SFAPD remote D-side TP control-plane port",
                )
                if envs.VLLM_ASCEND_SFA_DEBUG:
                    logger.info(
                        "MembPull P start_load_kv req %s: remote_host=%s, "
                        "remote_port=%s->%s, tp_rank=%s, tp_ratio=%s, local_block_ids=%s, "
                        "chunk_finish=%s, local_computed_tokens=%s, local_transed_tokens=%s",
                        req_id,
                        req_meta.remote_host,
                        old_remote_port,
                        req_meta.remote_port,
                        self.tp_rank,
                        tp_ratio,
                        req_meta.local_block_ids,
                        req_meta.chunk_finish,
                        req_meta.local_computed_tokens,
                        req_meta.local_transed_tokens,
                    )
            return
        raise RuntimeError("SFAPDCpuOffloadConnector P side supports memfabric pull only.")

    @staticmethod
    def _map_prefill_rank_to_decode_rank(
        *,
        prefill_tp_size: int,
        decode_tp_size: int,
        prefill_tp_rank: int,
    ) -> int:
        if prefill_tp_size < 1 or decode_tp_size < 1:
            raise ValueError(
                f"SFAPD P/D tensor parallel sizes must both be positive, got P={prefill_tp_size}, D={decode_tp_size}"
            )
        if prefill_tp_size < decode_tp_size or prefill_tp_size % decode_tp_size != 0:
            raise ValueError(
                "SFAPD requires P tensor parallel size to be greater than or "
                "equal to, and divisible by, D tensor parallel size; "
                f"got P={prefill_tp_size}, D={decode_tp_size}"
            )
        if not 0 <= prefill_tp_rank < prefill_tp_size:
            raise ValueError(f"SFAPD P tensor parallel rank {prefill_tp_rank} is outside [0, {prefill_tp_size})")
        tp_ratio = prefill_tp_size // decode_tp_size
        return prefill_tp_rank // tp_ratio

    def _build_producer_send_state(self) -> ProducerSendState:
        return ProducerSendState(
            last_layer_idx=self.last_layer_idx,
            layer_metadata=self.layer_metadata,
            p_session=global_memfabric_te.unique_id,
            main_group_idx=self.main_group_idx,
            indexer_group_idx=self.indexer_group_idx,
            block_sizes=tuple(self.block_size),
            layer_storage_slots=self.layer_storage_slots,
        )

    @staticmethod
    def _infer_layer_storage_slots(
        layer_metadata: dict[str, LayerMetadata],
    ) -> dict[int, tuple[int, ...]]:
        """Map each layer to the physical component slots it may overwrite."""
        slot_by_storage: dict[tuple[str, tuple[int, ...]], int] = {}
        layer_storage_slots: dict[int, tuple[int, ...]] = {}
        for layer_name, metadata in layer_metadata.items():
            main_count = metadata.main_tensor_count
            component_keys = [("main", tuple(metadata.kv_caches_base_addr[:main_count]))]
            if metadata.has_indexer:
                component_keys.append(("indexer", tuple(metadata.kv_caches_base_addr[main_count:])))
            slots = []
            for storage_key in component_keys:
                slot_id = slot_by_storage.setdefault(storage_key, len(slot_by_storage))
                slots.append(slot_id)
            layer_storage_slots[_layer_idx(layer_name)] = tuple(slots)
        return layer_storage_slots

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # memfabric pull mode only.
        assert self._backend == BACKEND_MEMFABRIC, "SFAPDCpuOffloadConnector P side supports memfabric pull only."
        layer2group_ids: dict[str, int] = {}
        for group_idx, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                layer2group_ids[layer_name] = group_idx

        num_blocks = self.kv_cache_config.num_blocks
        main_by_layer = {_layer_idx(name): name for name in kv_caches if "indexer" not in name.lower()}
        indexer_by_layer = {_layer_idx(name): name for name in kv_caches if "indexer" in name.lower()}
        if not main_by_layer:
            raise RuntimeError("SFAPD producer did not find main SFA KV cache layers")

        def _append_cache_tensors(
            layer_meta: LayerMetadata,
            cache_or_caches: Any,
            group_idx: int,
        ) -> None:
            if not isinstance(cache_or_caches, (list, tuple)):
                cache_or_caches = (cache_or_caches,)
            for single_kv_cache in cache_or_caches:
                tensor_num_blocks = single_kv_cache.shape[0]
                if tensor_num_blocks % num_blocks != 0:
                    raise ValueError("The external block size must be an integer multiple of the kernel block size.")
                layer_meta.tensor_group_idx.append(group_idx)
                layer_meta.kv_caches_base_addr.append(single_kv_cache.data_ptr())
                layer_meta.block_len.append(single_kv_cache.element_size() * math.prod(single_kv_cache.shape[1:]))
                layer_meta.block_size_scale.append(tensor_num_blocks // num_blocks)

        for physical_idx, main_name in sorted(main_by_layer.items()):
            layer_meta = LayerMetadata([], [], [], [])
            _append_cache_tensors(layer_meta, kv_caches[main_name], layer2group_ids[main_name])
            layer_meta.main_tensor_count = len(layer_meta.kv_caches_base_addr)
            if layer_meta.main_tensor_count != 2:
                raise RuntimeError(
                    f"SFAPD producer layer {main_name} must expose main K/V tensors, "
                    f"got {layer_meta.main_tensor_count} tensor(s)"
                )
            indexer_name = indexer_by_layer.get(physical_idx)
            if indexer_name is not None:
                _append_cache_tensors(
                    layer_meta,
                    kv_caches[indexer_name],
                    layer2group_ids[indexer_name],
                )
                layer_meta.has_indexer = True
            self.layer_metadata[main_name] = layer_meta
            self.index_to_name[physical_idx] = main_name

        self.last_layer_idx = max(main_by_layer)
        self.total_layers = self.last_layer_idx + 1

        # Infer physical storage slots directly from component addresses.
        # Main and indexer storage are tracked independently: a main-only layer
        # can still share its main slot with a later layer that owns an indexer.
        self.layer_storage_slots = self._infer_layer_storage_slots(self.layer_metadata)

        register_regions = collect_storage_merged_register_regions(kv_caches)
        validate_register_region_count(register_regions)
        global_memfabric_te.register_buffer(register_regions.ptrs, register_regions.lengths)

        ready_event = threading.Event()
        send_state = self._build_producer_send_state()
        self.kv_send_layer_thread = MembPullSendingThread(
            ready_event=ready_event,
            state=send_state,
        )
        self.kv_send_layer_thread.start()
        if not ready_event.wait(timeout=CONNECTOR_THREAD_STARTUP_TIMEOUT_SECONDS):
            self.kv_send_layer_thread.stop()
            raise RuntimeError("Timed out waiting for the SFAPD P-side send thread to start")
        if self.kv_send_layer_thread.startup_error is not None:
            error = self.kv_send_layer_thread.startup_error
            self.kv_send_layer_thread.stop()
            raise RuntimeError("SFAPD P-side send thread failed during startup") from error
        # Stash source tensors on the sending thread for env-gated verify
        # checksums (VLLM_ASCEND_MF_VERIFY=1): P sums its source blocks so
        # the user can compare against D's destination sums in the logs.
        self.kv_send_layer_thread._source_kv_caches = kv_caches
        logger.info(
            "MembPull P registered kv caches: layers=%d, p_session=%s",
            len(self.layer_metadata),
            global_memfabric_te.unique_id,
        )

    def _has_memfabric_pull_target(
        self,
        connector_metadata: KVConnectorMetadata,
        layer_idx: int,
        layer_group_indices: set[int],
    ) -> bool:
        for req_meta in getattr(connector_metadata, "requests", {}).values():
            has_endpoint = bool(req_meta.remote_host) and bool(req_meta.remote_port)
            if not has_endpoint:
                continue
            local_block_ids = req_meta.local_block_ids
            has_blocks = any(
                len(local_block_ids) > group_idx and bool(local_block_ids[group_idx])
                for group_idx in layer_group_indices
            )
            chunk_done = layer_idx == self.last_layer_idx and req_meta.chunk_finish
            if has_blocks or chunk_done:
                return True
        return False

    def on_kv_cache_written(self, layer_name: str, connector_metadata: KVConnectorMetadata) -> None:
        """Record scatter completion and dispatch the PD send pre-attention.

        The slot reuse gate (`wait_for_layer_send`) is untouched; only the
        D-notification is moved earlier so co-located pull overlaps attention.
        Idempotent per layer — save_kv_layer skips layers dispatched here.
        """
        if self._backend != BACKEND_MEMFABRIC or self.kv_send_layer_thread is None:
            return
        resolved_layer_name = layer_name or self.index_to_name.get(self.current_layer)
        if resolved_layer_name is None:
            return
        layer_idx = _layer_idx(resolved_layer_name)
        if layer_idx >= self.total_layers or layer_idx in self._pd_dispatched_layers:
            return
        if not getattr(connector_metadata, "requests", None):
            return
        self._pd_dispatched_layers.add(layer_idx)
        layer_group_indices = set(self.layer_metadata[resolved_layer_name].tensor_group_idx)
        if self._has_memfabric_pull_target(connector_metadata, layer_idx, layer_group_indices):
            self.kv_send_layer_thread.mark_layer_pending(layer_idx)
        # Fresh compute-stream event right after the scatter; it supersedes the
        # layer-end wait_event in the send thread.
        self.kv_send_layer_thread.record_p_save_event(layer_idx)
        self._enqueue_layer_send(resolved_layer_name, layer_idx, connector_metadata)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
        connector_metadata: KVConnectorMetadata,
        **kwargs,
    ) -> None:
        if self._backend != BACKEND_MEMFABRIC:
            raise RuntimeError("SFAPDCpuOffloadConnector P side supports memfabric pull only.")
        send_thread = self.kv_send_layer_thread
        if send_thread is None:
            raise RuntimeError(
                "SFAPD P-side send thread is unavailable; "
                "register_kv_caches() must complete before save_kv_layer()"
            )
        resolved_layer_name = layer_name or self.index_to_name.get(self.current_layer)
        if resolved_layer_name is None:
            return
        layer_idx = _layer_idx(resolved_layer_name)
        if layer_idx >= self.total_layers:
            self.current_layer += 1
            return
        if layer_idx in self._pd_dispatched_layers:
            self.current_layer += 1
            return
        if not getattr(connector_metadata, "requests", None):
            return
        self._pd_dispatched_layers.add(layer_idx)
        layer_group_indices = set(self.layer_metadata[resolved_layer_name].tensor_group_idx)
        if self._has_memfabric_pull_target(connector_metadata, layer_idx, layer_group_indices):
            send_thread.mark_layer_pending(layer_idx)
        # Fallback path (attention hook did not fire for this layer): record
        # scatter completion now and pick the reshape/event to wait on.
        send_thread.record_p_save_event(layer_idx)
        layer_attn_metadata = None
        if self.use_mla and hasattr(attn_metadata, "__getitem__"):
            try:
                layer_attn_metadata = attn_metadata[resolved_layer_name]
            except Exception:
                layer_attn_metadata = None
        if layer_attn_metadata is not None and hasattr(layer_attn_metadata, "reshape_cache_event"):
            wait_event = layer_attn_metadata.reshape_cache_event
        elif hasattr(attn_metadata, "reshape_cache_event"):
            wait_event = attn_metadata.reshape_cache_event
        else:
            wait_event = torch.npu.Event()
            wait_event.record()
        self._enqueue_layer_send(resolved_layer_name, layer_idx, connector_metadata, wait_event=wait_event)
        self.current_layer += 1

    def _enqueue_layer_send(
        self,
        layer_name: str,
        layer_idx: int,
        connector_metadata: KVConnectorMetadata,
        wait_event: torch.npu.Event | None = None,
    ) -> None:
        """Build and queue this layer's PD send task (READ_READY to D)."""
        assert self.kv_send_layer_thread is not None
        layer_send_task = SendTask(
            send_request={},
            wait_event=wait_event,
            layer_idx=layer_idx,
            layer_name=layer_name,
        )
        for req_id, req_meta in connector_metadata.requests.items():
            local_block_ids = req_meta.local_block_ids
            has_main = len(local_block_ids) > self.main_group_idx and bool(local_block_ids[self.main_group_idx])
            layer_has_indexer = self.layer_metadata[layer_name].has_indexer
            has_indexer = not layer_has_indexer or (
                len(local_block_ids) > self.indexer_group_idx and bool(local_block_ids[self.indexer_group_idx])
            )
            if not has_main or not has_indexer:
                continue
            layer_send_task.send_request[req_id] = self.update_decoder_info(req_id, req_meta)
        if layer_send_task.send_request:
            self.kv_send_layer_thread.send_queue.put(layer_send_task)
        else:
            self.kv_send_layer_thread._signal_layer_done(layer_idx)

    def _wait_for_pd_read_completion(
        self,
        event: threading.Event | None,
        error_getter: Callable[[], str | None],
        description: str,
    ) -> None:
        if event is None or self.kv_send_layer_thread is None:
            return
        while not event.wait(timeout=PD_READ_WAIT_LOG_INTERVAL_SECONDS):
            error = error_getter()
            if error is not None:
                raise RuntimeError(f"D failed to read {description}: {error}")
            if not self.kv_send_layer_thread.is_alive():
                startup_error = self.kv_send_layer_thread.startup_error
                detail = f": {startup_error}" if startup_error is not None else ""
                raise RuntimeError(
                    f"SFAPD P-side send thread stopped while waiting for D to read {description}{detail}"
                )
            logger.info("Waiting for D to read %s; keep waiting", description)

        error = error_getter()
        if error is not None:
            raise RuntimeError(f"D failed to read {description}: {error}")

    def wait_for_layer_send(self, layer_idx: int) -> None:
        """Block until D has read layer ``layer_idx``'s KV (buffer-reuse gate).

        In pull mode D reads P's KV via memfabric; this waits until D replies
        with READ_DONE or READ_FAILED before P reuses the KV buffer for a later
        layer, so D is no longer reading before P overwrites it.
        """
        if self.kv_send_layer_thread is None:
            return
        storage_slots = self.layer_storage_slots.get(layer_idx, ())
        if storage_slots:
            for slot_id in storage_slots:
                self._wait_for_pd_read_completion(
                    self.kv_send_layer_thread.get_storage_send_event(slot_id),
                    lambda slot_id=slot_id: self.kv_send_layer_thread.get_storage_error(slot_id),
                    f"physical KV storage slot {slot_id} for layer {layer_idx}",
                )

    def shutdown(self) -> None:
        if self.kv_send_layer_thread is not None:
            self.kv_send_layer_thread.stop()
