from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import regex as re
import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

CED_LAYERWISE_CONFIG_KEY = "CED_layerwise"

# Lease must outlive the longest possible step between session allocation and
# publication so an abandoned WRITING object cannot be reclaimed mid-write.
CED_SESSION_LEASE_TTL_MS = 120_000

# memcache batch_copy directions (kv_transfer.py KVTransferThread).
_MMC_COPY_L2G = 0  # save: device -> pool
_MMC_COPY_G2L = 1  # load: pool -> device

# batch_alloc returns UINT64_MAX for SSD-resident hits; anything at or above
# this bound is not a local DRAM GVA and cannot back a session write.
_SSD_HIT_GVA = (1 << 64) - 1
CED_GATE_POLL_INTERVAL_S = 0.1
CED_LEASE_SAFETY_MARGIN_S = 1.0

_CED_ARCHITECTURES = frozenset(
    {
        "DeepseekV41ForCausalLM",
        "DeepseekV41ForConditionalGeneration",
    }
)

_LAYER_INDEX_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def _layer_index(layer_name: str) -> int | None:
    match = _LAYER_INDEX_PATTERN.search(f".{layer_name}")
    return int(match.group(1)) if match else None


def _is_indexer_member(layer_name: str) -> bool:
    return ".indexer.k_cache" in layer_name


def _is_draft_member(layer_name: str) -> bool:
    return ".mtp." in f".{layer_name}"


def is_ced_model(vllm_config: VllmConfig) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    architectures = getattr(model_config, "architectures", None) or ()
    return any(architecture in _CED_ARCHITECTURES for architecture in architectures)


def resolve_ced_layerwise(vllm_config: VllmConfig) -> bool:
    """Resolve the ``CED_layerwise`` pool extra-config switch.

    Enables the CED (DeepSeek-V4.1) per-source-slice session save and the
    dependency-unit pipelined load on top of the non-layerwise bulk path.
    The switch is rejected together with ``use_layerwise`` and silently
    ignored on non-V4.1 models so existing deployments keep their behavior.
    """
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if not extra_config.get(CED_LAYERWISE_CONFIG_KEY, False):
        return False
    if extra_config.get("use_layerwise", False):
        raise ValueError(
            "CED_layerwise cannot be combined with use_layerwise: the CED path "
            "is a bulk (non-layerwise) pool extension. Set use_layerwise=false."
        )
    if not is_ced_model(vllm_config):
        logger.warning(
            "CED_layerwise only supports DeepSeek-V4.1 architectures %s; ignoring the switch.",
            sorted(_CED_ARCHITECTURES),
        )
        return False
    return True


@dataclass(frozen=True)
class CEDGroupSlice:
    """One producer-ready descriptor range of a bulk pool group object.

    ``member_start``/``member_end`` index the registered transfer descriptors,
    including scale and kernel segments, rather than resource names.
    """

    group_id: int
    member_start: int
    member_end: int
    trigger_layer: int
    slice_index: int = 0


@dataclass(frozen=True)
class CEDTriggerMap:
    """Derived save/load unit map for the CED pool path.

    Built from the cacheable cache groups plus the model topology, so
    ``swa_bounded_replay`` and MTP variations converge without hardcoded
    layer numbers. Groups without entries (compressor state, draft, unknown
    layouts) keep the existing step-end bulk save.
    """

    slices_by_group: dict[int, tuple[CEDGroupSlice, ...]] = field(default_factory=dict)
    slices_by_trigger: dict[int, tuple[CEDGroupSlice, ...]] = field(default_factory=dict)
    publish_layers: dict[int, int] = field(default_factory=dict)

    def trigger_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.slices_by_trigger))


def build_ced_trigger_map(
    group_layer_names: dict[int, Sequence[str]],
    kv_source_layer_ids: Sequence[int] = (),
    group_entry_offsets: dict[int, Sequence[int]] | None = None,
) -> CEDTriggerMap:
    """Map physical producers to registered bulk descriptor ranges.

    Registration sorts resources within each physical layer and expands
    index K/scale and kernel-page segments. Resource-list positions are
    therefore not descriptor positions. Use registration's layer boundaries.
    Unknown or incomplete groups remain wholly on the bulk path.
    """
    by_group: dict[int, tuple[CEDGroupSlice, ...]] = {}
    by_trigger: dict[int, list[CEDGroupSlice]] = {}
    for group_id, names in sorted(group_layer_names.items()):
        if not names or any(_is_draft_member(name) for name in names):
            continue
        layers: dict[int, list[str]] = {}
        for name in names:
            layer = _layer_index(name)
            if layer is None:
                break
            layers.setdefault(layer, []).append(name)
        else:
            physical_layers = sorted(layers)
            is_swa = all(name.endswith(".swa_cache") for name in names)
            is_mla = all(name.endswith(".long_kv_cache") or _is_indexer_member(name) for name in names)
            if not (is_swa or is_mla):
                continue
            if is_mla:
                if set(kv_source_layer_ids) - layers.keys():
                    logger.warning("CED group %d has no members for all KV sources; using bulk.", group_id)
                    continue
                if any(
                    not any(name.endswith(".long_kv_cache") for name in members)
                    or not any(_is_indexer_member(name) for name in members)
                    for members in layers.values()
                ):
                    continue
            offsets = None if group_entry_offsets is None else group_entry_offsets.get(group_id)
            if offsets is None:
                if group_entry_offsets is not None:
                    continue
                # Topology-only callers have not registered tensor views yet.
                offsets = [0]
                for layer in physical_layers:
                    offsets.append(offsets[-1] + len(layers[layer]))
            if (
                len(offsets) != len(physical_layers) + 1
                or offsets[0] != 0
                or any(a >= b for a, b in zip(offsets, offsets[1:]))
            ):
                continue
            if is_swa:
                slices = (CEDGroupSlice(group_id, 0, offsets[-1], physical_layers[-1]),)
            else:
                slices = tuple(
                    CEDGroupSlice(group_id, offsets[i], offsets[i + 1], layer, i)
                    for i, layer in enumerate(physical_layers)
                )
            by_group[group_id] = slices
            for group_slice in slices:
                by_trigger.setdefault(group_slice.trigger_layer, []).append(group_slice)
    return CEDTriggerMap(
        slices_by_group=by_group,
        slices_by_trigger={layer: tuple(slices) for layer, slices in by_trigger.items()},
        publish_layers={group: slices[-1].trigger_layer for group, slices in by_group.items()},
    )


def _kv_source_layer_ids(hf_config) -> tuple[int, ...]:
    source_ids = getattr(hf_config, "kv_source_layer_ids", None)
    return tuple(int(source_id) for source_id in source_ids) if source_ids else ()


def chunk_member_transfer(
    token_database: ChunkedTokenDatabase,
    group_id: int,
    block_id: int,
    start: int,
    end: int,
    member_start: int,
    member_end: int,
) -> list[tuple[int, int, int]]:
    """Select registered descriptors without duplicating bulk address math."""
    addrs, sizes, _ = token_database.prepare_value(start, end, [], kv_cache_group_id=group_id, block_id=block_id)
    entries = token_database.group_kv_caches_base_addr[group_id]
    if not 0 <= member_start <= member_end <= len(entries):
        raise ValueError("CED descriptor range is outside the registered group")
    segments = token_database.group_cache_entry_segments.get(group_id)
    block_size = token_database.get_block_size(group_id)
    local_start = start % block_size
    local_end = end - start // block_size * block_size
    active = []
    for index in range(len(entries)):
        segment_index, count = segments[index] if segments else (0, 1)
        tokens = block_size // count
        if max(local_start, segment_index * tokens) < min(local_end, (segment_index + 1) * tokens):
            active.append(index)
    if len(active) != len(addrs) or len(addrs) != len(sizes):
        raise ValueError("CED descriptor layout disagrees with bulk prepare_value")
    rows = []
    offset = 0
    for index, address, size in zip(active, addrs, sizes, strict=True):
        if member_start <= index < member_end:
            rows.append((address, offset, size))
        offset += size
    return rows


@dataclass
class CEDSaveChunks:
    """Chunk list of one (request, group) session, mirroring the bulk save."""

    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)


@dataclass
class CEDSaveSession:
    """One memcache WRITING session: the (request, group) pool object set."""

    req_meta: ReqMeta
    group_id: int
    chunks: CEDSaveChunks
    object_size: int
    store_job_id: int = 0
    live_keys: list[str] = field(default_factory=list)
    live_chunk_indices: list[int] = field(default_factory=list)
    live_gvas: list[int] = field(default_factory=list)
    completed_slices: set[int] = field(default_factory=set)
    failed: bool = False
    finished: bool = False
    commit_started: bool = False
    deadline: float = 0.0

    @property
    def req_id(self) -> str:
        return self.req_meta.req_id


@dataclass
class CEDSessionOpenTask:
    session: CEDSaveSession


@dataclass
class CEDSessionSliceTask:
    session: CEDSaveSession
    slice_index: int
    event: Any


@dataclass
class CEDLoadUnit:
    """One dependency unit: a registered descriptor range of one group.

    ``gate_layer`` is the first consumer layer of the member range (the KV
    source layer for full-context slot slices, the member's own layer for
    SWA member rows).
    """

    group_id: int
    member_start: int
    member_end: int
    gate_layer: int


@dataclass
class CEDLoadTask:
    """Back-to-back unit list of one request, drained in dependency order."""

    req_meta: ReqMeta
    token_len: int
    units: list[CEDLoadUnit]
    gate_events: dict[int, threading.Event] = field(default_factory=dict)
    remaining_by_gate: dict[int, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)
    prepared: bool = False
    chunks_by_group: dict[int, CEDSaveChunks] = field(default_factory=dict)
    gva_by_key: dict[str, int] = field(default_factory=dict)
    leased_keys: list[str] = field(default_factory=list)
    remaining_units: int = 0
    deadline: float = 0.0

    def settle_gate(self, gate_layer: int) -> None:
        with self.lock:
            remaining = self.remaining_by_gate.get(gate_layer, 0) - 1
            if remaining > 0:
                self.remaining_by_gate[gate_layer] = remaining
                return
            self.remaining_by_gate.pop(gate_layer, None)
        event = self.gate_events.get(gate_layer)
        if event is not None:
            event.set()

    def settle_all_gates(self) -> None:
        with self.lock:
            self.remaining_by_gate.clear()
            events = list(self.gate_events.values())
        for event in events:
            event.set()


@dataclass(frozen=True)
class CEDSessionSealTask:
    event: Any


@dataclass(frozen=True)
class CEDLoadUnitTask:
    task: CEDLoadTask
    unit: CEDLoadUnit


class CEDSessionSendThread(KVCacheStoreSendingThread):
    """Incremental writes for backends that guarantee exclusive allocation.

    Memcache release/1.3 batch_alloc returns shared allocations, so its
    backend deliberately does not enable this plane. Never compensate an
    uncertain commit with WRITE_FAIL: that SDK removes the key unconditionally.
    """

    def __init__(
        self,
        *args: Any,
        trigger_map: CEDTriggerMap | None = None,
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.ced_trigger_map = trigger_map or CEDTriggerMap()
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self._ced_lock = threading.Lock()
        self._ced_sessions: dict[tuple[str, int], CEDSaveSession] = {}
        self._ced_sessions_by_group: dict[int, list[CEDSaveSession]] = {}
        self._ced_claim_counts: dict[str, int] = {}
        # Keep the claim through the following bulk fallback job, even when
        # all session callbacks have already retired. Its offset is request-wide.
        self._ced_claimed_requests: set[str] = set()

    def open_ced_session(
        self, req_meta: ReqMeta, group_id: int, chunks: CEDSaveChunks, object_size: int
    ) -> CEDSaveSession:
        session = CEDSaveSession(req_meta=req_meta, group_id=group_id, chunks=chunks, object_size=object_size)
        with self._request_submit_lock:
            self.raise_if_failed()
            if self._stopping.is_set():
                raise RuntimeError("Cannot open a CED session on a stopped sender")
            with self._ced_lock:
                key = (session.req_id, group_id)
                if key in self._ced_sessions:
                    raise RuntimeError("A CED session is still using this request/group")
                with self.done_task_lock:
                    self._next_local_store_job_id -= 1
                    session.store_job_id = self._next_local_store_job_id
                    self.finished_requests.discard(session.req_id)
                    self.stored_requests.setdefault(session.req_id, set()).add(session.store_job_id)
                self._ced_claim_counts[session.req_id] = self._ced_claim_counts.get(session.req_id, 0) + 1
                self._ced_claimed_requests.add(session.req_id)
                self._ced_sessions[key] = session
                self._ced_sessions_by_group.setdefault(group_id, []).append(session)
            self.request_queue.put(CEDSessionOpenTask(session))
        return session

    def enqueue_slice_tasks(self, trigger_layer: int, event: Any) -> None:
        with self._ced_lock:
            tasks = [
                CEDSessionSliceTask(session, group_slice.slice_index, event)
                for group_slice in self.ced_trigger_map.slices_by_trigger.get(trigger_layer, ())
                for session in self._ced_sessions_by_group.get(group_slice.group_id, ())
                if not session.failed and not session.finished
            ]
        for task in tasks:
            self.add_request(task)

    def seal_ced_step(self, event: Any) -> None:
        # FIFO puts this after all early slices and before the normal batch
        # fence. Missing hooks and the replay cut use this final write event.
        self.add_request(CEDSessionSealTask(event))

    def _handle_request(self, req_meta: Any):
        if not isinstance(req_meta, (CEDSessionOpenTask, CEDSessionSliceTask, CEDSessionSealTask)):
            super()._handle_request(req_meta)
            return
        try:
            if isinstance(req_meta, CEDSessionSealTask):
                with self._ced_lock:
                    sessions = list(self._ced_sessions.values())
                for session in sessions:
                    for index in range(len(self.ced_trigger_map.slices_by_group[session.group_id])):
                        try:
                            self._handle_ced_slice(session, index, req_meta.event)
                        except Exception:
                            logger.exception("CED final slice failed for request %s", session.req_id)
                            self._fail_session(session)
                            break
            elif isinstance(req_meta, CEDSessionOpenTask):
                self._handle_ced_open(req_meta.session)
            else:
                self._handle_ced_slice(req_meta.session, req_meta.slice_index, req_meta.event)
        except Exception:
            logger.exception("CED save failed for request %s", req_meta.session.req_id)
            self._fail_session(req_meta.session)
        finally:
            self.request_queue.task_done()

    def _handle_stored_request(self, req_meta: ReqMeta) -> bool:
        succeeded = super()._handle_stored_request(req_meta)
        with self._ced_lock:
            claimed = req_meta.req_id in self._ced_claimed_requests
        return succeeded and not claimed

    def reset_saved_request(self, req_id: str) -> None:
        super().reset_saved_request(req_id)
        with self._ced_lock:
            self._ced_claimed_requests.discard(req_id)

    def _handle_ced_open(self, session: CEDSaveSession) -> None:
        if getattr(self.m_store, "supports_exclusive_allocation", False) is not True:
            raise RuntimeError("CED save requires exclusive object allocation")
        keys = session.chunks.keys
        if not keys:
            self._complete_session(session, published=False)
            return
        readable = self.m_store.batch_is_readable(keys)
        if readable is None or len(readable) != len(keys) or any(type(value) is not bool for value in readable):
            raise RuntimeError("Malformed CED readability response")
        missing = [i for i, value in enumerate(readable) if not value]
        if not missing:
            self._complete_session(session, published=False)
            return
        alloc_keys = [keys[i] for i in missing]
        session.deadline = time.monotonic() + CED_SESSION_LEASE_TTL_MS / 1000
        gvas = self.m_store.batch_alloc(alloc_keys, [session.object_size] * len(alloc_keys), CED_SESSION_LEASE_TTL_MS)
        if gvas is None or len(gvas) != len(alloc_keys):
            raise RuntimeError("Malformed CED allocation response")
        for key, index, gva in zip(alloc_keys, missing, gvas, strict=True):
            if isinstance(gva, bool) or not isinstance(gva, (int, np.integer)):
                raise RuntimeError("Invalid CED allocation address")
            if 0 < gva < _SSD_HIT_GVA:
                session.live_keys.append(key)
                session.live_chunk_indices.append(index)
                session.live_gvas.append(int(gva))
        if not session.live_keys:
            self._complete_session(session, published=False)

    def _handle_ced_slice(self, session: CEDSaveSession, slice_index: int, event: Any) -> None:
        if session.failed or session.finished:
            return
        slices = self.ced_trigger_map.slices_by_group[session.group_id]
        if not 0 <= slice_index < len(slices):
            raise ValueError("Invalid CED slice index")
        if slice_index in session.completed_slices:
            return
        event.synchronize()
        if time.monotonic() >= session.deadline - CED_LEASE_SAFETY_MARGIN_S:
            raise RuntimeError("CED write lease has insufficient remaining lifetime")
        group_slice = slices[slice_index]
        gvas, addrs, sizes = [], [], []
        for index, gva in zip(session.live_chunk_indices, session.live_gvas, strict=True):
            rows = chunk_member_transfer(
                self.token_database,
                session.group_id,
                session.chunks.block_ids[index],
                session.chunks.starts[index],
                session.chunks.ends[index],
                group_slice.member_start,
                group_slice.member_end,
            )
            if not rows:
                raise RuntimeError("A planned CED slice has no registered descriptors")
            for address, offset, size in rows:
                if offset + size > session.object_size:
                    raise RuntimeError("CED slice exceeds its allocated object")
                gvas.append(gva + offset)
                addrs.append(address)
                sizes.append(size)
        result = self._batch_copy_with_limits(
            np.asarray(gvas, dtype=np.int64),
            np.asarray(addrs, dtype=np.int64),
            np.asarray(sizes, dtype=np.int64),
            _MMC_COPY_L2G,
            self.max_transfer_blocks,
            self.max_transfer_bytes,
        )
        if result != 0:
            raise RuntimeError(f"CED slice copy failed with code {result}")
        session.completed_slices.add(slice_index)
        if len(session.completed_slices) == len(slices):
            # Set before the RPC: even an exception can mean remote success.
            session.commit_started = True
            results = self.m_store.batch_write_finish(session.live_keys, [0] * len(session.live_keys))
            if results is None or len(results) != len(session.live_keys) or any(value != 0 for value in results):
                raise RuntimeError("CED publication outcome is unknown")
            # WRITE_OK consumes the write lease in the SDK. Do not remove a
            # read lease here, which may belong to a concurrent reader.
            self._complete_session(session, published=True)

    def _fail_session(self, session: CEDSaveSession) -> None:
        if session.finished:
            return
        session.failed = True
        if session.live_keys and not session.commit_started:
            try:
                # Only this contract permits deleting an unpublished object.
                if getattr(self.m_store, "supports_exclusive_allocation", False) is True:
                    self.m_store.batch_write_finish(session.live_keys, [-1] * len(session.live_keys))
            except Exception:
                logger.exception("CED unpublished allocation cleanup failed")
        self._complete_session(session, published=False)

    def _complete_session(self, session: CEDSaveSession, published: bool) -> None:
        with self._ced_lock:
            if session.finished:
                return
            session.finished = True
            key = (session.req_id, session.group_id)
            if self._ced_sessions.get(key) is session:
                self._ced_sessions.pop(key)
            sessions = self._ced_sessions_by_group.get(session.group_id, [])
            if session in sessions:
                sessions.remove(session)
            if not sessions:
                self._ced_sessions_by_group.pop(session.group_id, None)
            count = self._ced_claim_counts.get(session.req_id, 0) - 1
            if count > 0:
                self._ced_claim_counts[session.req_id] = count
            else:
                self._ced_claim_counts.pop(session.req_id, None)
            with self.done_task_lock:
                jobs = self.stored_requests.get(session.req_id, set())
                jobs.discard(session.store_job_id)
                if not jobs:
                    self.stored_requests.pop(session.req_id, None)
                    self.finished_requests.add(session.req_id)


class CEDPipelinedRecvingThread(KVCacheStoreRecvingThread):
    """Copy leased bulk descriptor ranges before their first consumer.

    A task owns its error and completion events until the whole step drains.
    Enqueuing work never reports finished receiving to the scheduler.
    """

    def __init__(self, *args: Any, max_transfer_blocks: int = 0, max_transfer_bytes: int = 0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes

    def _handle_request(self, request_data: Any):
        if not isinstance(request_data, (CEDLoadTask, CEDLoadUnitTask)):
            super()._handle_request(request_data)
            return
        task = request_data.task if isinstance(request_data, CEDLoadUnitTask) else request_data
        units = [request_data.unit] if isinstance(request_data, CEDLoadUnitTask) else task.units
        if not isinstance(request_data, CEDLoadUnitTask):
            task.remaining_units = len(units)
        try:
            for unit in units:
                try:
                    if task.error is None:
                        if not task.prepared:
                            self._prepare_ced_load(task)
                        self._copy_ced_load_unit(task, unit)
                except Exception as exc:
                    # Publish the error before opening any gate. Do not rely on
                    # failed_requests, which get_transfer_results can consume.
                    task.error = exc
                    logger.exception("CED load failed for request %s", task.req_meta.req_id)
                    task.settle_all_gates()
                finally:
                    task.remaining_units -= 1
                    if task.remaining_units == 0:
                        self._finish_ced_load(task)
        finally:
            self.request_queue.task_done()

    def _prepare_ced_load(self, task: CEDLoadTask) -> None:
        request = task.req_meta
        masks = self.token_database.load_mask(request.block_hashes, task.token_len)
        for group_id in sorted({unit.group_id for unit in task.units}):
            task.chunks_by_group[group_id] = self._list_load_chunks(request, task.token_len, group_id, masks)
        expected_sizes: dict[str, int] = {}
        for group_id, chunks in task.chunks_by_group.items():
            for start, end, key, block_id in zip(
                chunks.starts, chunks.ends, chunks.keys, chunks.block_ids, strict=True
            ):
                _, sizes, _ = self.token_database.prepare_value(
                    start, end, [], kv_cache_group_id=group_id, block_id=block_id
                )
                size = sum(sizes)
                if key in expected_sizes and expected_sizes[key] != size:
                    raise RuntimeError("One CED key has inconsistent object sizes")
                expected_sizes[key] = size
        keys = list(expected_sizes)
        if keys:
            # for_load rewarms SSD-only blobs using the existing backend path.
            infos = self.m_store.batch_get_key_info(keys, for_load=True)
            if infos is None or len(infos) != len(keys):
                raise RuntimeError("Malformed CED load metadata")
            results = self.m_store.batch_add_lease(keys, CED_SESSION_LEASE_TTL_MS)
            if results is not None:
                task.leased_keys.extend(key for key, result in zip(keys, results) if result == 0)
            if results is None or len(results) != len(keys) or any(result != 0 for result in results):
                raise RuntimeError("Could not lease all CED load objects")
            task.deadline = time.monotonic() + CED_SESSION_LEASE_TTL_MS / 1000
            # Resolve again while the allocation is pinned. The pre-lease GVA
            # can have been evicted or changed by SSD rewarm.
            infos = self.m_store.batch_get_key_info(keys)
            if infos is None or len(infos) != len(keys):
                raise RuntimeError("Malformed leased CED metadata")
            for key, info in zip(keys, infos, strict=True):
                if info is None or info.size() != expected_sizes[key]:
                    raise RuntimeError("CED object size does not match the registered bulk layout")
                gva = next((int(gva) for gva in info.gva_list() if 0 < gva < _SSD_HIT_GVA), 0)
                if not gva:
                    raise RuntimeError("CED object has no leased DRAM address")
                task.gva_by_key[key] = gva
        task.prepared = True

    def _copy_ced_load_unit(self, task: CEDLoadTask, unit: CEDLoadUnit) -> None:
        chunks = task.chunks_by_group[unit.group_id]
        src_gvas, dst_addrs, sizes = [], [], []
        for index, key in enumerate(chunks.keys):
            rows = chunk_member_transfer(
                self.token_database,
                unit.group_id,
                chunks.block_ids[index],
                chunks.starts[index],
                chunks.ends[index],
                unit.member_start,
                unit.member_end,
            )
            if not rows:
                raise RuntimeError("A CED load dependency has no descriptors")
            for address, offset, size in rows:
                src_gvas.append(task.gva_by_key[key] + offset)
                dst_addrs.append(address)
                sizes.append(size)
        if src_gvas:
            if time.monotonic() >= task.deadline - CED_LEASE_SAFETY_MARGIN_S:
                raise RuntimeError("CED read lease has insufficient remaining lifetime")
            # SDK 1.3 BatchDataOperation waits for all transfer futures; this
            # gate is opened after copy completion, not merely submission.
            result = self._batch_copy_with_limits(
                np.asarray(src_gvas, dtype=np.int64),
                np.asarray(dst_addrs, dtype=np.int64),
                np.asarray(sizes, dtype=np.int64),
                _MMC_COPY_G2L,
                self.max_transfer_blocks,
                self.max_transfer_bytes,
            )
            if result != 0:
                raise RuntimeError(f"CED load copy failed with code {result}")
        task.settle_gate(unit.gate_layer)

    def _finish_ced_load(self, task: CEDLoadTask) -> None:
        try:
            if task.leased_keys and self.m_store.batch_remove_lease(list(task.leased_keys)) != 0:
                raise RuntimeError("Could not release CED read leases")
        except Exception as exc:
            task.error = task.error or exc
        finally:
            task.leased_keys.clear()
            if task.error is not None:
                with self.done_task_lock:
                    self.failed_requests.add(task.req_meta.req_id)
                # Conservative diagnostics; the consumer raises a fatal step
                # error rather than claiming that hybrid state can be retried.
                with self._invalid_block_ids_lock:
                    self._invalid_block_ids.update(
                        block for chunks in task.chunks_by_group.values() for block in chunks.block_ids if block > 0
                    )
            task.settle_all_gates()
            task.done.set()

    def _list_load_chunks(self, req_meta: ReqMeta, token_len: int, group_id: int, load_masks: Any) -> CEDSaveChunks:
        chunks = CEDSaveChunks()
        if group_id >= len(req_meta.block_ids_by_group):
            raise ValueError("Missing block table for a CED load group")
        block_ids = req_meta.block_ids_by_group[group_id]
        block_size = self._get_block_size(group_id)
        mask_num = req_meta.load_spec.vllm_cached_tokens // block_size * block_size
        iterator = self.token_database.process_token_key_strings_with_block_ids(
            token_len,
            req_meta.block_hashes,
            block_ids,
            mask_num,
            kv_cache_group_id=group_id,
            skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
            chunk_filter=lambda start: self.token_database.mask_allows_chunk(load_masks, group_id, start),
        )
        for start, end, key, _block_hash, block_id in iterator:
            chunks.starts.append(start)
            chunks.ends.append(end)
            chunks.keys.append(key)
            chunks.block_ids.append(block_id)
        return chunks


class CEDSaveWorker(KVPoolWorker):
    """Bulk-compatible CED worker with step-local producer/consumer plans.

    Only eager, non-speculative TP deployments are enabled initially. Other
    parallel layouts and async scheduler loads retain their existing bulk
    behavior. SDK 1.3 supports pipelined reads, but not exclusive session saves.
    """

    def __init__(self, vllm_config: VllmConfig, use_layerwise: bool, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config, use_layerwise, kv_cache_config)
        self.ced_trigger_map = CEDTriggerMap()
        self.ced_trigger_layers: tuple[int, ...] = ()
        self._ced_fallback_groups: dict[str, list[int]] = {}
        self._ced_fired_triggers: set[int] = set()
        self._ced_load_gates: dict[str, dict[int, threading.Event]] = {}
        self._ced_load_tasks: list[CEDLoadTask] = []
        self._ced_step_active = False
        self._ced_load_started = False
        compilation = vllm_config.compilation_config
        mode = getattr(compilation.cudagraph_mode, "name", str(compilation.cudagraph_mode))
        eager = bool(vllm_config.model_config.enforce_eager) or mode == "NONE"
        parallel = vllm_config.parallel_config
        self._ced_runtime_supported = (
            self._extra_config.get("backend", "mooncake") == "memcache"
            and eager
            and not self.load_async
            and self.pp_size == self.pcp_size == self.dcp_size == 1
            and not self.use_kvpp
            and not getattr(parallel, "use_ubatching", False)
            and not getattr(parallel, "use_sequence_parallel_moe", False)
            and vllm_config.speculative_config is None
            and self._extra_config.get("discard_partial_chunks", True) is True
        )
        self._ced_save_capable = False
        self._ced_load_capable = False
        if not self._ced_runtime_supported:
            logger.warning(
                "CED_layerwise requires eager memcache, load_async=false, full blocks and no "
                "PP/PCP/DCP/KVPP/microbatching/speculation; retaining bulk transfers."
            )

    def _cacheable_group_layer_names(self) -> dict[int, list[str]]:
        if self.kv_cache_config is None:
            return {}
        return {
            group_id: list(group.layer_names)
            for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups)
            if group_id in self.cacheable_group_ids
        }

    def _start_kv_transfer_threads(self) -> None:
        if self._transfer_threads_started:
            return
        self.ced_trigger_map = build_ced_trigger_map(
            self._cacheable_group_layer_names(),
            _kv_source_layer_ids(self.hf_config),
            self.group_layer_cache_entry_offsets,
        )
        self.ced_trigger_layers = self.ced_trigger_map.trigger_layers()
        mapped = self._ced_runtime_supported and bool(self.ced_trigger_layers)
        self._ced_save_capable = (
            mapped
            and not self.enable_kv_events
            and getattr(self.m_store, "supports_exclusive_allocation", False) is True
        )
        self._ced_load_capable = mapped
        if mapped and self.can_put and not self._ced_save_capable:
            logger.warning(
                "CED incremental saves require exclusive allocation. Memcache release/1.3 "
                "batch_alloc returns shared GVAs; preserving atomic bulk saves."
            )
        if self._ced_save_capable and self.can_put:
            ready = threading.Event()
            self.kv_send_thread = CEDSessionSendThread(
                self.m_store,
                self.token_database,
                self.grouped_block_size,
                self.tp_rank,
                self.tp_size,
                self.pcp_rank,
                self.pcp_size,
                self.dcp_size,
                self.put_step,
                self.kv_role,
                ready,
                self.group_uses_align_state,
                self.enable_kv_events,
                trigger_map=self.ced_trigger_map,
                max_transfer_blocks=int(self._extra_config.get("ced_max_transfer_blocks", 0)),
                max_transfer_bytes=int(self._extra_config.get("ced_max_transfer_bytes", 0)),
            )
            self.kv_send_thread.start()
            ready.wait()
            self._transfer_threads_started = True
        else:
            super()._start_kv_transfer_threads()
        if self._ced_load_capable:
            ready = threading.Event()
            self.kv_recv_thread = CEDPipelinedRecvingThread(
                self.m_store,
                self.token_database,
                self.grouped_block_size,
                self.tp_rank,
                self.tp_size,
                self.dcp_size,
                ready,
                invalid_block_ids=self._invalid_block_ids,
                invalid_block_ids_lock=self._invalid_block_ids_lock,
                record_operation=self._record_kv_connector_operation,
                max_transfer_blocks=int(self._extra_config.get("ced_max_transfer_blocks", 0)),
                max_transfer_bytes=int(self._extra_config.get("ced_max_transfer_bytes", 0)),
            )
            self.kv_recv_thread.start()
            ready.wait()

    def prepare_ced_step(self, metadata: Any) -> None:
        # bind happens before forward even for save-only steps. First drain
        # old target writes before any block table or gate can be replaced.
        self.finish_ced_loads()
        self._ced_load_tasks.clear()
        self._ced_load_gates.clear()
        self._ced_fired_triggers.clear()
        self._ced_fallback_groups.clear()
        self._ced_step_active = True
        self._ced_load_started = False
        if self._ced_save_capable and self.can_put:
            sender = self.kv_send_thread
            if isinstance(sender, CEDSessionSendThread):
                self._prepare_ced_sessions(sender, metadata)

    def start_load_kv(self, metadata: Any):
        if not self._ced_load_capable:
            return super().start_load_kv(metadata)
        if self._ced_load_started:
            return
        self._ced_load_started = True
        self.ensure_store_initialized()
        native = self.m_store.store
        required = ("batch_copy", "batch_get_key_info", "batch_add_lease", "batch_remove_lease")
        if not all(callable(getattr(native, name, None)) for name in required):
            return super().start_load_kv(metadata)
        handled = self._prepare_ced_loads(metadata)
        bulk_metadata = copy(metadata)
        bulk_metadata.requests = [request for request in metadata.requests if request.req_id not in handled]
        super().start_load_kv(bulk_metadata)

    def _prepare_ced_loads(self, metadata: Any) -> set[str]:
        receiver = self.kv_recv_thread
        if not isinstance(receiver, CEDPipelinedRecvingThread):
            return set()
        handled = set()
        submissions: list[CEDLoadUnitTask] = []
        for request in metadata.requests:
            spec = request.load_spec
            if spec is None or not spec.can_load:
                continue
            groups = set(request.kv_cache_group_ids or [0]) & set(self.cacheable_group_ids)
            if not groups or any(group not in self.ced_trigger_map.slices_by_group for group in groups):
                continue
            if spec.kvpool_cached_tokens % self.cache_transfer_granularity:
                continue
            units = self._build_ced_load_units(groups)
            if not units:
                continue
            snapshot = self._copy_req_meta_for_groups(request, sorted(groups))
            snapshot.skip_null_blocks_by_group = list(self.group_uses_align_state)
            gates = {unit.gate_layer: threading.Event() for unit in units}
            counts = {layer: sum(unit.gate_layer == layer for unit in units) for layer in gates}
            task = CEDLoadTask(
                snapshot,
                spec.kvpool_cached_tokens,
                units,
                gates,
                counts,
                remaining_units=len(units),
            )
            self._ced_load_tasks.append(task)
            self._ced_load_gates[request.req_id] = gates
            submissions.extend(CEDLoadUnitTask(task, unit) for unit in units)
            handled.add(request.req_id)
        # Interleave requests by first consumer, rather than finishing every
        # source of a long request before another request can start L0/L2.
        submissions.sort(key=lambda item: item.unit.gate_layer)
        for submission in submissions:
            receiver.add_request(submission)
        return handled

    def _build_ced_load_units(self, group_ids: set[int]) -> list[CEDLoadUnit]:
        units = []
        names_by_group = self._cacheable_group_layer_names()
        for group_id in sorted(group_ids):
            slices = self.ced_trigger_map.slices_by_group[group_id]
            names = names_by_group[group_id]
            if all(name.endswith(".swa_cache") for name in names):
                layers = sorted({_layer_index(name) for name in names})
                offsets = self.group_layer_cache_entry_offsets[group_id]
                units.extend(CEDLoadUnit(group_id, offsets[i], offsets[i + 1], layer) for i, layer in enumerate(layers))
            else:
                units.extend(
                    CEDLoadUnit(group_id, part.member_start, part.member_end, part.trigger_layer) for part in slices
                )
        units.sort(key=lambda unit: unit.gate_layer)
        return units

    def _wait_ced_event(self, event: threading.Event) -> None:
        receiver = self.kv_recv_thread
        if receiver is None:
            raise RuntimeError("CED receiver is not initialized")
        while not event.wait(timeout=CED_GATE_POLL_INTERVAL_S):
            receiver.raise_if_failed()
            if receiver._stopping.is_set() or not receiver.is_alive():
                raise RuntimeError("CED receiver stopped before completing a load")
        receiver.raise_if_failed()

    def wait_for_ced_layer_load(self, layer_idx: int) -> None:
        for task in self._ced_load_tasks:
            event = task.gate_events.get(layer_idx)
            if event is not None:
                self._wait_ced_event(event)
                if task.error is not None:
                    # No target writes may survive the failing forward.
                    self.finish_ced_loads()

    def finish_ced_loads(self) -> None:
        for task in self._ced_load_tasks:
            self._wait_ced_event(task.done)
        errors = [task.error for task in self._ced_load_tasks if task.error is not None]
        if errors:
            raise RuntimeError("CED pipelined load failed; this step's state cannot be reused") from errors[0]

    def _prepare_ced_sessions(self, send_thread: CEDSessionSendThread, metadata: Any) -> None:
        for request in metadata.requests:
            if not request.can_save:
                continue
            groups = list(request.kv_cache_group_ids or [0])
            self._ced_fallback_groups[request.req_id] = groups
            # Save only committed prefill pages. Speculation is gated above;
            # generated/decode extents and partial objects stay on bulk.
            if (
                request.num_prompt_tokens is None
                or request.target_token_len > request.num_prompt_tokens
                or request.token_len_chunk % self.cache_transfer_granularity
            ):
                continue
            snapshot = self._copy_req_meta_for_groups(request, groups)
            masks = self.token_database.store_mask(snapshot.token_len_chunk, snapshot.num_prompt_tokens)
            claimed = []
            for group_id in groups:
                if group_id not in self.ced_trigger_map.slices_by_group:
                    continue
                chunks = self._list_ced_save_chunks(send_thread, snapshot, group_id, masks)
                if not chunks.keys:
                    continue
                size = sum(self.token_database.group_block_len[group_id])
                send_thread.open_ced_session(snapshot, group_id, chunks, size)
                claimed.append(group_id)
            self._ced_fallback_groups[request.req_id] = [group for group in groups if group not in claimed]

    def _list_ced_save_chunks(
        self,
        send_thread: CEDSessionSendThread,
        request: ReqMeta,
        group_id: int,
        store_masks: Any,
    ) -> CEDSaveChunks:
        """Mirror the bulk save enumeration for one (request, group)."""
        chunks = CEDSaveChunks()
        token_len = request.token_len_chunk
        group_block_size = self.grouped_block_size[group_id]
        group_store_mask = store_masks[group_id] if store_masks is not None and group_id < len(store_masks) else None
        if group_store_mask is not None and not any(group_store_mask):
            return chunks
        if group_id >= len(request.block_ids_by_group):
            return chunks
        block_ids = request.block_ids_by_group[group_id]
        skip_null_blocks = group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]
        align_state_group = skip_null_blocks
        load_spec = request.load_spec
        skip_start = load_spec.vllm_cached_tokens if load_spec is not None else 0
        skip_end = (
            (
                load_spec.kvpool_store_skip_tokens
                if load_spec.kvpool_store_skip_tokens is not None
                else load_spec.kvpool_cached_tokens
            )
            if load_spec is not None
            else 0
        )
        save_start_token = send_thread.get_saved_offset(request.req_id)
        save_start = 0 if align_state_group else save_start_token

        def should_skip(start: int) -> bool:
            if start + group_block_size <= save_start:
                return True
            return skip_end > skip_start and start >= skip_start and start + group_block_size <= skip_end

        def chunk_filter(start: int) -> bool:
            return not should_skip(start)

        tp_replicas = self.put_step if self.dcp_size <= 1 and not align_state_group else 1
        iterator = self.token_database.process_token_key_strings_with_block_ids(
            token_len,
            request.block_hashes,
            block_ids,
            kv_cache_group_id=group_id,
            skip_null_blocks=skip_null_blocks,
            chunk_filter=chunk_filter,
            shard_rank=self.pcp_rank * tp_replicas + self.tp_rank % tp_replicas,
            shard_size=self.pcp_size * tp_replicas,
            chunk_mask=group_store_mask,
        )
        for start, end, key, _block_hash, block_id in iterator:
            chunks.starts.append(start)
            chunks.ends.append(end)
            chunks.keys.append(key)
            chunks.block_ids.append(block_id)
        return chunks

    def wait_for_save(self, connector_metadata: Any) -> None:
        sender = self.kv_send_thread
        if not isinstance(sender, CEDSessionSendThread):
            super().wait_for_save(connector_metadata)
            return
        event = torch.npu.Event()
        event.record()
        sender.seal_ced_step(event)
        requests = []
        for request in connector_metadata.requests:
            if not request.can_save:
                continue
            groups = self._ced_fallback_groups.get(request.req_id, list(request.kv_cache_group_ids or [0]))
            if groups:
                fallback = self._copy_req_meta_for_groups(request, groups)
                fallback.skip_null_blocks_by_group = list(self.group_uses_align_state)
                fallback.current_event = event
                requests.append(fallback)
        # Includes early tasks, cut/missing-hook slices and fallback bulk puts.
        self._previous_save_batch = sender.add_save_batch(requests)
        self._ced_step_active = False

    @staticmethod
    def _copy_req_meta_for_groups(request: ReqMeta, group_ids: list[int]) -> ReqMeta:
        snapshot = copy(request)
        snapshot.kv_cache_group_ids = list(group_ids)
        snapshot.block_ids_by_group = [list(blocks) for blocks in request.block_ids_by_group]
        snapshot.block_hashes = list(request.block_hashes)
        snapshot.load_spec = copy(request.load_spec) if request.load_spec is not None else None
        snapshot.token_ids = list(request.token_ids) if request.token_ids is not None else None
        return snapshot

    def on_ced_save_trigger(self, trigger_layer: int) -> None:
        if not self._ced_step_active or not self._ced_save_capable:
            return
        if trigger_layer in self._ced_fired_triggers or trigger_layer not in self.ced_trigger_map.slices_by_trigger:
            return
        sources = _kv_source_layer_ids(self.hf_config)
        if sources and getattr(self.vllm_config.cache_config, "swa_bounded_replay", False):
            if trigger_layer == max(sources):
                # Cache-only cut writes can be repeated during replay. The
                # final step event is the conservative last-write boundary.
                return
        sender = self.kv_send_thread
        if not isinstance(sender, CEDSessionSendThread):
            return
        sender.raise_if_failed()
        self._ced_fired_triggers.add(trigger_layer)
        event = torch.npu.Event()
        event.record()
        sender.enqueue_slice_tasks(trigger_layer, event)

    def shutdown(self) -> None:
        try:
            self.finish_ced_loads()
        finally:
            super().shutdown()

    def reset_store(self) -> None:
        self.finish_ced_loads()
        super().reset_store()
