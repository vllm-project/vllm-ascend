from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
_SSD_HIT_GVA = 1 << 60

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
    """One incrementally publishable byte range of a pool group object.

    ``member_start``/``member_end`` are indices into the cache group's
    layer-name list and must match the entry order used by
    ``ChunkedTokenDatabase.prepare_value`` so that the concatenation of all
    slices of a group equals the atomic-put object layout byte for byte.
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
) -> CEDTriggerMap:
    """Derive per-slice trigger points from cacheable group layer names.

    Full-context (MLA) groups get one slice per KV source slot, finalized by
    the source layer's own forward; the members of one slot are contiguous
    (long_kv, indexer[, folded]) in the group entry order. SWA groups get a
    single slice finalized by the group's last member layer. Draft groups
    and unrecognized layouts are left to the step-end bulk save.
    """
    slices_by_group: dict[int, list[CEDGroupSlice]] = {}
    slices_by_trigger: dict[int, list[CEDGroupSlice]] = {}
    publish_layers: dict[int, int] = {}

    for group_id, names in sorted(group_layer_names.items()):
        name_list = list(names)
        if not name_list or any(_is_draft_member(name) for name in name_list):
            continue
        member_layers = [_layer_index(name) for name in name_list]
        if any(layer is None for layer in member_layers):
            continue
        is_long_kv = [name.endswith(".long_kv_cache") for name in name_list]
        is_swa = [name.endswith(".swa_cache") for name in name_list]

        group_slices: list[CEDGroupSlice] = []
        if any(is_long_kv):
            by_source: dict[int, tuple[int, int]] = {}
            for member, layer in enumerate(member_layers):
                if not (is_long_kv[member] or _is_indexer_member(name_list[member])):
                    continue
                start, end = by_source.get(layer, (member, member + 1))
                by_source[layer] = (min(start, member), max(end, member + 1))
            for slice_index, source_layer in enumerate(sorted(by_source)):
                start, end = by_source[source_layer]
                group_slices.append(CEDGroupSlice(group_id, start, end, source_layer, slice_index))
            missing_sources = set(kv_source_layer_ids) - set(by_source)
            if missing_sources:
                logger.warning(
                    "CED trigger map: KV source layers %s have no members in group %d "
                    "(PP stage split or partial layout); those slots keep the bulk save.",
                    sorted(missing_sources),
                    group_id,
                )
        elif all(is_swa):
            group_slices.append(CEDGroupSlice(group_id, 0, len(name_list), max(member_layers), 0))

        if not group_slices:
            continue
        for group_slice in group_slices:
            slices_by_trigger.setdefault(group_slice.trigger_layer, []).append(group_slice)
        slices_by_group[group_id] = group_slices
        publish_layers[group_id] = max(group_slice.trigger_layer for group_slice in group_slices)

    return CEDTriggerMap(
        slices_by_group={group_id: tuple(slices) for group_id, slices in slices_by_group.items()},
        slices_by_trigger={trigger: tuple(slices) for trigger, slices in slices_by_trigger.items()},
        publish_layers=publish_layers,
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
    """(src_addr, dst_offset, size) rows of one chunk restricted to a slice.

    Mirrors ``ChunkedTokenDatabase.prepare_value``'s per-member math and adds
    the destination offset each member occupies in the atomic-put object
    layout, so session writes and atomic puts produce identical bytes. The
    offset accumulates over ALL of the chunk's members (including ones
    outside the slice) exactly as the atomic put concatenates them.
    """
    group_addrs = token_database.group_kv_caches_base_addr[group_id]
    block_lens = token_database.group_block_len[group_id]
    strides = token_database.group_block_stride.get(group_id)
    segments = token_database.group_cache_entry_segments.get(group_id)
    group_block_size = token_database.get_block_size(group_id)
    block_start = start // group_block_size * group_block_size
    start_in_block = start - block_start
    end_in_block = end - block_start

    rows: list[tuple[int, int, int]] = []
    dst_offset = 0
    for member in range(len(group_addrs)):
        segment_index, segment_count = segments[member] if segments else (0, 1)
        segment_tokens = group_block_size // segment_count
        segment_start = segment_index * segment_tokens
        copy_start = max(start_in_block, segment_start)
        copy_end = min(end_in_block, segment_start + segment_tokens)
        size = 0
        if copy_start < copy_end:
            block_len = block_lens[member % len(block_lens)]
            block_stride = strides[member % len(strides)] if strides else block_len
            byte_start = block_len * (copy_start - segment_start) // segment_tokens
            byte_end = block_len * (copy_end - segment_start) // segment_tokens
            size = byte_end - byte_start
            if member_start <= member < member_end:
                rows.append((group_addrs[member] + block_id * block_stride + byte_start, dst_offset, size))
        dst_offset += size
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
    """One dependency unit of a pipelined load: a member range of one group.

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


class CEDSessionSendThread(KVCacheStoreSendingThread):
    """Sending thread with a memcache WRITING-session data plane.

    Bulk ReqMeta requests and fence batches flow through the inherited
    handlers unchanged. CED sessions run open -> per-slice copy -> publish:
    keys are readability-prefiltered, allocated once per group object, filled
    slice by slice behind per-trigger npu events, and atomically published
    with ``batch_write_finish`` after the last slice. Failed sessions are
    revoked and retried by the next step's plan.

    The request-wide saved-offset is intentionally never advanced for
    CED-claimed requests: the plan's readability check is the dedup, so a
    revoked session's chunks are re-listed and retried on the next step.
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
        self.ced_trigger_map = trigger_map if trigger_map is not None else CEDTriggerMap()
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self._ced_lock = threading.Lock()
        self._ced_sessions: dict[tuple[str, int], CEDSaveSession] = {}
        self._ced_sessions_by_group: dict[int, list[CEDSaveSession]] = {}
        self._ced_claim_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Worker-facing session API
    # ------------------------------------------------------------------
    def open_ced_session(
        self, req_meta: ReqMeta, group_id: int, chunks: CEDSaveChunks, object_size: int
    ) -> CEDSaveSession:
        session = CEDSaveSession(req_meta=req_meta, group_id=group_id, chunks=chunks, object_size=object_size)
        with self._ced_lock:
            existing = self._ced_sessions.get((session.req_id, group_id))
            if existing is not None:
                self._revoke_session_locked(existing)
                self._finish_session_locked(existing)
            with self.done_task_lock:
                self._next_local_store_job_id -= 1
                session.store_job_id = self._next_local_store_job_id
                self.finished_requests.discard(session.req_id)
                self.stored_requests.setdefault(session.req_id, set()).add(session.store_job_id)
                self._ced_claim_counts[session.req_id] = self._ced_claim_counts.get(session.req_id, 0) + 1
            self._ced_sessions[(session.req_id, group_id)] = session
            self._ced_sessions_by_group.setdefault(group_id, []).append(session)
        self.add_request(CEDSessionOpenTask(session))
        return session

    def enqueue_slice_tasks(self, trigger_layer: int, event: Any) -> None:
        """Queue one slice copy for every live session of the trigger's groups."""
        slices = self.ced_trigger_map.slices_by_trigger.get(trigger_layer)
        if not slices:
            return
        with self._ced_lock:
            for group_slice in slices:
                for session in list(self._ced_sessions_by_group.get(group_slice.group_id, ())):
                    if session.failed:
                        continue
                    self.add_request(CEDSessionSliceTask(session, group_slice.slice_index, event))

    # ------------------------------------------------------------------
    # Thread handlers
    # ------------------------------------------------------------------
    def _handle_request(self, req_meta: Any):
        if isinstance(req_meta, CEDSessionOpenTask):
            try:
                self._handle_ced_open(req_meta.session)
            except Exception:
                logger.exception(
                    "CED session open failed for request %s group %d",
                    req_meta.session.req_id,
                    req_meta.session.group_id,
                )
                self._fail_session(req_meta.session)
            finally:
                self.request_queue.task_done()
            return
        if isinstance(req_meta, CEDSessionSliceTask):
            try:
                self._handle_ced_slice(req_meta.session, req_meta.slice_index, req_meta.event)
            except Exception:
                logger.exception(
                    "CED session slice failed for request %s group %d slice %d",
                    req_meta.session.req_id,
                    req_meta.session.group_id,
                    req_meta.slice_index,
                )
                self._fail_session(req_meta.session)
            finally:
                self.request_queue.task_done()
            return
        super()._handle_request(req_meta)

    def _handle_stored_request(self, req_meta: ReqMeta) -> bool:
        succeeded = super()._handle_stored_request(req_meta)
        if succeeded:
            with self._ced_lock:
                claimed = self._ced_claim_counts.get(req_meta.req_id, 0) > 0
            if claimed:
                # The group-filtered fallback job must not advance the
                # request-wide high-water for CED-claimed requests; the plan
                # readability check is the dedup and revoked chunks retry.
                return False
        return succeeded

    # ------------------------------------------------------------------
    # Session flow
    # ------------------------------------------------------------------
    def _handle_ced_open(self, session: CEDSaveSession) -> None:
        keys = session.chunks.keys
        if not keys:
            self._complete_session(session, published=False)
            return
        readable_fn = getattr(self.m_store, "batch_is_readable", None)
        if callable(readable_fn):
            readable = readable_fn(keys)
        else:
            readable = [bool(value) for value in self.m_store.exists(keys)]
        if len(readable) != len(keys):
            raise RuntimeError(f"CED readability probe returned {len(readable)} states for {len(keys)} keys")
        missing = [index for index, state in enumerate(readable) if not state]
        if not missing:
            self._complete_session(session, published=False)
            return
        alloc_keys = [keys[index] for index in missing]
        sizes = [session.object_size] * len(alloc_keys)
        gvas = self.m_store.batch_alloc(alloc_keys, sizes, CED_SESSION_LEASE_TTL_MS)
        if gvas is None or len(gvas) != len(alloc_keys):
            raise RuntimeError(
                f"CED allocation returned {0 if gvas is None else len(gvas)} GVAs for {len(alloc_keys)} keys"
            )
        kept_keys: list[str] = []
        kept_indices: list[int] = []
        kept_gvas: list[int] = []
        for key, chunk_index, gva in zip(alloc_keys, missing, gvas, strict=True):
            if not isinstance(gva, (int, np.integer)) or gva <= 0 or gva >= _SSD_HIT_GVA:
                # Another writer owns this object (committed or WRITING).
                continue
            kept_keys.append(key)
            kept_indices.append(chunk_index)
            kept_gvas.append(int(gva))
        session.live_keys = kept_keys
        session.live_chunk_indices = kept_indices
        session.live_gvas = kept_gvas
        if not kept_keys:
            self._complete_session(session, published=False)

    def _handle_ced_slice(self, session: CEDSaveSession, slice_index: int, event: Any) -> None:
        if session.failed:
            return
        slices = self.ced_trigger_map.slices_by_group.get(session.group_id, ())
        if slice_index < 0 or slice_index >= len(slices):
            return
        if slice_index in session.completed_slices:
            return
        if not session.live_keys:
            self._complete_session(session, published=False)
            return
        group_slice = slices[slice_index]
        event.synchronize()
        gvas: list[int] = []
        addrs: list[int] = []
        sizes: list[int] = []
        for chunk_index, gva in zip(session.live_chunk_indices, session.live_gvas, strict=True):
            rows = chunk_member_transfer(
                self.token_database,
                session.group_id,
                session.chunks.block_ids[chunk_index],
                session.chunks.starts[chunk_index],
                session.chunks.ends[chunk_index],
                group_slice.member_start,
                group_slice.member_end,
            )
            if not rows:
                continue
            for src_addr, dst_offset, size in rows:
                gvas.append(gva + dst_offset)
                addrs.append(src_addr)
                sizes.append(size)
        if gvas:
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
        if len(session.completed_slices) >= len(slices):
            results = self.m_store.batch_write_finish(session.live_keys, [0] * len(session.live_keys))
            if results is None or len(results) != len(session.live_keys) or any(int(value) != 0 for value in results):
                raise RuntimeError("CED session publication failed")
            remove_lease = getattr(self.m_store, "batch_remove_lease", None)
            if callable(remove_lease):
                remove_lease(session.live_keys)
            self._complete_session(session, published=True)

    def _fail_session(self, session: CEDSaveSession) -> None:
        if session.failed:
            return
        session.failed = True
        with self._ced_lock:
            self._revoke_session_locked(session)
            self._finish_session_locked(session)

    def _complete_session(self, session: CEDSaveSession, published: bool) -> None:
        with self._ced_lock:
            self._finish_session_locked(session)

    def _revoke_session_locked(self, session: CEDSaveSession) -> None:
        if not session.live_keys:
            return
        try:
            self.m_store.batch_write_finish(session.live_keys, [-1] * len(session.live_keys))
        except Exception:
            logger.exception("CED session revoke failed for request %s group %d", session.req_id, session.group_id)

    def _finish_session_locked(self, session: CEDSaveSession) -> None:
        self._ced_sessions.pop((session.req_id, session.group_id), None)
        sessions = self._ced_sessions_by_group.get(session.group_id)
        if sessions and session in sessions:
            sessions.remove(session)
            if not sessions:
                self._ced_sessions_by_group.pop(session.group_id, None)
        with self.done_task_lock:
            jobs = self.stored_requests.get(session.req_id)
            if jobs is not None and session.store_job_id in jobs:
                jobs.remove(session.store_job_id)
                remaining = len(jobs)
                if remaining == 0:
                    del self.stored_requests[session.req_id]
            else:
                remaining = len(jobs) if jobs else 0
            count = self._ced_claim_counts.get(session.req_id, 0) - 1
            if count > 0:
                self._ced_claim_counts[session.req_id] = count
            else:
                self._ced_claim_counts.pop(session.req_id, None)
        if remaining == 0:
            self.set_finished_request(session.req_id)


class CEDPipelinedRecvingThread(KVCacheStoreRecvingThread):
    """Recving thread draining CED dependency units back-to-back.

    Bulk load requests flow through the inherited handler unchanged. A CED
    task resolves every key's GVA once, then copies each unit's member range
    from the pool object straight into the local cache (GVA + object offset
    addressing, direction G2L), settling the unit's gate event on arrival.
    Failures mark the request failed, release every remaining gate, and the
    in-model gates raise so the step aborts instead of reading half-loaded
    KV.
    """

    def __init__(
        self,
        *args: Any,
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes

    def _handle_request(self, request_data: Any):
        if isinstance(request_data, CEDLoadTask):
            try:
                self._handle_ced_load(request_data)
            except Exception:
                logger.exception("CED pipelined load failed for request %s", request_data.req_meta.req_id)
                self._fail_ced_load(request_data)
            finally:
                self.request_queue.task_done()
            return
        super()._handle_request(request_data)

    def _handle_ced_load(self, task: CEDLoadTask) -> None:
        req_meta = task.req_meta
        try:
            load_masks = self.token_database.load_mask(req_meta.block_hashes, task.token_len)
        except AssertionError:
            load_masks = None
        unit_chunks: list[tuple[CEDLoadUnit, CEDSaveChunks]] = []
        for unit in task.units:
            chunks = self._list_load_chunks(req_meta, task.token_len, unit.group_id, load_masks)
            if chunks.keys:
                unit_chunks.append((unit, chunks))
        if not unit_chunks:
            task.settle_all_gates()
            return
        keys = []
        for _unit, chunks in unit_chunks:
            keys.extend(chunks.keys)
        keys = list(dict.fromkeys(keys))
        key_infos = self.m_store.batch_get_key_info(keys)
        if key_infos is None or len(key_infos) != len(keys):
            raise RuntimeError(f"CED load key resolution returned {len(key_infos or [])} infos for {len(keys)} keys")
        gva_by_key: dict[str, int] = {}
        for key, key_info in zip(keys, key_infos, strict=True):
            gvas = key_info.gva_list() if key_info is not None else []
            gva_by_key[key] = int(gvas[0]) if gvas else 0
        for unit, chunks in unit_chunks:
            src_gvas: list[int] = []
            dst_addrs: list[int] = []
            sizes: list[int] = []
            for index, key in enumerate(chunks.keys):
                gva = gva_by_key.get(key, 0)
                if gva <= 0:
                    with self._invalid_block_ids_lock:
                        self._invalid_block_ids.add(chunks.block_ids[index])
                    raise RuntimeError(f"CED load key {key} is not resolvable for request {req_meta.req_id}")
                rows = chunk_member_transfer(
                    self.token_database,
                    unit.group_id,
                    chunks.block_ids[index],
                    chunks.starts[index],
                    chunks.ends[index],
                    unit.member_start,
                    unit.member_end,
                )
                for device_addr, object_offset, size in rows:
                    src_gvas.append(gva + object_offset)
                    dst_addrs.append(device_addr)
                    sizes.append(size)
            if src_gvas:
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

    def _list_load_chunks(self, req_meta: ReqMeta, token_len: int, group_id: int, load_masks: Any) -> CEDSaveChunks:
        chunks = CEDSaveChunks()
        if group_id >= len(req_meta.block_ids_by_group):
            return chunks
        block_ids = req_meta.block_ids_by_group[group_id]
        group_block_size = self._get_block_size(group_id)
        load_spec = req_meta.load_spec
        mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size if load_spec else 0

        def chunk_filter(start: int) -> bool:
            return self.token_database.mask_allows_chunk(load_masks, group_id, start)

        iterator = self.token_database.process_token_key_strings_with_block_ids(
            token_len,
            req_meta.block_hashes,
            block_ids,
            mask_num,
            kv_cache_group_id=group_id,
            skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
            chunk_filter=chunk_filter,
        )
        for start, end, key, _block_hash, block_id in iterator:
            chunks.starts.append(start)
            chunks.ends.append(end)
            chunks.keys.append(key)
            chunks.block_ids.append(block_id)
        return chunks

    def _fail_ced_load(self, task: CEDLoadTask) -> None:
        task.settle_all_gates()
        self._mark_failed_requests([task.req_meta])


class CEDSaveWorker(KVPoolWorker):
    """Bulk pool worker extended with the CED incremental session save.

    Inherits the non-layerwise AscendStore behavior unchanged; the CED
    per-slice session save is layered on top and only activates for trigger
    points present in the derived map. Groups and steps without early
    triggers keep the step-end bulk save (draft groups, decode fallbacks,
    replay steps without derived slices).

    CED sessions never advance the request-wide saved-offset; the plan's
    readability probe is the dedup, so revoked or failed sessions retry on
    the next step. BlockStored KV events are not yet emitted for session
    saves.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, use_layerwise, kv_cache_config)
        self.ced_trigger_map = build_ced_trigger_map(
            self._cacheable_group_layer_names(),
            _kv_source_layer_ids(self.hf_config),
        )
        self._ced_fallback_groups: dict[str, list[int]] = {}
        self._ced_fired_triggers: set[int] = set()
        self._ced_load_gates: dict[str, dict[int, threading.Event]] = {}
        # The pipelined load rides the async-loading connector contract: the
        # scheduler holds the request in WAITING_FOR_REMOTE_KV until
        # get_finished reports it, and the early finished notification below
        # releases it while units stream in behind the in-model gates.
        self._ced_load_capable = (
            bool(self.ced_trigger_map.slices_by_group)
            and self.load_async
            and callable(getattr(self.m_store, "batch_get_key_info", None))
        )
        if self.ced_trigger_map.slices_by_group:
            if self.pp_size > 1:
                logger.warning("CED_layerwise session saves fall back to bulk under PP>1 in this revision.")
                self.ced_trigger_map = CEDTriggerMap()
            elif not (
                callable(getattr(self.m_store, "batch_alloc", None))
                and callable(getattr(self.m_store, "batch_write_finish", None))
            ):
                logger.warning(
                    "CED_layerwise requires a session-capable backend (memcache); falling back to bulk saves."
                )
                self.ced_trigger_map = CEDTriggerMap()
            elif self.enable_kv_event:
                logger.warning("CED_layerwise session saves do not emit BlockStored KV events yet.")
        self.ced_trigger_layers = self.ced_trigger_map.trigger_layers()
        if self.ced_trigger_layers:
            logger.info(
                "CED incremental pool path active: groups=%d trigger_layers=%s",
                len(self.ced_trigger_map.slices_by_group),
                list(self.ced_trigger_layers[:8]),
            )

    def _cacheable_group_layer_names(self) -> dict[int, list[str]]:
        if self.kv_cache_config is None:
            return {}
        groups = self.kv_cache_config.kv_cache_groups
        return {
            group_id: list(groups[group_id].layer_names)
            for group_id in range(len(groups))
            if group_id in self.cacheable_group_ids
        }

    # ------------------------------------------------------------------
    # Thread selection
    # ------------------------------------------------------------------
    def _start_kv_transfer_threads(self) -> None:
        if self._transfer_threads_started:
            return
        ced_active = bool(self.ced_trigger_layers)
        if ced_active and (self.can_put or self._ced_load_capable):
            if self.can_put:
                ready_event_sending = threading.Event()
                self.kv_send_thread = CEDSessionSendThread(
                    self.m_store,
                    self.token_database,
                    self.grouped_block_size,
                    self.tp_rank,
                    self.tp_size,
                    0 if self.use_kvpp else self.pcp_rank,
                    1 if self.use_kvpp else self.pcp_size,
                    self.dcp_size,
                    self.put_step,
                    self.kv_role,
                    ready_event_sending,
                    self.group_uses_align_state,
                    self.enable_kv_events,
                    worker=self if self.tp_mismatch else None,
                    trigger_map=self.ced_trigger_map,
                    max_transfer_blocks=int(self._extra_config.get("ced_max_transfer_blocks", 0)),
                    max_transfer_bytes=int(self._extra_config.get("ced_max_transfer_bytes", 0)),
                )
                self.kv_send_thread.start()
                ready_event_sending.wait()
            if self.load_async:
                self._start_ced_recving_thread()
            self._transfer_threads_started = True
            return
        super()._start_kv_transfer_threads()

    def _start_ced_recving_thread(self) -> None:
        ready_event = threading.Event()
        common_args = (
            self.m_store,
            self.token_database,
            self.grouped_block_size,
            self.tp_rank,
            self.tp_size,
            self.dcp_size,
            ready_event,
        )
        common_kwargs = {
            "invalid_block_ids": self._invalid_block_ids,
            "invalid_block_ids_lock": self._invalid_block_ids_lock,
            "worker": self if self.tp_mismatch else None,
            "record_operation": self._record_kv_connector_operation,
        }
        if self._ced_load_capable:
            self.kv_recv_thread = CEDPipelinedRecvingThread(
                *common_args,
                **common_kwargs,
                max_transfer_blocks=int(self._extra_config.get("ced_max_transfer_blocks", 0)),
                max_transfer_bytes=int(self._extra_config.get("ced_max_transfer_bytes", 0)),
            )
        else:
            self.kv_recv_thread = KVCacheStoreRecvingThread(*common_args, **common_kwargs)
        self.kv_recv_thread.start()
        ready_event.wait()

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------
    def start_load_kv(self, metadata: Any):
        self._ced_fired_triggers = set()
        self._ced_fallback_groups.clear()
        self._ced_load_gates.clear()
        if self._ced_load_capable and metadata.requests:
            self._prepare_ced_loads(metadata)
        if self.can_put and self.ced_trigger_layers and metadata.requests:
            send_thread = self.kv_send_thread
            if isinstance(send_thread, CEDSessionSendThread):
                self._prepare_ced_sessions(send_thread, metadata)
        super().start_load_kv(metadata)

    def _prepare_ced_loads(self, metadata: Any) -> None:
        recv_thread = self.kv_recv_thread
        if not isinstance(recv_thread, CEDPipelinedRecvingThread):
            return
        for request in metadata.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:
                continue
            group_ids = request.kv_cache_group_ids or [0]
            if any(group not in self.ced_trigger_map.slices_by_group for group in group_ids):
                # Draft or unrecognized groups keep the bulk load.
                continue
            token_len = request.token_len_chunk
            if (load_spec.kvpool_cached_tokens % self.cache_transfer_granularity != 0) and (
                load_spec.kvpool_cached_tokens == token_len - 1
            ):
                token_len = load_spec.kvpool_cached_tokens + 1
            else:
                token_len = load_spec.kvpool_cached_tokens
            units = self._build_ced_load_units(set(group_ids))
            if not units:
                continue
            gate_events = {unit.gate_layer: threading.Event() for unit in units}
            remaining: dict[int, int] = {}
            for unit in units:
                remaining[unit.gate_layer] = remaining.get(unit.gate_layer, 0) + 1
            recv_thread.add_request(
                CEDLoadTask(
                    req_meta=request,
                    token_len=token_len,
                    units=units,
                    gate_events=gate_events,
                    remaining_by_gate=remaining,
                )
            )
            # Early release: the request reports finished recving as soon as
            # its unit task is queued, so the scheduler starts the forward
            # while units stream in; the in-model gates enforce arrival.
            recv_thread.set_finished_request(request.req_id)
            self._ced_load_gates[request.req_id] = gate_events
            load_spec.can_load = False

    def _build_ced_load_units(self, group_ids: set[int]) -> list[CEDLoadUnit]:
        """One load unit per (group, consumer layer), members merged.

        Uses the physical layer -> (group, member index) map, so full-context
        slots gate at their KV source layer and every SWA member gates at its
        own layer without model-specific branching.
        """
        by_group_layer: dict[tuple[int, int], tuple[int, int]] = {}
        for local_layer, entries in self.physical_layer_to_group_layers.items():
            for group_id, member_index in entries:
                if group_id not in group_ids:
                    continue
                start, end = by_group_layer.get((group_id, local_layer), (member_index, member_index + 1))
                by_group_layer[(group_id, local_layer)] = (min(start, member_index), max(end, member_index + 1))
        units = [
            CEDLoadUnit(group_id, start, end, local_layer)
            for (group_id, local_layer), (start, end) in by_group_layer.items()
        ]
        units.sort(key=lambda unit: unit.gate_layer)
        return units

    def wait_for_ced_layer_load(self, layer_idx: int) -> None:
        """Block the forward at a consumer layer until its unit arrives.

        Polls the recving thread while waiting so a fatal transfer error
        surfaces here instead of hanging the step; request-level load
        failures raise so the step aborts before reading half-loaded KV.
        """
        gates = self._ced_load_gates
        if not gates:
            return
        recv_thread = self.kv_recv_thread
        if recv_thread is None:
            return
        pending = [(req_id, events[layer_idx]) for req_id, events in gates.items() if layer_idx in events]
        if not pending:
            return
        for _req_id, event in pending:
            while not event.wait(timeout=1.0):
                recv_thread.raise_if_failed()
        with recv_thread.done_task_lock:
            failed = [req_id for req_id, _event in pending if req_id in recv_thread.failed_requests]
        if failed:
            raise RuntimeError(f"CED pipelined load failed for requests {failed}")
        recv_thread.raise_if_failed()

    def _prepare_ced_sessions(self, send_thread: CEDSessionSendThread, metadata: Any) -> None:
        for request in metadata.requests:
            if request.can_save is None or not request.can_save:
                continue
            claimed: list[int] = []
            group_ids = request.kv_cache_group_ids or [0]
            try:
                store_masks = self.token_database.store_mask(request.token_len_chunk, request.num_prompt_tokens)
            except AssertionError:
                store_masks = None
            for group_id in group_ids:
                if group_id not in self.ced_trigger_map.slices_by_group:
                    continue
                chunks = self._list_ced_save_chunks(send_thread, request, group_id, store_masks)
                if not chunks.keys:
                    continue
                object_size = sum(self.token_database.group_block_len[group_id])
                send_thread.open_ced_session(request, group_id, chunks, object_size)
                claimed.append(group_id)
            if claimed:
                self._ced_fallback_groups[request.req_id] = [group for group in group_ids if group not in claimed]

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
        send_thread = self.kv_send_thread
        if not isinstance(send_thread, CEDSessionSendThread):
            # Fallback mode (backend without session support, PP>1): the
            # parent behavior covers every group through the bulk path.
            super().wait_for_save(connector_metadata)
            return
        requests: list[ReqMeta] = []
        current_event = None
        for request in connector_metadata.requests:
            if request.can_save is None or not request.can_save:
                continue
            unclaimed = self._ced_fallback_groups.pop(request.req_id, None)
            if unclaimed:
                fallback = self._copy_req_meta_for_groups(request, unclaimed)
                if current_event is None:
                    current_event = torch.npu.Event()
                    current_event.record()
                fallback.skip_null_blocks_by_group = self.group_uses_align_state
                fallback.current_event = current_event
                requests.append(fallback)
        # Always enqueue the fence batch: FIFO order makes it cover this
        # step's session tasks as well, so wait_for_previous_save fences both.
        self._previous_save_batch = send_thread.add_save_batch(requests)

    @staticmethod
    def _copy_req_meta_for_groups(request: ReqMeta, group_ids: list[int]) -> ReqMeta:
        return ReqMeta(
            req_id=request.req_id,
            token_len_chunk=request.token_len_chunk,
            block_ids_by_group=request.block_ids_by_group,
            block_hashes=request.block_hashes,
            can_save=request.can_save,
            load_spec=request.load_spec,
            is_last_chunk=request.is_last_chunk,
            kv_cache_group_ids=group_ids,
            skip_null_blocks_by_group=request.skip_null_blocks_by_group,
            num_prompt_tokens=request.num_prompt_tokens,
            token_ids=request.token_ids,
            original_block_size=request.original_block_size,
            save_end_token=request.save_end_token,
            target_token_len=request.target_token_len,
            save_start_token=request.save_start_token,
        )

    def on_ced_save_trigger(self, trigger_layer: int) -> None:
        """Handle a per-slice save trigger raised at a V4.1 layer boundary.

        The dsa_v41 hooks call this at every attention layer end and at the
        cut layer's cache-only write, so it is idempotent per step and a
        no-op for layers without derived slices.
        """
        if trigger_layer in self._ced_fired_triggers:
            return
        slices = self.ced_trigger_map.slices_by_trigger.get(trigger_layer)
        if not slices:
            return
        send_thread = self.kv_send_thread
        if not isinstance(send_thread, CEDSessionSendThread) or not send_thread.is_alive():
            return
        self._ced_fired_triggers.add(trigger_layer)
        event = torch.npu.Event()
        event.record()
        send_thread.enqueue_slice_tasks(trigger_layer, event)
