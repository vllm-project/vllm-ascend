# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared layouts executed through the pool's existing layerwise send queue.

This owns no scheduler, model state, or RPC server. The existing connector
supplies step metadata and layer readiness; the existing backend owns transport.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from vllm.logger import logger
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec, requires_padded_page_layout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    submit_layer_load_window,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import (
    StoreLayerLoadTask,
    StoreLayerTask,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata, block_hash_to_str
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_io import MemcacheStoreLayoutIO
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_layout import RequestStateLayout, SharedPoolLayout


def group_specs(group):
    spec = group.kv_cache_spec
    return {
        name: spec.kv_cache_specs[name] if isinstance(spec, UniformTypeKVCacheSpecs) else spec
        for name in group.layer_names
    }


def target_group_specs(group, draft_names=()):
    """Use runner-provided ownership to keep the target object schema stable."""
    return {
        name: item
        for name, item in group_specs(group).items()
        if not getattr(item, "is_draft", False) and name not in draft_names
    }


def transfers_mtp_cache(config):
    transfer = config.kv_transfer_config
    speculative = getattr(config, "speculative_config", None)
    return (
        getattr(transfer, "kv_connector", None) == "LayerwisePoolConnector"
        and getattr(speculative, "method", None) == "mtp"
    )


def draft_block_digest(hashes, block_hash_index):
    # The MTP input at the last slot is shifted by one token. The following
    # prefix hash commits to that token as well as the complete current block.
    next_index = block_hash_index + 1
    if next_index < len(hashes):
        return "mtp_" + block_hash_to_str(hashes[next_index])
    return None


def shared_draft_names(cache_config):
    """Resolve auxiliary indexer ownership from its declared main cache.

    Ascend MRV2 builds indexer specs before tagging ordinary attention specs
    as draft. Use the same main/indexer name relation as that runner instead
    of assuming the indexer flag is populated, or guessing from layer numbers.
    """
    from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendSFAIndexerCacheSpec

    all_specs = {name: spec for group in cache_config.kv_cache_groups for name, spec in group_specs(group).items()}
    draft = {name for name, spec in all_specs.items() if getattr(spec, "is_draft", False)}
    for name, spec in all_specs.items():
        if isinstance(spec, AscendSFAIndexerCacheSpec):
            suffix = ".indexer.k_cache"
        elif isinstance(spec, AscendIndexerKPoolTailSpec):
            suffix = ".indexer.tail_cache"
        else:
            continue
        if name.endswith(suffix):
            main_name = name.removesuffix(suffix) + ".attn"
            if main_name in draft:
                draft.add(name)
    return draft


def shared_pool_uses_layerwise(config):
    # Full graph replay does not execute the Python layer callbacks. Keep
    # pool I/O at the runner's step boundaries whenever graphs may run.
    extra = config.kv_transfer_config.kv_connector_extra_config
    return bool(extra.get("use_layerwise", False)) and config.model_config.enforce_eager


def validate_shared_pool(config, cache_config, *, worker=True):
    """Reject unsupported contracts before constructing native clients."""
    extra = config.kv_transfer_config.kv_connector_extra_config
    store_tp = extra["store_tp_size"]
    tp = config.parallel_config.tensor_parallel_size
    if type(store_tp) is not int or store_tp < tp or store_tp % tp:
        raise ValueError("store_tp_size must be a positive multiple of local TP")
    if not isinstance(extra.get("cache_prefix"), str) or not extra["cache_prefix"]:
        raise ValueError("Shared pool requires an explicit cache_prefix for model weights/epoch")
    if cache_config is None or not cache_config.kv_cache_groups:
        raise ValueError("Shared pool requires the actual cache configuration")
    p = config.parallel_config
    if any(
        size != 1
        for size in (p.pipeline_parallel_size, p.prefill_context_parallel_size, p.decode_context_parallel_size)
    ):
        raise ValueError("Shared pool currently requires PP=PCP=DCP=1")
    if not config.use_v2_model_runner:
        raise ValueError("Shared pool currently requires MRV2")
    if extra.get("backend", "mooncake").lower() != "memcache" or extra.get("load_async", False):
        raise ValueError("Shared pool currently requires MemCache with synchronous reads")
    if extra.get("discard_partial_chunks", True) is not True:
        raise ValueError("Shared pool currently publishes complete cache blocks only")
    if config.kv_events_config and config.kv_events_config.enable_kv_cache_events:
        raise ValueError("Shared pool KV events are not implemented")
    if extra.get("layerwise_num_shared_buffers", 0) or extra.get("layerwise_prefetch_layers", 0):
        raise ValueError("Shared pool buffer reuse requires separate lifetime validation")
    if not worker:
        # Per-owner validation belongs to the worker, before registration.
        # The scheduler must subsequently obtain its registered group manifest.
        return
    draft_names = shared_draft_names(cache_config)
    speculative = getattr(config, "speculative_config", None)
    if speculative is not None and speculative.use_eagle():
        if (
            not draft_names
            or not any(target_group_specs(group, draft_names) for group in cache_config.kv_cache_groups)
            or any(
                not hasattr(spec, "is_draft")
                for group in cache_config.kv_cache_groups
                for spec in group_specs(group).values()
            )
        ):
            raise ValueError("Shared pool MTP requires explicit target and draft cache ownership from the runner")
    for group in cache_config.kv_cache_groups:
        if getattr(group, "host_resident", False):
            raise ValueError("Shared pool does not yet support host-resident cache groups")
        specs = group_specs(group) if transfers_mtp_cache(config) else target_group_specs(group, draft_names)
        for item in specs.values():
            if is_circular_kv_cache_spec(item):
                # Only PD restores these request-lifetime pages. Ordinary
                # pooling leaves them to the model's aligned-prefix replay.
                continue
            if isinstance(item, MambaSpec):
                if item.mamba_cache_mode != "align" or len(item.shapes) != 2:
                    raise ValueError("Shared state requires aligned committed conv/recurrent checkpoints")
                if store_tp != tp:
                    raise ValueError("Shared state runtime currently requires matching TP and Store TP")
                continue
            if isinstance(item, MLAAttentionSpec):
                ratio = item.tokens_per_state
                if item.num_kv_heads != 1 or ratio < 1 or int(ratio) != ratio or item.block_size % ratio:
                    raise ValueError("Shared MLA requires one replicated KV head with integral aligned compression")
                continue
            # Both MHA and GQA use FullAttentionSpec. GDN byte layouts are
            # supported by SharedPoolLayout, but need checkpoint handoff first.
            if type(item) is not FullAttentionSpec:
                raise ValueError("Shared pool has no component adapter for this cache spec")
            if item.non_causal or item.sliding_window is not None or item.attention_chunk_size is not None:
                raise ValueError("Shared pool currently requires causal full-context attention")
            if item.tokens_per_state != 1 or item.num_heads != item.num_kv_heads:
                raise ValueError("Compressed/special head-slot content needs an explicit adapter")
            if item.max_tp_shards is not None and tp > item.max_tp_shards:
                raise ValueError("Replicated KV heads need a shared-shard ownership policy")
            if item.num_heads * tp % store_tp:
                raise ValueError("KV heads must divide into whole Store shards")
            expected = (item.head_size + item.head_size_v) * torch.empty((), dtype=item.dtype).element_size()
            if item.kv_quant_mode != KVQuantMode.NONE or item.state_content_size_bytes != expected:
                raise ValueError("Quantized/split-scale shared content needs an explicit adapter")


@dataclass(frozen=True)
class _SharedSlot:
    index: int
    base: int
    stride: int
    blocks: int
    group: int

    def contains(self, block, address, size):
        start = self.base + block * self.stride
        return 0 < block < self.blocks and start <= address < address + size <= start + self.stride


def _certified_shared_slots(config, caches):
    # The padded shared-slot allocator gives different cache groups views of
    # the same numbered physical pages. The scheduler allocates those IDs from
    # one BlockPool. Verify the runtime views still use that exact geometry.
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import KV_CACHE_BUFFER_ALIGNMENT, as_kv_cache_tensors

    owner_groups = {name: gid for gid, group in enumerate(config.kv_cache_groups) for name in group.layer_names}
    slots = {}
    for index, descriptor in enumerate(getattr(config, "kv_cache_tensors", ())):
        names = descriptor.layers
        groups = {owner_groups.get(name) for name in names}
        stride, blocks = descriptor.block_stride, config.num_blocks
        if (
            len(names) < 2
            or len(groups) != len(names)
            or None in groups
            or not all(name in caches for name in names)
            or descriptor.layer_stride != 0
            or descriptor.offset != 0
            or stride <= 0
            or blocks <= 0
            or descriptor.size != stride * blocks
        ):
            continue
        parts = [tuple(tensor for tensor in as_kv_cache_tensors(caches[name]) if tensor.numel()) for name in names]
        if not all(parts):
            continue
        tensors = [tensor for owner in parts for tensor in owner]
        storages = {(tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes()) for tensor in tensors}
        if len(storages) != 1:
            continue
        ((storage, storage_bytes),) = storages
        base = (storage + KV_CACHE_BUFFER_ALIGNMENT - 1) // KV_CACHE_BUFFER_ALIGNMENT * KV_CACHE_BUFFER_ALIGNMENT
        if base + descriptor.size > storage + storage_bytes:
            continue
        if all(_uses_slot_pages(tensor, base, stride, blocks) for tensor in tensors):
            slots.update((name, _SharedSlot(index, base, stride, blocks, owner_groups[name])) for name in names)
    return slots


def _uses_slot_pages(tensor, base, page_stride, blocks):
    if tensor.ndim < 1 or tensor.shape[0] % blocks or any(stride < 0 for stride in tensor.stride()):
        return False
    rows_per_page = tensor.shape[0] // blocks
    itemsize = tensor.element_size()
    span = 1 + sum(
        (length - 1) * stride
        for length, stride in zip((rows_per_page, *tensor.shape[1:]), tensor.stride(), strict=True)
    )
    start = tensor.data_ptr() - base
    return (
        tensor.stride(0) * rows_per_page * itemsize == page_stride
        and 0 <= start < start + span * itemsize <= page_stride
    )


def _exclusive_group_blocks(requests):
    owners = {}
    for request in requests:
        for gid, blocks in enumerate(request.block_ids_by_group):
            for block in blocks:
                if block > 0 and owners.setdefault(block, gid) != gid:
                    return False
    return True


def _disjoint_buffer_owners(caches, shared_slots=None):
    # Include stride gaps in the byte span. A conservative fallback keeps
    # shared or reused cache buffers on their existing per-layer copy path.
    intervals = []
    shared_slots = shared_slots or {}
    for owner, value in caches.items():
        tensors = (value,) if isinstance(value, torch.Tensor) else value
        for tensor in tensors:
            if not tensor.numel():
                continue
            start = tensor.data_ptr()
            strides = tensor.stride()
            if not start or any(stride < 0 for stride in strides):
                return set()
            width = 1 + sum((count - 1) * stride for count, stride in zip(tensor.shape, strides, strict=True))
            intervals.append((start, start + width * tensor.element_size(), owner))
    intervals.sort()
    overlapping = set()
    for i, (_, stop, owner) in enumerate(intervals):
        for j in range(i + 1, len(intervals)):
            start, _, other = intervals[j]
            if start >= stop:
                break
            owner_slot, other_slot = shared_slots.get(owner), shared_slots.get(other)
            proven = owner_slot is not None and other_slot is not None and owner_slot.index == other_slot.index
            if owner != other and not proven:
                overlapping.update((owner, other))
    return {owner for _, _, owner in intervals} - overlapping


class SharedPoolTransfer:
    def __init__(self, worker):
        self.worker = worker
        extra = worker.vllm_config.kv_transfer_config.kv_connector_extra_config
        self.layerwise = shared_pool_uses_layerwise(worker.vllm_config)
        self.io = MemcacheStoreLayoutIO(worker.m_store, int(extra.get("store_lease_ttl_ms", 600000)))
        self._coalesce_prefix_reads = (
            extra.get("coalesce_prefix_reads", False)
            and self.layerwise
            and worker.vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.layouts = {}
        self.draft_layouts = {}
        self.request_states = {}
        self.draft_request_states = {}
        self.regions = {}
        self.saves = []
        self.loads = []
        self.step = None
        self.completed_events = {}
        self.layer_bundles = {}
        self._executing_owner = None
        self._registered = False
        self._sender = None
        self._prefetch_enabled = (
            extra.get("shared_pool_prefetch", False)
            and self.layerwise
            and worker.vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self._prefetch_groups = ()
        self._prefetch_depth = 2
        self._prefetch_order = ()
        self._prefetch_position = {}
        self._prefetch_work = ()
        self._prefetch_next = 0
        self._prefetch_tasks = ()
        self._prefetch_sync = set()
        self._prefetch_pending = set()
        self._prefetch_shared_slots = {}
        self._receiver = None
        self._prefetch_events = {}
        self._prefetch_copy_lock = threading.Lock()
        self._lookup_executor = None
        if (
            extra.get("parallel_prefix_group_lookup", False)
            and worker.vllm_config.kv_transfer_config.kv_role == "kv_producer"
        ):
            self._lookup_executor = ThreadPoolExecutor(
                max_workers=2, initializer=worker.m_store.set_device, thread_name_prefix="pool-lookup"
            )

    def register(self, caches):
        worker = self.worker
        config = worker.kv_cache_config
        draft_names = shared_draft_names(config)
        mtp_cache = transfers_mtp_cache(worker.vllm_config)
        for gid, group in enumerate(config.kv_cache_groups):
            target = target_group_specs(group, draft_names)
            self._register_group(gid, group.kv_cache_spec.block_size, target, caches, self.layouts, self.request_states)
            if mtp_cache:
                draft = {name: spec for name, spec in group_specs(group).items() if name in draft_names}
                self._register_group(
                    gid, group.kv_cache_spec.block_size, draft, caches, self.draft_layouts, self.draft_request_states
                )
        worker.m_store.register_buffer(list(self.regions), list(self.regions.values()))
        if self._coalesce_prefix_reads:
            self.io.set_coalesced_read_regions(self.regions)
        layouts = [layout for _, layout, _ in self.prefix_layouts()] + [
            layout for _, layout in self.request_state_layouts()
        ]
        if any(layout.replicated for layout in layouts):
            from vllm_ascend.core.kv_cache_placement import build_layer_cache_bundles

            specs = {name: spec for layout in layouts for name, spec in layout.specs.items()}
            self.layer_bundles = build_layer_cache_bundles(specs)
        if self._prefetch_enabled:
            from vllm_ascend.core.kv_cache_placement import build_layer_cache_bundles

            specs = {name: spec for layout in layouts for name, spec in layout.specs.items()}
            bundles = build_layer_cache_bundles(specs, draft_names & specs.keys())
            if requires_padded_page_layout(
                spec for group in config.kv_cache_groups for spec in group_specs(group).values()
            ):
                self._prefetch_shared_slots = _certified_shared_slots(config, caches)
            disjoint = _disjoint_buffer_owners(caches, self._prefetch_shared_slots)
            self._prefetch_order = tuple(frozenset(owners) for owners in bundles.values())
            self._prefetch_groups = tuple(bundle for bundle in self._prefetch_order if disjoint.issuperset(bundle))
            if self._prefetch_groups:
                logger.info(
                    "Shared pool P prefix prefetch enabled for %d/%d layer groups (validated shared-slot owners=%d)"
                    " with %d layers of lookahead",
                    len(self._prefetch_groups),
                    len(bundles),
                    len(self._prefetch_shared_slots),
                    self._prefetch_depth,
                )
            else:
                logger.info("Shared pool P prefix prefetch disabled for overlapping cache buffers")
        self._registered = True

    def _register_group(self, gid, chunk_size, specs, caches, layouts, request_states):
        if not specs:
            return
        worker = self.worker
        config = worker.kv_cache_config
        extra = worker.vllm_config.kv_transfer_config.kv_connector_extra_config
        tensors = {}
        for name in specs:
            value = caches[name]
            # Some runners return stacked (K,V), others expose two views.
            tensors[name] = (
                tuple(value.unbind(0))
                if isinstance(value, torch.Tensor) and value.ndim == 5 and value.shape[0] == 2
                else worker._as_cache_tuple(value)
            )
        identity = KeyMetadata(worker.model_name, worker.tp_rank, 0, 0, gid)
        if all(is_circular_kv_cache_spec(item) for item in specs.values()):
            request_states[gid] = RequestStateLayout(
                identity, specs, tensors, config.num_blocks, extra["cache_prefix"], worker.tp_rank
            )
        else:
            layout = SharedPoolLayout(
                identity,
                specs,
                local_tp=worker.tp_size,
                store_tp=extra["store_tp_size"],
                rank=worker.tp_rank,
                chunk_size=chunk_size,
                hash_block_size=worker.hash_block_size,
                cache_prefix=extra["cache_prefix"],
            )
            layout.register(tensors, config.num_blocks)
            layouts[gid] = layout
        for entries in tensors.values():
            for tensor in entries:
                storage = tensor.untyped_storage()
                self.regions[storage.data_ptr()] = storage.nbytes()

    def prefix_layouts(self):
        for gid, layout in self.layouts.items():
            yield gid, layout, False
        for gid, layout in self.draft_layouts.items():
            yield gid, layout, True

    def request_state_layouts(self):
        yield from self.request_states.items()
        yield from self.draft_request_states.items()

    def describe_groups(self):
        if not self._registered:
            raise RuntimeError("Shared pool caches have not been registered")
        prefix_groups = set(self.layouts) | set(self.draft_layouts)
        state_groups = set(self.request_states) | set(self.draft_request_states)
        return {
            "version": 1,
            "group_count": len(self.worker.kv_cache_config.kv_cache_groups),
            "target_groups": sorted(prefix_groups | state_groups),
            "prefix_groups": sorted(prefix_groups),
            "stateful": bool(state_groups)
            or any(
                isinstance(spec, MambaSpec) or getattr(spec, "tokens_per_state", 1) != 1
                for _, layout, _ in self.prefix_layouts()
                for spec in layout.specs.values()
            ),
        }

    def _chunks(self, request, *, writing):
        load = request.load_spec
        for gid, layout, draft in self.prefix_layouts():
            if writing and layout.replicated and layout.writer_rank != 0:
                continue
            length = request.save_end_token if writing else load.kvpool_cached_tokens
            start = request.save_start_token if writing else load.vllm_cached_tokens
            # A last-token recompute still requires restoring its complete
            # immutable source block. Local APC blocks preceding it are skipped.
            first = math.ceil(start / layout.chunk_size) if writing else start // layout.chunk_size
            if writing and request.published_prompt_end:
                # MTP's next-token hash can become available only after D
                # starts computing. Only skip keys P could have published.
                published = request.published_prompt_end - (layout.hash_block_size if draft else 0)
                first = max(first, published // layout.chunk_size)
            last = length // layout.chunk_size if writing else math.ceil(length / layout.chunk_size)
            blocks = request.block_ids_by_group[gid]
            if layout.state and not writing:
                # A recurrent checkpoint replaces all earlier states. Loading
                # historical checkpoints is unnecessary and their allocator
                # slots may already be null in align mode.
                first = max(first, last - 1)
            for index in range(first, last):
                end = (index + 1) * layout.chunk_size
                hash_index = end // layout.hash_block_size - 1
                if hash_index >= len(request.block_hashes) or index >= len(blocks):
                    raise ValueError("Shared object has no committed prefix hash or allocated block")
                if blocks[index] <= 0:
                    if layout.state and writing and blocks[index] == 0:
                        # Align mode retains only materialized checkpoints.
                        # A null historical slot must never be published using
                        # the bytes of the current live state.
                        continue
                    raise ValueError("Shared full-attention object cannot reference a null block")
                if draft:
                    digest = draft_block_digest(request.block_hashes, hash_index)
                    if digest is None:
                        if writing:
                            continue  # The terminal block can only be published by PD.
                        raise ValueError("Shared MTP prefix is missing its next-token hash")
                else:
                    digest = block_hash_to_str(request.block_hashes[hash_index])
                yield layout, index * layout.chunk_size, blocks[index], digest

    def _plans(self, request, *, writing):
        return tuple(
            plan
            for layout, start, block, digest in self._chunks(request, writing=writing)
            for plan in layout.objects(start, block, digest)
        )

    def _write_plans(self, request, seen):
        if request.published_prompt_end or request.shared_pool_prefix_end <= request.save_start_token:
            plans = tuple(plan for plan in self._plans(request, writing=True) if plan.key not in seen)
            seen.update(plan.key for plan in plans)
            return plans

        # A warm restore often makes the same prefix eligible for saving.
        # Resolve its Store keys before building local address ranges.
        chunks, keys = [], []
        for layout, start, block, digest in self._chunks(request, writing=True):
            if block >= layout.num_blocks:
                raise ValueError("Invalid local block or Store chunk boundary")
            chunk_keys = [key for shard in layout.local_shards if (key := layout.key(shard, digest)) not in seen]
            if chunk_keys:
                chunks.append((layout, start, block, digest, chunk_keys))
                keys.extend(chunk_keys)
        if len(set(keys)) != len(keys):
            raise ValueError("One session must contain unique object keys")
        if not keys:
            return ()
        seen.update(keys)
        readable = self.io.batch_is_exist(keys)
        missing = {key for key, hit in zip(keys, readable, strict=True) if not hit}
        return tuple(
            plan
            for layout, start, block, digest, chunk_keys in chunks
            if any(key in missing for key in chunk_keys)
            for plan in layout.objects(start, block, digest)
            if plan.key in missing
        )

    def prepare_step(self, metadata):
        if self.step is metadata:
            return
        if self.loads or self.saves:
            raise RuntimeError("Previous shared pool step is not drained")
        self.step = metadata
        seen = set()
        try:
            for request in metadata.requests:
                if request.can_save and self.worker.can_put:
                    plans = self._write_plans(request, seen)
                    if request.published_prompt_end and not plans:
                        if request.event_id is not None:
                            self.completed_events[request.event_id] = 1
                        continue
                    # Recheck the remaining keys immediately before allocation.
                    self.saves.append((request, self.io.begin(plans, writing=True)))
        except Exception:
            self._abort_all()
            raise

    def start_load(self, metadata):
        try:
            for request in metadata.requests:
                if request.load_spec is not None and request.load_spec.can_load:
                    session = self.io.begin(self._plans(request, writing=False), writing=False)
                    self.loads.append((request, session))
                    if not self.layerwise:
                        self.io.copy(session)
                        self.io.finish(session)
            if self._prefetch_groups and self.loads:
                self._start_prefetch(metadata.requests)
        except Exception:
            self._abort_all()
            raise

    def _start_prefetch(self, requests):
        sessions = tuple(session for _, session in self.loads)
        pending = set()
        unsafe = set()
        slots = self._prefetch_shared_slots
        if slots and not _exclusive_group_blocks(requests):
            unsafe.update(slots)
        for request, session in self.loads:
            blocks = tuple(set(group) for group in request.block_ids_by_group) if slots else ()
            for plan in session.plans:
                # Full-block geometry is cached per physical slot. A layer's
                # ranges all fit that slot exactly when their bounds fit.
                bounds = getattr(plan, "owner_bounds", ()) or (
                    (part.owner, part.address, part.size) for part in plan.ranges
                )
                for owner, address, size in bounds:
                    pending.add(owner)
                    slot = slots.get(owner)
                    if slot is not None and (
                        slot.group >= len(blocks)
                        or plan.block_id not in blocks[slot.group]
                        or not slot.contains(plan.block_id, address, size)
                    ):
                        unsafe.add(owner)
        eligible = set(self._prefetch_groups)
        order = self._prefetch_order or self._prefetch_groups
        groups = tuple((bundle, bundle & pending) for bundle in order)
        if not any(owners and bundle in eligible and bundle.isdisjoint(unsafe) for bundle, owners in groups):
            return
        # Prepare each physical layer's ranges before it enters the receive queue.
        for session in sessions:
            self.io.prepare_layer_reads(session, tuple(bundle for bundle, _ in groups))
        if self._receiver is None:
            worker = self.worker
            self._receiver = KVCacheStoreLayerRecvingThread(
                worker.m_store,
                worker.token_database,
                worker.block_size,
                worker.tp_rank,
                worker.tp_size,
                worker.dcp_size,
                0,
                threading.Event(),
                threading.Event(),
                [threading.Event() for _ in groups],
                [],
                [],
                len(groups),
                group_builders=[],
            )
            self._receiver.start()
            while not self._receiver.ready_event.wait(timeout=0.1):
                if not self._receiver.is_alive():
                    raise RuntimeError("Layerwise receive thread exited before binding its NPU")
        self._receiver.raise_if_failed()
        events = self._receiver.layer_load_finished_events
        for event in events:
            event.clear()

        # The first receive waits for prior writes on the model stream.
        ready = torch.npu.Event()
        ready.record()

        self._prefetch_work = groups
        self._prefetch_pending = pending
        self._prefetch_sync = {
            index
            for index, (bundle, owners) in enumerate(groups)
            if owners and (bundle not in eligible or not bundle.isdisjoint(unsafe))
        }
        self._prefetch_position = {owner: index for index, (bundle, _) in enumerate(groups) for owner in bundle}
        self._prefetch_next = 0
        first = next(index for index, (_, owners) in enumerate(groups) if owners and index not in self._prefetch_sync)
        self._prefetch_tasks = tuple(
            StoreLayerLoadTask(
                index, ready if index == first else None, self.io, sessions, bundle, self._prefetch_copy_lock
            )
            for index, (bundle, _) in enumerate(groups)
        )
        # Start the first layer and its successors. A subsequent window opens
        # only when an execution hook reaches that physical layer.
        self._submit_prefetch_window(0)

    def _submit_prefetch_window(self, current):
        def submit(index):
            # A synchronous demand load must not queue behind future layers.
            if index in self._prefetch_sync:
                return False
            bundle, owners = self._prefetch_work[index]
            if not self._prefetch_pending.isdisjoint(owners):
                self._receiver.raise_if_failed()
                self._receiver.add_request(self._prefetch_tasks[index])
                self._prefetch_events.update(dict.fromkeys(bundle, self._receiver.layer_load_finished_events[index]))
            return True

        self._prefetch_next = submit_layer_load_window(
            current, self._prefetch_depth, len(self._prefetch_work), self._prefetch_next, submit
        )

    def _drain_prefetch(self):
        # All submitted copies must stop before releasing leases or pages.
        # Unsubmitted work remains the model's responsibility.
        active = bool(self._prefetch_work)
        self._prefetch_tasks = ()
        self._prefetch_work = ()
        self._prefetch_position.clear()
        self._prefetch_sync.clear()
        self._prefetch_pending.clear()
        self._prefetch_events.clear()
        if active:
            self._receiver.drain()

    def load_layer(self, owner, *, execution=True):
        # Some recurrent layers report readiness with an unnamed save hook.
        # Bind it to the preceding execution wait, never to a layer counter.
        # KVPP's advisory waits must not change the current execution owner.
        current = self._prefetch_position.get(owner)
        if execution:
            self._executing_owner = owner
            if current is not None:
                self._submit_prefetch_window(current)
        event = self._prefetch_events.get(owner)
        if event is not None:
            while not event.wait(timeout=0.1):
                self._receiver.raise_if_failed()
                if not self._receiver.is_alive():
                    raise RuntimeError("Layerwise receive thread exited before completing the current layer")
            self._receiver.raise_if_failed()
            return
        owners = self._prefetch_work[current][0] if current is not None else self.layer_bundles.get(owner, (owner,))
        if self._prefetch_work and self._prefetch_pending.isdisjoint(owners):
            return
        # Demand loads and the background worker share the read session.
        with self._prefetch_copy_lock if self._prefetch_work else nullcontext():
            for _, session in self.loads:
                self.io.copy(session, frozenset(owners))
        if current in self._prefetch_sync:
            self._prefetch_pending.difference_update(owners)
            self._prefetch_sync.remove(current)
            if execution:
                self._submit_prefetch_window(current)

    def resolve_save_owner(self, owner):
        if not owner:
            owner = self._executing_owner
            if not owner:
                raise RuntimeError("Unnamed cache save requires a preceding layer execution wait")
        self._executing_owner = None
        return owner

    def save_layer(self, owner, extra_sessions=()):
        owner = self.resolve_save_owner(owner)
        sessions = tuple(session for _, session in self.saves) + tuple(extra_sessions)
        if not sessions:
            return owner
        if self._sender is None:
            worker = self.worker
            self._sender = KVCacheStoreLayerSendingThread(
                worker.m_store,
                worker.token_database,
                worker.block_size,
                worker.tp_rank,
                worker.tp_size,
                worker.dcp_size,
                0,
                threading.Event(),
                0,
                [],
                [],
                group_builders=[],
            )
            self._sender.start()
        self._sender.raise_if_failed()
        event = torch.npu.Event()
        event.record()
        self._sender.add_request(
            StoreLayerTask(event, self.io, sessions, frozenset(self.layer_bundles.get(owner, (owner,))))
        )
        self._sender.raise_if_failed()
        return owner

    def drain_saves(self):
        # KDA and circular pages are live views, not private snapshots. Drain
        # before the next step, publication, cancellation or allocator release.
        if self._sender is not None:
            self._sender.drain()

    def finish_step(self):
        try:
            self._drain_prefetch()
            self.drain_saves()
            for request, session in self.loads:
                self.io.finish(session)
                if session.logical_read_segments:
                    logger.info(
                        "Ascend shared GET req=%s objects=%d logical_segments=%d submitted_segments=%d",
                        request.req_id,
                        len(session.plans),
                        session.logical_read_segments,
                        session.submitted_read_segments,
                    )
                else:
                    logger.info("Ascend shared GET req=%s objects=%d", request.req_id, len(session.plans))
            if self.saves and not self.layerwise:
                event = torch.npu.Event()
                event.record()
                event.synchronize()
            for request, session in self.saves:
                if not self.layerwise:
                    self.io.copy(session)
                self.io.finish(session)
                if request.event_id is not None:
                    self.completed_events[request.event_id] = 1
                logger.info("Ascend shared PUT req=%s objects=%d", request.req_id, len(session.plans))
        finally:
            self._abort_all()

    def lookup(self, token_len, hashes):
        if not self.layouts:
            return 0
        layouts = [(layout, draft) for _, layout, draft in self.prefix_layouts()]
        alignment = math.lcm(self.worker.cache_transfer_granularity, *(layout.chunk_size for layout, _ in layouts))
        upper = token_len // alignment * alignment

        def readable(layout, draft, end):
            hash_index = end // layout.hash_block_size - 1
            if hash_index >= len(hashes):
                return False
            digest = draft_block_digest(hashes, hash_index) if draft else block_hash_to_str(hashes[hash_index])
            if digest is None:
                return False
            keys = [layout.key(shard, digest) for shard in range(layout.store_tp)]
            return all(self.io.batch_is_exist(keys))

        attention = [(layout, draft) for layout, draft in layouts if not layout.state]
        checkpoints = [(layout, draft) for layout, draft in layouts if layout.state]
        if self._lookup_executor is None or len(attention) < 2:
            for layout, draft in attention:
                for end in range(layout.chunk_size, upper + 1, layout.chunk_size):
                    if not readable(layout, draft, end):
                        upper = (end - 1) // alignment * alignment
                        break
        else:
            bound_lock = threading.Lock()

            def scan_attention(layout, draft):
                nonlocal upper
                # A miss in either attention group bounds the usable prefix.
                for end in range(layout.chunk_size, token_len + 1, layout.chunk_size):
                    with bound_lock:
                        if end > upper:
                            break
                    if not readable(layout, draft, end):
                        with bound_lock:
                            upper = min(upper, (end - 1) // alignment * alignment)
                        break

            futures = [self._lookup_executor.submit(scan_attention, layout, draft) for layout, draft in attention]
            wait(futures)
            for future in futures:
                future.result()

        # Recurrent checkpoints are sparse and only the chosen boundary will
        # be loaded. Probe candidates newest first without scanning history.
        for end in range(upper, 0, -alignment):
            if all(readable(layout, draft, end) for layout, draft in checkpoints):
                return end
        return 0

    def _abort_all(self):
        errors = []
        try:
            self._drain_prefetch()
        except Exception as error:
            errors.append(error)
        try:
            self.drain_saves()
        except Exception as error:
            errors.append(error)
        for _, session in self.loads + self.saves:
            try:
                self.io.abort(session)
            except Exception as error:
                errors.append(error)
        self.loads.clear()
        self.saves.clear()
        self.step = None
        self._executing_owner = None
        if errors:
            raise RuntimeError("Shared pool lease cleanup failed") from errors[0]

    def abort_step(self):
        self._abort_all()

    def close(self):
        if self._lookup_executor is not None:
            self._lookup_executor.shutdown()
        errors = []
        try:
            self._abort_all()
        except Exception as error:
            errors.append(error)
        finally:
            if self._sender is not None:
                self._sender.stop()
                self._sender.join()
            if self._receiver is not None:
                self._receiver.stop()
                self._receiver.join()
        try:
            self.io.retry_cleanup()
            if self.regions:
                self.worker.m_store.unregister_buffer(list(self.regions), list(self.regions.values()))
                self.regions.clear()
        except Exception as error:
            errors.append(error)
        if errors:
            raise RuntimeError("Shared pool shutdown failed") from errors[0]
