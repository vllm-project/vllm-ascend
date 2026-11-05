# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronous execution of shared layouts inside the existing pool worker.

This owns no scheduler, model state, or RPC server. The existing connector
supplies step metadata and layer readiness; the existing backend owns transport.
"""

from __future__ import annotations

import math

import torch
from vllm.logger import logger
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata, block_hash_to_str
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_io import MemcacheStoreLayoutIO
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_layout import RequestStateLayout, SharedPoolLayout


def target_group_specs(group, draft_names=()):
    """Use runner-provided ownership; draft caches never enter shared objects."""
    spec = group.kv_cache_spec
    specs = {
        name: spec.kv_cache_specs[name] if isinstance(spec, UniformTypeKVCacheSpecs) else spec
        for name in group.layer_names
    }
    return {
        name: item for name, item in specs.items() if not getattr(item, "is_draft", False) and name not in draft_names
    }


def shared_draft_names(cache_config):
    """Resolve auxiliary indexer ownership from its declared main cache.

    Ascend MRV2 builds indexer specs before tagging ordinary attention specs
    as draft. Use the same main/indexer name relation as that runner instead
    of assuming the indexer flag is populated, or guessing from layer numbers.
    """
    from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendSFAIndexerCacheSpec

    all_specs = {}
    for group in cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        all_specs.update(
            {
                name: spec.kv_cache_specs[name] if isinstance(spec, UniformTypeKVCacheSpecs) else spec
                for name in group.layer_names
            }
        )
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
    if not config.use_v2_model_runner or not config.model_config.enforce_eager:
        raise ValueError("Shared pool currently requires MRV2 eager")
    if extra.get("backend", "mooncake").lower() != "memcache" or extra.get("load_async", False):
        raise ValueError("Shared pool currently requires synchronous MemCache transfers")
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
                for spec in target_group_specs(group).values()
            )
        ):
            raise ValueError("Shared pool MTP requires explicit target and draft cache ownership from the runner")
    for group in cache_config.kv_cache_groups:
        if getattr(group, "host_resident", False):
            raise ValueError("Shared pool does not yet support host-resident cache groups")
        for item in target_group_specs(group, draft_names).values():
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


class SharedPoolTransfer:
    def __init__(self, worker):
        self.worker = worker
        extra = worker.vllm_config.kv_transfer_config.kv_connector_extra_config
        self.layerwise = bool(extra.get("use_layerwise", False))
        self.io = MemcacheStoreLayoutIO(worker.m_store, int(extra.get("store_lease_ttl_ms", 600000)))
        self.layouts = {}
        self.request_states = {}
        self.regions = {}
        self.saves = []
        self.loads = []
        self.step = None
        self.completed_events = {}
        self.layer_bundles = {}
        self._executing_owner = None
        self._registered = False

    def register(self, caches):
        worker = self.worker
        config = worker.kv_cache_config
        extra = worker.vllm_config.kv_transfer_config.kv_connector_extra_config
        draft_names = shared_draft_names(config)
        for gid, group in enumerate(config.kv_cache_groups):
            spec = group.kv_cache_spec
            specs = target_group_specs(group, draft_names)
            if not specs:
                continue
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
                self.request_states[gid] = RequestStateLayout(
                    identity, specs, tensors, config.num_blocks, extra["cache_prefix"], worker.tp_rank
                )
            else:
                layout = SharedPoolLayout(
                    identity,
                    specs,
                    local_tp=worker.tp_size,
                    store_tp=extra["store_tp_size"],
                    rank=worker.tp_rank,
                    chunk_size=spec.block_size,
                    hash_block_size=worker.hash_block_size,
                    cache_prefix=extra["cache_prefix"],
                )
                layout.register(tensors, config.num_blocks)
                self.layouts[gid] = layout
            for entries in tensors.values():
                for tensor in entries:
                    storage = tensor.untyped_storage()
                    self.regions[storage.data_ptr()] = storage.nbytes()
        worker.m_store.register_buffer(list(self.regions), list(self.regions.values()))
        if self.request_states or any(layout.replicated for layout in self.layouts.values()):
            from vllm_ascend.core.kv_cache_placement import build_layer_cache_bundles

            specs = {
                name: spec
                for layout in (*self.layouts.values(), *self.request_states.values())
                for name, spec in layout.specs.items()
            }
            self.layer_bundles = build_layer_cache_bundles(specs)
        self._registered = True

    def describe_groups(self):
        if not self._registered:
            raise RuntimeError("Shared pool caches have not been registered")
        return {
            "version": 1,
            "group_count": len(self.worker.kv_cache_config.kv_cache_groups),
            "target_groups": sorted(set(self.layouts) | set(self.request_states)),
            "prefix_groups": sorted(self.layouts),
            "stateful": bool(self.request_states)
            or any(
                isinstance(spec, MambaSpec) or getattr(spec, "tokens_per_state", 1) != 1
                for layout in self.layouts.values()
                for spec in layout.specs.values()
            ),
        }

    def _plans(self, request, *, writing):
        result = []
        load = request.load_spec
        for gid, layout in self.layouts.items():
            if writing and layout.replicated and layout.writer_rank != 0:
                continue
            length = request.save_end_token if writing else load.kvpool_cached_tokens
            start = request.save_start_token if writing else load.vllm_cached_tokens
            # A last-token recompute still requires restoring its complete
            # immutable source block. Local APC blocks preceding it are skipped.
            first = math.ceil(start / layout.chunk_size) if writing else start // layout.chunk_size
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
                digest = block_hash_to_str(request.block_hashes[hash_index])
                result.extend(layout.objects(index * layout.chunk_size, blocks[index], digest))
        return tuple(result)

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
                    plans = tuple(plan for plan in self._plans(request, writing=True) if plan.key not in seen)
                    seen.update(plan.key for plan in plans)
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
        except Exception:
            self._abort_all()
            raise

    def load_layer(self, owner, *, execution=True):
        # Some recurrent layers report readiness with an unnamed save hook.
        # Bind it to the preceding execution wait, never to a layer counter.
        # KVPP's advisory waits must not change the current execution owner.
        if execution:
            self._executing_owner = owner
        for _, session in self.loads:
            self.io.copy(session, set(self.layer_bundles.get(owner, (owner,))))

    def resolve_save_owner(self, owner):
        if not owner:
            owner = self._executing_owner
            if not owner:
                raise RuntimeError("Unnamed cache save requires a preceding layer execution wait")
        self._executing_owner = None
        return owner

    def save_layer(self, owner):
        owner = self.resolve_save_owner(owner)
        if not self.saves:
            return owner
        event = torch.npu.Event()
        event.record()
        event.synchronize()
        for _, session in self.saves:
            self.io.copy(session, set(self.layer_bundles.get(owner, (owner,))))
        return owner

    def finish_step(self):
        try:
            for request, session in self.loads:
                self.io.finish(session)
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
        common = None
        for layout in self.layouts.values():
            hits = set()
            for start in range(0, token_len - layout.chunk_size + 1, layout.chunk_size):
                end = start + layout.chunk_size
                hash_index = end // layout.hash_block_size - 1
                if hash_index >= len(hashes):
                    break
                digest = block_hash_to_str(hashes[hash_index])
                keys = [layout.key(shard, digest) for shard in range(layout.store_tp)]
                if not all(self.io.batch_is_exist(keys)):
                    if layout.state:
                        # State checkpoints can be sparse; unlike attention,
                        # they do not require every historical block to exist.
                        continue
                    break
                hits.add(end)
            common = hits if common is None else common.intersection(hits)
        alignment = self.worker.cache_transfer_granularity
        # Rounding a checkpoint down would claim a state we never restored.
        return max((end for end in common or () if end % alignment == 0), default=0)

    def _abort_all(self):
        errors = []
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
        self._abort_all()
        self.io.retry_cleanup()
        if self.regions:
            self.worker.m_store.unregister_buffer(list(self.regions), list(self.regions.values()))
            self.regions.clear()
