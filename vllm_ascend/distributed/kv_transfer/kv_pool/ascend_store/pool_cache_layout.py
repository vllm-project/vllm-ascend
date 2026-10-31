# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Address/layout planning shared by ordinary pooling and pooled PD.

No store clients, queues, leases or request ownership live in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import (
        PoolObjectRef,
        PoolSaveRequest,
        PoolSourceProfile,
    )

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec, is_circular_kv_cache_spec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_layout import (
    MLA_PAGE_FORMAT_PREFIX,
    bind_token_page,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_physical_layer_index,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_writer import LayerWriteRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.prefix_object import (
    component_prefix_id,
    full_prefix_objects,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import TargetCacheSlice
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.state_snapshot import (
    bind_circular_snapshot,
    bind_mamba_checkpoint,
)
from vllm_ascend.distributed.kv_transfer.utils.cache_layout import kernel_cache_pages, kv_head_interval


@dataclass
class PoolCacheLayout:
    cache_config: Any
    bindings: dict
    specs: dict
    tp_rank: int
    tp_size: int
    dcp_rank: int
    dcp_size: int
    pcp_size: int
    pcp_rank: int
    total_num_kv_heads: int | None
    pp_rank: int
    base_layers: int
    pool_head_rank: int
    pool_layer_offset: int
    data_plane: str
    worker_id: str = ""
    _caches_registered: bool = True
    include_window_history: bool = False

    @classmethod
    def from_worker(cls, worker):
        fields = {
            name: getattr(worker, name)
            for name in cls.__dataclass_fields__
            if name not in ("data_plane", "include_window_history")
        }
        return cls(**fields, data_plane=worker.backend.layerwise_data_plane)

    def prefix_objects(self, request, profile, group_ids, *, history=False):
        """Reconstruct immutable keys/layouts without allocating model blocks.

        Virtual bindings retain real byte geometry but describe enough pages
        for this lookup. Only address-free object refs escape this method;
        its synthetic write ranges are never submitted to a store or device.
        """
        tables = []
        capacity = 2
        for group in self.cache_config.kv_cache_groups:
            specs = [self.specs[name] for name in group.layer_names]
            singleton = any(
                is_circular_kv_cache_spec(spec) or isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "none"
                for spec in specs
            )
            count = 1 if singleton else math.ceil(request.token_end / group.kv_cache_spec.block_size)
            tables.append(tuple(range(1, count + 1)))
            capacity = max(capacity, count + 1)
        bindings = {
            name: tuple(
                replace(part, shape=(capacity * part.block_size_scale, *part.shape[1:]), scheduler_blocks=capacity)
                for part in parts
            )
            for name, parts in self.bindings.items()
            if parts[0].group_id in group_ids
        }
        layout = replace(self, bindings=bindings, include_window_history=history)
        _, _, refs = layout.build_source_plan(replace(request, block_ids=tuple(tables)), profile)
        return refs

    def build_target_slices(
        self,
        block_ids: tuple[list[int], ...],
        token_start: int,
        token_end: int,
        *,
        group_token_starts: dict[int, int] | None = None,
    ) -> list[TargetCacheSlice]:
        if not self._caches_registered:
            raise RuntimeError("Pool cache buffers have not been registered")
        if len(block_ids) != len(self.cache_config.kv_cache_groups):
            raise ValueError("Destination block tables do not match local cache groups")
        targets = []
        for layer, components in self.bindings.items():
            spec = self.specs[layer]
            if isinstance(spec, MambaSpec):
                targets.extend(
                    bind_mamba_checkpoint(
                        components,
                        spec,
                        block_ids[components[0].group_id],
                        token_end,
                        rank=self.tp_rank * self.pcp_size + self.pcp_rank,
                        size=self.tp_size * self.pcp_size,
                    )
                )
                continue
            if is_circular_kv_cache_spec(spec):
                for component in components:
                    targets.append(bind_circular_snapshot(component, block_ids[component.group_id], token_end))
                continue
            mla = isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))
            replicated = isinstance(spec, (SlidingWindowSpec, SlidingWindowMLASpec))
            indexer_replicas = getattr(spec, "sfa_dcp_replicated_indexer_size", 1)
            if isinstance(spec, AscendSFAIndexerCacheSpec) and indexer_replicas > 1:
                replicated = True
            effective_dcp = 1 if replicated else self.dcp_size
            rank = 0 if replicated else self.dcp_rank
            if mla:
                head_start = 0
            elif isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                # Effective-DCP ranks share a head shard. PCP-aware group
                # placement is supplied by the connector's topology setup.
                dcp_tp_size = 1 if effective_dcp == 1 else effective_dcp // self.pcp_size
                head_tp_size = self.tp_size // dcp_tp_size
                total_heads = self.total_num_kv_heads or spec.num_kv_heads * head_tp_size
                head_start, head_end = kv_head_interval(self.tp_rank // dcp_tp_size, head_tp_size, total_heads)
                if head_end - head_start != spec.num_kv_heads:
                    raise ValueError("Local cache head count disagrees with its global head topology")
            else:
                raise ValueError(f"Pool state snapshot adapter required for {type(spec).__name__}")
            compression = getattr(spec, "tokens_per_state", 1)
            layer_start = max(token_start, (group_token_starts or {}).get(components[0].group_id, token_start))
            if isinstance(spec, SlidingWindowSpec) and not mla and not self.include_window_history:
                layer_start = max(layer_start, token_end - spec.sliding_window)
            logical_size = spec.block_size * indexer_replicas
            for component in components:
                pages = kernel_cache_pages(
                    block_ids[component.group_id],
                    logical_block_size=logical_size,
                    physical_scale=component.block_size_scale,
                    dcp_size=effective_dcp,
                    dcp_rank=rank,
                )
                for page in pages:
                    start, end = max(page.token_start, layer_start), min(page.token_end, token_end)
                    if start >= end:
                        continue
                    target = bind_token_page(
                        component,
                        spec,
                        page.physical_block_id,
                        token_start=page.token_start // compression,
                        head_start=head_start,
                        storage_rows=(page.token_end - page.token_start) // compression,
                    )
                    if target.cache.format_id.startswith(MLA_PAGE_FORMAT_PREFIX):
                        if start != page.token_start:
                            raise ValueError("Packed MLA restore cannot overwrite a partial local prefix page")
                        targets.append(target)
                        continue
                    row_start, row_end = (
                        max(target.cache.token_start, start // compression),
                        min(target.cache.token_end, end // compression),
                    )
                    if row_start < row_end:
                        address = target.address + (row_start - target.cache.token_start) * target.cache.token_stride
                        targets.append(
                            TargetCacheSlice(
                                replace(target.cache, token_start=row_start, token_end=row_end),
                                address,
                            )
                        )
        return targets

    def build_source_plan(
        self,
        request: PoolSaveRequest,
        profile: PoolSourceProfile,
        *,
        allow_empty_chunk: bool = False,
    ) -> tuple[dict[str, int], dict[str, tuple[LayerWriteRange, ...]], tuple[PoolObjectRef, ...]]:
        """Build shared full-prefix/component objects and backend-private objects.

        Objects carry explicit valid component slices, so unused/padded bytes
        in a physical page never become additional logical cache coverage.
        Ordinary component-major layouts retain their byte offsets and qualify
        the content identity with the actual format. GVA component/state/tail
        identities include their valid logical range and complete input digest
        when no complete prefix hash is available. Non-GVA range sessions keep
        exclusive attempt identities until backend ownership supports sharing.
        """
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import PoolObjectRef, PoolObjectSlice

        tables = tuple(list(group) for group in request.block_ids)
        # Revisit the unfinished page of each group, not the entire history.
        # A later chunk can complete a page whose first tokens were computed
        # in the preceding chunk; complete older pages are already in the pool.
        starts = {}
        if request.chunk_start:
            for group_id, group in enumerate(self.cache_config.kv_cache_groups):
                block_size = (
                    profile.group_block_sizes[group_id]
                    if profile.group_block_sizes
                    else group.kv_cache_spec.block_size * self.dcp_size
                )
                starts[group_id] = request.chunk_start // block_size * block_size
        pages = self.build_target_slices(tables, request.token_start, request.token_end, group_token_starts=starts)
        # Align mode retains independently addressable block-boundary states.
        # Publish those checkpoints for future prefix hits, in addition to the
        # final PD checkpoint. Null/evicted states cannot be advertised.
        for layer, components in self.bindings.items():
            spec = self.specs[layer]
            if not isinstance(spec, MambaSpec) or spec.mamba_cache_mode != "align":
                continue
            blocks = tables[components[0].group_id]
            first = (max(request.token_start, request.chunk_start) // spec.block_size + 1) * spec.block_size
            for boundary in range(first, request.token_end, spec.block_size):
                index = boundary // spec.block_size - 1
                if index < len(blocks) and blocks[index] > 0:
                    pages.extend(
                        bind_mamba_checkpoint(
                            components,
                            spec,
                            blocks,
                            boundary,
                            rank=self.tp_rank * self.pcp_size + self.pcp_rank,
                            size=self.tp_size * self.pcp_size,
                        )
                    )
        block_indices = [{block: index for index, block in enumerate(group)} for group in request.block_ids]
        keys = profile.key_identity.bind()
        objects: dict[str, int] = {}
        layers: dict[str, list[LayerWriteRange]] = {}
        refs = []
        covered = set()
        if self.data_plane == "gva" and self.pcp_size == self.dcp_size == 1:
            objects, layers, refs, covered = full_prefix_objects(
                self.bindings,
                pages,
                request,
                profile,
                rank=self.pool_head_rank,
                stage=self.pp_rank,
                group_block_sizes=[group.kv_cache_spec.block_size for group in self.cache_config.kv_cache_groups],
                base_layers=self.base_layers,
                global_layers=max(
                    (get_physical_layer_index(name, self.base_layers) + 1 for name in self.specs), default=0
                ),
                layer_offset=self.pool_layer_offset,
                state_rank=self.tp_rank,
                schema_qualified=True,
                excluded_groups=frozenset(
                    components[0].group_id
                    for layer, components in self.bindings.items()
                    if is_circular_kv_cache_spec(self.specs[layer])
                    or (isinstance(self.specs[layer], MambaSpec) and self.specs[layer].mamba_cache_mode != "align")
                ),
            )
        for page_index, page in enumerate(pages):
            if page_index in covered:
                continue
            cache = page.cache
            binding = self.bindings[cache.layer][int(cache.component.split("/", 1)[0])]
            physical_block = (page.address - binding.address) // binding.byte_strides[0]
            block_id = physical_block // binding.block_size_scale
            block_index = block_indices[binding.group_id][block_id]
            hashes = request.group_hashes[binding.group_id]
            block_hash = (
                hashes[block_index]
                if block_index < len(hashes)
                else (f"pd-tail-v1:{request.transfer_id}:{request.token_end}")
            )
            # Writer ownership cannot rely on an existence check followed by
            # allocation: another producer could race and overwrite a live key.
            # Include server incarnation and attempt until alias publication
            # has established the content-addressed object atomically.
            identity = (
                profile.engine_incarnation,
                request.transfer_id,
                request.attempt_id,
                self.worker_id,
                binding.group_id,
                block_hash,
                cache.layer,
                cache.component,
                cache.token_start,
                cache.token_end,
                cache.head_start,
                cache.head_end,
            )
            object_id = "pool-pd-v1:" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()
            if self.data_plane == "gva":
                object_id = component_prefix_id(
                    cache,
                    binding,
                    self.specs[cache.layer],
                    request,
                    group_block_size=self.cache_config.kv_cache_groups[binding.group_id].kv_cache_spec.block_size
                    * self.dcp_size,
                )
            key_rank = self.tp_rank if isinstance(self.specs[cache.layer], MambaSpec) else self.pool_head_rank
            key = keys.make_full_key(binding.group_id, object_id, key_rank, self.pp_rank)
            size = cache.span_bytes
            if key in objects:
                raise ValueError("Source plan generated a duplicate private object key")
            objects[key] = size
            layers.setdefault(cache.layer, []).append(LayerWriteRange(key, page.address, 0, size))
            refs.append(
                PoolObjectRef(
                    profile.profile_id,
                    binding.group_id,
                    object_id,
                    key_rank,
                    self.pp_rank,
                    size,
                    (PoolObjectSlice(cache, 0),),
                )
            )
        if not objects and self.bindings and not (allow_empty_chunk and not pages):
            raise ValueError("Source save request has no materialized cache ranges")
        return objects, {layer: tuple(ranges) for layer, ranges in layers.items()}, tuple(refs)
