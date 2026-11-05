# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend payload bindings for the upstream Store-shard addressing rules.

The local cache is never repacked. Attention values use layer/component/head/
token/content order, with K and V represented as separate components. Object
identity depends on the Store schema, not local TP, addresses or execution mode.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout import (
    HeadMajorStoreLayout,
    MambaStoreLayout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout import (
    KeyMetadata as UpstreamKeyMetadata,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, MLAAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata, PoolKey


@dataclass(frozen=True)
class PoolRange:
    owner: str
    address: int
    offset: int
    size: int


@dataclass(frozen=True)
class PoolObject:
    key: str
    size: int
    block_id: int
    ranges: tuple[PoolRange, ...]


class RequestStateLayout:
    """Replicated request-lifetime rings, outside ordinary prefix lookup.

    Like Mooncake V2's circular transfer, copy complete component content,
    validate physical shapes and skip allocator padding. The prompt boundary
    determines ring position; no tensor rotation or model change is required.
    """

    def __init__(self, metadata, specs, caches, num_blocks, cache_prefix, writer_rank):
        from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec

        if not specs or num_blocks <= 0 or not cache_prefix:
            raise ValueError("Request state requires owners, blocks and a model namespace")
        self.specs = specs
        self.writer_rank = writer_rank
        self.replicated = True
        self.num_blocks = num_blocks
        self._refs = []
        self._components = []
        schema = []
        for owner, spec in specs.items():
            if not is_circular_kv_cache_spec(spec):
                raise ValueError("Request state layout only accepts circular cache specs")
            descriptors = []
            for cache in caches[owner]:
                if cache.ndim < 2 or cache.shape[0] != num_blocks or not cache[0].is_contiguous():
                    raise ValueError("Circular cache requires contiguous content inside each physical page")
                size = cache[0].numel() * cache.element_size()
                stride = cache.stride(0) * cache.element_size()
                if size <= 0 or stride < size:
                    raise ValueError("Circular cache page content is empty or overlapping")
                self._components.append((owner, cache.data_ptr(), stride, size))
                self._refs.append(cache)
                descriptors.append((tuple(cache.shape[1:]), str(cache.dtype)))
            if not descriptors:
                raise ValueError("Circular owner has no registered state")
            schema.append(
                (owner, type(spec).__name__, spec.block_size, getattr(spec, "compress_ratio", None), tuple(descriptors))
            )
        fingerprint = hashlib.sha256(repr((cache_prefix, tuple(schema))).encode()).hexdigest()
        self.metadata = replace(
            metadata,
            head_or_tp_rank=0,
            kv_cache_group_id=0,
            cache_family="shared_v1_ring_" + fingerprint,
            cache_role="state",
        )

    def snapshot(self, block_id, digest, cache_tokens):
        if not 0 < block_id < self.num_blocks or cache_tokens <= 0:
            raise ValueError("Circular snapshot requires an allocated block and computed prefix")
        ranges, offset = [], 0
        for owner, base, stride, size in self._components:
            ranges.append(PoolRange(owner, base + block_id * stride, offset, size))
            offset += size
        key = PoolKey(self.metadata, f"pd_state_{cache_tokens}_{digest}").to_string()
        return (PoolObject(key, offset, block_id, tuple(ranges)),)


def _hnd_components(spec, caches, num_blocks):
    if len(caches) != 2:
        raise ValueError("Shared HND requires separate K and V tensors")
    components = []
    for cache, width in zip(caches, (spec.head_size, spec.head_size_v), strict=True):
        if (
            cache.ndim != 4
            or cache.shape[0] % num_blocks
            or cache.shape[1] != spec.num_heads
            or cache.shape[3] != width
            or cache.dtype != spec.dtype
            or cache.stride(3) != 1
            or cache.stride(2) != width
            or cache.stride(1) < cache.shape[2] * width
            or cache.stride(0) < cache.shape[1] * cache.stride(1)
        ):
            raise ValueError("Shared attention requires non-overlapping Ascend HND K/V views")
        kernels = cache.shape[0] // num_blocks
        if kernels * cache.shape[2] != spec.block_size:
            raise ValueError("HND kernel pages do not cover one scheduler block")
        view = cache.as_strided(
            (num_blocks, kernels, *cache.shape[1:]),
            (cache.stride(0) * kernels, *cache.stride()),
        )
        component_spec = replace(spec, state_content_bytes=width * cache.element_size())
        components.append((component_spec, view))
    return components


def _packed_state(spec, caches, num_blocks):
    # Extra replay/correction/ring states need an explicit persistence contract.
    if len(spec.shapes) != 2 or len(caches) != 2:
        raise ValueError("Shared state currently describes only committed conv/recurrent checkpoints")
    first = caches[0]
    storage = first.untyped_storage()
    stride = first.stride(0) * first.element_size()
    offset = 0
    for cache, shape, dtype in zip(caches, spec.shapes, spec.dtypes, strict=True):
        if (
            tuple(cache.shape) != (num_blocks, *shape)
            or cache.dtype != dtype
            or cache.untyped_storage().data_ptr() != storage.data_ptr()
            or cache.data_ptr() != first.data_ptr() + offset
            or cache.stride(0) * cache.element_size() != stride
            or not cache[0].is_contiguous()
        ):
            raise ValueError("State binding requires packed components with an explicit common page stride")
        offset += cache[0].numel() * cache.element_size()
    if offset != spec.state_content_size_bytes or stride < offset:
        raise ValueError("State content and page stride disagree")
    return torch.empty(0, dtype=torch.uint8, device=first.device).set_(
        storage, first.data_ptr() - storage.data_ptr(), (num_blocks, 1, 1, offset), (stride, offset, offset, 1)
    )


def _mla_components(spec, caches, num_blocks):
    """Bind Ascend's replicated N1D components without packing or dtype casts.

    Main latent/rope, packed main cache and indexer/scale all use the same
    descriptor. Their actual registered views define component widths/dtypes.
    Compression is expressed by tokens_per_state and mapped by StoreLayout.
    Incomplete compression groups belong to the separate circular tail state.
    """
    if spec.num_kv_heads != 1:
        raise ValueError("Replicated MLA requires one KV head")
    tokens_per_state = spec.tokens_per_state
    if tokens_per_state < 1 or int(tokens_per_state) != tokens_per_state:
        raise ValueError("MLA requires a positive integral tokens_per_state")
    tokens_per_state = int(tokens_per_state)
    if getattr(spec, "sfa_dcp_replicated_indexer_size", 1) != 1:
        raise ValueError("CP indexer replication needs an explicit token mapping")
    components = []
    for cache in caches:
        # Ascend exposes an empty rope view alongside packed latent content.
        # It owns no bytes and must not become an address descriptor.
        if cache.ndim == 4 and cache.shape[3] == 0:
            continue
        if (
            cache.ndim != 4
            or cache.shape[0] % num_blocks
            or cache.shape[2] != 1
            or cache.shape[3] <= 0
            or cache.stride(3) != 1
            or cache.stride(1) != cache.shape[3]
            or cache.stride(0) < cache.shape[1] * cache.shape[3]
        ):
            raise ValueError("MLA requires non-overlapping (page, token, 1, content) views")
        kernels = cache.shape[0] // num_blocks
        if kernels * cache.shape[1] * tokens_per_state != spec.block_size:
            raise ValueError("MLA kernel pages do not cover one scheduler block")
        width = cache.shape[3]
        component = FullAttentionSpec(
            block_size=spec.block_size,
            num_kv_heads=1,
            head_size=width,
            dtype=cache.dtype,
            state_content_bytes=width * cache.element_size(),
            tokens_per_state=tokens_per_state,
        )
        view = cache.as_strided(
            (num_blocks, kernels, 1, cache.shape[1], width),
            (kernels * cache.stride(0), cache.stride(0), cache.stride(2), cache.stride(1), 1),
        )
        components.append((component, view))
    if not components:
        raise ValueError("MLA cache has no persistent components")
    return components


class SharedPoolLayout:
    """One group's schema and public upstream address descriptors.

    HND components are flattened in owner/K/V order. State schemas use the
    upstream Mamba component split. Both share the same Store-shard mapping.
    The caller owns tensor readiness, checkpoint selection and buffer lifetime.
    """

    def __init__(
        self,
        metadata: KeyMetadata,
        specs: Mapping[str, FullAttentionSpec | MambaSpec],
        *,
        local_tp: int,
        store_tp: int,
        rank: int,
        chunk_size: int,
        hash_block_size: int,
        cache_prefix: str,
    ):
        if not specs or local_tp <= 0 or store_tp < local_tp or store_tp % local_tp or not 0 <= rank < local_tp:
            raise ValueError("Store TP must be a positive multiple of local TP, with a valid local rank")
        if not cache_prefix:
            raise ValueError("Shared pool requires an explicit cache_prefix identifying model weights/epoch")
        if chunk_size <= 0 or hash_block_size <= 0 or chunk_size % hash_block_size:
            raise ValueError("Store chunk must align with prefix hashes")
        self.specs = dict(specs)
        self.chunk_size = chunk_size
        # Group numbers may move when a local draft cache is added. The schema
        # already contains the ordered target owners; allocator group IDs must
        # not be part of persistent object identity.
        self.metadata = replace(metadata, kv_cache_group_id=0)
        self.local_tp, self.store_tp, self.rank = local_tp, store_tp, rank
        self.writer_rank = rank
        self.replicated = all(isinstance(spec, MLAAttentionSpec) for spec in specs.values())
        if self.replicated:
            self.local_tp, self.store_tp, self.rank = 1, 1, 0
        self.hash_block_size = hash_block_size
        self.cache_prefix = cache_prefix
        blocks = {spec.block_size for spec in specs.values()}
        if len(blocks) != 1 or next(iter(blocks)) % chunk_size:
            raise ValueError("Each group needs a common block size divisible by its Store chunk")
        self.block_size = next(iter(blocks))
        self.state = all(isinstance(spec, MambaSpec) for spec in specs.values())
        if self.state:
            if chunk_size != self.block_size:
                raise ValueError("Recurrent checkpoints cannot be sliced along tokens")
            layout_cls = MambaStoreLayout
            layout_specs = tuple(specs.values())
        elif self.replicated:
            # Actual Ascend views carry split and mixed-dtype component sizes.
            self.layout_cls = HeadMajorStoreLayout
            self.layer_layouts = {}
            self.component_views = {}
            self.num_blocks = 0
            self._refs = []
            return
        elif all(type(spec) is FullAttentionSpec for spec in specs.values()):
            layout_cls = HeadMajorStoreLayout
            # This schema includes both components, even before tensor registration.
            layout_specs = tuple(
                replace(spec, state_content_bytes=width * torch.empty((), dtype=spec.dtype).element_size())
                for spec in specs.values()
                for width in (spec.head_size, spec.head_size_v)
            )
        else:
            raise ValueError("Unsupported shared cache family; use an explicit component adapter")
        self.layout_cls = layout_cls
        self._initialize_layout(layout_specs)
        self.layer_layouts = {}
        self.component_views = {}
        self.num_blocks = 0
        self._refs = []

    def _initialize_layout(self, layout_specs):
        args = (layout_specs, self.local_tp, self.store_tp)
        fingerprint = (
            self.layout_cls.schema_fingerprint(*args)
            if self.state
            else self.layout_cls.schema_fingerprint(*args, self.chunk_size)
        )
        schema = (
            "ascend_shared_v1",
            tuple(self.specs),
            fingerprint,
            self.chunk_size,
            self.hash_block_size,
            self.store_tp,
            self.cache_prefix,
        )
        if self.replicated:
            # Equal widths do not imply equal quantization semantics (e.g.
            # packed C8 versus another packed encoding). Preserve the source
            # spec contract as well as the registered component byte format.
            fields = (
                "cache_dtype_str",
                "head_size",
                "scale_dim",
                "scale_dtype",
                "cache_sparse_sfa_c8",
                "cache_sparse_sfa_turboquant",
                "cache_sparse_li_c8",
                "cache_sparse_li_c4",
                "model_version",
                "tokens_per_state",
            )
            schema += (
                tuple(
                    (type(spec).__name__, tuple(getattr(spec, field, None) for field in fields))
                    for spec in self.specs.values()
                ),
            )
        self.schema = hashlib.sha256(repr(schema).encode()).hexdigest()
        self.family = f"shared_v1_{'state' if self.state else 'hnd_kv'}_{self.schema}"
        self.layout = self._make_layout(layout_specs)

    def _make_layout(self, specs):
        kwargs = dict(
            metadata=UpstreamKeyMetadata(self.metadata.model_name, self.rank, 0, 0, 0),
            block_size=self.block_size,
            hash_block_size=self.hash_block_size,
            local_tp_size=self.local_tp,
            store_tp_size=self.store_tp,
            tp_rank=self.rank,
            layer_specs=specs,
        )
        if not self.state:
            kwargs["store_chunk_size"] = self.chunk_size
        return self.layout_cls(**kwargs)

    @property
    def local_shards(self):
        return self.layout.local_shard_ids

    def key(self, shard: int, digest: str) -> str:
        if not 0 <= shard < self.store_tp:
            raise ValueError("Invalid Store shard")
        return PoolKey(
            replace(
                self.metadata,
                head_or_tp_rank=shard,
                cache_family=self.family,
                cache_role="state" if self.state else "kv",
            ),
            digest,
        ).to_string()

    def register(self, caches: Mapping[str, Sequence[torch.Tensor]], num_blocks: int):
        if num_blocks <= 0 or self.num_blocks:
            raise ValueError("Register a positive block count exactly once")
        views = []
        for name, spec in self.specs.items():
            tensors = tuple(caches[name])
            parts = (
                [(spec, _packed_state(spec, tensors, num_blocks))]
                if self.state
                else (
                    _mla_components(spec, tensors, num_blocks)
                    if self.replicated
                    else _hnd_components(spec, tensors, num_blocks)
                )
            )
            layout = self._make_layout(tuple(item[0] for item in parts))
            layer_views = [item[1] for item in parts]
            layout.register_kv_caches(layer_views, num_blocks)
            self.layer_layouts[name] = layout
            self.component_views[name] = parts
            views.extend(layer_views)
            self._refs.extend(tensors)
        if self.replicated:
            self._initialize_layout(tuple(spec for parts in self.component_views.values() for spec, _ in parts))
        self.layout.register_kv_caches(views, num_blocks)
        self.num_blocks = num_blocks

    def objects(self, start: int, block_id: int, digest: str) -> tuple[PoolObject, ...]:
        if not 0 <= block_id < self.num_blocks or start < 0 or start % self.chunk_size:
            raise ValueError("Invalid local block or Store chunk boundary")
        offset = start % self.block_size
        chunks = [(offset, offset + self.chunk_size)] * len(self.local_shards)
        addresses, sizes, _ = self.layout.prepare_values(chunks, [block_id], self.local_shards)
        owners = [
            (name, layout.prepare_values(chunks, [block_id], self.local_shards))
            for name, layout in self.layer_layouts.items()
        ]
        result = []
        for index, shard in enumerate(self.local_shards):
            ranges, actual = [], []
            cursor = 0
            for name, (layer_addrs, layer_sizes, _) in owners:
                for address, size in zip(layer_addrs[index], layer_sizes[index], strict=True):
                    ranges.append(PoolRange(name, int(address), cursor, int(size)))
                    actual.append((address, size))
                    cursor += int(size)
            if actual != list(zip(addresses[index], sizes[index], strict=True)):
                raise ValueError("Layer ranges disagree with the group's object byte order")
            result.append(PoolObject(self.key(shard, digest), cursor, block_id, tuple(ranges)))
        return tuple(result)

    def partial_objects(self, start: int, block_id: int, digest: str, valid_tokens: int) -> tuple[PoolObject, ...]:
        """PD-only compact tail, preserving the full object's component order.

        Upstream fixed-chunk templates include invalid tail bytes. Slice each
        validated head/token view instead; shard ownership stays upstream's.
        No ordinary-prefix lookup ever requests this key namespace.
        """
        if self.state or not 0 < valid_tokens < self.chunk_size:
            raise ValueError("Partial objects require an incomplete attention chunk")
        if not 0 <= block_id < self.num_blocks or start < 0 or start % self.chunk_size:
            raise ValueError("Invalid partial object boundary")
        result = []
        for shard_index, shard in enumerate(self.local_shards):
            ranges, cursor = [], 0
            for owner, parts in self.component_views.items():
                for component, view in parts:
                    heads = view.shape[2] // len(self.local_shards)
                    for head in range(shard_index * heads, (shard_index + 1) * heads):
                        # Only complete compressed rows are persistent here.
                        # Residual tokens are restored from the PD tail state.
                        ratio = component.tokens_per_state
                        token = int((start % self.block_size) // ratio)
                        stop = int((start % self.block_size + valid_tokens) // ratio)
                        while token < stop:
                            kernel, offset = divmod(token, view.shape[3])
                            count = min(stop - token, view.shape[3] - offset)
                            address = view[block_id, kernel, head, offset].data_ptr()
                            size = count * view.shape[4] * view.element_size()
                            ranges.append(PoolRange(owner, address, cursor, size))
                            cursor += size
                            token += count
            key = self.key(shard, f"partial_{valid_tokens}_{digest}")
            result.append(PoolObject(key, cursor, block_id, tuple(ranges)))
        return tuple(result)
