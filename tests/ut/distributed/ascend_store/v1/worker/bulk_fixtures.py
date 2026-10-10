"""Hybrid cache topologies with align-state and sparse original group identities."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    make_topology,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
)


def make_align_state_topology():
    base = make_topology(group_ids=(0, 1), physical_layers=(0,), tp_size=2)
    attention = KVPoolGroupTopology(
        0,
        FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=1, dtype=torch.float32),
        (KVPoolLayerTopology(0, ("layers.0.attention",)),),
        replace(base.groups[0].key_metadata, cache_family="attention"),
    )
    state = KVPoolGroupTopology(
        1,
        MambaSpec(
            block_size=8,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
        (KVPoolLayerTopology(1, ("layers.1.state",)),),
        replace(base.groups[1].key_metadata, cache_family="state"),
    )
    return replace(
        base,
        cache_transfer_granularity=8,
        hash_block_size=4,
        groups=(attention, state),
    )


def make_sparse_group_topology(*, block_size: int = 4, sliding_window: int = 8):
    base = make_topology(group_ids=(1, 3), physical_layers=(0,))
    attention = KVPoolGroupTopology(
        1,
        FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32),
        (KVPoolLayerTopology(0, ("layers.0.attention",)),),
        replace(base.groups[1].key_metadata, cache_family="attention"),
    )
    sliding = KVPoolGroupTopology(
        3,
        SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=sliding_window,
        ),
        (KVPoolLayerTopology(1, ("layers.1.swa",)),),
        replace(base.groups[3].key_metadata, cache_family="swa"),
    )
    return replace(base, groups=(attention, sliding), cache_transfer_granularity=block_size, hash_block_size=block_size)


class TensorBytesBackend(FakeBackend):
    """Copy actual CPU tensor bytes through the public Bulk Backend contract."""

    def __init__(self) -> None:
        super().__init__()
        self.objects: dict[str, bytes] = {}
        self.registered_regions: list[tuple[int, int]] = []

    def exists(self, keys):
        self.calls.append(("exists", tuple(keys)))
        return [int(key in self.objects) for key in keys]

    def register_buffer(self, addresses, sizes):
        self.registered_regions.extend(zip(addresses, sizes, strict=True))
        return super().register_buffer(addresses, sizes)

    def _assert_registered_ranges(self, addresses, sizes) -> None:
        for row_addresses, row_sizes in zip(addresses, sizes, strict=True):
            for address, size in zip(row_addresses, row_sizes, strict=True):
                assert any(
                    start <= address and address + size <= start + span for start, span in self.registered_regions
                )

    def store(self, keys, addresses, sizes):
        self._assert_registered_ranges(addresses, sizes)
        codes = self.put(keys, addresses, sizes)
        for key, row_addresses, row_sizes in zip(keys, addresses, sizes, strict=True):
            self.objects[key] = b"".join(
                ctypes.string_at(address, size) for address, size in zip(row_addresses, row_sizes, strict=True)
            )
        return codes

    def load(self, keys, addresses, sizes):
        self._assert_registered_ranges(addresses, sizes)
        codes = self.get(keys, addresses, sizes)
        for key, row_addresses, row_sizes in zip(keys, addresses, sizes, strict=True):
            payload = self.objects[key]
            assert sum(row_sizes) == len(payload)
            offset = 0
            for address, size in zip(row_addresses, row_sizes, strict=True):
                ctypes.memmove(address, payload[offset : offset + size], size)
                offset += size
        return codes


def make_multi_spec_caches():
    """Build MLA/indexer, private ring, and SWA views over padded shared pages."""
    num_blocks = 8
    block_size = 32
    ring_rows = 32
    page_padding_bytes = 16
    element_size = torch.float32.itemsize
    main_specs = {}
    window_specs = {}
    caches = {}
    state_name = "model.layers.2.compressor.state_cache"
    state_spec = CircularBufferSpec(
        block_size=ring_rows, num_kv_heads=1, head_size=1, head_size_v=0, dtype=torch.float32
    )
    for layer_id in (2, 3):
        ratio = 2 if layer_id == 2 else 1
        main_name = f"model.layers.{layer_id}.attn"
        index_name = f"model.layers.{layer_id}.indexer.k_cache"
        window_name = f"model.layers.{layer_id}.swa_cache"
        main = AscendMLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float32,
            tokens_per_state=ratio,
            model_version="deepseek_v41",
        )
        index = replace(main, head_size=2, scale_dim=1, scale_dtype=torch.float32)
        window = AscendSlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=2,
            dtype=torch.float32,
            sliding_window=block_size,
            model_version="deepseek_v41",
        )
        main_bytes = main.unpadded_page_size_bytes
        index_bytes = index.unpadded_page_size_bytes
        slot_bytes = (
            max(main_bytes + index_bytes, state_spec.page_size_bytes, window.page_size_bytes) + page_padding_bytes
        )
        page_stride = slot_bytes // element_size
        storage = torch.full((num_blocks * page_stride,), -1.0, dtype=torch.float32)
        token_rows = block_size // ratio
        caches[main_name] = torch.as_strided(storage, (num_blocks, token_rows, 4), (page_stride, 4, 1))
        caches[index_name] = torch.as_strided(
            storage, (num_blocks, token_rows, 3), (page_stride, 3, 1), storage_offset=main_bytes // element_size
        )
        caches[window_name] = torch.as_strided(storage, (num_blocks, block_size, 2), (page_stride, 2, 1))
        if layer_id == 2:
            caches[state_name] = torch.as_strided(storage, (num_blocks, ring_rows, 1), (page_stride, 1, 1))
        main_specs.update({index_name: index, main_name: main})
        window_specs[window_name] = window
    groups = [
        KVCacheGroupSpec(layer_names=list(main_specs), kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(main_specs)),
        KVCacheGroupSpec(layer_names=[state_name], kv_cache_spec=state_spec),
        KVCacheGroupSpec(
            layer_names=list(window_specs), kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(window_specs)
        ),
    ]
    cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        transfer_group_ids=(0, 1, 2),
        prefix_cache_retention_interval=None,
    )
    return cache_config, caches
