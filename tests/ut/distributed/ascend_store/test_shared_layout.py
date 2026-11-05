# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import pytest
import torch
from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_io import MemcacheStoreLayoutIO
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_layout import SharedPoolLayout

from .test_shared_io import MemoryBackend


def attention(tp, rank, *, source, block=4, chunk=2):
    heads = 8 // tp
    spec = FullAttentionSpec(block_size=block, num_kv_heads=heads, head_size=3, head_size_v=5, dtype=torch.float32)
    caches = {}
    for layer in range(2):
        tensors = []
        for component, width in enumerate((3, 5)):
            # Two kernel pages per scheduler block, including page gaps.
            physical = torch.full((6, heads, 2, width + 2), -99.0)
            # Padding is only between physical pages, not inside content.
            cache = physical.as_strided((6, heads, 2, width), (heads * 2 * (width + 2), 2 * width, width, 1))
            if source:
                for kernel in range(2):
                    for h in range(heads):
                        for t in range(2):
                            cache[2 + kernel, h, t] = (
                                torch.arange(width)
                                + 1000 * layer
                                + 100 * component
                                + 10 * (rank * heads + h)
                                + kernel * 2
                                + t
                            )
            tensors.append(cache)
        caches[f"model.layers.{layer}.attn"] = tuple(tensors)
    layout = SharedPoolLayout(
        KeyMetadata("test-model", rank, 0, 0),
        {name: spec for name in caches},
        local_tp=tp,
        store_tp=4,
        rank=rank,
        chunk_size=chunk,
        hash_block_size=2,
        cache_prefix="weights-v1",
    )
    layout.register(caches, 3)
    return layout, caches


def copy_modes(io, plans, writing, layerwise):
    session = io.begin(plans, writing=writing)
    owners = sorted({part.owner for plan in plans for part in plan.ranges})
    for layers in ({owner} for owner in owners) if layerwise else (None,):
        io.copy(session, layers)
    io.finish(session)


@pytest.mark.parametrize("writer_tp,reader_tp", [(1, 4), (2, 4), (4, 2), (4, 1)])
@pytest.mark.parametrize("write_layerwise,read_layerwise", [(False, False), (True, False), (False, True), (True, True)])
def test_hnd_shards_restore_distinct_k_v_and_kernel_pages(writer_tp, reader_tp, write_layerwise, read_layerwise):
    backend = MemoryBackend()
    for rank in range(writer_tp):
        layout, _ = attention(writer_tp, rank, source=True)
        plans = layout.objects(0, 1, "prefix-2") + layout.objects(2, 1, "prefix-4")
        copy_modes(MemcacheStoreLayoutIO(backend), plans, True, write_layerwise)
    assert len(backend.readable) == 8
    for rank in range(reader_tp):
        reader, caches = attention(reader_tp, rank, source=False)
        _, expected = attention(reader_tp, rank, source=True)
        plans = reader.objects(0, 2, "prefix-2") + reader.objects(2, 2, "prefix-4")
        assert all(plan.key in backend.readable for plan in plans)
        copy_modes(MemcacheStoreLayoutIO(backend), plans, False, read_layerwise)
        for name, tensors in caches.items():
            for actual, wanted in zip(tensors, expected[name]):
                torch.testing.assert_close(actual[4:6], wanted[2:4], rtol=0, atol=0)
                assert torch.all(actual[:4] == -99)
    assert not backend.leases


def state(tp, rank, source, orientation):
    qk, v = 8 // tp, 24 // tp
    width = 2 * qk + v
    conv_shape = (3, width) if orientation == "SD" else (width, 3)
    shape = (8 // tp, 3, 2)
    spec = MambaSpec(
        block_size=4,
        shapes=(conv_shape, shape),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    size = spec.state_content_size_bytes
    stride = size + 16
    backing = torch.full((3 * stride,), 0xA5, dtype=torch.uint8)
    conv = torch.empty(0, dtype=torch.bfloat16).set_(
        backing.untyped_storage(), 0, (3, *conv_shape), (stride // 2, conv_shape[1], 1)
    )
    recurrent = torch.empty(0, dtype=torch.float32).set_(
        backing.untyped_storage(), 3 * width * 2 // 4, (3, *shape), (stride // 4, 6, 2, 1)
    )
    if source:
        base = torch.cat(
            [
                torch.arange(rank * qk, (rank + 1) * qk),
                torch.arange(rank * qk, (rank + 1) * qk) + 32,
                torch.arange(rank * v, (rank + 1) * v) + 64,
            ]
        ).to(torch.bfloat16)
        history = base.unsqueeze(0).expand(3, -1) + torch.arange(3).unsqueeze(1)
        conv[1].copy_(history if orientation == "SD" else history.T)
        recurrent[1].copy_(torch.arange(rank * shape[0] * 6, (rank + 1) * shape[0] * 6).reshape(shape))
    layout = SharedPoolLayout(
        KeyMetadata("gdn", rank, 0, 0),
        {"model.layers.0.linear": spec},
        local_tp=tp,
        store_tp=4,
        rank=rank,
        chunk_size=4,
        hash_block_size=4,
        cache_prefix="weights-v1",
    )
    layout.register({"model.layers.0.linear": (conv, recurrent)}, 3)
    return layout, (conv, recurrent), backing


@pytest.mark.parametrize("orientation", ["SD", "DS"])
@pytest.mark.parametrize("writer_tp,reader_tp", [(2, 4), (4, 2)])
def test_gdn_uses_upstream_component_split_with_page_padding(monkeypatch, orientation, writer_tp, reader_tp):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", orientation)
    get_conv_state_layout.cache_clear()
    try:
        backend = MemoryBackend()
        for rank in range(writer_tp):
            layout, _, _ = state(writer_tp, rank, True, orientation)
            copy_modes(MemcacheStoreLayoutIO(backend), layout.objects(0, 1, "boundary"), True, True)
        for rank in range(reader_tp):
            layout, tensors, backing = state(reader_tp, rank, False, orientation)
            _, expected, _ = state(reader_tp, rank, True, orientation)
            copy_modes(MemcacheStoreLayoutIO(backend), layout.objects(0, 2, "boundary"), False, False)
            for actual, wanted in zip(tensors, expected):
                torch.testing.assert_close(actual[2], wanted[1], rtol=0, atol=0)
            assert torch.all(backing[-16:] == 0xA5)
    finally:
        get_conv_state_layout.cache_clear()


def test_schema_and_invalid_layout_do_not_silently_alias():
    layout, caches = attention(2, 0, source=True)
    spec = next(iter(layout.specs.values()))
    changed = SharedPoolLayout(
        KeyMetadata("test-model", 0, 0, 0),
        {name: replace(spec, head_size_v=4) for name in caches},
        local_tp=2,
        store_tp=4,
        rank=0,
        chunk_size=2,
        hash_block_size=2,
        cache_prefix="weights-v1",
    )
    assert changed.key(0, "prefix") != layout.key(0, "prefix")
    nhd = {name: tuple(t.transpose(1, 2) for t in tensors) for name, tensors in caches.items()}
    fresh = SharedPoolLayout(
        KeyMetadata("test-model", 0, 0, 0),
        layout.specs,
        local_tp=2,
        store_tp=4,
        rank=0,
        chunk_size=2,
        hash_block_size=2,
        cache_prefix="weights-v1",
    )
    with pytest.raises(ValueError, match="HND"):
        fresh.register(nhd, 3)
    with pytest.raises(ValueError, match="boundary"):
        layout.objects(1, 1, "prefix")
    with pytest.raises(ValueError, match="block"):
        layout.objects(0, 3, "prefix")


@pytest.mark.parametrize("layerwise", [False, True])
def test_flash_kda_allocated_page_layout(monkeypatch, layerwise):
    """Actual TP8 BF16/FP32 shapes and strides from the Flash baseline.

    Only the allocator block count is reduced. This tests address binding,
    not whether a model step produced a committed checkpoint at this boundary.
    """
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "SD")
    get_conv_state_layout.cache_clear()
    backend = MemoryBackend()
    restored = []
    try:
        for writing in (True, False):
            backing = torch.full((3 * 655360,), 0xA5, dtype=torch.uint8)
            conv = torch.empty(0, dtype=torch.bfloat16).set_(
                backing.untyped_storage(), 0, (3, 3, 3072), (327680, 3072, 1)
            )
            recurrent = torch.empty(0, dtype=torch.float32).set_(
                backing.untyped_storage(), 4608, (3, 8, 128, 128), (163840, 16384, 128, 1)
            )
            if writing:
                conv[1].fill_(3)
                recurrent[1].fill_(7)
            spec = MambaSpec(
                block_size=1024,
                shapes=((3, 3072), (8, 128, 128)),
                dtypes=(torch.bfloat16, torch.float32),
                mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            )
            layout = SharedPoolLayout(
                KeyMetadata("flash", 0, 0, 0),
                {"model.layers.0.self_attn": spec},
                local_tp=8,
                store_tp=8,
                rank=0,
                chunk_size=1024,
                hash_block_size=128,
                cache_prefix="flash-bf16",
            )
            layout.register({"model.layers.0.self_attn": (conv, recurrent)}, 3)
            plans = layout.objects(0, 1 if writing else 2, "checkpoint")
            assert sum(plan.size for plan in plans) == 542720
            copy_modes(MemcacheStoreLayoutIO(backend), plans, writing, layerwise if writing else False)
            assert torch.all(backing.reshape(3, 655360)[:, 542720:] == 0xA5)
            restored.append((conv, recurrent))
        for source, target in zip(restored[0], restored[1], strict=True):
            torch.testing.assert_close(source[1], target[2], rtol=0, atol=0)
    finally:
        get_conv_state_layout.cache_clear()
