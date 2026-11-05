# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_io import MemcacheStoreLayoutIO
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_layout import SharedPoolLayout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_transfer import (
    shared_draft_names,
    target_group_specs,
)

from .test_shared_io import MemoryBackend
from .test_shared_layout import attention, copy_modes


@pytest.mark.parametrize("valid", [1, 3, 4, 5, 7, 8])
@pytest.mark.parametrize("layerwise", [False, True])
def test_mixed_compressed_mla_rows_and_empty_rope(valid, layerwise):
    """Logical pages contain two kernel pages and mixed compression ratios.

    A partial compressed row is NOT a prefix value: its live values reside
    in a separate circular state. Destination padding must remain untouched.
    """
    backend = MemoryBackend()
    bound = []
    for source in (True, False):
        specs, caches = {}, {}
        for name, ratio in (("main", 1), ("indexer", 4)):
            rows, width = 4 // ratio, 3
            backing = torch.full((6, rows * width + 5), -7.0)
            value = backing.as_strided((6, rows, 1, width), (rows * width + 5, width, width, 1))
            if source:
                value[2:4] = torch.arange(2 * rows * width).reshape(2, rows, 1, width)
            specs[name] = MLAAttentionSpec(
                block_size=8,
                num_kv_heads=1,
                head_size=width,
                dtype=torch.float32,
                tokens_per_state=ratio,
            )
            caches[name] = (value, value[..., :0])
        layout = SharedPoolLayout(
            KeyMetadata("model", 0, 0, 0),
            specs,
            local_tp=1,
            store_tp=1,
            rank=0,
            chunk_size=8,
            hash_block_size=4,
            cache_prefix="epoch",
        )
        layout.register(caches, 3)
        plans = (
            layout.objects(0, 1 if source else 2, "prefix")
            if valid == 8
            else layout.partial_objects(0, 1 if source else 2, "prefix", valid)
        )
        assert plans[0].size == (valid + valid // 4) * 3 * 4
        copy_modes(MemcacheStoreLayoutIO(backend), plans, source, layerwise if source else False)
        bound.append(caches)
    for name, ratio in (("main", 1), ("indexer", 4)):
        expected = bound[0][name][0][2:4].reshape(-1, 3)
        actual = bound[1][name][0][4:6].reshape(-1, 3)
        rows = valid // ratio
        torch.testing.assert_close(actual[:rows], expected[:rows], rtol=0, atol=0)
        assert torch.all(actual[rows:] == -7)


def mla(tp, rank, source, *, group_id=0):
    # Split latent/rope and indexer/scales, including different dtypes and
    # physical page padding. No model-specific dimensions in the binding.
    caches = {}
    for name, components in (
        ("model.layers.0.attn", ((5, torch.float32), (3, torch.float32))),
        ("model.layers.0.indexer", ((7, torch.int8), (1, torch.float32))),
    ):
        tensors = []
        for width, dtype in components:
            backing = torch.full((6, 2 * width + 5), -7, dtype=dtype)
            value = backing.as_strided((6, 2, 1, width), (2 * width + 5, width, width, 1))
            if source:
                value[2:4] = torch.arange(4 * width, dtype=dtype).reshape(2, 2, 1, width)
            tensors.append(value)
        caches[name] = tuple(tensors)
    spec = MLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float32)
    layout = SharedPoolLayout(
        KeyMetadata("model", rank, 0, 0, group_id),
        {name: spec for name in caches},
        local_tp=tp,
        store_tp=8,
        rank=rank,
        chunk_size=4,
        hash_block_size=2,
        cache_prefix="epoch",
    )
    layout.register(caches, 3)
    return layout, caches


@pytest.mark.parametrize("writer_tp,reader_tp", [(1, 8), (8, 1), (4, 8)])
@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("valid", [1, 3, 4])
def test_replicated_mla_components_and_tail(writer_tp, reader_tp, layerwise, valid):
    backend = MemoryBackend()
    writer, expected = mla(writer_tp, 0, True)
    plans = writer.objects(0, 1, "prefix") if valid == 4 else writer.partial_objects(0, 1, "prefix", valid)
    copy_modes(MemcacheStoreLayoutIO(backend), plans, True, layerwise)
    for rank in range(reader_tp):
        reader, actual = mla(reader_tp, rank, False, group_id=9)
        incoming = reader.objects(0, 2, "prefix") if valid == 4 else reader.partial_objects(0, 2, "prefix", valid)
        assert [p.key for p in incoming] == [p.key for p in plans]
        copy_modes(MemcacheStoreLayoutIO(backend), incoming, False, False)
        for name in actual:
            for result, wanted in zip(actual[name], expected[name], strict=True):
                flat = result[4:6].reshape(4, -1)
                torch.testing.assert_close(flat[:valid], wanted[2:4].reshape(4, -1)[:valid], rtol=0, atol=0)
                assert torch.all(flat[valid:] == -7)
                assert torch.all(result[:4] == -7)
    assert len(backend.objects) == 1
    assert not backend.leases


def test_hnd_tail_uses_existing_store_shards():
    backend = MemoryBackend()
    for rank in range(2):
        layout, _ = attention(2, rank, source=True, chunk=4)
        copy_modes(MemcacheStoreLayoutIO(backend), layout.partial_objects(0, 1, "tail", 3), True, True)
    for rank in range(4):
        reader, actual = attention(4, rank, source=False, chunk=4)
        _, expected = attention(4, rank, source=True, chunk=4)
        copy_modes(MemcacheStoreLayoutIO(backend), reader.partial_objects(0, 2, "tail", 3), False, False)
        for name in actual:
            for result, wanted in zip(actual[name], expected[name], strict=True):
                torch.testing.assert_close(result[4], wanted[2], rtol=0, atol=0)
                torch.testing.assert_close(result[5, :, :1], wanted[3, :, :1], rtol=0, atol=0)
                assert torch.all(result[5, :, 1:] == -99)


def test_full_and_partial_namespaces_cannot_alias():
    layout, _ = mla(1, 0, True)
    full = layout.objects(0, 1, "same")[0]
    tail = layout.partial_objects(0, 1, "same", 3)[0]
    assert full.key != tail.key
    assert tail.size * 4 == full.size * 3
    with pytest.raises(ValueError):
        layout.partial_objects(0, 1, "same", 4)


def test_target_keys_do_not_depend_on_draft_group_number():
    target, caches = mla(8, 0, True)
    spec = next(iter(target.specs.values()))
    draft = SimpleNamespace(layer_names=["draft.layers.0.attn"], kv_cache_spec=replace(spec, is_draft=True))
    assert target_group_specs(draft) == {}
    target_group = SimpleNamespace(layer_names=list(caches), kv_cache_spec=spec)
    assert target_group_specs(target_group) == target.specs
    with_draft_group, _ = mla(4, 1, False, group_id=1)
    assert target.key(0, "hash") == with_draft_group.key(0, "hash")


def test_draft_indexer_inherits_exact_main_owner_without_runner_changes():
    from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendSFAIndexerCacheSpec

    main = MLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float32, is_draft=True)
    indexer = AscendSFAIndexerCacheSpec(block_size=4, num_kv_heads=1, head_size=7, dtype=torch.int8)
    tail = AscendIndexerKPoolTailSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=7,
        dtype=torch.float32,
        sliding_window=4,
        compress_ratio=4,
    )
    groups = [
        SimpleNamespace(layer_names=["draft.layers.0.self_attn.attn"], kv_cache_spec=main),
        SimpleNamespace(
            layer_names=["draft.layers.0.self_attn.indexer.k_cache", "model.layers.0.self_attn.indexer.k_cache"],
            kv_cache_spec=indexer,
        ),
        SimpleNamespace(
            layer_names=["draft.layers.0.self_attn.indexer.tail_cache", "model.layers.0.self_attn.indexer.tail_cache"],
            kv_cache_spec=tail,
        ),
    ]
    draft = shared_draft_names(SimpleNamespace(kv_cache_groups=groups))
    assert draft == {
        "draft.layers.0.self_attn.attn",
        "draft.layers.0.self_attn.indexer.k_cache",
        "draft.layers.0.self_attn.indexer.tail_cache",
    }
    assert list(target_group_specs(groups[1], draft)) == ["model.layers.0.self_attn.indexer.k_cache"]
    assert list(target_group_specs(groups[2], draft)) == ["model.layers.0.self_attn.indexer.tail_cache"]


def test_quantization_semantics_are_not_inferred_from_equal_payload_size():
    from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec

    keys = []
    for quantized in (False, True):
        spec = AscendSFAIndexerCacheSpec(
            block_size=4, num_kv_heads=1, head_size=8, dtype=torch.int8, cache_sparse_li_c8=quantized
        )
        layout = SharedPoolLayout(
            KeyMetadata("model", 0, 0, 0),
            {"layer.0": spec},
            local_tp=1,
            store_tp=1,
            rank=0,
            chunk_size=4,
            hash_block_size=4,
            cache_prefix="epoch",
        )
        layout.register({"layer.0": (torch.zeros((2, 4, 1, 8), dtype=torch.int8),)}, 2)
        keys.append(layout.key(0, "same-prefix"))
    assert keys[0] != keys[1]
