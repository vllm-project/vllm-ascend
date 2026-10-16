# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.cache_layout import (
    CacheComponentRole,
    MemoryKind,
    bind_layer_cache_views,
    build_mla_layer_cache_descriptor,
    resolve_dsa_main_cache_tensors,
    stable_model_fingerprint,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    KeyMetadata,
    PoolKey,
    build_layerwise_store_key,
)


def _a3_spec(*, c8: bool = False):
    return SimpleNamespace(
        block_size=4,
        num_kv_heads=1,
        sparse_head_dim=(8, 2, 4),
        dtype=torch.bfloat16,
        cache_sparse_c8=c8,
        c8_k_cache_dtype=torch.int8,
        c8_k_scale_cache_dtype=torch.float16,
    )


def _descriptor(layer_id: int = 0, *, mtp: bool = False):
    return build_mla_layer_cache_descriptor(
        layer_name=f"model.layers.{layer_id}.self_attn",
        layer_id=layer_id,
        group_id=0,
        cache_family_id="default",
        kv_cache_spec=_a3_spec(),
        is_mtp_layer=mtp,
    )


def test_a3_non_c8_descriptor_and_views():
    descriptor = _descriptor()
    assert [item.component_role for item in descriptor.components] == [
        CacheComponentRole.MAIN_KV,
        CacheComponentRole.MAIN_K_ROPE,
        CacheComponentRole.INDEXER_K,
    ]
    assert all(
        item.prefill_memory_kind is MemoryKind.DEVICE
        for item in descriptor.components
    )
    assert [item.decode_memory_kind for item in descriptor.components] == [
        MemoryKind.HOST_DEVICE_VISIBLE,
        MemoryKind.HOST_DEVICE_VISIBLE,
        MemoryKind.DEVICE,
    ]

    tensors = [
        torch.empty((3, *item.shape_per_block), dtype=item.dtype)
        for item in descriptor.components
    ]
    views = bind_layer_cache_views(descriptor, tensors)
    assert views[CacheComponentRole.MAIN_KV] is tensors[0]
    assert views[CacheComponentRole.INDEXER_K] is tensors[2]


def test_a3_c8_descriptor_has_indexer_scale():
    descriptor = build_mla_layer_cache_descriptor(
        layer_name="layer",
        layer_id=0,
        group_id=0,
        cache_family_id="default",
        kv_cache_spec=_a3_spec(c8=True),
    )
    assert descriptor.components[2].dtype is torch.int8
    assert descriptor.components[3].component_role is CacheComponentRole.INDEXER_SCALE
    assert descriptor.components[3].dtype is torch.float16


def test_dsa_main_cache_resolution_uses_merged_a5_c8_tensor():
    spec = _a3_spec(c8=True)
    spec.sparse_head_dim = (12, 0, 4)
    descriptor = build_mla_layer_cache_descriptor(
        layer_name="layer",
        layer_id=0,
        group_id=0,
        cache_family_id="default",
        kv_cache_spec=spec,
    )
    tensors = [
        torch.empty((3, *item.shape_per_block), dtype=item.dtype)
        for item in descriptor.components
    ]

    full_kv_cache, full_k_rope = resolve_dsa_main_cache_tensors(
        descriptor,
        tensors,
    )

    assert full_kv_cache is tensors[0]
    assert full_k_rope is tensors[0]
    assert full_k_rope is not tensors[1]


def test_bind_rejects_wrong_dtype_and_shape():
    descriptor = _descriptor()
    tensors = [
        torch.empty((2, *item.shape_per_block), dtype=item.dtype)
        for item in descriptor.components
    ]
    tensors[0] = tensors[0].float()
    with pytest.raises(TypeError, match="dtype"):
        bind_layer_cache_views(descriptor, tensors)


def test_layerwise_key_extends_but_does_not_change_ordinary_key():
    metadata = KeyMetadata("glm", 1, 2, 3, 4, 0, "kv", "default")
    ordinary = PoolKey(metadata, "abc")
    assert ordinary.to_string() == (
        "glm@pcp2@dcp3@head_or_tp_rank:1@pp_rank:4@group:0"
        "@cache_role:kv@cache_family:default@abc"
    )
    old_hash_input = hash(("glm", 1, 2, 3, 4, 0, "kv", "default", "abc"))
    assert hash(ordinary) == old_hash_input

    fingerprint = stable_model_fingerprint({"model": "glm", "dtype": "bf16"})
    layerwise = build_layerwise_store_key(
        metadata,
        "abc",
        model_fingerprint=fingerprint,
        cache_layout_version=1,
    )
    key = layerwise.to_string()
    assert "@cache_role:layerwise_kv" in key
    assert f"@model:{fingerprint}@layout:1" in key
    assert "request" not in key and "layer_id" not in key


def test_layerwise_key_fields_are_atomic():
    metadata = KeyMetadata(
        "glm",
        0,
        0,
        0,
        0,
        model_fingerprint="fingerprint",
    )
    with pytest.raises(ValueError, match="supplied together"):
        PoolKey(metadata, "hash").to_string()
