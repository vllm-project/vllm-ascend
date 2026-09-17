# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise mixed DFlash storage with real CPU tensors and production views.

These tests need Torch and vLLM, but do not execute an NPU operator. The small
12-block pool reproduces the cross-group alias without a multi-GiB allocation.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionBackendImpl
from vllm_ascend.core.dflash_cache import align_dflash_cache_specs, validate_dflash_cache_views
from vllm_ascend.worker.v2 import attn_utils

NUM_BLOCKS = 12
KERNEL_BLOCK_SIZE = 128
STORAGE_BLOCK_SIZE = 1536
SLIDING_WINDOW = 2048
CONV_PAGE_BYTES = 102400
STATE_PAGE_BYTES = 1572864
PAGE_BYTES = CONV_PAGE_BYTES + 2 * STATE_PAGE_BYTES
FULL_LAYER = "model.layers.3.self_attn.attn"
SWA_LAYER = "draft_model.layers.0.self_attn.attn"
MAMBA_LAYER = "model.layers.0.linear_attn"


@pytest.fixture
def mixed_cache(monkeypatch):
    config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=SimpleNamespace(
            method="dflash",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(layer_types=["sliding_attention"] * 4 + ["full_attention"]),
            ),
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5")),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        quant_config=None,
        additional_config={},
    )
    original_specs = {
        FULL_LAYER: FullAttentionSpec(
            block_size=STORAGE_BLOCK_SIZE,
            num_kv_heads=2,
            head_size=256,
            dtype=torch.bfloat16,
            page_size_padded=PAGE_BYTES,
        ),
        SWA_LAYER: SlidingWindowSpec(
            block_size=KERNEL_BLOCK_SIZE,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=SLIDING_WINDOW,
            page_size_padded=PAGE_BYTES,
        ),
        MAMBA_LAYER: MambaSpec(
            block_size=STORAGE_BLOCK_SIZE,
            shapes=((10, 5120), (24, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            page_size_padded=PAGE_BYTES,
        ),
    }
    specs = align_dflash_cache_specs(config, original_specs)
    groups = [KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec) for name, spec in specs.items()]
    cache_config = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups)
    # Distinct groups may use the same arena, but must use different physical
    # block IDs. They must therefore agree on every block's byte placement.
    backing = torch.zeros(NUM_BLOCKS * PAGE_BYTES, dtype=torch.uint8)
    raw = dict.fromkeys(specs, backing)
    layers = {name: SimpleNamespace(impl=SimpleNamespace()) for name in specs}
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args, **_kwargs: layers)
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda _: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
    monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda _: False)
    attention_groups = [
        AttentionGroup(
            backend=AscendAttentionBackend,
            layer_names=group.layer_names,
            kv_cache_spec=group.kv_cache_spec,
            kv_cache_group_id=index,
        )
        for index, group in enumerate(groups)
    ]
    caches = attn_utils._reshape_kv_cache_v2(
        attn_groups=attention_groups,
        kv_cache_raw_tensors=raw,
        cache_dtype="auto",
        kernel_block_sizes=[KERNEL_BLOCK_SIZE] * len(groups),
        shared_kv_cache_layers={},
        kv_cache_config=cache_config,
    )
    return SimpleNamespace(
        config=config,
        original_specs=original_specs,
        specs=specs,
        cache_config=cache_config,
        backing=backing,
        raw=raw,
        caches=caches,
        layers=layers,
    )


@pytest.fixture
def mixed_cache_width_two(mixed_cache, monkeypatch):
    if "layers" not in KVCacheTensor.__dataclass_fields__:
        pytest.skip("Requires standardized layer-strided KV cache descriptors")
    config = mixed_cache.config
    config.kv_transfer_config = None
    names_by_type = {
        FULL_LAYER: [FULL_LAYER, "model.layers.7.self_attn.attn"],
        SWA_LAYER: [SWA_LAYER, "draft_model.layers.1.self_attn.attn"],
        MAMBA_LAYER: [MAMBA_LAYER, "model.layers.1.linear_attn"],
    }
    groups = [
        KVCacheGroupSpec(layer_names=names, kv_cache_spec=mixed_cache.specs[name])
        for name, names in names_by_type.items()
    ]
    layer_bytes = NUM_BLOCKS * PAGE_BYTES
    descriptors = [
        KVCacheTensor(
            size=2 * layer_bytes,
            layers=list(group.layer_names),
            layer_stride=layer_bytes,
            block_stride=PAGE_BYTES,
            offset=0,
        )
        for group in groups
    ]
    cache_config = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=descriptors, kv_cache_groups=groups)
    specs = {name: group.kv_cache_spec for group in groups for name in group.layer_names}
    layers = {name: SimpleNamespace(impl=SimpleNamespace()) for name in specs}
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args, **_kwargs: layers)
    monkeypatch.setattr(attn_utils, "KVPPConfig", SimpleNamespace(from_vllm_config=lambda _: SimpleNamespace(size=1)))
    monkeypatch.setattr(attn_utils, "vllm_version_is", lambda _: False)
    # Exercise the production allocator, not only hand-constructed aliases:
    # three descriptors must overlay ONE two-layer backing allocation.
    raw = attn_utils._allocate_kv_cache(cache_config, shared_layers={}, device=torch.device("cpu"))
    attention_groups = [
        AttentionGroup(
            backend=AscendAttentionBackend,
            layer_names=group.layer_names,
            kv_cache_spec=group.kv_cache_spec,
            kv_cache_group_id=index,
        )
        for index, group in enumerate(groups)
    ]
    caches = attn_utils._reshape_kv_cache_v2(
        attn_groups=attention_groups,
        kv_cache_raw_tensors=raw,
        cache_dtype="auto",
        kernel_block_sizes=[KERNEL_BLOCK_SIZE] * len(groups),
        shared_kv_cache_layers={},
        kv_cache_config=cache_config,
    )
    return SimpleNamespace(
        config=config,
        cache_config=cache_config,
        specs=specs,
        names_by_type=names_by_type,
        raw=raw,
        caches=caches,
        layers=layers,
    )


def _physical_attention_view(cache, spec):
    return cache.view(NUM_BLOCKS, STORAGE_BLOCK_SIZE, spec.num_kv_heads, spec.head_size)


def test_alignment_preserves_sliding_semantics(mixed_cache):
    assert mixed_cache.original_specs[SWA_LAYER].block_size == KERNEL_BLOCK_SIZE
    spec = mixed_cache.specs[SWA_LAYER]
    assert type(spec) is SlidingWindowSpec
    assert spec.block_size == STORAGE_BLOCK_SIZE
    assert spec.sliding_window == SLIDING_WINDOW
    assert mixed_cache.specs[FULL_LAYER].sliding_window is None
    assert {spec.page_size_bytes for spec in mixed_cache.specs.values()} == {PAGE_BYTES}


def test_materialized_views_share_aligned_planes_and_install_null_guards(mixed_cache):
    base = mixed_cache.backing.data_ptr()
    for name in (FULL_LAYER, SWA_LAYER):
        spec = mixed_cache.specs[name]
        key, value = mixed_cache.caches[name]
        expected_shape = (
            NUM_BLOCKS * STORAGE_BLOCK_SIZE // KERNEL_BLOCK_SIZE,
            KERNEL_BLOCK_SIZE,
            spec.num_kv_heads,
            spec.head_size,
        )
        assert key.shape == value.shape == expected_shape
        assert key.dtype == value.dtype == torch.bfloat16
        assert key.is_contiguous() and value.is_contiguous()
        assert key.data_ptr() - base == NUM_BLOCKS * CONV_PAGE_BYTES
        assert value.data_ptr() - base == NUM_BLOCKS * (CONV_PAGE_BYTES + STATE_PAGE_BYTES)
        assert key.numel() * key.element_size() == NUM_BLOCKS * STATE_PAGE_BYTES
        impl = mixed_cache.layers[name].impl
        assert impl._dflash_null_block_size == STORAGE_BLOCK_SIZE
        assert impl._dflash_cache_slot_limit == NUM_BLOCKS * STORAGE_BLOCK_SIZE
    conv, state = mixed_cache.caches[MAMBA_LAYER]
    assert conv.shape == (NUM_BLOCKS, 10, 5120)
    assert state.shape == (NUM_BLOCKS, 24, 128, 128)
    assert conv.dtype == torch.bfloat16 and state.dtype == torch.float32
    assert conv.data_ptr() == base
    assert state.data_ptr() - base == NUM_BLOCKS * CONV_PAGE_BYTES
    validate_dflash_cache_views(mixed_cache.config, mixed_cache.cache_config, mixed_cache.raw, mixed_cache.caches)


def test_swa_block_one_cannot_overwrite_other_full_or_mamba_blocks(mixed_cache):
    full_key, full_value = [
        _physical_attention_view(cache, mixed_cache.specs[FULL_LAYER]) for cache in mixed_cache.caches[FULL_LAYER]
    ]
    swa_key, swa_value = [
        _physical_attention_view(cache, mixed_cache.specs[SWA_LAYER]) for cache in mixed_cache.caches[SWA_LAYER]
    ]
    conv, state = mixed_cache.caches[MAMBA_LAYER]
    # The old 128-token SWA planes overwrote full V blocks 10 and 11 even
    # though the allocator assigned SWA physical block 1, a different ID.
    full_value[10].fill_(11)
    full_value[11].fill_(22)
    full_key[3].fill_(55)
    conv.fill_(7)
    state[2].fill_(9)
    swa_key[1].fill_(33)
    swa_value[1].fill_(44)
    assert torch.all(full_value[10] == 11)
    assert torch.all(full_value[11] == 22)
    assert torch.all(full_key[3] == 55)
    assert torch.all(conv == 7)
    assert torch.all(state[2] == 9)
    assert torch.all(swa_key[1] == 33)
    assert torch.all(swa_value[1] == 44)


def test_width_two_allocator_uses_one_backing_with_separate_layer_arenas(mixed_cache_width_two):
    fixture = mixed_cache_width_two
    layer_bytes = NUM_BLOCKS * PAGE_BYTES
    storages = {raw.untyped_storage().data_ptr() for raw in fixture.raw.values()}
    assert len(storages) == 1
    base = next(iter(storages))
    for names in fixture.names_by_type.values():
        for index, name in enumerate(names):
            raw = fixture.raw[name]
            assert raw.untyped_storage().nbytes() == 2 * layer_bytes
            assert raw.numel() * raw.element_size() == layer_bytes
            assert raw.data_ptr() - base == index * layer_bytes
    validate_dflash_cache_views(fixture.config, fixture.cache_config, fixture.raw, fixture.caches)


def test_width_two_same_group_layers_keep_independent_state(mixed_cache_width_two):
    fixture = mixed_cache_width_two
    for names in fixture.names_by_type.values():
        # Both layers use the SAME physical block ID in one group's table,
        # but own different layer arenas within the standardized backing.
        for component in range(2):
            first, second = (fixture.caches[name][component] for name in names)
            if names[0] != MAMBA_LAYER:
                first = _physical_attention_view(first, fixture.specs[names[0]])
                second = _physical_attention_view(second, fixture.specs[names[1]])
            first[3].fill_(11 + component)
            second[3].fill_(21 + component)
            assert torch.all(first[3] == 11 + component)
            assert torch.all(second[3] == 21 + component)


def test_width_two_cross_group_writes_preserve_other_physical_blocks(mixed_cache_width_two):
    fixture = mixed_cache_width_two
    for index in range(2):
        full_name = fixture.names_by_type[FULL_LAYER][index]
        swa_name = fixture.names_by_type[SWA_LAYER][index]
        mamba_name = fixture.names_by_type[MAMBA_LAYER][index]
        full_key, full_value = [
            _physical_attention_view(cache, fixture.specs[full_name]) for cache in fixture.caches[full_name]
        ]
        swa_key, swa_value = [
            _physical_attention_view(cache, fixture.specs[swa_name]) for cache in fixture.caches[swa_name]
        ]
        conv, state = fixture.caches[mamba_name]
        # Within each layer arena, groups still share backing bytes and rely
        # on distinct live physical IDs. Check both historical alias victims.
        full_value[10].fill_(11)
        full_value[11].fill_(22)
        full_key[3].fill_(55)
        conv[2].fill_(7)
        state[2].fill_(9)
        swa_key[1].fill_(33)
        swa_value[1].fill_(44)
        assert torch.all(full_value[10] == 11)
        assert torch.all(full_value[11] == 22)
        assert torch.all(full_key[3] == 55)
        assert torch.all(conv[2] == 7)
        assert torch.all(state[2] == 9)
        assert torch.all(swa_key[1] == 33)
        assert torch.all(swa_value[1] == 44)
    validate_dflash_cache_views(fixture.config, fixture.cache_config, fixture.raw, fixture.caches)


def test_view_validation_rejects_wrong_actual_dtype(mixed_cache):
    key, value = mixed_cache.caches[SWA_LAYER]
    mixed_cache.caches[SWA_LAYER] = (key.view(torch.float16), value)
    with pytest.raises(ValueError):
        validate_dflash_cache_views(mixed_cache.config, mixed_cache.cache_config, mixed_cache.raw, mixed_cache.caches)


def test_view_validation_rejects_wrong_plane_offset(mixed_cache):
    key, value = mixed_cache.caches[SWA_LAYER]
    shifted_key = key.as_strided(key.shape, key.stride(), storage_offset=key.storage_offset() + 1)
    mixed_cache.caches[SWA_LAYER] = (shifted_key, value)
    with pytest.raises(ValueError):
        validate_dflash_cache_views(mixed_cache.config, mixed_cache.cache_config, mixed_cache.raw, mixed_cache.caches)


def test_view_validation_rejects_noncontiguous_plane(mixed_cache):
    key, value = mixed_cache.caches[SWA_LAYER]
    mixed_cache.caches[SWA_LAYER] = (key.transpose(-1, -2), value)
    with pytest.raises(ValueError):
        validate_dflash_cache_views(mixed_cache.config, mixed_cache.cache_config, mixed_cache.raw, mixed_cache.caches)


def test_view_validation_rejects_wrong_raw_allocation_size(mixed_cache):
    mixed_cache.raw[SWA_LAYER] = mixed_cache.raw[SWA_LAYER][:-2]
    with pytest.raises(ValueError):
        validate_dflash_cache_views(mixed_cache.config, mixed_cache.cache_config, mixed_cache.raw, mixed_cache.caches)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_slot_guard_masks_entire_physical_null_block_and_out_of_bounds(dtype):
    limit = NUM_BLOCKS * STORAGE_BLOCK_SIZE
    impl = SimpleNamespace(_dflash_null_block_size=STORAGE_BLOCK_SIZE, _dflash_cache_slot_limit=limit)
    slots = torch.arange(-2, limit + 2, dtype=dtype)
    before = slots.clone()
    masked = AscendAttentionBackendImpl._mask_dflash_cache_slots(impl, slots)
    expected = torch.full_like(slots, PAD_SLOT_ID)
    expected[STORAGE_BLOCK_SIZE + 2 : limit + 2] = torch.arange(STORAGE_BLOCK_SIZE, limit, dtype=dtype)
    torch.testing.assert_close(masked, expected)
    torch.testing.assert_close(slots, before)
    assert masked.dtype == dtype


def test_slot_guard_is_identity_outside_mixed_dflash():
    slots = torch.tensor([PAD_SLOT_ID, 0, KERNEL_BLOCK_SIZE - 1, STORAGE_BLOCK_SIZE], dtype=torch.int64)
    assert AscendAttentionBackendImpl._mask_dflash_cache_slots(SimpleNamespace(), slots) is slots
