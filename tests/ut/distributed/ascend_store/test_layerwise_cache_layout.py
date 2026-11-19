# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, UniformTypeKVCacheSpecs

from tests.ut.distributed.ascend_store.cache_layout_utils import make_cache_config, make_vllm_config
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
    LayerwiseCachePlanner,
    get_layerwise_gva_config,
    get_layerwise_offload_config,
    plan_layerwise_cache_reuse,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LayerwiseCacheConfig,
    allocate_cache_backing,
    apply_layerwise_kv_cache_plan,
    get_cache_views,
    get_layerwise_kv_cache_specs,
    get_layerwise_reuse_layout,
    get_physical_layer_index,
)


def _attention():
    return FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=8, dtype=torch.int8)


def _build(config, num_layers=27, buffers=6, **extra):
    vllm_config = make_vllm_config(num_layers, buffers, **extra)
    return LayerwiseCachePlanner(
        config,
        vllm_config,
        vllm_config.kv_transfer_config.kv_connector_extra_config,
        spec_reuse_rules={FullAttentionSpec: True},
    ).build()


def _names(start=0, end=27, suffix="attn"):
    return [f"model.layers.{i}.{suffix}" for i in range(start, end)]


def test_default_layout_keeps_one_buffer_per_layer():
    settings = LayerwiseCacheConfig.parse(27)
    layout = _build(make_cache_config([(_names(), _attention())]), buffers=None)
    assert settings.num_shared_buffers is None
    assert settings.num_prefetch_layers == 8
    assert settings.independent_layers == [0]
    assert not layout.has_layer_reuse
    assert layout.kv_cache_tensors == ()


def test_reuse_layout_matches_round_robin_buffer_assignments():
    names = _names()
    layout = _build(make_cache_config([(names, _attention())]))
    assert layout.prefetch_layer_map[7] == (1,)
    assert layout.prefetch_layer_map[8] == (2,)
    slots = [t.layers for t in layout.kv_cache_tensors]
    assert slots[0] == [names[0]]
    assert slots[1] == [names[i] for i in (1, 7, 13, 19, 25)]
    assert slots[2] == [names[i] for i in (2, 8, 14, 20, 26)]
    assert sorted(name for slot in slots for name in slot) == sorted(names)


@pytest.mark.parametrize(
    "value,expected", [([3, 5, 10], [3, 5, 10]), ([-1], [26]), ([1, 4], [1, 4]), ("all", list(range(27)))]
)
def test_independent_layer_parsing(value, expected):
    assert LayerwiseCacheConfig.parse(27, {"layerwise_independent_layers": value}).independent_layers == expected


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("layerwise_num_shared_buffers", True, TypeError),
        ("layerwise_num_shared_buffers", 0, ValueError),
        ("layerwise_independent_layers", 27, TypeError),
        ("layerwise_independent_layers", "1,4", TypeError),
        ("layerwise_independent_layers", [27], ValueError),
        ("layerwise_prefetch_layers", 0, ValueError),
    ],
)
def test_invalid_layout_config_is_rejected(key, value, error):
    with pytest.raises(error):
        LayerwiseCacheConfig.parse(27, {key: value})


def test_prefetch_count_can_be_overridden():
    assert (
        LayerwiseCacheConfig.parse(
            27, {"layerwise_num_shared_buffers": 6, "layerwise_prefetch_layers": 3}
        ).num_prefetch_layers
        == 3
    )


def test_reuse_config_is_scoped_to_layerwise_backend_connector():
    transfer = make_vllm_config().kv_transfer_config
    extra = transfer.kv_connector_extra_config
    assert get_layerwise_offload_config(transfer) is extra
    multi = replace(
        transfer,
        kv_connector="MultiConnector",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "OtherConnector", "kv_connector_extra_config": {"use_layerwise": True}},
                {"kv_connector": "AscendStoreConnector", "kv_connector_extra_config": extra},
            ]
        },
    )
    assert get_layerwise_offload_config(multi) is extra
    assert get_layerwise_gva_config(multi) is extra
    for changes in ({"use_layerwise": False}, {"layerwise_num_shared_buffers": None}):
        assert get_layerwise_offload_config(replace(transfer, kv_connector_extra_config={**extra, **changes})) is None
    assert get_layerwise_offload_config(replace(transfer, kv_connector="OtherConnector")) is None
    assert (
        get_layerwise_gva_config(replace(transfer, kv_connector_extra_config={**extra, "backend": "mooncake"})) is None
    )


@pytest.mark.parametrize("offset", [0, 32])
def test_standardized_descriptor_becomes_shared_backing_without_mutating_input(offset):
    names = _names(0, 6)
    config = make_cache_config([(names, _attention())], offset=offset)
    original = list(config.kv_cache_tensors)
    transfer = make_vllm_config(6, 2).kv_transfer_config
    layout = _build(config, 6, 2)
    planned = apply_layerwise_kv_cache_plan(config, transfer, layout)
    assert planned is not config
    assert config.kv_cache_tensors == original
    assert get_layerwise_reuse_layout(config) is None
    assert get_layerwise_reuse_layout(planned) is layout
    assert planned.kv_transfer_config is not transfer
    assert "_layerwise_reuse_layout" not in transfer.kv_connector_extra_config
    allocations = []

    def allocate(size):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8)

    raw = allocate_cache_backing(planned, get_layerwise_kv_cache_specs(planned), allocate, layout.cache_view_sizes)
    assert len(allocations) == 1
    assert raw[names[1]] is raw[names[3]] is raw[names[5]]
    assert raw[names[2]] is raw[names[4]]
    raw[names[1]].fill_(7)
    assert torch.all(raw[names[3]] == 7)
    assert torch.all(raw[names[0]] == 0)
    assert torch.all(raw[names[2]] == 0)


def test_different_physical_strides_use_separate_slots():
    names = _names(0, 6)
    small = _attention()
    large = replace(small, head_size=16, head_size_v=16)
    layout = _build(
        make_cache_config([(names[::2], small), (names[1::2], large)]), 6, 1, layerwise_independent_layers=[]
    )
    assert {tuple(t.layers) for t in layout.kv_cache_tensors} == {tuple(names[::2]), tuple(names[1::2])}


def test_stage_layout_uses_global_layer_numbers():
    layout = _build(make_cache_config([(_names(24, 45), _attention())]), 45, 3)
    assert layout.layers == tuple(range(24, 45))
    assert layout.independent_layers == []
    assert layout.prefetch_layer_map[27] == (24,)


@pytest.mark.parametrize(
    "name,index",
    [
        ("model.mtp.0.attn", 4),
        ("mtp.layers.0.attn", 4),
        ("model.mtp.layers.1.attn", 5),
        ("model.layers.4.attn", 4),
        ("notlayers.2.model.layers.3.attn", 2),
    ],
)
def test_physical_layer_index_supports_mtp_names(name, index):
    assert get_physical_layer_index(name, 4) == index


@pytest.mark.parametrize("replication", [1, 2])
@pytest.mark.parametrize("quantized", [False, True])
def test_indexer_native_capacity_is_preserved_and_not_rotated(replication, quantized):
    names = _names(0, 6)
    indexers = _names(1, 6, "indexer.k_cache")
    spec = AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8 if quantized else torch.bfloat16,
        scale_dim=1 if quantized else 0,
        scale_dtype=torch.float16,
        sfa_dcp_replicated_indexer_size=replication,
    )
    layout = _build(make_cache_config([(names, _attention()), (indexers, spec)]), 6, 2)
    assert layout.prefetch_layer_map[3] == (1,)
    assert not set(indexers) & layout.reused_cache_names
    assert all(layout.cache_view_sizes[name] == 3 * spec.page_size_bytes for name in indexers)
    assert len([t for t in layout.kv_cache_tensors if set(t.layers) & set(indexers)]) == len(indexers)


@pytest.mark.parametrize("mutation", ["bounds", "negative_offset", "zero_stride", "duplicate", "alignment"])
def test_invalid_native_views_are_rejected(mutation):
    config = make_cache_config([(_names(0, 4), replace(_attention(), dtype=torch.float16))])
    tensor = config.kv_cache_tensors[0]
    if mutation == "bounds":
        tensor.size -= 1
    elif mutation == "negative_offset":
        tensor.offset = -1
    elif mutation == "zero_stride":
        tensor.block_stride = 0
    elif mutation == "duplicate":
        tensor.layers[1] = tensor.layers[0]
    else:
        tensor.offset = 1
    with pytest.raises(ValueError):
        get_cache_views(config, get_layerwise_kv_cache_specs(config))


def test_padded_and_mixed_native_views_keep_their_byte_capacity():
    names = _names(0, 4)
    specs = {
        name: replace(_attention(), head_size=8 + 8 * (i % 2), head_size_v=8 + 8 * (i % 2))
        for i, name in enumerate(names)
    }
    config = make_cache_config([(names, UniformTypeKVCacheSpecs(block_size=2, kv_cache_specs=specs))])
    tensor = config.kv_cache_tensors[0]
    tensor.block_stride *= 2
    tensor.layer_stride = config.num_blocks * tensor.block_stride
    tensor.size = len(names) * tensor.layer_stride
    layout = _build(config, 4, 1)
    assert set(layout.cache_view_sizes.values()) == {tensor.layer_stride}
    assert layout.prefetch_layer_map == {2: (1,), 3: (2,)}


def _glm_mamba_config(stage_start, *, mode="align", checkpoints=0):
    names = _names(stage_start, 24 if stage_start == 0 else 45, "mixer")
    spec = MambaSpec(
        block_size=16,
        shapes=((8, 8),),
        dtypes=(torch.float32,),
        mamba_cache_mode=mode,
        num_prefill_checkpoint_blocks=checkpoints,
    )
    return make_cache_config([(names, spec)]), names


@pytest.mark.parametrize("stage_start", [0, 24])
def test_glm_aligned_mamba_three_buffers_reuses_only_after_save(stage_start):
    config, names = _glm_mamba_config(stage_start)
    vllm_config = make_vllm_config()
    layout = plan_layerwise_cache_reuse(config, vllm_config)
    assert layout is not None
    assert layout.layers == tuple(range(stage_start, stage_start + len(names)))
    assert layout.independent_layers == ([0] if stage_start == 0 else [])
    rotating = names[1:] if stage_start == 0 else names
    assert layout.reused_cache_names == frozenset(rotating)
    assert len(layout.kv_cache_tensors) == 3 + (stage_start == 0)
    config = apply_layerwise_kv_cache_plan(config, vllm_config.kv_transfer_config, layout)
    raw = allocate_cache_backing(
        config,
        get_layerwise_kv_cache_specs(config),
        lambda n: torch.zeros(n, dtype=torch.int8),
        layout.cache_view_sizes,
    )
    for i, name in enumerate(rotating):
        assert raw[name] is raw[rotating[i % 3]]
        if i >= 3:
            layer = layout.cache_layer_indices[name]
            assert layout.prefetch_layer_map[layer] == (layout.cache_layer_indices[rotating[i - 3]],)
    raw[rotating[0]].fill_(7)
    assert torch.all(raw[rotating[3]] == 7)
    assert torch.all(raw[rotating[1]] == 0)


@pytest.mark.parametrize("stage_start", [0, 24])
@pytest.mark.parametrize("mode,checkpoints", [("all", 0), ("align", 1)])
def test_glm_rejects_unsupported_mamba_state_semantics(stage_start, mode, checkpoints):
    config, _ = _glm_mamba_config(stage_start, mode=mode, checkpoints=checkpoints)
    with pytest.raises(NotImplementedError, match="unsupported state semantics"):
        plan_layerwise_cache_reuse(config, make_vllm_config())


def test_glm_keeps_tail_indexer_and_mtp_resident():
    config, names = _glm_mamba_config(0)
    state_spec = config.kv_cache_groups[0].kv_cache_spec
    draft = "model.mtp.0.mixer"
    tail = AscendIndexerKPoolTailSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        sliding_window=4,
        compress_ratio=4,
        ring_capacity=4,
    )
    indexer = AscendSFAIndexerCacheSpec(block_size=16, num_kv_heads=1, head_size=4, dtype=torch.float16)
    config = make_cache_config(
        [(names + [draft], state_spec), (["model.layers.0.tail"], tail), (["model.layers.0.indexer.k_cache"], indexer)]
    )
    layout = plan_layerwise_cache_reuse(config, make_vllm_config())
    assert layout is not None
    assert layout.draft_entry_names == frozenset({draft})
    assert not {draft, "model.layers.0.tail", "model.layers.0.indexer.k_cache"} & layout.reused_cache_names


@pytest.mark.parametrize("reason", ["disabled", "unknown_model", "empty", "no_rotation", "all_independent"])
def test_no_reuse_leaves_native_descriptors_unchanged(reason):
    config, _ = _glm_mamba_config(0)
    original = list(config.kv_cache_tensors)
    vllm_config = make_vllm_config()
    if reason == "disabled":
        vllm_config.kv_transfer_config = None
    elif reason == "unknown_model":
        vllm_config.model_config.hf_text_config.model_type = "deepseek_v4"
    elif reason == "empty":
        config.num_blocks = 0
    elif reason == "no_rotation":
        vllm_config.kv_transfer_config.kv_connector_extra_config["layerwise_num_shared_buffers"] = 45
    else:
        vllm_config.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = "all"
    assert plan_layerwise_cache_reuse(config, vllm_config) is None
    assert config.kv_cache_tensors == original


@pytest.mark.parametrize("constraint", ["override", "host_resident"])
def test_glm_rejects_incompatible_memory_settings(constraint):
    config, _ = _glm_mamba_config(0)
    vllm_config = make_vllm_config()
    if constraint == "override":
        vllm_config.cache_config.num_gpu_blocks_override = 8
    else:
        group = config.kv_cache_groups[0]
        config = make_cache_config(
            [
                (group.layer_names, group.kv_cache_spec),
                (["model.layers.44.mixer"], group.kv_cache_spec),
            ]
        )
        config.kv_cache_tensors[1].host_resident = True
    with pytest.raises(ValueError):
        plan_layerwise_cache_reuse(config, vllm_config)


@pytest.mark.parametrize(
    "draft,stage_only,buffers", [(False, False, 2), (True, False, 2), (False, True, 1), (True, False, 8)]
)
def test_physical_budget_counts_persistent_and_rotating_entries(draft, stage_only, buffers):
    # Memory accounting moved from NPUWorker's logical-layer factor to the
    # planner's actual byte descriptors. Cover the former budget cases here.
    names = _names(2 if stage_only else 0, 6)
    draft_name = "model.mtp.0.attn"
    if draft:
        names.append(draft_name)
    spec = _attention()
    indexers = _names(1, 4, "indexer.k_cache") if not stage_only else []
    indexer = AscendSFAIndexerCacheSpec(block_size=2, num_kv_heads=1, head_size=4, dtype=torch.int8, scale_dim=1)
    groups = [(names, spec)] + ([(indexers, indexer)] if indexers else [])
    config = make_cache_config(groups, num_blocks=1)
    layout = _build(config, 6, buffers)
    if buffers >= len(names):
        assert not layout.has_layer_reuse
        return
    resident_main = int(not stage_only) + int(draft)
    expected = (buffers + resident_main) * spec.page_size_bytes + len(indexers) * indexer.page_size_bytes
    assert {tensor.size for tensor in layout.kv_cache_tensors} == {expected}
    assert draft_name not in layout.reused_cache_names


def test_explicit_draft_specs_remain_resident_with_overlapping_layer_ids():
    spec = _attention()
    draft = replace(spec, is_draft=True, head_size=16, head_size_v=16)
    draft_names = [f"draft.model.layers.{i}.attn" for i in range(5)]
    layout = _build(make_cache_config([(_names(0, 6), spec), (draft_names, draft)], num_blocks=1), 6, 2)
    assert layout.draft_entry_names == frozenset(draft_names)
    assert not layout.draft_entry_names & layout.reused_cache_names
    assert {tensor.size for tensor in layout.kv_cache_tensors} == {3 * spec.page_size_bytes + 5 * draft.page_size_bytes}
