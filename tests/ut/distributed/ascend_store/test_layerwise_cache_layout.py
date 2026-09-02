from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    build_layerwise_cache_layout,
    build_layerwise_reuse_layout,
    get_gva_layerwise_config,
    get_layerwise_physical_layer_index,
    get_raw_cache_components,
)


def _make_full_attention_spec(
    *,
    num_kv_heads: int = 1,
    head_size: int = 8,
) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=2,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size,
        dtype=torch.int8,
    )


def _make_vllm_config(
    num_layers: int,
    num_shared_buffers: int,
    total_num_layers: int | None = None,
    layer_start: int = 0,
):
    model_config = MagicMock()
    model_config.get_num_layers.return_value = num_layers
    model_config.get_total_num_hidden_layers.return_value = num_layers if total_num_layers is None else total_num_layers
    model_config.get_layers_start_end_indices.return_value = (
        layer_start,
        layer_start + num_layers,
    )
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": True,
                "layerwise_num_shared_buffers": num_shared_buffers,
            },
        ),
        model_config=model_config,
        parallel_config=MagicMock(),
    )


def test_no_reuse_skips_topology_validation():
    spec = MambaSpec(
        block_size=2,
        shapes=((1,),),
        dtypes=(torch.int8,),
    )
    original_tensors = [
        KVCacheTensor(size=16, shared_by=["model.layers.0.self_attn"]),
        KVCacheTensor(size=16, shared_by=["model.layers.1.self_attn"]),
        KVCacheTensor(size=16, shared_by=["model.mtp.0.self_attn"]),
    ]
    layer_names = [tensor.shared_by[0] for tensor in original_tensors]
    kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=original_tensors.copy(),
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(layer_names, spec),
                ),
            )
        ],
    )

    assert apply_layerwise_kv_cache_plan(kv_cache_config, _make_vllm_config(2, 2)) is False

    assert kv_cache_config.kv_cache_tensors == original_tensors


def test_no_reuse_skips_multi_component_layer_validation():
    spec = MambaSpec(
        block_size=2,
        shapes=((1,),),
        dtypes=(torch.int8,),
    )
    suffixes = (
        "compressor.state_cache",
        "indexer.k_cache",
        "indexer.compressor.state_cache",
        "swa_cache",
        "attn",
    )
    layer_names = [f"model.layers.{layer}.self_attn.{suffix}" for layer in range(2) for suffix in suffixes]
    original_tensors = [KVCacheTensor(size=16, shared_by=[layer_name]) for layer_name in layer_names]
    kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=original_tensors.copy(),
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(layer_names, spec),
                ),
            )
        ],
    )

    assert apply_layerwise_kv_cache_plan(kv_cache_config, _make_vllm_config(2, 2)) is False
    assert kv_cache_config.kv_cache_tensors == original_tensors


def test_base_layers_are_merged_into_shared_slots():
    spec = _make_full_attention_spec()
    original_tensors = [
        KVCacheTensor(
            size=spec.page_size_bytes,
            shared_by=[f"model.layers.{layer}.self_attn"],
        )
        for layer in range(6)
    ]
    layer_names = [tensor.shared_by[0] for tensor in original_tensors]
    kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=original_tensors,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(layer_names, spec),
                ),
            )
        ],
    )

    apply_layerwise_kv_cache_plan(kv_cache_config, _make_vllm_config(6, 2))

    assert [tensor.shared_by for tensor in kv_cache_config.kv_cache_tensors] == [
        ["model.layers.0.self_attn"],
        ["model.layers.1.self_attn", "model.layers.3.self_attn", "model.layers.5.self_attn"],
        ["model.layers.2.self_attn", "model.layers.4.self_attn"],
    ]


def test_default_layout_keeps_one_buffer_per_layer():
    layout = build_layerwise_cache_layout(27)

    assert layout.has_layer_reuse is False
    assert layout.num_shared_buffers == 27
    assert layout.num_prefetch_layers == 8
    assert layout.independent_layers == [0]
    assert len(layout.storage_indices) == 27


def test_reuse_layout_matches_round_robin_buffer_assignments():
    layout = build_layerwise_cache_layout(27, {"layerwise_num_shared_buffers": 6})

    assert layout.has_layer_reuse is True
    assert layout.prefetch_layer_map[7] == 1
    assert layout.prefetch_layer_map[8] == 2
    assert layout.storage_indices[0] == [0]
    assert layout.storage_indices[1] == [1, 7, 13, 19, 25]
    assert layout.storage_indices[2] == [2, 8, 14, 20, 26]
    assert sorted(layer for slot in layout.storage_indices for layer in slot) == list(range(27))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([3, 5, 10], [3, 5, 10]),
        ([-1], [26]),
        ([1, 4], [1, 4]),
        ("all", list(range(27))),
    ],
)
def test_independent_layer_parsing(value, expected):
    layout = build_layerwise_cache_layout(27, {"layerwise_independent_layers": value})

    assert layout.independent_layers == expected


def test_invalid_layout_config_is_rejected():
    with pytest.raises(TypeError):
        build_layerwise_cache_layout(27, {"layerwise_num_shared_buffers": True})
    with pytest.raises(ValueError):
        build_layerwise_cache_layout(27, {"layerwise_num_shared_buffers": 0})
    with pytest.raises(TypeError):
        build_layerwise_cache_layout(27, {"layerwise_independent_layers": 27})
    with pytest.raises(TypeError):
        build_layerwise_cache_layout(27, {"layerwise_independent_layers": "1,4"})


def test_prefetch_count_can_be_overridden():
    layout = build_layerwise_cache_layout(
        27,
        {
            "layerwise_num_shared_buffers": 6,
            "layerwise_prefetch_layers": 3,
        },
    )

    assert layout.num_prefetch_layers == 3


def test_gva_config_is_scoped_to_memcache_layerwise_connector():
    ascend_store_config = {
        "backend": "memcache",
        "use_layerwise": True,
        "layerwise_num_shared_buffers": 2,
    }
    multi_config = SimpleNamespace(
        kv_connector="MultiConnector",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "OtherConnector",
                    "kv_connector_extra_config": {"use_layerwise": True},
                },
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_connector_extra_config": ascend_store_config,
                },
            ]
        },
    )
    unsupported = SimpleNamespace(
        kv_connector="AscendStoreConnector",
        kv_connector_extra_config={"backend": "mooncake", "use_layerwise": True},
    )

    assert get_gva_layerwise_config(multi_config) is ascend_store_config
    assert get_gva_layerwise_config(unsupported) is None


def test_incompatible_cache_specs_use_separate_slots():
    layer_names = [f"model.layers.{layer}.self_attn" for layer in range(4)]
    first_spec = _make_full_attention_spec()
    incompatible_spec = _make_full_attention_spec(
        num_kv_heads=2,
        head_size=4,
    )
    layer_specs = {layer_name: first_spec for layer_name in layer_names}
    layer_specs[layer_names[2]] = incompatible_spec
    kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=[
            KVCacheTensor(size=layer_specs[layer_name].page_size_bytes, shared_by=[layer_name])
            for layer_name in layer_names
        ],
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=layer_specs,
                ),
            )
        ],
    )

    apply_layerwise_kv_cache_plan(kv_cache_config, _make_vllm_config(4, 1))

    assert [tensor.shared_by for tensor in kv_cache_config.kv_cache_tensors] == [
        [layer_names[0]],
        [layer_names[1], layer_names[3]],
        [layer_names[2]],
    ]


def test_partial_layout_skips_tensor_merge():
    layer_names = [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    ]
    original_tensors = [KVCacheTensor(size=16, shared_by=[layer_name]) for layer_name in layer_names]
    spec = _make_full_attention_spec()
    kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=original_tensors.copy(),
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(layer_names, spec),
                ),
            )
        ],
    )
    vllm_config = _make_vllm_config(4, 1)
    vllm_config.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = []

    apply_layerwise_kv_cache_plan(kv_cache_config, vllm_config)

    assert kv_cache_config.kv_cache_tensors == original_tensors


def test_mtp_layer_does_not_hide_missing_base_layer():
    layer_names = [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
        "model.layers.2.self_attn",
        "model.mtp.0.self_attn",
    ]
    original_tensors = [KVCacheTensor(size=16, shared_by=[layer_name]) for layer_name in layer_names]
    spec = _make_full_attention_spec()
    kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=original_tensors.copy(),
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(layer_names, spec),
                ),
            )
        ],
    )
    vllm_config = _make_vllm_config(4, 1)
    vllm_config.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = []

    assert apply_layerwise_kv_cache_plan(kv_cache_config, vllm_config) is False
    assert kv_cache_config.kv_cache_tensors == original_tensors


def test_wrong_pp_base_layers_do_not_enable_tensor_merge():
    spec = _make_full_attention_spec()
    layer_names = [
        "model.layers.0.self_attn",
        "model.layers.2.self_attn",
        "model.layers.3.self_attn",
        "model.mtp.0.self_attn",
    ]
    original_tensors = [
        KVCacheTensor(
            size=spec.page_size_bytes,
            shared_by=[layer_name],
        )
        for layer_name in layer_names
    ]
    kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=original_tensors.copy(),
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(layer_names, spec),
                ),
            )
        ],
    )
    vllm_config = _make_vllm_config(
        num_layers=2,
        num_shared_buffers=1,
        total_num_layers=4,
        layer_start=2,
    )
    vllm_config.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = []

    assert apply_layerwise_kv_cache_plan(kv_cache_config, vllm_config) is False
    assert kv_cache_config.kv_cache_tensors == original_tensors


def test_layout_includes_mtp_layers():
    spec = _make_full_attention_spec()
    specs = {
        **{f"model.layers.{layer}.self_attn": spec for layer in range(4)},
        "model.mtp.0.self_attn": spec,
    }

    layout = build_layerwise_reuse_layout(
        specs,
        4,
        {"layerwise_num_shared_buffers": 2},
    )

    assert 4 in layout.layer_cache_specs
    assert layout.buffer_slots == ((0,), (1, 3), (2, 4))
    assert layout.prefetch_layer_map == {3: 1, 4: 2}


@pytest.mark.parametrize(
    ("layer_name", "expected"),
    [
        ("model.mtp.0.self_attn", 4),
        ("mtp.layers.0.self_attn", 4),
        ("model.mtp.layers.1.self_attn", 5),
        ("model.layers.4.self_attn", 4),
    ],
)
def test_physical_layer_index_supports_mtp_names(
    layer_name,
    expected,
):
    assert get_layerwise_physical_layer_index(layer_name, 4) == expected


def test_mtp_offset_uses_total_layers_with_pipeline_parallelism():
    spec = _make_full_attention_spec()
    layer_names = [
        "model.layers.2.self_attn.attn",
        "model.layers.3.self_attn.attn",
        "model.mtp.0.self_attn.attn",
    ]
    kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes, shared_by=[name]) for name in layer_names],
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(layer_names, spec),
                ),
            )
        ],
    )
    vllm_config = _make_vllm_config(
        num_layers=2,
        num_shared_buffers=1,
        total_num_layers=4,
        layer_start=2,
    )
    vllm_config.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = []

    layout = build_layerwise_reuse_layout(
        dict.fromkeys(layer_names, spec),
        4,
        vllm_config.kv_transfer_config.kv_connector_extra_config,
    )

    assert apply_layerwise_kv_cache_plan(kv_cache_config, vllm_config) is True
    assert kv_cache_config.kv_cache_tensors[0].shared_by == layer_names
    assert get_layerwise_physical_layer_index(layer_names[0], 4) == 2
    assert get_layerwise_physical_layer_index(layer_names[2], 4) == 4
    assert sorted(layout.layer_cache_specs) == [0, 1, 2]
    assert layout.buffer_slots == ((0, 1, 2),)
    assert layout.prefetch_layer_map == {1: 0, 2: 1}


def test_scheduler_slot_controls_reuse_for_each_present_component():
    main_spec = AscendMLAAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.int8,
        cache_sparse_sfa_c8=True,
    )
    indexer_spec = AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
    )
    specs = {
        **{f"model.layers.{layer}.self_attn.attn": main_spec for layer in range(6)},
        **{f"model.layers.{layer}.self_attn.indexer.k_cache": indexer_spec for layer in (1, 3, 5)},
    }

    layout = build_layerwise_reuse_layout(
        specs,
        6,
        {
            "layerwise_num_shared_buffers": 1,
            "layerwise_independent_layers": [],
        },
    )

    # The scheduler puts every layer in one slot. Each component role then gets
    # its own lane and the indexer lane only binds the layers that own it.
    assert layout.buffer_slots == ((0, 1, 2, 3, 4, 5),)
    assert sorted(len(components) for components in layout.component_lanes.values()) == [
        3,
        6,
    ]
    assert tuple(
        layer
        for layer in layout.buffer_slots[0]
        if any(".indexer." in named_spec.layer_name for named_spec in layout.layer_cache_specs[layer])
    ) == (
        1,
        3,
        5,
    )
    assert layout.prefetch_layer_map == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    assert len(layout.layer_cache_specs[0]) == 1
    assert len(layout.layer_cache_specs[1]) == 2


def test_cache_spec_roles_do_not_depend_on_order():
    main_spec = _make_sfa_main_spec()
    indexer_spec = _make_sfa_indexer_spec()
    indexer_name = "model.layers.0.self_attn.indexer.k_cache"
    main_name = "model.layers.0.self_attn.attn"
    specs = {
        indexer_name: indexer_spec,
        main_name: main_spec,
        "model.layers.1.self_attn.attn": main_spec,
    }

    layout = build_layerwise_reuse_layout(
        specs,
        2,
        {
            "layerwise_num_shared_buffers": 1,
            "layerwise_independent_layers": [],
        },
    )

    assert [named_spec.layer_name for named_spec in layout.layer_cache_specs[0]] == [
        main_name,
        indexer_name,
    ]


def test_single_indexer_spec_is_the_primary_spec():
    indexer_name = "model.layers.0.self_attn.indexer.k_cache"
    layout = build_layerwise_reuse_layout(
        {indexer_name: _make_sfa_indexer_spec()},
        1,
        {"layerwise_num_shared_buffers": 1},
    )

    assert [named_spec.layer_name for named_spec in layout.layer_cache_specs[0]] == [indexer_name]


def test_arbitrary_components_are_planned_independently_per_slot():
    spec = _make_full_attention_spec()
    component_names = {
        layer: [
            f"model.layers.{layer}.self_attn.component0",
            f"model.layers.{layer}.self_attn.component1",
        ]
        for layer in range(4)
    }
    component_names[3].append("model.layers.3.self_attn.component3")
    layer_names = [name for names in component_names.values() for name in names]
    kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes, shared_by=[name]) for name in layer_names],
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(layer_names, spec),
                ),
            )
        ],
    )
    vllm_config = _make_vllm_config(4, 2)
    vllm_config.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = []

    assert apply_layerwise_kv_cache_plan(kv_cache_config, vllm_config) is True

    assert [tensor.shared_by for tensor in kv_cache_config.kv_cache_tensors] == [
        [component_names[0][0], component_names[2][0]],
        [component_names[0][1], component_names[2][1]],
        [component_names[1][0], component_names[3][0]],
        [component_names[1][1], component_names[3][1]],
        [component_names[3][2]],
    ]


def test_multi_layer_slot_without_compatible_components_has_no_reuse():
    first_spec = _make_full_attention_spec()
    second_spec = _make_full_attention_spec(num_kv_heads=2, head_size=4)
    layout = build_layerwise_reuse_layout(
        {
            "model.layers.0.self_attn.attn": first_spec,
            "model.layers.1.self_attn.attn": second_spec,
        },
        2,
        {
            "layerwise_num_shared_buffers": 1,
            "layerwise_independent_layers": [],
        },
    )

    assert layout.buffer_slots == ((0, 1),)
    assert layout.has_layer_reuse is False
    assert layout.prefetch_layer_map == {}
    assert layout.independent_layers == [0, 1]


def test_raw_component_layout_is_stable_and_linear_in_num_blocks():
    spec = _make_sfa_indexer_spec()
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    (page_component,) = get_raw_cache_components(layer_name, spec, num_blocks=1)
    (actual_component,) = get_raw_cache_components(layer_name, spec, num_blocks=3)

    assert actual_component.reuse_key == page_component.reuse_key
    assert actual_component.alignment == page_component.alignment
    assert actual_component.size_bytes == page_component.size_bytes * 3
    assert len(actual_component.views) == len(page_component.views)


def test_actual_tensor_cannot_have_fewer_than_configured_blocks():
    spec = _make_full_attention_spec()
    layer_names = [f"model.layers.{layer}.self_attn.attn" for layer in range(2)]
    kv_cache_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes, shared_by=[name]) for name in layer_names],
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(layer_names, spec),
                ),
            )
        ],
    )
    vllm_config = _make_vllm_config(2, 1)
    vllm_config.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = []

    with pytest.raises(ValueError, match="fewer than the configured minimum"):
        apply_layerwise_kv_cache_plan(kv_cache_config, vllm_config)


def test_actual_tensors_can_have_different_extra_block_counts():
    spec = _make_full_attention_spec()
    layer_names = [f"model.layers.{layer}.self_attn.attn" for layer in range(2)]
    configured_num_blocks = 2
    actual_num_blocks = [3, 4]
    kv_cache_config = SimpleNamespace(
        num_blocks=configured_num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * num_blocks,
                shared_by=[name],
            )
            for name, num_blocks in zip(layer_names, actual_num_blocks, strict=True)
        ],
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(layer_names, spec),
                ),
            )
        ],
    )
    vllm_config = _make_vllm_config(2, 1)
    vllm_config.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = []

    assert apply_layerwise_kv_cache_plan(kv_cache_config, vllm_config) is True
    assert len(kv_cache_config.kv_cache_tensors) == 1
    assert kv_cache_config.kv_cache_tensors[0].size == spec.page_size_bytes * configured_num_blocks


def test_compatible_mixed_indexer_components_bind_one_raw_lane():
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    (bf16_component,) = get_raw_cache_components(
        layer_name,
        _make_unquantized_sfa_indexer_spec(),
        num_blocks=2,
    )
    (c8_component,) = get_raw_cache_components(
        "model.layers.1.self_attn.indexer.k_cache",
        _make_sfa_indexer_spec(),
        num_blocks=2,
    )
    raw = torch.zeros(
        max(bf16_component.size_bytes, c8_component.size_bytes),
        dtype=torch.int8,
    )

    bf16_views = bf16_component.bind(raw)
    c8_views = c8_component.bind(raw)

    assert bf16_component.reuse_key == c8_component.reuse_key
    assert len(bf16_views) == 1
    assert len(c8_views) == 2
    assert bf16_views[0].untyped_storage().data_ptr() == c8_views[0].untyped_storage().data_ptr()
    assert c8_views[0].untyped_storage().data_ptr() == c8_views[1].untyped_storage().data_ptr()


def test_multi_group_sfa_descriptors_are_merged_by_main_component():
    main_names = [
        *(f"model.layers.{layer}.self_attn.attn" for layer in range(4)),
        "model.mtp.0.self_attn.attn",
    ]
    indexer_names = [f"model.layers.{layer}.self_attn.indexer.k_cache" for layer in (1, 2, 4)]
    main_spec = AscendMLAAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
    )
    indexer_spec = AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=[
            *(KVCacheTensor(size=main_spec.page_size_bytes, shared_by=[name]) for name in main_names),
            *(KVCacheTensor(size=indexer_spec.page_size_bytes, shared_by=[name]) for name in indexer_names),
        ],
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=main_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(main_names, main_spec),
                ),
            ),
            SimpleNamespace(
                layer_names=indexer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(indexer_names, indexer_spec),
                ),
            ),
        ],
    )

    apply_layerwise_kv_cache_plan(
        kv_cache_config,
        _make_vllm_config(4, 1),
    )

    # One independent main tensor, one main tensor shared by every reused layer (incl.
    # MTP), and one indexer tensor shared only by the indexer-bearing layers.
    assert [tensor.shared_by for tensor in kv_cache_config.kv_cache_tensors] == [
        [main_names[0]],
        [main_names[1], main_names[2], main_names[3], main_names[4]],
        indexer_names,
    ]


def _make_sfa_main_spec(dtype=torch.int8) -> AscendMLAAttentionSpec:
    return AscendMLAAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=8,
        dtype=dtype,
        cache_sparse_sfa_c8=True,
    )


def _make_sfa_indexer_spec() -> AscendSFAIndexerCacheSpec:
    return AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
    )


def _make_unquantized_sfa_indexer_spec() -> AscendSFAIndexerCacheSpec:
    return AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.bfloat16,
        cache_sparse_li_c8=False,
    )


def test_mixed_li_c8_indexers_share_one_buffer_per_main_slot():
    main_spec = _make_sfa_main_spec()
    c8_spec = _make_sfa_indexer_spec()
    bf16_spec = _make_unquantized_sfa_indexer_spec()
    main_by_layer = {layer: f"model.layers.{layer}.self_attn.attn" for layer in range(6)}
    indexer_by_layer = {layer: f"model.layers.{layer}.self_attn.indexer.k_cache" for layer in range(6)}
    indexer_specs = {indexer_by_layer[layer]: c8_spec if layer % 2 == 0 else bf16_spec for layer in range(6)}
    num_blocks = 2
    kv_cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            *(
                KVCacheTensor(size=main_spec.page_size_bytes * num_blocks, shared_by=[name])
                for name in main_by_layer.values()
            ),
            *(
                KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=[name])
                for name, spec in indexer_specs.items()
            ),
        ],
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=list(main_by_layer.values()),
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(main_by_layer.values(), main_spec),
                ),
            ),
            SimpleNamespace(
                layer_names=list(indexer_by_layer.values()),
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=indexer_specs,
                ),
            ),
        ],
    )
    vllm_config = _make_vllm_config(6, 3)
    vllm_config.kv_transfer_config.kv_connector_extra_config["layerwise_independent_layers"] = []

    apply_layerwise_kv_cache_plan(kv_cache_config, vllm_config)

    main_tensors = [
        tensor
        for tensor in kv_cache_config.kv_cache_tensors
        if not any(".indexer." in name for name in tensor.shared_by)
    ]
    indexer_tensors = [
        tensor for tensor in kv_cache_config.kv_cache_tensors if any(".indexer." in name for name in tensor.shared_by)
    ]
    assert len(main_tensors) == 3
    assert len(indexer_tensors) == 3
    assert [tensor.shared_by for tensor in indexer_tensors] == [
        [indexer_by_layer[0], indexer_by_layer[3]],
        [indexer_by_layer[1], indexer_by_layer[4]],
        [indexer_by_layer[2], indexer_by_layer[5]],
    ]
    assert all(tensor.size == bf16_spec.page_size_bytes * num_blocks for tensor in indexer_tensors)


def test_component_sharing_merges_main_across_a_and_b_layers():
    # GLM5.2/SFA: A-class layers own main + indexer, B-class layers own main only. Every
    # main spec is identical, so one buffer's main tensor is shared by all layers in the
    # buffer while its indexer tensor is shared only by the buffer's A-class layers.
    main_spec = _make_sfa_main_spec()
    indexer_spec = _make_sfa_indexer_spec()
    a_layers = [1, 4]  # main + indexer
    b_layers = [2, 3, 5]  # main only
    # physical layer 0 is independent; layers 1..5 are reused; 6 = MTP (main only).
    main_by_layer = {
        0: "model.layers.0.self_attn.attn",
        **{layer: f"model.layers.{layer}.self_attn.attn" for layer in (*a_layers, *b_layers)},
        6: "model.mtp.0.self_attn.attn",
    }
    indexer_by_layer = {layer: f"model.layers.{layer}.self_attn.indexer.k_cache" for layer in a_layers}
    kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=[
            *(KVCacheTensor(size=main_spec.page_size_bytes, shared_by=[name]) for name in main_by_layer.values()),
            *(KVCacheTensor(size=indexer_spec.page_size_bytes, shared_by=[name]) for name in indexer_by_layer.values()),
        ],
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=list(main_by_layer.values()),
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(main_by_layer.values(), main_spec),
                ),
            ),
            SimpleNamespace(
                layer_names=list(indexer_by_layer.values()),
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(indexer_by_layer.values(), indexer_spec),
                ),
            ),
        ],
    )

    apply_layerwise_kv_cache_plan(kv_cache_config, _make_vllm_config(6, 2))

    main_shared_by = []
    indexer_shared_by = []
    for tensor in kv_cache_config.kv_cache_tensors:
        names = list(tensor.shared_by)
        if any(".indexer." in name for name in names):
            indexer_shared_by.append(names)
        else:
            main_shared_by.append(names)

    # 1 independent layer + 2 reused buffers == 3 main tensors.
    assert len(main_shared_by) == 3
    # The independent layer keeps its own main; every reused layer's main (incl. MTP)
    # lands in exactly one shared main tensor.
    assert main_shared_by[0] == [main_by_layer[0]]
    merged_reused_main = sorted(name for names in main_shared_by[1:] for name in names)
    assert merged_reused_main == sorted(main_by_layer[layer] for layer in (1, 2, 3, 4, 5, 6))
    # Indexer layers follow their main slots, so layers 1 and 4 use separate tensors.
    assert indexer_shared_by == [[indexer_by_layer[1]], [indexer_by_layer[4]]]


def test_non_attention_cache_spec_is_rejected():
    mamba_spec = MambaSpec(
        block_size=2,
        shapes=((1,),),
        dtypes=(torch.int8,),
    )
    layer_specs = {f"model.layers.{layer}.mixer": mamba_spec for layer in range(3)}

    with pytest.raises(NotImplementedError, match="attention cache specs only"):
        build_layerwise_reuse_layout(
            layer_specs,
            3,
            {"layerwise_num_shared_buffers": 1},
        )


def test_packed_cache_tensor_descriptors_are_rejected():
    layer_names = [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
        "model.layers.2.self_attn",
    ]
    spec = _make_full_attention_spec()
    kv_cache_config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=[
            KVCacheTensor(
                size=16,
                shared_by=[layer_name],
                offset=8,
                block_stride=32,
            )
            for layer_name in layer_names
        ],
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs=dict.fromkeys(layer_names, spec),
                ),
            )
        ],
    )

    with pytest.raises(NotImplementedError, match="pre-shared or packed"):
        apply_layerwise_kv_cache_plan(
            kv_cache_config,
            _make_vllm_config(3, 1),
        )
