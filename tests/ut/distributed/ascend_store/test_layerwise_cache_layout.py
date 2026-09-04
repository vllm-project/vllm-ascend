from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheTensor, MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    build_layerwise_cache_layout,
    build_layerwise_reuse_layout,
    get_layerwise_physical_layer_index,
    get_layerwise_reuse_config,
)
from vllm_ascend.utils import vllm_version_is


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


def _make_vllm_config(num_layers: int, num_shared_buffers: int):
    model_config = MagicMock()
    model_config.get_num_layers.return_value = num_layers
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


@pytest.mark.skipif(
    not vllm_version_is("0.27.1"),
    reason="layerwise tensor merge is disabled on the standardized KV cache layout",
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

    apply_layerwise_kv_cache_plan(kv_cache_config, _make_vllm_config(2, 2))

    assert kv_cache_config.kv_cache_tensors == original_tensors


@pytest.mark.skipif(
    not vllm_version_is("0.27.1"),
    reason="layerwise tensor merge is disabled on the standardized KV cache layout",
)
def test_base_layers_are_merged_into_shared_slots():
    original_tensors = [KVCacheTensor(size=16, shared_by=[f"model.layers.{layer}.self_attn"]) for layer in range(6)]
    layer_names = [tensor.shared_by[0] for tensor in original_tensors]
    spec = _make_full_attention_spec()
    kv_cache_config = SimpleNamespace(
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


def test_reuse_config_is_scoped_to_layerwise_protocol_connector():
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
    not_opted_in = SimpleNamespace(
        kv_connector="AscendStoreConnector",
        kv_connector_extra_config={"backend": "memcache"},
    )

    assert get_layerwise_reuse_config(multi_config) is ascend_store_config
    assert get_layerwise_reuse_config(unsupported) is None
    assert get_layerwise_reuse_config(not_opted_in) is None


@pytest.mark.skipif(
    not vllm_version_is("0.27.1"),
    reason="layerwise tensor merge is disabled on the standardized KV cache layout",
)
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
        kv_cache_tensors=[KVCacheTensor(size=32, shared_by=[layer_name]) for layer_name in layer_names],
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


@pytest.mark.skipif(
    not vllm_version_is("0.27.1"),
    reason="layerwise tensor merge is disabled on the standardized KV cache layout",
)
def test_partial_layout_skips_tensor_merge():
    layer_names = [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    ]
    original_tensors = [KVCacheTensor(size=16, shared_by=[layer_name]) for layer_name in layer_names]
    spec = _make_full_attention_spec()
    kv_cache_config = SimpleNamespace(
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


def test_main_spec_controls_reuse_regardless_of_indexer():
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

    # Identical main specs put every layer in one buffer; the indexer subset rides along.
    assert layout.buffer_slots == ((0, 1, 2, 3, 4, 5),)
    assert tuple(layer for layer in layout.buffer_slots[0] if layout.layer_cache_specs[layer].indexer is not None) == (
        1,
        3,
        5,
    )
    assert layout.prefetch_layer_map == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    assert layout.layer_cache_specs[0].indexer is None
    assert layout.layer_cache_specs[1].indexer is not None


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

    assert layout.layer_cache_specs[0].main.layer_name == main_name
    assert layout.layer_cache_specs[0].indexer is not None
    assert layout.layer_cache_specs[0].indexer.layer_name == indexer_name


def test_single_indexer_spec_is_the_primary_spec():
    indexer_name = "model.layers.0.self_attn.indexer.k_cache"
    layout = build_layerwise_reuse_layout(
        {indexer_name: _make_sfa_indexer_spec()},
        1,
        {"layerwise_num_shared_buffers": 1},
    )

    assert layout.layer_cache_specs[0].main.layer_name == indexer_name
    assert layout.layer_cache_specs[0].indexer is None


def test_ambiguous_multi_spec_layer_is_rejected():
    main_spec = _make_sfa_main_spec()
    specs = {
        "model.layers.0.self_attn.attn": main_spec,
        "model.layers.0.self_attn.other_cache": main_spec,
    }

    with pytest.raises(ValueError, match="multiple cache specs"):
        build_layerwise_reuse_layout(
            specs,
            1,
            {"layerwise_num_shared_buffers": 1},
        )


@pytest.mark.skipif(
    not vllm_version_is("0.27.1"),
    reason="layerwise tensor merge is disabled on the standardized KV cache layout",
)
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


@pytest.mark.skipif(
    not vllm_version_is("0.27.1"),
    reason="layerwise tensor merge is disabled on the standardized KV cache layout",
)
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


@pytest.mark.skipif(
    not vllm_version_is("0.27.1"),
    reason="layerwise tensor merge is disabled on the standardized KV cache layout",
)
def test_packed_cache_tensor_descriptors_are_rejected():
    layer_names = [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
        "model.layers.2.self_attn",
    ]
    spec = _make_full_attention_spec()
    kv_cache_config = SimpleNamespace(
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
