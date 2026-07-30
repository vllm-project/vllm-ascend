from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.kv_cache_interface import KVCacheTensor

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    build_layerwise_cache_layout,
    get_gva_layerwise_config,
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


def test_no_reuse_skips_topology_validation():
    original_tensors = [
        KVCacheTensor(size=16, shared_by=["model.layers.0.self_attn"]),
        KVCacheTensor(size=16, shared_by=["model.layers.1.self_attn"]),
        KVCacheTensor(size=16, shared_by=["model.mtp.0.self_attn"]),
    ]
    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=original_tensors.copy(),
        kv_cache_groups=[object(), object()],
    )

    apply_layerwise_kv_cache_plan(kv_cache_config, _make_vllm_config(2, 2))

    assert kv_cache_config.kv_cache_tensors == original_tensors


def test_base_layers_are_merged_into_shared_slots():
    original_tensors = [KVCacheTensor(size=16, shared_by=[f"model.layers.{layer}.self_attn"]) for layer in range(6)]
    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=original_tensors,
        kv_cache_groups=[object()],
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


def test_reuse_layout_matches_round_robin_storage_slots():
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
    with pytest.raises(ValueError):
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
