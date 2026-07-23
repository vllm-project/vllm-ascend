from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_config import (
    _DEFAULT_MAX_PREFETCH_LAYERS,
    get_gva_layerwise_config,
    get_layer_load_start_block,
    get_layerwise_config,
    get_layerwise_independent_layers,
    get_layerwise_kv_cache_num_tensors,
    get_layerwise_num_prefetch_layers,
    get_layerwise_num_shared_buffers,
    get_layerwise_storage_indices,
)


def test_default_config_keeps_one_buffer_per_layer():
    config = get_layerwise_config(27)

    assert config.has_layer_reuse is False
    assert config.num_shared_buffers == 27
    assert config.independent_layers == [0, 26]


def test_reuse_plan_matches_round_robin_storage_slots():
    extra_config = {"layerwise_num_shared_buffers": 6}
    config = get_layerwise_config(27, extra_config)
    storage_indices = get_layerwise_storage_indices(27, extra_config)

    assert config.has_layer_reuse is True
    assert config.prefetch_layer_map[7] == 1
    assert config.prefetch_layer_map[8] == 2
    assert storage_indices[:2] == [[0], [26]]
    assert storage_indices[2] == [1, 7, 13, 19, 25]
    assert sorted(layer for slot in storage_indices for layer in slot) == list(range(27))
    assert get_layerwise_kv_cache_num_tensors(27, extra_config) == 8


def test_mtp_layer_is_included_in_reuse_decision():
    extra_config = {"layerwise_num_shared_buffers": 2}

    assert get_layerwise_config(4, extra_config).has_layer_reuse is False
    assert get_layerwise_config(5, extra_config).has_layer_reuse is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [("3,5,10", [3, 5, 10]), ("-1", [26]), ([1, 4], [1, 4]), ("all", list(range(27)))],
)
def test_independent_layer_parsing(value, expected):
    assert get_layerwise_independent_layers(27, {"layerwise_independent_layers": value}) == expected


def test_invalid_layerwise_config_is_rejected():
    with pytest.raises(TypeError):
        get_layerwise_num_shared_buffers(27, {"layerwise_num_shared_buffers": True})
    with pytest.raises(ValueError):
        get_layerwise_num_shared_buffers(27, {"layerwise_num_shared_buffers": 0})
    with pytest.raises(ValueError):
        get_layerwise_independent_layers(27, {"layerwise_independent_layers": 27})


def test_prefetch_default_is_bounded_and_can_be_overridden():
    assert get_layerwise_num_prefetch_layers(6) == 6
    assert get_layerwise_num_prefetch_layers(100) == _DEFAULT_MAX_PREFETCH_LAYERS
    assert get_layerwise_num_prefetch_layers(6, {"layerwise_prefetch_layers": 3}) == 3


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


def test_shared_layers_reload_full_prefix_but_independent_layers_do_not():
    independent_layers = [0, 26]

    assert get_layer_load_start_block(0, independent_layers, 32, 16, True) == 2
    assert get_layer_load_start_block(5, independent_layers, 32, 16, True) == 0
    assert get_layer_load_start_block(5, independent_layers, 32, 16, False) == 2
