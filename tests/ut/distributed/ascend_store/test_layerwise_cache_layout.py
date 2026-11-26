# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
    LayerwiseCachePlanner,
    get_layerwise_offload_config,
    plan_layerwise_cache_reuse,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LayerwiseCacheConfig,
    allocate_cache_backing,
    apply_layerwise_kv_cache_plan,
    get_cache_views,
    get_layerwise_reuse_layout,
    get_physical_layer_index,
)


def _native_config(groups, num_blocks=3):
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * group.kv_cache_spec.page_size_bytes,
                layers=[name],
                layer_stride=0,
                block_stride=group.kv_cache_spec.page_size_bytes,
            )
            for group in groups
            for name in group.layer_names
        ],
    )


def _runtime(num_layers, num_buffers, model_type="glm5_next_text", **extra):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            get_total_num_hidden_layers=lambda: num_layers,
            hf_text_config=SimpleNamespace(model_type=model_type),
        ),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        kv_transfer_config=KVTransferConfig(
            kv_connector="AscendStoreConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": True,
                "layerwise_num_shared_buffers": num_buffers,
                **extra,
            },
        ),
    )


def _plan(config, runtime, rules):
    return LayerwiseCachePlanner(
        config, runtime, runtime.kv_transfer_config.kv_connector_extra_config, spec_reuse_rules=rules
    ).build()


def _attention_spec(dtype=torch.int8, head_size=8):
    return FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=head_size, head_size_v=head_size, dtype=dtype)


def test_default_layout_keeps_native_buffers():
    settings = LayerwiseCacheConfig.parse(27)
    assert settings.num_shared_buffers is None
    assert settings.num_prefetch_layers == 8
    assert settings.independent_layers == [0]
    names = [f"model.layers.{index}.self_attn" for index in range(27)]
    config = _native_config([KVCacheGroupSpec(names, _attention_spec())])
    assert plan_layerwise_cache_reuse(config, _runtime(27, None)) is None


def test_reuse_layout_matches_round_robin_buffer_assignments():
    names = [f"model.layers.{index}.self_attn" for index in range(27)]
    config = _native_config([KVCacheGroupSpec(names, _attention_spec())])
    runtime = _runtime(27, 6)
    layout = _plan(config, runtime, {FullAttentionSpec: True})
    assert layout.prefetch_layer_map[7] == (1,)
    assert layout.prefetch_layer_map[8] == (2,)
    assert [tensor.layers for tensor in layout.kv_cache_tensors][:3] == [[names[0]], names[1::6], names[2::6]]
    assert sorted(name for tensor in layout.kv_cache_tensors for name in tensor.layers) == sorted(names)


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
    layout = LayerwiseCacheConfig.parse(27, {"layerwise_independent_layers": value})

    assert layout.independent_layers == expected


def test_invalid_layout_config_is_rejected():
    with pytest.raises(TypeError):
        LayerwiseCacheConfig.parse(27, {"layerwise_num_shared_buffers": True})
    with pytest.raises(ValueError):
        LayerwiseCacheConfig.parse(27, {"layerwise_num_shared_buffers": 0})
    with pytest.raises(TypeError):
        LayerwiseCacheConfig.parse(27, {"layerwise_independent_layers": 27})
    with pytest.raises(TypeError):
        LayerwiseCacheConfig.parse(27, {"layerwise_independent_layers": "1,4"})


def test_prefetch_count_can_be_overridden():
    layout = LayerwiseCacheConfig.parse(
        27,
        {
            "layerwise_num_shared_buffers": 6,
            "layerwise_prefetch_layers": 3,
        },
    )

    assert layout.num_prefetch_layers == 3


def test_reuse_config_is_scoped_to_layerwise_backend_connector():
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

    assert get_layerwise_offload_config(multi_config) is ascend_store_config
    for connector_name in (
        "AscendStoreConnector",
        "MooncakeConnectorStoreV1",
    ):
        direct_config = SimpleNamespace(
            kv_connector=connector_name,
            kv_connector_extra_config=ascend_store_config,
        )
        assert get_layerwise_offload_config(direct_config) is ascend_store_config
    assert get_layerwise_offload_config(unsupported) is None
    assert get_layerwise_offload_config(not_opted_in) is None


@pytest.mark.parametrize(
    ("layer_name", "expected"),
    [
        ("model.mtp.0.self_attn", 4),
        ("mtp.layers.0.self_attn", 4),
        ("model.mtp.layers.1.self_attn", 5),
        ("model.layers.4.self_attn", 4),
        ("notlayers.2.model.layers.3.attn", 2),
    ],
)
def test_physical_layer_index_supports_mtp_names(
    layer_name,
    expected,
):
    assert get_physical_layer_index(layer_name, 4) == expected


def test_pp_stage_uses_global_independent_layers():
    names = [f"model.layers.{index}.attn" for index in range(4, 8)]
    config = _native_config([KVCacheGroupSpec(names, _attention_spec())])
    runtime = _runtime(8, 1, layerwise_independent_layers=[0, 6])
    layout = _plan(config, runtime, {FullAttentionSpec: True})
    assert layout.layers == (4, 5, 6, 7)
    assert layout.independent_layers == [6]
    assert layout.prefetch_layer_map == {5: (4,), 7: (5,)}


@pytest.mark.parametrize("num_buffers", [None, 6])
@pytest.mark.parametrize("model_type", ["glm5_next_text", "deepseek_v4"])
def test_no_reuse_preserves_native_descriptors(num_buffers, model_type):
    state = MambaSpec(block_size=2, shapes=((4,),), dtypes=(torch.int8,), mamba_cache_mode="align")
    names = [f"model.layers.{index}.state" for index in range(6)]
    config = _native_config([KVCacheGroupSpec(names, state)])
    original_tensors = tuple(config.kv_cache_tensors)
    assert plan_layerwise_cache_reuse(config, _runtime(6, num_buffers, model_type)) is None
    assert tuple(config.kv_cache_tensors) == original_tensors


@pytest.mark.parametrize("num_blocks", [1, 3])
def test_registered_glm_reuse_preserves_resident_components_and_native_views(num_blocks):
    names = [f"model.layers.{index}.self_attn.attn" for index in range(4)]
    indexer_names = [f"model.layers.{index}.self_attn.indexer.k_cache" for index in (1, 2, 3)]
    draft = "model.mtp.0.self_attn.attn"
    main = AscendMLAAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, model_version="glm5_next"
    )
    indexer = replace(main, head_size=4, tokens_per_state=2)
    config = _native_config(
        [
            KVCacheGroupSpec(names + [draft], main),
            KVCacheGroupSpec(indexer_names, indexer),
        ],
        num_blocks,
    )
    runtime = _runtime(4, 1)
    layout = plan_layerwise_cache_reuse(config, runtime)
    assert layout is not None
    assert layout.draft_entry_names == {draft}
    assert layout.reused_cache_names == set(names[1:])
    assert layout.prefetch_layer_map == {2: (1,), 3: (2,)}
    assert set(layout.cache_view_sizes) == set(names + indexer_names + [draft])
    specs = dict.fromkeys(names + [draft], main) | dict.fromkeys(indexer_names, indexer)
    native_views = get_cache_views(config, specs)
    assert layout.cache_view_sizes == {name: end - start for name, (_, start, end) in native_views.items()}
    worker_config = apply_layerwise_kv_cache_plan(config, runtime.kv_transfer_config, layout)
    assert get_layerwise_reuse_layout(worker_config) is layout
    assert get_layerwise_reuse_layout(config) is None
    allocated_sizes: list[int] = []

    def allocate(size):
        allocated_sizes.append(size)
        return torch.empty(size, dtype=torch.uint8)

    tensors = allocate_cache_backing(worker_config, specs, allocate, layout.cache_view_sizes)
    # Independent layer, one rotating slot, and the persistent MTP cache.
    expected_bytes = num_blocks * (3 * main.page_size_bytes + 3 * indexer.page_size_bytes)
    assert allocated_sizes == [expected_bytes]
    assert tensors[names[1]] is tensors[names[2]] is tensors[names[3]]
    offsets = {tensors[name].data_ptr() for name in [names[0], names[1], draft, *indexer_names]}
    assert len(offsets) == 6


def test_layout_resolves_two_source_dependencies_on_one_target_layer():
    attention = _attention_spec()
    state = MambaSpec(block_size=2, shapes=((attention.page_size_bytes,),), dtypes=(torch.int8,))
    attn_names = ["model.layers.0.attn", "model.layers.2.attn"]
    state_names = ["model.layers.1.state", "model.layers.2.state"]
    config = _native_config([KVCacheGroupSpec(attn_names, attention), KVCacheGroupSpec(state_names, state)])
    layout = _plan(config, _runtime(3, 1, layerwise_independent_layers=[]), {FullAttentionSpec: True, MambaSpec: True})
    assert layout.prefetch_layer_map == {2: (0, 1)}
    assert set(layout.kv_cache_tensors[0].layers) == set(attn_names + state_names)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"size": 1}, "exceeds its backing"),
        ({"block_stride": 0}, "positive block strides"),
        ({"offset": 1}, "is not aligned"),
    ],
)
def test_invalid_native_view_is_rejected(changes, message):
    names = [f"model.layers.{index}.attn" for index in range(3)]
    config = _native_config([KVCacheGroupSpec(names, _attention_spec(torch.float16))])
    config.kv_cache_tensors[-1] = replace(config.kv_cache_tensors[-1], **changes)
    with pytest.raises(ValueError, match=message):
        _plan(config, _runtime(3, 1), {FullAttentionSpec: True})
