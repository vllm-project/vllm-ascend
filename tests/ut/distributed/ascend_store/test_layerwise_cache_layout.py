# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.layerwise_cache_config import (
    get_layerwise_layout,
    kv_cache_allocation_bytes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    build_layerwise_cache_layout,
    get_layerwise_policy,
    get_layerwise_reuse_config,
    get_physical_layer_index,
    infer_layerwise_reuse_layout,
    validate_layerwise_reuse_layout,
)


def spec(reuse=True, **kwargs):
    return FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=8,
        head_size_v=8,
        dtype=torch.int8,
        reuse=reuse,
        **kwargs,
    )


def runtime(buffers=2, independent=()):
    return SimpleNamespace(
        model_config=SimpleNamespace(get_num_layers=lambda _: 6),
        parallel_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": True,
                "layerwise_num_shared_buffers": buffers,
                "layerwise_independent_layers": list(independent),
            },
        ),
    )


def cache(specs, domains=None, n=3):
    if domains is None:
        domains = [list(specs)]
    tensors = []
    for names in domains:
        page = specs[names[0]].page_size_bytes
        tensors.append(
            KVCacheTensor(
                size=n * page * len(names),
                layers=names,
                layer_stride=n * page,
                block_stride=page,
                offset=0,
            )
        )
    return KVCacheConfig(
        n,
        tensors,
        [
            KVCacheGroupSpec(
                list(specs),
                UniformTypeKVCacheSpecs(block_size=2, kv_cache_specs=specs),
            )
        ],
    )


def names(count=6, suffix="attn"):
    return [f"model.layers.{i}.{suffix}" for i in range(count)]


def test_independent_tensors_never_merge_even_with_identical_specs():
    members = names()
    config = cache(dict.fromkeys(members, spec()), [[name] for name in members])
    before = deepcopy(config.kv_cache_tensors)
    apply_layerwise_kv_cache_plan(config, runtime())
    assert config.kv_cache_tensors == before
    assert not get_layerwise_layout(config).has_layer_reuse


def test_contiguous_layer_regions_rotate_inside_original_tensor():
    members = names()
    config = cache(dict.fromkeys(members, spec()))
    apply_layerwise_kv_cache_plan(config, runtime())
    assert [t.layers for t in config.kv_cache_tensors] == [members[::2], members[1::2]]
    assert all(t.layer_stride == 0 and t.size == 3 * 32 for t in config.kv_cache_tensors)
    assert get_layerwise_layout(config).predecessors == {2: (0,), 4: (2,), 3: (1,), 5: (3,)}


def test_mixed_resident_region_is_retained_and_not_added_to_rotation():
    members = names()
    specs = dict.fromkeys(members, spec())
    specs[members[2]] = spec(False)
    config = cache(specs)
    apply_layerwise_kv_cache_plan(config, runtime())
    assert next(t for t in config.kv_cache_tensors if members[2] in t.layers).layers == [members[2]]
    assert not specs[members[2]].reuse
    assert get_layerwise_layout(config).resident_layer_names == frozenset([members[2]])


def test_independent_storage_domains_have_independent_slot_sequences():
    main, extra = names(), names(suffix="extra")
    config = cache(dict.fromkeys(main + extra, spec()), [main, extra])
    apply_layerwise_kv_cache_plan(config, runtime())
    assert [t.layers for t in config.kv_cache_tensors] == [main[::2], main[1::2], extra[::2], extra[1::2]]


@pytest.mark.parametrize("different_groups", [False, True])
def test_different_original_offsets_keep_separate_regions_on_the_same_backing(different_groups):
    members = names()
    config = cache(dict.fromkeys(members, spec()), [members[:3], members[3:]])
    if different_groups:
        config.kv_cache_groups = [KVCacheGroupSpec(members[:3], spec()), KVCacheGroupSpec(members[3:], spec())]
    for index, tensor in enumerate(config.kv_cache_tensors):
        tensor.backing_id = "shared_bank"
        tensor.size = 6 * 3 * 32
        tensor.offset = index * 3 * 3 * 32
    apply_layerwise_kv_cache_plan(config, runtime())
    assert [t.layers for t in config.kv_cache_tensors] == [
        [members[0], members[2]],
        [members[1]],
        [members[3], members[5]],
        [members[4]],
    ]
    assert {t.size for t in config.kv_cache_tensors} == {4 * 3 * 32}
    assert len({t.offset for t in config.kv_cache_tensors}) == 4
    assert {t.backing_id for t in config.kv_cache_tensors} == {"shared_bank"}


def test_cross_group_descriptors_with_different_layer_strides_keep_separate_regions():
    # Both groups have a member in each logical buffer.
    members = names(4)
    first, second = members[:2], members[2:]
    config = cache(dict.fromkeys(first + second, spec()), [first, second])
    config.kv_cache_groups = [KVCacheGroupSpec(first, spec()), KVCacheGroupSpec(second, spec())]
    for index, tensor in enumerate(config.kv_cache_tensors):
        tensor.backing_id = "bank"
        tensor.layer_stride *= index + 1
        tensor.size = 3 * 3 * 32
    before = deepcopy(config.kv_cache_tensors)
    apply_layerwise_kv_cache_plan(config, runtime())
    assert config.kv_cache_tensors == before
    assert all(len(slot) == 1 for slot in get_layerwise_layout(config).buffer_slots)


def test_resident_regions_coalesce_without_aliasing_or_changing_intent():
    rotating, resident = names(), names(suffix="history")
    config = cache({**dict.fromkeys(rotating, spec()), **dict.fromkeys(resident, spec(False))}, [rotating, resident])
    # These components have independent group lifetimes. Their original
    # storage regions are disjoint within one explicitly identified backing.
    config.kv_cache_groups = [
        KVCacheGroupSpec(rotating, spec()),
        KVCacheGroupSpec(resident, spec(False)),
    ]
    for index, tensor in enumerate(config.kv_cache_tensors):
        tensor.backing_id = "bank"
        tensor.size = 12 * 3 * 32
        tensor.offset = index * 6 * 3 * 32
    apply_layerwise_kv_cache_plan(config, runtime())
    tensors = config.kv_cache_tensors
    assert len(tensors) == 3
    assert tensors[0].layers == resident
    assert tensors[0].layer_stride == 3 * 32
    assert tensors[0].block_stride == 32
    assert tensors[1].offset >= tensors[0].offset + len(resident) * tensors[0].layer_stride
    assert get_layerwise_layout(config).resident_layer_names == frozenset(resident)
    before = deepcopy(config)
    apply_layerwise_kv_cache_plan(config, runtime())
    assert config == before


def test_same_size_different_backing_ids_remain_independent():
    main, extra = names(), names(suffix="extra")
    config = cache(dict.fromkeys(main + extra, spec()), [main, extra])
    for index, tensor in enumerate(config.kv_cache_tensors):
        tensor.backing_id = str(index)
    apply_layerwise_kv_cache_plan(config, runtime())
    assert len(config.kv_cache_tensors) == 4
    assert [t.backing_id for t in config.kv_cache_tensors] == ["0", "0", "1", "1"]
    assert kv_cache_allocation_bytes(config.kv_cache_tensors) == 2 * 2 * 3 * 32


def test_cross_group_slot_waits_for_previous_layer_in_its_buffer():
    count, buffers = 6, 2
    members = names(count)
    config = cache(dict.fromkeys(members, spec()))
    config.kv_cache_tensors[0].backing_id = "shared_pool"
    config.kv_cache_groups = [KVCacheGroupSpec([name], spec()) for name in members]
    groups_before = deepcopy(config.kv_cache_groups)
    apply_layerwise_kv_cache_plan(config, runtime(buffers=buffers))
    layout = get_layerwise_layout(config)
    assert layout.has_layer_reuse
    assert layout.predecessors == {i: (i - buffers,) for i in range(buffers, count)}
    assert [t.layers for t in config.kv_cache_tensors] == [members[i::buffers] for i in range(buffers)]
    assert [t.offset for t in config.kv_cache_tensors] == [i * 3 * 32 for i in range(buffers)]
    assert kv_cache_allocation_bytes(config.kv_cache_tensors) == buffers * 3 * 32
    assert config.kv_cache_groups == groups_before
    before = deepcopy(config)
    apply_layerwise_kv_cache_plan(config, runtime(buffers=buffers))
    assert config == before


@pytest.mark.parametrize("invalid", ["capacity", "partial_overlap", "same_group_alias"])
def test_invalid_shared_geometry_is_rejected_atomically(invalid):
    members = names()
    config = cache(dict.fromkeys(members, spec()), [members[:3], members[3:]])
    first, second = config.kv_cache_tensors
    first.backing_id = second.backing_id = "bank"
    first.size = second.size = 6 * 3 * 32
    second.offset = 3 * 3 * 32
    if invalid == "capacity":
        second.size += 32
    elif invalid == "partial_overlap":
        second.offset = 16
    else:
        second.offset = 0
    before = deepcopy(config)
    with pytest.raises((ValueError, NotImplementedError)):
        apply_layerwise_kv_cache_plan(config, runtime())
    assert config == before


def test_explicit_packed_state_retains_component_stride():
    members = names(suffix="state")
    state = MambaSpec(block_size=2, shapes=((4,), (2, 2)), dtypes=(torch.int16, torch.float32), reuse=True)
    config = cache(dict.fromkeys(members, state))
    tensor = config.kv_cache_tensors[0]
    tensor.backing_id = "state_bank"
    tensor.block_stride = 32
    tensor.layer_stride = 3 * 32
    tensor.size = 6 * tensor.layer_stride
    apply_layerwise_kv_cache_plan(config, runtime())
    assert [t.layers for t in config.kv_cache_tensors] == [members[::2], members[1::2]]
    assert all(t.block_stride == 32 and t.size == 2 * 3 * 32 for t in config.kv_cache_tensors)


def test_packed_attention_uses_physical_geometry_not_padded_group_page():
    members = names()
    config = cache(dict.fromkeys(members, spec(page_size_padded=64)))
    tensor = config.kv_cache_tensors[0]
    tensor.backing_id = "bank"
    tensor.block_stride = 32
    tensor.layer_stride = 3 * 32
    tensor.size = 6 * tensor.layer_stride
    apply_layerwise_kv_cache_plan(config, runtime())
    assert all(t.block_stride == 32 and t.size == 2 * 3 * 32 for t in config.kv_cache_tensors)


@pytest.mark.parametrize("shared", [False, True])
def test_equal_byte_size_different_dtype_is_not_layout_compatibility(shared):
    members = names()
    specs = {name: spec() for name in members}
    for name in members[2:4]:
        specs[name] = replace(specs[name], head_size=4, head_size_v=4, dtype=torch.float16)
    config = cache(specs)
    if shared:
        config.kv_cache_tensors[0].backing_id = "shared_pool"
        config.kv_cache_groups = [KVCacheGroupSpec([name], specs[name]) for name in members]
    apply_layerwise_kv_cache_plan(config, runtime(2))
    assert {tuple(t.layers) for t in config.kv_cache_tensors} == {
        (members[0], members[4]),
        (members[1], members[5]),
        (members[2],),
        (members[3],),
    }


def test_simultaneous_components_cannot_share_a_slot():
    main, extra = names(), names(suffix="extra")
    ordered = [name for pair in zip(main, extra) for name in pair]
    config = cache(dict.fromkeys(ordered, spec()))
    apply_layerwise_kv_cache_plan(config, runtime(2))
    assert len(config.kv_cache_tensors) == 4
    assert all(len({name.split(".")[2] for name in t.layers}) == len(t.layers) for t in config.kv_cache_tensors)


def test_sparse_component_uses_its_physical_layers_buffer_and_single_predecessor():
    main = names(6)
    extra = [name for i, name in enumerate(names(6, suffix="extra")) if i != 2]
    config = cache(dict.fromkeys(main + extra, spec()), [main, extra])
    apply_layerwise_kv_cache_plan(config, runtime(2))
    layout = get_layerwise_layout(config)
    assert layout.predecessors == {i: (i - 2,) for i in range(2, 6)}
    assert layout.layer_buffers == {i: i % 2 for i in range(6)}
    assert [t.layers for t in config.kv_cache_tensors] == [
        main[::2],
        main[1::2],
        [extra[0], extra[3]],
        [extra[1], extra[2], extra[4]],
    ]


def test_single_buffer_requires_at_most_one_participating_layer():
    members = names(3)
    config = cache(dict.fromkeys(members, spec()))
    before = deepcopy(config)
    with pytest.raises(ValueError, match="at least two buffers"):
        apply_layerwise_kv_cache_plan(config, runtime(1))
    assert config == before
    apply_layerwise_kv_cache_plan(config, runtime(1, independent=[0, 2]))
    assert get_layerwise_layout(config).layer_buffers == {1: 0}
    assert not get_layerwise_layout(config).has_layer_reuse


@pytest.mark.parametrize("same_group", [True, False])
def test_same_layer_different_offsets_share_only_the_logical_buffer(same_group):
    main = names(5)
    extra = names(5, suffix="extra")[4]
    config = cache(dict.fromkeys([*main, extra], spec()), [main, [extra]])
    if not same_group:
        config.kv_cache_groups = [KVCacheGroupSpec(main, spec()), KVCacheGroupSpec([extra], spec())]
    first, second = config.kv_cache_tensors
    first.backing_id = second.backing_id = "bank"
    second.offset = first.size
    first.size = second.size = first.size + second.size
    apply_layerwise_kv_cache_plan(config, runtime())
    layout = get_layerwise_layout(config)
    main_region = next(t for t in config.kv_cache_tensors if main[4] in t.layers)
    extra_region = next(t for t in config.kv_cache_tensors if extra in t.layers)
    assert main_region.offset != extra_region.offset
    assert main_region.layers == main[::2] and extra_region.layers == [extra]
    assert layout.layer_buffers[4] == 0 and layout.predecessors[4] == (2,)
    apply_layerwise_kv_cache_plan(config, runtime())


def test_published_plan_rejects_an_extra_predecessor_or_a_split_layer():
    config = cache(dict.fromkeys(names(), spec()))
    apply_layerwise_kv_cache_plan(config, runtime())
    layout = get_layerwise_layout(config)
    for invalid in (
        replace(layout, predecessors={**layout.predecessors, 4: (1, 2)}),
        replace(layout, layer_buffers={**layout.layer_buffers, 4: 1}),
        replace(layout, buffer_slots=(tuple(names()[::3]), tuple(n for n in names() if n not in names()[::3]))),
    ):
        with pytest.raises(ValueError, match="buffer assignment"):
            validate_layerwise_reuse_layout(config, invalid)


def test_excluded_reuse_is_rejected_before_publishing_independent_layers():
    config = cache(dict.fromkeys(names(), spec()))
    before = deepcopy(config)
    with pytest.raises(ValueError, match="conflicts"):
        apply_layerwise_kv_cache_plan(config, runtime(independent=[0]), excluded_layer_names={"model.layers.2.attn"})
    assert all(s.reuse for s in config.kv_cache_groups[0].kv_cache_spec.kv_cache_specs.values())
    assert config == before


def test_small_bucket_validates_intent_but_preserves_original_allocation():
    config = cache(dict.fromkeys(names(2), spec()))
    before = deepcopy(config.kv_cache_tensors)
    apply_layerwise_kv_cache_plan(config, runtime(2))
    assert config.kv_cache_tensors == before
    assert all(s.reuse for s in config.kv_cache_groups[0].kv_cache_spec.kv_cache_specs.values())


@pytest.mark.parametrize("field,value", [("layer_stride", 0), ("block_stride", 16), ("size", 1), ("offset", -1)])
def test_ambiguous_or_out_of_bounds_geometry_fails_atomically(field, value):
    config = cache(dict.fromkeys(names(), spec()))
    setattr(config.kv_cache_tensors[0], field, value)
    before = deepcopy(config)
    with pytest.raises((ValueError, NotImplementedError)):
        apply_layerwise_kv_cache_plan(config, runtime())
    assert config == before


def test_disabled_layerwise_does_not_validate_or_change_intent():
    config = cache(dict.fromkeys(names(), spec()))
    vc = runtime(independent=[2])
    vc.kv_transfer_config = None
    before = deepcopy(config)
    apply_layerwise_kv_cache_plan(config, vc)
    assert config == before


def test_unsupported_requested_cache_is_rejected_even_without_actual_alias():
    state = MambaSpec(block_size=2, shapes=((1,),), dtypes=(torch.int8,), reuse=True)
    config = cache({names(1)[0]: state})
    with pytest.raises(NotImplementedError, match="save/restore"):
        apply_layerwise_kv_cache_plan(config, runtime())


def test_reuse_field_survives_replace_and_does_not_change_group_equality():
    original = spec()
    assert replace(original, block_size=4).reuse
    assert original == replace(original, reuse=False)


def test_uniform_reuse_survives_merge_and_only_splits_at_placement():
    members = names(4)
    merged = FullAttentionSpec.merge([spec() for _ in members])
    assert merged.reuse
    config = cache(dict.fromkeys(members, merged))
    original = KVCacheGroupSpec(members, merged)
    config.kv_cache_groups = [original]
    assert config.transfer_groups[0] is original
    apply_layerwise_kv_cache_plan(config, runtime(independent=[0]))
    (group,) = config.kv_cache_groups
    assert config.transfer_groups[0] is group
    assert group.layer_names == members
    assert {name: s.reuse for name, s in group.kv_cache_spec.kv_cache_specs.items()} == {
        name: i != 0 for i, name in enumerate(members)
    }
    assert original.kv_cache_spec is merged and merged.reuse


def test_inference_and_published_validation_have_separate_inputs():
    config = cache(dict.fromkeys(names(), spec()))
    vc = runtime()
    policy = get_layerwise_policy(config, 6, vc.kv_transfer_config.kv_connector_extra_config)
    original = deepcopy(config)
    _, inferred = infer_layerwise_reuse_layout(config, policy, base_layers=6)
    assert config == original
    assert get_layerwise_layout(config) is None
    apply_layerwise_kv_cache_plan(config, vc)
    assert get_layerwise_layout(config) == inferred
    placed = deepcopy(config)
    validate_layerwise_reuse_layout(config, inferred, policy=policy, base_layers=6)
    assert config == placed
    with pytest.raises(ValueError, match="already placed"):
        infer_layerwise_reuse_layout(config, policy, base_layers=6)
    config.num_blocks += 1
    with pytest.raises(ValueError, match="policy/capacity"):
        validate_layerwise_reuse_layout(config, inferred, base_layers=6)


def test_worker_projection_preserves_per_name_reuse_after_group_merge():
    from vllm.v1.core.kv_cache_planning import _project_kv_cache_groups_to_worker

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        get_layerwise_kv_cache_specs,
    )

    members = names(4)
    specs = {name: spec(reuse=i % 2 == 0) for i, name in enumerate(members)}
    group_spec = UniformTypeKVCacheSpecs(block_size=2, kv_cache_specs=specs)
    group = KVCacheGroupSpec(members, group_spec)
    projected = _project_kv_cache_groups_to_worker([group], {name: specs[name] for name in members[1:3]})
    expanded = get_layerwise_kv_cache_specs(KVCacheConfig(1, [], projected))
    assert {name: s.reuse for name, s in expanded.items()} == {members[1]: False, members[2]: True}
    assert projected[0].layer_names == members[1:3]


@pytest.mark.parametrize("field,value", [("host_resident", True), ("enable_kv_transfer", False)])
def test_explicit_request_requires_device_save_restore(field, value):
    config = cache(dict.fromkeys(names(), spec()))
    setattr(config.kv_cache_groups[0], field, value)
    before = deepcopy(config)
    with pytest.raises(ValueError, match="save/restore"):
        apply_layerwise_kv_cache_plan(config, runtime())
    assert config == before


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

    assert get_layerwise_reuse_config(multi_config) is ascend_store_config
    for connector_name in (
        "AscendStoreConnector",
        "MooncakeConnectorStoreV1",
    ):
        direct_config = SimpleNamespace(
            kv_connector=connector_name,
            kv_connector_extra_config=ascend_store_config,
        )
        assert get_layerwise_reuse_config(direct_config) is ascend_store_config
    assert get_layerwise_reuse_config(unsupported) is None
    assert get_layerwise_reuse_config(not_opted_in) is None


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
