# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical allocation and Qwen4 descriptor-driven reuse."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.layerwise_cache_config import get_layerwise_layout, kv_cache_allocation_bytes
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    get_layerwise_kv_cache_specs,
)
from vllm_ascend.models.qwen4_exp.cache_config import (
    GDN,
    HIDDEN,
    PLE,
    QSA_COMPRESSED,
    QSA_MAIN,
    build_qwen4_kv_cache_layout,
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_pool_bytes_per_block,
    make_qsa_kernel_page_cache,
    make_slab_view,
)

BUDGET = 12556015104
BYTES_PER_BLOCK_WITH_TWO_RESIDENT_LAYERS = 3392640
PREFIX = "language_model.model.layers."


def sample_groups():
    main = FullAttentionSpec(
        reuse=True, block_size=768, num_kv_heads=1, head_size=256, head_size_v=256, dtype=torch.bfloat16
    )
    raw = CircularBufferSpec(
        block_size=4, num_kv_heads=1, head_size=140, head_size_v=0, dtype=torch.bfloat16, page_size_padded=786432
    )
    compressed = MLAAttentionSpec(
        reuse=True,
        block_size=768,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
        page_size_padded=786432,
    )
    gdn = MambaSpec(
        reuse=True,
        block_size=768,
        shapes=((3, 1280), (6, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=786432,
        mamba_cache_mode="align",
    )
    ple = MambaSpec(
        block_size=768,
        shapes=((3072, 30),),
        dtypes=(torch.bfloat16,),
        page_size_padded=786432,
        mamba_cache_mode="align",
        tp_replicated=True,
    )

    def group(specs):
        return KVCacheGroupSpec(
            list(specs), UniformTypeKVCacheSpecs(block_size=next(iter(specs.values())).block_size, kv_cache_specs=specs)
        )

    sources = [f"{PREFIX}{layer}.self_attn" for layer in range(3, 48, 4)]
    states = [f"{PREFIX}{layer}.linear_attn" for layer in range(48) if layer % 4 != 3]
    groups = [
        group({f"{source}.indexer.raw_key_cache": raw for source in sources}),
        group(
            {
                **{f"{source}.attn": main for source in sources},
                **{f"{source}.indexer.compressed_key_cache": compressed for source in sources},
            }
        ),
        *[group(dict.fromkeys(states[start : start + 12], gdn)) for start in (0, 12, 24)],
        group({f"{PREFIX}1.ple": ple}),
    ]

    return groups


def runtime_config(independent=(0, 1)):
    return SimpleNamespace(
        additional_config={},
        use_v2_model_runner=True,
        kv_transfer_config=SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": True,
                "layerwise_num_shared_buffers": 2,
                "layerwise_independent_layers": list(independent),
            },
        ),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=0),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, decode_context_parallel_size=1, prefill_context_parallel_size=1, world_size=1
        ),
        model_config=SimpleNamespace(
            get_num_layers=lambda _: 48,
            hf_text_config=SimpleNamespace(),
            hf_config=SimpleNamespace(),
            enforce_eager=True,
        ),
        speculative_config=None,
    )


def make_config(n=None):
    vc = runtime_config()
    groups = sample_groups()
    budget = BUDGET if n is None else get_qwen4_exp_pool_bytes_per_block(groups, vc) * n
    return get_qwen4_exp_kv_cache_config(vc, groups, budget)


def make_views(config, device="cpu", alignment=1):
    """Use the same descriptors and original component geometry as the runner."""

    def allocate(size):
        backing = torch.zeros(size + alignment - 1, dtype=torch.int8, device=device)
        offset = -backing.data_ptr() % alignment
        return backing[offset : offset + size]

    original = build_qwen4_kv_cache_layout(config.kv_cache_groups, config.num_blocks)
    shared_size = next(t.size for t in config.kv_cache_tensors if original.owner(t.layers[0]).role not in (PLE, HIDDEN))
    shared = allocate(shared_size)
    backings, caches = {"shared": shared}, {}
    for owner in original.owners:
        descriptor = next(t for t in config.kv_cache_tensors if owner.layer_name in t.layers)
        layer_offset = descriptor.offset + descriptor.layers.index(owner.layer_name) * descriptor.layer_stride
        spec = owner.spec
        if owner.role in (PLE, HIDDEN):
            backing = allocate(descriptor.size)
            backings[owner.layer_name] = backing
        else:
            backing = shared
        if owner.role == QSA_MAIN:
            value = make_qsa_kernel_page_cache(
                backing,
                dtype=spec.dtype,
                num_blocks=config.num_blocks,
                manager_block_size=spec.block_size,
                kernel_block_size=128,
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                head_size_v=spec.head_size_v,
                storage_offset=layer_offset,
                block_stride=descriptor.block_stride,
            )
        elif owner.role in (GDN, PLE):
            value = [
                make_slab_view(
                    backing,
                    dtype=dtype,
                    num_blocks=config.num_blocks,
                    item_shape=shape,
                    storage_offset=layer_offset + (original.region("r2").offset if index else 0),
                    block_stride=descriptor.block_stride,
                )
                for index, (shape, dtype) in enumerate(zip(spec.shapes, spec.dtypes))
            ]
        else:
            pages = spec.block_size // 128 if owner.role == QSA_COMPRESSED else 1
            states = spec.get_num_kernel_states(128) if owner.role == QSA_COMPRESSED else spec.block_size
            value = make_slab_view(
                backing,
                dtype=spec.dtype,
                num_blocks=config.num_blocks * pages,
                item_shape=(spec.num_kv_heads, states, spec.head_size),
                storage_offset=layer_offset,
            ).transpose(1, 2)
        caches[owner.layer_name] = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    return backings, caches


def test_capacity_counts_backings_once_and_preserves_groups():
    groups = sample_groups()
    before = copy.deepcopy(groups)
    config = make_config()
    assert config.num_blocks == 3700
    assert get_qwen4_exp_pool_bytes_per_block(groups, runtime_config()) == BYTES_PER_BLOCK_WITH_TWO_RESIDENT_LAYERS
    assert kv_cache_allocation_bytes(config.kv_cache_tensors) == BYTES_PER_BLOCK_WITH_TWO_RESIDENT_LAYERS * 3700
    assert config.kv_cache_groups == before
    assert all(t.layer_stride >= 0 for t in config.kv_cache_tensors)
    assert sorted(name for t in config.kv_cache_tensors for name in t.layers) == sorted(
        name for g in groups for name in g.layer_names
    )


def test_descriptors_preserve_rotation_components_and_resident_regions():
    config = make_config(3)
    backings, caches = make_views(config)
    assert len(backings) == 2
    main3 = caches[PREFIX + "3.self_attn.attn"]
    main11 = caches[PREFIX + "11.self_attn.attn"]
    assert [t.data_ptr() for t in main3] == [t.data_ptr() for t in main11]
    gdn = caches[PREFIX + "2.linear_attn"]
    assert gdn[1].data_ptr() - gdn[0].data_ptr() == 3 * 1280 * 2
    assert gdn[0].stride(0) * gdn[0].element_size() == 786432
    assert gdn[1].stride(0) * gdn[1].element_size() == 786432
    raw = [caches[PREFIX + f"{layer}.self_attn.indexer.raw_key_cache"][0] for layer in range(3, 48, 4)]
    assert len({t.data_ptr() for t in raw}) == 12
    assert caches[PREFIX + "1.ple"][0].is_contiguous()
    raw_descriptors = [t for t in config.kv_cache_tensors if t.layers[0].endswith("raw_key_cache")]
    assert len(raw_descriptors) == 1
    assert len(raw_descriptors[0].layers) == 12
    assert raw_descriptors[0].layer_stride == 3 * 1120
    assert raw_descriptors[0].block_stride == 1120
    assert len(config.kv_cache_tensors) == 6
    assert main3[0].data_ptr() == caches[PREFIX + "5.linear_attn"][0].data_ptr()
    assert main3[0].data_ptr() != gdn[0].data_ptr()


@pytest.mark.parametrize("independent", [(), (0, 1)])
def test_whole_layers_rotate_across_qsa_and_gdn_groups(independent):
    vc = runtime_config(independent)
    groups = sample_groups()
    cost = get_qwen4_exp_pool_bytes_per_block(groups, vc)
    config = get_qwen4_exp_kv_cache_config(vc, groups, 3 * cost)
    layout = get_layerwise_layout(config)
    start = len(independent)
    assert layout.layer_buffers == {layer: (layer - start) % 2 for layer in range(start, 48)}
    assert layout.predecessors == {layer: (layer - 2,) for layer in range(start + 2, 48)}
    main_or_gdn = [
        t
        for t in config.kv_cache_tensors
        if t.layer_stride == 0 and any(name.endswith((".self_attn.attn", ".linear_attn")) for name in t.layers)
    ]
    assert len(main_or_gdn) == 2
    for buffer, tensor in enumerate(main_or_gdn):
        expected = [
            PREFIX + str(layer) + (".self_attn.attn" if layer % 4 == 3 else ".linear_attn")
            for layer in range(start + buffer, 48, 2)
        ]
        assert tensor.layers == expected
    compressed = next(t for t in config.kv_cache_tensors if t.layers[0].endswith("compressed_key_cache"))
    assert len(compressed.layers) == 12 and compressed.layer_stride == 0
    assert len({t.offset for t in [*main_or_gdn, compressed]}) == 3
    assert [g.layer_names for g in config.kv_cache_groups] == [g.layer_names for g in groups]


def test_apply_is_idempotent_and_rejects_changed_policy():
    config = make_config(2)
    before = copy.deepcopy(config)
    apply_layerwise_kv_cache_plan(config, runtime_config())
    assert config == before
    vc = runtime_config()
    vc.kv_transfer_config.kv_connector_extra_config["layerwise_num_shared_buffers"] = 3
    with pytest.raises(ValueError, match="policy/capacity"):
        apply_layerwise_kv_cache_plan(config, vc)


def test_cross_group_writes_and_same_group_restore_preserve_valid_blocks(monkeypatch):
    from tests.ut.models.test_qwen4_exp_cache_config import _allocate_mrv2_kv_cache

    config = make_config(3)
    _, caches, _ = _allocate_mrv2_kv_cache(config, monkeypatch)
    main = caches[PREFIX + "3.self_attn.attn"]
    gdn5 = caches[PREFIX + "5.linear_attn"]
    gdn9 = caches[PREFIX + "9.linear_attn"]
    assert main[0].data_ptr() == gdn5[0].data_ptr() == gdn9[0].data_ptr()
    # Main owns block 1; the GDN group owns block 2 in the same slab.
    main_block = [part[part.shape[0] // 3 : 2 * (part.shape[0] // 3)] for part in main]
    for index, part in enumerate(main_block):
        part.fill_(10 + index)
    for index, part in enumerate(gdn5):
        part[2].fill_(20 + index)
    snapshot = [part[2].clone() for part in gdn5]
    for part in gdn9:
        part[2].fill_(30)
    for index, (part, saved) in enumerate(zip(gdn5, snapshot, strict=True)):
        assert torch.all(part[2] == 30)
        part[2].copy_(saved)
        assert torch.all(part[2] == 20 + index)
    for index, part in enumerate(main_block):
        assert torch.all(part == 10 + index)


@pytest.mark.parametrize(
    ("independent", "resident"), [([0, 3], {0, 3}), ([0, 1], {0, 1}), (None, {0}), ("all", set(range(48)))]
)
def test_independent_layers_apply_after_grouping_in_probe_and_scheduler(independent, resident):
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config

    vc = runtime_config()
    extra = vc.kv_transfer_config.kv_connector_extra_config
    if independent is None:
        del extra["layerwise_independent_layers"]
    else:
        extra["layerwise_independent_layers"] = independent
    groups = sample_groups()
    declared = get_layerwise_kv_cache_specs(KVCacheConfig(1, [], groups))
    flags = {name: spec.reuse for name, spec in declared.items()}
    expected = {
        name: flag and int(name.removeprefix(PREFIX).split(".")[0]) not in resident for name, flag in flags.items()
    }
    assert flags[PREFIX + "1.linear_attn"] and flags[PREFIX + "3.self_attn.attn"]
    assert not flags[PREFIX + "3.self_attn.indexer.raw_key_cache"]
    cost = get_qwen4_exp_pool_bytes_per_block(groups, vc)
    config = get_qwen4_exp_kv_cache_config(vc, groups, cost * 4)
    assert config.num_blocks == 4
    assert kv_cache_allocation_bytes(config.kv_cache_tensors) == cost * 4
    scheduler = generate_scheduler_kv_cache_config([config])
    for published in (config, scheduler):
        assert [g.layer_names for g in published.kv_cache_groups] == [g.layer_names for g in groups]
        assert {name: spec.reuse for name, spec in get_layerwise_kv_cache_specs(published).items()} == expected
        assert get_layerwise_layout(published).resident_layer_names == {
            name for name, flag in expected.items() if not flag
        }
    assert {
        name: spec.reuse for name, spec in get_layerwise_kv_cache_specs(KVCacheConfig(1, [], groups)).items()
    } == flags


def test_disabled_layerwise_keeps_original_layout():
    vc = runtime_config(range(48))
    vc.kv_transfer_config = None
    config = get_qwen4_exp_kv_cache_config(vc, sample_groups(), BUDGET)
    assert config.num_blocks == 1228
    assert get_layerwise_layout(config) is None


def test_override_cannot_exceed_budget():
    vc = runtime_config()
    vc.cache_config.num_gpu_blocks_override = BUDGET // BYTES_PER_BLOCK_WITH_TWO_RESIDENT_LAYERS + 1
    with pytest.raises(ValueError, match="exceeds"):
        get_qwen4_exp_kv_cache_config(vc, sample_groups(), BUDGET)


def test_mixed_component_intent_is_rejected_without_coercion():
    groups = sample_groups()
    name = PREFIX + "3.self_attn.indexer.compressed_key_cache"
    specs = groups[1].kv_cache_spec.kv_cache_specs
    specs[name] = replace(specs[name], reuse=False)
    with pytest.raises(NotImplementedError, match="Mixed reuse intent"):
        get_qwen4_exp_kv_cache_config(runtime_config(), groups, BUDGET)
    assert not specs[name].reuse


def test_coalescing_preserves_every_production_view(monkeypatch):
    from tests.ut.models.test_qwen4_exp_cache_config import _allocate_mrv2_kv_cache

    config = make_config(3)
    split = copy.deepcopy(config)
    # Reconstruct the previous representation: rotating descriptors split by
    # group, with all original offsets, strides and slot dependencies intact.
    split.kv_cache_tensors = []
    for descriptor in config.kv_cache_tensors:
        if descriptor.layer_stride:
            split.kv_cache_tensors.append(copy.deepcopy(descriptor))
            continue
        for group in config.kv_cache_groups:
            members = [name for name in descriptor.layers if name in group.layer_names]
            if members:
                split.kv_cache_tensors.append(replace(descriptor, layers=members))
    assert len(split.kv_cache_tensors) == 11
    assert config.kv_cache_groups == split.kv_cache_groups
    assert get_layerwise_layout(config) == get_layerwise_layout(split)
    raw, caches, _ = _allocate_mrv2_kv_cache(config, monkeypatch)
    assert caches[PREFIX + "3.self_attn.attn"][0].data_ptr() == caches[PREFIX + "11.self_attn.attn"][0].data_ptr()
    allocations = {t.data_ptr(): t.numel() for t in raw.values()}
    assert len(allocations) == 2
    assert sum(allocations.values()) == 3 * BYTES_PER_BLOCK_WITH_TWO_RESIDENT_LAYERS
    before_backings, before_caches = make_views(split)
    for name, previous in before_caches.items():
        current = caches[name]
        current = current if isinstance(current, (tuple, list)) else (current,)
        old_backing = before_backings.get(name, before_backings["shared"])
        for actual, expected in zip(current, previous, strict=True):
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.stride() == expected.stride()
            assert actual.data_ptr() - raw[name].data_ptr() == expected.data_ptr() - old_backing.data_ptr()


def test_changed_descriptor_is_rejected_before_runner_reuses_it():
    config = make_config(2)
    config.kv_cache_tensors[0].offset += 16
    with pytest.raises(ValueError, match="descriptors changed"):
        apply_layerwise_kv_cache_plan(config, runtime_config())


@pytest.mark.parametrize("independent", [[0, 1], [0, 3]])
def test_cross_rank_capacity_rebuilds_offsets_at_common_block_count(monkeypatch, independent):
    from vllm.v1.core import kv_cache_config_builder, kv_cache_planning

    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        get_layerwise_kv_cache_specs,
    )
    from vllm_ascend.patch.platform import patch_kv_cache_utils  # noqa: F401

    vc = runtime_config(independent)
    vc.model_config.original_max_model_len = vc.model_config.max_model_len = 768
    vc.model_config.is_encoder_decoder = False
    vc.parallel_config.tensor_parallel_size = 1
    vc.scheduler_config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=768)
    vc.attention_config = SimpleNamespace(hisparse_config=None)
    vc.cache_config.mamba_cache_mode = "align"
    groups = sample_groups()
    specs = get_layerwise_kv_cache_specs(KVCacheConfig(1, [], groups))
    cost = get_qwen4_exp_pool_bytes_per_block(groups, vc)
    monkeypatch.setattr(kv_cache_config_builder, "get_kv_cache_config_builder", lambda _: AscendKVCacheConfigBuilder())
    configs = kv_cache_planning.get_kv_cache_configs(vc, [specs, specs], [100 * cost, 80 * cost], groups)
    expected = get_qwen4_exp_kv_cache_config(vc, groups, 80 * cost)
    for config in configs:
        flags = {name: spec.reuse for name, spec in get_layerwise_kv_cache_specs(config).items()}
        assert flags[PREFIX + "1.linear_attn"] == (1 not in independent)
        assert flags[PREFIX + "3.self_attn.attn"] == (3 not in independent)
        assert config.num_blocks == 80
        assert config.kv_cache_tensors == expected.kv_cache_tensors
        assert get_layerwise_layout(config) == get_layerwise_layout(expected)


@pytest.mark.parametrize("override", [None, 3651])
def test_worker_profiles_through_common_planner_without_scaling_physical_budget(monkeypatch, override):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        get_layerwise_kv_cache_specs,
    )
    from vllm_ascend.worker import worker as worker_module

    vc = runtime_config()
    vc.cache_config.num_gpu_blocks_override = override
    groups = sample_groups()
    specs = get_layerwise_kv_cache_specs(KVCacheConfig(1, [], groups))
    worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
    worker.vllm_config = vc
    worker.model_config = vc.model_config
    worker.parallel_config = vc.parallel_config
    worker.model_runner = SimpleNamespace(speculator=None)
    monkeypatch.setattr(worker, "get_kv_cache_spec", lambda: specs)
    monkeypatch.setattr(worker_module, "get_kv_cache_groups", lambda *_: groups)

    cost = get_qwen4_exp_pool_bytes_per_block(groups, vc)
    budget = 100 * cost
    assert worker._scale_kv_cache_memory_for_multi_group(budget) == budget
    assert worker._gva_layerwise_memory_info == (48, 2, 1.0)
    assert vc.cache_config.num_gpu_blocks_override == override
    assert cost == BYTES_PER_BLOCK_WITH_TWO_RESIDENT_LAYERS
