# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical allocation, serialization and Qwen4 descriptor-driven reuse."""

import copy
import pickle
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
)
from vllm_ascend.models.qwen4_exp.cache_config import (
    GDN,
    HIDDEN,
    PLE,
    QSA_COMPRESSED,
    QSA_MAIN,
    build_qwen4_kv_cache_layout,
    declare_qwen4_reuse,
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_pool_bytes_per_block,
    make_qsa_kernel_page_cache,
    make_qwen4_original_tensors,
    make_slab_view,
)

BUDGET = 12556015104
PREFIX = "language_model.model.layers."


def sample_groups():
    main = FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=256, head_size_v=256, dtype=torch.bfloat16)
    raw = CircularBufferSpec(
        block_size=4, num_kv_heads=1, head_size=140, head_size_v=0, dtype=torch.bfloat16, page_size_padded=786432
    )
    compressed = MLAAttentionSpec(
        block_size=768, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, tokens_per_state=4, page_size_padded=786432
    )
    gdn = MambaSpec(
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
        *[group({name: gdn for name in states[start : start + 12]}) for start in (0, 12, 24)],
        group({f"{PREFIX}1.ple": ple}),
    ]

    declared = declare_qwen4_reuse(
        {name: spec for group_spec in groups for name, spec in group_spec.kv_cache_spec.kv_cache_specs.items()}
    )
    for group_spec in groups:
        group_spec.kv_cache_spec = replace(
            group_spec.kv_cache_spec,
            kv_cache_specs={name: declared[name] for name in group_spec.layer_names},
        )
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
    assert config.num_blocks == 3648
    assert get_qwen4_exp_pool_bytes_per_block(groups, runtime_config()) == 3441792
    assert kv_cache_allocation_bytes(config.kv_cache_tensors) == 3441792 * 3648 <= BUDGET
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
    gdn_descriptors = [t for t in config.kv_cache_tensors if t.layers[0].endswith("linear_attn")]
    assert len(config.kv_cache_tensors) == 9
    assert len(gdn_descriptors) == 3
    assert gdn_descriptors[0].layers == [PREFIX + "0.linear_attn", PREFIX + "1.linear_attn"]
    group_ids = {name: i for i, group in enumerate(config.kv_cache_groups) for name in group.layer_names}
    for descriptor in gdn_descriptors[1:]:
        assert {group_ids[name] for name in descriptor.layers} == {2, 3, 4}
        assert descriptor.layer_stride == 0
        assert len({tuple(t.data_ptr() for t in caches[name]) for name in descriptor.layers}) == 1


def test_apply_is_idempotent_and_rejects_changed_policy():
    config = make_config(2)
    before = copy.deepcopy(config)
    apply_layerwise_kv_cache_plan(config, runtime_config())
    assert config == before
    vc = runtime_config()
    vc.kv_transfer_config.kv_connector_extra_config["layerwise_num_shared_buffers"] = 3
    with pytest.raises(ValueError, match="policy/capacity"):
        apply_layerwise_kv_cache_plan(config, vc)


def test_slot_result_survives_scheduler_simplification_and_pickle():
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config

    config = make_config(2)
    scheduler = generate_scheduler_kv_cache_config([config])
    transported = pickle.loads(pickle.dumps(config))
    assert get_layerwise_layout(scheduler) == get_layerwise_layout(transported)
    assert transported.kv_cache_tensors == config.kv_cache_tensors
    assert get_layerwise_layout(config).predecessors[11] == (3,)
    assert get_layerwise_layout(config).predecessors[5] == (2,)


@pytest.mark.parametrize("conflict", ["independent", "excluded"])
def test_explicit_reuse_conflicts_do_not_mutate_configuration(conflict):
    groups = sample_groups()
    original = build_qwen4_kv_cache_layout(groups, 2)
    config = KVCacheConfig(2, make_qwen4_original_tensors(original), groups)
    before = copy.deepcopy(config)
    vc = runtime_config([0, 1, 3] if conflict == "independent" else [0, 1])
    excluded = {PREFIX + "3.self_attn.attn"} if conflict == "excluded" else None
    with pytest.raises(ValueError, match="conflicts"):
        apply_layerwise_kv_cache_plan(config, vc, excluded_layer_names=excluded)
    assert config == before


def test_disabled_layerwise_keeps_original_layout_even_with_conflicting_intent():
    vc = runtime_config(range(48))
    vc.kv_transfer_config = None
    config = get_qwen4_exp_kv_cache_config(vc, sample_groups(), BUDGET)
    assert config.num_blocks == 1228
    assert get_layerwise_layout(config) is None


def test_override_cannot_exceed_budget():
    vc = runtime_config()
    vc.cache_config.num_gpu_blocks_override = 3649
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
    assert len(split.kv_cache_tensors) == 13
    assert config.kv_cache_groups == split.kv_cache_groups
    assert get_layerwise_layout(config) == get_layerwise_layout(split)
    raw, caches, _ = _allocate_mrv2_kv_cache(config, monkeypatch)
    assert caches[PREFIX + "3.self_attn.attn"][0].data_ptr() == caches[PREFIX + "11.self_attn.attn"][0].data_ptr()
    allocations = {t.data_ptr(): t.numel() for t in raw.values()}
    assert len(allocations) == 2
    assert sum(allocations.values()) == 3 * 3441792
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


def test_cross_rank_capacity_rebuilds_offsets_at_common_block_count(monkeypatch):
    from vllm.v1.core import kv_cache_config_builder, kv_cache_planning

    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        get_layerwise_kv_cache_specs,
    )
    from vllm_ascend.patch.platform import patch_kv_cache_utils  # noqa: F401

    vc = runtime_config()
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
    expected = make_config(80)
    for config in configs:
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
    assert worker._gva_layerwise_memory_info == (48, 6, 1.0)
    assert vc.cache_config.num_gpu_blocks_override == override
    assert cost == 3441792
