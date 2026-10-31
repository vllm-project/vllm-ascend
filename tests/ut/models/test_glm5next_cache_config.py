# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GLM-Next cache grouping, physical layout, and capacity accounting."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import (
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendMLAAttentionSpec
from vllm_ascend.models.glm5next.cache_config import (
    GLM5_NEXT_CACHE_ALIGNMENT,
    _get_glm5_next_cache_layout,
    get_glm5_next_alignment_reserve,
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_max_memory_usage,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers


@pytest.mark.parametrize("pool", [4, 16])
@pytest.mark.parametrize("speculative_blocks", [0, 5])
def test_platform_builder_accounts_for_both_glm_page_classes(pool, speculative_blocks):
    planning = pytest.importorskip("vllm.v1.core.kv_cache_planning")
    if not hasattr(planning, "_pool_bytes_per_block_from_builder"):
        pytest.skip("Requires the upstream builder pool-accounting hook")
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from vllm_ascend.platform import NPUPlatform

    config = make_config()
    config.attention_config = SimpleNamespace(hisparse_config=None)
    specs = {
        name: replace(spec, num_speculative_blocks=speculative_blocks) if isinstance(spec, MambaSpec) else spec
        for name, spec in make_specs(pool).items()
    }
    groups = get_glm5_next_kv_cache_groups(config, specs)
    physical = get_glm5_next_kv_cache_config(config, groups, get_glm5_next_pool_bytes_per_block(groups))
    assert len({tensor.size for tensor in physical.kv_cache_tensors}) == 2
    expected = sum(tensor.size for tensor in physical.kv_cache_tensors)
    builder = resolve_obj_by_qualname(NPUPlatform.get_kv_cache_config_builder_cls(config))()
    config.cache_config.num_gpu_blocks_override = 7
    assert planning._pool_bytes_per_block_from_builder(builder, config, groups) == expected
    assert config.cache_config.num_gpu_blocks_override == 7


def test_platform_builder_preserves_default_non_glm_accounting():
    pytest.importorskip("vllm.v1.core.kv_cache_planning")
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    groups = [KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=make_specs()["model.layers.3.attn"])]
    groups[0].kv_cache_spec = replace(groups[0].kv_cache_spec, model_version=None)
    assert AscendKVCacheConfigBuilder()._get_pool_bytes_per_block(make_config(), groups) is None


def _ratio_kwargs(ratio: int) -> dict[str, int]:
    return {"tokens_per_state": ratio}


def make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        max_in_flight_tokens=32,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            mamba_cache_mode="none",
            enable_prefix_caching=False,
        ),
    )


def make_specs(pool: int = 16):
    specs = {
        "model.layers.3.attn": MLAAttentionSpec(
            block_size=512,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            model_version="glm5_next",
        ),
        "model.layers.3.indexer.k_cache": MLAAttentionSpec(
            block_size=512,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            **_ratio_kwargs(pool),
        ),
        "model.layers.3.indexer.tail_cache": AscendIndexerKPoolTailSpec(
            block_size=pool,
            sliding_window=pool,
            compress_ratio=pool,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float32,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
        ),
    }
    for layer_idx in range(3):
        specs[f"model.layers.{layer_idx}.linear_attn"] = MambaSpec(
            block_size=512,
            shapes=((3, 16), (1, 16, 16)),
            dtypes=(torch.bfloat16, torch.float32),
        )
    return specs


def make_full_prefill_specs():
    templates = make_specs(pool=4)
    specs = {}
    for layer in range(45):
        if layer % 4 == 3:
            for name, spec in templates.items():
                if not isinstance(spec, MambaSpec):
                    specs[name.replace("layers.3.", f"layers.{layer}.")] = replace(spec)
        else:
            specs[f"model.layers.{layer}.linear_attn"] = replace(
                templates["model.layers.0.linear_attn"], mamba_cache_mode="align"
            )
    return specs


@pytest.mark.parametrize("buffers", [1, 2, 3, 4])
def test_prefill_reuse_increases_blocks_with_the_same_physical_budget(buffers):
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        apply_layerwise_kv_cache_plan,
    )

    config = make_config()
    config.model_config.get_num_layers = lambda _: 45
    groups = get_glm5_next_kv_cache_groups(config, make_full_prefill_specs())
    layout = _get_glm5_next_cache_layout(groups)
    old_cost = get_glm5_next_pool_bytes_per_block(groups)
    budget = 128 * old_cost
    baseline = get_glm5_next_kv_cache_config(config, groups, budget)
    small_layers = set(layout.indexer_names) | set(layout.tail_names)
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="AscendStoreConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={
            "backend": "memcache",
            "use_layerwise": True,
            "layerwise_num_shared_buffers": buffers,
        },
    )
    reserve = get_glm5_next_alignment_reserve(config, groups)
    plan = get_glm5_next_kv_cache_config(config, groups, budget - reserve)
    new_cost = (buffers + 1) * layout.main_page_size + 11 * layout.small_page_size
    assert get_glm5_next_pool_bytes_per_block(groups, config) == new_cost
    assert AscendKVCacheConfigBuilder()._get_pool_bytes_per_block(config, groups) == new_cost
    assert plan.num_blocks == (budget - reserve) // new_cost > baseline.num_blocks
    assert sum(tensor.size for tensor in plan.kv_cache_tensors) == plan.num_blocks * new_cost <= budget - reserve
    assert reserve == len(plan.kv_cache_tensors) * GLM5_NEXT_CACHE_ALIGNMENT
    assert len(plan.kv_cache_tensors) == buffers + 1 + 11
    assert generate_scheduler_kv_cache_config([plan]).layerwise_kv_cache_reuse
    names = [name for tensor in plan.kv_cache_tensors for name in get_kv_cache_tensor_layers(tensor)]
    assert len(names) == len(set(names)) == 67
    for tensor in plan.kv_cache_tensors:
        assert tensor.size == plan.num_blocks * tensor.block_stride
        assert (plan.num_blocks - 1) * tensor.block_stride < tensor.size
    small = [tensor for tensor in plan.kv_cache_tensors if set(get_kv_cache_tensor_layers(tensor)) & small_layers]
    assert len(small) == 11
    assert all(len(get_kv_cache_tensor_layers(tensor)) == 2 for tensor in small)
    original_tensors = plan.kv_cache_tensors
    apply_layerwise_kv_cache_plan(plan, config)
    assert plan.kv_cache_tensors is original_tensors
    required_ids = get_glm5_next_max_memory_usage(config, groups) // new_cost
    assert get_glm5_next_max_memory_usage(config, groups) == required_ids * new_cost
    # Replanning an unequal rank must preserve the agreed global ID capacity.
    smaller = get_glm5_next_kv_cache_config(config, groups, (plan.num_blocks - 1) * new_cost)
    assert smaller.num_blocks == plan.num_blocks - 1


def test_alignment_reserve_tracks_real_backings_and_preserves_explicit_block_counts():
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    config = make_config()
    config.model_config.get_num_layers = lambda _: 45
    groups = get_glm5_next_kv_cache_groups(config, make_full_prefill_specs())
    assert get_glm5_next_alignment_reserve(config, groups) == 0
    for reuse, slots in [(False, 23), (True, 15)]:
        config.kv_transfer_config = SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": reuse,
                "layerwise_num_shared_buffers": 3,
            },
        )
        reserve = get_glm5_next_alignment_reserve(config, groups)
        assert reserve == slots * GLM5_NEXT_CACHE_ALIGNMENT
        for blocks in (7, 11):
            plan = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(config, groups, blocks)
            assert plan.num_blocks == blocks
            assert len(plan.kv_cache_tensors) == slots


@pytest.mark.parametrize("reuse", [False, True])
def test_worker_reserves_alignment_before_planning_global_blocks(reuse):
    from vllm_ascend.worker.worker import NPUWorker

    config = make_config()
    config.model_config.get_num_layers = lambda _: 45
    config.model_config.hf_text_config = SimpleNamespace(model_type="glm5_next_text")
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="AscendStoreConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={
            "backend": "memcache",
            "use_layerwise": reuse,
            "layerwise_num_shared_buffers": 3,
        },
    )
    specs = make_full_prefill_specs()
    groups = get_glm5_next_kv_cache_groups(config, specs)
    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = config
    worker.model_config = config.model_config
    worker.get_kv_cache_spec = lambda: specs
    budget = 8 * 1024**3
    remaining = worker._reserve_glm5_next_cache_alignment(budget)
    reserve = get_glm5_next_alignment_reserve(config, groups)
    assert remaining == budget - reserve
    assert worker._reserve_glm5_next_cache_alignment(reserve - 1) == 0
    plan = get_glm5_next_kv_cache_config(config, groups, remaining)
    allocated = sum(tensor.size for tensor in plan.kv_cache_tensors) + reserve
    assert allocated <= budget < allocated + get_glm5_next_pool_bytes_per_block(groups, config)


@pytest.mark.parametrize("buffers", [1, 2, 3, 4])
@pytest.mark.parametrize("role", ["kv_producer", "kv_both"])
def test_glm_ring_preserves_ordinary_packs_and_group_overwrite_order(buffers, role):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        build_layerwise_reuse_layout,
        get_layerwise_kv_cache_specs,
    )

    config = make_config()
    config.model_config.get_num_layers = lambda _: 45
    extra = {
        "backend": "memcache",
        "use_layerwise": True,
        "layerwise_num_shared_buffers": buffers,
        "layerwise_independent_layers": [0, 1, 2, 3],
        "layerwise_prefetch_layers": 8,
    }
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="AscendStoreConnector", kv_role=role, kv_connector_extra_config=extra
    )
    groups = get_glm5_next_kv_cache_groups(config, make_full_prefill_specs())
    plan = get_glm5_next_kv_cache_config(config, groups, 1 << 30)
    layout = build_layerwise_reuse_layout(get_layerwise_kv_cache_specs(plan), 45, extra)
    assert layout.independent_layers == [0, 1, 2, 3]
    assert set(layout.buffer_slots[0]) == {0, 1, 2, 3}
    assert tuple(tuple(sorted(slot)) for slot in layout.buffer_slots[1:]) == tuple(
        tuple(layer for start in range(4 + 4 * slot, 45, 4 * buffers) for layer in range(start, min(start + 4, 45)))
        for slot in range(buffers)
    )
    assert layout.prefetch_layer_map == {layer: layer - 4 * buffers for layer in range(4 + 4 * buffers, 45)}
    assert layout.num_prefetch_layers == 8  # Logical layers, independent of the raw tensor count.
    main = [t for t in plan.kv_cache_tensors if not any(".indexer." in name for name in t.layers)]
    assert len(main) == buffers + 1
    assert set(main[0].layers) == {"model.layers.3.attn", *(f"model.layers.{i}.linear_attn" for i in range(3))}
    assert len(plan.kv_cache_tensors) == buffers + 12
    placement = {name: tensor for tensor in main for name in tensor.layers}
    for start in range(4, 45, 4):
        pack_names = [name for name in placement if start <= int(name.split("layers.")[1].split(".")[0]) < start + 4]
        assert len({id(placement[name]) for name in pack_names}) == 1
    if buffers == 4:
        assert layout.prefetch_layer_map[20] == 4
        assert layout.prefetch_layer_map[21] == 5
        assert layout.prefetch_layer_map[44] == 28
        assert "model.layers.29.linear_attn" in placement["model.layers.44.linear_attn"].layers


@pytest.mark.parametrize("independent,buffers", [("all", 4), ([0, 1, 2, 3], 11), ([], 12)])
def test_glm_ring_without_physical_savings_keeps_ordinary_layout(independent, buffers):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        build_layerwise_reuse_layout,
        get_layerwise_kv_cache_specs,
    )

    config = make_config()
    config.model_config.get_num_layers = lambda _: 45
    groups = get_glm5_next_kv_cache_groups(config, make_full_prefill_specs())
    ordinary = get_glm5_next_kv_cache_config(config, groups, 1 << 30)
    extra = {
        "backend": "memcache",
        "use_layerwise": True,
        "layerwise_num_shared_buffers": buffers,
        "layerwise_independent_layers": independent,
    }
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="AscendStoreConnector", kv_role="kv_producer", kv_connector_extra_config=extra
    )
    plan = get_glm5_next_kv_cache_config(config, groups, 1 << 30)
    assert plan.kv_cache_tensors == ordinary.kv_cache_tensors
    assert not getattr(plan, "layerwise_kv_cache_reuse", False)
    assert not build_layerwise_reuse_layout(get_layerwise_kv_cache_specs(plan), 45, extra).has_layer_reuse


def test_glm_draft_inventory_is_resident_and_charged_to_capacity():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        build_layerwise_reuse_layout,
        get_layerwise_kv_cache_specs,
    )

    config = make_config()
    config.model_config.get_num_layers = lambda _: 45
    config.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    extra = {
        "backend": "memcache",
        "use_layerwise": True,
        "layerwise_num_shared_buffers": 4,
        "layerwise_independent_layers": [0, 1, 2, 3],
    }
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="AscendStoreConnector", kv_role="kv_producer", kv_connector_extra_config=extra
    )
    specs = make_full_prefill_specs()
    for name, spec in make_specs(pool=4).items():
        if not isinstance(spec, MambaSpec):
            specs[name.replace("layers.3.", "layers.45.")] = replace(spec)
    groups = get_glm5_next_kv_cache_groups(config, specs)
    plan = get_glm5_next_kv_cache_config(config, groups, 1 << 30)
    layout = build_layerwise_reuse_layout(get_layerwise_kv_cache_specs(plan), 45, extra)
    assert set(layout.layer_cache_specs) == set(range(45))
    assert 45 not in layout.prefetch_layer_map
    draft = [t for t in plan.kv_cache_tensors if any("layers.45." in name for name in t.layers)]
    assert len(draft) == 2
    assert all(all("layers.45." in name for name in t.layers) for t in draft)
    assert len(plan.kv_cache_tensors) == 18  # 5+11 target and 1+1 draft
    ordinary = _get_glm5_next_cache_layout(groups)
    cost = 6 * ordinary.main_page_size + 12 * ordinary.small_page_size
    assert get_glm5_next_pool_bytes_per_block(groups, config) == cost
    assert sum(t.size for t in plan.kv_cache_tensors) == cost * plan.num_blocks


@pytest.mark.parametrize(
    "unsupported", ["consumer", "pd", "data_parallel", "pipeline", "prefill_cp", "decode_cp", "graph", "v2", "kv_dtype"]
)
def test_prefill_reuse_rejects_unsupported_execution_before_allocation(unsupported):
    config = make_config()
    config.model_config.get_num_layers = lambda _: 45
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="AscendStoreConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={"backend": "memcache", "use_layerwise": True, "layerwise_num_shared_buffers": 3},
    )
    if unsupported == "consumer":
        config.kv_transfer_config.kv_role = "kv_consumer"
    elif unsupported == "pd":
        transfer = config.kv_transfer_config
        transfer.kv_connector_extra_config = {
            "connectors": [
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_connector_extra_config": transfer.kv_connector_extra_config,
                }
            ]
        }
        transfer.kv_connector = "MultiConnector"
    elif unsupported == "graph":
        config.model_config.enforce_eager = False
    elif unsupported == "v2":
        config.use_v2_model_runner = True
    elif unsupported == "kv_dtype":
        pass
    else:
        field = {
            "data_parallel": "data_parallel_size",
            "pipeline": "pipeline_parallel_size",
            "prefill_cp": "prefill_context_parallel_size",
            "decode_cp": "decode_context_parallel_size",
        }[unsupported]
        setattr(config.parallel_config, field, 2)
    specs = make_full_prefill_specs()
    if unsupported == "kv_dtype":
        specs["model.layers.3.attn"] = replace(specs["model.layers.3.attn"], dtype=torch.int8)
    groups = get_glm5_next_kv_cache_groups(config, specs)
    with pytest.raises((ValueError, NotImplementedError)):
        get_glm5_next_kv_cache_config(config, groups, 1 << 30)


@pytest.fixture(autouse=True)
def register_cache_specs():
    # Match production: vLLM registers built-in specs before the Ascend hook.
    register_all_kvcache_specs(None)


@pytest.mark.parametrize("pool", [4, 16])
def test_groups_share_block_ids_and_pack_two_page_classes(pool):
    config = make_config()
    specs = make_specs(pool)
    groups = get_glm5_next_kv_cache_groups(config, dict(reversed(list(specs.items()))))
    layout = _get_glm5_next_cache_layout(groups)
    assert layout is not None
    assert groups[0].layer_names == [
        "model.layers.3.attn",
        "model.layers.3.indexer.k_cache",
    ]
    assert len(groups) == 5  # full, state, and three interleaved KDA groups
    assert layout.main_slot_count == layout.small_slot_count == 1

    bytes_per_block = layout.main_page_size + layout.small_page_size
    budget = 20 * bytes_per_block
    plan = get_glm5_next_kv_cache_config(config, groups, budget)
    assert plan.num_blocks == 20
    assert len(plan.kv_cache_tensors) == 2
    assert all(isinstance(tensor, KVCacheTensor) for tensor in plan.kv_cache_tensors)
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {
        20 * layout.main_page_size,
        20 * layout.small_page_size,
    }
    assert all(
        tensor.offset == 0 and tensor.layer_stride == 0 and tensor.block_stride == tensor.size // plan.num_blocks
        for tensor in plan.kv_cache_tensors
    )

    placements = {
        layer_name: tensor for tensor in plan.kv_cache_tensors for layer_name in get_kv_cache_tensor_layers(tensor)
    }
    main = placements[layout.mla_names[0]]
    indexer = placements[layout.indexer_names[0]]
    state = placements[layout.tail_names[0]]
    assert main.offset == 0
    assert indexer.offset == 0
    assert state is indexer
    assert set(get_kv_cache_tensor_layers(main)) == {
        layout.mla_names[0],
        *(group.layer_names[0] for group in layout.mamba_groups),
    }
    assert set(get_kv_cache_tensor_layers(indexer)) == {
        layout.indexer_names[0],
        layout.tail_names[0],
    }

    # Scheduler groups consume disjoint IDs from the shared global BlockPool.
    required_blocks = sum(
        (group.kv_cache_spec.max_memory_usage_bytes(config) + group.kv_cache_spec.page_size_bytes - 1)
        // group.kv_cache_spec.page_size_bytes
        for group in groups
    )
    assert get_glm5_next_pool_bytes_per_block(groups) == bytes_per_block
    assert get_glm5_next_max_memory_usage(config, groups) == required_blocks * bytes_per_block


def test_standalone_mtp_layout_has_no_mamba_groups():
    specs = {name: spec for name, spec in make_specs().items() if not isinstance(spec, MambaSpec)}
    groups = get_glm5_next_kv_cache_groups(make_config(), specs)
    layout = _get_glm5_next_cache_layout(groups)
    assert len(groups) == 2
    assert layout is not None
    assert layout.mamba_groups == ()


def test_pipeline_projection_supports_a_mamba_only_worker():
    config = make_config()
    groups = get_glm5_next_kv_cache_groups(config, make_specs())
    local_mamba_name = groups[2].layer_names[0]
    projected_groups = [
        KVCacheGroupSpec(
            [local_mamba_name] if group is groups[2] else [],
            group.kv_cache_spec,
        )
        for group in groups
    ]

    layout = _get_glm5_next_cache_layout(projected_groups)
    assert layout is not None
    assert layout.mla_names == layout.indexer_names == layout.tail_names == ()
    assert layout.main_slot_count == 1
    assert layout.small_slot_count == 0

    budget = 10 * layout.main_page_size
    plan = get_glm5_next_kv_cache_config(config, projected_groups, available_memory=budget)
    assert plan.num_blocks == 10
    assert len(plan.kv_cache_tensors) == 1
    assert get_kv_cache_tensor_layers(plan.kv_cache_tensors[0]) == [local_mamba_name]


def test_missing_paired_cache_is_rejected():
    specs = make_specs()
    del specs["model.layers.3.indexer.tail_cache"]
    with pytest.raises(ValueError, match="requires"):
        get_glm5_next_kv_cache_groups(make_config(), specs)


def test_misaligned_logical_block_is_rejected():
    specs = make_specs(pool=16)
    object.__setattr__(
        specs["model.layers.3.indexer.k_cache"],
        "tokens_per_state",
        15,
    )
    with pytest.raises(ValueError, match="divisible"):
        get_glm5_next_kv_cache_groups(make_config(), specs)


@pytest.mark.parametrize("use_sparse", [False, True])
@pytest.mark.parametrize("packed_main", [False, True])
def test_worker_preserves_glm5_next_physical_pool_budget(monkeypatch, use_sparse, packed_main):
    from vllm_ascend.worker.worker import NPUWorker

    config = make_config()
    config.model_config.max_model_len = 1048576
    config.model_config.hf_text_config = SimpleNamespace(model_type="glm5_next", index_topk=2048, index_kpool=16)
    config.cache_config.get_resolved_kv_cache_layout = lambda: SimpleNamespace(
        is_layer_compact=True, is_block_compact=True
    )
    specs = make_specs()
    if packed_main:
        specs["model.layers.3.attn"] = AscendMLAAttentionSpec(
            block_size=512,
            num_kv_heads=1,
            head_size=528,
            dtype=torch.int8,
            cache_sparse_sfa_c8=True,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
        )
    groups = get_glm5_next_kv_cache_groups(config, specs)
    # Supply exactly enough physical memory for the configured context. The
    # generic discount incorrectly makes this valid shared pool fail admission.
    required = get_glm5_next_max_memory_usage(config, groups)
    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = config
    worker.model_config = config.model_config
    worker.get_kv_cache_spec = lambda: specs
    worker.model_runner = SimpleNamespace(
        use_sparse=use_sparse,
        use_compress=False,
        supports_standardized_shared_kv_backing=True,
    )
    monkeypatch.setattr("vllm_ascend.worker.worker.get_kv_cache_groups", lambda *_args: groups)
    budget = worker._scale_kv_cache_memory_for_multi_group(required)

    assert budget == required
    plan = get_glm5_next_kv_cache_config(config, groups, budget)
    assert plan.num_blocks * get_glm5_next_pool_bytes_per_block(groups) == required
    assert sum(tensor.size for tensor in plan.kv_cache_tensors) == required
