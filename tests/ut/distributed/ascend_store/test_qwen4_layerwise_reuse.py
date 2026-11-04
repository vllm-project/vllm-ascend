# SPDX-License-Identifier: Apache-2.0
"""Packed Qwen owner bindings, explicit dispatch, and normal GVA restores."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import LayerBatchBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    get_layerwise_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import SharedBlockData
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.models.layerwise import get_model_layerwise_handler
from vllm_ascend.models.qwen4_exp.cache_config import (
    GDN,
    PLE,
    QSA_COMPRESSED,
    QSA_MAIN,
    QWEN4_CACHE_ALIGNMENT,
    build_qwen4_kv_cache_layout,
    get_qwen4_exp_allocation_overhead,
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_max_memory_usage,
    get_qwen4_exp_pool_bytes_per_block,
)
from vllm_ascend.models.qwen4_exp.layerwise import (
    allocate_qwen4_cache,
    apply_qwen4_layerwise_kv_cache_plan,
    bind_qwen4_layerwise_cache_layout,
    build_qwen4_reuse_layout,
    view_qwen4_cache,
)
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.worker import NPUWorker

PREFIX = "model.layers"
QSA_LAYERS = tuple(range(3, 48, 4))


def main_name(layer):
    suffix = "self_attn.attn" if layer in QSA_LAYERS else "linear_attn"
    return f"{PREFIX}.{layer}.{suffix}"


def indexer_name(layer, kind):
    return f"{PREFIX}.{layer}.self_attn.indexer.{kind}_key_cache"


def make_config():
    main = FullAttentionSpec(block_size=256, num_kv_heads=1, head_size=64, head_size_v=64, dtype=torch.bfloat16)
    raw = CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=32, head_size_v=0, dtype=torch.bfloat16)
    compressed = MLAAttentionSpec(
        block_size=256, num_kv_heads=1, head_size=32, dtype=torch.bfloat16, tokens_per_state=4
    )
    gdn = MambaSpec(block_size=256, shapes=((4, 64), (64, 128)), dtypes=(torch.bfloat16, torch.float32))
    ple = MambaSpec(block_size=256, shapes=((3, 64),), dtypes=(torch.bfloat16,))
    composite = {
        name: spec
        for i in QSA_LAYERS
        for name, spec in ((main_name(i), main), (indexer_name(i, "compressed"), compressed))
    }
    groups = [
        KVCacheGroupSpec([indexer_name(i, "raw") for i in QSA_LAYERS], raw),
        KVCacheGroupSpec(list(composite), UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=composite)),
    ]
    gdn_layers = [i for i in range(48) if i not in QSA_LAYERS]
    groups.extend(
        KVCacheGroupSpec([main_name(i) for i in gdn_layers[start : start + 12]], gdn) for start in (0, 12, 24)
    )
    groups.append(KVCacheGroupSpec([f"{PREFIX}.1.ple"], ple))
    extra = {
        "backend": "memcache",
        "use_layerwise": True,
        "layerwise_num_shared_buffers": 2,
        "layerwise_independent_layers": [0],
    }
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_layers=lambda _: 48,
            architecture="Qwen4ExpForCausalLM",
            hf_config=SimpleNamespace(architectures=["Qwen4ExpForCausalLM"]),
        ),
        parallel_config=None,
        kv_transfer_config=SimpleNamespace(kv_connector="AscendStoreConnector", kv_connector_extra_config=extra),
        cache_config=SimpleNamespace(num_gpu_blocks_override=3, prefix_cache_retention_interval=1),
    )
    kv_config = get_qwen4_exp_kv_cache_config(config, groups, available_memory=10_000_000)
    assert kv_config is not None
    return config, kv_config


def test_qwen_plan_keeps_descriptors_and_auxiliary_slots():
    config, kv_config = make_config()
    before = deepcopy(kv_config)
    layout = apply_layerwise_kv_cache_plan(kv_config, config, model_specific_plan=apply_qwen4_layerwise_kv_cache_plan)
    assert layout is not None
    assert kv_config == before
    assert {name for tensor in kv_config.kv_cache_tensors for name in tensor.layers} == set(
        get_layerwise_kv_cache_specs(kv_config)
    )
    assert layout.buffer_slots == ((0,), tuple(range(1, 48, 2)), tuple(range(2, 48, 2)))
    assert layout.prefetch_layer_map == {i: i - 2 for i in range(3, 48)}
    names = []
    for i, layer in layout.layer_cache_specs.items():
        assert layer.main.layer_name == main_name(i)
        names.append(layer.main.layer_name)
        names.extend(spec.layer_name for spec in layer.extra_main_specs)
        if i in QSA_LAYERS:
            assert layer.indexer.layer_name == indexer_name(i, "compressed")
            assert [spec.layer_name for spec in layer.extra_main_specs] == [indexer_name(i, "raw")]
            names.append(layer.indexer.layer_name)
        else:
            assert layer.indexer is None
    assert len(names) == len(set(names)) == 73
    assert set(names) == set(get_layerwise_kv_cache_specs(kv_config))
    assert layout.layer_cache_specs[1].extra_main_specs[0].layer_name == f"{PREFIX}.1.ple"


@pytest.mark.parametrize("buffers,independent", [(1, []), (2, [0]), (3, [0, 1])])
def test_qwen_slot_capacity_and_fences(buffers, independent):
    config, kv_config = make_config()
    extra = config.kv_transfer_config.kv_connector_extra_config
    extra.update(layerwise_num_shared_buffers=buffers, layerwise_independent_layers=independent)
    layout = apply_layerwise_kv_cache_plan(kv_config, config, model_specific_plan=apply_qwen4_layerwise_kv_cache_plan)
    assert layout is not None
    for slot, members in enumerate(layout.buffer_slots):
        for i in members:
            assert layout.cache_slot_map[main_name(i)] == slot
        for previous, current in zip(members, members[1:]):
            assert layout.prefetch_layer_map[current] == previous
    for ordinal, i in enumerate(QSA_LAYERS):
        assert layout.cache_slot_map[indexer_name(i, "raw")] == ordinal
        assert layout.cache_slot_map[indexer_name(i, "compressed")] == layout.cache_slot_map[main_name(i)]


def test_qwen_no_reuse_and_invalid_plans():
    config, kv_config = make_config()
    extra = config.kv_transfer_config.kv_connector_extra_config
    specs = get_layerwise_kv_cache_specs(kv_config)
    extra["layerwise_num_shared_buffers"] = 48
    assert not build_qwen4_reuse_layout(specs, 48, extra).has_layer_reuse
    extra["layerwise_num_shared_buffers"] = 12
    with pytest.raises(ValueError, match="slab count"):
        build_qwen4_reuse_layout(specs, 48, extra)
    extra["layerwise_num_shared_buffers"] = 2
    del specs[indexer_name(3, "compressed")]
    with pytest.raises(ValueError, match="raw and compressed"):
        build_qwen4_reuse_layout(specs, 48, extra)
    with pytest.raises(ValueError, match="original slots"):
        bind_qwen4_layerwise_cache_layout(
            build_qwen4_kv_cache_layout(kv_config.kv_cache_groups, 3), {indexer_name(3, "raw"): 1}
        )


def test_shared_main_state_writes_preserve_auxiliary_regions():
    config, kv_config = make_config()
    reuse = apply_layerwise_kv_cache_plan(kv_config, config, model_specific_plan=apply_qwen4_layerwise_kv_cache_plan)
    physical = bind_qwen4_layerwise_cache_layout(
        build_qwen4_kv_cache_layout(kv_config.kv_cache_groups, 3), reuse.cache_slot_map
    )
    allocated = []

    def allocate(size):
        allocated.append(size)
        return torch.zeros(size, dtype=torch.int8)

    raw = allocate_qwen4_cache(physical, allocate)
    views = {
        owner.layer_name: view_qwen4_cache(physical, owner.layer_name, raw[owner.layer_name], 256)
        for owner in physical.owners
    }
    auxiliary = [views[indexer_name(i, kind)] for i in QSA_LAYERS for kind in ("raw", "compressed")]
    auxiliary.extend(views[f"{PREFIX}.1.ple"])
    auxiliary.extend(views[main_name(0)])
    for i, tensor in enumerate(auxiliary):
        tensor.fill_(i + 1)
    snapshots = [tensor.clone() for tensor in auxiliary]
    for layer in range(1, 48):
        for tensor in views[main_name(layer)]:
            tensor.fill_(layer)
    for tensor, expected in zip(auxiliary, snapshots, strict=True):
        torch.testing.assert_close(tensor, expected)
    assert views[main_name(1)][0].data_ptr() == views[main_name(3)][0].data_ptr()
    assert views[main_name(2)][0].data_ptr() != views[main_name(3)][0].data_ptr()
    assert len({views[indexer_name(i, "compressed")].data_ptr() for i in QSA_LAYERS}) == 1
    assert len({views[indexer_name(i, "raw")].data_ptr() for i in QSA_LAYERS}) == 12
    assert allocated[0] == kv_config.kv_cache_tensors[0].size
    assert len(allocated) == 2  # Packed backing and private PLE.
    original = build_qwen4_kv_cache_layout(kv_config.kv_cache_groups, 3)
    assert original.backing_size - physical.backing_size == 9 * original.slot_backing_size
    assert type(physical) is type(original)
    assert physical.slot_count == 3 and physical.raw_slot_count == original.raw_slot_count == 12
    assert physical.slot_backing_size == original.slot_backing_size
    assert physical.raw_slot_stride == original.raw_slot_stride
    for name in ("r0", "r1", "r2", "r5"):
        assert physical.region(name) == original.region(name)


@pytest.mark.parametrize("buffers,independent", [(1, []), (2, [0]), (3, [0, 1])])
def test_compact_capacity_uses_the_same_hbm_budget(buffers, independent):
    config, kv_config = make_config()
    groups = kv_config.kv_cache_groups
    config.cache_config.num_gpu_blocks_override = None
    extra = config.kv_transfer_config.kv_connector_extra_config
    extra.update(layerwise_num_shared_buffers=buffers, layerwise_independent_layers=independent)
    original_config = deepcopy(config)
    original_config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = False
    original_cost = get_qwen4_exp_pool_bytes_per_block(groups, original_config)
    compact_cost = get_qwen4_exp_pool_bytes_per_block(groups, config)
    probe = build_qwen4_kv_cache_layout(groups, 1)
    assert original_cost - compact_cost == (12 - buffers - len(independent)) * probe.slot_backing_size
    budget = original_cost * 4
    original = get_qwen4_exp_kv_cache_config(original_config, groups, budget)
    compact = get_qwen4_exp_kv_cache_config(config, groups, budget)
    assert compact.num_blocks > original.num_blocks == 4
    assert compact.num_blocks == budget // compact_cost
    physical = build_qwen4_kv_cache_layout(groups, compact.num_blocks, vllm_config=config)
    allocations = []

    def allocate(size):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8)

    raw = allocate_qwen4_cache(physical, allocate)
    assert sum(allocations) == compact.num_blocks * compact_cost <= budget
    assert sum(allocations) + compact_cost > budget
    # Exercise the highest new block, including compressed kernel-page views.
    for owner in physical.owners:
        views = view_qwen4_cache(physical, owner.layer_name, raw[owner.layer_name], 128)
        for view in views if isinstance(views, (list, tuple)) else [views]:
            view[-1].fill_(1)
            assert torch.all(view[-1] == 1)
    # Cross-rank replanning must preserve the chosen minimum block count.
    replanned = get_qwen4_exp_kv_cache_config(config, groups, compact.num_blocks * compact_cost)
    assert replanned.num_blocks == compact.num_blocks


def test_compact_descriptors_match_allocated_views():
    config, kv_config = make_config()
    physical = build_qwen4_kv_cache_layout(kv_config.kv_cache_groups, kv_config.num_blocks, vllm_config=config)
    raw = allocate_qwen4_cache(physical, lambda size: torch.zeros(size, dtype=torch.int8))
    for descriptor in kv_config.kv_cache_tensors:
        for index, name in enumerate(descriptor.layers):
            owner = physical.owner(name)
            views = view_qwen4_cache(physical, name, raw[name], 256)
            anchor = views[0] if isinstance(views, (tuple, list)) else views
            assert descriptor.size == raw[name].numel()
            assert anchor.data_ptr() - raw[name].data_ptr() == descriptor.offset + index * descriptor.layer_stride
            assert anchor.stride(0) * anchor.element_size() == descriptor.block_stride
            if owner.role in (QSA_MAIN, GDN, QSA_COMPRESSED) and len(descriptor.layers) > 1:
                assert descriptor.layer_stride == 0
            if owner.role != PLE:
                region = physical.region(physical.region_name_for_owner(name))
                assert descriptor.offset + index * descriptor.layer_stride + region.size <= descriptor.size


def test_compact_binding_rejects_incomplete_and_noncontiguous_maps():
    config, kv_config = make_config()
    original = build_qwen4_kv_cache_layout(kv_config.kv_cache_groups, 3)
    reuse = build_qwen4_reuse_layout(
        get_layerwise_kv_cache_specs(kv_config), 48, config.kv_transfer_config.kv_connector_extra_config
    )
    incomplete = dict(reuse.cache_slot_map)
    del incomplete[main_name(1)]
    with pytest.raises(ValueError, match="every main/state owner"):
        bind_qwen4_layerwise_cache_layout(original, incomplete)
    noncontiguous = dict(reuse.cache_slot_map)
    for i in range(1, 48, 2):
        noncontiguous[main_name(i)] = 3
    with pytest.raises(ValueError, match="contiguous"):
        bind_qwen4_layerwise_cache_layout(original, noncontiguous)
    mismatched = dict(reuse.cache_slot_map)
    mismatched[indexer_name(3, "compressed")] = 2
    with pytest.raises(ValueError, match="share its main KV slot"):
        bind_qwen4_layerwise_cache_layout(original, mismatched)
    compact = bind_qwen4_layerwise_cache_layout(original, reuse.cache_slot_map)
    assert bind_qwen4_layerwise_cache_layout(compact, reuse.cache_slot_map) == compact


@pytest.mark.parametrize(
    "mode", ["disabled", "missing_count", "all_independent", "other_architecture", "no_connector", "block_key_backend"]
)
def test_disabled_compaction_keeps_original_capacity(mode):
    config, kv_config = make_config()
    extra = config.kv_transfer_config.kv_connector_extra_config
    if mode == "disabled":
        extra["use_layerwise"] = False
    elif mode == "missing_count":
        del extra["layerwise_num_shared_buffers"]
    elif mode == "all_independent":
        extra["layerwise_independent_layers"] = "all"
    elif mode == "other_architecture":
        config.model_config.architecture = "OtherForCausalLM"
    elif mode == "no_connector":
        config.kv_transfer_config = None
    else:
        extra["backend"] = "mooncake"
    groups = kv_config.kv_cache_groups
    assert build_qwen4_kv_cache_layout(groups, 3, vllm_config=config) == build_qwen4_kv_cache_layout(groups, 3)
    assert get_qwen4_exp_pool_bytes_per_block(groups, config) == get_qwen4_exp_pool_bytes_per_block(groups)


def test_compact_admission_and_one_block_boundary(monkeypatch):
    config, kv_config = make_config()
    groups = kv_config.kv_cache_groups
    cost = get_qwen4_exp_pool_bytes_per_block(groups, config)
    config.cache_config.num_gpu_blocks_override = None
    for budget in (cost, 2 * cost - 1):
        assert get_qwen4_exp_kv_cache_config(config, groups, budget).num_blocks == 1
    for group in groups:
        monkeypatch.setattr(
            type(group.kv_cache_spec), "max_memory_usage_bytes", lambda spec, _: 2 * spec.page_size_bytes
        )
    assert get_qwen4_exp_max_memory_usage(config, groups) == len(groups) * 2 * cost
    config.use_v2_model_runner = False
    with pytest.raises(ValueError, match="Model Runner V2"):
        get_qwen4_exp_pool_bytes_per_block(groups, config)


def test_runner_allocates_and_reshapes_the_compact_plan(monkeypatch):
    config, kv_config = make_config()
    config.additional_config = {}
    config.cache_config.cache_dtype = "auto"
    config.model_config.hf_text_config = SimpleNamespace()
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
    raw = attn_utils._allocate_kv_cache(kv_config, {}, torch.device("cpu"))
    specs = get_layerwise_kv_cache_specs(kv_config)
    groups = [
        SimpleNamespace(kv_cache_group_id=i, kv_cache_spec=specs[name], layer_names=[name])
        for i, group in enumerate(kv_config.kv_cache_groups)
        for name in group.layer_names
    ]
    views = attn_utils._reshape_kv_cache_v2(groups, raw, "auto", [128] * len(kv_config.kv_cache_groups), {}, kv_config)
    assert set(views) == set(specs)
    assert views[main_name(1)][0].data_ptr() == views[main_name(3)][0].data_ptr()
    cost = get_qwen4_exp_pool_bytes_per_block(kv_config.kv_cache_groups, config)
    allocations = {tensor.untyped_storage().data_ptr(): tensor.numel() for tensor in raw.values()}
    assert sum(allocations.values()) == kv_config.num_blocks * cost
    for owner_views in views.values():
        for view in owner_views if isinstance(owner_views, (tuple, list)) else [owner_views]:
            view[-1].fill_(1)
            assert torch.all(view[-1] == 1)


def test_worker_keeps_physical_budget_without_double_scaling(monkeypatch):
    config, kv_config = make_config()
    specs = get_layerwise_kv_cache_specs(kv_config)
    worker = object.__new__(NPUWorker)
    worker.vllm_config = config
    worker.model_config = config.model_config
    worker.parallel_config = config.parallel_config
    worker.get_kv_cache_spec = Mock(return_value=specs)
    monkeypatch.setattr("vllm_ascend.worker.worker.get_kv_cache_groups", lambda *_: kv_config.kv_cache_groups)
    budget = 10_000_000
    reserve = get_qwen4_exp_allocation_overhead(kv_config.kv_cache_groups, config)
    assert reserve == 2 * QWEN4_CACHE_ALIGNMENT  # One shared backing and private PLE, not one per descriptor.
    assert worker._scale_kv_cache_memory_for_multi_group(budget) == budget - reserve
    assert worker._scale_kv_cache_memory_for_multi_group(reserve - 1) == 0
    assert worker._get_layerwise_kv_cache_memory_info(specs, config.kv_transfer_config.kv_connector_extra_config) == (
        48,
        3,
        1.0,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_aligned_allocations_fit_the_same_budget(monkeypatch, enabled):
    config, kv_config = make_config()
    config.cache_config.num_gpu_blocks_override = None
    config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = enabled
    config.additional_config = {}
    config.model_config.hf_text_config = SimpleNamespace()
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
    worker = object.__new__(NPUWorker)
    worker.vllm_config = config
    worker.get_kv_cache_spec = Mock(return_value=get_layerwise_kv_cache_specs(kv_config))
    monkeypatch.setattr("vllm_ascend.worker.worker.get_kv_cache_groups", lambda *_: kv_config.kv_cache_groups)
    physical_budget = 10_000_000
    planner_budget = worker._scale_kv_cache_memory_for_multi_group(physical_budget)
    plan = get_qwen4_exp_kv_cache_config(config, kv_config.kv_cache_groups, planner_budget)
    raw = attn_utils._allocate_kv_cache(plan, {}, torch.device("cpu"))
    storage_bytes = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in raw.values()}
    cost = get_qwen4_exp_pool_bytes_per_block(plan.kv_cache_groups, config)
    assert sum(storage_bytes.values()) <= physical_budget
    assert sum(storage_bytes.values()) + cost > physical_budget
    assert all(tensor.data_ptr() % QWEN4_CACHE_ALIGNMENT == 0 for tensor in raw.values())


def test_shared_qsa_kv_and_compressed_roundtrip_preserves_raw_states():
    config, kv_config = make_config()
    physical = build_qwen4_kv_cache_layout(kv_config.kv_cache_groups, 3, vllm_config=config)
    raw = allocate_qwen4_cache(physical, lambda size: torch.zeros(size, dtype=torch.int8))
    views = {
        owner.layer_name: view_qwen4_cache(physical, owner.layer_name, raw[owner.layer_name], 128)
        for owner in physical.owners
    }
    states = [views[indexer_name(i, "raw")] for i in QSA_LAYERS]
    for i, state in enumerate(states):
        state.fill_(i + 1)
    saved = {}
    for i in QSA_LAYERS:
        parts = (*views[main_name(i)], views[indexer_name(i, "compressed")])
        for component, part in enumerate(parts):
            part.fill_(i + component)
        saved[i] = tuple(part.clone() for part in parts)
    # Restore each layer after every other QSA owner has overwritten its slot.
    for i in QSA_LAYERS:
        parts = (*views[main_name(i)], views[indexer_name(i, "compressed")])
        for part, checkpoint in zip(parts, saved[i], strict=True):
            part.copy_(checkpoint)
        for part, checkpoint in zip(parts, saved[i], strict=True):
            torch.testing.assert_close(part, checkpoint)
        for ordinal, state in enumerate(states):
            assert torch.all(state == ordinal + 1)


def test_gva_restore_includes_main_and_compressed_indexer():
    # The QSA group contains two main views and the compressed indexer.
    database = SimpleNamespace(
        group_block_len={0: [8, 8, 16]},
        group_kv_caches_base_addr={0: [100, 200, 300]},
        group_block_stride={0: [32, 32, 64]},
        group_layer_cache_entry_offsets={0: [0, 3]},
    )
    builder = LayerBatchBuilder(database, 32, 1)
    shared = SharedBlockData(np.array([2]), np.array([1000]), ["request"], [True])
    result = builder.build_addrs(shared, 0)
    assert result.addr_array.tolist() == [164, 264, 428]
    assert result.size_array.tolist() == [8, 8, 16]
    assert result.gvas_array.tolist() == [1000, 1008, 1016]


@pytest.mark.parametrize("previous_step", [True, False])
def test_running_and_external_load_restore_whole_qsa_group(previous_step):
    worker = object.__new__(LayerwiseKVPoolWorker)
    worker.use_block_key_layerwise = False
    worker.layerwise_offload = True
    worker.independent_layers = [0]
    worker.grouped_block_size = [256, 256]
    worker.hash_block_size = 256
    worker.num_layers = 48
    worker.layer_load_tasks = {3: []}
    worker._layerwise_load_snapshot_start = Mock(return_value=None)
    worker._resolve_layerwise_load_partial = Mock(return_value=(None, 512))
    worker._masked_block_runs_cached = Mock(return_value=[(0, 2)])
    request = SimpleNamespace(
        load_spec=SimpleNamespace(
            can_load=True,
            kvpool_cached_tokens=512,
            kvpool_store_skip_tokens=None,
            vllm_cached_tokens=512,
            layerwise_previous_step=previous_step,
        ),
        block_hashes=[b"a", b"b"],
        partial_load_gva_per_group=(),
        last_block_gva=None,
        load_masks=None,
    )
    worker._process_load_for_layer_batch([request], 3, 1, 0)
    worker._masked_block_runs_cached.assert_called_once_with("load", request, 1, 0, 2, None)
    (task,) = worker.layer_load_tasks[3]
    assert task.group_id == 1
    assert task.layer_idx_in_group == 0
    (block_range,) = task.block_ranges
    assert block_range.request is request
    assert (block_range.start_block, block_range.end_block) == (0, 2)


@pytest.mark.parametrize("result", [None, object()])
def test_model_callback_returns_directly_without_generic_planning(result):
    # These objects cannot enter the generic path: callback owns all validation.
    kv_config, config = object(), object()
    excluded = {"draft"}
    callback = Mock(return_value=result)
    assert (
        apply_layerwise_kv_cache_plan(kv_config, config, excluded_layer_names=excluded, model_specific_plan=callback)
        is result
    )
    callback.assert_called_once_with(kv_config, config, excluded)


@pytest.mark.parametrize("architecture", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"])
def test_handler_is_selected_by_model_architecture(architecture):
    config, kv_config = make_config()
    config.model_config.architecture = architecture
    handler = get_model_layerwise_handler(config.model_config)
    assert handler.apply_plan is apply_qwen4_layerwise_kv_cache_plan
    assert handler.build_reuse_layout is build_qwen4_reuse_layout
    assert handler.memory_scale == 1.0
    # Dispatch follows the resolved architecture, not a fallback in hf_config.
    config.model_config.architecture = "OtherForCausalLM"
    assert get_model_layerwise_handler(config.model_config) is None
    # Qwen-like cache names alone cannot select the model-specific planner.
    with pytest.raises(NotImplementedError, match="attention cache specs only"):
        apply_layerwise_kv_cache_plan(kv_config, config)


def test_model_plan_disabled_empty_and_excluded_owners():
    config, kv_config = make_config()
    with pytest.raises(ValueError, match="excluded draft"):
        apply_qwen4_layerwise_kv_cache_plan(kv_config, config, {main_name(3)})
    config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = False
    assert apply_qwen4_layerwise_kv_cache_plan(kv_config, config) is None
    config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
    kv_config.kv_cache_tensors = []
    assert apply_qwen4_layerwise_kv_cache_plan(kv_config, config) is None
