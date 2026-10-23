# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.models.qwen4_exp.cache_config import (
    TRANSFER_ALIGNMENT,
    _get_cache_layout,
    _iter_row_slots,
    allocate_qwen4_exp_kv_cache_tensors,
    build_layer_tuples,
    get_qwen4_exp_allocation_overhead,
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_kv_cache_configs,
    get_qwen4_exp_kv_cache_groups,
    get_qwen4_exp_pool_bytes_per_block,
    is_qwen4_exp_cache,
    project_qwen4_exp_cache_groups,
    reshape_qwen4_exp_kv_cache_tensors,
    split_tensor,
)
from vllm_ascend.models.qwen4_exp.qsa import AscendQSACompressedKeyCache, upstream_indexer

_LAYER_TYPES = tuple(["linear_attention"] * 3 + ["qwen_sparse_attention"]) * 12


def _config(num_blocks: int | None = None, cycles: int = 12):
    cache_config = SimpleNamespace(
        num_gpu_blocks_override=num_blocks,
        prefix_cache_retention_interval=None,
        get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(layer_types=list(_LAYER_TYPES[: cycles * 4])),
        ),
        cache_config=cache_config,
    )


def _specs(cycles: int = 12) -> dict[str, KVCacheSpec]:
    main = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    raw = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=130,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    compressed = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
        model_version="qwen4_exp",
    )
    gdn = MambaSpec(
        block_size=128,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        shapes=((1, 1280), (1, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
    )
    ple = MambaSpec(
        block_size=128,
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        shapes=((10240,),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
        tp_replicated=True,
    )
    specs: dict[str, KVCacheSpec] = {}
    for index, layer_type in enumerate(_LAYER_TYPES[: cycles * 4]):
        if layer_type == "linear_attention":
            specs[f"model.layers.{index}.linear_attn"] = gdn
        else:
            source = f"model.layers.{index}.self_attn"
            specs[f"{source}.attn"] = main
            specs[f"{source}.indexer.compressed_key_cache"] = compressed
            specs[f"{source}.indexer.raw_key_cache"] = raw
    specs["model.ple"] = ple
    return specs


def _allocate(config, *, aligned=False):
    backings = []

    def allocator(size, alignment):
        storage = torch.zeros(size + (alignment if aligned else 0), dtype=torch.int8)
        start = (-storage.data_ptr()) % alignment if aligned else 0
        backing = storage[start : start + size]
        backings.append(backing)
        return backing

    raw = allocate_qwen4_exp_kv_cache_tensors(config, allocator)
    return backings, raw, reshape_qwen4_exp_kv_cache_tensors(config, raw)


def test_three_groups_retain_original_specs_and_mixed_tp():
    specs = _specs()
    groups = get_qwen4_exp_kv_cache_groups(_config(), specs)
    assert [len(group.layer_names) for group in groups] == [24, 37, 12]
    topology, _, _ = build_layer_tuples(specs, _LAYER_TYPES)
    assert len(topology) == 12
    assert topology[0].gdn == tuple(f"model.layers.{i}.linear_attn" for i in range(3))
    assert all(type(group) is KVCacheGroupSpec for group in groups)
    assert groups[1].layer_names[-1] == "model.ple"
    for group in groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for name, spec in group.kv_cache_spec.kv_cache_specs.items():
            assert spec is specs[name]
    assert groups[1].kv_cache_spec.kv_cache_specs["model.ple"].tp_replicated
    assert not specs["model.layers.0.linear_attn"].tp_replicated
    # The global uniform rule remains strict; the model-specific merge is local.
    assert UniformTypeKVCacheSpecs.from_specs(groups[1].kv_cache_spec.kv_cache_specs) is None
    assert not groups[2].enable_kv_transfer
    assert not groups[2].kv_cache_spec.prefix_cacheable


@pytest.mark.parametrize("cycles", [1, 2, 12])
def test_topology_is_order_independent_and_serializable(cycles):
    specs, model_config = _specs(cycles), _config(cycles=cycles)
    groups = get_qwen4_exp_kv_cache_groups(model_config, specs)
    reverse = get_qwen4_exp_kv_cache_groups(model_config, dict(reversed(list(specs.items()))))
    assert groups == reverse == pickle.loads(pickle.dumps(groups)) == copy.deepcopy(groups)
    rows = _get_cache_layout(groups)
    assert rows == ((65536, max(cycles, 3)), (65536, 3 * cycles), (2560, 3 * cycles), (8192, cycles))
    assert _get_cache_layout(reverse) == rows
    assert list(_iter_row_slots(reverse)) == list(_iter_row_slots(groups))
    config = get_qwen4_exp_kv_cache_config(model_config, groups, 3 * get_qwen4_exp_pool_bytes_per_block(groups))
    assert config == pickle.loads(pickle.dumps(config)) == copy.deepcopy(config)


@pytest.mark.parametrize("aligned", [False, True])
def test_two_backings_split_rows_budget_and_model_binding(aligned):
    model_config = _config(cycles=1)
    model_config.kv_transfer_config = object() if aligned else None
    groups = get_qwen4_exp_kv_cache_groups(model_config, _specs(1))
    geometry = _get_cache_layout(groups)
    row_sizes = [3 * page * slots for page, slots in geometry]
    overhead = get_qwen4_exp_allocation_overhead(model_config, groups)
    assert overhead == (2 * TRANSFER_ALIGNMENT if aligned else 0)
    config = get_qwen4_exp_kv_cache_config(model_config, groups, sum(row_sizes) + overhead)
    assert config.num_blocks == 3 and config.kv_cache_layout == "LBNHC"
    tensors = config.kv_cache_tensors
    assert len(tensors) == 4 and all(type(t) is KVCacheTensor for t in tensors)
    assert [t.size for t in tensors] == [sum(row_sizes[:3])] * 3 + [row_sizes[3]]
    assert [t.offset for t in tensors] == [0, row_sizes[0], sum(row_sizes[:2]), 0]
    assert [t.block_stride for t in tensors] == [65536, 65536, 2560, 8192]
    assert [t.layer_stride for t in tensors] == [3 * t.block_stride for t in tensors]
    backings, raw, caches = _allocate(config, aligned=aligned)
    assert len(backings) == 2
    assert backings[0].untyped_storage().data_ptr() != backings[1].untyped_storage().data_ptr()
    assert sum(b.untyped_storage().nbytes() for b in backings) == sum(row_sizes) + overhead
    for row, slot, name, component in _iter_row_slots(groups):
        cache = caches[name]
        view = cache[component] if isinstance(cache, tuple) else cache
        descriptor = tensors[row]
        backing = backings[int(row == 3)]
        assert view.data_ptr() == backing.data_ptr() + descriptor.offset + slot * descriptor.layer_stride
        assert view.data_ptr() == raw[name][component].data_ptr()
        assert view.stride(0) * view.element_size() == descriptor.block_stride
        assert view.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    assert caches["model.layers.3.self_attn.attn"][0].shape == (3, 128, 1, 256)
    assert caches["model.layers.3.self_attn.indexer.compressed_key_cache"].shape == (3, 1, 32, 128)
    assert caches["model.ple"][0].data_ptr() == backings[0].data_ptr() + 2 * tensors[0].layer_stride


def test_group_payload_isolation_aliasing_and_padding():
    groups = get_qwen4_exp_kv_cache_groups(_config(cycles=2), _specs(2))
    config = get_qwen4_exp_kv_cache_config(_config(cycles=2), groups, 4 * get_qwen4_exp_pool_bytes_per_block(groups))
    backings, raw, caches = _allocate(config)
    masks = [torch.zeros_like(backing, dtype=torch.bool) for backing in backings]
    for group_id, group in enumerate(groups):
        for marker, name in enumerate(group.layer_names, 1):
            views = caches[name] if isinstance(caches[name], tuple) else (caches[name],)
            for view, byte_view in zip(views, raw[name], strict=True):
                view[group_id].fill_(marker)
                backing_id = int(view.untyped_storage().data_ptr() == backings[1].untyped_storage().data_ptr())
                start = byte_view[group_id].data_ptr() - backings[backing_id].data_ptr()
                size = byte_view.shape[1]
                assert not masks[backing_id][start : start + size].any()
                masks[backing_id][start : start + size] = True
    for group_id, group in enumerate(groups):
        for marker, name in enumerate(group.layer_names, 1):
            for view in caches[name] if isinstance(caches[name], tuple) else (caches[name],):
                assert torch.all(view[group_id] == marker)
                assert not torch.count_nonzero(view[3])
    for backing, mask in zip(backings, masks, strict=True):
        assert not torch.count_nonzero(backing[~mask])
    key, value = caches["model.layers.3.self_attn.attn"]
    ring = caches["model.layers.3.self_attn.indexer.raw_key_cache"]
    assert value.data_ptr() == ring.data_ptr()
    assert key.data_ptr() == caches["model.layers.0.linear_attn"][1].data_ptr()
    ring[0].fill_(7)
    assert value[0].flatten()[0] == 7
    assert not torch.count_nonzero(key[3])


def test_allocator_storage_offset_is_included():
    groups = get_qwen4_exp_kv_cache_groups(_config(cycles=1), _specs(1))
    config = get_qwen4_exp_kv_cache_config(_config(2, cycles=1), groups, 10**7)
    storages = []

    def allocator(size, alignment):
        storage = torch.zeros(size + 32, dtype=torch.int8)
        storages.append(storage)
        return storage[16 : 16 + size]

    raw = allocate_qwen4_exp_kv_cache_tensors(config, allocator)
    ple = raw["model.ple"][0]
    assert ple.data_ptr() == storages[0].data_ptr() + 16 + 2 * config.kv_cache_tensors[0].layer_stride
    ple.fill_(1)
    assert not storages[0][:16].any() and not storages[0][-16:].any()


@pytest.mark.parametrize("sizes", [(16, 16, 16), (0, 32, 16)])
def test_split_is_contiguous_disjoint_and_zero_copy(sizes):
    storage = torch.zeros(80, dtype=torch.int8)
    backing = storage[16:64]
    parts = split_tensor(backing, sizes)
    start = backing.data_ptr()
    for part, size in zip(parts, sizes, strict=True):
        assert part.is_contiguous()
        assert part.untyped_storage().data_ptr() == storage.data_ptr()
        if size:
            assert part.data_ptr() == start
        start += size
    parts[-1].fill_(11)
    assert torch.all(storage[48:64] == 11)
    assert not torch.count_nonzero(storage[:48])


@pytest.mark.parametrize("sizes", [(16, 16), (15, 17, 16), (-16, 64), ()])
def test_invalid_split_boundaries(sizes):
    with pytest.raises(ValueError):
        split_tensor(torch.zeros(48, dtype=torch.int8), sizes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_size", 256),
        ("mamba_cache_mode", "all"),
        ("num_speculative_blocks", 5),
        ("num_prefill_checkpoint_blocks", 1),
        ("prefill_checkpoint_alignment", 16),
    ],
)
def test_gdn_ple_lifecycle_mismatch_is_rejected(field, value):
    specs = _specs(1)
    specs["model.ple"] = replace(specs["model.ple"], **{field: value})
    with pytest.raises(ValueError, match="lifetimes"):
        get_qwen4_exp_kv_cache_groups(_config(cycles=1), specs)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_gdn",
        "duplicate_gdn",
        "missing_qsa",
        "missing_ring",
        "wrong_qsa",
        "multiple_ple",
        "topology",
        "coverage",
    ],
)
def test_invalid_topology_fails_before_allocation(failure):
    specs, model_config = _specs(1), _config(cycles=1)
    if failure == "missing_gdn":
        del specs["model.layers.0.linear_attn"]
    elif failure == "duplicate_gdn":
        specs["model.layers.0.duplicate"] = specs["model.layers.0.linear_attn"]
    elif failure == "missing_qsa":
        specs = {name: spec for name, spec in specs.items() if ".layers.3." not in name}
    elif failure == "missing_ring":
        del specs["model.layers.3.self_attn.indexer.raw_key_cache"]
    elif failure == "wrong_qsa":
        name = "model.layers.3.self_attn.indexer.raw_key_cache"
        specs[name.replace("self_attn", "other_attn")] = specs.pop(name)
    elif failure == "multiple_ple":
        specs["model.ple_copy"] = specs["model.ple"]
    elif failure == "topology":
        model_config.model_config.hf_text_config.layer_types[0] = "qwen_sparse_attention"
    else:
        name = "model.layers.3.self_attn.indexer.compressed_key_cache"
        specs[name] = replace(specs[name], block_size=256)
    with pytest.raises(ValueError):
        get_qwen4_exp_kv_cache_groups(model_config, specs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mixed_state_dtypes_and_component_order(dtype):
    specs = _specs(1)
    for name, spec in list(specs.items()):
        if name.endswith(".linear_attn"):
            specs[name] = replace(spec, dtypes=(torch.bfloat16, dtype))
    groups = get_qwen4_exp_kv_cache_groups(_config(cycles=1), specs)
    config = get_qwen4_exp_kv_cache_config(_config(cycles=1), groups, 2 * get_qwen4_exp_pool_bytes_per_block(groups))
    _, _, caches = _allocate(config)
    conv, ssm = caches["model.layers.0.linear_attn"]
    assert conv.dtype == torch.bfloat16 and ssm.dtype == dtype
    assert conv.shape == (2, 1, 1280) and ssm.shape == (2, 1, 128, 128)
    assert conv.stride(0) * conv.element_size() != ssm.stride(0) * ssm.element_size()


def test_pipeline_projection_preserves_topology_group_ids_and_local_specs():
    specs = _specs(2)
    groups = get_qwen4_exp_kv_cache_groups(_config(cycles=2), specs)
    local = {name: spec for name, spec in specs.items() if ".layers.4." in name}
    projected = project_qwen4_exp_cache_groups(groups, local)
    assert len(projected) == 3
    assert projected[0].layer_names == projected[2].layer_names == []
    assert projected[1].layer_names == ["model.layers.4.linear_attn"]
    assert all(type(group) is KVCacheGroupSpec for group in projected)
    assert is_qwen4_exp_cache(pickle.loads(pickle.dumps(projected)))
    page = get_qwen4_exp_pool_bytes_per_block(projected)
    assert page < get_qwen4_exp_pool_bytes_per_block(groups)
    config = get_qwen4_exp_kv_cache_config(_config(cycles=2), projected, 2 * page)
    backings, _, caches = _allocate(config)
    assert len(backings) == 1 and caches.keys() == local.keys()
    assert project_qwen4_exp_cache_groups(groups, {}) == []
    assert {(row, slot) for row, slot, _, _ in _iter_row_slots(projected)} == {(1, 3), (2, 3)}


@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("draft_prefix", ["model.layers.4.", "model.mtp.layers.0."])
def test_mtp_and_hidden_keep_separate_lifetimes(draft_prefix, aligned):
    model_config = _config(cycles=1)
    model_config.kv_transfer_config = object() if aligned else None
    specs = _specs(1)
    for name, spec in list(specs.items()):
        if ".layers.3." in name:
            specs[name.replace("model.layers.3.", draft_prefix)] = spec
    specs["hidden"] = HiddenStateCacheSpec(block_size=128, num_kv_heads=1, head_size=256, dtype=torch.bfloat16)
    groups = get_qwen4_exp_kv_cache_groups(model_config, specs)
    assert [len(g.layer_names) for g in groups] == [2, 4, 1, 2, 1, 1]
    assert [g.is_eagle_group for g in groups] == [False, False, False, True, True, False]
    overhead = get_qwen4_exp_allocation_overhead(model_config, groups)
    assert overhead == (3 * TRANSFER_ALIGNMENT if aligned else 0)
    budget = 2 * get_qwen4_exp_pool_bytes_per_block(groups) + overhead
    config = get_qwen4_exp_kv_cache_config(model_config, groups, budget)
    assert config.num_blocks == 2
    assert len(config.kv_cache_tensors) == 5
    backings, _, caches = _allocate(config, aligned=aligned)
    assert len(backings) == 3
    assert sum(backing.untyped_storage().nbytes() for backing in backings) == budget
    assert caches["hidden"].untyped_storage().data_ptr() == backings[2].untyped_storage().data_ptr()


def test_scheduler_representative_preserves_block_demand():
    config = _config(cycles=1)
    config.model_config.max_model_len = 512
    config.parallel_config = SimpleNamespace(decode_context_parallel_size=1, tensor_parallel_size=1)
    config.cache_config.mamba_cache_mode = "align"
    specs = _specs(1)
    groups = get_qwen4_exp_kv_cache_groups(config, specs)
    counts = []
    for group in groups:
        uniform = group.kv_cache_spec
        representative = uniform.first_spec
        count = uniform.max_memory_usage_bytes(config) // uniform.page_size_bytes
        assert count == representative.max_memory_usage_bytes(config) // representative.page_size_bytes
        assert uniform.max_num_blocks_per_req(config, 512) == representative.max_num_blocks_per_req(config, 512)
        counts.append(count)
    assert counts == [4, 5, 1]


def test_capacity_and_descriptor_validation():
    groups = get_qwen4_exp_kv_cache_groups(_config(cycles=1), _specs(1))
    page = get_qwen4_exp_pool_bytes_per_block(groups)
    with pytest.raises(ValueError, match="Insufficient"):
        get_qwen4_exp_kv_cache_config(_config(cycles=1), groups, page - 1)
    config = get_qwen4_exp_kv_cache_config(_config(2, cycles=1), groups, page)
    assert config.num_blocks == 2
    config.kv_cache_tensors[0].size -= 1
    with pytest.raises(ValueError, match="descriptor size"):
        _allocate(config)


@pytest.mark.parametrize("layout", [KVCacheLayout.BLNHC, KVCacheLayout.BLHNC])
def test_reject_unsupported_layout(layout):
    config = _config(cycles=1)
    config.cache_config.get_resolved_kv_cache_layout = lambda: layout
    groups = get_qwen4_exp_kv_cache_groups(config, _specs(1))
    with pytest.raises(ValueError, match="requires LBNHC"):
        get_qwen4_exp_kv_cache_config(config, groups, 10**9)


def test_indexer_owner_emits_marked_spec():
    config = _config(cycles=1)
    config.compilation_config = SimpleNamespace(static_forward_context={})
    owner = upstream_indexer.QSACompressedKeyCache(
        head_size=128,
        dtype=torch.bfloat16,
        cache_config=SimpleNamespace(block_size=128),
        prefix="model.layers.3.self_attn.indexer.compressed_key_cache",
        vllm_config=config,
        compress_ratio=4,
    )
    assert isinstance(owner, AscendQSACompressedKeyCache)
    spec = owner.get_kv_cache_spec(config)
    assert (spec.block_size, spec.num_states, spec.page_size_bytes) == (128, 32, 8192)
    assert is_qwen4_exp_cache({"renamed_owner": spec})


@pytest.mark.parametrize("override", [None, 40])
def test_worker_replan_accounts_for_fixed_alignment_overhead(monkeypatch, override):
    from vllm.v1.core import kv_cache_utils as utils

    config = _config(override, cycles=1)
    config.kv_transfer_config = object()
    config.speculative_config = None
    config.model_config.original_max_model_len = 512
    config.model_config.max_model_len = 512
    config.parallel_config = SimpleNamespace(decode_context_parallel_size=1, tensor_parallel_size=1)
    config.cache_config.mamba_cache_mode = "align"
    specs = _specs(1)
    groups = get_qwen4_exp_kv_cache_groups(config, specs)
    page = get_qwen4_exp_pool_bytes_per_block(groups)
    overhead = get_qwen4_exp_allocation_overhead(config, groups)
    monkeypatch.setattr(utils, "get_kv_cache_groups", get_qwen4_exp_kv_cache_groups)
    monkeypatch.setattr(utils, "_project_kv_cache_groups_to_worker", project_qwen4_exp_cache_groups)
    monkeypatch.setattr(utils, "get_kv_cache_config_from_groups", get_qwen4_exp_kv_cache_config)
    # Exercise real admission checks with the model's physical byte formula.
    monkeypatch.setattr(
        utils,
        "_max_memory_usage_bytes_from_groups",
        lambda cfg, grps: 10 * get_qwen4_exp_pool_bytes_per_block(grps) + overhead,
    )
    configs = get_qwen4_exp_kv_cache_configs(config, [specs, specs], [40 * page + overhead, 60 * page + overhead])
    assert [cfg.num_blocks for cfg in configs] == [40, 40]
    assert [cfg.kv_tp_replicas for cfg in configs] == [1, 1]
    assert all(cfg.kv_cache_tensors[0].size + cfg.kv_cache_tensors[3].size == 40 * page for cfg in configs)


def test_worker_replan_preserves_projected_layout_and_alignment(monkeypatch):
    from vllm.v1.core import kv_cache_utils as utils

    config, specs = _config(cycles=2), _specs(2)
    config.kv_transfer_config = object()
    config.model_config.original_max_model_len = 128
    config.model_config.max_model_len = 128
    config.parallel_config = SimpleNamespace(decode_context_parallel_size=1, tensor_parallel_size=1)
    first = {name: spec for name, spec in specs.items() if name == "model.ple" or int(name.split(".")[2]) < 4}
    workers = [first, {name: spec for name, spec in specs.items() if name not in first}]
    groups = get_qwen4_exp_kv_cache_groups(config, specs)
    projected = [project_qwen4_exp_cache_groups(groups, local) for local in workers]
    pages = [get_qwen4_exp_pool_bytes_per_block(local) for local in projected]
    overheads = [get_qwen4_exp_allocation_overhead(config, local) for local in projected]
    monkeypatch.setattr(utils, "get_kv_cache_config_from_groups", get_qwen4_exp_kv_cache_config)
    monkeypatch.setattr(
        utils,
        "_max_memory_usage_bytes_from_groups",
        lambda cfg, local: (
            5 * get_qwen4_exp_pool_bytes_per_block(local) + get_qwen4_exp_allocation_overhead(cfg, local)
        ),
    )
    configs = get_qwen4_exp_kv_cache_configs(
        config, workers, [blocks * page + overhead for blocks, page, overhead in zip((40, 60), pages, overheads)]
    )
    for result, local, page in zip(configs, projected, pages):
        assert result.num_blocks == 40
        assert [group.layer_names for group in result.kv_cache_groups] == [group.layer_names for group in local]
        assert result.kv_cache_tensors[0].size + result.kv_cache_tensors[3].size == 40 * page
        assert result.kv_cache_tensors[1].layer_stride == 40 * result.kv_cache_tensors[1].block_stride


@pytest.mark.parametrize(
    "layer_name",
    [
        "model.layers.3.self_attn.attn",
        "model.layers.3.self_attn.indexer.compressed_key_cache",
        "model.layers.3.self_attn.indexer.raw_key_cache",
        "model.layers.0.linear_attn",
        "model.ple",
    ],
)
def test_partial_pipeline_row_budget_matches_actual_allocations(layer_name):
    model_config, specs = _config(cycles=1), _specs(1)
    model_config.kv_transfer_config = object()
    groups = get_qwen4_exp_kv_cache_groups(model_config, specs)
    projected = project_qwen4_exp_cache_groups(groups, {layer_name: specs[layer_name]})
    budget = 3 * get_qwen4_exp_pool_bytes_per_block(projected) + get_qwen4_exp_allocation_overhead(
        model_config, projected
    )
    config = get_qwen4_exp_kv_cache_config(model_config, projected, budget)
    backings, _, caches = _allocate(config, aligned=True)
    assert config.num_blocks == 3
    assert caches.keys() == {layer_name}
    assert sum(backing.untyped_storage().nbytes() for backing in backings) == budget


@pytest.mark.parametrize("qsa_type", ["full_attention", "qwen_sparse_attention"])
@pytest.mark.parametrize("qsa_index", range(4))
def test_cycle_order_uses_original_model_indices(qsa_index, qsa_type):
    config, original = _config(cycles=1), _specs(1)
    gdn_indices = [i for i in range(4) if i != qsa_index]
    config.model_config.hf_text_config.layer_types = [
        qsa_type if i == qsa_index else "linear_attention" for i in range(4)
    ]
    specs = {}
    for name, spec in original.items():
        if ".layers.3." in name:
            name = name.replace(".layers.3.", f".layers.{qsa_index}.")
        elif name.endswith("linear_attn"):
            old_index = int(name.split(".")[2])
            name = name.replace(f".layers.{old_index}.", f".layers.{gdn_indices[old_index]}.")
        specs[name] = spec
    groups = get_qwen4_exp_kv_cache_groups(config, dict(reversed(list(specs.items()))))
    projected = project_qwen4_exp_cache_groups(
        groups, {f"model.layers.{gdn_indices[1]}.linear_attn": original["model.layers.0.linear_attn"]}
    )
    assert [(row, slot) for row, slot, _, _ in _iter_row_slots(projected)] == [(2, 1), (1, 1)]


def test_heterogeneous_payloads_use_common_row_page_without_copy():
    specs, config = _specs(2), _config(cycles=2)
    specs["model.ple"] = replace(specs["model.ple"], shapes=((40000,),), dtypes=(torch.int32,))
    name = "model.layers.1.linear_attn"
    specs[name] = replace(specs[name], shapes=((1, 1291), (1, 128, 128)), dtypes=(torch.float32, torch.float32))
    groups = get_qwen4_exp_kv_cache_groups(config, specs)
    cache_config = get_qwen4_exp_kv_cache_config(config, groups, 3 * get_qwen4_exp_pool_bytes_per_block(groups))
    _, raw, caches = _allocate(cache_config)
    assert [t.block_stride for t in cache_config.kv_cache_tensors] == [160000, 65536, 5168, 8192]
    assert raw[name][0].shape == (3, 5164)
    assert not caches[name][0].is_contiguous()
    assert caches[name][0].data_ptr() == raw[name][0].data_ptr()
    for row, _, layer, component in _iter_row_slots(groups):
        assert raw[layer][component].stride(0) == cache_config.kv_cache_tensors[row].block_stride


def test_qsa_backends_support_the_resolved_layout():
    from vllm_ascend.models.qwen4_exp.qsa import AscendQSABackend, AscendQSAStateBackend

    assert AscendQSABackend.supported_kv_cache_layouts() == (KVCacheLayout.LBNHC,)
    assert AscendQSAStateBackend.supported_kv_cache_layouts() == (KVCacheLayout.LBNHC,)
    main = _specs(1)["model.layers.3.self_attn.attn"]
    assert AscendQSABackend.customize_spec(main) is main


def test_mixed_page_layout_validation_is_model_specific():
    from vllm.v1.core.kv_cache_utils import validate_kv_cache_layout

    groups = get_qwen4_exp_kv_cache_groups(_config(cycles=1), _specs(1))
    validate_kv_cache_layout(KVCacheLayout.LBNHC, groups)
    with pytest.raises(ValueError, match="requires LBNHC"):
        validate_kv_cache_layout(KVCacheLayout.BLNHC, groups)
    # The generic multi-group planner still rejects mixed-page LBNHC.
    ordinary = [
        KVCacheGroupSpec([str(i)], spec)
        for i, spec in enumerate([_specs(1)["model.layers.3.self_attn.attn"], _specs(1)["model.layers.0.linear_attn"]])
    ]
    with pytest.raises(ValueError, match="mixed page sizes"):
        validate_kv_cache_layout(KVCacheLayout.LBNHC, ordinary)
