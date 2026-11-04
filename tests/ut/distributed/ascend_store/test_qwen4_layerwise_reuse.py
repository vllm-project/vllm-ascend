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
    build_qwen4_kv_cache_layout,
    get_qwen4_exp_kv_cache_config,
)
from vllm_ascend.models.qwen4_exp.layerwise import (
    allocate_qwen4_cache,
    apply_qwen4_layerwise_kv_cache_plan,
    bind_qwen4_layerwise_cache_layout,
    build_qwen4_reuse_layout,
    view_qwen4_cache,
)

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
    assert len(kv_config.kv_cache_tensors) == 7
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
        assert layout.cache_slot_map[indexer_name(i, "compressed")] == ordinal


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
    assert len({views[indexer_name(i, "compressed")].data_ptr() for i in QSA_LAYERS}) == 12
    assert allocated[0] == kv_config.kv_cache_tensors[0].size
    assert len(allocated) == 2  # Packed backing and private PLE.


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
