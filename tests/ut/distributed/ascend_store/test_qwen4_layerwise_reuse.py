# SPDX-License-Identifier: Apache-2.0
"""Packed Qwen owner bindings and selective GVA restores on CPU storage."""

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
    build_layerwise_reuse_layout,
    get_layerwise_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import SharedBlockData
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.models.qwen4_exp.cache_config import (
    allocate_qwen4_cache,
    build_qwen4_kv_cache_layout,
    get_qwen4_exp_kv_cache_config,
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
        model_config=SimpleNamespace(get_num_layers=lambda _: 48),
        parallel_config=None,
        kv_transfer_config=SimpleNamespace(kv_connector="AscendStoreConnector", kv_connector_extra_config=extra),
        cache_config=SimpleNamespace(num_gpu_blocks_override=3, prefix_cache_retention_interval=1),
    )
    kv_config = get_qwen4_exp_kv_cache_config(config, groups, available_memory=10_000_000)
    assert kv_config is not None
    return config, kv_config


def test_qwen_plan_keeps_descriptors_and_resident_owners():
    config, kv_config = make_config()
    before = deepcopy(kv_config)
    layout = apply_layerwise_kv_cache_plan(kv_config, config)
    assert layout is not None
    assert kv_config == before
    assert len(kv_config.kv_cache_tensors) == 7
    assert layout.buffer_slots == ((0,), tuple(range(1, 48, 2)), tuple(range(2, 48, 2)))
    assert layout.prefetch_layer_map == {i: i - 2 for i in range(3, 48)}
    assert len(layout.reused_cache_names()) == 47
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
    layout = apply_layerwise_kv_cache_plan(kv_config, config)
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
    assert not build_layerwise_reuse_layout(specs, 48, extra).has_layer_reuse
    extra["layerwise_num_shared_buffers"] = 12
    with pytest.raises(ValueError, match="slab count"):
        build_layerwise_reuse_layout(specs, 48, extra)
    extra["layerwise_num_shared_buffers"] = 2
    del specs[indexer_name(3, "compressed")]
    with pytest.raises(ValueError, match="raw and compressed"):
        build_layerwise_reuse_layout(specs, 48, extra)
    with pytest.raises(ValueError, match="original slots"):
        build_qwen4_kv_cache_layout(kv_config.kv_cache_groups, 3, cache_slot_map={indexer_name(3, "raw"): 1})


def test_shared_main_state_writes_preserve_all_resident_caches():
    config, kv_config = make_config()
    reuse = apply_layerwise_kv_cache_plan(kv_config, config)
    physical = build_qwen4_kv_cache_layout(kv_config.kv_cache_groups, 3, cache_slot_map=reuse.cache_slot_map)
    allocated = []

    def allocate(size):
        allocated.append(size)
        return torch.zeros(size, dtype=torch.int8)

    raw = allocate_qwen4_cache(physical, allocate)
    views = {
        owner.layer_name: view_qwen4_cache(physical, owner.layer_name, raw[owner.layer_name], 256)
        for owner in physical.owners
    }
    resident = [views[indexer_name(i, kind)] for i in QSA_LAYERS for kind in ("raw", "compressed")]
    resident.extend(views[f"{PREFIX}.1.ple"])
    resident.extend(views[main_name(0)])
    for i, tensor in enumerate(resident):
        tensor.fill_(i + 1)
    snapshots = [tensor.clone() for tensor in resident]
    for layer in range(1, 48):
        for tensor in views[main_name(layer)]:
            tensor.fill_(layer)
    for tensor, expected in zip(resident, snapshots, strict=True):
        torch.testing.assert_close(tensor, expected)
    assert views[main_name(1)][0].data_ptr() == views[main_name(3)][0].data_ptr()
    assert views[main_name(2)][0].data_ptr() != views[main_name(3)][0].data_ptr()
    assert len({views[indexer_name(i, "compressed")].data_ptr() for i in QSA_LAYERS}) == 12
    assert allocated[0] == kv_config.kv_cache_tensors[0].size
    assert len(allocated) == 2  # Packed backing and private PLE.


def test_selective_gva_restore_preserves_remote_offsets():
    database = SimpleNamespace(
        group_block_len={0: [8, 4, 16]},
        group_kv_caches_base_addr={0: [100, 200, 300]},
        group_block_stride={0: [32, 8, 64]},
        group_layer_cache_entry_offsets={0: [0, 1, 3]},
    )
    builder = LayerBatchBuilder(database, 28, 2)
    shared = SharedBlockData(np.array([2]), np.array([1000]), ["request"], [True])
    # Layer 1's second entry must keep its offset after the resident first entry.
    result = builder.build_addrs(shared, 1, (1,))
    assert result.addr_array.tolist() == [428]
    assert result.size_array.tolist() == [16]
    assert result.gvas_array.tolist() == [1012]
    with pytest.raises(ValueError, match="out of range"):
        builder.build_addrs(shared, 1, (2,))


def test_running_load_excludes_indexer_but_external_hit_includes_it():
    config, kv_config = make_config()
    layout = apply_layerwise_kv_cache_plan(kv_config, config)
    worker = object.__new__(LayerwiseKVPoolWorker)
    worker._layerwise_reuse_layout = layout
    worker.group_layer_cache_entry_offsets = {1: [0, 3]}
    worker.group_cache_entry_names = {1: [main_name(3), main_name(3), indexer_name(3, "compressed")]}
    worker._process_load_for_layer_batch_impl = Mock()
    running = SimpleNamespace(load_spec=SimpleNamespace(layerwise_previous_step=True))
    external = SimpleNamespace(load_spec=SimpleNamespace(layerwise_previous_step=False))
    worker._process_load_for_layer_batch([running, external], 3, 1, 0)
    calls = worker._process_load_for_layer_batch_impl.call_args_list
    assert calls[0].args[0] == [running, external]
    assert calls[0].kwargs == {"cache_entry_indices": (0, 1), "restore_reused": True}
    assert calls[1].args[0] == [external]
    assert calls[1].kwargs == {"cache_entry_indices": (2,), "restore_reused": False}
