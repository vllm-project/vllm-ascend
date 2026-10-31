# /*Copyright (c) 2025 Huawei Technologies Co., Ltd.
# *
# * Licensed under the OpenSSL license (the "License").  You may not use
# * this file except in compliance with the License.  You can obtain a copy
# * in the file LICENSE in the source distribution or at
# * https://www.openssl.org/source/license.html
# */

"""Regression tests for PP stage-local layer numbering."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)


def test_reuse_layout_remap_to_stage_local():
    stage_globals = list(range(38, 79))

    local_map, local_independent = KVPoolWorker._remap_layout_to_stage_local(
        {42: 39, 43: 40, 44: 41, 45: 42}, [38], stage_globals
    )

    assert local_map == {4: 1, 5: 2, 6: 3, 7: 4}
    assert local_independent == [0]
    assert KVPoolWorker._remap_layout_to_stage_local({3: 0, 4: 1}, [0], list(range(38))) == ({3: 0, 4: 1}, [0])


def _make_pp_layer_count_worker(pp_rank, pp_size=2, initial_num_layers=4, use_layerwise_transfer=False, num_groups=2):
    layer_offset = pp_rank * 4
    group_layer_names = [
        [f"model.layers.{layer_offset + i}.attn" for i in range(group_id, 4, num_groups)]
        for group_id in range(num_groups)
    ]
    worker = KVPoolWorker.__new__(KVPoolWorker)
    attention_spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.uint8)
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=names, kv_cache_spec=attention_spec) for names in group_layer_names
        ],
    )
    worker.hf_config = SimpleNamespace(num_hidden_layers=8)
    worker.pp_size = pp_size
    worker.pp_rank = pp_rank
    worker.pp_layer_offset = layer_offset
    worker.total_layers = 8
    worker.num_layers = initial_num_layers
    worker.num_kv_cache_groups = num_groups
    # _init_layerwise_config skips non-cacheable groups; both attention
    # groups here are prefix-cacheable.
    worker.cacheable_group_ids = list(range(num_groups))
    worker.use_layerwise_transfer = use_layerwise_transfer
    worker.use_layerwise = False
    worker._extra_config = {}

    worker._init_layerwise_config()

    assert worker.num_layers == 4
    assert worker.layerwise_key_layers == initial_num_layers
    worker.num_blocks = 2
    worker.kv_caches = {
        layer_name: torch.empty((2, 16, 2, 1, 1), dtype=torch.uint8)
        for names in group_layer_names
        for layer_name in names
    }
    worker.group_kv_caches_base_addr = {}
    worker.group_block_len = {}
    worker.group_block_stride = {}
    worker.group_layer_cache_entry_offsets = {}
    worker.group_cache_entry_segments = {}
    worker.group_num_layers = {}
    for group_id, names in enumerate(group_layer_names):
        worker._infer_cache_group_metadata(group_id, names)

    layers_per_group = 4 // num_groups
    assert worker.group_num_layers == dict.fromkeys(range(num_groups), layers_per_group)
    assert all(len(addresses) == layers_per_group for addresses in worker.group_kv_caches_base_addr.values())
    assert worker.group_layer_cache_entry_offsets == {
        group_id: list(range(layers_per_group + 1)) for group_id in range(num_groups)
    }

    worker.tp_rank = 0
    worker.dcp_size = 1
    worker.put_step = 1
    # _is_layerwise_save_owner gates the save path; attention groups never use
    # align state, and a bare producer owns every save.
    worker.kv_role = "kv_producer"
    worker.consumer_is_to_put = False
    worker.group_uses_align_state = [False] * num_groups
    worker.grouped_block_size = [16] * num_groups
    worker.hash_block_size = 16
    worker.cache_coordinator = None
    worker.use_block_key_layerwise = False
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    # Supply backend allocation results; keep task and address construction real.
    worker._prepare_load_gvas = Mock()
    worker._alloc_gvas_for_save = Mock()
    worker._masked_runs_cache = {}
    # Builders only need the registered geometry; no real backend client.
    worker.token_database = worker
    return worker, layer_offset


@pytest.mark.parametrize("pp_rank", [0, 1])
@pytest.mark.parametrize("is_save", [True, False])
@pytest.mark.parametrize(
    ("initial_num_layers", "use_layerwise_transfer"),
    [(4, False), (4, True), (2, True)],
    ids=["existing-non-layerwise", "gva-no-expansion", "gva-physical-expansion"],
)
def test_pp_stage_maps_multiple_cache_groups_to_local_layers(
    pp_rank, is_save, initial_num_layers, use_layerwise_transfer, monkeypatch
):
    worker, layer_offset = _make_pp_layer_count_worker(
        pp_rank, initial_num_layers=initial_num_layers, use_layerwise_transfer=use_layerwise_transfer
    )
    assert worker.physical_layer_to_group_layers == {
        0: [(0, 0)],
        1: [(1, 0)],
        2: [(0, 1)],
        3: [(1, 1)],
    }
    request = ReqMeta(
        req_id="pp-layer",
        token_len_chunk=16,
        block_ids_by_group=[[1], [1]],
        block_hashes=[b"hash"],
        can_save=is_save,
        load_spec=None if is_save else LoadSpec(0, 16, True),
        block_ids_by_group_np=[np.array([1]), np.array([1])],
        block_gvas_by_group_np=[np.array([4096]), np.array([8192])],
        load_block_gvas_by_group_np=[np.array([4096]), np.array([8192])],
    )
    # This is the real production builder entry point, before request binding.
    builders = worker._build_group_layer_builders()
    worker.process_layer_data([request])
    tasks_by_layer = worker.layer_save_tasks if is_save else worker.layer_load_tasks
    assert len(tasks_by_layer) == 4
    assert all(len(tasks) == 1 for tasks in tasks_by_layer)
    metas = [builders[task.group_id].build(task, is_save) for tasks in tasks_by_layer for task in tasks]
    for local_layer, meta in enumerate(metas):
        cache = worker.kv_caches[f"model.layers.{layer_offset + local_layer}.attn"]
        assert meta is not None
        assert meta.addr_array.tolist() == [cache[1].data_ptr()]

    # Both trailing physical layers must have real tasks and correctly selected
    # cache addresses, even when the original logical extent stopped at layer 2.
    assert worker.layerwise_key_layers == worker.num_layers == 4
    assert tasks_by_layer[-1][0].layer_id == 3
    assert tasks_by_layer[-1][0].layer_idx_in_group == 1
    assert tasks_by_layer[-1][0].final_group_layer
    if is_save:
        # Exercise the real final-layer submission; only device events and the
        # blocking sender drain are stubbed, as no transfer thread is started.
        worker.current_layer = worker.num_layers - 1
        worker.sync_save_events = [Mock() for _ in range(worker.num_layers)]
        worker.layer_save_finished_events = [Mock() for _ in range(worker.num_layers)]
        worker.kv_send_thread = Mock()
        worker._attention_saved_layers = set()
        worker.block_key_hybrid = False
        worker._wait_for_final_layer_save = Mock()
        monkeypatch.setitem(KVPoolWorker.save_kv_layer.__globals__, "_LW_DEFER_LAST_SAVE", False)

        worker.save_kv_layer(None)

        worker.sync_save_events[-1].record.assert_called_once_with()
        worker.kv_send_thread.add_stored_request.assert_called_once_with(request.req_id)
        worker.kv_send_thread.add_request.assert_called_once_with(tasks_by_layer[-1])
        worker._wait_for_final_layer_save.assert_called_once_with(worker.num_layers, worker.kv_send_thread)
        assert worker.current_layer == worker.num_layers


@pytest.mark.parametrize(
    ("pp_size", "use_layerwise_transfer", "num_groups", "expected_key_layers"),
    [(1, True, 2, 2), (2, False, 2, 2), (1, True, 1, 4)],
    ids=["pp1-multigroup-unchanged", "non-layerwise-pp-unchanged", "pp1-single-group-existing-sync"],
)
def test_group_builders_preserve_other_layer_count_branches(
    pp_size, use_layerwise_transfer, num_groups, expected_key_layers
):
    worker, _ = _make_pp_layer_count_worker(
        pp_rank=0,
        pp_size=pp_size,
        initial_num_layers=2,
        use_layerwise_transfer=use_layerwise_transfer,
        num_groups=num_groups,
    )
    original_key_offset = worker.layerwise_key_layer_offset
    original_group_bytes = {group_id: tuple(lengths) for group_id, lengths in worker.group_block_len.items()}

    builders = worker._build_group_layer_builders()

    assert worker.num_layers == 4
    assert worker.layerwise_key_layers == expected_key_layers
    assert worker.layerwise_key_layer_offset == original_key_offset
    assert {group_id: tuple(lengths) for group_id, lengths in worker.group_block_len.items()} == original_group_bytes
    assert len(builders) == num_groups
