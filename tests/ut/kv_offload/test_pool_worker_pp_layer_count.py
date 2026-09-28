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

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import LayerBatchBuilder
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


@pytest.mark.parametrize("pp_rank", [0, 1])
@pytest.mark.parametrize("is_save", [True, False])
def test_pp_stage_maps_multiple_cache_groups_to_local_layers(pp_rank, is_save):
    layer_offset = pp_rank * 4
    group_layer_names = [
        [f"model.layers.{layer_offset + i}.attn" for i in [0, 2]],
        [f"model.layers.{layer_offset + i}.attn" for i in [1, 3]],
    ]
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=names) for names in group_layer_names]
    )
    worker.hf_config = SimpleNamespace(num_hidden_layers=8)
    worker.pp_size = 2
    worker.pp_rank = pp_rank
    worker.pp_layer_offset = layer_offset
    worker.total_layers = 8
    worker.num_layers = 4
    worker.num_kv_cache_groups = 2
    worker.use_layerwise_transfer = False
    worker.use_layerwise = False
    worker._extra_config = {}

    worker._init_layerwise_config()

    assert worker.num_layers == worker.layerwise_key_layers == 4
    assert worker.physical_layer_to_group_layers == {
        0: [(0, 0)],
        1: [(1, 0)],
        2: [(0, 1)],
        3: [(1, 1)],
    }

    worker.num_blocks = 2
    worker.kv_caches = {
        layer_name: torch.empty((2, 2), dtype=torch.uint8) for names in group_layer_names for layer_name in names
    }
    worker.group_kv_caches_base_addr = {}
    worker.group_block_len = {}
    worker.group_block_stride = {}
    worker.group_layer_cache_entry_offsets = {}
    worker.group_num_layers = {}
    for group_id, names in enumerate(group_layer_names):
        worker._infer_cache_group_metadata(group_id, names)

    assert worker.group_num_layers == {0: 2, 1: 2}
    assert all(len(addresses) == 2 for addresses in worker.group_kv_caches_base_addr.values())
    assert worker.group_layer_cache_entry_offsets == {0: [0, 1, 2], 1: [0, 1, 2]}

    worker.tp_rank = 0
    worker.dcp_size = 1
    worker.put_step = 1
    worker.grouped_block_size = [16, 16]
    worker.hash_block_size = 16
    worker.cache_coordinator = None
    worker.use_block_key_layerwise = False
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    # Supply backend allocation results; keep task and address construction real.
    worker._prepare_load_gvas = Mock()
    worker._alloc_gvas_for_save = Mock()
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
    worker.process_layer_data([request])
    builders = [
        LayerBatchBuilder(worker, sum(worker.group_block_len[group]), worker.group_num_layers[group], group_id=group)
        for group in range(2)
    ]
    tasks_by_layer = worker.layer_save_tasks if is_save else worker.layer_load_tasks
    assert len(tasks_by_layer) == 4
    assert all(len(tasks) == 1 for tasks in tasks_by_layer)
    metas = [builders[task.group_id].build(task, is_save) for tasks in tasks_by_layer for task in tasks]
    for local_layer, meta in enumerate(metas):
        cache = worker.kv_caches[f"model.layers.{layer_offset + local_layer}.attn"]
        assert meta is not None
        assert meta.addr_array.tolist() == [cache[1].data_ptr()]
