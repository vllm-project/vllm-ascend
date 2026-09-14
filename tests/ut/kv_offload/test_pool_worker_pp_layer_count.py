# /*Copyright (c) 2025 Huawei Technologies Co., Ltd.
# *
# * Licensed under the OpenSSL license (the "License").  You may not use
# * this file except in compliance with the License.  You can obtain a copy
# * in the file LICENSE in the source distribution or at
# * https://www.openssl.org/source/license.html
# */

"""Regression tests for PP stage-local layer numbering."""

from types import SimpleNamespace

import torch

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


def test_nonzero_pp_stage_maps_multiple_cache_groups_to_local_layers():
    group_layer_names = [
        ["model.layers.4.attn", "model.layers.6.attn"],
        ["model.layers.5.attn", "model.layers.7.attn"],
    ]
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=names) for names in group_layer_names]
    )
    worker.hf_config = SimpleNamespace(num_hidden_layers=8)
    worker.num_layers = 4
    worker.num_kv_cache_groups = 2
    worker.use_layerwise_transfer = False
    worker.use_layerwise = False
    worker._extra_config = {}

    worker._init_layerwise_config()

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
