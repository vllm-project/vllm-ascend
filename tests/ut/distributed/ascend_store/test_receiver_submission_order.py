# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import kv_transfer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import LayerBatchReqMeta


@pytest.mark.parametrize("reuse_ready,gate_ready", [(True, True), (True, False), (False, True)])
def test_receiver_prepares_host_arrays_early_but_keeps_all_device_fences(monkeypatch, reuse_ready, gate_ready):
    trace = []
    receiver = kv_transfer.KVCacheStoreLayerRecvingThread.__new__(kv_transfer.KVCacheStoreLayerRecvingThread)
    receiver._wait_for_reuse_saves = lambda data: trace.append("reuse_fence") or reuse_ready
    receiver._wait_for_event = lambda gate: trace.append("compute_gate") or gate_ready
    receiver._stagger_h2d_submit = lambda layer: None
    receiver.external_slot_release_waiter = lambda layer: trace.append("external_readers")
    receiver.load_disabled = threading.Event()
    receiver.max_transfer_blocks = receiver.max_transfer_bytes = 0
    receiver.final_layer_id = 1
    receiver.layer_load_finished_events = [threading.Event()]
    metas = [
        LayerBatchReqMeta(
            req_ids=["r"],
            layer_id=0,
            is_last_chunks=[False],
            gvas_array=np.asarray([100 + i]),
            addr_array=np.asarray([200 + i]),
            size_array=np.asarray([32]),
        )
        for i in range(2)
    ]
    receiver.group_builders = [SimpleNamespace(build_addrs=Mock(return_value=meta)) for meta in metas]
    tasks = [SimpleNamespace(group_id=i, layer_idx_in_group=0, shared_block_data=object()) for i in range(2)]
    data = SimpleNamespace(transfer_tasks=tasks, layer_id=0, attention_start_gate=object())
    concatenate = np.concatenate

    def assemble(arrays):
        trace.append("host_assembly")
        return concatenate(arrays)

    def copy(gvas, addresses, sizes, direction, *limits):
        trace.append("device_copy")
        assert gvas.tolist() == [100, 101]
        assert addresses.tolist() == [200, 201]
        assert sizes.tolist() == [32, 32]
        assert direction == 1
        return 0

    monkeypatch.setattr(kv_transfer.np, "concatenate", assemble)
    receiver._batch_copy_with_limits = copy
    assert receiver._copy_gva_layers(data) == (reuse_ready and gate_ready)
    expected = ["reuse_fence"]
    if reuse_ready:
        expected += ["host_assembly"] * 3 + ["compute_gate"]
        if gate_ready:
            expected += ["external_readers", "device_copy"]
    assert trace == expected
