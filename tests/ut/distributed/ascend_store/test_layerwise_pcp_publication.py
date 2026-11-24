# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pool_worker


def _publication_worker(monkeypatch, tp, pcp, offload=True, transfer=True):
    events = []
    monkeypatch.setattr(pool_worker, "get_tp_group", lambda: SimpleNamespace(barrier=lambda: events.append("tp")))
    monkeypatch.setattr(pool_worker, "get_pcp_group", lambda: SimpleNamespace(barrier=lambda: events.append("pcp")))
    monkeypatch.setattr(pool_worker, "reset_attention_compute_start_gate", lambda: None)
    worker = pool_worker.LayerwiseKVPoolWorker.__new__(pool_worker.LayerwiseKVPoolWorker)
    worker.use_layerwise = True
    worker.use_layerwise_transfer = transfer
    worker.layerwise_offload = offload
    worker.tp_size, worker.pcp_size = tp, pcp
    worker.use_kvpp = False
    worker.kv_recv_thread = None
    worker._save_disabled = set()
    worker._drain_deferred_last_save = lambda: events.append("save_drained")
    worker.process_layer_data = lambda requests: events.append("load_prepared")
    return worker, events


@pytest.mark.parametrize(
    ("tp", "pcp", "offload", "transfer"),
    [
        (1, 8, True, True),
        (2, 1, True, True),
        (2, 8, True, True),
        (1, 1, True, True),
        (1, 8, False, True),
        (1, 8, True, False),
    ],
)
def test_previous_step_publication_fenced_before_pcp_load(monkeypatch, tp, pcp, offload, transfer):
    worker, events = _publication_worker(monkeypatch, tp, pcp, offload, transfer)
    worker.prepare_layerwise_step(SimpleNamespace(requests=[]))
    expected = ["save_drained"]
    if transfer and offload:
        if tp > 1:
            expected.append("tp")
        if pcp > 1:
            expected.append("pcp")
    assert events == expected + ["load_prepared"]


def test_failed_publication_fence_does_not_prepare_loads(monkeypatch):
    worker, events = _publication_worker(monkeypatch, 1, 8)

    def fail():
        raise RuntimeError("publication failed")

    monkeypatch.setattr(pool_worker, "get_pcp_group", lambda: SimpleNamespace(barrier=fail))
    with pytest.raises(RuntimeError, match="publication failed"):
        worker.prepare_layerwise_step(SimpleNamespace(requests=[]))
    assert events == ["save_drained"]
