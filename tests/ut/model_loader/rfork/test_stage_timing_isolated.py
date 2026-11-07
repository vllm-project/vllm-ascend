# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import logging
from types import SimpleNamespace

import pytest
import regex as re
import torch


def _record(caplog):
    messages = [r.getMessage() for r in caplog.records if "scope=register_memory_region" in r.getMessage()]
    assert len(messages) == 1
    match = re.search(r"total_s=(\d+\.\d{6}) stages_s=(\{.*\})$", messages[0])
    assert match
    total = float(match.group(1))
    stages = json.loads(match.group(2))
    assert sum(stages.values()) == pytest.approx(total, abs=len(stages) * 1e-6)
    return messages[0], stages


@pytest.mark.parametrize("fail", [False, True])
def test_registration_timing_covers_inventory_native_calls_and_rollback(tensor_runtime, monkeypatch, caplog, fail):
    transfer = tensor_runtime.transfer_backend
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda tensor: True)
    monkeypatch.setattr(layout, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(transfer, "read_npu_format", lambda tensor: 0)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2))
    tensor = model.weight
    blocks = [{"address": tensor.data_ptr(), "size": 8, "state": "active_allocated"}]
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(memory=SimpleNamespace(memory_snapshot=lambda: [{"blocks": blocks}])),
        raising=False,
    )
    success = SimpleNamespace(is_error=lambda: False)
    failed = SimpleNamespace(is_error=lambda: True, to_string=lambda: "failed")
    backend = tensor_runtime.RForkTransferBackend(tp_rank=1)
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda _: failed if fail else success,
        batch_unregister_memory=lambda _: success,
    )
    backend._memory_registration_cls = lambda *fields: fields
    backend.transfer_session_id = "registered-session"
    identity = SimpleNamespace(global_rank=5, is_draft_model=True)

    with caplog.at_level(logging.INFO, logger=transfer.logger.name):
        assert backend.register_memory_region(model, purpose="destination", identity=identity) is not fail

    message, stages = _record(caplog)
    assert "relation=nested" in message
    assert "purpose=destination model_kind=draft global_rank=5 tp_rank=1 session=registered-session" in message
    assert f"outcome={'failed' if fail else 'success'}" in message
    assert {
        "lock_wait",
        "engine_init_overhead",
        "device_check",
        "collect",
        "manifest",
        "digest",
        "memory_snapshot",
        "block_plan",
        "native_calls",
        "lock_release",
    } <= stages.keys()
    assert ("registration_rollback" in stages) is fail
    assert ("layout_summary" in stages) is not fail


def test_registration_timing_records_exception(tensor_runtime, monkeypatch, caplog):
    transfer = tensor_runtime.transfer_backend
    backend = tensor_runtime.RForkTransferBackend()
    monkeypatch.setattr(backend, "_initialize_transfer_engine", lambda *_: (_ for _ in ()).throw(RuntimeError("init")))
    with caplog.at_level(logging.INFO, logger=transfer.logger.name), pytest.raises(RuntimeError, match="init"):
        backend.register_memory_region(object())
    message, stages = _record(caplog)
    assert "outcome=exception" in message
    assert "engine_init_overhead" in stages


@pytest.mark.parametrize("purpose", ["destination", "seed_service"])
def test_session_labels_registration_purpose(runtime, purpose):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.transfer_backend.register_memory_region.return_value = False
    model = object()

    if purpose == "destination":
        assert not session.register_receiver_weights(model)
    else:
        session.register_local_weights_for_seed(model)

    session.transfer_backend.register_memory_region.assert_called_once_with(
        model, purpose=purpose, identity=runtime.identity
    )
