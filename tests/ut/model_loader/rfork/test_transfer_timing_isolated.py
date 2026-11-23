# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(("tp_rank", "expected_level"), [(0, logging.INFO), (2, logging.DEBUG)])
def test_weight_transfer_completion_is_info_only_for_tp0(tensor_runtime, monkeypatch, caplog, tp_rank, expected_level):
    transfer_backend = tensor_runtime.transfer_backend
    manifest = sys.modules["vllm_ascend.model_loader.rfork.manifest"]
    monkeypatch.setattr(manifest, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(transfer_backend, "read_npu_format", lambda tensor: 0)

    tensor = torch.arange(4, dtype=torch.float32)
    backend = tensor_runtime.RForkTransferBackend(tp_rank=tp_rank)
    backend.transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend._registered_transferable_tensors = [("weight", tensor)]
    seed_info = tensor_runtime.SeedTransferInfo(
        "seed-session",
        {"weight": (1234, 4, 4, [4], "float32")},
        formats={"weight": 0},
    )

    with caplog.at_level(logging.DEBUG, logger=transfer_backend.logger.name):
        assert backend.read_weights_from_seed(object(), seed_info, True)

    completion = [record for record in caplog.records if "RFork weight transfer completed" in record.message]
    assert len(completion) == 1
    assert completion[0].levelno == expected_level
    assert f"tp_rank={tp_rank}" in completion[0].message
    assert "elapsed=" in completion[0].message
    assert "throughput=" in completion[0].message


def test_failed_transfer_does_not_log_successful_completion(tensor_runtime, monkeypatch, caplog):
    transfer_backend = tensor_runtime.transfer_backend
    manifest = sys.modules["vllm_ascend.model_loader.rfork.manifest"]
    monkeypatch.setattr(manifest, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(transfer_backend, "read_npu_format", lambda tensor: 0)

    tensor = torch.arange(4, dtype=torch.float32)
    backend = tensor_runtime.RForkTransferBackend(tp_rank=0)
    backend.transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(
            is_error=lambda: True,
            to_string=lambda: "transfer failed",
        )
    )
    backend._registered_transferable_tensors = [("weight", tensor)]
    seed_info = tensor_runtime.SeedTransferInfo(
        "seed-session",
        {"weight": (1234, 4, 4, [4], "float32")},
        formats={"weight": 0},
    )

    with caplog.at_level(logging.INFO, logger=transfer_backend.logger.name):
        assert not backend.read_weights_from_seed(object(), seed_info, True)

    assert not any("RFork weight transfer completed" in record.message for record in caplog.records)
