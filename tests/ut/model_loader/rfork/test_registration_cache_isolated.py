# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import regex as re
import torch


def test_tensor_collection_deduplicates_exact_impl_alias_but_keeps_distinct_view(tensor_runtime, monkeypatch):
    tensor_layout = tensor_runtime.tensor_layout
    weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model = torch.nn.Module()
    model.register_parameter("weight", weight)
    model.impl = SimpleNamespace(weight=weight, view=weight[:2])
    monkeypatch.setattr(tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_layout.collect_transferable_tensors(model, processed_layout=True)

    assert [(name, tensor.numel()) for name, tensor in collected] == [("weight", 4), ("impl.view", 2)]


@pytest.mark.parametrize("processed_layout", [False, True])
def test_collector_excludes_scheduler_sized_topk_indices_buffer(tensor_runtime, monkeypatch, processed_layout):
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    def make_model(max_num_batched_tokens):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(2))
        model.topk_indices_buffer = torch.empty(max_num_batched_tokens, 2048, dtype=torch.int32)
        model.indexer_op = torch.nn.Module()
        model.indexer_op.impl = SimpleNamespace(
            packed_weight=torch.ones(3),
            topk_indices_buffer=model.topk_indices_buffer,
        )
        return model

    manifests = []
    for max_num_batched_tokens in (2048, 4096):
        collected = tensor_runtime.tensor_layout.collect_transferable_tensors(
            make_model(max_num_batched_tokens), processed_layout
        )
        assert all(name.rsplit(".", 1)[-1] != "topk_indices_buffer" for name, _ in collected)
        manifests.append([(name, tuple(tensor.shape)) for name, tensor in collected])

    assert manifests[0] == manifests[1]
    assert ("weight", (2,)) in manifests[0]
    assert ("indexer_op.impl.packed_weight", (3,)) in manifests[0]


def test_layout_summary_is_one_bounded_info_record_with_fixed_digests(tensor_runtime, caplog):
    tensors = [(f"weight_{index}", torch.arange(4, dtype=torch.float32)) for index in range(6)]
    formats = {name: 29 for name, _ in tensors}

    with caplog.at_level(logging.INFO, logger=tensor_runtime.tensor_layout.logger.name):
        tensor_runtime.tensor_layout.log_tensor_layout_summary(
            tensors,
            stage="receiver_before_read",
            session_id="receiver-session",
            peer_session_id="seed-session",
            processed_layout=True,
            known_formats=formats,
        )

    records = [record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.getMessage()]
    assert len(records) == 1
    message = records[0]
    assert "tensors=6" in message
    assert "session=receiver-session peer_session=seed-session" in message
    assert len(re.findall(r"(?:semantic|physical)_digest=[0-9a-f]{64}", message)) == 2
    assert "weight_0" in message and "weight_2" in message
    assert "weight_3" not in message and "weight_5" not in message


def test_layout_summary_includes_npu_format_and_physical_size(tensor_runtime, monkeypatch, caplog):
    class _NPUTensorProxy:
        device = SimpleNamespace(type="npu")

        def __init__(self, tensor):
            self._tensor = tensor

        def __getattr__(self, name):
            return getattr(self._tensor, name)

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            get_npu_format=lambda tensor: 29,
            get_storage_size=lambda tensor: tensor.numel() + 8,
        ),
    )
    tensor = _NPUTensorProxy(torch.arange(4, dtype=torch.float32))

    with caplog.at_level(logging.INFO, logger=tensor_runtime.tensor_layout.logger.name):
        tensor_runtime.tensor_layout.log_tensor_layout_summary(
            [("weight", tensor)],
            stage="registered",
            session_id="seed-session",
            processed_layout=True,
        )

    message = next(
        record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.getMessage()
    )
    assert "physical_nonlogical_tensors=1" in message
    assert "formats={'29': 1}" in message
    assert "'npu_format': 29" in message
    assert "'npu_storage_numel': 12" in message


def test_post_load_layout_summary_is_observational(tensor_runtime, monkeypatch, caplog):
    tensor_layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(tensor_layout, "is_tensor_on_transfer_device", lambda tensor: True)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    original = model.weight.detach().clone()
    backend = tensor_runtime.RForkTransferBackend()
    backend.transfer_session_id = "receiver-session"

    with caplog.at_level(logging.INFO, logger=tensor_layout.logger.name):
        backend.log_model_layout_summary(
            model,
            False,
            stage="receiver_after_post_load",
            peer_session_id="seed-session",
        )

    message = next(record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.message)
    assert "stage=receiver_after_post_load" in message
    assert "session=receiver-session peer_session=seed-session" in message
    assert "tensors=1" in message
    torch.testing.assert_close(model.weight, original)


def test_post_load_layout_diagnostic_failure_does_not_escape(tensor_runtime, monkeypatch, caplog):
    transfer_backend = tensor_runtime.transfer_backend
    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        Mock(side_effect=RuntimeError("inspection failed")),
    )
    backend = tensor_runtime.RForkTransferBackend()

    with caplog.at_level(logging.INFO, logger=transfer_backend.logger.name):
        backend.log_model_layout_summary(object(), False, stage="receiver_after_post_load")

    assert "unavailable=RuntimeError:inspection failed" in caplog.text
