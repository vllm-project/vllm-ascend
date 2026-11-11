# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import regex as re
import torch

from .rfork_test_support import _load_module


def test_tensor_collection_deduplicates_exact_impl_alias_but_keeps_distinct_view(tensor_runtime, monkeypatch):
    tensor_layout = tensor_runtime.tensor_layout
    weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model = torch.nn.Module()
    model.register_parameter("weight", weight)
    model.impl = SimpleNamespace(weight=weight, view=weight[:2])
    monkeypatch.setattr(tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_layout.collect_transferable_tensors(model, processed_layout=True)

    assert [(name, tensor.numel()) for name, tensor in collected] == [("weight", 4), ("impl.view", 2)]
    assert collected[0][1] is weight


@pytest.mark.parametrize("processed_layout", [False, True])
def test_bfs_discards_plain_leaves_before_classification(tensor_runtime, monkeypatch, processed_layout):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda _tensor: True)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2))
    model.register_buffer("scale", torch.tensor(1.0))
    model.impl = SimpleNamespace(weight=model.weight, packed_weight=torch.ones(3))
    expected = layout.collect_transferable_tensors(model, processed_layout)

    leaves = [None, True, False, 1, 1.5, 1j, "text", b"bytes"]
    model.public_leaves = leaves
    model.impl.metadata = SimpleNamespace(
        strings=[f"value_{index}" for index in range(10_000)],
        values=leaves,
        tuple_values=tuple(leaves),
        mapping=dict(enumerate(leaves)),
    )
    classify = Mock(wraps=layout._tensor_child_key)
    monkeypatch.setattr(layout, "_tensor_child_key", classify)

    collected = layout.collect_transferable_tensors(model, processed_layout)

    assert [(name, id(tensor)) for name, tensor in collected] == [(name, id(tensor)) for name, tensor in expected]
    assert len(collected) == 3
    assert classify.call_count < 30
    assert all(
        type(call.args[2]) not in {str, bytes, int, float, bool, complex, type(None)}
        for call in classify.call_args_list
    )


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
        model = make_model(max_num_batched_tokens)
        collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, processed_layout)
        assert all(tensor is not model.topk_indices_buffer for _, tensor in collected)
        manifests.append({name: tuple(tensor.shape) for name, tensor in collected})

    assert manifests[0] == manifests[1]
    assert set(manifests[0].values()) == {(2,), (3,)}


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


@pytest.mark.parametrize("processed_layout", [False, True])
@pytest.mark.parametrize("exclude_shared", [False, True])
def test_session_reuses_inventory_and_final_digest(runtime, monkeypatch, processed_layout, exclude_shared):
    prefix = "vllm_ascend.model_loader.rfork"
    layout = sys.modules[f"{prefix}.tensor_layout"]
    transfer = _load_module(monkeypatch, f"{prefix}.transfer_backend", "transfer_backend.py")
    monkeypatch.setattr(runtime.session, "RForkTransferBackend", transfer.RForkTransferBackend)
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda tensor: True)
    monkeypatch.setattr(layout, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(transfer, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(sys.modules[f"{prefix}.manifest"], "read_npu_format", lambda tensor: 0)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model.register_buffer("scale", torch.tensor(1.0))
    inventory = layout.collect_transferable_tensors(model, processed_layout)
    expected_digest = layout.build_structural_digest(inventory)
    excluded = [(model.scale.data_ptr(), model.scale.element_size())] if exclude_shared else None
    success = SimpleNamespace(is_error=lambda: False)
    blocks = [
        {"address": tensor.data_ptr(), "size": tensor.numel() * tensor.element_size(), "state": "active_allocated"}
        for _, tensor in inventory
    ]
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(memory=SimpleNamespace(memory_snapshot=lambda: [{"blocks": blocks}])),
        raising=False,
    )
    collect = Mock(wraps=layout.collect_transferable_tensors)
    monkeypatch.setattr(transfer, "collect_transferable_tensors", collect)
    monkeypatch.setattr(runtime.session, "collect_transferable_tensors", collect)
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    backend = session.transfer_backend
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda registrations: success,
        batch_unregister_memory=lambda addresses: success,
        batch_transfer_sync_read=lambda *args: success,
        finalize=lambda: success,
    )
    backend._memory_registration_cls = lambda *fields: fields
    backend.transfer_session_id = "receiver-session"
    backend._is_initialized = True
    monkeypatch.setattr(session, "_ensure_lease_release_retry_locked", lambda: None)
    monkeypatch.setattr(session, "_run_seed_heartbeat", lambda *args: None)
    monkeypatch.setattr(session.planner, "report_seed_once", lambda *args, **kwargs: True)
    monkeypatch.setattr(session.planner, "remove_seed", lambda: True)
    monkeypatch.setattr(runtime.session, "start_rfork_server", Mock(return_value=Mock(is_alive=True, port=1234)))
    verify = Mock(wraps=session.planner.verify_structural_digest)
    monkeypatch.setattr(session.planner, "verify_structural_digest", verify)

    try:
        assert session.register_destination(model, processed_layout, excluded)
        assert collect.call_count == 1
        assert session.planner.structural_digest == expected_digest
        assert len(backend._registered_transferable_tensors) == (1 if exclude_shared else 2)
        seed_info = runtime.types.SeedTransferInfo(
            "seed-session", dict(backend.weight_manifest), formats=dict(backend.weight_formats)
        )
        monkeypatch.setattr(runtime.session, "fetch_seed_transfer_info", lambda *args: seed_info)
        session.state = runtime.types.RForkLifecycleState.LEASED
        session.seed_lease = runtime.lease
        assert session.transfer_from_seed(model, processed_layout)
        assert collect.call_count == 1
        session.seed_lease = None
        model.eval()
        digest = session.log_transferred_model_layout(model, processed_layout)
        assert digest == expected_digest
        assert collect.call_count == 2
        result = session.start_seed_service(model, processed_layout, excluded, structural_digest=digest)
        # Checkpoint-layout promotion needs a new registration after post-load processing.
        assert collect.call_count == 2 + (not processed_layout)
        assert result is runtime.types.RForkSeedServiceStartResult.STARTED
        if processed_layout:
            verify.assert_called_once_with(expected_digest)
        else:
            verify.assert_not_called()
    finally:
        session.seed_lease = None
        assert session.shutdown()


def test_bfs_preserves_device_empty_meta_and_layout_checks(tensor_runtime, monkeypatch):
    model = torch.nn.Module()
    weight = torch.ones(2)
    cpu = torch.ones(3)
    model.impl = {"weight": weight, "cpu": cpu, "empty": torch.empty(0), "meta": torch.empty(2, device="meta")}
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda tensor: tensor is not cpu)

    collected = layout.collect_transferable_tensors(model, True)

    assert len(collected) == 1
    assert collected[0][1] is weight
    model.impl["gapped"] = torch.ones(4)[::2]
    with pytest.raises(ValueError, match="gapped or overlapping"):
        layout.collect_transferable_tensors(model, True)


@pytest.mark.parametrize("processed_layout", [False, True])
def test_registered_tensors_stay_canonical_over_shallower_aliases(tensor_runtime, monkeypatch, processed_layout):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.inner = torch.nn.Linear(2, 2, bias=False)
    model.layer.inner.register_buffer("scale", torch.ones(2))
    before = layout.collect_transferable_tensors(model, processed_layout)
    # Plain attribute aliases are one or two edges shallower than the registered tensors.
    model.weight_alias = model.layer.inner.weight.data
    model.impl = SimpleNamespace(scale=model.layer.inner.scale.view(2))

    after = layout.collect_transferable_tensors(model, processed_layout)

    assert [name for name, _ in after] == [name for name, _ in before]
    assert [name for name, _ in after] == ["layer.inner.weight", "layer.inner.scale"]
    assert after[0][1] is model.layer.inner.weight
    assert isinstance(after[0][1], torch.nn.Parameter)
    assert after[1][1] is model.layer.inner.scale


@pytest.mark.parametrize("exclude_shared", [False, True])
def test_registration_digest_reads_formats_only_for_excluded_tensors(tensor_runtime, monkeypatch, exclude_shared):
    layout = tensor_runtime.tensor_layout
    transfer = tensor_runtime.transfer_backend
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(4))
    model.register_buffer("shared", torch.ones(2))
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda _tensor: True)
    monkeypatch.setattr(transfer, "read_npu_format", lambda _tensor: 29)
    layout_reads = Mock(return_value=29)
    monkeypatch.setattr(layout, "read_npu_format", layout_reads)
    inventory = layout.collect_transferable_tensors(model, True)
    blocks = [
        {"address": tensor.data_ptr(), "size": tensor.numel() * tensor.element_size(), "state": "active_allocated"}
        for _, tensor in inventory
    ]
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(memory=SimpleNamespace(memory_snapshot=lambda: [{"blocks": blocks}])),
        raising=False,
    )
    success = SimpleNamespace(is_error=lambda: False)
    backend = tensor_runtime.RForkTransferBackend()
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda registrations: success,
        batch_unregister_memory=lambda addresses: success,
    )
    backend._memory_registration_cls = lambda *fields: fields
    excluded = [(model.shared.data_ptr(), model.shared.numel() * model.shared.element_size())]

    assert backend.register_memory_region(model, True, excluded if exclude_shared else None)

    assert layout_reads.call_count == (1 if exclude_shared else 0)
    layout_reads.reset_mock()
    assert backend.registered_structural_digest == layout.build_structural_digest(inventory)
    assert backend.unregister_memory_region()
    assert backend.registered_structural_digest is None
