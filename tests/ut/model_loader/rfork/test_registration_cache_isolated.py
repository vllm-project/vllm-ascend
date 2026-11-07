# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ctypes
import enum
import logging
import sys
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import regex as re
import torch

from .rfork_test_support import _load_module
from .session_test_utils import join_seed_worker


def test_tensor_collection_deduplicates_exact_impl_alias_but_keeps_distinct_view(tensor_runtime, monkeypatch):
    tensor_layout = tensor_runtime.tensor_layout
    weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model = torch.nn.Module()
    model.register_parameter("weight", weight)
    model.impl = SimpleNamespace(weight=weight, view=weight[:2])
    monkeypatch.setattr(tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_layout.collect_transferable_tensors(model)

    assert [(name, tensor.numel()) for name, tensor in collected] == [("weight", 4), ("impl.view", 2)]
    assert collected[0][1] is weight


def test_bfs_discards_plain_leaves_before_classification(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda _tensor: True)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2))
    model.register_buffer("scale", torch.tensor(1.0))
    model.impl = SimpleNamespace(weight=model.weight, packed_weight=torch.ones(3))
    expected = layout.collect_transferable_tensors(model)

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

    collected = layout.collect_transferable_tensors(model)

    assert [(name, id(tensor)) for name, tensor in collected] == [(name, id(tensor)) for name, tensor in expected]
    assert len(collected) == 3
    assert classify.call_count < 30
    assert all(
        type(call.args[2]) not in {str, bytes, int, float, bool, complex, type(None)}
        for call in classify.call_args_list
    )


@pytest.mark.parametrize("number_type, value", [(int, 1), (float, 1.5), (complex, 1j)])
def test_bfs_preserves_scan_scope_and_numeric_subclasses(tensor_runtime, monkeypatch, number_type, value):
    Number = type("Number", (number_type,), {})

    class Implementation:
        class_weight = torch.ones(4)

        def __call__(self):
            pass

        def method(self):
            pass

    number = Number(value)
    number.weight = torch.ones(2)
    monkeypatch.setattr(Implementation.method, "weight", torch.ones(6), raising=False)
    impl: Any = Implementation()
    impl.weight = torch.tensor(1.0)
    impl.number = number
    impl._private = torch.ones(3)
    function = lambda: None
    monkeypatch.setattr(function, "weight", torch.ones(5), raising=False)
    impl.function = function
    impl.class_object = Implementation
    impl.bound_method = impl.method
    impl.module = torch.nn.Linear(2, 2)
    for name, base, leaf_value in (("text", str, "text"), ("bytes", bytes, b"bytes")):
        leaf = type("Leaf", (base,), {})(leaf_value)
        leaf.weight = torch.ones(7)
        setattr(impl, name, leaf)
    model = torch.nn.Module()
    model.impl = impl
    model.other = SimpleNamespace(weight=torch.ones(8))
    model._private = torch.ones(9)
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_tensor_on_transfer_device", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model)

    assert {id(tensor) for _, tensor in collected} == {
        id(impl.weight),
        id(number.weight),
    }


def test_collector_excludes_scheduler_sized_topk_indices_buffer(tensor_runtime, monkeypatch):
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
        collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model)
        assert all(tensor is not model.topk_indices_buffer for _, tensor in collected)
        manifests.append({name: tuple(tensor.shape) for name, tensor in collected})

    assert manifests[0] == manifests[1]
    assert set(manifests[0].values()) == {(2,), (3,)}


def test_collector_excludes_locally_derived_rope_caches(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda _tensor: True)
    model = torch.nn.Module()
    cache = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    model.register_buffer("cos_sin_cache", cache, persistent=False)
    cos, sin = cache.view(-1, 2, 4).repeat(1, 1, 2).chunk(2, dim=1)
    derived = {"_rope_derived_cos_cache": cos.squeeze(1), "_rope_derived_sin_cache": sin.squeeze(1)}
    for name, tensor in derived.items():
        model.register_buffer(name, tensor, persistent=False)
        assert not layout.is_non_overlapping_dense_tensor(tensor)

    collected = layout.collect_transferable_tensors(model)

    assert len(collected) == 1
    assert collected[0][0] == "cos_sin_cache"
    assert collected[0][1] is cache
    buffers = dict(model.named_buffers())
    assert all(buffers[name] is tensor for name, tensor in derived.items())
    assert not model.state_dict()
    model.register_buffer("other_cache", cos.squeeze(1), persistent=False)
    with pytest.raises(ValueError, match="gapped or overlapping"):
        layout.collect_transferable_tensors(model)


def test_layout_summary_reports_tensor_count_bytes_and_formats(tensor_runtime, caplog):
    tensors = [(f"weight_{index}", torch.arange(4, dtype=torch.float32)) for index in range(6)]
    formats = {name: 29 for name, _ in tensors}

    with caplog.at_level(logging.INFO, logger=tensor_runtime.tensor_layout.logger.name):
        tensor_runtime.tensor_layout.log_tensor_layout_summary(
            tensors,
            stage="receiver_before_read",
            session_id="receiver-session",
            peer_session_id="seed-session",
            known_formats=formats,
        )

    records = [record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.getMessage()]
    assert len(records) == 1
    assert "tensors=6 bytes=96 formats={'29': 6}" in records[0]
    assert "session=receiver-session peer_session=seed-session" in records[0]


def test_layout_summary_errors_do_not_interrupt_loading(tensor_runtime, caplog):
    bad_tensor = SimpleNamespace(numel=lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))
    with caplog.at_level(logging.INFO, logger=tensor_runtime.tensor_layout.logger.name):
        tensor_runtime.tensor_layout.log_tensor_layout_summary(
            [("weight", bad_tensor)], stage="registration", session_id="session", known_formats={"weight": 0}
        )
    assert "unavailable=RuntimeError" in caplog.text


def test_final_layout_summary_is_observational(tensor_runtime, monkeypatch, caplog):
    tensor_layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(tensor_layout, "is_tensor_on_transfer_device", lambda tensor: True)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    original = model.weight.detach().clone()
    backend = tensor_runtime.RForkTransferBackend()
    backend.transfer_session_id = "receiver-session"
    monkeypatch.setattr(tensor_runtime.transfer_backend, "read_npu_format", lambda tensor: 0)
    backend.registered_structural_digest = tensor_layout.build_structural_digest(
        tensor_layout.collect_transferable_tensors(model), known_formats={"weight": 0}
    )

    with caplog.at_level(logging.INFO, logger=tensor_layout.logger.name):
        backend.validate_final_layout(model, peer_session_id="seed-session")

    message = next(record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.message)
    assert "stage=receiver_after_transfer_finalize" in message
    assert "session=receiver-session peer_session=seed-session" in message
    assert "tensors=1" in message
    torch.testing.assert_close(model.weight, original)


def test_final_layout_collection_failure_propagates(tensor_runtime, monkeypatch):
    transfer = tensor_runtime.transfer_backend
    monkeypatch.setattr(transfer, "collect_transferable_tensors", Mock(side_effect=ValueError("inspection failed")))
    with pytest.raises(ValueError, match="inspection failed"):
        tensor_runtime.RForkTransferBackend().validate_final_layout(object())


def test_final_layout_format_failure_propagates(tensor_runtime, monkeypatch):
    transfer = tensor_runtime.transfer_backend
    monkeypatch.setattr(transfer, "collect_transferable_tensors", lambda model: [("weight", torch.ones(1))])
    monkeypatch.setattr(transfer, "read_npu_format", lambda tensor: None)
    with pytest.raises(RuntimeError, match="final NPU storage format"):
        tensor_runtime.RForkTransferBackend().validate_final_layout(object())


def test_final_layout_log_failure_does_not_change_validation(tensor_runtime, monkeypatch):
    transfer = tensor_runtime.transfer_backend
    layout = tensor_runtime.tensor_layout
    tensor = torch.ones(1)
    monkeypatch.setattr(transfer, "collect_transferable_tensors", lambda model: [("weight", tensor)])
    monkeypatch.setattr(transfer, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(transfer, "log_tensor_layout_summary", Mock(side_effect=RuntimeError("log failed")))
    backend = transfer.RForkTransferBackend()
    backend.registered_structural_digest = layout.build_structural_digest([("weight", tensor)], {"weight": 0})
    backend.validate_final_layout(object())


def test_final_layout_detects_changed_tensor(tensor_runtime, monkeypatch):
    transfer = tensor_runtime.transfer_backend
    layout = tensor_runtime.tensor_layout
    own = torch.ones(1)
    shared = torch.ones(2)
    tensors = [("own", own), ("shared", shared)]
    monkeypatch.setattr(transfer, "collect_transferable_tensors", lambda model: tensors)
    monkeypatch.setattr(transfer, "read_npu_format", lambda tensor: 0)
    backend = transfer.RForkTransferBackend()
    backend.registered_structural_digest = layout.build_structural_digest(tensors, {"own": 0, "shared": 0})
    tensors[1] = ("shared", torch.ones(3))

    with pytest.raises(RuntimeError, match="structure differs"):
        backend.validate_final_layout(object())


@pytest.mark.parametrize("deferred", [False, True])
def test_session_reuses_inventory_and_final_digest(runtime, monkeypatch, deferred):
    prefix = "vllm_ascend.model_loader.rfork"
    layout = sys.modules[f"{prefix}.tensor_layout"]
    transfer = _load_module(monkeypatch, f"{prefix}.transfer_backend", "transfer_backend.py")
    monkeypatch.setattr(runtime.session, "RForkTransferBackend", transfer.RForkTransferBackend)
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda tensor: True)
    monkeypatch.setattr(layout, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(transfer, "read_npu_format", lambda tensor: 0)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model.register_buffer("scale", torch.tensor(1.0))
    inventory = layout.collect_transferable_tensors(model)
    expected_digest = layout.build_structural_digest(inventory)
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
    monkeypatch.setattr(session, "_run_seed_heartbeat", lambda *args: None)
    monkeypatch.setattr(session.planner, "report_seed_once", lambda *args, **kwargs: True)
    monkeypatch.setattr(session.planner, "remove_seed", lambda: True)
    monkeypatch.setattr(runtime.session, "create_rfork_server", Mock(return_value=Mock(is_alive=True, port=1234)))

    try:
        assert session.register_receiver_weights(model)
        assert collect.call_count == 1
        assert session.planner.structural_digest == expected_digest
        assert len(backend._registered_tensors) == 2
        weights = dict(backend.weight_manifest)
        seed_info = runtime.types.SeedTransferInfo("seed-session", weights)
        monkeypatch.setattr(runtime.session, "fetch_seed_transfer_info", lambda *args: seed_info)
        session.seed_lease = runtime.lease
        release = threading.Event()

        def wait_for_release():
            assert release.wait(3)
            session.seed_lease = None

        monkeypatch.setattr(session, "_retry_seed_lease_release", wait_for_release)
        assert session.read_weights_from_seed()
        assert session._seed_worker_thread is None
        assert session.seed_lease is runtime.lease
        assert collect.call_count == 1
        if not deferred:
            release.set()
        model.eval()
        session.validate_transferred_model_layout(model)
        assert collect.call_count == 2
        result = session.schedule_seed_service()
        assert result is True
        if deferred:
            release.set()
        join_seed_worker(session, 2)
        assert collect.call_count == 2
        assert session.state is runtime.types.RForkLifecycleState.SERVING
    finally:
        session.seed_lease = None
        assert session.shutdown()
        assert backend._registered_tensors is None


def test_bfs_preserves_device_empty_meta_and_layout_checks(tensor_runtime, monkeypatch):
    model = torch.nn.Module()
    weight = torch.ones(2)
    cpu = torch.ones(3)
    model.impl = {"weight": weight, "cpu": cpu, "empty": torch.empty(0), "meta": torch.empty(2, device="meta")}
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda tensor: tensor is not cpu)

    collected = layout.collect_transferable_tensors(model)

    assert len(collected) == 1
    assert collected[0][1] is weight
    model.impl["gapped"] = torch.ones(4)[::2]
    with pytest.raises(ValueError, match="gapped or overlapping"):
        layout.collect_transferable_tensors(model)


@pytest.mark.parametrize("matching_aliases", [True, False])
def test_bfs_ids_match_seed_reads(tensor_runtime, monkeypatch, matching_aliases):
    def make_model(shared_value, own_value, reverse, matching_aliases=True):
        model = torch.nn.Module()
        shared = torch.full((4,), shared_value, dtype=torch.float32)
        own = torch.full((3,), own_value, dtype=torch.float32)
        entries = [("a", shared), ("b", shared if matching_aliases else shared.clone()), ("own", own)]
        model.impl = dict(reversed(entries) if reverse else entries)
        return model

    layout = tensor_runtime.tensor_layout
    transfer = tensor_runtime.transfer_backend
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    source = make_model(7, 5, False)
    destination = make_model(11, 13, True, matching_aliases)
    source_tensors = layout.collect_transferable_tensors(source)
    destination_tensors = layout.collect_transferable_tensors(destination)
    assert (
        layout.build_structural_digest(source_tensors) == layout.build_structural_digest(destination_tensors)
    ) == matching_aliases
    reads = []

    def read(_session, client_ptrs, seed_ptrs, lengths):
        for client_ptr, seed_ptr, length in zip(client_ptrs, seed_ptrs, lengths, strict=True):
            reads.append((client_ptr, seed_ptr, length))
            ctypes.memmove(client_ptr, seed_ptr, length)
        return SimpleNamespace(is_error=lambda: False)

    backend = tensor_runtime.RForkTransferBackend()
    backend.transfer_engine = SimpleNamespace(batch_transfer_sync_read=read)
    backend._registered_tensors = destination_tensors
    monkeypatch.setattr(
        transfer,
        "collect_transferable_tensors",
        lambda model: pytest.fail("transfer must use the registered inventory"),
    )
    seed_info = tensor_runtime.SeedTransferInfo(
        "seed-session",
        {name: tensor.data_ptr() for name, tensor in source_tensors},
    )

    success = backend.read_weights_from_seed(seed_info)
    if not matching_aliases:
        assert not success
        assert reads == []
        return
    assert success
    assert len(reads) == 2
    assert destination.impl["a"] is destination.impl["b"]
    torch.testing.assert_close(destination.impl["a"], torch.full((4,), 7.0))
    torch.testing.assert_close(destination.impl["own"], torch.full((3,), 5.0))


class _KeyKind(enum.Enum):
    SCALE = 1


def test_bfs_ids_distinguish_typed_container_keys(tensor_runtime):
    layout = tensor_runtime.tensor_layout
    # 1, True and 1.0 collide as dict keys but may still appear under different parents.
    keys = [1, "1", True, "true", 1.0, (1,), "(1,)", 1.5, _KeyKind.SCALE, "SCALE", torch.float16, None, "None"]

    labels = [layout._format_tensor_key(key) for key in keys]

    assert None not in labels
    assert len(set(labels)) == len(keys)


def test_bfs_rejects_container_keys_without_a_stable_label(tensor_runtime, monkeypatch):
    class OpaqueKey:
        pass

    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    model = torch.nn.Module()
    model.impl = {OpaqueKey(): torch.ones(2)}

    with pytest.raises(ValueError, match="stable tensor ID.*OpaqueKey"):
        layout.collect_transferable_tensors(model)


def test_registered_tensors_stay_canonical_over_shallower_aliases(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.inner = torch.nn.Linear(2, 2, bias=False)
    model.layer.inner.register_buffer("scale", torch.ones(2))
    before = layout.collect_transferable_tensors(model)
    # Plain attribute aliases are one or two edges shallower than the registered tensors.
    model.weight_alias = model.layer.inner.weight.data
    model.impl = SimpleNamespace(scale=model.layer.inner.scale.view(2))

    after = layout.collect_transferable_tensors(model)

    assert [name for name, _ in after] == [name for name, _ in before]
    assert [name for name, _ in after] == ["layer.inner.weight", "layer.inner.scale"]
    assert after[0][1] is model.layer.inner.weight
    assert isinstance(after[0][1], torch.nn.Parameter)
    assert after[1][1] is model.layer.inner.scale


def test_layout_error_names_the_full_tensor_path(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.impl = SimpleNamespace(packed={"w": torch.ones(4)[::2]})

    with pytest.raises(ValueError, match=re.escape("'block.impl.packed[\"w\"]'")):
        layout.collect_transferable_tensors(model)


@pytest.mark.parametrize("missing_shared_format", [False, True])
def test_registration_digest_reads_formats_for_all_tensors(tensor_runtime, monkeypatch, missing_shared_format):
    layout = tensor_runtime.tensor_layout
    transfer = tensor_runtime.transfer_backend
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(4))
    model.register_buffer("shared", torch.ones(2))
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda _tensor: True)
    transfer_reads = Mock(side_effect=lambda tensor: None if missing_shared_format and tensor is model.shared else 29)
    monkeypatch.setattr(transfer, "read_npu_format", transfer_reads)
    layout_reads = Mock(return_value=29)
    monkeypatch.setattr(layout, "read_npu_format", layout_reads)
    inventory = layout.collect_transferable_tensors(model)
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

    registered = backend.register_memory_region(model)
    if missing_shared_format:
        assert not registered
        return
    assert registered

    assert transfer_reads.call_count == 2
    assert layout_reads.call_count == 0
    layout_reads.reset_mock()
    assert backend.registered_structural_digest == layout.build_structural_digest(inventory)
    assert backend.unregister_memory_region()
    assert backend.registered_structural_digest is None
