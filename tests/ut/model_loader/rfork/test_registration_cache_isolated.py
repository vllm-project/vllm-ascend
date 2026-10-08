# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ctypes
import enum
import logging
import sys
from types import SimpleNamespace
from typing import Any
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
@pytest.mark.parametrize("number_type, value", [(int, 1), (float, 1.5), (complex, 1j)])
def test_bfs_preserves_scan_scope_and_numeric_subclasses(
    tensor_runtime, monkeypatch, processed_layout, number_type, value
):
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

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, processed_layout)

    assert {id(tensor) for _, tensor in collected} == {
        id(impl.weight),
        id(number.weight),
    }


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
        digest = backend.log_model_layout_summary(
            model,
            False,
            stage="receiver_after_post_load",
            peer_session_id="seed-session",
        )

    message = next(record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.message)
    assert "stage=receiver_after_post_load" in message
    assert "session=receiver-session peer_session=seed-session" in message
    assert "tensors=1" in message
    assert digest == tensor_layout.build_structural_digest(tensor_layout.collect_transferable_tensors(model, False))
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


@pytest.mark.parametrize("processed_layout", [False, True])
@pytest.mark.parametrize("exclude_shared", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_session_reuses_inventory_and_final_digest(runtime, monkeypatch, processed_layout, exclude_shared, deferred):
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

    with pytest.raises(RuntimeError, match="unavailable"):
        backend.snapshot_registered_tensor_inventory()
    try:
        assert session.register_destination(model, processed_layout, excluded)
        assert collect.call_count == 1
        assert session.planner.structural_digest == expected_digest
        assert len(backend._registered_transferable_tensors) == (1 if exclude_shared else 2)
        assert backend.snapshot_registered_tensor_inventory() == inventory
        backend.snapshot_registered_tensor_inventory().clear()
        assert len(backend.snapshot_registered_tensor_inventory()) == 2
        seed_info = runtime.types.SeedTransferInfo(
            "seed-session", dict(backend.weight_manifest), formats=dict(backend.weight_formats)
        )
        monkeypatch.setattr(runtime.session, "fetch_seed_transfer_info", lambda *args: seed_info)
        session.state = runtime.types.RForkLifecycleState.LEASED
        session.seed_lease = runtime.lease
        assert session.transfer_from_seed(model, processed_layout)
        assert collect.call_count == 1
        if not deferred:
            session.seed_lease = None
        model.eval()
        digest = session.log_transferred_model_layout(model, processed_layout)
        assert digest == expected_digest
        assert collect.call_count == 2
        result = session.start_seed_service(model, processed_layout, excluded, structural_digest=digest)
        # Checkpoint-layout promotion needs a new registration after post-load processing.
        expected_scans = 2 + (not processed_layout)
        assert collect.call_count == expected_scans
        if deferred:
            assert result is runtime.types.RForkSeedServiceStartResult.DEFERRED
            verify.assert_not_called()
            model.register_buffer("late_buffer", torch.ones(1))
            live_digest = layout.build_structural_digest(layout.collect_transferable_tensors(model, processed_layout))
            assert live_digest != expected_digest
            session.seed_lease = None
            session._promote_deferred_seed()
            assert collect.call_count == expected_scans + 1
            verify.assert_called_once_with(live_digest)
        else:
            assert result is runtime.types.RForkSeedServiceStartResult.STARTED
            verify.assert_called_once_with(expected_digest)
    finally:
        session.seed_lease = None
        assert session.shutdown()
    with pytest.raises(RuntimeError, match="unavailable"):
        backend.snapshot_registered_tensor_inventory()


@pytest.mark.parametrize("processed_layout", [False, True])
def test_bfs_ids_preserve_graph_structure_and_ignore_order(tensor_runtime, monkeypatch, processed_layout):
    def make_model(reverse):
        model = torch.nn.Module()
        parameter = torch.nn.Parameter(torch.ones(2))
        buffer = torch.ones(3)
        child = torch.nn.Module()
        child.weight = torch.nn.Parameter(torch.ones(4))
        shared = {"weight": torch.ones(5)}
        alias = torch.ones(14)
        pairs = lambda entries: reversed(entries) if reverse else entries
        for name in pairs(["a", "z"]):
            model.register_parameter(name, parameter)
        for name in pairs(["b", "y"]):
            model.register_buffer(name, buffer)
        for name in pairs(["left", "right"]):
            model.add_module(name, child)
        child.add_module("back", model)
        entries = [
            ("a", shared),
            ("b", shared),
            ("b.weight", torch.ones(6)),
            (1, torch.ones(7)),
            ("1", torch.ones(8)),
            ((1, "a"), torch.ones(9)),
            ("topk_indices_buffer", {"weight": torch.ones(10)}),
            ("state", SimpleNamespace(weight=torch.ones(12))),
            ("alias", dict(pairs([("topk_indices_buffer", alias), ("valid", alias)]))),
        ]
        model.impl = dict(pairs(entries))
        model.impl["cycle"] = model.impl
        model.metadata = model.impl
        object.__setattr__(model, "unregistered", torch.nn.Linear(15, 15, bias=False))
        return model

    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    manifests = []
    for reverse in (False, True):
        model = make_model(reverse)
        manifest = {
            name: tuple(tensor.shape) for name, tensor in layout.collect_transferable_tensors(model, processed_layout)
        }
        del model.impl["alias"]["topk_indices_buffer"]
        assert manifest == {
            name: tuple(tensor.shape) for name, tensor in layout.collect_transferable_tensors(model, processed_layout)
        }
        manifests.append(manifest)

    assert manifests[0] == manifests[1]
    assert len(manifests[0]) == 11
    assert set(manifests[0].values()) == {(size,) for size in (*range(2, 11), 12, 14)}


@pytest.mark.parametrize("short_path", [False, True])
def test_bfs_chooses_shortest_path_then_smallest_id(tensor_runtime, monkeypatch, short_path):
    layout = tensor_runtime.tensor_layout
    shared = {"weight": torch.ones(2)}
    model = torch.nn.Module()
    # Enqueue the worse candidate first to catch premature expansion/ID finalization.
    model.impl = {name: {"shared": shared} for name in ("b", "a")}
    if short_path:
        model.impl["short"] = shared["weight"]
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)

    collected = layout.collect_transferable_tensors(model, True)

    expected = 'impl["short"]' if short_path else 'impl["a"]["shared"]["weight"]'
    assert [name for name, _ in collected] == [expected]


@pytest.mark.parametrize("processed_layout", [False, True])
@pytest.mark.parametrize("contains_tensor", [False, True])
def test_bfs_expands_shared_diamond_once(tensor_runtime, monkeypatch, processed_layout, contains_tensor):
    class CountingDict(dict):
        scans = 0

        def items(self):
            self.scans += 1
            return super().items()

    depth = 30  # Enumerating every alias would produce over a billion tensor paths.
    weight = torch.ones(2)
    value = CountingDict(value=weight if contains_tensor else 1)
    nodes = [value]
    for _ in range(depth):
        value = CountingDict(left=value, right=value)
        nodes.append(value)
    model = torch.nn.Module()
    model.impl = {"left": value, "right": value, "weight": weight}
    layout = tensor_runtime.tensor_layout
    eligible = Mock(return_value=True)
    child_path = Mock(wraps=layout._tensor_child_path)
    monkeypatch.setattr(layout, "is_transferable_tensor", eligible)
    monkeypatch.setattr(layout, "_tensor_child_path", child_path)

    collected = layout.collect_transferable_tensors(model, processed_layout)

    assert len(collected) == 1
    assert all(node.scans == 1 for node in nodes)
    assert eligible.call_count == 1
    # One path per BFS edge, one for the collected tensor's layout check, and one for the
    # deepest edge back to the already collected weight, which still needs the duplicate-label check.
    assert child_path.call_count == 2 * depth + 5 + contains_tensor


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


@pytest.mark.parametrize("exclude_shared, matching_aliases", [(False, True), (True, True), (False, False)])
def test_bfs_ids_match_seed_reads_and_shared_exclusions(tensor_runtime, monkeypatch, exclude_shared, matching_aliases):
    def make_model(shared_value, own_value, reverse, matching_aliases=True):
        model = torch.nn.Module()
        shared = torch.full((4,), shared_value, dtype=torch.float32)
        own = torch.full((3,), own_value, dtype=torch.float32)
        entries = [("a", shared), ("b", shared if matching_aliases else shared.clone()), ("own", own)]
        model.impl = dict(reversed(entries) if reverse else entries)
        return model

    layout = tensor_runtime.tensor_layout
    transfer = tensor_runtime.transfer_backend
    manifest = sys.modules["vllm_ascend.model_loader.rfork.manifest"]
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(manifest, "read_npu_format", lambda _tensor: 0)
    source = make_model(7, 5, False)
    destination = make_model(11, 13, True, matching_aliases)
    source_tensors = layout.collect_transferable_tensors(source, True)
    destination_tensors = layout.collect_transferable_tensors(destination, True)
    assert (
        layout.build_structural_digest(source_tensors) == layout.build_structural_digest(destination_tensors)
    ) == matching_aliases
    shared = destination.impl["a"]
    excluded = [(shared.data_ptr(), shared.numel() * shared.element_size())] if exclude_shared else []
    reads = []

    def read(_session, client_ptrs, seed_ptrs, lengths):
        for client_ptr, seed_ptr, length in zip(client_ptrs, seed_ptrs, lengths, strict=True):
            reads.append((client_ptr, seed_ptr, length))
            ctypes.memmove(client_ptr, seed_ptr, length)
        return SimpleNamespace(is_error=lambda: False)

    backend = tensor_runtime.RForkTransferBackend()
    backend.transfer_engine = SimpleNamespace(batch_transfer_sync_read=read)
    backend._registered_transferable_tensors, _ = transfer._split_tensors_by_excluded_blocks(
        destination_tensors, excluded
    )
    backend.excluded_weight_blocks = excluded
    seed_info = tensor_runtime.SeedTransferInfo(
        "seed-session",
        {
            name: (tensor.data_ptr(), tensor.numel(), tensor.element_size(), tuple(tensor.shape), str(tensor.dtype))
            for name, tensor in source_tensors
        },
        formats={name: 0 for name, _ in source_tensors},
    )

    success = backend.read_weights_from_seed(destination, seed_info, True)
    if not matching_aliases:
        assert not success
        assert reads == []
        return
    assert success
    assert len(reads) == (1 if exclude_shared else 2)
    assert destination.impl["a"] is destination.impl["b"]
    torch.testing.assert_close(destination.impl["a"], torch.full((4,), 11.0 if exclude_shared else 7.0))
    torch.testing.assert_close(destination.impl["own"], torch.full((3,), 5.0))


class _KeyKind(enum.Enum):
    SCALE = 1


@pytest.mark.parametrize(
    "key",
    [_KeyKind.SCALE, torch.float16, float("nan"), float("inf"), (1, "a"), (_KeyKind.SCALE,), True, None],
)
def test_bfs_ids_are_stable_for_non_json_container_keys(tensor_runtime, monkeypatch, key):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)

    def make_model():
        model = torch.nn.Module()
        model.impl = SimpleNamespace(scales={key: torch.ones(3)})
        return model

    first = layout.collect_transferable_tensors(make_model(), True)
    second = layout.collect_transferable_tensors(make_model(), True)

    assert len(first) == 1
    assert [name for name, _ in first] == [name for name, _ in second]


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
        layout.collect_transferable_tensors(model, True)


@pytest.mark.parametrize("reverse", [False, True])
def test_bfs_collects_the_alias_object_named_by_the_chosen_id(tensor_runtime, monkeypatch, reverse):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    storage = torch.ones(4)
    # Distinct tensor objects over the same range share one signature key.
    aliases = {"a": storage.view(4), "b": storage.view(4)}
    model = torch.nn.Module()
    model.impl = dict(reversed(aliases.items()) if reverse else aliases.items())

    collected = layout.collect_transferable_tensors(model, True)

    assert [name for name, _ in collected] == ['impl["a"]']
    assert collected[0][1] is aliases["a"]


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


def test_bfs_ids_are_readable_and_quote_ambiguous_segments(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].impl = SimpleNamespace(
        scales={1: torch.ones(1), "1": torch.ones(2), "a.b": torch.ones(3)},
        stack=[torch.ones(4)],
    )
    setattr(model.layers[0].impl, "a.b", torch.ones(5))
    model.layers[0].impl.a = SimpleNamespace(b=torch.ones(6))

    names = {name: tensor.numel() for name, tensor in layout.collect_transferable_tensors(model, True)}

    assert names == {
        "layers.0.impl.scales[1]": 1,
        'layers.0.impl.scales["1"]': 2,
        'layers.0.impl.scales["a.b"]': 3,
        "layers.0.impl.stack[0]": 4,
        'layers.0.impl["a.b"]': 5,
        "layers.0.impl.a.b": 6,
    }


def test_layout_error_names_the_full_tensor_path(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.impl = SimpleNamespace(packed={"w": torch.ones(4)[::2]})

    with pytest.raises(ValueError, match=re.escape("'block.impl.packed[\"w\"]'")):
        layout.collect_transferable_tensors(model, True)


def test_structural_digest_reuses_known_registration_formats(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    tensors = [("weight", torch.ones(2)), ("shared", torch.ones(3))]
    monkeypatch.setattr(layout, "read_npu_format", lambda _tensor: 29)
    expected = layout.build_structural_digest(tensors)
    reads = Mock(return_value=29)
    monkeypatch.setattr(layout, "read_npu_format", reads)

    assert layout.build_structural_digest(tensors, known_formats={"weight": 29}) == expected
    # Only the tensor without a registered format (e.g. a target-shared one) is read again.
    assert reads.call_count == 1


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


class _Bits(enum.IntFlag):
    LOW = 1
    HIGH = 2


@pytest.mark.parametrize("pair", [(_Bits(4), _Bits(8)), (_Bits.LOW | _Bits.HIGH, _Bits(3) | _Bits(4))])
def test_bfs_ids_encode_unnamed_flag_members_by_value(tensor_runtime, monkeypatch, pair):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    source_key, destination_key = pair

    def make_model(key):
        model = torch.nn.Module()
        model.impl = SimpleNamespace(scales={key: torch.ones(2)})
        return model

    source = layout.collect_transferable_tensors(make_model(source_key), True)
    destination = layout.collect_transferable_tensors(make_model(destination_key), True)

    assert [name for name, _ in source] != [name for name, _ in destination]
    assert layout.build_structural_digest(source) != layout.build_structural_digest(destination)
    assert layout._format_tensor_key(_Bits(4)) != layout._format_tensor_key(4)


def test_unnamed_flag_member_keys_cannot_feed_a_mismatched_seed_read(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    manifest = sys.modules["vllm_ascend.model_loader.rfork.manifest"]
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(manifest, "read_npu_format", lambda _tensor: 0)

    def make_model(key, value):
        model = torch.nn.Module()
        model.impl = {key: torch.full((2,), value)}
        return model

    source = make_model(_Bits(4), 7.0)
    destination = make_model(_Bits(8), 11.0)
    source_tensors = layout.collect_transferable_tensors(source, True)
    backend = tensor_runtime.RForkTransferBackend()
    backend.transfer_engine = SimpleNamespace(batch_transfer_sync_read=Mock(side_effect=AssertionError("no read")))
    backend._registered_transferable_tensors = layout.collect_transferable_tensors(destination, True)
    seed_info = tensor_runtime.SeedTransferInfo(
        "seed-session",
        {
            name: (tensor.data_ptr(), tensor.numel(), tensor.element_size(), tuple(tensor.shape), str(tensor.dtype))
            for name, tensor in source_tensors
        },
        formats={name: 0 for name, _ in source_tensors},
    )

    assert not backend.read_weights_from_seed(destination, seed_info, True)
    torch.testing.assert_close(destination.impl[_Bits(8)], torch.full((2,), 11.0))


@pytest.mark.parametrize(
    "make_keys",
    [
        lambda: (float("nan"), float("nan")),
        lambda: ((float("nan"), 1), (float("nan"), 1)),
    ],
)
@pytest.mark.parametrize("swapped", [False, True])
def test_bfs_rejects_distinct_container_keys_with_one_label(tensor_runtime, monkeypatch, make_keys, swapped):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    first, second = make_keys()
    # Different leaf names keep final IDs unique, so only the dict-level check can see the collision.
    subtrees = [{"left": torch.ones(2)}, {"right": torch.ones(3)}]
    if swapped:
        subtrees.reverse()
    model = torch.nn.Module()
    model.impl = {first: subtrees[0], second: subtrees[1]}
    assert len(model.impl) == 2

    with pytest.raises(ValueError, match="one tensor ID label"):
        layout.collect_transferable_tensors(model, True)


def test_bfs_rejects_duplicate_labels_even_when_one_entry_is_already_visited(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    shared = {"weight": torch.ones(2)}
    model = torch.nn.Module()
    model.impl = {"shared": shared, "nested": {float("nan"): shared, float("nan"): {"other": torch.ones(3)}}}

    with pytest.raises(ValueError, match="one tensor ID label"):
        layout.collect_transferable_tensors(model, True)


class _NanRole(enum.Enum):
    A = float("nan")
    B = float("nan")
    C = (float("nan"), 1)
    D = (float("nan"), 1)


class _AliasRole(enum.Enum):
    PRIMARY = 1
    ALIAS = 1


@pytest.mark.parametrize("pair", [(_NanRole.A, _NanRole.B), (_NanRole.C, _NanRole.D)])
def test_plain_enum_keys_are_labeled_by_member_name(tensor_runtime, pair):
    layout = tensor_runtime.tensor_layout
    first, second = pair

    assert first is not second
    assert layout._format_tensor_key(first) != layout._format_tensor_key(second)
    # Aliases resolve to the canonical member, so they share its label.
    assert _AliasRole.ALIAS is _AliasRole.PRIMARY
    assert layout._format_tensor_key(_AliasRole.ALIAS) == layout._format_tensor_key(_AliasRole.PRIMARY)


@pytest.mark.parametrize("pair", [(_NanRole.A, _NanRole.B), (_NanRole.C, _NanRole.D)])
def test_nan_valued_enum_keys_cannot_feed_a_mismatched_seed_read(tensor_runtime, monkeypatch, pair):
    layout = tensor_runtime.tensor_layout
    manifest = sys.modules["vllm_ascend.model_loader.rfork.manifest"]
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(manifest, "read_npu_format", lambda _tensor: 0)

    def make_model(key, value):
        model = torch.nn.Module()
        model.impl = {key: torch.full((2,), value)}
        return model

    source_key, destination_key = pair
    source = make_model(source_key, 7.0)
    destination = make_model(destination_key, 11.0)
    source_tensors = layout.collect_transferable_tensors(source, True)
    destination_tensors = layout.collect_transferable_tensors(destination, True)
    assert layout.build_structural_digest(source_tensors) != layout.build_structural_digest(destination_tensors)
    backend = tensor_runtime.RForkTransferBackend()
    backend.transfer_engine = SimpleNamespace(batch_transfer_sync_read=Mock(side_effect=AssertionError("no read")))
    backend._registered_transferable_tensors = destination_tensors
    seed_info = tensor_runtime.SeedTransferInfo(
        "seed-session",
        {
            name: (tensor.data_ptr(), tensor.numel(), tensor.element_size(), tuple(tensor.shape), str(tensor.dtype))
            for name, tensor in source_tensors
        },
        formats={name: 0 for name, _ in source_tensors},
    )

    assert not backend.read_weights_from_seed(destination, seed_info, True)
    backend.transfer_engine.batch_transfer_sync_read.assert_not_called()
    torch.testing.assert_close(destination.impl[destination_key], torch.full((2,), 11.0))
