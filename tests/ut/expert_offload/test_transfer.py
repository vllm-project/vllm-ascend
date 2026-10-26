# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.transfer import TorchExpertTransfer
from vllm_ascend.expert_offload.types import PreparedExpertTensor


def _tensor(count, pointer, *, physical_elements=None, storage_bytes=None):
    physical_elements = physical_elements if physical_elements is not None else 12 * count
    storage_bytes = storage_bytes if storage_bytes is not None else physical_elements * 4
    storage = SimpleNamespace(nbytes=lambda: storage_bytes, data_ptr=lambda: pointer)
    return SimpleNamespace(
        device="npu:0",
        dtype=torch.float32,
        shape=(count, 2, 3),
        stride=lambda: (6, 3, 1),
        storage_offset=lambda: 0,
        element_size=lambda: 4,
        untyped_storage=lambda: storage,
        physical_elements=physical_elements,
        npu_format=29,
    )


def _runtime():
    return SimpleNamespace(
        get_storage_size=lambda tensor: tensor.physical_elements,
        get_npu_format=lambda tensor: tensor.npu_format,
    )


def test_nz_copy_uses_physical_padding_and_slot_offset():
    source, destination = _tensor(1, 1000), _tensor(3, 2000)
    assert TorchExpertTransfer._physical_copy_parameters(destination, source, 2, _runtime()) == (2096, 1000, 48)


@pytest.mark.parametrize("slot", [-1, 3, True])
def test_physical_copy_rejects_invalid_slot(slot):
    with pytest.raises(ValueError, match="out of range"):
        TorchExpertTransfer._physical_copy_parameters(_tensor(3, 2000), _tensor(1, 1000), slot, _runtime())


@pytest.mark.parametrize("field,value", [("npu_format", 2), ("device", "npu:1"), ("dtype", torch.float16)])
def test_physical_copy_rejects_layout_mismatch(field, value):
    destination, source = _tensor(3, 2000), _tensor(1, 1000)
    setattr(source, field, value)
    with pytest.raises(ValueError, match="physical layouts"):
        TorchExpertTransfer._physical_copy_parameters(destination, source, 0, _runtime())


@pytest.mark.parametrize("elements,storage_bytes", [(35, 144), (36, 143)])
def test_physical_copy_rejects_truncated_or_nonuniform_storage(elements, storage_bytes):
    destination = _tensor(3, 2000, physical_elements=elements, storage_bytes=storage_bytes)
    with pytest.raises(ValueError, match="physical storage"):
        TorchExpertTransfer._physical_copy_parameters(destination, _tensor(1, 1000), 0, _runtime())


def test_physical_copy_rejects_storage_view():
    destination = _tensor(3, 2000)
    destination.storage_offset = lambda: 1
    with pytest.raises(ValueError, match="physical layouts"):
        TorchExpertTransfer._physical_copy_parameters(destination, _tensor(1, 1000), 0, _runtime())


def _npu_transfer(monkeypatch, error=0, error_on_call=1):
    events: list[object] = []
    runtime = _runtime()
    monkeypatch.setitem(sys.modules, "torch_npu", runtime)
    monkeypatch.setattr(torch, "device", lambda device: SimpleNamespace(type="npu"))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda device: SimpleNamespace(npu_stream=123, synchronize=lambda: events.append("sync"))
        ),
        raising=False,
    )

    def memcpy(dst, dst_size, src, size, kind, stream):
        events.append((dst, src, size, kind))
        assert dst_size == size
        assert stream == 123
        return error if len(events) == error_on_call else 0

    monkeypatch.setitem(sys.modules, "acl", SimpleNamespace(rt=SimpleNamespace(memcpy_async=memcpy)))
    names = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    destination = {name: _tensor(3, 2000 + i * 1000) for i, name in enumerate(names)}
    prepared = {
        name: PreparedExpertTensor(
            data=SimpleNamespace(
                shape=(8, 48),
                dtype=torch.uint8,
                device=SimpleNamespace(type="cpu"),
                data_ptr=lambda i=i: 10000 + i * 1000,
                is_contiguous=lambda: True,
                is_pinned=lambda: True,
            ),
            shape=(2, 3),
            stride=(6, 3, 1),
            dtype=torch.float32,
            device="npu:0",
            npu_format=29,
            expert_bytes=48,
        )
        for i, name in enumerate(names)
    }

    def forbidden_materialize(*args):
        raise AssertionError("runtime must not materialize or format-cast experts")

    adapter = SimpleNamespace(materialize=forbidden_materialize)
    return adapter, destination, SimpleNamespace(prepared=prepared), events


def test_npu_transfer_copies_native_batch_and_fences_once(monkeypatch):
    adapter, destination, store, events = _npu_transfer(monkeypatch)
    TorchExpertTransfer.copy_experts(adapter, store, destination, [(7, 2), (1, 0)], "npu:0")
    assert events == [
        *[(2096 + i * 1000, 10336 + i * 1000, 48, 1) for i in range(4)],
        *[(2000 + i * 1000, 10048 + i * 1000, 48, 1) for i in range(4)],
        "sync",
    ]


def test_npu_transfer_checks_entire_bundle_before_mutating_storage(monkeypatch):
    adapter, destination, store, events = _npu_transfer(monkeypatch)
    destination["w2_weight_scale"].npu_format = 2
    with pytest.raises(ValueError, match="physical layouts"):
        TorchExpertTransfer.copy_experts(adapter, store, destination, [(7, 0)], "npu:0")
    assert events == []


def test_npu_transfer_propagates_acl_error(monkeypatch):
    adapter, destination, store, events = _npu_transfer(monkeypatch, error=42)
    with pytest.raises(RuntimeError, match="ACL error 42"):
        TorchExpertTransfer.copy_experts(adapter, store, destination, [(7, 0)], "npu:0")
    assert events == [(2000, 10336, 48, 1), "sync"]


def test_partial_native_batch_is_drained_before_propagating_failure(monkeypatch):
    adapter, destination, store, events = _npu_transfer(monkeypatch, error=42, error_on_call=3)
    with pytest.raises(RuntimeError, match="ACL error 42"):
        TorchExpertTransfer.copy_experts(adapter, store, destination, [(7, 0), (1, 2)], "npu:0")
    assert events == [*[(2000 + i * 1000, 10336 + i * 1000, 48, 1) for i in range(3)], "sync"]


@pytest.mark.parametrize("expert_id,slot", [(8, 0), (-1, 0), (True, 0), (1, 3), (1, -1), (1, True)])
def test_invalid_later_load_is_rejected_before_any_copy(monkeypatch, expert_id, slot):
    adapter, destination, store, events = _npu_transfer(monkeypatch)
    with pytest.raises(ValueError, match="out of range"):
        TorchExpertTransfer.copy_experts(adapter, store, destination, [(7, 0), (expert_id, slot)], "npu:0")
    assert events == []


def test_native_transfer_rejects_unpinned_sources(monkeypatch):
    adapter, destination, store, events = _npu_transfer(monkeypatch)
    store.prepared["w13_weight"].data.is_pinned = lambda: False
    with pytest.raises(ValueError, match="pinned bounded"):
        TorchExpertTransfer.copy_experts(adapter, store, destination, [(7, 0)], "npu:0")
    assert events == []


def test_no_misses_do_not_submit_or_synchronize(monkeypatch):
    adapter, destination, store, events = _npu_transfer(monkeypatch)
    TorchExpertTransfer.copy_experts(adapter, store, destination, [], "npu:0")
    assert events == []


def test_native_transfer_rejects_incomplete_bundle(monkeypatch):
    adapter, destination, store, events = _npu_transfer(monkeypatch)
    del store.prepared["w2_weight_scale"]
    with pytest.raises(RuntimeError, match="incomplete"):
        TorchExpertTransfer.copy_experts(adapter, store, destination, [(7, 0)], "npu:0")
    assert events == []
