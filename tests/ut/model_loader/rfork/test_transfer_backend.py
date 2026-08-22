#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from types import SimpleNamespace

import torch

import vllm_ascend.model_loader.rfork.transfer_backend as transfer_backend
from vllm_ascend.model_loader.rfork.aligned_memory import (
    AlignedStorageError,
    materialize_aligned_storage,
)
from vllm_ascend.model_loader.rfork.transfer_backend import (
    RForkTransferBackend,
    _collect_checkpoint_layout_tensors,
    _collect_processed_layout_tensors,
    _parse_weight_info,
    _reshape_tensor_to_seed_shape,
    get_remote_instance_transfer_engine_info,
)


def test_parse_weight_info_keeps_backward_compatibility():
    assert _parse_weight_info([1, 2, 4]) == (1, 2, 4, None)


def test_parse_weight_info_accepts_shape_metadata_from_json():
    assert _parse_weight_info([1, 6, 2, [2, 3]]) == (1, 6, 2, (2, 3))


def test_parse_weight_info_rejects_invalid_shape_metadata():
    assert _parse_weight_info([1, 6, 2, ["2", 3]]) is None
    assert _parse_weight_info([1, 6, 2, -1]) is None


def test_reshape_tensor_to_seed_shape_updates_tensor_metadata_only():
    tensor = torch.arange(6).reshape(2, 3)
    original_ptr = tensor.data_ptr()

    assert _reshape_tensor_to_seed_shape("weight", tensor, (1, 2, 3))

    assert tuple(tensor.shape) == (1, 2, 3)
    assert tensor.data_ptr() == original_ptr


def test_reshape_tensor_to_seed_shape_rejects_numel_mismatch():
    tensor = torch.arange(6).reshape(2, 3)

    assert not _reshape_tensor_to_seed_shape("weight", tensor, (2, 2))
    assert tuple(tensor.shape) == (2, 3)


def test_recv_from_source_refreshes_registered_shape_after_reshape(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": (2, 3)}

    monkeypatch.setattr(
        transfer_backend,
        "_iter_transferable_tensors",
        lambda model, processed_layout: iter([("weight", tensor)]),
    )
    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: (
            "seed-session",
            {"weight": [1, tensor.numel(), tensor.element_size()]},
            {"weight": [1, 2, 3]},
        ),
    )

    assert backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
    assert tuple(tensor.shape) == (1, 2, 3)
    assert backend.rfork_transfer_engine_weights_shape_dict["weight"] == (1, 2, 3)


def test_recv_from_source_reuses_registered_transferable_tensors(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": (2, 3)}
    backend._registered_transferable_tensors = [("weight", tensor)]

    def fail_if_rescanned(model, processed_layout):
        raise AssertionError("recv_from_source should reuse the registered tensor cache")

    monkeypatch.setattr(transfer_backend, "_iter_transferable_tensors", fail_if_rescanned)
    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: (
            "seed-session",
            {"weight": [1, tensor.numel(), tensor.element_size(), [2, 3]]},
            None,
        ),
    )

    assert backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
    assert backend._registered_transferable_tensors is None


def test_transferable_tensor_scan_depends_on_runtime_layout(monkeypatch):
    class _RuntimeImpl:
        def __init__(self):
            self.runtime_weight = torch.ones(2)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("buffer", torch.ones(2))
            self.runtime_constant = torch.ones(2)
            self.impl = _RuntimeImpl()

    monkeypatch.setattr(transfer_backend, "_is_transferable_tensor", lambda tensor: True)
    model = _Model()

    processed_names = {name for name, _ in _collect_processed_layout_tensors(model)}
    checkpoint_names = {name for name, _ in _collect_checkpoint_layout_tensors(model)}

    assert processed_names == {"weight", "buffer", "runtime_constant", "impl.runtime_weight"}
    assert checkpoint_names == {"weight", "buffer", "impl.runtime_weight"}


def test_get_remote_instance_transfer_engine_info_non_200_returns_three_values(monkeypatch):
    monkeypatch.setattr(
        transfer_backend.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=503),
    )

    assert get_remote_instance_transfer_engine_info("http://seed", "seed-key") == (None, None, None)


def test_materialize_aligned_storage_preserves_values_and_shared_storage():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            source = torch.arange(8, dtype=torch.float32)
            self.left = torch.nn.Parameter(source[:4])
            self.right = torch.nn.Parameter(source[4:])

    model = _Model()
    left_values = model.left.detach().clone()
    right_values = model.right.detach().clone()
    storage = materialize_aligned_storage(
        model,
        [("left", model.left), ("right", model.right)],
        copy_values=True,
        alignment=256,
    )

    assert storage.backing_view.data_ptr() % 256 == 0
    assert model.left.untyped_storage().data_ptr() == model.right.untyped_storage().data_ptr()
    assert torch.equal(model.left, left_values)
    assert torch.equal(model.right, right_values)
    assert len(storage.registrations) == 1
    assert storage.registrations[0].backing_addr == storage.backing_view.data_ptr()


def test_materialize_aligned_storage_rejects_noncontiguous_tensor():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 3).transpose(0, 1))

    model = _Model()
    try:
        materialize_aligned_storage(
            model,
            [("weight", model.weight)],
            copy_values=True,
            alignment=256,
        )
    except AlignedStorageError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("noncontiguous RFork tensor should be rejected")


def test_register_memory_region_uses_backing_aware_api_for_aligned_route(monkeypatch):
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))

    class _MemoryRegistration:
        def __init__(self, logical_addr, logical_length, backing_addr, backing_length):
            self.logical_addr = logical_addr
            self.logical_length = logical_length
            self.backing_addr = backing_addr
            self.backing_length = backing_length

    captured = []
    result = SimpleNamespace(is_error=lambda: False)
    engine = SimpleNamespace(
        batch_register_memory_ex=lambda registrations, location: captured.extend(registrations) or result
    )
    model = _Model()
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = engine
    backend._route_policy = "auto"
    backend._memory_registration_type = _MemoryRegistration
    backend._registered_transferable_tensors = None
    backend._aligned_storage = None

    monkeypatch.setattr(
        transfer_backend,
        "_iter_transferable_tensors",
        lambda model, processed_layout: iter([("weight", model.weight)]),
    )

    assert backend.register_memory_region(model, False)
    assert captured
    assert captured[0].backing_addr % (2 * 1024 * 1024) == 0
    assert captured[0].logical_addr == model.weight.data_ptr()
    assert captured[0].logical_length == model.weight.numel() * model.weight.element_size()
