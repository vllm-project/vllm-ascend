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

import gc
import sys
import threading
import weakref
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_ascend.model_loader.rfork.transfer_backend as transfer_backend
from vllm_ascend.model_loader.rfork.transfer_backend import (
    RForkTransferBackend,
    _collect_checkpoint_layout_tensors,
    _collect_processed_layout_tensors,
    _parse_weight_info,
    _reshape_tensor_to_seed_shape,
    _split_tensors_by_excluded_blocks,
    _subtract_weight_blocks,
    get_remote_instance_transfer_engine_info,
)


def test_parse_weight_info_keeps_backward_compatibility():
    assert _parse_weight_info([1, 2, 4]) == (1, 2, 4, None)


def test_backend_initialization_rejects_transfer_engine_without_memory_registration(monkeypatch):
    yr_module = ModuleType("yr")
    datasystem_module = ModuleType("yr.datasystem")
    datasystem_module.__dict__["TransferEngine"] = object
    datasystem_module.__dict__["ErrorCode"] = SimpleNamespace(kNotReady=1)
    yr_module.__dict__["datasystem"] = datasystem_module
    monkeypatch.setitem(sys.modules, "yr", yr_module)
    monkeypatch.setitem(sys.modules, "yr.datasystem", datasystem_module)

    with pytest.raises(ImportError, match="MemoryRegistration"):
        RForkTransferBackend()


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


def test_recv_from_source_retains_registered_transferable_tensor_owners(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    tensor_ref = weakref.ref(tensor)
    tensor_numel = tensor.numel()
    tensor_element_size = tensor.element_size()
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": (2, 3)}
    registered_tensors = [("weight", tensor)]
    backend._registered_transferable_tensors = registered_tensors

    def fail_if_rescanned(model, processed_layout):
        raise AssertionError("recv_from_source should reuse the registered tensor cache")

    monkeypatch.setattr(transfer_backend, "_iter_transferable_tensors", fail_if_rescanned)
    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: (
            "seed-session",
            {"weight": [1, tensor_numel, tensor_element_size, [2, 3]]},
            None,
        ),
    )

    assert backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
    assert backend._registered_transferable_tensors is registered_tensors
    del registered_tensors
    del tensor
    gc.collect()
    assert tensor_ref() is not None


def test_recv_from_source_keeps_registered_tensors_when_seed_metadata_is_unavailable(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace()
    registered_tensors = [("weight", tensor)]
    backend._registered_transferable_tensors = registered_tensors

    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: (None, None, None),
    )

    assert not backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
    assert backend._registered_transferable_tensors is registered_tensors


def test_unregister_memory_region_releases_registered_tensor_owners():
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_unregister_memory=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_info_dict = {"weight": (tensor.data_ptr(), tensor.numel(), 1)}
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": tuple(tensor.shape)}
    backend.registered_weight_blocks = [(tensor.data_ptr(), tensor.numel() * tensor.element_size())]
    backend._registered_transferable_tensors = [("weight", tensor)]

    assert backend.unregister_memory_region()
    assert backend._registered_transferable_tensors is None
    assert backend.rfork_transfer_engine_weights_info_dict is None
    assert backend.rfork_transfer_engine_weights_shape_dict is None
    assert backend.registered_weight_blocks == []


def test_unregister_memory_region_keeps_tracking_when_engine_fails():
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_unregister_memory=lambda *args: SimpleNamespace(
            is_error=lambda: True, to_string=lambda: "mock unregister error"
        )
    )
    stale_blocks = [(4096, 128)]
    backend.registered_weight_blocks = list(stale_blocks)
    backend.rfork_transfer_engine_weights_info_dict = {"weight": (4096, 128, 2)}
    backend.rfork_transfer_engine_weights_shape_dict = {"weight": (64,)}
    backend._registered_transferable_tensors = []

    assert not backend.unregister_memory_region()
    assert backend.registered_weight_blocks == stale_blocks
    assert backend.rfork_transfer_engine_weights_info_dict is not None
    assert backend.rfork_transfer_engine_weights_shape_dict is not None


def test_register_memory_region_retries_stale_blocks_before_registering(monkeypatch):
    storage = torch.arange(10, dtype=torch.float32)
    stale_blocks = [(storage.data_ptr() + 4096, 256)]
    backend, registered_calls, unregistered_calls = _make_register_memory_region_backend(
        monkeypatch,
        [("weight", storage)],
        [
            {
                "address": storage.data_ptr(),
                "size": storage.numel() * storage.element_size(),
                "state": "active_allocated",
            }
        ],
        stale_blocks=stale_blocks,
    )

    assert backend.register_memory_region(object(), True)

    assert unregistered_calls == [[stale_blocks[0][0]]]
    assert registered_calls == [[(storage.data_ptr(), 40, storage.data_ptr(), 40)]]
    assert backend.registered_weight_blocks == [(storage.data_ptr(), 40)]


def test_register_memory_region_aborts_when_stale_unregister_fails(monkeypatch):
    storage = torch.arange(10, dtype=torch.float32)
    stale_blocks = [(storage.data_ptr() + 4096, 256)]
    backend, registered_calls, unregistered_calls = _make_register_memory_region_backend(
        monkeypatch,
        [("weight", storage)],
        [
            {
                "address": storage.data_ptr(),
                "size": storage.numel() * storage.element_size(),
                "state": "active_allocated",
            }
        ],
        stale_blocks=stale_blocks,
        unregister_error=True,
    )

    assert not backend.register_memory_region(object(), True)

    assert registered_calls == []
    assert unregistered_calls == [[stale_blocks[0][0]]]
    assert backend.registered_weight_blocks == stale_blocks


def test_register_memory_region_uses_logical_ranges_with_allocator_backing(monkeypatch):
    tensor = torch.arange(8, dtype=torch.uint8)
    logical = tensor[2:6]
    backing_start = tensor.data_ptr()
    backing_size = tensor.numel() * tensor.element_size()
    registrations: list[tuple[int, int, int, int]] = []

    class _MemoryRegistration:
        def __init__(self, logical_addr, logical_length, backing_addr, backing_length):
            self.values = (logical_addr, logical_length, backing_addr, backing_length)

    def batch_register_memory_ex(items):
        registrations.extend(item.values for item in items)
        return SimpleNamespace(is_error=lambda: False)

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(batch_register_memory_ex=batch_register_memory_ex)
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    monkeypatch.setattr(transfer_backend, "_iter_transferable_tensors", lambda *args: iter([("weight", logical)]))
    monkeypatch.setattr(
        transfer_backend.torch,
        "npu",
        SimpleNamespace(
            memory=SimpleNamespace(
                memory_snapshot=lambda: [
                    {"blocks": [{"address": backing_start, "size": backing_size, "state": "active_allocated"}]}
                ]
            )
        ),
        raising=False,
    )

    assert backend.register_memory_region(object(), False)
    assert registrations == [(logical.data_ptr(), logical.numel(), backing_start, backing_size)]
    assert backend.registered_memory_addresses == [logical.data_ptr()]


def test_extended_registration_merges_overlapping_logical_ranges(monkeypatch):
    tensor = torch.arange(8, dtype=torch.uint8)
    first = tensor[1:5]
    second = tensor[3:7]
    backing_start = tensor.data_ptr()
    registrations: list[tuple[int, ...]] = []

    class _MemoryRegistration:
        def __init__(self, *values):
            self.values = values

    def batch_register_memory_ex(items):
        registrations.extend(item.values for item in items)
        return SimpleNamespace(is_error=lambda: False)

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(batch_register_memory_ex=batch_register_memory_ex)
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    monkeypatch.setattr(
        transfer_backend,
        "_iter_transferable_tensors",
        lambda *args: iter([("first", first), ("second", second)]),
    )
    monkeypatch.setattr(
        transfer_backend.torch,
        "npu",
        SimpleNamespace(
            memory=SimpleNamespace(
                memory_snapshot=lambda: [
                    {"blocks": [{"address": backing_start, "size": tensor.numel(), "state": "active_allocated"}]}
                ]
            )
        ),
        raising=False,
    )

    assert backend.register_memory_region(object(), False)
    assert registrations == [(first.data_ptr(), 6, backing_start, tensor.numel())]


def test_register_memory_region_rejects_transfer_engine_without_extended_registration(monkeypatch):
    tensor = torch.arange(4, dtype=torch.uint8)
    calls = []

    def batch_register_memory(addresses, sizes):
        calls.append((list(addresses), list(sizes)))

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(batch_register_memory=batch_register_memory)
    backend._memory_registration_cls = None
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    monkeypatch.setattr(transfer_backend, "_iter_transferable_tensors", lambda *args: iter([("weight", tensor)]))
    monkeypatch.setattr(
        transfer_backend.torch,
        "npu",
        SimpleNamespace(
            memory=SimpleNamespace(
                memory_snapshot=lambda: [
                    {"blocks": [{"address": tensor.data_ptr(), "size": tensor.numel(), "state": "active_allocated"}]}
                ]
            )
        ),
        raising=False,
    )

    assert not backend.register_memory_region(object(), False)
    assert calls == []
    assert backend.registered_weight_blocks == []
    assert backend.registered_memory_addresses == []


def test_unregister_uses_logical_registration_addresses():
    calls = []

    def batch_unregister_memory(addresses):
        calls.append(addresses)
        return SimpleNamespace(is_error=lambda: False)

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(batch_unregister_memory=batch_unregister_memory)
    backend.registered_weight_blocks = [(100, 1000)]
    backend.registered_memory_addresses = [120, 400]
    backend._registered_transferable_tensors = []

    assert backend.unregister_memory_region()
    assert calls == [[120, 400]]


def test_unregister_splits_batches_and_preserves_only_unfinished_addresses(monkeypatch):
    calls = []

    def batch_unregister_memory(addresses):
        calls.append(list(addresses))
        return SimpleNamespace(
            is_error=lambda: len(calls) == 2,
            to_string=lambda: "mock unregister error",
        )

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(batch_unregister_memory=batch_unregister_memory)
    backend.registered_weight_blocks = [(100, 1000)]
    backend.registered_memory_addresses = [120, 400, 800]
    backend._registered_transferable_tensors = [("weight", torch.ones(1))]
    monkeypatch.setattr(transfer_backend, "MAX_MEMORY_REGISTRATION_BATCH_ITEMS", 2)

    assert not backend.unregister_memory_region()
    assert calls == [[120, 400], [800]]
    assert backend.registered_memory_addresses == [800]
    assert backend._registered_transferable_tensors is not None

    assert backend.unregister_memory_region()
    assert calls[-1] == [800]
    assert backend.registered_memory_addresses == []
    assert backend._registered_transferable_tensors is None


def test_finalize_retries_not_ready_and_clears_state_only_after_success(monkeypatch):
    attempts: list[bool] = []
    sleep_calls: list[float] = []

    class _Ret:
        def __init__(self, error, code):
            self.error = error
            self.code = code

        def is_error(self):
            return self.error

        def get_code(self):
            return self.code

        def to_string(self):
            return self.code

    results = iter([_Ret(True, "not-ready"), _Ret(False, "ok")])

    def finalize():
        attempts.append(True)
        return next(results)

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(finalize=finalize)
    backend.rfork_transfer_engine_session_id = "session"
    backend.registered_weight_blocks = [(1, 1)]
    backend.registered_memory_addresses = [1]
    backend._registered_transferable_tensors = [("weight", torch.ones(1))]
    backend._not_ready_error_code = "not-ready"
    backend._is_initialized = True

    monkeypatch.setattr(transfer_backend.time, "sleep", sleep_calls.append)

    assert backend.finalize_transfer_engine()
    assert len(attempts) == 2
    assert sleep_calls == [transfer_backend.TRANSFER_ENGINE_FINALIZE_RETRY_INTERVAL_SEC]
    assert backend._is_initialized is False
    assert backend.rfork_transfer_engine_session_id is None
    assert backend.registered_weight_blocks == []
    assert backend._registered_transferable_tensors is None


def test_finalize_failure_preserves_registered_tensor_owners():
    owners = [("weight", torch.ones(1))]
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        finalize=lambda: SimpleNamespace(
            is_error=lambda: True,
            get_code=lambda: "fatal",
            to_string=lambda: "fatal",
        )
    )
    backend.rfork_transfer_engine_session_id = "session"
    backend.registered_weight_blocks = [(1, 1)]
    backend.registered_memory_addresses = [1]
    backend._registered_transferable_tensors = owners
    backend._not_ready_error_code = "not-ready"
    backend._is_initialized = True

    assert not backend.finalize_transfer_engine()
    assert backend._is_initialized is True
    assert backend.registered_weight_blocks == [(1, 1)]
    assert backend._registered_transferable_tensors is owners


def test_finalize_stops_after_bounded_not_ready_retries(monkeypatch):
    attempts: list[bool] = []
    sleep_calls: list[float] = []
    result = SimpleNamespace(
        is_error=lambda: True,
        get_code=lambda: "not-ready",
        to_string=lambda: "not-ready",
    )

    def finalize():
        attempts.append(True)
        return result

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(finalize=finalize)
    backend._not_ready_error_code = "not-ready"
    backend._is_initialized = True
    monkeypatch.setattr(transfer_backend.time, "sleep", sleep_calls.append)

    assert not backend.finalize_transfer_engine(max_attempts=3, retry_interval_sec=0.25)
    assert len(attempts) == 3
    assert sleep_calls == [0.25, 0.25]


def test_finalize_is_idempotent_after_success():
    calls = []

    def finalize():
        calls.append(True)

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(finalize=finalize)
    backend._is_initialized = False

    assert backend.finalize_transfer_engine()
    assert calls == []


def test_transfer_lifecycle_lock_serializes_concurrent_calls():
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend._lifecycle_lock = threading.RLock()
    first_entered = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    calls = []

    def recv_locked(*args):
        calls.append(args)
        if len(calls) == 1:
            first_entered.set()
            assert release_first.wait(1)
        return True

    backend.__dict__["_recv_from_source_locked"] = recv_locked
    args = (object(), "127.0.0.1", 8000, "key", False)
    first = threading.Thread(target=backend.recv_from_source, args=args)

    def run_second():
        backend.recv_from_source(*args)
        second_done.set()

    second = threading.Thread(target=run_second)

    first.start()
    assert first_entered.wait(1)
    second.start()
    assert not second_done.wait(0.05)
    assert len(calls) == 1
    release_first.set()
    first.join(1)
    second.join(1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(calls) == 2


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


def test_subtract_weight_blocks_cuts_excluded_ranges():
    assert _subtract_weight_blocks([(0, 100)], []) == [(0, 100)]
    assert _subtract_weight_blocks([(0, 100)], [(0, 100)]) == []
    assert _subtract_weight_blocks([(0, 100)], [(10, 20)]) == [(0, 10), (30, 70)]
    assert _subtract_weight_blocks([(0, 100)], [(-50, 60)]) == [(10, 90)]
    assert _subtract_weight_blocks([(0, 100)], [(90, 50)]) == [(0, 90)]
    assert _subtract_weight_blocks([(0, 100)], [(200, 10)]) == [(0, 100)]
    assert _subtract_weight_blocks(
        [(0, 100), (200, 100)],
        [(50, 10), (240, 10)],
    ) == [(0, 50), (60, 40), (200, 40), (250, 50)]


def test_split_tensors_by_excluded_blocks_separates_shared_storage():
    storage = torch.arange(10)
    shared_tensor = storage[:3]
    own_tensor = torch.arange(4)

    kept_tensors, excluded_names = _split_tensors_by_excluded_blocks(
        [("model.embed_tokens.weight", shared_tensor), ("layers.0.fc.weight", own_tensor)],
        [(storage.data_ptr(), shared_tensor.numel() * shared_tensor.element_size())],
    )

    assert [name for name, _ in kept_tensors] == ["layers.0.fc.weight"]
    assert excluded_names == ["model.embed_tokens.weight"]


def test_split_tensors_by_excluded_blocks_noop_without_exclusion():
    tensor = torch.arange(6)
    kept_tensors, excluded_names = _split_tensors_by_excluded_blocks([("weight", tensor)], [])

    assert kept_tensors == [("weight", tensor)]
    assert excluded_names == []


def _make_register_memory_region_backend(
    monkeypatch,
    tensors,
    snapshot_blocks,
    stale_blocks=(),
    unregister_error=False,
):
    registered_calls = []
    unregistered_calls = []

    class _MemoryRegistration:
        def __init__(self, logical_addr, logical_length, backing_addr, backing_length):
            self.values = (logical_addr, logical_length, backing_addr, backing_length)

    def batch_register_memory_ex(registrations):
        registered_calls.append([registration.values for registration in registrations])
        return SimpleNamespace(is_error=lambda: False)

    def batch_unregister_memory(addresses):
        unregistered_calls.append(list(addresses))
        return SimpleNamespace(is_error=lambda: unregister_error, to_string=lambda: "mock unregister error")

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_register_memory_ex=batch_register_memory_ex,
        batch_unregister_memory=batch_unregister_memory,
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = list(stale_blocks)
    backend.registered_memory_addresses = [address for address, _ in stale_blocks]
    monkeypatch.setattr(
        transfer_backend,
        "_iter_transferable_tensors",
        lambda model, processed_layout: iter(tensors),
    )
    snapshot = [{"blocks": snapshot_blocks}]
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(memory=SimpleNamespace(memory_snapshot=lambda: snapshot)),
        raising=False,
    )
    return backend, registered_calls, unregistered_calls


def test_register_memory_region_skips_shared_weights(monkeypatch):
    storage = torch.arange(100, dtype=torch.float32)
    shared_weight = storage[:10]
    own_weight = storage[20:32].reshape(2, 6)
    backend, registered_calls, _ = _make_register_memory_region_backend(
        monkeypatch,
        [("model.embed_tokens.weight", shared_weight), ("layers.0.fc.weight", own_weight)],
        [
            {
                "address": storage.data_ptr(),
                "size": storage.numel() * storage.element_size(),
                "state": "active_allocated",
            }
        ],
    )

    excluded_blocks = [(storage.data_ptr(), 40)]
    assert backend.register_memory_region(object(), True, exclude_blocks=excluded_blocks)

    assert set(backend.rfork_transfer_engine_weights_info_dict) == {"layers.0.fc.weight"}
    assert [name for name, _ in backend._registered_transferable_tensors] == ["layers.0.fc.weight"]
    assert backend.excluded_weight_blocks == excluded_blocks
    assert registered_calls == [
        [
            (
                own_weight.data_ptr(),
                own_weight.numel() * own_weight.element_size(),
                storage.data_ptr() + 40,
                360,
            )
        ]
    ]
    assert backend.registered_weight_blocks == [(storage.data_ptr() + 40, 360)]


def test_register_memory_region_registers_all_weights_without_exclusion(monkeypatch):
    storage = torch.arange(100, dtype=torch.float32)
    weight_a = storage[:10]
    weight_b = storage[20:32]
    backend, registered_calls, _ = _make_register_memory_region_backend(
        monkeypatch,
        [("a.weight", weight_a), ("b.weight", weight_b)],
        [
            {
                "address": storage.data_ptr(),
                "size": storage.numel() * storage.element_size(),
                "state": "active_allocated",
            }
        ],
    )

    assert backend.register_memory_region(object(), True)

    assert set(backend.rfork_transfer_engine_weights_info_dict) == {"a.weight", "b.weight"}
    assert registered_calls == [
        [
            (weight_a.data_ptr(), 40, storage.data_ptr(), 400),
            (weight_b.data_ptr(), 48, storage.data_ptr(), 400),
        ]
    ]
    assert backend.excluded_weight_blocks == []


def test_register_memory_region_splits_large_registration_batches(monkeypatch):
    storage = torch.arange(12, dtype=torch.uint8)
    weights = [storage[0:2], storage[4:6], storage[8:10]]
    backend, registered_calls, _ = _make_register_memory_region_backend(
        monkeypatch,
        [(f"weight_{index}", weight) for index, weight in enumerate(weights)],
        [
            {
                "address": storage.data_ptr(),
                "size": storage.numel(),
                "state": "active_allocated",
            }
        ],
    )
    monkeypatch.setattr(transfer_backend, "MAX_MEMORY_REGISTRATION_BATCH_ITEMS", 2)

    assert backend.register_memory_region(object(), False)
    assert [len(batch) for batch in registered_calls] == [2, 1]
    assert backend.registered_memory_addresses == [weight.data_ptr() for weight in weights]


def test_register_memory_region_skips_empty_batch_after_excluding_all_weights(monkeypatch):
    storage = torch.arange(10, dtype=torch.float32)
    backend, registered_calls, _ = _make_register_memory_region_backend(
        monkeypatch,
        [("model.embed_tokens.weight", storage)],
        [
            {
                "address": storage.data_ptr(),
                "size": storage.numel() * storage.element_size(),
                "state": "active_allocated",
            }
        ],
    )

    excluded_blocks = [(storage.data_ptr(), storage.numel() * storage.element_size())]
    assert backend.register_memory_region(object(), True, exclude_blocks=excluded_blocks)

    assert backend.rfork_transfer_engine_weights_info_dict == {}
    assert backend.rfork_transfer_engine_weights_shape_dict == {}
    assert backend._registered_transferable_tensors == []
    assert backend.registered_weight_blocks == []
    assert registered_calls == []


def test_recv_from_source_defensively_skips_pre_registered_weights(monkeypatch):
    storage = torch.arange(100, dtype=torch.float32)
    shared_weight = storage[:10]
    own_weight = storage[20:32]
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_shape_dict = {}
    backend.excluded_weight_blocks = [(storage.data_ptr(), 40)]
    backend._registered_transferable_tensors = None

    monkeypatch.setattr(
        transfer_backend,
        "_iter_transferable_tensors",
        lambda model, processed_layout: iter(
            [
                ("model.embed_tokens.weight", shared_weight),
                ("layers.0.fc.weight", own_weight),
            ]
        ),
    )

    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: (
            "seed-session",
            {"layers.0.fc.weight": [7, own_weight.numel(), own_weight.element_size()]},
            None,
        ),
    )

    assert backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
    assert backend.rfork_transfer_engine_weights_shape_dict == {"layers.0.fc.weight": tuple(own_weight.shape)}


def test_recv_from_source_fails_for_unknown_weight_outside_shared_blocks(monkeypatch):
    own_weight = torch.arange(12, dtype=torch.float32)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.rfork_transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.rfork_transfer_engine_weights_shape_dict = {}
    backend.excluded_weight_blocks = [(4096, 128)]
    backend._registered_transferable_tensors = [("layers.0.fc.weight", own_weight)]

    monkeypatch.setattr(
        transfer_backend,
        "get_remote_instance_transfer_engine_info",
        lambda *args: ("seed-session", {}, None),
    )

    assert not backend.recv_from_source(object(), "127.0.0.1", 8000, "seed-key", True)
