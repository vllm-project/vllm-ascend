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
import random
import sys
import threading
import weakref
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_ascend.model_loader.rfork.transfer_backend as transfer_backend
from vllm_ascend.model_loader.rfork.transfer_backend import (
    RForkTransferBackend,
    _select_weight_blocks,
    _split_tensors_by_excluded_blocks,
    iter_transfer_chunks,
)
from vllm_ascend.model_loader.rfork.types import SeedTransferInfo


def _append_and_return(items: list[Any], value: Any, result: Any) -> Any:
    items.append(value)
    return result


def _extend_and_return(items: list[Any], values: Any, result: Any) -> Any:
    items.extend(values)
    return result


def test_weight_block_index_handles_nested_ranges_and_touching_endpoints():
    snapshot = [
        {
            "blocks": [
                {"address": address, "size": size, "state": state}
                for address, size, state in [
                    (200, 10, "active_allocated"),
                    (90, 10, "active_allocated"),
                    (180, 10, "active_allocated"),
                    (190, 10, "active_allocated"),
                    (185, 2, "inactive"),
                    (True, 10, "active_allocated"),
                ]
            ]
        }
    ]
    assert _select_weight_blocks(snapshot, [(100, 200), (110, 120)]) == [(180, 20)]
    assert _select_weight_blocks(snapshot, []) == []


def test_weight_block_index_matches_exhaustive_overlap_search():
    rng = random.Random(42)
    for _ in range(30):
        ranges = [(start, start + rng.randrange(1, 100)) for start in rng.sample(range(1, 500), 70)]
        blocks = [(start, rng.randrange(1, 30)) for start in rng.sample(range(1, 600), 100)]
        snapshot = [{"blocks": [{"address": a, "size": s, "state": "active_allocated"} for a, s in blocks]}]
        selected = sorted((a, s) for a, s in blocks if any(a < end and a + s > start for start, end in ranges))
        expected: list[tuple[int, int]] = []
        for address, size in selected:
            if not expected or expected[-1][0] + expected[-1][1] < address:
                expected.append((address, size))
            else:
                start, length = expected[-1]
                expected[-1] = (start, max(start + length, address + size) - start)
        assert _select_weight_blocks(snapshot, ranges) == expected


@pytest.mark.parametrize("gap", [0, 1])
def test_registration_requires_full_coverage_across_allocator_blocks(monkeypatch, gap):
    tensor = torch.arange(8, dtype=torch.uint8)
    start = tensor.data_ptr()
    backend, registrations, _ = _make_register_memory_region_backend(
        monkeypatch,
        [("weight", tensor)],
        [
            {"address": start + 4 + gap, "size": 4, "state": "active_allocated"},
            {"address": start, "size": 4, "state": "active_allocated"},
        ],
    )
    assert backend.register_memory_region(object(), False) is (gap == 0)
    assert bool(registrations) is (gap == 0)


def test_backend_initialization_rejects_transfer_engine_without_memory_registration(monkeypatch):
    yr_module = ModuleType("yr")
    datasystem_module = ModuleType("yr.datasystem")
    datasystem_module.TransferEngine = object  # type: ignore[attr-defined]
    datasystem_module.ErrorCode = SimpleNamespace(kNotReady=1)  # type: ignore[attr-defined]
    yr_module.datasystem = datasystem_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yr", yr_module)
    monkeypatch.setitem(sys.modules, "yr.datasystem", datasystem_module)

    with pytest.raises(ImportError, match="MemoryRegistration"):
        RForkTransferBackend()


def test_iter_transfer_chunks_splits_single_large_tensor():
    chunk_limit = transfer_backend.MAX_TRANSFER_CHUNK_BYTES
    chunks = list(
        iter_transfer_chunks(
            ["large"],
            [10_000],
            [20_000],
            [chunk_limit + 17],
        )
    )

    assert len(chunks) == 2
    assert [length for _, _, _, lengths in chunks for length in lengths] == [chunk_limit, 17]
    assert all(sum(lengths) <= chunk_limit for _, _, _, lengths in chunks)
    assert all(len(lengths) <= transfer_backend.MAX_TRANSFER_CHUNK_WEIGHTS for _, _, _, lengths in chunks)
    assert chunks[1][1] == [10_000 + chunk_limit]
    assert chunks[1][2] == [20_000 + chunk_limit]


def test_read_weights_from_seed_refreshes_registered_shape_after_reshape(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.weight_manifest = {"weight": (tensor.data_ptr(), tensor.numel(), tensor.element_size(), (2, 3), "int64")}
    backend.weight_shapes = {"weight": (2, 3)}

    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        lambda model, processed_layout: [("weight", tensor)],
    )
    seed_info = SeedTransferInfo(
        "seed-session",
        {"weight": [1, tensor.numel(), tensor.element_size(), [1, 2, 3], "int64"]},
        {"weight": [1, 2, 3]},
    )

    assert backend.read_weights_from_seed(object(), seed_info, True)
    assert tuple(tensor.shape) == (1, 2, 3)
    assert backend.weight_shapes["weight"] == (1, 2, 3)
    assert backend.weight_manifest["weight"][3] == (1, 2, 3)
    assert backend.weight_manifest["weight"][4] == "int64"


def test_read_weights_from_seed_retains_registered_transferable_tensor_owners(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    tensor_ref = weakref.ref(tensor)
    tensor_numel = tensor.numel()
    tensor_element_size = tensor.element_size()
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.weight_shapes = {"weight": (2, 3)}
    registered_tensors = [("weight", tensor)]
    backend._registered_transferable_tensors = registered_tensors

    def fail_if_rescanned(model, processed_layout):
        raise AssertionError("read_weights_from_seed should reuse the registered tensor cache")

    monkeypatch.setattr(transfer_backend, "collect_transferable_tensors", fail_if_rescanned)
    seed_info = SeedTransferInfo(
        "seed-session",
        {"weight": [1, tensor_numel, tensor_element_size, [2, 3], "int64"]},
        None,
    )

    assert backend.read_weights_from_seed(object(), seed_info, True)
    assert backend._registered_transferable_tensors is registered_tensors
    del registered_tensors
    del tensor
    gc.collect()
    assert tensor_ref() is not None


def test_read_weights_from_seed_keeps_registered_tensors_when_seed_metadata_is_invalid(monkeypatch):
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace()
    registered_tensors = [("weight", tensor)]
    backend._registered_transferable_tensors = registered_tensors

    seed_info = SeedTransferInfo("", {})

    assert not backend.read_weights_from_seed(object(), seed_info, True)
    assert backend._registered_transferable_tensors is registered_tensors


def test_unregister_memory_region_releases_registered_tensor_owners():
    tensor = torch.arange(6).reshape(2, 3)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_unregister_memory=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.weight_manifest = {"weight": (tensor.data_ptr(), tensor.numel(), 1)}
    backend.weight_shapes = {"weight": tuple(tensor.shape)}
    backend.registered_weight_blocks = [(tensor.data_ptr(), tensor.numel() * tensor.element_size())]
    backend.registered_memory_addresses = [tensor.data_ptr()]
    backend._registered_transferable_tensors = [("weight", tensor)]

    assert backend.unregister_memory_region()
    assert backend._registered_transferable_tensors is None
    assert backend.weight_manifest is None
    assert backend.weight_shapes is None
    assert backend.registered_weight_blocks == []


def test_unregister_memory_region_failure_preserves_state_for_retry():
    tensor = torch.arange(6).reshape(2, 3)
    blocks = [(tensor.data_ptr(), tensor.numel() * tensor.element_size())]
    owners = [("weight", tensor)]
    attempts = []

    class _Ret:
        def __init__(self, error):
            self.error = error

        def is_error(self):
            return self.error

        def to_string(self):
            return "failure"

    def unregister(_addresses):
        attempts.append(_addresses)
        return _Ret(len(attempts) == 1)

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(batch_unregister_memory=unregister)
    backend.weight_manifest = {"weight": (tensor.data_ptr(), 6, 1)}
    backend.weight_shapes = {"weight": (2, 3)}
    backend.registered_weight_blocks = blocks
    backend.registered_memory_addresses = [blocks[0][0]]
    backend._registered_transferable_tensors = owners

    assert not backend.unregister_memory_region()
    assert backend.registered_weight_blocks == blocks
    assert backend._registered_transferable_tensors is owners
    assert backend.weight_manifest is not None
    assert backend.weight_shapes is not None

    assert backend.unregister_memory_region()
    assert attempts == [[blocks[0][0]], [blocks[0][0]]]
    assert backend.registered_weight_blocks == []
    assert backend._registered_transferable_tensors is None


def test_register_memory_region_uses_logical_ranges_with_allocator_backing(monkeypatch):
    tensor = torch.arange(8, dtype=torch.uint8)
    logical = tensor[2:6]
    backing_start = tensor.data_ptr()
    backing_size = tensor.numel() * tensor.element_size()
    registrations: list[tuple[int, int, int, int]] = []

    class _MemoryRegistration:
        def __init__(self, logical_addr, logical_length, backing_addr, backing_length):
            self.values = (logical_addr, logical_length, backing_addr, backing_length)

    class _Ret:
        def is_error(self):
            return False

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda items: _extend_and_return(
            registrations, (item.values for item in items), _Ret()
        )
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    backend.weight_manifest = None
    backend.weight_shapes = None
    backend._registered_transferable_tensors = None

    monkeypatch.setattr(transfer_backend, "find_non_npu_state_tensors", lambda _model: [])
    monkeypatch.setattr(transfer_backend, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(transfer_backend, "collect_transferable_tensors", lambda *args: [("weight", logical)])
    monkeypatch.setattr(
        transfer_backend.torch,
        "npu",
        SimpleNamespace(
            memory=SimpleNamespace(
                memory_snapshot=lambda: [
                    {
                        "blocks": [
                            {
                                "address": backing_start,
                                "size": backing_size,
                                "state": "active_allocated",
                            }
                        ]
                    }
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
    registrations: list[tuple[int, int, int, int]] = []

    class _MemoryRegistration:
        def __init__(self, *values):
            self.values = values

    class _Ret:
        def is_error(self):
            return False

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda items: _extend_and_return(
            registrations, (item.values for item in items), _Ret()
        )
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    backend._registered_transferable_tensors = None
    monkeypatch.setattr(transfer_backend, "find_non_npu_state_tensors", lambda _model: [])
    monkeypatch.setattr(transfer_backend, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        lambda *args: [("first", first), ("second", second)],
    )
    monkeypatch.setattr(
        transfer_backend.torch,
        "npu",
        SimpleNamespace(
            memory=SimpleNamespace(
                memory_snapshot=lambda: [
                    {
                        "blocks": [
                            {
                                "address": backing_start,
                                "size": tensor.numel(),
                                "state": "active_allocated",
                            }
                        ]
                    }
                ]
            )
        ),
        raising=False,
    )

    assert backend.register_memory_region(object(), False)
    assert registrations == [(first.data_ptr(), 6, backing_start, tensor.numel())]


def test_finalize_retries_not_ready_and_clears_state_only_after_success():
    attempts: list[Any] = []

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
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(finalize=lambda: _append_and_return(attempts, True, next(results)))
    backend.transfer_session_id = "session"
    backend.weight_manifest = {"weight": (1, 1, 1)}
    backend.weight_shapes = {"weight": (1,)}
    backend.registered_weight_blocks = [(1, 1)]
    backend.registered_memory_addresses = [1]
    backend._registered_transferable_tensors = [("weight", torch.ones(1))]
    backend._not_ready_error_code = "not-ready"
    backend._is_initialized = True

    assert backend.finalize_transfer_engine()
    assert len(attempts) == 2
    assert backend._is_initialized is False
    assert backend.transfer_session_id is None
    assert backend.registered_weight_blocks == []
    assert backend._registered_transferable_tensors is None


def test_finalize_failure_preserves_registered_tensor_owners():
    owners = [("weight", torch.ones(1))]

    class _Ret:
        def is_error(self):
            return True

        def get_code(self):
            return "fatal"

        def to_string(self):
            return "fatal"

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(finalize=lambda: _Ret())
    backend.transfer_session_id = "session"
    backend.registered_weight_blocks = [(1, 1)]
    backend.registered_memory_addresses = [1]
    backend._registered_transferable_tensors = owners
    backend._not_ready_error_code = "not-ready"
    backend._is_initialized = True

    assert not backend.finalize_transfer_engine()
    assert backend._is_initialized is True
    assert backend.registered_weight_blocks == [(1, 1)]
    assert backend._registered_transferable_tensors is owners


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

    backend._read_weights_from_seed_locked = recv_locked  # type: ignore[method-assign]
    args = (object(), SeedTransferInfo("session", {}), False)
    first = threading.Thread(target=backend.read_weights_from_seed, args=args)

    def recv_second():
        backend.read_weights_from_seed(*args)
        second_done.set()

    second = threading.Thread(target=recv_second)

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


def test_register_memory_region_rejects_second_registration_without_overwriting(monkeypatch):
    tensor = torch.arange(4)
    old_blocks = [(123, 64)]
    old_info = {"old": (123, 4, 4)}
    old_owners = [("old", tensor)]
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace()
    backend.registered_weight_blocks = old_blocks
    backend.registered_memory_addresses = [123]
    backend.weight_manifest = old_info
    backend.weight_shapes = {"old": (4,)}
    backend._registered_transferable_tensors = old_owners

    # A failed unregister keeps its state so the next attempt can retry it.
    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        lambda *args: pytest.fail("register_memory_region must not rescan before stale unregister succeeds"),
    )
    assert not backend.register_memory_region(object(), False)
    assert backend.registered_weight_blocks == old_blocks
    assert backend.weight_manifest is old_info
    assert backend._registered_transferable_tensors is old_owners


def test_register_memory_region_retries_stale_blocks_before_registering(monkeypatch):
    storage = torch.arange(10, dtype=torch.float32)
    stale_blocks = [(storage.data_ptr() + 4096, 256)]
    backend, registrations, unregistered_calls = _make_register_memory_region_backend(
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
    assert registrations == [(storage.data_ptr(), storage.numel() * storage.element_size(), storage.data_ptr(), 40)]
    assert backend.registered_weight_blocks == [(storage.data_ptr(), 40)]


def test_register_memory_region_aborts_when_stale_unregister_fails(monkeypatch):
    storage = torch.arange(10, dtype=torch.float32)
    stale_blocks = [(storage.data_ptr() + 4096, 256)]
    backend, registrations, unregistered_calls = _make_register_memory_region_backend(
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

    assert registrations == []
    assert unregistered_calls == [[stale_blocks[0][0]]]
    assert backend.registered_weight_blocks == stale_blocks


def test_unregister_splits_batches_and_preserves_only_unfinished_addresses(monkeypatch):
    calls = []

    def batch_unregister_memory(addresses):
        calls.append(list(addresses))
        return SimpleNamespace(
            is_error=lambda: len(calls) == 2,
            to_string=lambda: "mock unregister error",
        )

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(batch_unregister_memory=batch_unregister_memory)
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
    backend.transfer_engine = SimpleNamespace(finalize=finalize)
    backend._not_ready_error_code = "not-ready"
    backend._is_initialized = True
    monkeypatch.setattr(transfer_backend.time, "sleep", sleep_calls.append)

    assert not backend.finalize_transfer_engine(max_attempts=3, retry_interval_sec=0.25)
    assert len(attempts) == 3
    assert sleep_calls == [0.25, 0.25]


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


def _make_register_memory_region_backend(
    monkeypatch,
    tensors,
    snapshot_blocks,
    stale_blocks=(),
    unregister_error=False,
):
    registrations: list[tuple[int, int, int, int]] = []
    unregistered_calls = []
    monkeypatch.setattr(
        transfer_backend, "is_transferable_tensor", lambda t: isinstance(t, torch.Tensor) and t.numel() > 0
    )

    class _MemoryRegistration:
        def __init__(self, *values):
            self.values = values

    class _Ret:
        def is_error(self):
            return False

    def batch_register_memory_ex(items):
        registrations.extend(item.values for item in items)
        return _Ret()

    def batch_unregister_memory(addresses):
        unregistered_calls.append(list(addresses))
        return SimpleNamespace(is_error=lambda: unregister_error, to_string=lambda: "mock unregister error")

    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=batch_register_memory_ex,
        batch_unregister_memory=batch_unregister_memory,
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = list(stale_blocks)
    backend.registered_memory_addresses = [address for address, _ in stale_blocks]
    backend.weight_manifest = None
    backend.weight_shapes = None
    backend._registered_transferable_tensors = None
    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        lambda model, processed_layout: list(tensors),
    )
    snapshot = [{"blocks": snapshot_blocks}]
    monkeypatch.setattr(
        transfer_backend.torch,
        "npu",
        SimpleNamespace(memory=SimpleNamespace(memory_snapshot=lambda: snapshot)),
        raising=False,
    )
    return backend, registrations, unregistered_calls


def test_register_memory_region_skips_shared_weights(monkeypatch):
    storage = torch.arange(100, dtype=torch.float32)
    shared_weight = storage[:10]
    own_weight = storage[20:32].reshape(2, 6)
    backend, registrations, _ = _make_register_memory_region_backend(
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

    assert set(backend.weight_manifest) == {"layers.0.fc.weight"}
    assert [name for name, _ in backend._registered_transferable_tensors] == ["layers.0.fc.weight"]
    assert backend.excluded_weight_blocks == excluded_blocks
    assert registrations == [
        (
            own_weight.data_ptr(),
            own_weight.numel() * own_weight.element_size(),
            storage.data_ptr() + 40,
            360,
        )
    ]


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
    assert len(registered_calls) == 3
    assert backend.registered_memory_addresses == [weight.data_ptr() for weight in weights]


def test_register_memory_region_skips_empty_batch_after_excluding_all_weights(monkeypatch):
    storage = torch.arange(10, dtype=torch.float32)
    backend, registrations, _ = _make_register_memory_region_backend(
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

    assert backend.weight_manifest == {}
    assert backend.weight_shapes == {}
    assert backend._registered_transferable_tensors == []
    assert backend.registered_weight_blocks == []
    assert registrations == []


def test_read_weights_from_seed_defensively_skips_pre_registered_weights(monkeypatch):
    storage = torch.arange(100, dtype=torch.float32)
    shared_weight = storage[:10]
    own_weight = storage[20:32]
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.weight_shapes = {}
    backend.excluded_weight_blocks = [(storage.data_ptr(), 40)]
    backend._registered_transferable_tensors = None

    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        lambda model, processed_layout: [
            ("model.embed_tokens.weight", shared_weight),
            ("layers.0.fc.weight", own_weight),
        ],
    )

    seed_info = SeedTransferInfo(
        "seed-session",
        {
            "layers.0.fc.weight": [
                7,
                own_weight.numel(),
                own_weight.element_size(),
                list(own_weight.shape),
                "float32",
            ]
        },
        None,
    )

    assert backend.read_weights_from_seed(object(), seed_info, True)
    assert backend.weight_shapes == {"layers.0.fc.weight": tuple(own_weight.shape)}


def test_read_weights_from_seed_fails_for_unknown_weight_outside_shared_blocks(monkeypatch):
    own_weight = torch.arange(12, dtype=torch.float32)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_transfer_sync_read=lambda *args: SimpleNamespace(is_error=lambda: False)
    )
    backend.weight_shapes = {}
    backend.excluded_weight_blocks = [(4096, 128)]
    backend._registered_transferable_tensors = [("layers.0.fc.weight", own_weight)]

    seed_info = SeedTransferInfo("seed-session", {}, None)

    assert not backend.read_weights_from_seed(object(), seed_info, True)


@pytest.mark.parametrize(
    "entry",
    [
        [1, 6, 4],
        [1, 6, 4, [6]],
        [1, 6, 4, [6], "int32"],
        [1, 6, 4, [7], "float32"],
    ],
)
def test_incompatible_manifest_rejected_before_native_read(monkeypatch, entry):
    tensor = torch.ones(6, dtype=torch.float32)
    backend = RForkTransferBackend.__new__(RForkTransferBackend)
    calls = []
    backend.transfer_engine = SimpleNamespace(batch_transfer_sync_read=lambda *args: calls.append(args))
    backend._registered_transferable_tensors = [("weight", tensor)]
    seed_info = SeedTransferInfo("session", {"weight": entry}, None)

    assert not backend.read_weights_from_seed(object(), seed_info, True)
    assert calls == []
