# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that the E2E probe rejects failed or silently skipped Backend copies on CPU."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.e2e.common.kv_pool.ascendstore_v1_probe import (
    _snapshot_buffers,
    clear_worker_local_kv,
    collect_worker_io_probe,
    install_worker_io_probe,
)


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    cache = torch.arange(1, 33, dtype=torch.float32).reshape(2, 16)
    raw_bytes = cache.view(torch.uint8).flatten()
    objects: dict[str, torch.Tensor] = {}
    backend = SimpleNamespace(mode="copy")
    bindings = tuple(
        SimpleNamespace(
            remote_object=SimpleNamespace(
                key=f"key-{index}",
                chunk=SimpleNamespace(token_range=SimpleNamespace(start_token=index * 4, end_token=(index + 1) * 4)),
            ),
            local_region=SimpleNamespace(memory=SimpleNamespace(addresses=(cache[index].data_ptr(),), sizes=(64,))),
        )
        for index in range(2)
    )

    def store(batches):
        evidence = []
        for batch in batches:
            for binding in batch.bindings:
                memory = binding.local_region.memory
                offset = memory.addresses[0] - cache.data_ptr()
                objects[binding.remote_object.key] = raw_bytes[offset : offset + memory.sizes[0]].clone()
                evidence.append(SimpleNamespace(result_code=0))
        return SimpleNamespace(succeeded=True, source_release_confirmed=True, binding_evidence=evidence)

    def get(keys, addresses, sizes):
        if backend.mode == "fail":
            return [-1] * len(keys)
        if backend.mode == "short":
            return [0]
        if backend.mode == "missing":
            return None
        if backend.mode in ("copy", "partial"):
            for index, (key, buffers, lengths) in enumerate(zip(keys, addresses, sizes, strict=True)):
                if index and backend.mode == "partial":
                    break
                offset = buffers[0] - cache.data_ptr()
                raw_bytes[offset : offset + lengths[0]].copy_(objects[key])
        return [0] * len(keys)

    def load(selected_bindings):
        if not selected_bindings:
            return ()
        keys = [binding.remote_object.key for binding in selected_bindings]
        addresses = [list(binding.local_region.memory.addresses) for binding in selected_bindings]
        sizes = [list(binding.local_region.memory.sizes) for binding in selected_bindings]
        return tuple(SimpleNamespace(result_code=code) for code in backend.get(keys, addresses, sizes))

    backend.get = get
    runtime = SimpleNamespace(
        _resources=SimpleNamespace(kv_caches={"layer": cache}, backend=backend),
        _backend_io=SimpleNamespace(store=store, load=load),
        _program=SimpleNamespace(topology=SimpleNamespace(cache_transfer_granularity=4)),
        fence_previous_store=Mock(),
    )
    connector_type = type(
        "AscendStoreV1Connector",
        (),
        {"__module__": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector"},
    )
    connector = connector_type()
    connector.runtime = runtime
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.kv_transfer", SimpleNamespace(get_kv_transfer_group=lambda: connector)
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=Mock()), raising=False)
    return SimpleNamespace(runtime=runtime, bindings=bindings, cache=cache, connector=connector)


def test_probe_verifies_real_copy_after_erasing_source(worker: SimpleNamespace) -> None:
    assert install_worker_io_probe(worker) == 4
    expected = worker.cache.clone()
    worker.runtime._backend_io.store((SimpleNamespace(bindings=worker.bindings),))
    cold_evidence = clear_worker_local_kv(worker)
    worker.runtime.fence_previous_store.assert_called_once()
    assert torch.count_nonzero(worker.cache) == 0
    assert cold_evidence["get_calls"] == 0
    assert cold_evidence["local_kv_cleared"]

    worker.runtime._backend_io.load(worker.bindings)
    assert torch.equal(worker.cache, expected)
    warm_evidence = collect_worker_io_probe(worker)
    assert warm_evidence["get_calls"] == 1
    assert warm_evidence["loaded_keys"] == cold_evidence["stored_keys"]
    assert warm_evidence["loaded_ranges"] == ((0, 4), (4, 8))
    assert warm_evidence["loaded_bytes"] == cold_evidence["stored_bytes"] == 128


@pytest.mark.parametrize("mode", ["noop", "partial", "fail", "short", "missing"])
def test_probe_rejects_false_get_success(worker: SimpleNamespace, mode: str) -> None:
    install_worker_io_probe(worker)
    worker.runtime._backend_io.store((SimpleNamespace(bindings=worker.bindings),))
    clear_worker_local_kv(worker)
    worker.runtime._resources.backend.mode = mode
    with pytest.raises(AssertionError):
        worker.runtime._backend_io.load(worker.bindings)
    assert collect_worker_io_probe(worker)["get_calls"] == 0


def test_probe_does_not_count_an_empty_load_as_get(worker: SimpleNamespace) -> None:
    install_worker_io_probe(worker)
    assert worker.runtime._backend_io.load(()) == ()
    assert collect_worker_io_probe(worker)["get_calls"] == 0


def test_probe_rejects_a_different_connector(worker: SimpleNamespace) -> None:
    worker.connector.__class__.__module__ = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"
    with pytest.raises(AssertionError):
        install_worker_io_probe(worker)


def test_snapshot_uses_storage_offsets_and_rejects_out_of_bounds() -> None:
    storage = torch.arange(16, dtype=torch.int32)
    cache = storage[3::2]
    snapshot = _snapshot_buffers((cache,), (storage.data_ptr() + 4,), (8,))[0]
    assert torch.equal(snapshot, storage.view(torch.uint8)[4:12])
    storage.zero_()
    assert torch.count_nonzero(snapshot) > 0
    with pytest.raises(AssertionError, match="outside registered KV storage"):
        _snapshot_buffers((cache,), (storage.data_ptr() + 60,), (8,))
