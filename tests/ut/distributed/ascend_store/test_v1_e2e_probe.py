# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the production/v1 equivalence probe on CPU."""

import sys
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.engine import EngineCoreOutputs, UtilityOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

from tests.e2e.common.kv_pool.ascendstore_v1_probe import (
    _snapshot_buffers,
    clear_worker_local_kv,
    collect_worker_io_probe,
    install_worker_io_probe,
)

_PRODUCTION_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector"
_V1_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector"


@pytest.fixture
def worker_factory(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], SimpleNamespace]:
    def make_worker(implementation: str) -> SimpleNamespace:
        cache = torch.arange(1, 33, dtype=torch.float32).reshape(2, 16)
        raw_bytes = cache.view(torch.uint8).flatten()
        objects: dict[str, torch.Tensor] = {}
        backend = SimpleNamespace(mode="copy")
        keys = ["key-0", "key-1"]
        addresses = [[cache.data_ptr()], [cache.data_ptr() + 64]]
        sizes = [[64], [64]]

        def store(keys_to_store, address_rows, size_rows):
            for key, key_addresses, key_sizes in zip(keys_to_store, address_rows, size_rows, strict=True):
                offset = key_addresses[0] - cache.data_ptr()
                objects[key] = raw_bytes[offset : offset + key_sizes[0]].clone()
            return None if implementation == "production" else [0] * len(keys_to_store)

        def load(keys_to_load, address_rows, size_rows):
            if backend.mode == "fail":
                return [-1] * len(keys_to_load)
            if backend.mode == "short":
                return [0]
            if backend.mode == "missing":
                return None
            if backend.mode in ("copy", "partial"):
                for index, (key, key_addresses, key_sizes) in enumerate(
                    zip(keys_to_load, address_rows, size_rows, strict=True)
                ):
                    if index and backend.mode == "partial":
                        break
                    offset = key_addresses[0] - cache.data_ptr()
                    raw_bytes[offset : offset + key_sizes[0]].copy_(objects[key])
            return [0] * len(keys_to_load)

        backend.exists = lambda lookup_keys: [int(key in objects) for key in lookup_keys]
        backend.put = store
        backend.get = load
        backend.store = store
        backend.load = load
        if implementation == "production":
            connector_type = type("AscendStoreConnector", (), {"__module__": _PRODUCTION_MODULE})
            connector = connector_type()
            connector.connector_worker = SimpleNamespace(
                m_store=backend,
                kv_caches={"layer": cache},
                cache_transfer_granularity=4,
                wait_for_previous_save=Mock(),
            )
            wait_for_store = connector.connector_worker.wait_for_previous_save
            store_method = "put"
            load_method = "get"
        elif implementation == "v1":
            connector_type = type("AscendStoreV1Connector", (), {"__module__": _V1_MODULE})
            connector = connector_type()
            connector.worker = SimpleNamespace(
                _resources=SimpleNamespace(kv_caches={"layer": cache}, backend=backend),
                _topology=SimpleNamespace(cache_transfer_granularity=4),
                fence_previous_store=Mock(),
            )
            wait_for_store = connector.worker.fence_previous_store
            store_method = "store"
            load_method = "load"
        else:
            raise AssertionError(f"Unknown implementation: {implementation}")

        monkeypatch.setitem(
            sys.modules,
            "vllm.distributed.kv_transfer",
            SimpleNamespace(get_kv_transfer_group=lambda: connector),
        )
        monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=Mock()), raising=False)
        return SimpleNamespace(
            backend=backend,
            cache=cache,
            connector=connector,
            keys=keys,
            addresses=addresses,
            sizes=sizes,
            wait_for_store=wait_for_store,
            store_method=store_method,
            load_method=load_method,
        )

    return make_worker


def test_probe_proves_the_common_store_lookup_load_contract(
    worker_factory: Callable[[str], SimpleNamespace], monkeypatch: pytest.MonkeyPatch
) -> None:
    summaries = []
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    for implementation in ("production", "v1"):
        worker = worker_factory(implementation)
        assert install_worker_io_probe(worker) == 4
        expected = worker.cache.clone()
        getattr(worker.backend, worker.store_method)(worker.keys, worker.addresses, worker.sizes)
        cold_evidence = clear_worker_local_kv(worker)
        worker.wait_for_store.assert_called_once()
        assert torch.count_nonzero(worker.cache) == 0
        assert cold_evidence["get_calls"] == cold_evidence["lookup_calls"] == 0
        assert cold_evidence["local_kv_cleared"]

        assert all(worker.backend.exists(worker.keys))
        getattr(worker.backend, worker.load_method)(worker.keys, worker.addresses, worker.sizes)
        assert torch.equal(worker.cache, expected)
        warm_evidence = collect_worker_io_probe(worker)
        assert warm_evidence == {
            "stored_keys": tuple(worker.keys),
            "stored_bytes": 128,
            "lookup_calls": 1,
            "lookup_keys": tuple(worker.keys),
            "get_calls": 1,
            "loaded_keys": tuple(worker.keys),
            "loaded_bytes": 128,
            "local_kv_cleared": True,
        }
        assert getattr(worker.backend, worker.load_method)([], [], []) == []
        assert collect_worker_io_probe(worker)["get_calls"] == 1

        outputs = EngineCoreOutputs(utility_output=UtilityOutput(call_id=1, result=UtilityResult([warm_evidence])))
        frames = MsgpackEncoder().encode(outputs)
        decoded = MsgpackDecoder(EngineCoreOutputs).decode(frames).utility_output
        assert decoded is not None and decoded.result is not None
        assert decoded.result.result == [warm_evidence]
        summaries.append(warm_evidence)

    assert summaries[0] == summaries[1]


def test_probe_rejects_failed_malformed_or_silently_skipped_loads(
    worker_factory: Callable[[str], SimpleNamespace],
) -> None:
    worker = worker_factory("v1")
    install_worker_io_probe(worker)
    worker.backend.store(worker.keys, worker.addresses, worker.sizes)
    clear_worker_local_kv(worker)
    assert all(worker.backend.exists(worker.keys))
    for mode in ("noop", "partial", "fail", "short", "missing"):
        worker.cache.zero_()
        worker.backend.mode = mode
        with pytest.raises(AssertionError):
            worker.backend.load(worker.keys, worker.addresses, worker.sizes)
    assert collect_worker_io_probe(worker)["get_calls"] == 0


def test_probe_rejects_an_unsupported_connector(worker_factory: Callable[[str], SimpleNamespace]) -> None:
    unsupported = worker_factory("v1")
    unsupported.connector.__class__.__module__ = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"
    with pytest.raises(AssertionError, match="Unsupported connector"):
        install_worker_io_probe(unsupported)


def test_snapshot_uses_storage_offsets_and_rejects_out_of_bounds() -> None:
    storage = torch.arange(16, dtype=torch.int32)
    cache = storage[3::2]
    snapshot = _snapshot_buffers((cache,), (storage.data_ptr() + 4,), (8,))[0]
    assert torch.equal(snapshot, storage.view(torch.uint8)[4:12])
    storage.zero_()
    assert torch.count_nonzero(snapshot) > 0
    with pytest.raises(AssertionError, match="outside registered KV storage"):
        _snapshot_buffers((cache,), (storage.data_ptr() + 60,), (8,))
