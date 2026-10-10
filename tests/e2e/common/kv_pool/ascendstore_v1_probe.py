# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-local observations for AscendStore production/v1 equivalence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

_PRODUCTION_CONNECTOR = (
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
    "AscendStoreConnector",
)
_V1_CONNECTOR = (
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector",
    "AscendStoreV1Connector",
)


def install_worker_io_probe(worker: Any) -> int:
    """Observe the common whole-object Backend contract without replacing it."""
    # Import in the Worker, not the pytest process that submits this callable.
    from vllm.distributed.kv_transfer import get_kv_transfer_group

    connector = get_kv_transfer_group()
    connector_kind = (type(connector).__module__, type(connector).__name__)
    if connector_kind == _PRODUCTION_CONNECTOR:
        kv_pool_worker = connector.connector_worker
        assert kv_pool_worker is not None
        backend = kv_pool_worker.m_store
        kv_caches = kv_pool_worker.kv_caches
        granularity = kv_pool_worker.cache_transfer_granularity
        wait_for_store: Callable[[], Any] = kv_pool_worker.wait_for_previous_save
        store_method_name = "put"
        load_method_name = "get"
    elif connector_kind == _V1_CONNECTOR:
        kv_pool_worker = connector.worker
        assert kv_pool_worker is not None
        backend = kv_pool_worker._resources.backend
        kv_caches = kv_pool_worker._resources.kv_caches
        granularity = kv_pool_worker._topology.cache_transfer_granularity
        wait_for_store = kv_pool_worker.fence_previous_store
        store_method_name = "store"
        load_method_name = "load"
    else:
        raise AssertionError(f"Unsupported connector for AscendStore equivalence probe: {connector_kind!r}")

    assert kv_caches, "The Worker must register KV buffers before installing the probe"
    caches = tuple(
        cache
        for cache_or_caches in kv_caches.values()
        for cache in ((cache_or_caches,) if isinstance(cache_or_caches, torch.Tensor) else cache_or_caches)
    )
    probe: dict[str, Any] = {
        "caches": caches,
        "wait_for_store": wait_for_store,
        "stored_buffers": {},
        "lookup_calls": 0,
        "lookup_keys": [],
        "get_calls": 0,
        "loaded_keys": [],
        "loaded_bytes": 0,
        "local_kv_cleared": False,
        "load_started": False,
    }
    original_exists = backend.exists
    original_store = getattr(backend, store_method_name)
    original_load = getattr(backend, load_method_name)

    def record_exists(keys: list[str]) -> Any:
        result = original_exists(keys)
        if probe["local_kv_cleared"] and not probe["load_started"]:
            values = tuple(result)
            assert len(values) == len(keys) and all(bool(value) for value in values), (
                "Lookup did not observe every object stored by the cold request"
            )
            probe["lookup_calls"] += 1
            probe["lookup_keys"].extend(keys)
        return result

    def record_store(
        keys: list[str],
        addresses: list[list[int]],
        sizes: list[list[int]],
    ) -> Any:
        snapshots = {
            key: _snapshot_buffers(caches, key_addresses, key_sizes)
            for key, key_addresses, key_sizes in zip(keys, addresses, sizes, strict=True)
        }
        # Both connectors reach their Backend only after the Store source is ready.
        result = original_store(keys, addresses, sizes)
        if result is not None:
            _assert_successful_results("Store", keys, result)
        probe["stored_buffers"].update(snapshots)
        return result

    def record_load(
        keys: list[str],
        addresses: list[list[int]],
        sizes: list[list[int]],
    ) -> Any:
        if not keys:
            return original_load(keys, addresses, sizes)
        assert probe["local_kv_cleared"], "Load must occur after erasing local KV, not during the cold request"
        probe["load_started"] = True
        result = original_load(keys, addresses, sizes)
        assert result is not None, "The real Backend Load returned no per-object evidence"
        _assert_successful_results("Load", keys, result)
        for key, key_addresses, key_sizes in zip(keys, addresses, sizes, strict=True):
            assert key in probe["stored_buffers"], f"Load used a key not written by the cold request: {key}"
            actual = _snapshot_buffers(caches, key_addresses, key_sizes)
            expected = probe["stored_buffers"][key]
            assert len(actual) == len(expected), f"Load changed the buffer count for {key}"
            for actual_buffer, expected_buffer in zip(actual, expected, strict=True):
                assert torch.equal(actual_buffer, expected_buffer), f"Load did not restore the stored bytes for {key}"
            probe["loaded_bytes"] += sum(key_sizes)
        probe["get_calls"] += 1
        probe["loaded_keys"].extend(keys)
        return result

    backend.exists = record_exists
    setattr(backend, store_method_name, record_store)
    setattr(backend, load_method_name, record_load)
    worker._ascendstore_equivalence_probe = probe
    return granularity


def clear_worker_local_kv(worker: Any) -> dict[str, Any]:
    """Drain Store, erase local bytes, and leave external objects intact."""
    probe = worker._ascendstore_equivalence_probe
    probe["wait_for_store"]()
    assert probe["stored_buffers"], "The cold request did not Store any objects"
    assert probe["get_calls"] == 0, "The cold request unexpectedly loaded remote KV"
    assert any(buffer.any() for buffers in probe["stored_buffers"].values() for buffer in buffers), (
        "Stored KV must contain nonzero bytes so a no-op Load cannot pass"
    )
    for cache in probe["caches"]:
        cache.zero_()
    torch.npu.synchronize()
    probe["local_kv_cleared"] = True
    return collect_worker_io_probe(worker)


def collect_worker_io_probe(worker: Any) -> dict[str, Any]:
    """Return transport-stable observations, excluding retained source bytes."""
    probe = worker._ascendstore_equivalence_probe
    return {
        "stored_keys": tuple(probe["stored_buffers"]),
        "stored_bytes": sum(buffer.numel() for buffers in probe["stored_buffers"].values() for buffer in buffers),
        "lookup_calls": probe["lookup_calls"],
        "lookup_keys": tuple(probe["lookup_keys"]),
        "get_calls": probe["get_calls"],
        "loaded_keys": tuple(probe["loaded_keys"]),
        "loaded_bytes": probe["loaded_bytes"],
        "local_kv_cleared": probe["local_kv_cleared"],
    }


def _assert_successful_results(operation: str, keys: list[str], results: Sequence[int]) -> None:
    values = tuple(int(value) for value in results)
    assert len(values) == len(keys), f"{operation} returned {len(values)} results for {len(keys)} objects"
    assert all(value == 0 for value in values), f"{operation} returned failed evidence: {values}"


def _snapshot_buffers(
    caches: tuple[torch.Tensor, ...], addresses: Sequence[int], sizes: Sequence[int]
) -> tuple[torch.Tensor, ...]:
    snapshots = []
    for address, size in zip(addresses, sizes, strict=True):
        assert size > 0, "A Backend buffer must select a positive byte count"
        for cache in caches:
            storage = cache.untyped_storage()
            offset = address - storage.data_ptr()
            if offset >= 0 and offset + size <= storage.nbytes():
                # Byte offsets refer to storage, not the possibly strided logical Tensor view.
                view = torch.empty(0, dtype=torch.uint8, device=cache.device).set_(storage, offset, (size,), (1,))
                snapshots.append(view.to("cpu", copy=True))
                break
        else:
            raise AssertionError(f"Backend buffer [{address}, {address + size}) is outside registered KV storage")
    return tuple(snapshots)
