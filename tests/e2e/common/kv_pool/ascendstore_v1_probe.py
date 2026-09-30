# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-local observations for the AscendStore v1 transfer smoke test."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def install_worker_io_probe(worker: Any) -> int:
    """Wrap real Backend calls without changing their inputs or results."""
    # Import in the Worker, not the pytest process that submits this callable.
    from vllm.distributed.kv_transfer import get_kv_transfer_group

    connector = get_kv_transfer_group()
    assert type(connector).__module__ == "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector"
    assert type(connector).__name__ == "AscendStoreV1Connector"
    runtime = connector.runtime
    assert runtime is not None
    kv_caches = runtime._resources.kv_caches
    assert kv_caches, "The Worker must register KV buffers before installing the probe"
    caches = tuple(
        cache
        for cache_or_caches in kv_caches.values()
        for cache in ((cache_or_caches,) if isinstance(cache_or_caches, torch.Tensor) else cache_or_caches)
    )
    probe: dict[str, Any] = {
        "runtime": runtime,
        "caches": caches,
        "stored_buffers": {},
        "get_calls": 0,
        "loaded_keys": [],
        "loaded_ranges": [],
        "loaded_bytes": 0,
        "local_kv_cleared": False,
    }
    backend_io = runtime._backend_io
    original_store, original_load = backend_io.store, backend_io.load
    original_get = runtime._resources.backend.get

    def record_store(batches: Any) -> Any:
        bindings = tuple(binding for batch in batches for binding in batch.bindings)
        snapshots = {
            binding.remote_object.key: _snapshot_buffers(
                caches, binding.local_region.memory.addresses, binding.local_region.memory.sizes
            )
            for binding in bindings
        }
        # Store reaches this boundary only after its SourceReady event has been synchronized.
        evidence = original_store(batches)
        assert evidence.succeeded and evidence.source_release_confirmed, "The real Backend Store did not succeed"
        assert len(evidence.binding_evidence) == len(bindings)
        assert all(item.result_code == 0 for item in evidence.binding_evidence)
        probe["stored_buffers"].update(snapshots)
        return evidence

    def record_get(keys: list[str], addresses: list[list[int]], sizes: list[list[int]]) -> Any:
        assert probe["local_kv_cleared"], "GET must occur after erasing local KV, not during the cold request"
        result = original_get(keys, addresses, sizes)
        assert result is not None and len(result) == len(keys), "GET returned missing or unaligned results"
        assert all(code == 0 for code in result), f"The real Backend GET failed: {result}"
        for key, key_addresses, key_sizes in zip(keys, addresses, sizes, strict=True):
            assert key in probe["stored_buffers"], f"GET used a key not written by the cold request: {key}"
            actual = _snapshot_buffers(caches, key_addresses, key_sizes)
            expected = probe["stored_buffers"][key]
            assert len(actual) == len(expected), f"GET changed the buffer count for {key}"
            for actual_buffer, expected_buffer in zip(actual, expected, strict=True):
                assert torch.equal(actual_buffer, expected_buffer), f"GET did not restore the stored bytes for {key}"
            probe["loaded_bytes"] += sum(key_sizes)
        probe["get_calls"] += 1
        probe["loaded_keys"].extend(keys)
        return result

    def record_load(bindings: Any) -> Any:
        evidence = original_load(bindings)
        assert len(evidence) == len(bindings) and all(item.result_code == 0 for item in evidence)
        probe["loaded_ranges"].extend(
            (binding.remote_object.chunk.token_range.start_token, binding.remote_object.chunk.token_range.end_token)
            for binding in bindings
        )
        return evidence

    runtime._resources.backend.get = record_get
    backend_io.store, backend_io.load = record_store, record_load
    worker._ascendstore_v1_probe = probe
    return runtime._program.topology.cache_transfer_granularity


def clear_worker_local_kv(worker: Any) -> dict[str, Any]:
    """Drain Store before destroying the local bytes; leave external objects intact."""
    probe = worker._ascendstore_v1_probe
    probe["runtime"].fence_previous_store()
    assert probe["stored_buffers"], "The cold request did not Store any objects"
    assert probe["get_calls"] == 0, "The cold request unexpectedly loaded remote KV"
    assert any(buffer.any() for buffers in probe["stored_buffers"].values() for buffer in buffers), (
        "Stored KV must contain nonzero bytes so a no-op GET cannot pass"
    )
    for cache in probe["caches"]:
        cache.zero_()
    torch.npu.synchronize()
    probe["local_kv_cleared"] = True
    return collect_worker_io_probe(worker)


def collect_worker_io_probe(worker: Any) -> dict[str, Any]:
    """Return transport-stable transfer evidence, not the retained source snapshots."""
    probe = worker._ascendstore_v1_probe
    return {
        "stored_keys": tuple(probe["stored_buffers"]),
        "stored_bytes": sum(buffer.numel() for buffers in probe["stored_buffers"].values() for buffer in buffers),
        "get_calls": probe["get_calls"],
        "loaded_keys": tuple(probe["loaded_keys"]),
        # Untyped nested tuples become lists across the EngineCore utility response codec.
        "loaded_ranges": [list(token_range) for token_range in probe["loaded_ranges"]],
        "loaded_bytes": probe["loaded_bytes"],
        "local_kv_cleared": probe["local_kv_cleared"],
    }


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
