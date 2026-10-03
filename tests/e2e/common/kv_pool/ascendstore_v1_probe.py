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
        "load_ranges_by_batch": {},
        "get_calls": 0,
        "loaded_keys": [],
        "loaded_ranges": [],
        "loaded_bytes": 0,
        "local_kv_cleared": False,
    }
    backend_io = runtime._backend_io
    original_store_batch = backend_io.store_batch
    original_load_batch = backend_io.load_batch
    original_build_load_batch = runtime._build_load_batch

    def record_store_batch(batch: Any, layer_id: int | None = None) -> Any:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.backend.arguments import (
            materialize_rule_ranges,
        )

        arguments = materialize_rule_ranges(runtime._bound_rules, batch, layer_id=layer_id, store=True)
        snapshots = {
            key: _snapshot_buffers(caches, addresses, sizes)
            for key, addresses, sizes in zip(
                arguments.keys,
                arguments.addresses,
                arguments.sizes,
                strict=True,
            )
        }
        # Store reaches this boundary only after its SourceReady event has been synchronized.
        completions = original_store_batch(batch, layer_id)
        evidence = tuple(item for completion in completions for item in completion.evidence.transfer_evidence)
        assert all(
            completion.evidence.succeeded and completion.evidence.source_release_confirmed for completion in completions
        ), "The real Backend Store did not succeed"
        assert len(evidence) == len(arguments.sources)
        assert all(item.result_code == 0 for item in evidence)
        probe["stored_buffers"].update(snapshots)
        return completions

    def record_load_batch(batch: Any, layer_id: int | None = None) -> Any:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.backend.arguments import (
            materialize_rule_ranges,
        )

        arguments = materialize_rule_ranges(runtime._bound_rules, batch, layer_id=layer_id, store=False)
        if not arguments.sources:
            return original_load_batch(batch, layer_id)
        assert probe["local_kv_cleared"], "GET must occur after erasing local KV, not during the cold request"
        completions = original_load_batch(batch, layer_id)
        evidence = tuple(item for completion in completions for item in completion.transfer_evidence)
        assert len(evidence) == len(arguments.sources) and all(item.result_code == 0 for item in evidence), (
            "The real Backend GET returned missing, unaligned, or failed evidence"
        )
        for key, key_addresses, key_sizes in zip(
            arguments.keys,
            arguments.addresses,
            arguments.sizes,
            strict=True,
        ):
            assert key in probe["stored_buffers"], f"GET used a key not written by the cold request: {key}"
            actual = _snapshot_buffers(caches, key_addresses, key_sizes)
            expected = probe["stored_buffers"][key]
            assert len(actual) == len(expected), f"GET changed the buffer count for {key}"
            for actual_buffer, expected_buffer in zip(actual, expected, strict=True):
                assert torch.equal(actual_buffer, expected_buffer), f"GET did not restore the stored bytes for {key}"
            probe["loaded_bytes"] += sum(key_sizes)
        probe["get_calls"] += 1
        probe["loaded_keys"].extend(arguments.keys)
        probe["loaded_ranges"].extend(probe["load_ranges_by_batch"].pop(id(batch), ()))
        return completions

    def record_build_load_batch(commands: Any) -> Any:
        batch = original_build_load_batch(commands)
        granularity = runtime._spec.topology.cache_transfer_granularity
        probe["load_ranges_by_batch"][id(batch)] = tuple(
            (start, min(start + granularity, command.load_range.end_token))
            for command in commands
            for start in range(command.load_range.start_token, command.load_range.end_token, granularity)
        )
        return batch

    backend_io.store_batch = record_store_batch
    backend_io.load_batch = record_load_batch
    runtime._build_load_batch = record_build_load_batch
    worker._ascendstore_v1_probe = probe
    return runtime._spec.topology.cache_transfer_granularity


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
