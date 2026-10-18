# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded byte checks outside Layerwise callbacks and outside performance runs."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

MAX_SNAPSHOT_BYTES = 128 << 20


class KVByteAccuracy:
    """Record transfer coordinates only; snapshot after the request and Store fence.

    The accuracy workload has one request and one output token. No subsequent
    request or local-cache reset may reuse its buffers before ``finish``.
    """

    def __init__(self, backend: Any, caches: dict, production: bool, layerwise: bool, directory: str) -> None:
        self.backend = backend
        self.caches = tuple(
            tensor for value in caches.values() for tensor in ((value,) if isinstance(value, torch.Tensor) else value)
        )
        self.directory = Path(directory)
        self.records: dict[str, list] = {"store": [], "load": []}
        self.failures: list[str] = []
        self.originals: dict[str, Any] = {}
        methods = ("batch_copy_put", "batch_copy_get") if layerwise else ("put", "get")
        if not production:
            methods = ("copy_key_range_store", "copy_key_range_load") if layerwise else ("store", "load")
        for operation, name in zip(("store", "load"), methods, strict=True):
            original = getattr(backend, name)
            self.originals[name] = original
            setattr(backend, name, self._observe(operation, original, production and not layerwise))

    def _observe(self, operation: str, original: Any, production_bulk: bool) -> Any:
        def transfer(keys, addresses, sizes, offsets=None):
            try:
                result = (
                    original(keys, addresses, sizes) if offsets is None else original(keys, addresses, sizes, offsets)
                )
            except BaseException as error:
                self.failures.append(f"{operation}: {type(error).__name__}: {error}")
                raise
            # Retain errors even when a Worker catches them; fail at the phase boundary,
            # without raising new exceptions inside a production background thread.
            valid = isinstance(result, (tuple, list)) and len(result) == len(keys)
            if valid:
                valid = (
                    all(value is True for value in result)
                    if production_bulk and operation == "store"
                    else all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in result)
                )
            if not valid:
                self.failures.append(f"{operation} returned invalid or failed results: {result!r}")
            for index, (key, row_addresses, row_sizes) in enumerate(zip(keys, addresses, sizes, strict=True)):
                position = 0
                for column, (address, size) in enumerate(zip(row_addresses, row_sizes, strict=True)):
                    offset = position if offsets is None else offsets[index][column]
                    self.records[operation].append((key, address, size, offset))
                    position += size
            return result

        return transfer

    def finish(self, operation: str) -> dict[str, Any]:
        """One batched device-to-host snapshot, then exact comparison on the host."""
        assert not self.failures, self.failures
        records = self.records[operation]
        assert records, f"No {operation} byte ranges observed"
        total_bytes = sum(size for _, _, size, _ in records)
        assert 0 < total_bytes <= MAX_SNAPSHOT_BYTES, "Accuracy snapshot exceeded its bounded workload"
        views = []
        for _, address, size, offset in records:
            assert size > 0 and offset >= 0
            for cache in self.caches:
                storage = cache.untyped_storage()
                local_offset = address - storage.data_ptr()
                if local_offset >= 0 and local_offset + size <= storage.nbytes():
                    views.append(
                        torch.empty(0, dtype=torch.uint8, device=cache.device).set_(
                            storage, local_offset, (size,), (1,)
                        )
                    )
                    break
            else:
                raise AssertionError("Transfer range lies outside registered KV storage")
        # No CPU transfer, synchronization or equality check runs in a layer callback.
        if views[0].device.type != "cpu":
            torch.npu.synchronize()
        snapshot = torch.cat(views).cpu().numpy().tobytes()
        cursor = 0
        objects: dict[str, list[tuple[int, bytes]]] = {}
        for key, _, size, offset in records:
            objects.setdefault(key, []).append((offset, snapshot[cursor : cursor + size]))
            cursor += size
        manifest = {}
        if operation == "store":
            self.directory.mkdir(parents=True, exist_ok=False)
        for key, ranges in objects.items():
            path = self.directory / (sha256(key.encode()).hexdigest() + ".bin")
            cursor = 0
            pieces = []
            for offset, payload in sorted(ranges):
                assert offset == cursor, f"{operation} has gaps or overlapping bytes for {key}"
                pieces.append(payload)
                cursor += len(payload)
            actual = b"".join(pieces)
            assert any(actual), f"{operation} contains only zero bytes for {key}"
            if operation == "store":
                path.write_bytes(actual)
            else:
                assert path.exists(), f"Load used an object absent from Store: {key}"
                assert actual == path.read_bytes(), f"Loaded KV differs byte-for-byte from saved KV: {key}"
            manifest[key] = len(actual)
        self.records[operation].clear()
        return {"objects": manifest, "bytes": total_bytes}

    def clear_local_kv(self) -> None:
        for cache in self.caches:
            cache.zero_()
        torch.npu.synchronize()

    def close(self) -> None:
        for name, original in self.originals.items():
            setattr(self.backend, name, original)


def compare_saved_bytes(reference: Path, candidate: Path, expected: dict, actual: dict) -> None:
    """Compare actual bytes, not output tokens or digest-only approximations."""
    assert expected and expected == actual, "Production/v1 saved different object keys or byte counts"
    for key in expected:
        filename = sha256(key.encode()).hexdigest() + ".bin"
        assert (reference / filename).read_bytes() == (candidate / filename).read_bytes(), (
            f"Production/v1 saved different KV bytes: {key}"
        )
