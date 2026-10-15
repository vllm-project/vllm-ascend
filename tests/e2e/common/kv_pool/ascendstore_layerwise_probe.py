# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only KeyRange publication, destination-byte and per-layer readiness checks."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any

import torch

RANGE_METHODS = MappingProxyType(
    {
        "production": (
            "batch_put_start",
            "batch_copy_put",
            "batch_commit",
            "batch_revoke",
            "batch_get_start",
            "batch_copy_get",
            "batch_get_end",
        ),
        "v1": (
            "start_key_range_store",
            "copy_key_range_store",
            "commit_key_range_store",
            "revoke_key_range_store",
            "start_key_range_load",
            "copy_key_range_load",
            "finish_key_range_load",
        ),
    }
)


class LayerwiseRangeProbe:
    """Retain cold Store bytes and inspect real Load destinations before attention consumes each layer.

    The probe owns its hooks and CPU snapshots. It serializes observations, never holds its lock across a
    layer wait, and is used only in a separate correctness server, not in timed performance runs.
    """

    def __init__(self, connector: Any, backend: Any, caches: dict[str, Any], implementation: str) -> None:
        self.connector = connector
        self.backend = backend
        self.caches = {
            name: (value,) if isinstance(value, torch.Tensor) else tuple(value) for name, value in caches.items()
        }
        assert self.caches and all(self.caches.values()), "No registered layer KV tensors"
        self.lock = threading.RLock()
        self.hooks: list[tuple[Any, str, Any]] = []
        self.objects: dict[str, int] = {}
        self.stored: dict[str, dict[int, tuple[str, torch.Tensor]]] = {}
        self.loaded: dict[str, dict[int, int]] = {}
        self.checked: dict[str, set[int]] = {}
        self.open_stores: set[str] = set()
        self.committed: set[str] = set()
        self.open_loads: set[str] = set()
        self.finished_loads: set[str] = set()
        self.layer_waits: dict[str, int] = {}
        self.poisoning_armed = False
        self.poisoned_bytes = 0
        self.checked_bytes = 0
        names = RANGE_METHODS[implementation]
        (
            self.original_start_store,
            self.original_copy_store,
            self.original_commit,
            self.original_revoke,
            self.original_start_load,
            self.original_copy_load,
            self.original_finish_load,
        ) = (getattr(backend, name) for name in names)
        self.original_wait = connector.wait_for_layer_load
        try:
            wrappers = (
                self.start_store,
                self.copy_store,
                self.commit,
                self.revoke,
                self.start_load,
                self.copy_load,
                self.finish_load,
            )
            for name, wrapper in zip(names, wrappers, strict=True):
                self._install(backend, name, wrapper)
            self._install(connector, "wait_for_layer_load", self.wait_for_layer)
        except BaseException:
            self.remove()
            raise

    def _install(self, owner: Any, name: str, replacement: Any) -> None:
        self.hooks.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def remove(self) -> None:
        for owner, name, original in reversed(self.hooks):
            setattr(owner, name, original)
        self.hooks.clear()

    def start_store(self, keys: Sequence[str], sizes: Sequence[int]) -> Any:
        with self.lock:
            assert not self.poisoning_armed, "Unexpected Store during the full-hit validation phase"
            for key, size in zip(keys, sizes, strict=True):
                assert key not in self.objects and size > 0, "Repeated key or invalid object size"
                self.objects[key], self.stored[key] = size, {}
            result = self.original_start_store(keys, sizes)
            self._successful(keys, result)
            self.open_stores.update(keys)
            return result

    def copy_store(self, keys: Sequence[str], addresses: Any, sizes: Any, offsets: Any) -> Any:
        with self.lock:
            assert set(keys) <= self.open_stores, "Store range has no live put session"
            # Both implementations enter this method only after the layer's source-ready event.
            for key, pointers, lengths, starts in zip(keys, addresses, sizes, offsets, strict=True):
                for pointer, size, offset in zip(pointers, lengths, starts, strict=True):
                    layer, view = self._byte_view(pointer, size)
                    assert 0 <= offset < offset + size <= self.objects[key], "Range outside object"
                    for previous, (_, snapshot) in self.stored[key].items():
                        assert offset + size <= previous or previous + snapshot.numel() <= offset, "Overlapping ranges"
                    self.stored[key][offset] = (layer, view.to("cpu", copy=True))
            result = self.original_copy_store(keys, addresses, sizes, offsets)
            self._successful(keys, result)
            return result

    def commit(self, keys: Sequence[str]) -> Any:
        with self.lock:
            assert set(keys) <= self.open_stores, "Commit has no live put session"
            for key in keys:
                cursor = 0
                for offset, (_, snapshot) in sorted(self.stored[key].items()):
                    assert offset == cursor, "Commit before complete object coverage"
                    cursor += snapshot.numel()
                assert cursor == self.objects[key], "Commit before complete object coverage"
                assert {layer for layer, _ in self.stored[key].values()} == set(self.caches), "Commit misses layers"
            assert tuple(self.backend.exists(keys)) == (0,) * len(keys), "Uncommitted object is readable"
            result = self.original_commit(keys)
            self._successful(keys, result)
            assert tuple(self.backend.exists(keys)) == (1,) * len(keys), "Commit did not publish complete objects"
            self.open_stores.difference_update(keys)
            self.committed.update(keys)
            return result

    def revoke(self, keys: Sequence[str]) -> Any:
        # Preserve real cleanup even when another assertion has failed in the sender thread.
        with self.lock:
            result = self.original_revoke(keys)
            self.open_stores.difference_update(keys)
            return result

    def arm_destination_poisoning(self) -> dict[str, Any]:
        with self.lock:
            assert self.objects and set(self.objects) == self.committed and not self.open_stores
            assert not self.loaded and not self.open_loads, "Cold requests unexpectedly loaded External KV"
            self.poisoning_armed = True
            return self.summary()

    def start_load(self, keys: Sequence[str]) -> Any:
        with self.lock:
            assert self.poisoning_armed, "Load must follow local cache reset and arming destination poisoning"
            assert set(keys) <= self.committed, "Load key was not published by the cold requests"
            assert not set(keys) & (self.open_loads | self.finished_loads), "Repeated get session"
            result = self.original_start_load(keys)
            self._successful(keys, result)
            for key in keys:
                self.loaded[key], self.checked[key] = {}, set()
            self.open_loads.update(keys)
            return result

    def copy_load(self, keys: Sequence[str], addresses: Any, sizes: Any, offsets: Any) -> Any:
        with self.lock:
            assert set(keys) <= self.open_loads, "Load range has no live get session"
            destination_views = []
            for key, pointers, lengths, starts in zip(keys, addresses, sizes, offsets, strict=True):
                for pointer, size, offset in zip(pointers, lengths, starts, strict=True):
                    assert offset in self.stored[key] and offset not in self.loaded[key], "Unknown/repeated Load range"
                    expected_layer, expected = self.stored[key][offset]
                    layer, view = self._byte_view(pointer, size)
                    assert layer == expected_layer and size == expected.numel(), "Load changed layer/range mapping"
                    assert bool((expected != 255).any()), "Reference is indistinguishable from poisoned bytes"
                    destination_views.append(view)
            # Poison only the actual destinations, including relocated blocks; leave unused/sentinel KV untouched.
            # 0xff bytes encode NaNs in this test's BF16 caches, so a no-op Load cannot pass by retaining old KV.
            for view in destination_views:
                view.fill_(255)
                self.poisoned_bytes += view.numel()
            if any(view.device.type != "cpu" for view in destination_views):
                torch.npu.synchronize()
            result = self.original_copy_load(keys, addresses, sizes, offsets)
            self._successful(keys, result)
            for key, pointers, starts in zip(keys, addresses, offsets, strict=True):
                self.loaded[key].update(zip(starts, pointers, strict=True))
            return result

    def finish_load(self, keys: Sequence[str]) -> Any:
        with self.lock:
            assert set(keys) <= self.open_loads, "Get end has no live session"
            for key in keys:
                assert self.loaded[key].keys() == self.stored[key].keys(), "Get end before all ranges completed"
            result = self.original_finish_load(keys)
            assert result is None or result == 0, f"Get session end failed: {result}"
            self.open_loads.difference_update(keys)
            self.finished_loads.update(keys)
            return result

    def wait_for_layer(self, layer_name: str, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            keys = tuple(self.open_loads)
        # The real wait must make progress on its sender/receiver threads without this observation lock.
        result = self.original_wait(layer_name, *args, **kwargs)
        with self.lock:
            for key in keys:
                ranges = [
                    (offset, snapshot) for offset, (layer, snapshot) in self.stored[key].items() if layer == layer_name
                ]
                assert ranges, f"No stored ranges for layer {layer_name}"
                for offset, expected in ranges:
                    assert offset in self.loaded[key], "Layer wait returned before its ranges completed"
                    assert offset not in self.checked[key], "Repeated layer consumption"
                    _, actual = self._byte_view(self.loaded[key][offset], expected.numel())
                    assert torch.equal(actual.to("cpu", copy=True), expected), "Layer wait returned incorrect KV bytes"
                    self.checked[key].add(offset)
                    self.checked_bytes += expected.numel()
            if keys:
                self.layer_waits[layer_name] = self.layer_waits.get(layer_name, 0) + 1
        return result

    def verify_complete(self) -> dict[str, Any]:
        with self.lock:
            assert self.poisoning_armed and self.committed
            assert not self.open_stores and not self.open_loads, "Range sessions leaked"
            assert self.finished_loads == self.committed, "Some published objects were not loaded"
            assert set(self.layer_waits) == set(self.caches), "Some layers were not verified before compute"
            for key in self.committed:
                assert self.checked[key] == set(self.stored[key]), "Restored ranges were not checked before compute"
            assert self.checked_bytes == self.poisoned_bytes == sum(self.objects.values()) > 0
            return {**self.summary(), "status": "passed"}

    def summary(self) -> dict[str, Any]:
        return {
            "stored_objects": len(self.objects),
            "committed_objects": len(self.committed),
            "stored_bytes": sum(self.objects.values()),
            "checked_bytes_before_compute": self.checked_bytes,
            "completed_get_sessions": len(self.finished_loads),
            "open_put_sessions": len(self.open_stores),
            "open_get_sessions": len(self.open_loads),
            "layer_waits": dict(self.layer_waits),
            "destination_poisoning_armed": self.poisoning_armed,
            "poisoned_destination_bytes": self.poisoned_bytes,
        }

    def _byte_view(self, address: int, size: int) -> tuple[str, torch.Tensor]:
        assert size > 0, "Empty KV range"
        matches = []
        for layer, tensors in self.caches.items():
            for cache in tensors:
                # Logical view bounds identify a layer even if several layers share one allocation.
                span = 1 + sum((count - 1) * stride for count, stride in zip(cache.shape, cache.stride()))
                if cache.data_ptr() <= address and address + size <= cache.data_ptr() + span * cache.element_size():
                    storage = cache.untyped_storage()
                    view = torch.empty(0, dtype=torch.uint8, device=cache.device).set_(
                        storage, address - storage.data_ptr(), (size,), (1,)
                    )
                    matches.append((layer, view))
        assert len({layer for layer, _ in matches}) == 1, "KV address has no unique registered physical layer"
        return matches[0]

    @staticmethod
    def _successful(keys: Sequence[str], results: Sequence[int]) -> None:
        assert tuple(results) == (0,) * len(keys), f"KeyRange operation failed: {results}"
