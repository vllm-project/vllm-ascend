# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the diagnostic itself; these do not establish native/NPU correctness."""

import threading
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch

from tests.e2e.common.kv_pool.ascendstore_layerwise_probe import RANGE_METHODS, LayerwiseRangeProbe
from tests.e2e.pull_request.one_card.ascendstore_v1_layerwise import AscendStoreLayerwiseWorkerExtension


class RangeBackend:
    def __init__(self, caches, implementation):
        self.caches = caches
        self.objects = {}
        self.published = set()
        self.no_op_load = False
        self.load_code = 0
        self.end_code = None if implementation == "v1" else 0
        self.premature_publication = False
        for name, method in zip(
            RANGE_METHODS[implementation],
            (
                self.start_store,
                self.copy_store,
                self.commit,
                self.revoke,
                self.start_load,
                self.copy_load,
                self.finish_load,
            ),
            strict=True,
        ):
            setattr(self, name, method)

    def exists(self, keys):
        return [int(key in self.published or self.premature_publication) for key in keys]

    def start_store(self, keys, sizes):
        self.objects.update((key, torch.empty(size, dtype=torch.uint8)) for key, size in zip(keys, sizes, strict=True))
        return [0] * len(keys)

    def copy_store(self, keys, addresses, sizes, offsets):
        for key, pointers, lengths, starts in zip(keys, addresses, sizes, offsets, strict=True):
            for pointer, size, offset in zip(pointers, lengths, starts, strict=True):
                self.objects[key][offset : offset + size].copy_(self.view(pointer, size))
        return [0] * len(keys)

    def commit(self, keys):
        self.published.update(keys)
        return [0] * len(keys)

    def revoke(self, keys):
        return [0] * len(keys)

    def start_load(self, keys):
        return [0] * len(keys)

    def copy_load(self, keys, addresses, sizes, offsets):
        if not self.no_op_load:
            for key, pointers, lengths, starts in zip(keys, addresses, sizes, offsets, strict=True):
                for pointer, size, offset in zip(pointers, lengths, starts, strict=True):
                    self.view(pointer, size).copy_(self.objects[key][offset : offset + size])
        return [self.load_code] * len(keys)

    def finish_load(self, keys):
        return self.end_code

    def view(self, pointer, size):
        for cache in self.caches.values():
            storage = cache.untyped_storage()
            if storage.data_ptr() <= pointer and pointer + size <= storage.data_ptr() + storage.nbytes():
                return torch.empty(0, dtype=torch.uint8).set_(storage, pointer - storage.data_ptr(), (size,), (1,))
        raise AssertionError("Unknown pointer")


@pytest.fixture(params=["production", "v1"])
def chain(request):
    caches = {f"layer{index}": torch.arange(1, 17, dtype=torch.float32) for index in range(2)}
    connector = SimpleNamespace(wait_for_layer_load=Mock(return_value="waited"))
    backend = RangeBackend(caches, request.param)
    original_methods = [getattr(backend, name) for name in RANGE_METHODS[request.param]]
    original_wait = connector.wait_for_layer_load
    probe = LayerwiseRangeProbe(connector, backend, caches, request.param)
    names = RANGE_METHODS[request.param]

    def call(operation, *args):
        return getattr(backend, names[operation])(*args)

    def ranges(layer, destination=False):
        # Use a different physical block for Load to catch accidental source-address comparisons.
        pointer = caches[f"layer{layer}"].data_ptr() + (32 if destination else 0)
        return ["key"], [[pointer, pointer + 16]], [[16, 16]], [[layer * 32, layer * 32 + 16]]

    yield SimpleNamespace(
        probe=probe,
        backend=backend,
        connector=connector,
        call=call,
        ranges=ranges,
        caches=caches,
        originals=original_methods,
        original_wait=original_wait,
        names=names,
    )
    probe.remove()


def publish(chain):
    chain.call(0, ["key"], [64])
    for layer in range(2):
        chain.call(1, *chain.ranges(layer))
    chain.call(2, ["key"])
    chain.probe.arm_destination_poisoning()
    chain.call(4, ["key"])


def test_complete_ranges_and_last_layer_session_end_before_wait_returns(chain):
    publish(chain)
    for layer in range(2):
        chain.call(5, *chain.ranges(layer, destination=True))
        if layer == 1:
            chain.probe.original_wait = lambda *args: chain.call(6, ["key"])
        chain.connector.wait_for_layer_load(f"layer{layer}")
    result = chain.probe.verify_complete()
    assert result["status"] == "passed"
    assert result["checked_bytes_before_compute"] == result["poisoned_destination_bytes"] == 64
    assert result["completed_get_sessions"] == 1
    assert result["layer_waits"] == {"layer0": 1, "layer1": 1}
    chain.probe.remove()
    assert [getattr(chain.backend, name) for name in chain.names] == chain.originals
    assert chain.connector.wait_for_layer_load == chain.original_wait


def test_missing_range_prevents_commit(chain):
    chain.call(0, ["key"], [64])
    chain.call(1, *chain.ranges(0))
    with pytest.raises(AssertionError, match="complete object coverage"):
        chain.call(2, ["key"])
    assert not chain.backend.published


def test_full_bytes_without_all_layers_prevents_commit(chain):
    chain.call(0, ["key"], [32])
    chain.call(1, *chain.ranges(0))
    with pytest.raises(AssertionError, match="misses layers"):
        chain.call(2, ["key"])


def test_overlapping_store_range_fails(chain):
    chain.call(0, ["key"], [64])
    chain.call(1, *chain.ranges(0))
    with pytest.raises(AssertionError, match="Overlapping"):
        chain.call(1, *chain.ranges(0))


def test_premature_visibility_fails(chain):
    chain.backend.premature_publication = True
    with pytest.raises(AssertionError, match="Uncommitted object is readable"):
        publish(chain)


@pytest.mark.parametrize("fault", ["noop", "corruption"])
def test_success_code_cannot_hide_incorrect_native_load(chain, fault):
    publish(chain)
    chain.backend.no_op_load = fault == "noop"
    if fault == "corruption":
        chain.backend.objects["key"][0] ^= 1
    chain.call(5, *chain.ranges(0, destination=True))
    with pytest.raises(AssertionError, match="incorrect KV bytes"):
        chain.connector.wait_for_layer_load("layer0")


def test_early_wait_and_early_session_end_fail(chain):
    publish(chain)
    with pytest.raises(AssertionError, match="before its ranges completed"):
        chain.connector.wait_for_layer_load("layer0")
    with pytest.raises(AssertionError, match="before all ranges completed"):
        chain.call(6, ["key"])


def test_native_failure_and_unclosed_session_fail(chain):
    publish(chain)
    chain.backend.load_code = -1
    with pytest.raises(AssertionError, match="KeyRange operation failed"):
        chain.call(5, *chain.ranges(0, destination=True))
    with pytest.raises(AssertionError, match="sessions leaked"):
        chain.probe.verify_complete()


def test_session_end_failure_is_not_ignored(chain):
    publish(chain)
    for layer in range(2):
        chain.call(5, *chain.ranges(layer, destination=True))
        chain.connector.wait_for_layer_load(f"layer{layer}")
    chain.backend.end_code = -1
    with pytest.raises(AssertionError, match="session end failed"):
        chain.call(6, ["key"])


def test_layer_wait_does_not_hold_observer_lock(chain):
    publish(chain)
    failures = []

    def load_in_receiver():
        try:
            chain.call(5, *chain.ranges(0, destination=True))
        except BaseException as error:
            failures.append(error)

    def wait_in_compute(*args):
        receiver = threading.Thread(target=load_in_receiver, daemon=True)
        receiver.start()
        receiver.join(timeout=2)
        assert not receiver.is_alive(), "Observer lock blocked the real receiver"

    chain.probe.original_wait = wait_in_compute
    chain.connector.wait_for_layer_load("layer0")
    assert not failures


def test_shared_allocation_is_classified_by_layer_view():
    storage = torch.arange(32, dtype=torch.float32)
    caches = {"layer0": storage[:16], "layer1": storage[16:]}
    backend = RangeBackend(caches, "v1")
    probe = LayerwiseRangeProbe(SimpleNamespace(wait_for_layer_load=Mock()), backend, caches, "v1")
    try:
        assert probe._byte_view(caches["layer1"].data_ptr(), 32)[0] == "layer1"
    finally:
        probe.remove()


@pytest.mark.parametrize("implementation", ["production", "v1"])
def test_worker_drains_the_layerwise_owner_not_the_bulk_sender(monkeypatch, implementation):
    extension = AscendStoreLayerwiseWorkerExtension()
    store = object()
    backend = SimpleNamespace(store=store, _store=store)
    worker = SimpleNamespace(_drain_attention_transfers=Mock(), fence_previous_store=Mock(), _pending_store_batch=None)
    monkeypatch.setattr(extension, "_layerwise_resources", lambda: (None, worker, backend, {}, implementation))
    assert extension._drain_stores()[:2] == (backend, store)
    if implementation == "production":
        worker._drain_attention_transfers.assert_called_once_with(drain_recv=False, full=True)
        worker.fence_previous_store.assert_not_called()
    else:
        worker.fence_previous_store.assert_called_once_with()
        worker._drain_attention_transfers.assert_not_called()


def test_worker_restores_hooks_even_if_drain_raises(monkeypatch):
    extension = AscendStoreLayerwiseWorkerExtension()
    probe = cast(LayerwiseRangeProbe, SimpleNamespace(remove=Mock()))
    extension._layerwise_probe = probe
    monkeypatch.setattr(extension, "_drain_stores", Mock(side_effect=RuntimeError("sender failed")))
    with pytest.raises(RuntimeError, match="sender failed"):
        extension.remove_layerwise_validation()
    cast(Mock, probe.remove).assert_called_once_with()
    assert not hasattr(extension, "_layerwise_probe")
