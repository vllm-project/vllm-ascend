# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pytest

from vllm_ascend.attention.cache_layout import (
    LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT,
)
from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import (
    AscendMultiConnector,
)


class FakeNpuEvent:
    def __init__(self):
        self.record_calls = 0
        self.synchronize_calls = 0

    def record(self):
        self.record_calls += 1

    def synchronize(self):
        self.synchronize_calls += 1


class ResultObservedFuture(Future[None]):
    def __init__(self) -> None:
        super().__init__()
        self.result_called = threading.Event()

    def result(self, timeout: float | None = None) -> None:
        self.result_called.set()
        return super().result(timeout)


class FakeConnector:
    def __init__(self):
        self.load_order: list[str] = []
        self.load_batches: list[tuple[str, ...]] = []
        self.load_events: dict[str, threading.Event] = {}
        self.load_errors: dict[str, BaseException] = {}
        self.save_sources: dict[str, Future[None]] = {}
        self.start_calls = 0
        self.wait_for_save_calls = 0
        self.shutdown_calls = 0

    def start_load_kv(self, _forward_context: Any, **_kwargs) -> None:
        self.start_calls += 1

    def wait_for_layer_load(self, layer_name: str) -> None:
        self.load_order.append(layer_name)
        self.load_events.setdefault(layer_name, threading.Event()).set()
        error = self.load_errors.get(layer_name)
        if error is not None:
            raise error

    def wait_for_layer_loads(self, layer_names: tuple[str, ...]) -> None:
        self.load_batches.append(layer_names)
        for layer_name in layer_names:
            self.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        _kv_layer: Any,
        _attn_metadata: Any,
        **_kwargs,
    ) -> Future[None] | None:
        return self.save_sources.get(layer_name)

    def wait_for_save(self) -> None:
        self.wait_for_save_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def make_pingpong_connector(
    store: FakeConnector,
    p2p: FakeConnector,
    num_layers: int = 9,
) -> AscendMultiConnector:
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [store, p2p]
    connector._layerwise_workspace_fences_enabled = True
    connector._execution_id = -1
    connector._workspace_generation = 0
    connector._workspace_fence_lock = threading.Lock()
    connector._workspace_fences = {}
    connector._workspace_layer_order = tuple(
        f"layer.{layer_id}" for layer_id in range(num_layers)
    )
    connector._workspace_layer_to_index = {
        layer_name: layer_index
        for layer_index, layer_name in enumerate(connector._workspace_layer_order)
    }
    connector._workspace_layer_to_arena = {
        layer_name: layer_index % LAYERWISE_PREFILL_WORKSPACE_ARENA_COUNT
        for layer_index, layer_name in enumerate(connector._workspace_layer_order)
    }
    connector._layer_load_futures = {}
    connector._non_workspace_source_futures = {}
    connector._layer_load_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="test-layerwise-prefetch",
    )
    connector._store_connector = store
    return connector


def test_initial_four_layer_get_and_rolling_bank_reuse(monkeypatch) -> None:
    store = FakeConnector()
    p2p = FakeConnector()
    layer_zero_source: Future[None] = Future()
    layer_two_source: Future[None] = Future()
    store.save_sources["layer.0"] = layer_zero_source
    store.save_sources["layer.2"] = layer_two_source
    connector = make_pingpong_connector(store, p2p, num_layers=7)
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.ascend_multi_connector.torch.npu.Event",
        FakeNpuEvent,
    )

    try:
        connector.start_load_kv(object())
        connector.wait_for_layer_load("layer.0")
        assert store.load_batches == [
            tuple(f"layer.{layer_id}" for layer_id in range(4))
        ]
        connector.wait_for_layer_load("layer.2")

        layer_four_started = store.load_events.setdefault(
            "layer.4", threading.Event()
        )
        layer_six_started = store.load_events.setdefault(
            "layer.6", threading.Event()
        )
        for layer_id in range(4):
            connector.save_kv_layer(
                f"layer.{layer_id}",
                object(),
                object(),
            )
        assert not layer_four_started.wait(timeout=0.05)

        layer_zero_source.set_result(None)
        assert layer_four_started.wait(timeout=1)
        connector.wait_for_layer_load("layer.4")
        assert store.load_batches[1] == tuple(
            f"layer.{layer_id}" for layer_id in range(4, 6)
        )
        assert not layer_six_started.wait(timeout=0.05)

        layer_two_source.set_result(None)
        assert layer_six_started.wait(timeout=1)
        connector.wait_for_layer_load("layer.6")
        assert store.load_batches[2] == ("layer.6",)

        connector.wait_for_save()
        assert store.wait_for_save_calls == 1
        assert p2p.wait_for_save_calls == 1
    finally:
        connector.shutdown()


def test_grouped_prefetch_drains_all_bank_fences_after_one_failure(
    monkeypatch,
) -> None:
    store = FakeConnector()
    p2p = FakeConnector()
    connector = make_pingpong_connector(store, p2p, num_layers=2)
    connector._execution_id = 0
    layer_zero_source: Future[None] = Future()
    layer_one_source: Future[None] = Future()
    store.save_sources["layer.0"] = layer_zero_source
    store.save_sources["layer.1"] = layer_one_source
    layer_zero_compute_release = FakeNpuEvent()
    layer_one_compute_release = FakeNpuEvent()
    npu_events = iter(
        [
            layer_zero_compute_release,
            layer_one_compute_release,
        ]
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.ascend_multi_connector.torch.npu.Event",
        lambda: next(npu_events),
    )

    try:
        connector.save_kv_layer("layer.0", object(), object())
        connector.save_kv_layer("layer.1", object(), object())
        layer_zero_source.set_exception(RuntimeError("source failed"))
        with ThreadPoolExecutor(max_workers=1) as executor:
            wait_future = executor.submit(
                connector._wait_for_workspace_arenas, (0, 1)
            )
            assert not wait_future.done()
            layer_one_source.set_result(None)
            with pytest.raises(RuntimeError, match="Workspace source transfer failed"):
                wait_future.result()
        assert layer_zero_compute_release.synchronize_calls == 1
        assert layer_one_compute_release.synchronize_calls == 1
        assert connector._workspace_fences == {}
    finally:
        connector.shutdown()


def test_pingpong_get_failure_still_drains_children() -> None:
    store = FakeConnector()
    p2p = FakeConnector()
    store.load_errors["layer.0"] = RuntimeError("get failed")
    connector = make_pingpong_connector(store, p2p)

    try:
        connector.start_load_kv(object())
        with pytest.raises(RuntimeError, match="get failed"):
            connector.wait_for_layer_load("layer.0")
        with pytest.raises(RuntimeError, match="get failed"):
            connector.wait_for_save()
        assert store.wait_for_save_calls == 1
        assert p2p.wait_for_save_calls == 1
    finally:
        connector.shutdown()


def test_non_workspace_mtp_source_is_drained_before_save_finishes() -> None:
    store = FakeConnector()
    p2p = FakeConnector()
    mtp_source: Future[None] = Future()
    p2p.save_sources["layer.mtp"] = mtp_source
    connector = make_pingpong_connector(store, p2p)

    try:
        returned = connector.save_kv_layer("layer.mtp", object(), object())
        assert returned == [mtp_source]

        with ThreadPoolExecutor(max_workers=1) as executor:
            wait_future = executor.submit(connector.wait_for_save)
            assert not wait_future.done()
            mtp_source.set_result(None)
            wait_future.result(timeout=1)

        assert connector._non_workspace_source_futures == {}
        assert store.wait_for_save_calls == 1
        assert p2p.wait_for_save_calls == 1
    finally:
        connector.shutdown()


def test_non_workspace_mtp_source_failure_is_propagated_and_cleared() -> None:
    store = FakeConnector()
    p2p = FakeConnector()
    mtp_source: Future[None] = Future()
    p2p.save_sources["layer.mtp"] = mtp_source
    connector = make_pingpong_connector(store, p2p)

    try:
        connector.save_kv_layer("layer.mtp", object(), object())
        mtp_source.set_exception(RuntimeError("mtp source failed"))

        with pytest.raises(RuntimeError, match="mtp source failed"):
            connector.wait_for_save()

        assert connector._non_workspace_source_futures == {}
        assert store.wait_for_save_calls == 1
        assert p2p.wait_for_save_calls == 1
    finally:
        connector.shutdown()


def test_repeated_mtp_waits_for_previous_source_before_cache_write() -> None:
    store = FakeConnector()
    p2p = FakeConnector()
    mtp_source = ResultObservedFuture()
    p2p.save_sources["layer.mtp"] = mtp_source
    connector = make_pingpong_connector(store, p2p)

    try:
        connector.save_kv_layer("layer.mtp", object(), object())

        with ThreadPoolExecutor(max_workers=1) as executor:
            wait_future = executor.submit(
                connector.wait_for_layer_load, "layer.mtp"
            )
            assert mtp_source.result_called.wait(timeout=1)
            assert store.load_order == []
            assert p2p.load_order == []

            mtp_source.set_result(None)
            wait_future.result(timeout=1)

        assert store.load_order == ["layer.mtp"]
        assert p2p.load_order == ["layer.mtp"]
        assert connector._non_workspace_source_futures == {}
    finally:
        if not mtp_source.done():
            mtp_source.set_result(None)
        connector.shutdown()


def test_non_workspace_wait_only_drains_the_repeated_layer() -> None:
    store = FakeConnector()
    p2p = FakeConnector()
    first_source: Future[None] = Future()
    p2p.save_sources["layer.mtp.0"] = first_source
    connector = make_pingpong_connector(store, p2p)

    try:
        connector.save_kv_layer("layer.mtp.0", object(), object())
        connector.wait_for_layer_load("layer.mtp.1")

        assert store.load_order == ["layer.mtp.1"]
        assert p2p.load_order == ["layer.mtp.1"]
        assert connector._non_workspace_source_futures == {
            "layer.mtp.0": [first_source]
        }

        first_source.set_result(None)
        connector.wait_for_save()
    finally:
        if not first_source.done():
            first_source.set_result(None)
        connector.shutdown()
