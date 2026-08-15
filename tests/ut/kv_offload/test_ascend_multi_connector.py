"""Tests for Ascend-specific MultiConnector allocation fan-out."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import (  # noqa: E402
    AscendMultiConnector,
)


class _FakeBlocks:
    def __init__(self) -> None:
        self.empty = object()

    def new_empty(self):
        return self.empty


def _make_connector(*, requires_full_blocks: bool = False):
    return SimpleNamespace(
        requires_full_blocks_on_update_after_alloc=requires_full_blocks,
        update_state_after_alloc=MagicMock(),
    )


def test_update_state_after_alloc_forwards_full_blocks_to_observer():
    chosen = _make_connector()
    full_blocks_observer = _make_connector(requires_full_blocks=True)
    unrelated = _make_connector()
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [chosen, full_blocks_observer, unrelated]
    connector._requests_to_connector = {"req-0": 0}
    request = SimpleNamespace(request_id="req-0")
    blocks = _FakeBlocks()

    connector.update_state_after_alloc(request, blocks, num_external_tokens=16)

    chosen.update_state_after_alloc.assert_called_once_with(request, blocks, 16)
    full_blocks_observer.update_state_after_alloc.assert_called_once_with(
        request,
        blocks,
        16,
    )
    unrelated.update_state_after_alloc.assert_called_once_with(request, blocks.empty, 0)


def test_update_state_after_alloc_forwards_observer_without_chosen_connector():
    full_blocks_observer = _make_connector(requires_full_blocks=True)
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [full_blocks_observer]
    connector._requests_to_connector = {}
    request = SimpleNamespace(request_id="req-0")
    blocks = _FakeBlocks()

    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)

    full_blocks_observer.update_state_after_alloc.assert_called_once_with(
        request,
        blocks,
        0,
    )


def test_layerwise_pd_completion_is_wired_and_provider_runs_first():
    call_order = []
    provider = SimpleNamespace(
        is_producer=True,
        connector_worker=object(),
        wait_for_layer_send=MagicMock(),
        wait_for_layer_load=MagicMock(side_effect=lambda *_: call_order.append("pd-load")),
        save_kv_layer=MagicMock(side_effect=lambda *_args, **_kwargs: call_order.append("pd-save")),
        on_kv_cache_written=MagicMock(side_effect=lambda *_: call_order.append("pd-written")),
    )
    store = SimpleNamespace(
        set_layerwise_pd_transfer_waiter=MagicMock(),
        wait_for_layer_load=MagicMock(side_effect=lambda *_: call_order.append("store-load")),
        save_kv_layer=MagicMock(side_effect=lambda *_args, **_kwargs: call_order.append("store-save")),
        on_kv_cache_written=MagicMock(side_effect=lambda *_: call_order.append("store-written")),
    )
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    # Put the store first to verify the dependency does not rely on config order.
    connector._connectors = [store, provider]

    connector._configure_layerwise_pd_completion()

    waiter = store.set_layerwise_pd_transfer_waiter.call_args.args[0]
    assert waiter == provider.wait_for_layer_send

    connector.wait_for_layer_load("model.layers.7.self_attn")
    connector.save_kv_layer("model.layers.7.self_attn", object(), object())
    connector.on_kv_cache_written("model.layers.7.self_attn")
    assert call_order == [
        "pd-load",
        "store-load",
        "pd-save",
        "store-save",
        "pd-written",
        "store-written",
    ]
