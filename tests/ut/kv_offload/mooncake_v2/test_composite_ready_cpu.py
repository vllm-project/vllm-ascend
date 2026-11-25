# SPDX-License-Identifier: Apache-2.0
"""Composite PD readiness executes production methods with CPU events."""

import queue
import threading
from concurrent.futures import Future
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.ut.attention.test_mla_layerwise_load_cpu import load

PUSH = "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/push_worker.py"
OWNER = "MooncakePushConnectorWorker"
CONNECTOR = "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py"


def worker(cache_names=("model.layers.2.long", "model.layers.2.swa")):
    layer = ("model", 2)
    return SimpleNamespace(
        kv_cache_config=SimpleNamespace(dspark_draft_layer_names={"draft"}),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    "model.layers.2.v41_attn": SimpleNamespace(kv_transfer_cache_names=cache_names),
                }
            )
        ),
        _layer_indices={"model.layers.2.long": 0, "model.layers.2.swa": 1, "draft": 2},
        _layer_index_by_name={"model.layers.2.long": layer, "model.layers.2.swa": layer, "draft": ("draft", 0)},
        _layer_names_by_index={layer: ["model.layers.2.long", "model.layers.2.swa"], ("draft", 0): ["draft"]},
        kv_transfer_config=SimpleNamespace(is_kv_producer=True),
        _prefilling_request_ids=("r",),
        _shared_layer_indices={},
        _last_layer_index=None,
        _process_sending_requests=Mock(),
        _send_completions=Mock(),
    )


@pytest.mark.parametrize("draft_alias", [False, True])
@pytest.mark.parametrize("target_name", ["model.layers.2.v41_attn", "model.layers.2.long"])
def test_target_and_draft_use_their_own_ready_hooks(draft_alias, target_name):
    obj = worker()
    if draft_alias:
        obj.vllm_config.compilation_config.static_forward_context["draft.v41_attn"] = SimpleNamespace(
            kv_transfer_cache_names=("draft",)
        )
    load(PUSH, "_register_composite_layer_aliases", {}, OWNER)(obj)
    state = SimpleNamespace(
        error=None, metadata=SimpleNamespace(num_computed_tokens=16, num_scheduled_tokens=16), available_layers={}
    )
    obj._reqs_send_state = {"r": state}
    save = load(PUSH, "save_kv_layer", {"torch": SimpleNamespace(npu=SimpleNamespace(Event=Mock))}, OWNER)
    save(obj, target_name)
    assert {name: end for name, (end, _) in state.available_layers.items()} == {
        "model.layers.2.long": 32,
        "model.layers.2.swa": 32,
    }
    save(obj, "draft.v41_attn" if draft_alias else "draft")
    assert state.available_layers["draft"][0] == 32
    assert state.available_layers["draft"][1] is not state.available_layers["model.layers.2.long"][1]


@pytest.mark.parametrize("num_external_tokens", [8, 16])
def test_hbm_prefix_is_ready_while_external_prefix_waits_in_same_batch(num_external_tokens):
    obj = worker()
    load(PUSH, "_register_composite_layer_aliases", {}, OWNER)(obj)
    states = {
        name: SimpleNamespace(destinations={1: object()}, available_layers={}, local_request_id=name, error=None)
        for name in ("hbm", "pool")
    }
    obj._reqs_send_state = states
    meta = SimpleNamespace(
        push_send_requests={
            name: SimpleNamespace(
                request_id=name, num_computed_tokens=16, num_external_tokens=external, num_scheduled_tokens=16
            )
            for name, external in (("hbm", 0), ("pool", num_external_tokens))
        },
        requests_to_send={},
        push_aborted_requests={},
    )
    start = load(
        PUSH,
        "start_load_kv",
        {"torch": SimpleNamespace(npu=SimpleNamespace(Event=Mock)), "SendReqInfo": lambda name: states[name]},
        OWNER,
    )
    start(obj, meta)
    ready_prefix = {name: end for name, (end, _) in states["hbm"].available_layers.items()}
    assert ready_prefix == dict.fromkeys(obj._layer_indices, 16)
    assert not states["pool"].available_layers
    save = load(PUSH, "save_kv_layer", {"torch": SimpleNamespace(npu=SimpleNamespace(Event=Mock))}, OWNER)
    save(obj, "model.layers.2.v41_attn")
    for state in states.values():
        assert state.available_layers["model.layers.2.long"][0] == 32
        assert state.available_layers["model.layers.2.swa"][0] == 32


@pytest.mark.parametrize("configuration", ["consumer", "resident", "offload"])
def test_push_advertises_the_worker_layout(configuration):
    transfer_config = SimpleNamespace(is_kv_producer=configuration != "consumer")
    vllm_config = SimpleNamespace(kv_transfer_config=transfer_config)
    cache_config = SimpleNamespace(kv_transfer_config=transfer_config)
    worker = SimpleNamespace(kv_transfer_config=transfer_config)
    entry = "model.layers.24.self_attn"
    layout = SimpleNamespace(layers=(24, 25), cache_layer_indices={entry: 24}, reused_cache_names={entry})
    get_layout = Mock(return_value=None if configuration == "resident" else layout)
    init_worker = load(
        PUSH,
        "__init__",
        {
            "super": lambda: SimpleNamespace(__init__=Mock()),
            "get_layerwise_reuse_layout": get_layout,
            "threading": threading,
            "queue": queue,
            "SizedDict": Mock(),
        },
        OWNER,
    )
    init_worker(worker, vllm_config, "p", cache_config)
    shared = configuration == "offload"
    assert worker._shared_layer_indices == ({entry: 0} if shared else {})
    registry = load(PUSH, "supports_layerwise_buffer_reuse", {}, OWNER)
    worker.supports_layerwise_buffer_reuse = registry(worker)
    facade = SimpleNamespace(connector_worker=None)
    supports_reuse = load(
        CONNECTOR,
        "supports_layerwise_buffer_reuse",
        {"MooncakePushConnectorWorker": type(worker)},
        "MooncakePushConnector",
    )
    assert supports_reuse(facade) is False
    facade.connector_worker = worker
    assert supports_reuse(facade) is shared
    if configuration == "consumer":
        get_layout.assert_not_called()
    else:
        get_layout.assert_called_once_with(cache_config)


def test_last_shared_hook_submits_writes_once():
    obj = worker()
    shared, resident = "model.layers.2.long", "model.layers.2.swa"
    other_shared = "model.layers.3.long"
    obj._shared_layer_indices = {shared: 0, other_shared: 1}
    obj._last_layer_index = obj._layer_index_by_name[shared]
    target = (0, 0, 0)
    event = Mock()
    resident_write: Future[str | None] = Future()
    shared_write: Future[str | None] = Future()
    metadata_future = Mock()
    metadata_future.done.return_value = False
    destination = SimpleNamespace(
        metadata_future=metadata_future, required_layers={shared, resident}, pending_layers=set(), completion=None
    )
    state = SimpleNamespace(
        available_layers={resident: (32, event)},
        destinations={target: destination},
        metadata=SimpleNamespace(num_computed_tokens=16, num_scheduled_tokens=16),
        consumer_engine_id="d",
        expected_ranks={target},
        error=None,
    )
    obj._reqs_send_state = {"r": state}
    obj._pending_sends = []
    obj._sending_futures = {"d": [Future()]}
    obj._sender_executor = SimpleNamespace(submit=Mock(side_effect=[resident_write, shared_write]))
    obj._execute_layer_transfer = Mock()
    obj._build_layer_transfer_request = Mock(return_value=(SimpleNamespace(request_id="r"), event))
    process = load(
        PUSH,
        "_process_sending_requests",
        {"LayerTransferBatch": lambda layer_name, *_: SimpleNamespace(layer_name=layer_name)},
        OWNER,
    )
    obj._process_sending_requests = Mock(wraps=MethodType(process, obj))
    submit = load(PUSH, "_submit_shared_layer_writes", {}, OWNER)
    obj._submit_shared_layer_writes = MethodType(submit, obj)
    save = load(PUSH, "save_kv_layer", {"torch": SimpleNamespace(npu=SimpleNamespace(Event=lambda: event))}, OWNER)

    save(obj, shared)

    obj._process_sending_requests.assert_called_once_with(layer_hook=True, force=True, wait_for_metadata=True)
    obj._send_completions.assert_called_once_with()
    metadata_future.result.assert_called_once()
    assert {batch.layer_name: future for batch, future in obj._pending_sends} == {
        resident: resident_write,
        shared: shared_write,
    }
    assert state.available_layers == {resident: (32, event)}
    obj._pending_sends.append((SimpleNamespace(layer_name=other_shared), Future()))
    waited: list[Future[str | None]] = []
    wait_futures = Mock(side_effect=waited.extend)
    wait = load(PUSH, "wait_for_layer_reuse", {"wait_futures": wait_futures}, OWNER)
    wait(obj, 0)
    assert waited == [shared_write]


def test_last_shared_hook_without_new_kv_checks_completion():
    obj = worker()
    shared = "model.layers.2.long"
    obj._shared_layer_indices = {shared: 0}
    obj._last_layer_index = obj._layer_index_by_name[shared]
    obj._reqs_send_state = {"r": SimpleNamespace(error="Push transfer failed")}
    obj._submit_shared_layer_writes = Mock()
    save = load(PUSH, "save_kv_layer", {}, OWNER)

    save(obj, shared)

    obj._submit_shared_layer_writes.assert_not_called()
    obj._process_sending_requests.assert_called_once_with(layer_hook=True, force=True)
    obj._send_completions.assert_called_once_with()
