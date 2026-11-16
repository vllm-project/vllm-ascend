# SPDX-License-Identifier: Apache-2.0
"""Composite PD readiness executes production methods with CPU events."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.ut.attention.test_mla_layerwise_load_cpu import load

PUSH = "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/push_worker.py"
OWNER = "MooncakePushConnectorWorker"


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
        _last_layer_index=None,
        _process_sending_requests=Mock(),
        _send_completions=Mock(),
    )


def test_composite_owner_is_early_ready_but_component_and_draft_are_not():
    obj = worker()
    load(PUSH, "_register_composite_layer_aliases", {}, OWNER)(obj)
    state = SimpleNamespace(
        error=None, metadata=SimpleNamespace(num_computed_tokens=16, num_scheduled_tokens=16), available_layers={}
    )
    obj._reqs_send_state = {"r": state}
    save = load(PUSH, "save_kv_layer", {"torch": SimpleNamespace(npu=SimpleNamespace(Event=Mock))}, OWNER)
    save(obj, "model.layers.2.long")
    save(obj, "draft")
    assert not state.available_layers
    save(obj, "model.layers.2.v41_attn")
    assert {name: end for name, (end, _) in state.available_layers.items()} == {
        "model.layers.2.long": 32,
        "model.layers.2.swa": 32,
    }


@pytest.mark.parametrize(
    "names",
    [
        ("missing",),
        ("model.layers.2.long",),
        ("model.layers.2.long", "model.layers.2.long"),
        ("model.layers.2.long", "draft"),
    ],
)
def test_invalid_ownership_fails_registration(names):
    with pytest.raises(ValueError):
        load(PUSH, "_register_composite_layer_aliases", {}, OWNER)(worker(names))


def test_external_prefix_is_not_marked_ready_before_model_load_hook():
    obj = worker()
    load(PUSH, "_register_composite_layer_aliases", {}, OWNER)(obj)
    state = SimpleNamespace(destinations={1: object()}, available_layers={}, local_request_id="p")
    obj._reqs_send_state = {"r": state}
    meta = SimpleNamespace(
        push_send_requests={"p": SimpleNamespace(request_id="r", num_computed_tokens=16)},
        requests_to_send={},
        push_aborted_requests={},
    )
    start = load(
        PUSH,
        "start_load_kv",
        {"torch": SimpleNamespace(npu=SimpleNamespace(Event=Mock)), "SendReqInfo": lambda _: state},
        OWNER,
    )
    start(obj, meta)
    assert not state.available_layers


def test_ascend_registration_replaces_upstream_layerwise_connector(monkeypatch):
    class Factory:
        _registry = {"MooncakeLayerwiseConnector": ("upstream", "MooncakeLayerwiseConnector")}

        @classmethod
        def register_connector(cls, name, module, class_name):
            if name in cls._registry:
                raise ValueError(f"duplicate connector: {name}")
            cls._registry[name] = (module, class_name)

    offloading = ModuleType("vllm.v1.kv_offload.factory")
    offloading.OffloadingSpecFactory = SimpleNamespace(_registry={}, register_spec=Mock())
    monkeypatch.setitem(sys.modules, offloading.__name__, offloading)
    scope = {"KVConnectorFactory": Factory, "logger": Mock()}
    path = "vllm_ascend/distributed/kv_transfer/__init__.py"
    load(path, "_register_with_class_name_alias", scope)
    load(path, "register_connector", scope)()
    assert Factory._registry["MooncakeLayerwiseConnector"] == (
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector",
        "MooncakeLayerwiseConnector",
    )
