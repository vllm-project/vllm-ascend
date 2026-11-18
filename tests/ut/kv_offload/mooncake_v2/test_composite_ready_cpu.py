# SPDX-License-Identifier: Apache-2.0
"""Composite PD readiness executes production methods with CPU events."""

from types import SimpleNamespace
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
