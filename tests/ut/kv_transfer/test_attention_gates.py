# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import attention_fence


@pytest.fixture
def fake_npu(monkeypatch):
    events = []

    def make_event():
        event = Mock()
        events.append(event)
        return event

    monkeypatch.setattr(
        attention_fence.torch,
        "npu",
        SimpleNamespace(Event=make_event, current_stream=lambda: "compute"),
        raising=False,
    )
    return events


def test_early_load_gate_does_not_record_save_event(fake_npu):
    load_gate, save_gate = attention_fence.reset_attention_gates()
    save_event = Mock()
    save_gate.event = save_event
    save_gate.on_start = Mock()

    attention_fence.record_attention_prefetch_start()

    assert load_gate.wait(timeout=0)
    assert len(fake_npu) == 1
    fake_npu[0].record.assert_called_once_with("compute")
    save_event.record.assert_not_called()
    save_gate.on_start.assert_not_called()

    attention_fence.record_attention_compute_start()
    attention_fence.record_attention_compute_start()
    save_event.record.assert_called_once_with("compute")
    save_gate.on_start.assert_called_once_with()
    assert len(fake_npu) == 1


def test_attention_window_records_both_gates_and_finishes_on_error(fake_npu):
    load_gate, save_gate = attention_fence.reset_attention_gates()
    save_gate.event = Mock()
    on_start = Mock()
    save_gate.on_start = on_start
    save_gate.on_finish = Mock()

    with pytest.raises(RuntimeError, match="attention failed"), attention_fence.attention_transfer_window():
        raise RuntimeError("attention failed")

    assert load_gate.wait(timeout=0)
    save_gate.event.record.assert_called_once_with("compute")
    save_gate.on_start.assert_called_once_with()
    save_gate.on_finish.assert_called_once_with()


def test_cancel_releases_load_without_starting_save(fake_npu):
    load_gate, save_gate = attention_fence.reset_attention_gates()
    save_gate.event = Mock()
    on_start = Mock()
    save_gate.on_start = on_start

    save_gate.cancel()
    load_gate.cancel()
    attention_fence.record_attention_compute_start()

    assert load_gate.wait(timeout=0)
    on_start.assert_not_called()
    assert len(fake_npu) == 1


def test_recurrent_layer_leaves_save_event_for_post_compute_hook(fake_npu):
    load_gate, save_gate = attention_fence.reset_attention_gates()
    save_gate.on_finish = Mock()

    attention_fence.record_attention_prefetch_start()
    attention_fence.record_attention_compute_start()

    assert load_gate.wait(timeout=0)
    assert save_gate.event is None
    assert not save_gate._recorded


@pytest.mark.parametrize("route", ["unrelated", "direct", "multi_connector"])
def test_connector_wait_opens_only_ascend_store_prefetch_gate(route):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/utils.py"
    functions: list[ast.stmt] = [
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_supports_attention_prefetch_gate", "wait_for_kv_layer_from_connector"}
    ]
    connector = SimpleNamespace(
        has_connector_metadata=lambda: True,
        wait_for_layer_load=Mock(),
        supports_attention_prefetch_gate=route == "direct",
        sub_connectors=(SimpleNamespace(supports_attention_prefetch_gate=True),) if route == "multi_connector" else (),
    )
    record_prefetch = Mock()
    scope = {
        "Any": object,
        "has_kv_transfer_group": lambda: True,
        "is_v1_kv_transfer_group": lambda: True,
        "get_kv_transfer_group": lambda: connector,
        "get_forward_context": lambda: SimpleNamespace(attn_metadata={}, no_compile_layers={}),
        "_kv_layer_transfer_connectors": lambda *_args: (connector,),
        "record_attention_prefetch_start": record_prefetch,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)

    cast(Callable[[str], None], scope["wait_for_kv_layer_from_connector"])("model.layers.0.attn")

    connector.wait_for_layer_load.assert_called_once_with("model.layers.0.attn")
    assert record_prefetch.call_count == int(route != "unrelated")
