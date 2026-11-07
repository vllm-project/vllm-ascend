# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from vllm_ascend.models.qwen4_exp import qsa


def test_qsa_restores_before_updates_and_opens_prefetch_after_updates(monkeypatch):
    order = []
    main = SimpleNamespace(num_actual_tokens=2, slot_mapping=torch.arange(2))
    raw = SimpleNamespace(num_actual_tokens=2, token_to_req=None, logical_positions=None, seq_lens=None)
    monkeypatch.setattr(qsa, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"main": main, "raw": raw}))
    monkeypatch.setattr(qsa, "wait_for_kv_layer_from_connector", lambda name: order.append("load"))
    saved = []

    def save_cache(layer_name, cache):
        saved.append((layer_name, cache))
        order.append("save")

    monkeypatch.setattr(qsa, "maybe_save_kv_layer_to_connector", save_cache)

    @contextmanager
    def window():
        order.append("gate_open")
        yield
        order.append("gate_close")

    monkeypatch.setattr(qsa, "attention_transfer_window", window)

    class Indexer:
        raw_key_cache = SimpleNamespace(prefix="raw")
        output_width = 4

        def __call__(self, *args):
            order.append("indexer_update")
            return torch.zeros(2, self.output_width, dtype=torch.int32)

    owner = SimpleNamespace(
        layer_name="main",
        kv_cache=(torch.zeros(1), torch.zeros(1)),
        indexer=Indexer(),
        topk_indices_buffer=torch.zeros(2, 4, dtype=torch.int32),
        impl=SimpleNamespace(
            do_kv_cache_update=lambda *args: order.append("main_kv_update"),
            forward_qsa=lambda *args, **kwargs: order.append("attention"),
        ),
    )
    value = torch.zeros(2, 4)
    qsa.AscendQwen4ExpQSAAttention._run_qsa(owner, value, torch.arange(2), value, value, value, value)
    assert order == ["load", "indexer_update", "main_kv_update", "gate_open", "attention", "gate_close", "save"]
    assert saved[0][0] == "main"
    assert isinstance(saved[0][1], list)
    assert all(actual is expected for actual, expected in zip(saved[0][1], owner.kv_cache, strict=True))


def test_qsa_overlaps_indexer_prolog_with_main_qkv_preparation(monkeypatch):
    order = []
    main_metadata = SimpleNamespace(num_actual_tokens=2, slot_mapping=torch.arange(2))
    side_metadata = SimpleNamespace(
        num_actual_tokens=2,
        token_to_req=None,
        logical_positions=None,
        seq_lens=None,
    )
    monkeypatch.setattr(
        qsa,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"main": main_metadata, "raw": side_metadata}),
    )
    monkeypatch.setattr(qsa, "wait_for_kv_layer_from_connector", lambda name: order.append("load"))
    monkeypatch.setattr(qsa, "maybe_save_kv_layer_to_connector", lambda *args: order.append("save"))

    @contextmanager
    def transfer_window():
        order.append("gate_open")
        yield
        order.append("gate_close")

    monkeypatch.setattr(qsa, "attention_transfer_window", transfer_window)

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def record_event(self):
            event = f"{self.name}_event"
            order.append(f"{self.name}_record")
            return event

        def wait_event(self, event):
            order.append(f"{self.name}_wait_{event}")

    main_stream = FakeStream("main")
    indexer_stream = FakeStream("indexer")
    monkeypatch.setattr(qsa, "current_stream", lambda: main_stream)

    @contextmanager
    def stream_switch(stream):
        assert stream is indexer_stream
        order.append("indexer_enter")
        yield
        order.append("indexer_exit")

    monkeypatch.setattr(qsa, "npu_stream_switch", stream_switch)

    class Indexer:
        raw_key_cache = SimpleNamespace(prefix="raw")
        output_width = 4

        def prepare_prolog(self, hidden_states, positions):
            order.append("indexer_prolog")
            return hidden_states, side_metadata

        def select_prepared(self, prepared, out):
            order.append("indexer_select")
            return out

    value = torch.zeros(2, 4)
    owner = SimpleNamespace(
        layer_name="main",
        kv_cache=(torch.zeros(1), torch.zeros(1)),
        indexer=Indexer(),
        indexer_prolog_stream=indexer_stream,
        topk_indices_buffer=torch.zeros(2, 4, dtype=torch.int32),
        num_heads=1,
        num_kv_heads=1,
        head_dim=4,
        _supports_indexer_prolog_overlap=lambda qkv: True,
        _project_qkv_gate=lambda qkv, positions: (
            order.append("main_qkv_prepare") or value,
            value,
            value,
            value,
        ),
        impl=SimpleNamespace(
            do_kv_cache_update=lambda *args: order.append("main_kv_update"),
            forward_qsa=lambda *args, **kwargs: order.append("attention"),
        ),
    )

    qsa.AscendQwen4ExpQSAAttention._run_qsa(
        owner,
        value,
        torch.arange(2),
        None,
        None,
        None,
        value,
        qkv=torch.zeros(2, 4),
        gate=value.clone(),
    )

    assert order == [
        "load",
        "main_record",
        "indexer_enter",
        "indexer_wait_main_event",
        "indexer_prolog",
        "indexer_record",
        "indexer_exit",
        "main_qkv_prepare",
        "main_kv_update",
        "main_wait_indexer_event",
        "indexer_select",
        "gate_open",
        "attention",
        "gate_close",
        "save",
    ]
