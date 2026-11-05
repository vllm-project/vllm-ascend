# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from vllm_ascend.models.qwen4_exp import qsa
from vllm_ascend.utils import vllm_version_is


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

        def __call__(self, *args, **kwargs):
            order.append("indexer_update")
            selected = torch.zeros(2, self.output_width, dtype=torch.int32)
            return selected if vllm_version_is("0.32.0") else (selected, None)

    owner = SimpleNamespace(
        layer_name="main",
        kv_cache=(torch.zeros(1), torch.zeros(1)),
        indexer=Indexer(),
        topk_indices_buffer=torch.zeros(2, 4, dtype=torch.int32),
        impl=SimpleNamespace(
            do_kv_cache_update=lambda *args: order.append("main_kv_update"),
            forward_qsa=lambda *args, **kwargs: order.append("attention"),
            forward=lambda *args, **kwargs: order.append("attention"),
        ),
    )
    value = torch.zeros(2, 4)
    qsa.AscendQwen4ExpQSAAttention._run_qsa(owner, value, torch.arange(2), value, value, value, value)
    assert order == ["load", "indexer_update", "main_kv_update", "gate_open", "attention", "gate_close", "save"]
    assert saved[0][0] == "main"
    assert isinstance(saved[0][1], list)
    assert all(actual is expected for actual, expected in zip(saved[0][1], owner.kv_cache, strict=True))
