# SPDX-License-Identifier: Apache-2.0
"""Step5's custom DSA path must participate in the per-layer KV lifecycle."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# Match the worker's operator bootstrap before importing the model adapter.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.step5 import model, sparse_attention


@pytest.mark.parametrize("failure", [None, "load", "compute"])
def test_dsa_layer_waits_before_write_and_saves_after_compute(monkeypatch, failure):
    events = []

    def wait(name):
        assert name == "language_model.model.layers.3.self_attn.attn"
        events.append("load")
        if failure == "load":
            raise RuntimeError("load failed")

    def compute(*args):
        events.append("compute")
        if failure == "compute":
            raise RuntimeError("compute failed")

    monkeypatch.setattr(model, "wait_for_kv_layer_from_connector", wait, raising=False)
    monkeypatch.setattr(model, "maybe_save_kv_layer_to_connector", lambda *args: events.append("save"), raising=False)
    projection = MagicMock(return_value=(torch.zeros(2, 12), torch.zeros(2, 4)))
    projection.split_indexer.return_value = (None, None, None, None)
    layer = SimpleNamespace(
        layer_name="language_model.model.layers.3.self_attn.attn",
        qkv_indexer_proj=projection,
        use_fused_qknorm_rope=False,
        q_size=4,
        kv_size=4,
        num_heads=1,
        num_kv_heads=1,
        head_dim=4,
        q_norm=lambda x: x,
        k_norm=lambda x: x,
        use_rope=False,
        _finish_indexer=lambda *args: (None, None, None, None),
        _write_kv=lambda *args: events.append("write"),
        kv_cache=(torch.zeros(1), torch.zeros(1)),
        impl=SimpleNamespace(forward=compute),
        o_proj=lambda x: (x, None),
    )
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            model.AscendStep5DSAAttention.forward(layer, torch.arange(2), torch.zeros(2, 4))
        assert events == (["load"] if failure == "load" else ["load", "write", "compute"])
    else:
        model.AscendStep5DSAAttention.forward(layer, torch.arange(2), torch.zeros(2, 4))
        assert events == ["load", "write", "compute", "save"]


@pytest.mark.parametrize("verify", [False, True])
def test_summary_update_precedes_attention_transfer_window(monkeypatch, verify):
    events = []

    @contextmanager
    def window():
        events.append("attention_start")
        yield
        events.append("attention_finish")

    md = SimpleNamespace(
        num_actual_tokens=4,
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=2 if verify else 1,
        decode=SimpleNamespace(query_len=2 if verify else 1),
    )
    monkeypatch.setattr(sparse_attention, "AscendStep5DSAMetadata", SimpleNamespace)
    monkeypatch.setattr(sparse_attention, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"layer": md}))
    monkeypatch.setattr(sparse_attention, "attention_transfer_window", window, raising=False)
    impl = SimpleNamespace(
        _update_summary_state=lambda *args: events.append("summary"),
        _forward_sparse_gqa_decode=lambda **kwargs: events.append("decode"),
        _forward_sparse_gqa_verify=lambda **kwargs: events.append("verify"),
        _forward_sparse_gqa_prefill=lambda **kwargs: events.append("prefill"),
    )
    tensor = torch.zeros(4, 1, 4)
    sparse_attention.AscendStep5DSAImpl.forward(
        impl,
        SimpleNamespace(layer_name="layer"),
        tensor,
        (tensor, tensor),
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        torch.arange(4),
    )
    assert events == ["summary", "attention_start", "verify" if verify else "decode", "prefill", "attention_finish"]
