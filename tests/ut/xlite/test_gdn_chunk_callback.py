# Copyright (c) 2026 ouyangqing. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the xlite GDN chunk callback (CPU-only).

The callback delegates the pure-prefill GDN recurrent core to the fused
``npu_chunk_gated_delta_rule`` CANN op. Tests exercise the pure-Python
layout conversion, the fallback accounting and the consecutive-failure
self-disable with the CANN op mocked out.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

try:
    import xlite._C  # noqa: F401

    _HAVE_XLITE = True
except Exception:  # pragma: no cover - CI CPU environments have no xlite wheel
    _HAVE_XLITE = False
    import enum

    class _AttnKind(enum.IntEnum):
        MHA = 0
        MLA = 1
        HYBRID = 2
        DSA = 3

    xlite_mod = types.ModuleType("xlite")
    xlite_c = types.ModuleType("xlite._C")
    xlite_c.AttnMHA = _AttnKind.MHA
    xlite_c.AttnMLA = _AttnKind.MLA
    xlite_c.AttnHybrid = _AttnKind.HYBRID
    xlite_c.AttnDSA = _AttnKind.DSA
    xlite_c.AttnMeta = type("AttnMeta", (), {})
    xlite_c.AttnMetaV2 = type("AttnMetaV2", (), {})
    xlite_c.Runtime = MagicMock
    xlite_c.ScoringFuncSoftmax = 0
    xlite_c.ScoringFuncSigmoid = 1
    for _name in (
        "graph_capture_begin",
        "graph_capture_end",
        "graph_destroy",
        "graph_exec",
        "set_gdn_chunk_callback",
    ):
        setattr(xlite_c, _name, MagicMock())
    sys.modules.setdefault("xlite", xlite_mod)
    sys.modules.setdefault("xlite._C", xlite_c)

import vllm_ascend.xlite.xlite as xlite_mod_under_test  # noqa: E402
from vllm_ascend.xlite.xlite import XliteWrapper  # noqa: E402


def _wrapper():
    """XliteWrapper without running __init__ (no xlite runtime on CPU)."""
    w = object.__new__(XliteWrapper)
    w._gdn_chunk_cb_registered = False
    w._gdn_chunk_cb_fails = 0
    w._gdn_chunk_cb_disabled = False
    return w


def test_gdn_chunk_call_passes_native_head_layout(monkeypatch):
    """q/k go to the op at their native key-head count (Nk < Nv, no expansion)."""
    B, H, Hk, K, V, T = 2, 4, 2, 8, 16, 6
    state = torch.zeros(B, H, K, V)
    q = torch.randn(T, Hk * K)
    k = torch.randn(T, Hk * K)
    v = torch.randn(T, H * V)
    beta = torch.randn(T, H)
    g = torch.randn(T, H)
    out = torch.zeros(T, H * V)
    cu_seqlens = torch.tensor([0, 3, 6])

    captured = {}

    def fake_op(q_in, k_in, v_in, *, beta, initial_state, actual_seq_lengths, scale, g):
        captured["q_shape"] = tuple(q_in.shape)
        captured["k_shape"] = tuple(k_in.shape)
        captured["v_shape"] = tuple(v_in.shape)
        captured["beta_shape"] = tuple(beta.shape)
        captured["g_dtype"] = g.dtype
        captured["state_vk"] = initial_state
        captured["asl"] = actual_seq_lengths.tolist()
        captured["scale"] = scale
        o = torch.ones(T, H, V)
        final_state = torch.ones(B, H, V, K)
        return o, final_state

    fake_npu = SimpleNamespace(npu_chunk_gated_delta_rule=fake_op)
    monkeypatch.setattr(xlite_mod_under_test, "torch_npu", fake_npu)
    monkeypatch.setattr(torch.npu, "synchronize", MagicMock(), raising=False)

    w = _wrapper()
    ok = XliteWrapper._gdn_chunk_call(w, q, k, v, beta, g, state, out, cu_seqlens, [3, 3])

    assert ok is True
    # Nk grouping: q/k at native key-head count, no repeat-interleave.
    assert captured["q_shape"] == (T, Hk, K)
    assert captured["k_shape"] == (T, Hk, K)
    assert captured["v_shape"] == (T, H, V)
    assert captured["beta_shape"] == (T, H)
    assert captured["g_dtype"] == torch.float32
    # xlite keeps state as [B, H, K, V]; the op expects [B, H, V, K].
    assert captured["state_vk"].shape == (B, H, V, K)
    assert torch.equal(captured["state_vk"], state.transpose(-1, -2))
    assert captured["asl"] == [3, 3]
    assert captured["scale"] == pytest.approx(K**-0.5)
    # Outputs written back: out in token-major [T, H*V]; state transposed back.
    assert torch.equal(out.view(T, H, V), torch.ones(T, H, V))
    assert torch.equal(state, torch.ones(B, H, K, V))


def test_gdn_chunk_callback_success_keeps_fails_at_zero(monkeypatch):
    w = _wrapper()
    monkeypatch.setattr(XliteWrapper, "_gdn_chunk_call", lambda *a, **k: True)

    ok = XliteWrapper._gdn_chunk_callback(w, None, None, None, None, None, None, None, None, None)

    assert ok is True
    assert w._gdn_chunk_cb_fails == 0
    assert w._gdn_chunk_cb_disabled is False


def test_gdn_chunk_callback_self_disables_after_four_failures(monkeypatch):
    w = _wrapper()

    def boom(*a, **k):
        raise RuntimeError("fused op unavailable")

    monkeypatch.setattr(XliteWrapper, "_gdn_chunk_call", boom)

    for i in range(1, 5):
        ok = XliteWrapper._gdn_chunk_callback(w, None, None, None, None, None, None, None, None, None)
        assert ok is False
        assert w._gdn_chunk_cb_fails == i
        # Self-disable triggers exactly at the 4th consecutive failure.
        assert w._gdn_chunk_cb_disabled is (i >= 4)

    # Once disabled, the callback short-circuits without touching the call.
    calls = {"n": 0}

    def counting_call(*a, **k):
        calls["n"] += 1
        return True

    monkeypatch.setattr(XliteWrapper, "_gdn_chunk_call", counting_call)
    assert XliteWrapper._gdn_chunk_callback(w, None, None, None, None, None, None, None, None, None) is False
    assert calls["n"] == 0
