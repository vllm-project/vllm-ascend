# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the INT2 decode execution mixin.

The tier-merge numeric path (incl. the empty-segment arms) is device-bound;
its behavioral evidence is the device contract battery (including the
empty-segment arms) and the serve parity legs. At module level we pin the
structural contract (entry-point presence, explicit imports).
"""

import inspect

from vllm_ascend.attention.int2.decode import Int2DecodeMixin


def test_decode_entry_points():
    assert callable(getattr(Int2DecodeMixin, "_forward_int2_decode_threetier", None))


def test_explicit_import_blocks():
    import vllm_ascend.attention.int2.decode as dmod

    assert "from vllm_ascend.attention.int2.ctx import (" in inspect.getsource(dmod)


def test_batched_tnd_packs_each_request_at_its_own_offset(monkeypatch):
    """B >= 2: TND key/value must be packed-varlen (sum of lens, request i's
    rows at the cumulative offset). The pre-fix code allocated max(lens) and
    overwrote rows [0, n) per request, handing every request the LAST
    request's KV."""
    import types

    import torch
    import torch_npu

    H, KV, D = 4, 2, 8
    PH = D // 4
    BS = 16
    SLOTS, CAP = 2, 64

    captured = {}

    def fake_fia(**kwargs):
        captured.update(kwargs)
        return torch.zeros(2, H, D), None

    monkeypatch.setattr(torch_npu, "npu_fused_infer_attention_score", fake_fia)

    class Backend(Int2DecodeMixin):
        _TT_RECENT = 512
        num_heads, num_kv_heads, head_size, scale = H, KV, D, 1.0
        _oscar_R_k = None
        _oscar_R_v = None
        num_blocks = 8
        key_cache = (torch.arange(8 * BS * KV * PH, dtype=torch.uint8) % 7).reshape(8, BS, KV, PH)
        value_cache = (torch.arange(8 * BS * KV * PH, dtype=torch.uint8) % 5).reshape(8, BS, KV, PH)
        k_scale_cache = torch.ones(8, BS, KV)
        k_zero_cache = torch.zeros(8, BS, KV)
        v_scale_cache = torch.ones(8, BS, KV)
        v_zero_cache = torch.zeros(8, BS, KV)
        vllm_config = types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=BS))

        def _ensure_tt(self, device):
            return None

        def _r164_bind_slots(self, *a, **k):
            return None

        def _oscar_layer_ord(self, layer):
            return 0

        def _tt_rid(self, b):
            return f"r{b}"

        def _tt_slot(self, b):
            return b

        def _tt_write_hp(self, *a, **k):
            return None

        def _dequantize_packed_int2(self, packed, s, z, dtype):
            t, hkv, ph = packed.shape
            return packed[:, :, :1].float().expand(t, hkv, D).contiguous()

    # per-slot HP buffers with distinct row markers (slot b row i = 100*b + i)
    _hk = (
        (
            100 * torch.arange(SLOTS, dtype=torch.float32)[:, None, None, None]
            + torch.arange(CAP, dtype=torch.float32)[None, :, None, None]
        )
        .expand(SLOTS, CAP, KV, D)
        .contiguous()
    )
    _hv = 2 * _hk

    be = Backend()
    be._tt_pref = {"r0": 8, "r1": 4}
    be._tt_rcnt = {"r0": 4, "r1": 2}
    be._tt_rcnt_seen = {}
    be._tt_hp_bufs = lambda lid, dev, dt: (_hk, _hv)

    md = types.SimpleNamespace(
        seq_lens_list=[40, 24],
        actual_seq_lengths_q=[1, 1],
        block_tables=torch.arange(2 * 8).reshape(2, 8) % 8,
        attn_mask=None,
    )
    q = torch.arange(2 * H * D, dtype=torch.float32).reshape(2, H, D)
    kr = torch.arange(2 * KV * D, dtype=torch.float32).reshape(2, KV, D)
    vr = torch.arange(2 * KV * D, dtype=torch.float32).reshape(2, KV, D)
    out = torch.zeros(2, H, D)
    layer = types.SimpleNamespace(layer_idx=0)

    be._forward_int2_decode_threetier(q, kr, vr, md, out, layer)

    key = captured["key"]
    kvlen = list(captured["actual_seq_lengths_kv"])
    assert kvlen == [40, 24]
    assert key.shape[0] == sum(kvlen), "TND buffer must be sized sum(lens), not max(lens)"
    # the write loop advances _tt_rcnt (+1 per request) before the gather
    # loop reads it, so the recent window is rcnt+1 wide at gather time
    assert torch.equal(key[0:8], _hk[0, 0:8])  # r0 prefix(8)
    assert torch.equal(key[35:40], _hk[0, 8:13])  # r0 recent(4+1)
    # request 1: rows [40, 64) at its OWN offset -- prefix(4) from slot1 rows 0..3,
    # recent 2+1 rows 4..6 (pre-fix, these values sat at rows [4, 7) of the
    # shared buffer instead, i.e. inside request 0's span)
    assert torch.equal(key[40:44], _hk[1, 0:4])
    assert torch.equal(key[61:64], _hk[1, 4:7])
