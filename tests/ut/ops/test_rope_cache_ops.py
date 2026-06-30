from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm_ascend.ops import rope_cache_ops


def test_mla_prolog_v3_by_cache_uses_true_by_cache_op(monkeypatch):
    calls = {}

    def fake_native_op(**kwargs):
        calls.update(kwargs)
        return "ok"

    def fake_get_torch_npu_op(name):
        assert name == "npu_mla_prolog_v3_by_cache"
        return fake_native_op

    def fake_get_rope_cache(rotary_emb, ref_tensor):
        assert rotary_emb is rotary
        assert ref_tensor is token_x
        return rotary.cos_sin_cache

    monkeypatch.setattr(rope_cache_ops, "_get_torch_npu_op", fake_get_torch_npu_op)
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", fake_get_rope_cache)

    token_x = torch.randn(1, 1)
    weight_dq = torch.randn(1, 1)
    weight_uq_qr = torch.randn(1, 1)
    weight_uk = torch.randn(1, 1)
    weight_dkv_kr = torch.randn(1, 1)
    rmsnorm_gamma_cq = torch.ones(1)
    rmsnorm_gamma_ckv = torch.ones(1)
    positions = torch.tensor([0], dtype=torch.long)
    rotary = SimpleNamespace(cos_sin_cache=torch.randn(4, 2), is_neox_style=True)
    kv_cache = torch.empty(1, 1)
    kr_cache = torch.empty(1, 1)
    cache_index = torch.tensor([0], dtype=torch.long)

    result = rope_cache_ops.mla_prolog_v3_by_cache(
        token_x=token_x,
        weight_dq=weight_dq,
        weight_uq_qr=weight_uq_qr,
        weight_uk=weight_uk,
        weight_dkv_kr=weight_dkv_kr,
        rmsnorm_gamma_cq=rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv=rmsnorm_gamma_ckv,
        positions=positions,
        rotary_emb=rotary,
        kv_cache=kv_cache,
        kr_cache=kr_cache,
        cache_index=cache_index,
        ref_tensor=token_x,
        cache_mode="PA_BSND",
    )

    assert result == "ok"
    assert calls["token_x"] is token_x
    assert calls["positions"] is positions
    assert calls["cos_sin_cache"] is rotary.cos_sin_cache
    assert calls["kv_cache"] is kv_cache
    assert calls["kr_cache"] is kr_cache
    assert calls["cache_index"] is cache_index
    assert calls["cache_mode"] == "PA_BSND"


def test_rotary_siso_by_cache_accepts_valid_1d_positions(monkeypatch):
    calls = {}

    def fake_rope_forward(qk, **kwargs):
        calls["qk"] = qk
        calls.update(kwargs)
        return qk + 1

    def fake_get_rope_cache(rotary_emb, ref_tensor):
        assert rotary_emb is rotary
        calls["cache_ref"] = ref_tensor
        return rotary.cos_sin_cache

    monkeypatch.setattr(rope_cache_ops, "_get_triton_rope_forward_siso", lambda: fake_rope_forward)
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", fake_get_rope_cache)

    x = torch.zeros(2, 1, 4)
    positions = torch.tensor([0, 1], dtype=torch.long)
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 4), rotary_dim=4, is_neox_style=True)

    out = rope_cache_ops.rotary_siso_by_cache(x, positions, rotary)

    assert torch.equal(out, x + 1)
    assert calls["cache_ref"] is calls["qk"]
    assert calls["positions"] is positions
    assert calls["rope_dim"] == 4
    assert calls["is_neox_style"] is True


def test_rotary_siso_by_cache_normalizes_strided_positions(monkeypatch):
    calls = {}

    def fake_rope_forward(qk, **kwargs):
        calls.update(kwargs)
        return qk + 1

    monkeypatch.setattr(rope_cache_ops, "_get_triton_rope_forward_siso", lambda: fake_rope_forward)
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", lambda rotary_emb, ref_tensor: rotary_emb.cos_sin_cache)

    x = torch.zeros(2, 1, 4)
    positions = torch.tensor([0, 99, 1, 100], dtype=torch.int32)[::2]
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 4), rotary_dim=4, is_neox_style=True)

    out = rope_cache_ops.rotary_siso_by_cache(x, positions, rotary)

    assert torch.equal(out, x + 1)
    assert calls["positions"].is_contiguous()
    assert calls["positions"].dtype == torch.int32
    assert torch.equal(calls["positions"], torch.tensor([0, 1], dtype=torch.int32))


def test_rotary_siso_by_cache_rejects_non_1d_positions():
    x = torch.zeros(2, 1, 4)
    positions = torch.zeros(1, 2, dtype=torch.long)
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 4), rotary_dim=4, is_neox_style=True)

    try:
        rope_cache_ops.rotary_siso_by_cache(x, positions, rotary)
    except ValueError as exc:
        assert "expects 1D positions" in str(exc)
    else:
        raise AssertionError("rotary_siso_by_cache should reject non-1D positions")


def test_interleave_rope_by_cache_passes_strided_view_to_triton(monkeypatch):
    calls = {}

    def fake_interleave(qk, **kwargs):
        calls["qk"] = qk
        calls.update(kwargs)
        return qk.contiguous() + 1

    def fake_get_rope_cache(rotary_emb, ref_tensor):
        assert rotary_emb is rotary
        calls["cache_ref"] = ref_tensor
        return rotary.cos_sin_cache

    monkeypatch.setattr(
        rope_cache_ops,
        "_get_torch_npu_op",
        lambda name: None,
    )
    monkeypatch.setattr(rope_cache_ops, "_get_c_ascend_op", lambda name: None)
    monkeypatch.setattr(rope_cache_ops, "_get_triton_interleave_rope_by_cache", lambda: fake_interleave)
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", fake_get_rope_cache)

    base = torch.arange(16, dtype=torch.float32).reshape(2, 1, 1, 8)
    x = base[..., ::2]
    positions = torch.tensor([0, 99, 1, 100], dtype=torch.long)[::2]
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 4), rotary_dim=4, is_neox_style=True)

    out = rope_cache_ops.interleave_rope_by_cache(x, positions, rotary)

    assert calls["qk"].shape == (2, 1, 4)
    assert not calls["qk"].is_contiguous()
    assert calls["cache_ref"] is calls["qk"]
    assert calls["positions"].is_contiguous()
    assert torch.equal(calls["positions"], torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(out, (calls["qk"].contiguous() + 1).reshape(x.shape))


def test_interleave_rope_by_cache_falls_back_to_c_ascend_by_cache(monkeypatch):
    calls = {}

    def fake_native(qk, positions, cos_sin_cache, rope_dim, is_neox_style):
        calls["qk"] = qk
        calls["positions"] = positions
        calls["cos_sin_cache"] = cos_sin_cache
        calls["rope_dim"] = rope_dim
        calls["is_neox_style"] = is_neox_style
        return qk + 1

    def fake_get_c_ascend_op(name):
        assert name == "interleave_rope_by_cache"
        return fake_native

    def fake_get_rope_cache(rotary_emb, ref_tensor):
        assert rotary_emb is rotary
        calls["cache_ref"] = ref_tensor
        return rotary.cos_sin_cache

    monkeypatch.setattr(
        rope_cache_ops,
        "_get_torch_npu_op",
        lambda name: None,
    )
    monkeypatch.setattr(rope_cache_ops, "_get_c_ascend_op", fake_get_c_ascend_op)
    monkeypatch.setattr(
        rope_cache_ops,
        "_get_triton_interleave_rope_by_cache",
        lambda: (_ for _ in ()).throw(AssertionError("Triton should not be used")),
    )
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", fake_get_rope_cache)

    x = torch.zeros(2, 1, 1, 32)
    positions = torch.tensor([0, 99, 1, 100], dtype=torch.int32)[::2]
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 32), rotary_dim=32, is_neox_style=True)

    out = rope_cache_ops.interleave_rope_by_cache(x, positions, rotary)

    assert calls["qk"].shape == (2, 1, 32)
    assert calls["qk"].stride(-1) == 1
    assert calls["cache_ref"] is calls["qk"]
    assert calls["positions"].is_contiguous()
    assert calls["positions"].dtype == torch.int32
    assert torch.equal(calls["positions"], torch.tensor([0, 1], dtype=torch.int32))
    assert calls["cos_sin_cache"] is rotary.cos_sin_cache
    assert calls["rope_dim"] == 32
    assert calls["is_neox_style"] is True
    assert torch.equal(out, (calls["qk"] + 1).reshape(x.shape))


def test_interleave_rope_by_cache_skips_native_for_unaligned_half_rope_dim(monkeypatch):
    calls = {}

    def fake_native(*args, **kwargs):
        raise AssertionError("C native op must not be used when rope_dim / 2 is not 16-aligned")

    def fake_triton(qk, **kwargs):
        calls["qk"] = qk
        calls.update(kwargs)
        return qk.contiguous() + 1

    monkeypatch.setattr(
        rope_cache_ops,
        "_get_torch_npu_op",
        lambda name: None,
    )
    monkeypatch.setattr(rope_cache_ops, "_get_c_ascend_op", lambda name: fake_native)
    monkeypatch.setattr(rope_cache_ops, "_get_triton_interleave_rope_by_cache", lambda: fake_triton)
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", lambda rotary_emb, ref_tensor: rotary_emb.cos_sin_cache)

    x = torch.zeros(2, 1, 1, 16)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 16), rotary_dim=16, is_neox_style=True)

    out = rope_cache_ops.interleave_rope_by_cache(x, positions, rotary)

    assert calls["rope_dim"] == 16
    assert torch.equal(out, (calls["qk"].contiguous() + 1).reshape(x.shape))


def test_interleave_rope_by_cache_falls_back_to_native_by_cache_for_non_neox(monkeypatch):
    calls = {}

    def fake_native(qk, positions, cos_sin_cache, rope_dim, is_neox_style):
        calls["qk"] = qk
        calls["positions"] = positions
        calls["cos_sin_cache"] = cos_sin_cache
        calls["rope_dim"] = rope_dim
        calls["is_neox_style"] = is_neox_style
        return qk + 1

    monkeypatch.setattr(
        rope_cache_ops,
        "_get_torch_npu_op",
        lambda name: None,
    )
    monkeypatch.setattr(rope_cache_ops, "_get_c_ascend_op", lambda name: fake_native)
    monkeypatch.setattr(
        rope_cache_ops,
        "_get_triton_interleave_rope_by_cache",
        lambda: (_ for _ in ()).throw(AssertionError("Triton should not be used")),
    )
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", lambda rotary_emb, ref_tensor: rotary_emb.cos_sin_cache)

    x = torch.zeros(2, 1, 1, 32)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 32), rotary_dim=32, is_neox_style=False)

    out = rope_cache_ops.interleave_rope_by_cache(x, positions, rotary)

    assert calls["qk"].shape == (2, 1, 32)
    assert calls["positions"].dtype == torch.int64
    assert calls["cos_sin_cache"] is rotary.cos_sin_cache
    assert calls["rope_dim"] == 32
    assert calls["is_neox_style"] is False
    assert torch.equal(out, (calls["qk"] + 1).reshape(x.shape))


def test_mla_preprocess_by_cache_normalizes_int32_positions(monkeypatch):
    calls = {}

    def fake_native_op(*args, **kwargs):
        calls["positions"] = args[8]
        calls["cos_sin_cache"] = args[9]
        calls["kwargs"] = kwargs

    def fake_get_c_ascend_op(name):
        assert name == "mla_preprocess_by_cache"
        return fake_native_op

    def fake_get_rope_cache(rotary_emb, ref_tensor):
        assert rotary_emb is rotary
        assert ref_tensor is hidden_states
        return rotary.cos_sin_cache

    monkeypatch.setattr(rope_cache_ops, "_get_c_ascend_op", fake_get_c_ascend_op)
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", fake_get_rope_cache)

    hidden_states = torch.zeros(2, 4)
    wd_qkv = torch.empty(1)
    deq_scale_qkv = torch.empty(1)
    gamma1 = torch.ones(1)
    beta1 = None
    wu_q = torch.empty(1)
    qb_deq_scl = torch.empty(1)
    gamma2 = torch.ones(1)
    positions = torch.tensor([0, 99, 1, 100], dtype=torch.int32)[::2]
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(8, 4), is_neox_style=True)
    W_UK_T = torch.empty(1)
    k_nope = torch.empty(1)
    k_pe = torch.empty(1)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int32)

    rope_cache_ops.mla_preprocess_by_cache(
        hidden_states,
        wd_qkv,
        deq_scale_qkv,
        gamma1,
        beta1,
        wu_q,
        qb_deq_scl,
        gamma2,
        positions,
        rotary,
        W_UK_T,
        k_nope,
        k_pe,
        slot_mapping,
    )

    normalized_positions = calls["positions"]
    assert normalized_positions.dtype == torch.int32
    assert normalized_positions.is_contiguous()
    assert torch.equal(normalized_positions, torch.tensor([0, 1], dtype=torch.int32))
    assert calls["cos_sin_cache"] is rotary.cos_sin_cache
    assert calls["kwargs"]["is_neox_style"] is True


def test_interleave_rope_by_cache_does_not_fallback_to_materialized_native(monkeypatch):
    queried_ops = []

    def fake_get_torch_npu_op(name):
        queried_ops.append(name)
        if name == "npu_interleave_rope_by_cache":
            return None
        if name == "npu_interleave_rope":
            return fake_native_op
        return None

    def fake_native_op(x_arg, cos_arg, sin_arg):
        raise AssertionError("materialized npu_interleave_rope must not be used")

    monkeypatch.setattr(rope_cache_ops, "_get_torch_npu_op", fake_get_torch_npu_op)
    monkeypatch.setattr(rope_cache_ops, "_get_c_ascend_op", lambda name: None)
    monkeypatch.setattr(rope_cache_ops, "_get_triton_interleave_rope_by_cache", lambda: None)

    x = torch.zeros(2, 1, 1, 4)
    positions = torch.tensor([0, 1], dtype=torch.long)
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 4), rotary_dim=4, is_neox_style=True)

    with pytest.raises(RuntimeError, match="true by-cache backend"):
        rope_cache_ops.interleave_rope_by_cache(x, positions, rotary)

    assert queried_ops == ["npu_interleave_rope_by_cache"]


def test_interleave_rope_by_cache_prefers_torch_npu_by_cache(monkeypatch):
    calls = {}

    def fake_torch_npu_op(qk, positions, cos_sin_cache, rope_dim, is_neox_style):
        calls["qk"] = qk
        calls["positions"] = positions
        calls["cos_sin_cache"] = cos_sin_cache
        calls["rope_dim"] = rope_dim
        calls["is_neox_style"] = is_neox_style
        return qk + 1

    def fake_get_torch_npu_op(name):
        assert name == "npu_interleave_rope_by_cache"
        return fake_torch_npu_op

    monkeypatch.setattr(rope_cache_ops, "_get_torch_npu_op", fake_get_torch_npu_op)
    monkeypatch.setattr(
        rope_cache_ops,
        "_get_c_ascend_op",
        lambda name: (_ for _ in ()).throw(AssertionError("C Ascend should not be used before torch_npu")),
    )
    monkeypatch.setattr(
        rope_cache_ops,
        "_get_triton_interleave_rope_by_cache",
        lambda: (_ for _ in ()).throw(AssertionError("Triton should not be used before torch_npu")),
    )
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", lambda rotary_emb, ref_tensor: rotary_emb.cos_sin_cache)

    x = torch.zeros(2, 1, 1, 32)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 32), rotary_dim=32, is_neox_style=False)

    out = rope_cache_ops.interleave_rope_by_cache(x, positions, rotary)

    assert calls["qk"] is x
    assert torch.equal(calls["positions"], positions)
    assert calls["cos_sin_cache"] is rotary.cos_sin_cache
    assert calls["rope_dim"] == 32
    assert calls["is_neox_style"] is False
    assert torch.equal(out, x + 1)


def test_kv_rmsnorm_rope_cache_by_cache_does_not_fallback_to_materialized_native(monkeypatch):
    queried_ops = []

    def fake_get_torch_npu_op(name):
        queried_ops.append(name)
        if name == "npu_kv_rmsnorm_rope_cache_by_cache":
            return None
        if name == "npu_kv_rmsnorm_rope_cache":
            return fake_native_op
        return None

    def fake_native_op(kv_arg, weight_arg, cos_arg, sin_arg, slots_arg, rope_cache_arg, nope_cache_arg, **kwargs):
        raise AssertionError("materialized npu_kv_rmsnorm_rope_cache must not be used")

    monkeypatch.setattr(rope_cache_ops, "_get_torch_npu_op", fake_get_torch_npu_op)
    monkeypatch.setattr(rope_cache_ops, "_get_c_ascend_op", lambda name: None)
    monkeypatch.setattr(rope_cache_ops, "_get_triton_kv_rmsnorm_rope_cache_by_cache", lambda: None)

    kv = torch.zeros(2, 1, 1, 6)
    weight = torch.ones(2)
    positions = torch.tensor([0, 99, 1, 100], dtype=torch.long)[::2]
    slots = torch.tensor([4, 5], dtype=torch.int32)
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 8), is_neox_style=True)
    rope_cache = torch.zeros(8, 1, 1, 4)
    nope_cache = torch.zeros(8, 1, 1, 2)

    with pytest.raises(RuntimeError, match="true by-cache backend"):
        rope_cache_ops.kv_rmsnorm_rope_cache_by_cache(
            kv,
            weight,
            positions,
            rotary,
            slots,
            rope_cache,
            nope_cache,
            epsilon=1e-6,
            cache_mode="PA",
        )

    assert queried_ops == ["npu_kv_rmsnorm_rope_cache_by_cache"]


def test_kv_rmsnorm_rope_cache_by_cache_triton_accepts_int32_slots(monkeypatch):
    calls = {}

    def fake_get_torch_npu_op(name):
        raise AssertionError(f"torch_npu fallback should not be used before Triton: {name}")

    def fake_triton_op(kv, weight, positions, cos_sin_cache, slots, *args, **kwargs):
        calls["positions"] = positions
        calls["slots"] = slots
        calls["cos_sin_cache"] = cos_sin_cache
        calls["kwargs"] = kwargs
        return "ok"

    def fake_get_rope_cache(rotary_emb, ref_tensor):
        assert rotary_emb is rotary
        assert ref_tensor is kv
        return rotary.cos_sin_cache

    monkeypatch.setattr(rope_cache_ops, "_get_torch_npu_op", fake_get_torch_npu_op)
    monkeypatch.setattr(rope_cache_ops, "_get_c_ascend_op", lambda name: None)
    monkeypatch.setattr(rope_cache_ops, "_get_triton_kv_rmsnorm_rope_cache_by_cache", lambda: fake_triton_op)
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", fake_get_rope_cache)

    kv = torch.zeros(2, 1, 1, 6)
    weight = torch.ones(2)
    positions = torch.tensor([0, 1], dtype=torch.long)
    slots = torch.tensor([4, 5], dtype=torch.int32)
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 8), is_neox_style=True)
    rope_cache = torch.zeros(8, 1, 1, 4)
    nope_cache = torch.zeros(8, 1, 1, 2)

    result = rope_cache_ops.kv_rmsnorm_rope_cache_by_cache(
        kv,
        weight,
        positions,
        rotary,
        slots,
        rope_cache,
        nope_cache,
        epsilon=1e-6,
        cache_mode="PA",
    )

    assert result == "ok"
    assert calls["positions"].is_contiguous()
    assert torch.equal(calls["positions"], torch.tensor([0, 1], dtype=torch.long))
    assert calls["slots"] is slots
    assert calls["slots"].dtype == torch.int32
    assert calls["cos_sin_cache"] is rotary.cos_sin_cache
    assert calls["kwargs"]["is_output_kv"] is False


def test_kv_rmsnorm_rope_cache_by_cache_uses_c_ascend_by_cache_before_triton(monkeypatch):
    calls = {}
    expected = ("rope_cache", "nope_cache", "out_rope", "out_nope")

    def fake_c_ascend_op(kv, weight, positions, cos_sin_cache, slots, *args):
        calls["kv"] = kv
        calls["weight"] = weight
        calls["positions"] = positions
        calls["cos_sin_cache"] = cos_sin_cache
        calls["slots"] = slots
        calls["args"] = args
        return expected

    def fake_get_c_ascend_op(name):
        assert name == "kv_rmsnorm_rope_cache_by_cache"
        return fake_c_ascend_op

    def fake_get_rope_cache(rotary_emb, ref_tensor):
        assert rotary_emb is rotary
        assert ref_tensor is kv
        return rotary.cos_sin_cache

    def fail_triton():
        raise AssertionError("Triton fallback should not be used when C by-cache backend is available")

    monkeypatch.setattr(
        rope_cache_ops,
        "_get_torch_npu_op",
        lambda name: (_ for _ in ()).throw(AssertionError("torch_npu fallback should not be used before C Ascend")),
    )
    monkeypatch.setattr(rope_cache_ops, "_get_c_ascend_op", fake_get_c_ascend_op)
    monkeypatch.setattr(rope_cache_ops, "_get_triton_kv_rmsnorm_rope_cache_by_cache", fail_triton)
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", fake_get_rope_cache)

    kv = torch.zeros(2, 1, 1, 48)
    weight = torch.ones(16)
    positions = torch.tensor([0, 1], dtype=torch.long)
    slots = torch.tensor([4, 5], dtype=torch.int32)
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 32), rotary_dim=32, is_neox_style=False)
    rope_cache = torch.zeros(8, 1, 1, 32)
    nope_cache = torch.zeros(8, 1, 1, 16)

    result = rope_cache_ops.kv_rmsnorm_rope_cache_by_cache(
        kv,
        weight,
        positions,
        rotary,
        slots,
        rope_cache,
        nope_cache,
        epsilon=1e-6,
        cache_mode="PA",
        is_output_kv=True,
    )

    assert result == expected
    assert calls["kv"] is kv
    assert calls["weight"] is weight
    assert calls["positions"] is positions
    assert calls["slots"] is slots
    assert calls["cos_sin_cache"] is rotary.cos_sin_cache
    assert calls["args"][-5:] == (1e-6, 32, False, True, False)


def test_kv_rmsnorm_rope_cache_by_cache_uses_c_ascend_with_negative_slots(monkeypatch):
    calls = {}
    expected = ("rope_cache", "nope_cache", "out_rope", "out_nope")

    def fake_c_ascend_op(kv, weight, positions, cos_sin_cache, slots, *args):
        calls["kv"] = kv
        calls["weight"] = weight
        calls["positions"] = positions
        calls["cos_sin_cache"] = cos_sin_cache
        calls["slots"] = slots
        calls["args"] = args
        return expected

    def fake_get_c_ascend_op(name):
        assert name == "kv_rmsnorm_rope_cache_by_cache"
        return fake_c_ascend_op

    def fake_get_rope_cache(rotary_emb, ref_tensor):
        assert rotary_emb is rotary
        assert ref_tensor is kv
        return rotary.cos_sin_cache

    def fail_native(name):
        raise AssertionError(f"torch_npu native backend should not be queried for negative slots: {name}")

    def fail_triton():
        raise AssertionError("Triton fallback should not be used when C by-cache backend is available")

    monkeypatch.setattr(rope_cache_ops, "_get_torch_npu_op", fail_native)
    monkeypatch.setattr(rope_cache_ops, "_get_c_ascend_op", fake_get_c_ascend_op)
    monkeypatch.setattr(rope_cache_ops, "_get_triton_kv_rmsnorm_rope_cache_by_cache", fail_triton)
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", fake_get_rope_cache)

    kv = torch.zeros(2, 1, 1, 48)
    weight = torch.ones(16)
    positions = torch.tensor([0, 1], dtype=torch.long)
    slots = torch.tensor([-1, 5], dtype=torch.int64)
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 32), is_neox_style=False)
    rope_cache = torch.zeros(8, 1, 1, 32)
    nope_cache = torch.zeros(8, 1, 1, 16)

    result = rope_cache_ops.kv_rmsnorm_rope_cache_by_cache(
        kv,
        weight,
        positions,
        rotary,
        slots,
        rope_cache,
        nope_cache,
        epsilon=1e-6,
        cache_mode="PA",
        is_output_kv=True,
        allow_negative_slots=True,
    )

    assert result == expected
    assert calls["kv"] is kv
    assert calls["weight"] is weight
    assert calls["positions"] is positions
    assert calls["slots"] is slots
    assert torch.equal(calls["slots"], torch.tensor([-1, 5], dtype=torch.int64))
    assert calls["cos_sin_cache"] is rotary.cos_sin_cache
    assert calls["args"][-5:] == (1e-6, 32, False, True, False)


def test_kv_rmsnorm_rope_cache_by_cache_uses_torch_npu_by_cache_as_last_fallback(monkeypatch):
    calls = {}
    expected = ("rope_cache", "nope_cache", "out_rope", "out_nope")

    def fake_torch_npu_op(kv, weight, positions, cos_sin_cache, slots, rope_cache, nope_cache, **kwargs):
        calls["kv"] = kv
        calls["weight"] = weight
        calls["positions"] = positions
        calls["cos_sin_cache"] = cos_sin_cache
        calls["slots"] = slots
        calls["rope_cache"] = rope_cache
        calls["nope_cache"] = nope_cache
        calls["kwargs"] = kwargs
        return expected

    def fake_get_torch_npu_op(name):
        assert name == "npu_kv_rmsnorm_rope_cache_by_cache"
        return fake_torch_npu_op

    monkeypatch.setattr(rope_cache_ops, "_get_torch_npu_op", fake_get_torch_npu_op)
    monkeypatch.setattr(rope_cache_ops, "_get_c_ascend_op", lambda name: None)
    monkeypatch.setattr(rope_cache_ops, "_get_triton_kv_rmsnorm_rope_cache_by_cache", lambda: None)
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", lambda rotary_emb, ref_tensor: rotary_emb.cos_sin_cache)

    kv = torch.zeros(2, 1, 1, 48)
    weight = torch.ones(16)
    positions = torch.tensor([0, 1], dtype=torch.long)
    slots = torch.tensor([4, 5], dtype=torch.int32)
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 32), rotary_dim=32, is_neox_style=False)
    rope_cache = torch.zeros(8, 1, 1, 32)
    nope_cache = torch.zeros(8, 1, 1, 16)

    result = rope_cache_ops.kv_rmsnorm_rope_cache_by_cache(
        kv,
        weight,
        positions,
        rotary,
        slots,
        rope_cache,
        nope_cache,
        epsilon=1e-6,
        cache_mode="PA",
        is_output_kv=True,
    )

    assert result == expected
    assert calls["kv"] is kv
    assert calls["weight"] is weight
    assert torch.equal(calls["positions"], positions)
    assert calls["cos_sin_cache"] is rotary.cos_sin_cache
    assert calls["slots"].dtype == torch.int64
    assert torch.equal(calls["slots"], slots.to(torch.int64))
    assert calls["rope_cache"] is rope_cache
    assert calls["nope_cache"] is nope_cache
    assert calls["kwargs"]["epsilon"] == 1e-6
    assert calls["kwargs"]["cache_mode"] == "PA"
    assert calls["kwargs"]["is_output_kv"] is True
    assert calls["kwargs"]["rope_dim"] == 32
    assert calls["kwargs"]["is_neox_style"] is False


def test_kv_rmsnorm_rope_cache_and_interleave_by_cache_uses_c_ascend(monkeypatch):
    calls = {}
    expected = ("q_out", "rope_cache", "nope_cache", "out_rope", "out_nope")

    def fake_c_ascend_op(kv, weight, q, positions, cos_sin_cache, slots, rope_cache, nope_cache, *args):
        calls["kv"] = kv
        calls["weight"] = weight
        calls["q"] = q
        calls["positions"] = positions
        calls["cos_sin_cache"] = cos_sin_cache
        calls["slots"] = slots
        calls["rope_cache"] = rope_cache
        calls["nope_cache"] = nope_cache
        calls["args"] = args
        return expected

    def fake_get_c_ascend_op(name):
        assert name == "kv_rmsnorm_rope_cache_and_interleave_by_cache"
        return fake_c_ascend_op

    def fake_get_rope_cache(rotary_emb, ref_tensor):
        assert rotary_emb is rotary
        assert ref_tensor is kv
        return rotary.cos_sin_cache

    monkeypatch.setattr(rope_cache_ops, "_get_c_ascend_op", fake_get_c_ascend_op)
    monkeypatch.setattr(rope_cache_ops, "get_rope_cache", fake_get_rope_cache)

    q = torch.zeros(2, 2, 32)
    kv = torch.zeros(2, 1, 1, 48)
    weight = torch.ones(16)
    positions = torch.tensor([0, 1], dtype=torch.long)
    slots = torch.tensor([4, 5], dtype=torch.int32)
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(4, 32), is_neox_style=False)
    rope_cache = torch.zeros(8, 1, 1, 32)
    nope_cache = torch.zeros(8, 1, 1, 16)

    result = rope_cache_ops.kv_rmsnorm_rope_cache_and_interleave_by_cache(
        q,
        kv,
        weight,
        positions,
        rotary,
        slots,
        rope_cache,
        nope_cache,
        epsilon=1e-6,
        cache_mode="PA",
        is_output_kv=True,
    )

    assert result == expected
    assert calls["kv"] is kv
    assert calls["weight"] is weight
    assert calls["q"] is q
    assert calls["positions"] is positions
    assert calls["slots"] is slots
    assert calls["cos_sin_cache"] is rotary.cos_sin_cache
    assert calls["rope_cache"] is rope_cache
    assert calls["nope_cache"] is nope_cache
    assert calls["args"][-5:] == (1e-6, 32, False, True, False)


def test_kv_rmsnorm_rope_cache_and_interleave_by_cache_skips_negative_slots(monkeypatch):
    monkeypatch.setattr(
        rope_cache_ops,
        "_get_c_ascend_op",
        lambda name: (_ for _ in ()).throw(AssertionError("fused op should not be queried")),
    )

    result = rope_cache_ops.kv_rmsnorm_rope_cache_and_interleave_by_cache(
        torch.zeros(1, 1, 16),
        torch.zeros(1, 1, 1, 32),
        torch.ones(16),
        torch.tensor([0], dtype=torch.long),
        SimpleNamespace(cos_sin_cache=torch.zeros(4, 16), is_neox_style=False),
        torch.tensor([-1], dtype=torch.int32),
        torch.zeros(8, 1, 1, 16),
        torch.zeros(8, 1, 1, 16),
        epsilon=1e-6,
        allow_negative_slots=True,
    )

    assert result is None


def test_split_qkv_by_cache_backend_helpers_report_missing_backend(monkeypatch):
    monkeypatch.setattr(rope_cache_ops, "_get_vllm_op", lambda name: None)

    assert not rope_cache_ops.has_split_qkv_tp_rmsnorm_rope_by_cache_backend()
    assert not rope_cache_ops.has_split_qkv_rmsnorm_mrope_by_cache_backend()


def test_split_qkv_rmsnorm_mrope_by_cache_fails_fast_without_backend(monkeypatch):
    monkeypatch.setattr(rope_cache_ops, "_get_vllm_op", lambda name: None)

    with pytest.raises(RuntimeError, match="refusing to materialize sin/cos tensors"):
        rope_cache_ops.split_qkv_rmsnorm_mrope_by_cache(
            torch.zeros(1, 4),
            torch.ones(4),
            torch.ones(4),
            torch.tensor([0], dtype=torch.long),
            SimpleNamespace(cos_sin_cache=torch.zeros(4, 4)),
            num_q_heads=1,
            num_kv_heads=1,
            head_size=4,
            eps=1e-6,
            mrope_section=[1, 1, 2],
            is_interleaved=False,
        )


def _require_real_npu():
    pytest.importorskip("torch_npu")
    npu_mod = getattr(torch, "npu", None)
    is_available_attr = getattr(npu_mod, "is_available", None)
    if not callable(is_available_attr):
        pytest.skip("NPU is required")
    is_available = cast(Callable[[], bool], is_available_attr)
    try:
        available = is_available()
    except Exception:
        pytest.skip("NPU is required")
    if available is not True:
        pytest.skip("NPU is required")
    set_device_attr = getattr(npu_mod, "set_device", None)
    if not callable(set_device_attr):
        pytest.skip("NPU is required")
    set_device = cast(Callable[[int], None], set_device_attr)
    set_device(0)


def _make_cos_sin_cache(max_position: int, rope_dim: int, dtype: torch.dtype) -> torch.Tensor:
    angles = torch.linspace(-0.8, 0.8, max_position * (rope_dim // 2), dtype=torch.float32)
    angles = angles.reshape(max_position, rope_dim // 2)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to("npu", dtype=dtype)


def _interleave_rope_reference(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    rope_dim = x.shape[-1]
    half_dim = rope_dim // 2
    cache = cos_sin_cache[positions].to(torch.float32)
    cos, sin = cache[:, :half_dim], cache[:, half_dim:rope_dim]
    even = x[..., 0::2].to(torch.float32)
    odd = x[..., 1::2].to(torch.float32)

    out = torch.empty_like(x)
    out[..., :half_dim] = (even * cos[:, None, :] - odd * sin[:, None, :]).to(x.dtype)
    out[..., half_dim:] = (odd * cos[:, None, :] + even * sin[:, None, :]).to(x.dtype)
    return out


def test_c_interleave_rope_by_cache_matches_native_layout(monkeypatch):
    _require_real_npu()
    rope_cache_ops.clear_rope_cache_op_capability_cache()
    if rope_cache_ops._get_c_ascend_op("interleave_rope_by_cache") is None:
        pytest.skip("C Ascend interleave_rope_by_cache op is unavailable")

    monkeypatch.setattr(rope_cache_ops, "_get_torch_npu_op", lambda name: None)
    monkeypatch.setattr(
        rope_cache_ops,
        "_get_triton_interleave_rope_by_cache",
        lambda: (_ for _ in ()).throw(AssertionError("C backend should be used")),
    )

    dtype = torch.float16
    num_tokens, num_heads, rope_dim = 5, 3, 64
    x = torch.arange(num_tokens * num_heads * rope_dim, device="npu", dtype=torch.float32)
    x = (x.reshape(num_tokens, num_heads, 1, rope_dim) / 127).to(dtype)
    positions = torch.tensor([0, 3, 1, 7, 2], device="npu", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(8, rope_dim, dtype)
    rotary = SimpleNamespace(cos_sin_cache=cos_sin_cache, rotary_dim=rope_dim, is_neox_style=False)

    actual = rope_cache_ops.interleave_rope_by_cache(x, positions, rotary)
    expected = _interleave_rope_reference(x.squeeze(2), positions, cos_sin_cache).reshape_as(x)
    torch.npu.synchronize()

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_c_kv_rmsnorm_rope_cache_by_cache_matches_native_layout(monkeypatch):
    _require_real_npu()
    rope_cache_ops.clear_rope_cache_op_capability_cache()
    if rope_cache_ops._get_c_ascend_op("kv_rmsnorm_rope_cache_by_cache") is None:
        pytest.skip("C Ascend kv_rmsnorm_rope_cache_by_cache op is unavailable")

    monkeypatch.setattr(rope_cache_ops, "_get_torch_npu_op", lambda name: None)
    monkeypatch.setattr(
        rope_cache_ops,
        "_get_triton_kv_rmsnorm_rope_cache_by_cache",
        lambda: (_ for _ in ()).throw(AssertionError("C backend should be used")),
    )

    dtype = torch.float16
    num_tokens, nope_dim, rope_dim = 4, 32, 64
    epsilon = 1e-6
    kv = torch.arange(num_tokens * (nope_dim + rope_dim), device="npu", dtype=torch.float32)
    kv = (kv.reshape(num_tokens, 1, 1, nope_dim + rope_dim) / 113).to(dtype)
    weight = torch.linspace(0.7, 1.3, nope_dim, device="npu", dtype=torch.float32).to(dtype)
    positions = torch.tensor([2, 0, 5, 1], device="npu", dtype=torch.int32)
    slots = torch.tensor([0, 3, 4, 7], device="npu", dtype=torch.int32)
    cos_sin_cache = _make_cos_sin_cache(8, rope_dim, dtype)
    rotary = SimpleNamespace(cos_sin_cache=cos_sin_cache, rotary_dim=rope_dim, is_neox_style=False)
    kv_cache_rope = torch.full((2, 4, 1, rope_dim), -9, device="npu", dtype=dtype)
    kv_cache_nope = torch.full((2, 4, 1, nope_dim), -9, device="npu", dtype=dtype)

    cache_rope, cache_nope, out_rope, out_nope = rope_cache_ops.kv_rmsnorm_rope_cache_by_cache(
        kv,
        weight,
        positions,
        rotary,
        slots,
        kv_cache_rope,
        kv_cache_nope,
        epsilon=epsilon,
        is_output_kv=True,
    )
    torch.npu.synchronize()

    nope_in = kv[:, 0, 0, :nope_dim].to(torch.float32)
    rstd = torch.rsqrt(nope_in.square().mean(dim=-1, keepdim=True) + epsilon)
    expected_nope = (nope_in * weight.to(torch.float32) * rstd).to(dtype)
    rope_in = kv[:, 0, 0, nope_dim:]
    expected_rope = _interleave_rope_reference(rope_in[:, None, :], positions, cos_sin_cache).squeeze(1)
    expected_out_rope = expected_rope.reshape_as(out_rope)
    expected_out_nope = expected_nope.reshape_as(out_nope)
    flat_cache_rope = kv_cache_rope.reshape(-1, 1, rope_dim)[:, 0, :]
    flat_cache_nope = kv_cache_nope.reshape(-1, 1, nope_dim)[:, 0, :]

    torch.testing.assert_close(out_rope, expected_out_rope, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_nope, expected_out_nope, rtol=1e-3, atol=1e-3)
    slot_indices = slots.cpu()
    torch.testing.assert_close(
        cache_rope.reshape(-1, 1, rope_dim)[slot_indices, 0],
        expected_rope,
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        cache_nope.reshape(-1, 1, nope_dim)[slot_indices, 0],
        expected_nope,
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        flat_cache_rope[slot_indices],
        expected_rope,
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        flat_cache_nope[slot_indices],
        expected_nope,
        rtol=1e-3,
        atol=1e-3,
    )


def test_c_kv_rmsnorm_rope_cache_and_interleave_by_cache_matches_native_layout():
    _require_real_npu()
    rope_cache_ops.clear_rope_cache_op_capability_cache()
    if rope_cache_ops._get_c_ascend_op("kv_rmsnorm_rope_cache_and_interleave_by_cache") is None:
        pytest.skip("C Ascend kv_rmsnorm_rope_cache_and_interleave_by_cache op is unavailable")

    dtype = torch.float16
    num_tokens, num_heads, nope_dim, rope_dim = 4, 2, 32, 64
    epsilon = 1e-6
    q = torch.arange(num_tokens * num_heads * rope_dim, device="npu", dtype=torch.float32)
    q = (q.reshape(num_tokens, num_heads, rope_dim) / 131).to(dtype)
    kv = torch.arange(num_tokens * (nope_dim + rope_dim), device="npu", dtype=torch.float32)
    kv = (kv.reshape(num_tokens, 1, 1, nope_dim + rope_dim) / 113).to(dtype)
    weight = torch.linspace(0.7, 1.3, nope_dim, device="npu", dtype=torch.float32).to(dtype)
    positions = torch.tensor([2, 0, 5, 1], device="npu", dtype=torch.int64)
    slots = torch.tensor([0, 3, 4, 7], device="npu", dtype=torch.int32)
    cos_sin_cache = _make_cos_sin_cache(8, rope_dim, dtype)
    rotary = SimpleNamespace(cos_sin_cache=cos_sin_cache, rotary_dim=rope_dim, is_neox_style=False)
    kv_cache_rope = torch.full((2, 4, 1, rope_dim), -9, device="npu", dtype=dtype)
    kv_cache_nope = torch.full((2, 4, 1, nope_dim), -9, device="npu", dtype=dtype)

    q_out, cache_rope, cache_nope, out_rope, out_nope = rope_cache_ops.kv_rmsnorm_rope_cache_and_interleave_by_cache(
        q,
        kv,
        weight,
        positions,
        rotary,
        slots,
        kv_cache_rope,
        kv_cache_nope,
        epsilon=epsilon,
        is_output_kv=True,
    )
    torch.npu.synchronize()

    nope_in = kv[:, 0, 0, :nope_dim].to(torch.float32)
    rstd = torch.rsqrt(nope_in.square().mean(dim=-1, keepdim=True) + epsilon)
    expected_nope = (nope_in * weight.to(torch.float32) * rstd).to(dtype)
    rope_in = kv[:, 0, 0, nope_dim:]
    expected_q = _interleave_rope_reference(q, positions, cos_sin_cache)
    expected_rope = _interleave_rope_reference(rope_in[:, None, :], positions, cos_sin_cache).squeeze(1)
    slot_indices = slots.cpu()

    torch.testing.assert_close(q_out, expected_q, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_rope, expected_rope.reshape_as(out_rope), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_nope, expected_nope.reshape_as(out_nope), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        cache_rope.reshape(-1, 1, rope_dim)[slot_indices, 0],
        expected_rope,
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        cache_nope.reshape(-1, 1, nope_dim)[slot_indices, 0],
        expected_nope,
        rtol=1e-3,
        atol=1e-3,
    )
