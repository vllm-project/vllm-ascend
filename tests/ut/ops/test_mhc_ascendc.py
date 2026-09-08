# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity tests: AscendC fused mHC vs the upstream torch decomposition.

Compares, at the real GLM-5.3-Flash shapes (T=8, hidden_size=4096,
hc_mult=4, mix_hc=24, sinkhorn_iters=20, rms_norm_eps=1e-5, hc_eps=1e-6):

* ``hc_pre_ascendc``         vs ``mhc_pre_torch`` (+ fused input RMSNorm)
* ``hc_post_ascendc``        vs ``mhc_post_torch``
* ``fused_post_pre_ascendc`` vs ``mhc_post_torch`` + ``mhc_pre_torch``
* the "non-mHC layer" skip branch (MTP/plain layers must not touch mHC ops)
* the fallback branches (unsupported shapes / ``GLM53_HC_ASCENDC=0``)
"""

from __future__ import annotations

import pytest
import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch

from vllm_ascend.ops import mhc_ascendc as m
from vllm_ascend.utils import enable_custom_op

# GLM-5.3-Flash text_config values the fused kernels were built for.
HIDDEN = 4096
HC_MULT = 4
MIX_HC = (2 + HC_MULT) * HC_MULT  # 24
SINKHORN = 20
RMS_EPS = 1e-5
HC_EPS = 1e-6
T = 8
DEVICE = "npu"

# Task tolerance (the comb's Sinkhorn normalisation order differs between
# the fused kernel and the torch decomposition).
ATOL = 1e-2

enable_custom_op()
torch.npu.set_device(0)


def _fused_ops_loadable() -> bool:
    try:
        return m.probe_available(HIDDEN, HC_MULT)
    except Exception:
        return False


FUSED_OPS = _fused_ops_loadable()


def _diff_stats(got: torch.Tensor, ref: torch.Tensor) -> str:
    got32, ref32 = got.float(), ref.float()
    diff = (got32 - ref32).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (ref32.abs() + 1e-6)).max().item()
    # "how many bf16 rounding steps at the tensor's output scale" — 1 bf16 ulp
    # of the largest |ref| is denom * 2**-7, so a value < 1 means the two
    # implementations only disagree by the final bf16 cast.
    denom = ref32.abs().max().item()
    ulp_scale = denom * 2.0**-7 if denom > 0 else float("nan")
    return (
        f"max|d|={max_abs:.3e} max_rel_elem={max_rel:.3e} "
        f"|d|/(|ref|max*2^-7)={max_abs / ulp_scale:.2f} "
        f"|ref|max={denom:.3e} dtype={got.dtype} shape={tuple(got.shape)}"
    )


def assert_close(label: str, got: torch.Tensor, ref: torch.Tensor, atol: float = ATOL) -> None:
    assert torch.allclose(got.float(), ref.float(), atol=atol, rtol=atol), (
        f"{label}: {m.__name__} disagrees with the torch reference "
        f"({atol=}); {_diff_stats(got, ref)}"
    )


def make_inputs(t: int = T, hidden: int = HIDDEN, hc_mult: int = HC_MULT):
    mix_hc = (2 + hc_mult) * hc_mult
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(t, hc_mult, hidden, generator=gen).to(DEVICE, torch.bfloat16) * 0.5
    hc_fn = torch.randn(mix_hc, hc_mult * hidden, generator=gen).to(DEVICE) * 0.02
    hc_scale = torch.randn(3, generator=gen).to(DEVICE) * 0.05
    hc_base = torch.randn(mix_hc, generator=gen).to(DEVICE) * 0.05
    norm_weight = torch.randn(hidden, generator=gen).to(DEVICE, torch.bfloat16)
    return x, hc_fn.float(), hc_scale.float(), hc_base.float(), norm_weight


def rms_norm_ref(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Bit-identical to patch_triton.py::_mhc_rms_norm (current NPU path)."""
    if w is None:
        return x
    xf = x.float()
    var = xf.square().mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * w.float()).to(x.dtype)


def pre_ref(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    post, comb, layer_input = mhc_pre_torch(
        residual, hc_fn, hc_scale, hc_base, RMS_EPS, HC_EPS, HC_EPS, 2.0, SINKHORN
    )
    return post, comb, rms_norm_ref(layer_input, norm_weight, RMS_EPS)


@pytest.fixture(autouse=True)
def _reset_availability():
    m.reset_availability()
    yield
    m.reset_availability()


def test_pre() -> None:
    x, fn, scale, base, w = make_inputs()

    # With fused input RMSNorm (production path).
    y, post, comb = m.hc_pre_ascendc(
        x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    post_r, comb_r, y_r = pre_ref(x, fn, scale, base, w)
    assert_close("y", y, y_r)
    assert_close("post", post, post_r)
    assert_close("comb", comb, comb_r)
    assert post.shape == (T, HC_MULT, 1), f"post shape {post.shape} != (T, hc, 1)"

    # Without RMSNorm (the model norm is a separate op).
    y2, post2, comb2 = m.hc_pre_ascendc(x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS)
    assert_close("y", y2, pre_ref(x, fn, scale, base)[2])
    assert_close("post", post2, post_r)
    assert_close("comb", comb2, comb_r)

    # Packed x layout [T, hc*d].
    xp = x.reshape(T, HC_MULT * HIDDEN)
    y3, post3, comb3 = m.hc_pre_ascendc(
        xp, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    assert_close("y", y3, y_r)
    assert_close("post", post3, post_r)
    assert_close("comb", comb3, comb_r)

    # post_keepdim=False (raw operator layout).
    _, post4, _ = m.hc_pre_ascendc(
        x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS, post_keepdim=False
    )
    assert post4.shape == (T, HC_MULT), f"post4 shape {post4.shape}"

    # 4-D batched x [1, T, hc, d].
    x4 = x.unsqueeze(0)
    y5, post5, comb5 = m.hc_pre_ascendc(
        x4, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    assert y5.shape == (1, T, HIDDEN), y5.shape
    assert post5.shape == (1, T, HC_MULT, 1), post5.shape
    assert comb5.shape == (1, T, HC_MULT, HC_MULT), comb5.shape
    assert_close("y", y5[0], y_r)
    assert_close("post", post5[0], post_r)
    assert_close("comb", comb5[0], comb_r)
    # After a successful supported-shape call, the op path must have been
    # taken exactly when the fused ops are loadable in this environment.
    assert m._PRE_AVAILABLE is FUSED_OPS, (
        f"_PRE_AVAILABLE={m._PRE_AVAILABLE}, fused ops loadable={FUSED_OPS}"
    )


def test_post() -> None:
    x_branch = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=DEVICE) * 0.5
    residual = torch.randn(T, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=DEVICE) * 0.5
    _, fn, scale, base, _ = make_inputs()
    post_r, comb_r, _ = pre_ref(residual, fn, scale, base)

    # Production gate layout post=[T, hc, 1].
    out = m.hc_post_ascendc(x_branch, residual, post_r, comb_r)
    ref = mhc_post_torch(x_branch, residual, post_r, comb_r)
    assert_close("residual", out, ref)
    assert out.shape == (T, HC_MULT, HIDDEN), out.shape

    # Operator gate layout post=[T, hc].
    out2 = m.hc_post_ascendc(x_branch, residual, post_r.squeeze(-1), comb_r)
    assert_close("residual", out2, ref)

    # Packed residual [T, hc*d].
    out3 = m.hc_post_ascendc(x_branch, residual.reshape(T, -1), post_r, comb_r)
    assert_close("residual", out3, ref.reshape(T, -1))

    # Batched [1, T, ...].
    out4 = m.hc_post_ascendc(
        x_branch.unsqueeze(0),
        residual.unsqueeze(0),
        post_r.unsqueeze(0).squeeze(-1),
        comb_r.unsqueeze(0),
    )
    assert out4.shape == (1, T, HC_MULT, HIDDEN), out4.shape
    assert_close("residual", out4[0], ref)
    assert m._POST_AVAILABLE is FUSED_OPS, (
        f"_POST_AVAILABLE={m._POST_AVAILABLE}, fused ops loadable={FUSED_OPS}"
    )


def test_fused() -> None:
    x_branch = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=DEVICE) * 0.5
    _, fn, scale, base, w = make_inputs()
    residual = torch.randn(T, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=DEVICE) * 0.5
    post, comb, _ = pre_ref(residual, fn, scale, base)

    res_c, post_c, comb_c, layer_input_c = m.fused_post_pre_ascendc(
        x_branch, residual, post, comb, fn, scale, base, SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=w, layer_norm_eps=RMS_EPS,
    )

    ref_res = mhc_post_torch(x_branch, residual, post, comb)
    ref_post, ref_comb, ref_li = pre_ref(ref_res, fn, scale, base, w)
    assert_close("residual", res_c, ref_res)
    assert_close("post", post_c, ref_post)
    assert_close("comb", comb_c, ref_comb)
    assert_close("layer_input", layer_input_c, ref_li)
    assert res_c.shape == (T, HC_MULT, HIDDEN)
    assert post_c.shape == (T, HC_MULT, 1)
    assert comb_c.shape == (T, HC_MULT, HC_MULT)
    assert layer_input_c.shape == (T, HIDDEN)

    # The fused entry must equal the two separate hc_post + hc_pre calls.
    res_s = m.hc_post_ascendc(x_branch, residual, post, comb)
    y_s, post_s, comb_s = m.hc_pre_ascendc(
        res_s, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    assert torch.equal(res_c, res_s)
    assert torch.equal(post_c, post_s)
    assert torch.equal(comb_c, comb_s)
    assert torch.equal(layer_input_c, y_s)


def _layer_forward(
    hidden_states: torch.Tensor,
    *,
    mhc: bool,
    attn_out_fn,
    mlp_fn,
    params: dict,
    calls: dict,
) -> torch.Tensor:
    """Mirror of Glm5NextDecoderLayer.forward's mHC control flow (eager, no SP).

    Verifies the branch structure the AscendC wiring must preserve:
    non-mHC (MTP) layers take the plain path and never touch mHC ops.
    """
    from vllm.model_executor.layers.mhc import hc_contract, hc_expand

    if not mhc:
        x = hidden_states
        x = attn_out_fn(x)
        x = mlp_fn(x)
        return x  # no residual/post/comb plumbing at all

    n = params["n"]
    x = hidden_states
    x = hc_expand(x, n)
    residual = x
    _, post, comb, x = _fake_fused_pre(
        x, params, calls,
        m.hc_pre_ascendc(
            x, params["fn"], params["scale"], params["base"], params["n"],
            SINKHORN, RMS_EPS, HC_EPS,
            norm_weight=params["w"], layer_norm_eps=RMS_EPS,
        ),
    )
    x = attn_out_fn(x)
    res, post, comb, x = m.fused_post_pre_ascendc(
        x, residual, post, comb, params["fn"], params["scale"], params["base"],
        SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=params["w"], layer_norm_eps=RMS_EPS,
    )
    residual = res
    x = mlp_fn(x)
    x = m.hc_post_ascendc(x, residual, post, comb)
    return hc_contract(x, n)


def _fake_fused_pre(x, params, calls, pre_out):
    """hc_pre returns (y, post, comb); the layer re-orders to (res, post, comb, x)."""
    y, post, comb = pre_out
    return x, post, comb, y


def test_skip_branch() -> None:
    calls = {"pre": 0, "fused": 0, "post": 0}
    orig = (m.hc_pre_ascendc, m.hc_post_ascendc, m.fused_post_pre_ascendc)

    def counting_pre(*a, **k):
        calls["pre"] += 1
        return orig[0](*a, **k)

    def counting_post(*a, **k):
        calls["post"] += 1
        return orig[1](*a, **k)

    def counting_fused(*a, **k):
        calls["fused"] += 1
        return orig[2](*a, **k)

    m.hc_pre_ascendc = counting_pre
    m.hc_post_ascendc = counting_post
    m.fused_post_pre_ascendc = counting_fused
    try:
        _, fn, scale, base, w = make_inputs(t=4)
        params = {"n": HC_MULT, "fn": fn, "scale": scale, "base": base, "w": w}

        def attn(t_: torch.Tensor) -> torch.Tensor:
            return t_ * 1.0 + 0.1

        def mlp(t_: torch.Tensor) -> torch.Tensor:
            return t_ * 0.5

        layer_in = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device=DEVICE)

        # non-mHC / MTP layer: plain path, mHC ops must not be entered
        out_plain = _layer_forward(
            layer_in, mhc=False, attn_out_fn=attn, mlp_fn=mlp, params=params, calls=calls
        )
        assert calls == {"pre": 0, "fused": 0, "post": 0}, calls
        ref_plain = mlp(attn(layer_in))
        assert torch.equal(out_plain, ref_plain), "plain (non-mHC) path must be untouched"

        # mHC layer: 1 pre + 1 fused + 1 post entry points; the fused entry
        # internally re-enters hc_post + hc_pre, hence pre/post count 2 each.
        out_mhc = _layer_forward(
            layer_in, mhc=True, attn_out_fn=attn, mlp_fn=mlp, params=params, calls=calls
        )
        assert calls == {"pre": 2, "fused": 1, "post": 2}, calls
        assert out_mhc.shape == layer_in.shape, (out_mhc.shape, layer_in.shape)
        assert m._PRE_AVAILABLE is FUSED_OPS and m._POST_AVAILABLE is FUSED_OPS
    finally:
        m.hc_pre_ascendc, m.hc_post_ascendc, m.fused_post_pre_ascendc = orig
        m.reset_availability()


def test_fallback_unsupported_shapes() -> None:
    # hidden_size not in {4096, 7168}
    x, fn, scale, base, w = make_inputs(t=4, hidden=5120, hc_mult=HC_MULT)
    y, post, comb = m.hc_pre_ascendc(
        x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    post_r, comb_r, y_r = pre_ref(x, fn, scale, base, w)
    assert m._PRE_AVAILABLE is False, "unsupported d must flip availability to False"
    assert_close("y(d=5120)", y, y_r)
    assert_close("post(d=5120)", post, post_r)
    assert_close("comb(d=5120)", comb, comb_r)
    m.reset_availability()

    # hc_mult != 4
    x, fn, scale, base, w = make_inputs(t=4, hidden=HIDDEN, hc_mult=2)
    y, post, comb = m.hc_pre_ascendc(
        x, fn, scale, base, 2, SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    post_r, comb_r, y_r = pre_ref(x, fn, scale, base, w)
    assert m._PRE_AVAILABLE is False
    assert_close("y(hc=2)", y, y_r)
    assert_close("post(hc=2)", post, post_r)
    m.reset_availability()

    # hc_post_mult_value != 2.0 (kernel hard-codes 2.0)
    x, fn, scale, base, w = make_inputs(t=4)
    y, post, comb = m.hc_pre_ascendc(
        x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
        hc_post_mult_value=1.5, norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    post_r, comb_r, y_r = pre_ref(x, fn, scale, base, w)
    assert m._PRE_AVAILABLE is False
    assert_close("y(post_mult=1.5)", y, y_r)


def test_env_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLM53_HC_ASCENDC", "0")
    x, fn, scale, base, w = make_inputs()
    y, post, comb = m.hc_pre_ascendc(
        x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    post_r, comb_r, y_r = pre_ref(x, fn, scale, base, w)
    assert_close("y", y, y_r)
    assert_close("post", post, post_r)
    assert_close("comb", comb, comb_r)
    assert not m.is_available(), "GLM53_HC_ASCENDC=0 must force the torch path"


def test_dispatch_counts() -> None:
    """The fused path must dispatch far fewer eager ops (host-side cost)."""
    if not FUSED_OPS:
        pytest.skip("fused mHC ops not loadable; the torch path dispatches equally")
    from torch.utils._python_dispatch import TorchDispatchMode

    class Counter(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[override]
            self.count += 1
            return func(*args, **(kwargs or {}))

    x, fn, scale, base, w = make_inputs(t=8)

    with Counter() as c_asc:
        m.hc_pre_ascendc(
            x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
            norm_weight=w, layer_norm_eps=RMS_EPS,
        )
    m.reset_availability()
    with Counter() as c_torch:
        pre_ref(x, fn, scale, base, w)
    torch.npu.synchronize()
    assert c_asc.count < c_torch.count, (
        f"fused hc_pre dispatched {c_asc.count} ops, torch path {c_torch.count}"
    )
