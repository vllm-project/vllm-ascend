#!/usr/bin/env python3
"""Numerical parity test: AscendC fused mHC vs the upstream torch decomposition.

Compares, at the real GLM-5.3-Flash shapes (T=8, hidden_size=4096,
hc_mult=4, mix_hc=24, sinkhorn_iters=20, rms_norm_eps=1e-5, hc_eps=1e-6):

* ``hc_pre_ascendc``        vs ``mhc_pre_torch`` (+ fused input RMSNorm)
* ``hc_post_ascendc``       vs ``mhc_post_torch``
* ``fused_post_pre_ascendc`` vs ``mhc_post_torch`` + ``mhc_pre_torch``
* the "non-mHC layer" skip branch (MTP/plain layers must not touch mHC ops)
* the fallback branches (unsupported shapes / ``GLM53_HC_ASCENDC=0``)

Run on one card:  ASCEND_RT_VISIBLE_DEVICES=0 python3 test_mhc_ascendc.py
"""

from __future__ import annotations

import os
import sys

# Run from any checkout location; falls through to the installed package
# when the repo layout is absent.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if os.path.isdir(os.path.join(_REPO_ROOT, "vllm_ascend")) and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

torch.manual_seed(0)

import torch_npu  # noqa: E402

from vllm_ascend.utils import enable_custom_op  # noqa: E402

enable_custom_op()
torch.npu.set_device(0)

from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch  # noqa: E402

from vllm_ascend.ops import mhc_ascendc as m  # noqa: E402

# Standard GLM-5.3-Flash text-config constants (inlined so the test is
# self-contained and runs on any machine / in CI).
DEVICE = "npu"

HIDDEN = 4096
HC_MULT = 4
MIX_HC = (2 + HC_MULT) * HC_MULT  # 24
SINKHORN = 20
RMS_EPS = 1e-5
HC_EPS = 1e-6
T = 8

ATOL = 1e-2  # task tolerance (comb's Sinkhorn normalisation order differs)

_PASS: list[str] = []
_FAIL: list[str] = []


def _stats(name: str, got: torch.Tensor, ref: torch.Tensor) -> float:
    got32, ref32 = got.float(), ref.float()
    diff = (got32 - ref32).abs()
    max_abs = diff.max().item()
    denom = ref32.abs().max().item()
    max_rel = (diff / (ref32.abs() + 1e-6)).max().item()
    # "how many bf16 rounding steps at the tensor's output scale" — 1 bf16 ulp
    # of the largest |ref| is denom * 2**-7, so a value < 1 means the two
    # implementations only disagree by the final bf16 cast.
    ulp_scale = denom * 2.0**-7 if denom > 0 else float("nan")
    print(
        f"    {name:<10} max|Δ|={max_abs:.3e}  max_rel_elem={max_rel:.3e}  "
        f"|Δ|/(|ref|max·2^-7)={max_abs / ulp_scale:.2f}  |ref|max={denom:.3e}  "
        f"dtype={got.dtype} shape={tuple(got.shape)}"
    )
    return max_abs


def _assert_close(label: str, got: torch.Tensor, ref: torch.Tensor, atol: float = ATOL) -> None:
    max_abs = _stats(label, got, ref)
    if not torch.allclose(got.float(), ref.float(), atol=atol, rtol=atol):
        _FAIL.append(f"{label}: max|Δ|={max_abs:.3e} > atol {atol}")
        print(f"    !! FAIL {label}")
        return
    _PASS.append(f"{label} (max|Δ|={max_abs:.3e})")


def make_inputs(t: int = T, hidden: int = HIDDEN, hc_mult: int = HC_MULT, packed: bool = False):
    mix_hc = (2 + hc_mult) * hc_mult
    x = torch.randn(t, hc_mult, hidden, dtype=torch.bfloat16, device=DEVICE) * 0.5
    if packed:
        x = x.reshape(t, hc_mult * hidden)
    hc_fn = (torch.randn(mix_hc, hc_mult * hidden, dtype=torch.float32, device=DEVICE) * 0.02)
    hc_scale = torch.randn(3, dtype=torch.float32, device=DEVICE) * 0.05
    hc_base = torch.randn(mix_hc, dtype=torch.float32, device=DEVICE) * 0.05
    norm_weight = torch.randn(hidden, dtype=torch.bfloat16, device=DEVICE)
    return x, hc_fn, hc_scale, hc_base, norm_weight


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


# --------------------------------------------------------------------------
def test_pre() -> None:
    print("\n[1] hc_pre_ascendc vs mhc_pre_torch  (T=8, hidden=4096, hc_mult=4)")
    m.reset_availability()
    x, fn, scale, base, w = make_inputs()

    print("  -- with fused input RMSNorm (production path)")
    y, post, comb = m.hc_pre_ascendc(
        x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    post_r, comb_r, y_r = pre_ref(x, fn, scale, base, w)
    _assert_close("y", y, y_r)
    _assert_close("post", post, post_r)
    _assert_close("comb", comb, comb_r)
    assert post.shape == (T, HC_MULT, 1), f"post shape {post.shape} != (T, hc, 1)"

    print("  -- without RMSNorm (A3-style: model norm is a separate op)")
    y2, post2, comb2 = m.hc_pre_ascendc(x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS)
    _assert_close("y", y2, pre_ref(x, fn, scale, base)[2])
    _assert_close("post", post2, post_r)
    _assert_close("comb", comb2, comb_r)
    assert post2.shape == (T, HC_MULT, 1)

    print("  -- packed x layout [T, hc*d] (A3 calling convention)")
    xp = x.reshape(T, HC_MULT * HIDDEN)
    y3, post3, comb3 = m.hc_pre_ascendc(
        xp, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS, norm_weight=w, layer_norm_eps=RMS_EPS
    )
    _assert_close("y", y3, y_r)
    _assert_close("post", post3, post_r)
    _assert_close("comb", comb3, comb_r)

    print("  -- post_keepdim=False (raw operator layout)")
    _, post4, _ = m.hc_pre_ascendc(
        x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS, post_keepdim=False
    )
    assert post4.shape == (T, HC_MULT), f"post4 shape {post4.shape}"

    print("  -- 4-D batched x [1, T, hc, d]")
    x4 = x.unsqueeze(0)
    y5, post5, comb5 = m.hc_pre_ascendc(
        x4, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS, norm_weight=w, layer_norm_eps=RMS_EPS
    )
    assert y5.shape == (1, T, HIDDEN), y5.shape
    assert post5.shape == (1, T, HC_MULT, 1), post5.shape
    assert comb5.shape == (1, T, HC_MULT, HC_MULT), comb5.shape
    _assert_close("y", y5[0], y_r)
    _assert_close("post", post5[0], post_r)
    _assert_close("comb", comb5[0], comb_r)
    assert m._PRE_AVAILABLE is True, "AscendC hc_pre was not used"


def test_post() -> None:
    print("\n[2] hc_post_ascendc vs mhc_post_torch")
    m.reset_availability()
    x_branch = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=DEVICE) * 0.5
    residual = torch.randn(T, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=DEVICE) * 0.5
    _, fn, scale, base, _ = make_inputs()
    post_r, comb_r, _ = pre_ref(residual, fn, scale, base)

    print("  -- production gate layout post=[T, hc, 1]")
    out = m.hc_post_ascendc(x_branch, residual, post_r, comb_r)
    ref = mhc_post_torch(x_branch, residual, post_r, comb_r)
    _assert_close("residual", out, ref)
    assert out.shape == (T, HC_MULT, HIDDEN), out.shape

    print("  -- operator gate layout post=[T, hc]")
    out2 = m.hc_post_ascendc(x_branch, residual, post_r.squeeze(-1), comb_r)
    _assert_close("residual", out2, ref)

    print("  -- packed residual [T, hc*d]")
    out3 = m.hc_post_ascendc(x_branch, residual.reshape(T, -1), post_r, comb_r)
    _assert_close("residual", out3, ref.reshape(T, -1))

    print("  -- batched [1, T, ...]")
    out4 = m.hc_post_ascendc(
        x_branch.unsqueeze(0),
        residual.unsqueeze(0),
        post_r.unsqueeze(0).squeeze(-1),
        comb_r.unsqueeze(0),
    )
    assert out4.shape == (1, T, HC_MULT, HIDDEN), out4.shape
    _assert_close("residual", out4[0], ref)
    assert m._POST_AVAILABLE is True, "AscendC hc_post was not used"


def test_fused() -> None:
    print("\n[3] fused_post_pre_ascendc vs mhc_post_torch + mhc_pre_torch")
    m.reset_availability()
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
    _assert_close("residual", res_c, ref_res)
    _assert_close("post", post_c, ref_post)
    _assert_close("comb", comb_c, ref_comb)
    _assert_close("layer_input", layer_input_c, ref_li)
    assert res_c.shape == (T, HC_MULT, HIDDEN)
    assert post_c.shape == (T, HC_MULT, 1)
    assert comb_c.shape == (T, HC_MULT, HC_MULT)
    assert layer_input_c.shape == (T, HIDDEN)

    print("  -- equals separate hc_post + hc_pre calls")
    res_s = m.hc_post_ascendc(x_branch, residual, post, comb)
    y_s, post_s, comb_s = m.hc_pre_ascendc(
        res_s, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    assert torch.equal(res_c, res_s)
    assert torch.equal(post_c, post_s)
    assert torch.equal(comb_c, comb_s)
    assert torch.equal(layer_input_c, y_s)
    print("    bitwise identical to the two separate entries")


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
    print("\n[4] non-mHC (MTP) layer skip branch")
    m.reset_availability()
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
        assert m._PRE_AVAILABLE is True and m._POST_AVAILABLE is True
        print(f"    non-mHC layer mHC calls: 0; mHC layer entry points: "
              f"pre=1, fused=1, post=1 (calls counted incl. fused's internal post+pre: {calls})")
    finally:
        m.hc_pre_ascendc, m.hc_post_ascendc, m.fused_post_pre_ascendc = orig
        m.reset_availability()


def test_fallback_unsupported_shapes() -> None:
    print("\n[5] fallback on shapes outside the operator envelope")
    m.reset_availability()

    # hidden_size not in {4096, 7168}
    x, fn, scale, base, w = make_inputs(t=4, hidden=5120, hc_mult=HC_MULT)
    y, post, comb = m.hc_pre_ascendc(
        x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    post_r, comb_r, y_r = pre_ref(x, fn, scale, base, w)
    assert m._PRE_AVAILABLE is False, "unsupported d must flip availability to False"
    _assert_close("y(d=5120)", y, y_r)
    _assert_close("post(d=5120)", post, post_r)
    _assert_close("comb(d=5120)", comb, comb_r)
    print("    d=5120 -> torch fallback, numerics identical")
    m.reset_availability()

    # hc_mult != 4
    x, fn, scale, base, w = make_inputs(t=4, hidden=HIDDEN, hc_mult=2)
    y, post, comb = m.hc_pre_ascendc(
        x, fn, scale, base, 2, SINKHORN, RMS_EPS, HC_EPS,
        norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    post_r, comb_r, y_r = pre_ref(x, fn, scale, base, w)
    assert m._PRE_AVAILABLE is False
    _assert_close("y(hc=2)", y, y_r)
    _assert_close("post(hc=2)", post, post_r)
    print("    hc_mult=2 -> torch fallback, numerics identical")
    m.reset_availability()

    # hc_post_mult_value != 2.0 (kernel hard-codes 2.0)
    x, fn, scale, base, w = make_inputs(t=4)
    y, post, comb = m.hc_pre_ascendc(
        x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
        hc_post_mult_value=1.5, norm_weight=w, layer_norm_eps=RMS_EPS,
    )
    post_r, comb_r, y_r = pre_ref(x, fn, scale, base, w)
    assert m._PRE_AVAILABLE is False
    _assert_close("y(post_mult=1.5)", y, y_r)
    print("    hc_post_mult_value=1.5 -> torch fallback (kernel pins 2.0)")
    m.reset_availability()


def test_env_kill_switch() -> None:
    print("\n[6] GLM53_HC_ASCENDC=0 kill switch")
    m.reset_availability()
    os.environ["GLM53_HC_ASCENDC"] = "0"
    try:
        x, fn, scale, base, w = make_inputs()
        y, post, comb = m.hc_pre_ascendc(
            x, fn, scale, base, HC_MULT, SINKHORN, RMS_EPS, HC_EPS,
            norm_weight=w, layer_norm_eps=RMS_EPS,
        )
        post_r, comb_r, y_r = pre_ref(x, fn, scale, base, w)
        _assert_close("y", y, y_r)
        _assert_close("post", post, post_r)
        _assert_close("comb", comb, comb_r)
        print(f"    is_available()={m.is_available()} (torch path forced, no NPU op call)")
    finally:
        os.environ.pop("GLM53_HC_ASCENDC", None)
        m.reset_availability()


def test_dispatch_counts() -> None:
    print("\n[7] eager dispatch count per hc_pre (host-side cost proxy)")
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except Exception as exc:  # pragma: no cover
        print(f"    skipped: {exc!r}")
        return

    class Counter(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[override]
            self.count += 1
            return func(*args, **(kwargs or {}))

    m.reset_availability()
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
    print(f"    AscendC hc_pre : {c_asc.count} dispatched ops (1 fused kernel + 1 rms_norm)")
    print(f"    torch hc_pre   : {c_torch.count} dispatched ops (Sinkhorn loop included)")
    print(f"    reduction      : {c_torch.count / max(c_asc.count, 1):.1f}x fewer dispatches")


def main() -> int:
    print(f"config: hidden={HIDDEN} hc_mult={HC_MULT} mix_hc={MIX_HC} "
          f"sinkhorn={SINKHORN} rms_eps={RMS_EPS} hc_eps={HC_EPS} T={T}")
    print(f"fused ops registered: {m.is_available()}  probe: {m.probe_available(HIDDEN, HC_MULT)}")
    m.reset_availability()

    test_pre()
    test_post()
    test_fused()
    test_skip_branch()
    test_fallback_unsupported_shapes()
    test_env_kill_switch()
    test_dispatch_counts()

    print("\n================ summary ================")
    print(f"ascendc op used: pre={m._PRE_AVAILABLE is not False}, post={m._POST_AVAILABLE is not False}")
    for line in _PASS:
        print(f"  PASS {line}")
    for line in _FAIL:
        print(f"  FAIL {line}")
    print(f"{len(_PASS)} passed, {len(_FAIL)} failed")
    torch.npu.synchronize()
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
