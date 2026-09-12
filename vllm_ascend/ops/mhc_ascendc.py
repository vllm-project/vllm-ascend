# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AscendC fused mHC (multi-hyper-connection) operators for GLM-5.3-Flash.

The upstream/fallback mHC implementation
(``vllm.model_executor.kernels.mhc.torch``) decomposes every hc_pre into
~340 eager ops (the 20-round Sinkhorn loop alone accounts for 117 dispatches),
which is the single largest host-side bottleneck of this model on Ascend
(~43% of host time in eager mode).  ``vllm_ascend_C`` ships the fused
kernels ``npu_hc_pre_v2`` (mix projection + RMS-normalize + pre/post/comb
gates + 20-round Sinkhorn + stream contraction, one launch) and
``npu_hc_post`` (deferred residual combination, one launch); this module is
the thin, validating, self-falling-back bridge between the two contracts.

Contract notes (verified against ``vllm_ascend/csrc/torch_binding.cpp``):

``npu_hc_pre_v2(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters,
norm_eps, hc_eps) -> (y, post, comb)``

* ``x``            bf16, 3-D ``[T, hc, d]`` or 4-D ``[B, S, hc, d]``
* ``hc_fn``        fp32, ``[(2 + hc) * hc, hc * d]`` = ``[24, hc * d]``
* ``hc_scale``     fp32, ``[3]``
* ``hc_base``      fp32, ``[(2 + hc) * hc]`` = ``[24]``
* ``y``            bf16, ``[T, d]`` / ``[B, S, d]`` -- the *contracted*
  layer input (``sum_i pre_i * residual_i``), NOT the packed
  ``[T, hc * d]`` layout: only the mHC residual stream stays packed.
* ``post``         fp32, ``[T, hc]`` / ``[B, S, hc]``
* ``comb``         fp32, ``[T, hc, hc]`` / ``[B, S, hc, hc]``

Kernel-enforced limits (TORCH_CHECK in op_host):

* ``hc_mult == 4`` (``HC_PRE_HC_LIMIT``, our config is exactly at the limit)
* ``d in {4096, 7168}`` (``HC_PRE_D_LIMIT`` / ``_EXTEND``)
* ``hc_fn.shape[0] == 24`` (``HC_PRE_MIX_HC_LIMIT``)
* ``hc_post_mult_value`` is fixed to 2.0 inside the kernel

``npu_hc_post(x, residual, post, comb) -> out`` takes ``[B, S, d]`` /
``[B, S, hc, d]`` / ``[B, S, hc]`` / ``[B, S, hc, hc]`` and returns
``[B, S, hc, d]`` (same dtype as ``residual``).

The wrappers below additionally accept the *packed* residual layout
(``[..., hc * d]``) and, by default, hand the ``post`` mix back with the
trailing singleton dim restored (``[..., hc, 1]``) so that the return
contract matches ``vllm.model_executor.kernels.mhc.torch.mhc_pre_torch``
bit-for-bit in shape -- that is what the vendored
``Glm5NextDecoderLayer``/``MHCPreOp``/``MHCFusedPostPreOp`` plumbing feeds
back into ``mhc_post_torch`` (``post_term = post[..., None] * x.unsqueeze(-2)``).
Set ``post_keepdim=False`` to get the raw operator layout instead.

Set ``GLM53_HC_ASCENDC=0`` to force the torch fallback for the whole module.
"""

from __future__ import annotations

import os
import sys
from typing import NamedTuple

import torch

try:  # torch_npu is a hard requirement on Ascend, but keep the module importable off-device
    import torch_npu  # noqa: F401

    _TORCH_NPU_OK = True
except Exception:  # pragma: no cover - non-NPU host (CPU import / docs build)
    _TORCH_NPU_OK = False

# Operator-enforced limits (mirrors csrc/torch_binding.cpp).
HC_PRE_HC_LIMIT = 4
HC_PRE_D_LIMIT = 4096
HC_PRE_D_LIMIT_EXTEND = 7168
HC_PRE_MIX_HC_LIMIT = 24
# aclnnHcPost / aclnnHcPre bake the post-mix multiplier in; a different
# value from the config would silently change numerics, so it is validated.
KERNEL_HC_POST_MULT_VALUE = 2.0

_HC_SUPPORTED_D = (HC_PRE_D_LIMIT, HC_PRE_D_LIMIT_EXTEND)

_ENV_FLAG = "GLM53_HC_ASCENDC"
_LOG_PREFIX = "[mhc-ascendc]"


class HCPreOutput(NamedTuple):
    y: torch.Tensor  # bf16 [..., d] contracted layer input
    post: torch.Tensor  # fp32 [..., hc] (or [..., hc, 1] with post_keepdim)
    comb: torch.Tensor  # fp32 [..., hc, hc]


class HCFusedPostPreOutput(NamedTuple):
    residual: torch.Tensor  # bf16 [..., hc, d] updated residual stream
    post: torch.Tensor  # fp32 [..., hc] (or [..., hc, 1] with post_keepdim)
    comb: torch.Tensor  # fp32 [..., hc, hc]
    layer_input: torch.Tensor  # bf16 [..., d] contracted layer input


# --------------------------------------------------------------------------
# Availability plumbing (mirrors ops/triton/kda/kda.py's AscendC pattern)
# --------------------------------------------------------------------------
# None = never tried, True = operator working, False = operator broken
# (fall back from now on).
_PRE_AVAILABLE: bool | None = None
_POST_AVAILABLE: bool | None = None
_OPS_REGISTERED: bool | None = None


def _env_enabled() -> bool:
    """``GLM53_HC_ASCENDC`` gates the whole module (default: enabled)."""
    raw = os.getenv(_ENV_FLAG)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "off", "no")


def _ensure_ops_registered() -> bool:
    """Import ``vllm_ascend_C`` once and check both ops are registered."""
    global _OPS_REGISTERED
    if _OPS_REGISTERED is not None:
        return _OPS_REGISTERED
    if not _env_enabled() or not _TORCH_NPU_OK:
        _OPS_REGISTERED = False
        return _OPS_REGISTERED
    try:
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()
    except Exception:  # pragma: no cover - already enabled / vllm missing
        pass
    try:
        import vllm_ascend.vllm_ascend_C  # noqa: F401

        ns = torch.ops._C_ascend
        _OPS_REGISTERED = hasattr(ns, "npu_hc_pre_v2") and hasattr(ns, "npu_hc_post")
    except Exception:
        _OPS_REGISTERED = False
    return _OPS_REGISTERED


def reset_availability() -> None:
    """Forget the cached probe results (used by tests)."""
    global _PRE_AVAILABLE, _POST_AVAILABLE, _OPS_REGISTERED
    _PRE_AVAILABLE = None
    _POST_AVAILABLE = None
    _OPS_REGISTERED = None


def is_available() -> bool:
    """True when the fused operators are usable on the current device."""
    return _env_enabled() and _ensure_ops_registered()


def probe_available(hidden_size: int = HC_PRE_D_LIMIT, hc_mult: int = HC_PRE_HC_LIMIT) -> bool:
    """Run a minimal pre+post on the current device and report success.

    Allocates one token worth of state; intended for tests / startup logging,
    not for the hot path (the hot path probes lazily on first call instead).
    """
    if not is_available():
        return False
    device = torch.device("npu", torch.npu.current_device())
    try:
        x = torch.zeros(1, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
        hc_fn = torch.zeros(HC_PRE_MIX_HC_LIMIT, hc_mult * hidden_size, dtype=torch.float32, device=device)
        hc_scale = torch.zeros(3, dtype=torch.float32, device=device)
        hc_base = torch.zeros(HC_PRE_MIX_HC_LIMIT, dtype=torch.float32, device=device)
        y, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(x, hc_fn, hc_scale, hc_base, hc_mult, 1, 1e-6, 1e-6)
        torch.ops._C_ascend.npu_hc_post(y.view(1, 1, hidden_size), x.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0))
        torch.npu.synchronize()
        return True
    except Exception as op_err:
        print(
            f"{_LOG_PREFIX} probe failed, fused mHC unusable: {op_err!r:.600}",
            flush=True,
            file=sys.stderr,
        )
        return False


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------
def _check_hc_mult(hc_mult: int) -> None:
    if hc_mult != HC_PRE_HC_LIMIT:
        raise ValueError(f"{_LOG_PREFIX} operator only supports hc_mult={HC_PRE_HC_LIMIT}, got {hc_mult}")


def _check_param_dtypes(hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor) -> None:
    if hc_fn.dtype != torch.float32 or hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32:
        raise ValueError(
            f"{_LOG_PREFIX} hc_fn/hc_scale/hc_base must be float32, got {hc_fn.dtype}/{hc_scale.dtype}/{hc_base.dtype}"
        )


def _streams_d(x: torch.Tensor, hc_mult: int) -> int:
    d = x.shape[-1]
    if hc_mult * d != x.shape[-2] * x.shape[-1]:
        raise ValueError(f"{_LOG_PREFIX} x shape {tuple(x.shape)} is inconsistent with hc_mult={hc_mult}")
    return d


def _stream_layout(x: torch.Tensor, hc_mult: int, name: str) -> tuple[torch.Tensor, int, tuple[int, ...]]:
    """Return (x as [..., hc_mult, d], d, outer_shape), no envelope check.

    Accepts both the packed residual layout ``[..., hc_mult * d]`` (what
    ``hc_expand``+``view`` produce in A3) and the stream layout
    ``[..., hc_mult, d]`` (what our vendored ``hc_expand`` produces).
    """
    if x.dim() < 2:
        raise ValueError(f"{_LOG_PREFIX} {name} must be at least 2-D, got {tuple(x.shape)}")
    if x.dim() >= 3 and x.shape[-2] == hc_mult:
        return x, _streams_d(x, hc_mult), tuple(x.shape[:-2])
    if x.shape[-1] % hc_mult:
        raise ValueError(f"{_LOG_PREFIX} {name} last dim {x.shape[-1]} is not divisible by hc_mult={hc_mult}")
    d = x.shape[-1] // hc_mult
    x_stream = x.reshape(*x.shape[:-1], hc_mult, d)
    return x_stream, d, tuple(x_stream.shape[:-2])


def _to_stream_layout(x: torch.Tensor, hc_mult: int, name: str) -> tuple[torch.Tensor, int, tuple[int, ...]]:
    """``_stream_layout`` + the operator's hidden_size envelope."""
    x_stream, d, outer = _stream_layout(x, hc_mult, name)
    if d not in _HC_SUPPORTED_D:
        raise ValueError(f"{_LOG_PREFIX} operator only supports hidden_size in {_HC_SUPPORTED_D}, got {d}")
    return x_stream, d, outer


def _post_to_2d(post: torch.Tensor, hc_mult: int, name: str) -> torch.Tensor:
    """[..., hc] or [..., hc, 1] -> [..., hc] (the operator's layout)."""
    if post.shape[-1] == 1 and post.dim() >= 2 and post.shape[-2] == hc_mult:
        post = post.squeeze(-1)
    if post.dim() < 1 or post.shape[-1] != hc_mult:
        raise ValueError(
            f"{_LOG_PREFIX} {name} must end with hc={hc_mult} (optionally with a trailing 1), got {tuple(post.shape)}"
        )
    return post


def _check_pre_params(hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, d: int, hc_mult: int) -> None:
    mix_hc = (2 + hc_mult) * hc_mult
    if hc_fn.dim() != 2 or hc_fn.shape != (mix_hc, hc_mult * d):
        raise ValueError(f"{_LOG_PREFIX} hc_fn must be {(mix_hc, hc_mult * d)}, got {tuple(hc_fn.shape)}")
    if hc_scale.dim() != 1 or hc_scale.shape[0] != 3:
        raise ValueError(f"{_LOG_PREFIX} hc_scale must be (3,), got {tuple(hc_scale.shape)}")
    if hc_base.dim() != 1 or hc_base.shape[0] != mix_hc:
        raise ValueError(f"{_LOG_PREFIX} hc_base must be ({mix_hc},), got {tuple(hc_base.shape)}")


def _rms_norm(layer_input: torch.Tensor, norm_weight: torch.Tensor | None, norm_eps: float) -> torch.Tensor:
    """Input RMSNorm fused into the pre block (matches CUDA/tilelang semantics).

    ``torch_npu.npu_rms_norm`` is bit-identical to the eager reference the
    current patch uses and costs a single launch; the eager path is kept for
    non-NPU/older-stack environments.
    """
    if norm_weight is None:
        return layer_input
    if _TORCH_NPU_OK and layer_input.is_npu:
        try:
            return torch_npu.npu_rms_norm(layer_input, norm_weight, norm_eps)[0]
        except Exception:  # pragma: no cover - older CANN without the op
            pass
    xf = layer_input.float()
    var = xf.square().mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + norm_eps) * norm_weight.float()).to(layer_input.dtype)


# --------------------------------------------------------------------------
# Torch fallbacks (upstream vllm kernels, the pre-AscendC behaviour)
# --------------------------------------------------------------------------
def _pre_torch(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    hc_post_mult_value: float,
    *,
    norm_weight: torch.Tensor | None,
    layer_norm_eps: float,
    post_keepdim: bool,
) -> HCPreOutput:
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch

    # mhc_pre_torch derives hc_mult from residual.shape[-2], so normalize the
    # packed [..., hc * d] layout (which the operator accepts) first.
    residual, _, _ = _stream_layout(residual, hc_mult, "residual")
    post_mix, comb_mix, layer_input = mhc_pre_torch(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
        norm_eps,
        hc_eps,
        hc_eps,
        hc_post_mult_value,
        sinkhorn_iters,
    )
    post = post_mix.squeeze(-1) if not post_keepdim else post_mix
    return HCPreOutput(_rms_norm(layer_input, norm_weight, layer_norm_eps), post, comb_mix)


def _post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    return mhc_post_torch(x, residual, post, comb)


def infer_hc_mult(residual: torch.Tensor) -> int:
    """hc_mult of a stream- or packed-layout residual.

    ``mhc_pre_torch`` derives it from ``residual.shape[-2]``, which is only
    correct for the stream layout ``[..., hc, d]`` our ``hc_expand`` produces;
    A3 feeds the packed ``[..., hc * d]`` layout instead.  Models with an
    hc_mult other than the operator's limit fall through to the torch
    derivation (the packed case is then ambiguous and stays unsupported).
    """
    if residual.dim() >= 2 and residual.shape[-1] % HC_PRE_HC_LIMIT == 0:
        if residual.dim() < 3 or residual.shape[-2] == HC_PRE_HC_LIMIT:
            return HC_PRE_HC_LIMIT  # stream [..., hc, d] or packed [..., hc * d]
    return int(residual.shape[-2])


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def hc_pre_ascendc(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = HC_PRE_HC_LIMIT,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    *,
    hc_post_mult_value: float = KERNEL_HC_POST_MULT_VALUE,
    norm_weight: torch.Tensor | None = None,
    layer_norm_eps: float = 0.0,
    post_keepdim: bool = True,
) -> HCPreOutput:
    """mHC pre block: mix projection + gates + Sinkhorn + stream contraction.

    ``y`` (a.k.a. ``layer_input``) is the bf16 contracted layer input
    ``sum_i pre_i * residual_i`` with the layer's input RMSNorm applied when
    ``norm_weight`` is given; ``post``/``comb`` are the fp32 gate matrices the
    next ``hc_post`` consumes.  Falls back to
    ``vllm.model_executor.kernels.mhc.torch.mhc_pre_torch`` when the fused
    operator is unavailable or the shapes are outside its support envelope.
    """
    global _PRE_AVAILABLE

    if _PRE_AVAILABLE is False or not _ensure_ops_registered():
        return _pre_torch(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult,
            sinkhorn_iters,
            norm_eps,
            hc_eps,
            hc_post_mult_value,
            norm_weight=norm_weight,
            layer_norm_eps=layer_norm_eps,
            post_keepdim=post_keepdim,
        )

    try:
        y, post, comb = _run_hc_pre(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult,
            sinkhorn_iters,
            norm_eps,
            hc_eps,
            hc_post_mult_value,
            norm_weight,
            layer_norm_eps,
            post_keepdim,
        )
    except Exception as op_err:
        if _PRE_AVAILABLE is True:
            raise
        print(
            f"{_LOG_PREFIX} hc_pre op failed, falling back to torch mhc_pre: "
            f"{op_err!r:.600} | x: {tuple(x.shape)} {x.dtype} | hc_fn: {tuple(hc_fn.shape)} "
            f"{hc_fn.dtype} | hc_mult: {hc_mult}",
            flush=True,
            file=sys.stderr,
        )
        _PRE_AVAILABLE = False
        return _pre_torch(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult,
            sinkhorn_iters,
            norm_eps,
            hc_eps,
            hc_post_mult_value,
            norm_weight=norm_weight,
            layer_norm_eps=layer_norm_eps,
            post_keepdim=post_keepdim,
        )

    if _PRE_AVAILABLE is None:
        print(f"{_LOG_PREFIX} hc_pre active (first call ok)", flush=True, file=sys.stderr)
    _PRE_AVAILABLE = True
    return y, post, comb


def _run_hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    hc_post_mult_value: float,
    norm_weight: torch.Tensor | None,
    layer_norm_eps: float,
    post_keepdim: bool,
) -> HCPreOutput:
    _check_hc_mult(hc_mult)
    if hc_post_mult_value != KERNEL_HC_POST_MULT_VALUE:
        raise ValueError(
            f"{_LOG_PREFIX} operator hard-codes hc_post_mult_value="
            f"{KERNEL_HC_POST_MULT_VALUE}, got {hc_post_mult_value}"
        )
    if hc_fn.dim() != 2 or hc_fn.dtype != torch.float32:
        raise ValueError(f"{_LOG_PREFIX} hc_fn must be 2-D float32, got {hc_fn.dim()}D {hc_fn.dtype}")
    _check_param_dtypes(hc_fn, hc_scale, hc_base)

    x_stream, d, outer = _to_stream_layout(x, hc_mult, "x")
    _check_pre_params(hc_fn, hc_scale, hc_base, d, hc_mult)
    if x_stream.dtype != torch.bfloat16:
        raise ValueError(f"{_LOG_PREFIX} x must be bfloat16, got {x_stream.dtype}")

    y, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
        x_stream.contiguous(),
        hc_fn.contiguous(),
        hc_scale.contiguous(),
        hc_base.contiguous(),
        hc_mult,
        sinkhorn_iters,
        norm_eps,
        hc_eps,
    )
    # [T, d] -> outer + [d]  (4-D inputs keep their batch dims)
    y = y.reshape(*outer, d) if outer else y.reshape(d)
    # [T, hc, hc] -> outer + [hc, hc]
    comb = comb.reshape(*outer, hc_mult, hc_mult) if outer else comb.reshape(hc_mult, hc_mult)
    if post_keepdim:
        # Match mhc_pre_torch's [..., hc, 1] so the vendored plumbing
        # (post_term = post * x.unsqueeze(-2)) keeps working unchanged.
        post = post.reshape(*outer, hc_mult, 1) if outer else post.reshape(hc_mult, 1)
    else:
        post = post.reshape(*outer, hc_mult) if outer else post.reshape(hc_mult)
    y = _rms_norm(y, norm_weight, layer_norm_eps)
    return HCPreOutput(y, post, comb)


def hc_post_ascendc(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """mHC post block: ``residual_out_j = sum_i comb_ji * residual_i + post_j * x``.

    ``x`` is the branch output ``[..., d]``, ``residual`` the mHC residual
    stream ``[..., hc, d]`` (packed ``[..., hc * d]`` also accepted),
    ``post_layer_mix``/``comb_res_mix`` the fp32 gates returned by the
    previous ``hc_pre`` (``post`` with or without its trailing singleton).
    Falls back to ``vllm.model_executor.kernels.mhc.torch.mhc_post_torch``.
    """
    global _POST_AVAILABLE

    if _POST_AVAILABLE is False or not _ensure_ops_registered():
        return _post_torch(x, residual, post_layer_mix, comb_res_mix)

    try:
        out = _run_hc_post(x, residual, post_layer_mix, comb_res_mix)
    except Exception as op_err:
        if _POST_AVAILABLE is True:
            raise
        print(
            f"{_LOG_PREFIX} hc_post op failed, falling back to torch mhc_post: "
            f"{op_err!r:.600} | x: {tuple(x.shape)} {x.dtype} | residual: "
            f"{tuple(residual.shape)} {residual.dtype}",
            flush=True,
            file=sys.stderr,
        )
        _POST_AVAILABLE = False
        return _post_torch(x, residual, post_layer_mix, comb_res_mix)

    if _POST_AVAILABLE is None:
        print(f"{_LOG_PREFIX} hc_post active (first call ok)", flush=True, file=sys.stderr)
    _POST_AVAILABLE = True
    return out


def _run_hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    hc_mult = HC_PRE_HC_LIMIT
    if residual.dim() < 2:
        raise ValueError(f"{_LOG_PREFIX} residual must be at least 2-D, got {tuple(residual.shape)}")
    out_shape = tuple(residual.shape)
    if (
        residual.dim() >= 2
        and residual.shape[-1] % hc_mult == 0
        and (residual.dim() < 3 or residual.shape[-2] != hc_mult)
    ):
        # packed [..., hc * d] -> [..., hc, d]
        d = residual.shape[-1] // hc_mult
        residual = residual.reshape(*residual.shape[:-1], hc_mult, d)
    if residual.shape[-2] != hc_mult:
        raise ValueError(f"{_LOG_PREFIX} residual must end with hc={hc_mult}, got {tuple(residual.shape)}")
    d = residual.shape[-1]

    if x.shape != residual.shape[:-2] + (d,):
        raise ValueError(
            f"{_LOG_PREFIX} x {tuple(x.shape)} must match residual stream dims {tuple(residual.shape[:-2] + (d,))}"
        )
    if residual.dtype != x.dtype:
        raise ValueError(f"{_LOG_PREFIX} x.dtype {x.dtype} must match residual.dtype {residual.dtype}")
    if post_layer_mix.dtype != comb_res_mix.dtype:
        raise ValueError(f"{_LOG_PREFIX} post.dtype {post_layer_mix.dtype} must match comb.dtype {comb_res_mix.dtype}")

    post2d = _post_to_2d(post_layer_mix, hc_mult, "post_layer_mix")
    if post2d.shape != residual.shape[:-1]:
        raise ValueError(
            f"{_LOG_PREFIX} post {tuple(post_layer_mix.shape)} must match residual stream "
            f"count {tuple(residual.shape[:-1])}"
        )
    if comb_res_mix.shape != residual.shape[:-2] + (hc_mult, hc_mult):
        raise ValueError(
            f"{_LOG_PREFIX} comb {tuple(comb_res_mix.shape)} must be {tuple(residual.shape[:-2] + (hc_mult, hc_mult))}"
        )

    # Operator contract: [B, S, d] / [B, S, hc, d] / [B, S, hc] / [B, S, hc, hc]
    x4 = x.reshape(1, -1, d)
    residual4 = residual.reshape(1, -1, hc_mult, d)
    post4 = post2d.reshape(1, -1, hc_mult)
    comb4 = comb_res_mix.reshape(1, -1, hc_mult, hc_mult)
    out = torch.ops._C_ascend.npu_hc_post(
        x4.contiguous(),
        residual4.contiguous(),
        post4.contiguous(),
        comb4.contiguous(),
    )
    return out.reshape(*out_shape)


def fused_post_pre_ascendc(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    *,
    hc_post_mult_value: float = KERNEL_HC_POST_MULT_VALUE,
    norm_weight: torch.Tensor | None = None,
    layer_norm_eps: float = 0.0,
    post_keepdim: bool = True,
) -> HCFusedPostPreOutput:
    """Post-attn ``hc_post`` fused with the pre-FFN ``hc_pre`` (2 launches).

    Drop-in for ``MHCFusedPostPreOp.forward_*``: returns
    ``(residual_cur, post_cur, comb_cur, layer_input_cur)`` in the same order
    and shapes as ``mhc_post_torch`` + ``mhc_pre_torch``.  The fused
    post+pre operator pair keeps the deferred gate application and the next
    block's mix/Sinkhorn in two kernel launches instead of ~340 eager ops.
    """
    residual_cur = hc_post_ascendc(x, residual, post_layer_mix, comb_res_mix)
    y, post_cur, comb_cur = hc_pre_ascendc(
        residual_cur,
        hc_fn,
        hc_scale,
        hc_base,
        sinkhorn_iters=sinkhorn_iters,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        hc_post_mult_value=hc_post_mult_value,
        norm_weight=norm_weight,
        layer_norm_eps=layer_norm_eps,
        post_keepdim=post_keepdim,
    )
    return HCFusedPostPreOutput(residual_cur, post_cur, comb_cur, y)
