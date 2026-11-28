# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape-generic Verify for the four-token Qwen A3 RecoverSSM layout."""

import torch

from .dispatch import _is_a3


def supports_model_verify(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    *,
    checkpoint_state,
    replay_cache,
    query_start_loc,
    state_indices,
    spec_query_len,
    scale,
    softplus_beta,
    softplus_threshold,
    use_qk_l2norm_in_kernel,
    out,
):
    rows = state_indices.numel()
    if (
        spec_query_len != 4
        or q.ndim != 4
        or not use_qk_l2norm_in_kernel
        or softplus_beta != 1.0
        or softplus_threshold != 20.0
        or scale not in (None, 128**-0.5)
    ):
        return False

    tokens = q.shape[1]
    if not 0 <= tokens <= rows * 4 or state_indices.ndim != 1:
        return False

    if any(
        t.device != q.device
        for t in (k, v, a, b, A_log, dt_bias, checkpoint_state, replay_cache, query_start_loc, state_indices)
    ):
        return False
    if any(t.shape != (1, tokens, 8, 128) or t.stride()[-2:] != (128, 1) or t.dtype != torch.bfloat16 for t in (q, k)):
        return False
    if v.shape != (1, tokens, 24, 128) or v.stride()[-2:] != (128, 1) or v.dtype != torch.bfloat16:
        return False
    if any(
        t.shape not in ((tokens, 24), (1, tokens, 24)) or t.stride(-1) != 1 or t.dtype != torch.bfloat16 for t in (a, b)
    ):
        return False
    if any(
        t.shape != (24,) or t.stride(0) != 1 or t.dtype not in (torch.bfloat16, torch.float32) for t in (A_log, dt_bias)
    ):
        return False
    if (
        checkpoint_state.ndim != 4
        or checkpoint_state.shape[1:] != (24, 128, 128)
        or checkpoint_state.stride()[1:] != (16384, 128, 1)
        or checkpoint_state.stride(0) < 24 * 128 * 128
        or checkpoint_state.stride(0) % 128
        or checkpoint_state.dtype != torch.bfloat16
    ):
        return False
    pages = checkpoint_state.shape[0]
    if (
        not pages
        or replay_cache.shape != (pages, 24, 4, 257)
        or replay_cache.stride(0) < 24 * 4 * 257
        or replay_cache.stride()[1:] != (1028, 257, 1)
        or replay_cache.dtype != torch.float32
    ):
        return False
    if (
        query_start_loc.shape != (rows + 1,)
        or query_start_loc.stride(0) != 1
        or query_start_loc.dtype != torch.int32
        or state_indices.dtype != torch.int32
        or state_indices.stride(0) < 1
    ):
        return False
    return out is None or (
        out.shape == v.shape and out.is_contiguous() and out.dtype == v.dtype and out.device == v.device
    )


def _run_model_verify(**kwargs):
    from ._model_verify_kernel import run_model_verify

    return run_model_verify(**kwargs)


def verify_model(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    checkpoint_state: torch.Tensor,
    replay_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    scale: float | None = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_qk_l2norm_in_kernel: bool = True,
    out: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> torch.Tensor:
    """Run the model ABI on A3 with 8 Q/K heads, 24 V heads and head size 128.

    Each request may supply zero to four tokens and one checkpoint page index.
    The checkpoint stays read-only. Valid active pages receive FP32 replay
    records. The caller may supply a non-overlapping contiguous output view.
    Supplying z, a BF16 norm weight and epsilon applies sigmoid-gated RMSNorm
    to Verify's BF16 result. Omit all three to apply the model's norm separately.
    """
    args = dict(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        checkpoint_state=checkpoint_state,
        replay_cache=replay_cache,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        out=out,
    )
    if not _is_a3(q.device) or not supports_model_verify(
        **args,
        spec_query_len=spec_query_len,
        scale=scale,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    ):
        raise ValueError("Inputs do not match the A3 TileLang RecoverSSM Verify layout")

    if any(value is not None for value in (z, norm_weight, norm_eps)) and (
        z is None
        or z.shape != v.shape
        or z.stride()[-2:] != (128, 1)
        or z.dtype != torch.bfloat16
        or z.device != v.device
        or norm_weight is None
        or norm_weight.shape != (128,)
        or not norm_weight.is_contiguous()
        or norm_weight.dtype != torch.bfloat16
        or norm_weight.device != v.device
        or norm_eps is None
        or norm_eps <= 0
    ):
        raise ValueError("Fused sigmoid RMSNorm requires BF16 z and weight and positive epsilon")

    if q.shape[1] == 0:
        return out if out is not None else torch.empty_like(v, memory_format=torch.contiguous_format)
    if checkpoint_state.shape[0] * (checkpoint_state.stride(0) // 128) > 2**31:
        raise ValueError("Checkpoint row offsets exceed int32")

    return _run_model_verify(**args, z=z, norm_weight=norm_weight, norm_eps=norm_eps)
