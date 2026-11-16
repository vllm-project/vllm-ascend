# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.mhc_post_collapse_norm import MHC_POST_COLLAPSE_NORM

MHC_MAX_ROWS = 72

_recording_mhc: ContextVar[bool] = ContextVar("dsv41_mhc_recording", default=False)


@contextmanager
def record_mhc_overlap():
    token = _recording_mhc.set(True)
    try:
        yield
    finally:
        _recording_mhc.reset(token)


@cache
def _coefficient_stream(device: torch.device) -> torch.npu.Stream:
    return torch.npu.Stream(device=device)


def _coefficients_on_aux(updated, fn, scale, base, iters, norm_eps, hc_eps):
    main = torch.npu.current_stream()
    auxiliary = _coefficient_stream(updated.device)
    auxiliary.wait_stream(main)
    with torch.npu.stream(auxiliary):
        post, comb, pre = torch.ops._C_ascend.dsv41_hc_pre_coefficients(
            updated, fn, scale, base, iters, norm_eps, hc_eps
        )
    updated.record_stream(auxiliary)
    for coefficients in (post, comb, pre):
        coefficients.record_stream(main)
    return post, comb, pre


def _wait_for_coefficients(residual):
    torch.npu.current_stream().wait_stream(_coefficient_stream(residual.device))


def _prepare_attention(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    iters: int,
    norm_eps: float,
    hc_eps: float,
    overlap: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coefficients = torch.ops._C_ascend.dsv41_hc_pre_coefficients
    if overlap and _recording_mhc.get():
        _wait_for_coefficients(residual)
        coefficients = _coefficients_on_aux

    updated = torch.empty_like(residual)
    normalized = torch.empty_like(x, memory_format=torch.contiguous_format)
    MHC_POST_COLLAPSE_NORM(
        x.contiguous(), residual, post, comb, pre, weight, updated, normalized, normalized, norm_eps, False
    )
    return updated, normalized, *coefficients(updated, fn, scale, base, iters, norm_eps, hc_eps)


def _prepare_moe(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    iters: int,
    norm_eps: float,
    hc_eps: float,
    overlap: bool,
    wait_for_attention: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coefficients = torch.ops._C_ascend.dsv41_hc_pre_coefficients
    if overlap and _recording_mhc.get():
        if wait_for_attention:
            _wait_for_coefficients(residual)
        coefficients = _coefficients_on_aux

    updated = torch.empty_like(residual)
    normalized = torch.empty_like(x, memory_format=torch.contiguous_format)
    routing_input = normalized.new_empty(normalized.shape, dtype=torch.float32)
    MHC_POST_COLLAPSE_NORM(
        x.contiguous(), residual, post, comb, pre, weight, updated, normalized, routing_input, norm_eps, True
    )
    return updated, normalized, routing_input, *coefficients(updated, fn, scale, base, iters, norm_eps, hc_eps)


def _finish_overlap(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    if _recording_mhc.get():
        _wait_for_coefficients(residual)
    return _native_post(x, residual, post, comb)


def _native_post(x, residual, post, comb):
    return torch.ops._C_ascend.npu_hc_post(
        x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
    ).squeeze(0)


def _fake_attention(x, residual, post, comb, pre, weight, fn, scale, base, iters, norm_eps, hc_eps, overlap):
    return (
        torch.empty_like(residual),
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(post),
        torch.empty_like(comb),
        torch.empty_like(pre),
    )


def _fake_moe(
    x, residual, post, comb, pre, weight, fn, scale, base, iters, norm_eps, hc_eps, overlap, wait_for_attention
):
    return (
        torch.empty_like(residual),
        torch.empty_like(x, memory_format=torch.contiguous_format),
        x.new_empty(x.shape, dtype=torch.float32),
        torch.empty_like(post),
        torch.empty_like(comb),
        torch.empty_like(pre),
    )


def _fake_post(x, residual, post, comb):
    return torch.empty_like(residual)


direct_register_custom_op(
    op_name="dsv41_mhc_attention_input",
    op_func=_prepare_attention,
    mutates_args=[],
    fake_impl=_fake_attention,
    dispatch_key="PrivateUse1",
)


direct_register_custom_op(
    op_name="dsv41_mhc_moe_input",
    op_func=_prepare_moe,
    mutates_args=[],
    fake_impl=_fake_moe,
    dispatch_key="PrivateUse1",
)


direct_register_custom_op(
    op_name="dsv41_mhc_finish_overlap",
    op_func=_finish_overlap,
    mutates_args=[],
    fake_impl=_fake_post,
    dispatch_key="PrivateUse1",
)
