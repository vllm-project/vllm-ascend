# SPDX-License-Identifier: Apache-2.0
"""Fused QSA output-gate activation for Ascend."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _qsa_sigmoid_mul_kernel(
    output,
    gate,
    output_row_stride,
    gate_row_stride,
    width,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < width
    values = tl.load(
        output + row * output_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gates = tl.load(
        gate + row * gate_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + row * output_row_stride + offsets,
        values * tl.sigmoid(gates),
        mask=mask,
    )


def qsa_sigmoid_mul_(output: torch.Tensor, gate: torch.Tensor) -> None:
    """Apply the QSA sigmoid gate in-place with one Triton launch."""
    if output.shape != gate.shape:
        raise ValueError("QSA output and gate must have the same shape")
    if output.ndim != 2:
        raise ValueError("QSA output and gate must be two-dimensional")
    if output.stride(1) != 1 or gate.stride(1) != 1:
        raise ValueError("QSA output and gate must be contiguous in the last dimension")

    rows, width = output.shape
    if rows == 0:
        return
    block = triton.next_power_of_2(width)
    _qsa_sigmoid_mul_kernel[(rows,)](
        output,
        gate,
        output.stride(0),
        gate.stride(0),
        width,
        BLOCK=block,
        num_warps=8,
    )


def _qsa_sigmoid_mul_fake(output: torch.Tensor, gate: torch.Tensor) -> None:
    return None


direct_register_custom_op(
    op_name="qwen4_exp_ascend_qsa_sigmoid_mul_",
    op_func=qsa_sigmoid_mul_,
    fake_impl=_qsa_sigmoid_mul_fake,
    mutates_args=["output"],
    dispatch_key="PrivateUse1",
)


__all__ = ["qsa_sigmoid_mul_"]
