# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit(do_not_specialize=["numel"])
def attention_sigmoid_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    numel,
    WIDTH: tl.constexpr,
    X_ROW: tl.constexpr,
    X_COL: tl.constexpr,
    G_ROW: tl.constexpr,
    G_COL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(numel, BLOCK), tl.num_programs(0)):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        if X_ROW == WIDTH and X_COL == 1:
            xo = offsets
        else:
            xo = (offsets // WIDTH) * X_ROW + (offsets % WIDTH) * X_COL
        if G_ROW == WIDTH and G_COL == 1:
            go = offsets
        else:
            go = (offsets // WIDTH) * G_ROW + (offsets % WIDTH) * G_COL
        x = tl.load(x_ptr + xo, mask=mask, other=0).to(tl.float32)
        g = tl.load(gate_ptr + go, mask=mask, other=0).to(tl.float32)
        # Match the original sigmoid tensor's rounding before multiplication.
        scale = tl.sigmoid(g).to(gate_ptr.dtype.element_ty).to(tl.float32)
        tl.store(out_ptr + offsets, x * scale, mask=mask)


def _validate_inputs(x: torch.Tensor, gate: torch.Tensor) -> None:
    if x.ndim != 2 or gate.ndim != 2 or x.shape != gate.shape:
        raise ValueError("Attention output and gate must have the same 2-D shape")
    if x.device != gate.device or x.dtype != gate.dtype:
        raise ValueError("Attention output and gate must have matching device and dtype")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Attention gating requires FP16, BF16 or FP32")


def attention_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Return x * sigmoid(gate) without mutating or aliasing either input."""
    _validate_inputs(x, gate)
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    numel = x.numel()
    if numel == 0:
        return output
    init_device_properties_triton()
    block = 512 if numel <= 8192 else 2048 if numel < 65536 else 4096
    grid = (min(triton.cdiv(numel, block), get_vectorcore_num()),)
    attention_sigmoid_mul_kernel[grid](
        x,
        gate,
        output,
        numel,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        gate.stride(0),
        gate.stride(1),
        BLOCK=block,
        multibuffer=False,
    )
    return output


def attention_sigmoid_mul_fake(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    _validate_inputs(x, gate)
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="ascend_attention_sigmoid_mul",
    op_func=attention_sigmoid_mul,
    fake_impl=attention_sigmoid_mul_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
