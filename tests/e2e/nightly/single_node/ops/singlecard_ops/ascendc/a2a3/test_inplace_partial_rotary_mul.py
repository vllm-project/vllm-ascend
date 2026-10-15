import gc

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

NOPE_DIM = 128  # start of partial_slice
HEAD_DIM = 192  # end of partial_slice (rotation range [NOPE, HEAD))
ROPE_DIM = HEAD_DIM - NOPE_DIM
NUM_HEADS = 16


def golden_inplace_partial_rotary_mul(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, negate_sin: bool = False
) -> torch.Tensor:
    """CPU golden for interleave mode: y[j] = x[j]*cos[j] +/- x[j^1]*sin[j].

    The sin term is negated on even positions (kernel default mask 0x5555,
    standard interleave RoPE).
    x: (T, 1, H, D); cos/sin: (T, 1, 1, R) fp32, indexed element-wise,
    last dim equals the rotation span R.
    negate_sin=True is equivalent to passing -sin (legacy path).
    """
    x32 = x.float()
    c = cos.float()
    s = sin.float()
    if negate_sin:
        s = -s
    x_rot = x32[..., NOPE_DIM:]
    # swap even/odd positions: x_swap[j] = x[j^1]
    x_swap = torch.empty_like(x_rot)
    x_swap[..., 0::2] = x_rot[..., 1::2]
    x_swap[..., 1::2] = x_rot[..., 0::2]
    # negate the sin term on even positions (kernel default mask 0x5555)
    sin_term = x_swap * s
    sin_term[..., 0::2] *= -1
    y = x32.clone()
    y[..., NOPE_DIM:] = x_rot * c + sin_term
    return y


@pytest.mark.parametrize(
    "num_tokens",
    [1, 16, 128, 1000],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 8e-3, 8e-3),
        (torch.float32, 1e-5, 1e-5),
    ],
)
def test_inplace_partial_rotary_mul(num_tokens: int, dtype, atol, rtol):
    torch.manual_seed(0)

    # x: (T, 1, H, D); cos/sin: (T, 1, 1, R) fp32, no repeat_interleave
    x = torch.randn(num_tokens, 1, NUM_HEADS, HEAD_DIM, dtype=dtype)
    base = torch.randn(num_tokens, 1, 1, ROPE_DIM, dtype=torch.float32)
    cos = torch.cos(base)
    sin = torch.sin(base)

    # ---- 1. positive sin: negate_sin=False ----
    out = x.clone().npu()
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        out,
        cos.npu(),
        sin.npu(),
        rotary_mode="interleave",
        partial_slice=[NOPE_DIM, HEAD_DIM],
        negate_sin=False,
    )
    ref = golden_inplace_partial_rotary_mul(x, cos, sin, negate_sin=False)
    torch.testing.assert_close(out.cpu(), ref.to(dtype), atol=atol, rtol=rtol)

    # ---- 2. negative sin: raw sin + negate_sin=True ----
    out_neg = x.clone().npu()
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        out_neg,
        cos.npu(),
        sin.npu(),
        rotary_mode="interleave",
        partial_slice=[NOPE_DIM, HEAD_DIM],
        negate_sin=True,
    )
    ref_neg = golden_inplace_partial_rotary_mul(x, cos, sin, negate_sin=True)
    torch.testing.assert_close(out_neg.cpu(), ref_neg.to(dtype), atol=atol, rtol=rtol)

    # ---- 3. cross check: negate_sin=True equals -sin input (legacy), bit-wise ----
    out_neg2 = x.clone().npu()
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        out_neg2,
        cos.npu(),
        (-sin).npu(),
        rotary_mode="interleave",
        partial_slice=[NOPE_DIM, HEAD_DIM],
        negate_sin=False,
    )
    assert torch.equal(out_neg.cpu(), out_neg2.cpu())

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
