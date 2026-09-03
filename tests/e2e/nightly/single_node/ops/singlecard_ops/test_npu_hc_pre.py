import gc

import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

HC_MULT = 4
DSV4_FLASH_HIDDEN_SIZE = 4096
HIDDEN_SIZE = DSV4_FLASH_HIDDEN_SIZE
EXTENDED_HIDDEN_SIZE = 7168
MIX_HC = 24
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6
HF32_MANTISSA_BITS = 10
FP32_MANTISSA_BITS = 23
Y_DIFF_THRESHOLD = 4e-3
Y_REQUIRED_PASS_RATE = 0.98
AUX_DIFF_THRESHOLD = 1e-4
AUX_REQUIRED_PASS_RATE = 0.995


def _make_hc_pre_inputs(shape: tuple[int, ...]):
    torch.manual_seed(1024)
    hidden_size = shape[-1]
    fan_in = HC_MULT * hidden_size
    x = (torch.rand(shape, dtype=torch.float32) * 2).to(torch.bfloat16)
    hc_fn = (
        torch.rand(
            MIX_HC,
            fan_in,
            dtype=torch.float32,
        )
        / fan_in
    )
    hc_scale = torch.rand(3, dtype=torch.float32) * 2
    hc_base = torch.rand(MIX_HC, dtype=torch.float32) * 2
    return x, hc_fn, hc_scale, hc_base


def _to_hf32(tensor: torch.Tensor) -> torch.Tensor:
    dropped_mantissa_bits = FP32_MANTISSA_BITS - HF32_MANTISSA_BITS
    mantissa_mask = ~((1 << dropped_mantissa_bits) - 1)
    bits = tensor.contiguous().view(torch.int32)
    return (bits & mantissa_mask).view(torch.float32)


def _hc_pre_cpu(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_float = x.float()
    x_flat = x_float.flatten(-2)
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)

    # HcPre performs this matmul in HF32 round-toward-zero mode.
    mixes = F.linear(_to_hf32(x_flat), _to_hf32(hc_fn)) * inv_rms
    pre, post, comb_frag = mixes.split(
        [HC_MULT, HC_MULT, HC_MULT * HC_MULT],
        dim=-1,
    )
    comb_frag = comb_frag.unflatten(-1, (HC_MULT, HC_MULT))

    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[HC_MULT : 2 * HC_MULT])
    comb_frag = comb_frag * hc_scale[2] + hc_base[2 * HC_MULT :].view(HC_MULT, HC_MULT)

    comb_frag = comb_frag.softmax(-1) + HC_EPS
    comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITERS - 1):
        comb_frag = comb_frag / (comb_frag.sum(-1, keepdim=True) + HC_EPS)
        comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)

    y = (pre.unsqueeze(-1) * x_float).sum(dim=-2).to(x.dtype)
    return y, post, comb_frag


def _assert_close_with_pass_rate(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    diff_threshold: float,
    required_pass_rate: float,
):
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    abs_diff = (actual - expected).abs()
    magnitude = torch.maximum(actual.abs(), expected.abs())
    close = (abs_diff <= diff_threshold) | (
        abs_diff / magnitude.clamp_min(torch.finfo(torch.float32).tiny) <= diff_threshold
    )
    pass_rate = close.float().mean().item()
    max_abs_diff = abs_diff.max().item()
    assert pass_rate >= required_pass_rate, (
        f"pass rate {pass_rate:.2%} is below {required_pass_rate:.2%}; max absolute difference: {max_abs_diff}"
    )


def _compare_hc_pre_with_cpu(shape: tuple[int, ...]):
    x, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(shape)
    expected_y, expected_post, expected_comb_frag = _hc_pre_cpu(
        x,
        hc_fn,
        hc_scale,
        hc_base,
    )
    y, post, comb_frag = torch.ops._C_ascend.npu_hc_pre_v2(
        x.npu(),
        hc_fn.npu(),
        hc_scale.npu(),
        hc_base.npu(),
        HC_MULT,
        HC_SINKHORN_ITERS,
        NORM_EPS,
        HC_EPS,
    )

    batch_shape = shape[:-2]
    assert y.shape == (*batch_shape, shape[-1])
    assert post.shape == (*batch_shape, HC_MULT)
    assert comb_frag.shape == (*batch_shape, HC_MULT, HC_MULT)
    assert y.dtype == torch.bfloat16
    assert post.dtype == torch.float32
    assert comb_frag.dtype == torch.float32
    _assert_close_with_pass_rate(
        y,
        expected_y,
        diff_threshold=Y_DIFF_THRESHOLD,
        required_pass_rate=Y_REQUIRED_PASS_RATE,
    )
    _assert_close_with_pass_rate(
        post,
        expected_post,
        diff_threshold=AUX_DIFF_THRESHOLD,
        required_pass_rate=AUX_REQUIRED_PASS_RATE,
    )
    _assert_close_with_pass_rate(
        comb_frag,
        expected_comb_frag,
        diff_threshold=AUX_DIFF_THRESHOLD,
        required_pass_rate=AUX_REQUIRED_PASS_RATE,
    )


@torch.inference_mode()
def test_npu_hc_pre_v2_bf16_3d_input():
    _compare_hc_pre_with_cpu((2, HC_MULT, HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_npu_hc_pre_v2_bf16_4d_input():
    _compare_hc_pre_with_cpu((1, 2, HC_MULT, HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_npu_hc_pre_v2_bf16_dsv4_flash_hidden_size():
    _compare_hc_pre_with_cpu((4, HC_MULT, DSV4_FLASH_HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_npu_hc_pre_v2_bf16_extended_hidden_size():
    _compare_hc_pre_with_cpu((2, HC_MULT, EXTENDED_HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
