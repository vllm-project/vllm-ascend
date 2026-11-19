import gc

import pytest
import torch
import torch_npu

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, E402

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

HC_MULT = 4
HIDDEN_SIZE = 4096
MIX_HC = 24
NORM_EPS = 1e-6
HC_EPS = 1e-6
HC_POST_SCALE = 2.0

pytestmark = pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.CANN_MEGAMHC)
    or not torch.ops._C_ascend.is_mega_mhc_available(),
    reason="MegaMhc and MhcSinkhorn require an A5 CANN ops-transformer package",
)


def _assert_close_with_pass_rate(actual, expected, *, diff_threshold, required_pass_rate):
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


@pytest.mark.parametrize("shape", [(2, HIDDEN_SIZE), (1, 2, HIDDEN_SIZE)])
@pytest.mark.parametrize("sinkhorn_iters", [1, 20, 100])
@torch.inference_mode()
def test_npu_mega_mhc_matches_post_pre_rms_norm(shape, sinkhorn_iters):
    torch.manual_seed(1024)
    leading_shape = shape[:-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    residual = torch.randn((*leading_shape, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16)
    post = torch.rand((*leading_shape, HC_MULT, 1), dtype=torch.float32)
    comb = torch.randn((*leading_shape, HC_MULT, HC_MULT), dtype=torch.float32).softmax(-1)
    hc_fn = torch.randn(MIX_HC, HC_MULT * HIDDEN_SIZE, dtype=torch.float32) / (HC_MULT * HIDDEN_SIZE)
    hc_scale = torch.rand(3, dtype=torch.float32)
    hc_base = torch.randn(MIX_HC, dtype=torch.float32) * 0.1
    norm_weight = torch.randn(HIDDEN_SIZE, dtype=torch.float32)

    npu_tensors = [tensor.npu() for tensor in (x, residual, post, comb, hc_fn, hc_scale, hc_base, norm_weight)]
    x_npu, residual_npu, post_npu, comb_npu, fn_npu, scale_npu, base_npu, weight_npu = npu_tensors
    actual = torch.ops._C_ascend.npu_mega_mhc(
        x_npu,
        residual_npu,
        post_npu,
        comb_npu,
        fn_npu,
        scale_npu,
        base_npu,
        weight_npu,
        HC_MULT,
        sinkhorn_iters,
        NORM_EPS,
        NORM_EPS,
        HC_EPS,
        HC_POST_SCALE,
    )

    token_count = x_npu.numel() // HIDDEN_SIZE
    expected_residual = (
        torch.ops._C_ascend.npu_hc_post(
            x_npu.reshape(token_count, HIDDEN_SIZE).unsqueeze(0),
            residual_npu.reshape(token_count, HC_MULT, HIDDEN_SIZE).unsqueeze(0),
            post_npu.reshape(token_count, HC_MULT, 1).squeeze(-1).unsqueeze(0),
            comb_npu.reshape(token_count, HC_MULT, HC_MULT).unsqueeze(0),
        )
        .squeeze(0)
        .reshape(residual_npu.shape)
    )
    expected_y, expected_post, expected_comb = torch.ops._C_ascend.npu_hc_pre_v2(
        expected_residual,
        fn_npu,
        scale_npu,
        base_npu,
        HC_MULT,
        sinkhorn_iters,
        NORM_EPS,
        HC_EPS,
    )
    expected_y = torch_npu.npu_rms_norm(expected_y, weight_npu, epsilon=NORM_EPS)[0]
    expected = (expected_residual, expected_post.unsqueeze(-1), expected_comb, expected_y)

    for actual_value, expected_value in zip(actual, expected):
        assert actual_value.shape == expected_value.shape
        assert actual_value.dtype == expected_value.dtype
        _assert_close_with_pass_rate(
            actual_value,
            expected_value,
            diff_threshold=4e-3 if actual_value.dtype == torch.bfloat16 else 1e-4,
            required_pass_rate=0.98 if actual_value.dtype == torch.bfloat16 else 0.995,
        )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
