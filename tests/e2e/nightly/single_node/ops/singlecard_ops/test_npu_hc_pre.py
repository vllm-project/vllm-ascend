import gc

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

HC_MULT = 4
HIDDEN_SIZE = 4096
MIX_HC = 24
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6


def _make_hc_pre_inputs(shape: tuple[int, ...]):
    torch.manual_seed(1024)
    x = torch.randn(shape, dtype=torch.bfloat16, device="npu")
    hc_fn = (
        torch.randn(
            MIX_HC,
            HC_MULT * HIDDEN_SIZE,
            dtype=torch.float32,
            device="npu",
        )
        * 0.01
    )
    hc_scale = torch.randn(3, dtype=torch.float32, device="npu") * 0.01
    hc_base = torch.randn(MIX_HC, dtype=torch.float32, device="npu") * 0.01
    return x, hc_fn, hc_scale, hc_base


def _compare_hc_pre_outputs(shape: tuple[int, ...]):
    x, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(shape)
    expected = torch.ops._C_ascend.npu_hc_pre(x, hc_fn, hc_scale, hc_base, HC_MULT, HC_SINKHORN_ITERS, NORM_EPS, HC_EPS)
    actual = torch.ops._C_ascend.npu_hc_pre_v2(
        x, hc_fn, hc_scale, hc_base, HC_MULT, HC_SINKHORN_ITERS, NORM_EPS, HC_EPS
    )

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor.cpu(),
            expected_tensor.cpu(),
            atol=5e-2,
            rtol=5e-2,
        )


@torch.inference_mode()
def test_npu_hc_pre_v1_v2_bf16_3d_input():
    _compare_hc_pre_outputs((2, HC_MULT, HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_npu_hc_pre_v1_v2_bf16_4d_input():
    _compare_hc_pre_outputs((1, 2, HC_MULT, HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
