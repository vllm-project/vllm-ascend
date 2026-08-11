import gc

import pytest
import torch
import torch_npu


torch_npu.npu.config.allow_internal_format = True

MXFP8_GROUP_SIZE = 32


def swiglu_no_interleaved_with_alpha_and_limit_fp32(
    x: torch.Tensor,
    gemm1_alpha: float,
    gemm1_limit: float,
) -> torch.Tensor:
    input_dtype = x.dtype
    gate, up = x.chunk(2, dim=-1)
    gate, up = gate.to(torch.float32), up.to(torch.float32)
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    gate = gate * torch.sigmoid(gate * gemm1_alpha)
    up = (up + 1)
    return (gate * up).to(input_dtype)


def _make_mxfp8_identity_weight(hidden_size: int, dst_type: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.eye(hidden_size, dtype=dst_type, device="cpu").transpose(0, 1).to("npu")

    num_groups = (hidden_size + MXFP8_GROUP_SIZE - 1) // MXFP8_GROUP_SIZE
    padded_num_groups = num_groups + num_groups % 2
    weight_scale = torch.ones(
        (hidden_size, padded_num_groups),
        dtype=torch.float8_e8m0fnu,
        device="cpu",
    ).view(torch.uint8)
    weight_scale = weight_scale.reshape(hidden_size, padded_num_groups // 2, 2).transpose(0, 1).to("npu")
    return weight, weight_scale


def _dequantize_mxfp8(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    identity_weight, identity_weight_scale = _make_mxfp8_identity_weight(quantized.shape[-1], quantized.dtype)
    print(identity_weight.shape, identity_weight_scale.shape)
    return torch_npu.npu_quant_matmul(
        quantized,
        identity_weight,
        identity_weight_scale,
        scale_dtype=torch.float8_e8m0fnu,
        pertoken_scale=scale,
        pertoken_scale_dtype=torch.float8_e8m0fnu,
        bias=None,
        output_dtype=output_dtype,
        group_sizes=[1, 1, MXFP8_GROUP_SIZE],
    )


def _assert_dequantized_close(
    actual: torch.Tensor,
    actual_scale: torch.Tensor,
    expected: torch.Tensor,
    expected_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> None:
    actual_dequant = _dequantize_mxfp8(actual, actual_scale, output_dtype)
    expected_dequant = _dequantize_mxfp8(expected, expected_scale, output_dtype)
    # 千分之五精度标准
    torch.testing.assert_close(actual_dequant.cpu(), expected_dequant.cpu(), atol=1, rtol=5e-3)

@pytest.mark.parametrize("seqlen", [1, 128, 128*1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_swiglu_mx_quant_matches_dynamic_mx_quant(dtype: torch.dtype, seqlen: int):
    if not hasattr(torch.ops._C_ascend, "swiglu_mx_quant"):
        pytest.skip("swiglu_mx_quant custom op is not available")

    x = torch.randn((seqlen, 6144), dtype=dtype, device="npu")
    dst_type = torch.float8_e4m3fn
    gemm1_alpha = 1.702
    gemm1_limit = 7.0

    golden_act = swiglu_no_interleaved_with_alpha_and_limit_fp32(x, gemm1_alpha, gemm1_limit)
    expected, expected_scale = torch_npu.npu_dynamic_mx_quant(golden_act, dst_type=dst_type, scale_alg=1)

    actual, actual_scale = torch_npu.npu_swiglu_mx_quant(
        x,
        group_index=None,
        dst_type=dst_type,
        activate_dim=-1,
        activate_left=True,
        swiglu_mode=1,
        clamp_limit=gemm1_limit,
        glu_alpha=gemm1_alpha,
        glu_bias=1.0,
        group_mode=0,
        axis=-1,
        round_mode="rint",
        scale_alg=1,
        max_dtype_value=0.0,
    )

    _assert_dequantized_close(actual, actual_scale, expected, expected_scale, dtype)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
