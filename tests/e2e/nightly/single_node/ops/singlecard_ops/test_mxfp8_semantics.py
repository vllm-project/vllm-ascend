# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.device.mxfp_compat import FLOAT8_E8M0FNU_DTYPE


BLOCK_SIZE = 32


def _scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    return scale.contiguous().view(torch.uint8).reshape(scale.shape[0], -1)


def _dequant_mxfp8(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    values_fp32 = values.float()
    scale_u8 = _scale_bytes(scale)
    num_blocks = values.shape[-1] // BLOCK_SIZE
    blocked = values_fp32.reshape(*values.shape[:-1], num_blocks, BLOCK_SIZE)
    multiplier = torch.exp2(scale_u8.float() - 127.0)
    return (blocked * multiplier.unsqueeze(-1)).reshape_as(values_fp32)


def _vllm_style_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = x.shape[-1] // BLOCK_SIZE
    blocked = x.float().reshape(*x.shape[:-1], num_blocks, BLOCK_SIZE)
    amax = blocked.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    scale_u8 = (torch.floor(torch.log2(amax)) + 127.0).clamp(0, 254).to(torch.uint8)
    multiplier = torch.exp2(scale_u8.float() - 127.0)
    values = (blocked / multiplier.unsqueeze(-1)).reshape_as(x).to(torch.float8_e4m3fn)
    return values, scale_u8


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = actual.float() - expected.float()
    return (diff.norm() / expected.float().norm().clamp_min(1e-12)).item()


def _print_diff(tag: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = actual.float() - expected.float()
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    print(
        f"{tag}: shape={tuple(actual.shape)} "
        f"abs_mean={diff.abs().mean().item():.6e} "
        f"max_abs={diff.abs().max().item():.6e} "
        f"rel_l2={_relative_l2(actual, expected):.6e} "
        f"cosine={cosine.item():.8f}"
    )


@pytest.mark.parametrize("m,n,k", [(7, 64, 128)])
def test_mxfp8_scale_semantics_and_quant_matmul(m: int, n: int, k: int) -> None:
    if FLOAT8_E8M0FNU_DTYPE is None:
        pytest.skip("The runtime does not expose float8_e8m0fnu")
    if not hasattr(torch_npu, "npu_dynamic_mx_quant"):
        pytest.skip("The runtime does not expose npu_dynamic_mx_quant")

    torch.manual_seed(7)
    device = torch.device("npu")

    # Different block magnitudes make exponent-selection differences visible.
    x_gain = torch.exp2(torch.arange(k // BLOCK_SIZE, device=device).float() - 2.0)
    w_gain = torch.exp2(torch.arange(k // BLOCK_SIZE, device=device).float() - 1.0)
    x = (
        torch.randn(m, k // BLOCK_SIZE, BLOCK_SIZE, device=device)
        * x_gain.view(1, -1, 1)
    ).reshape(m, k).to(torch.bfloat16)
    weight = (
        torch.randn(n, k // BLOCK_SIZE, BLOCK_SIZE, device=device)
        * w_gain.view(1, -1, 1)
    ).reshape(n, k).to(torch.bfloat16)

    # Compare two valid MXFP8 activation encodings through their decoded values.
    x_vllm_q, x_vllm_scale = _vllm_style_quant(x)
    x_npu_q, x_npu_scale = torch_npu.npu_dynamic_mx_quant(
        x, dst_type=torch.float8_e4m3fn
    )
    x_vllm_dequant = _dequant_mxfp8(x_vllm_q, x_vllm_scale)
    x_npu_dequant = _dequant_mxfp8(x_npu_q, x_npu_scale)

    scale_delta = x_vllm_scale.to(torch.int16) - _scale_bytes(x_npu_scale).to(
        torch.int16
    )
    print(
        "activation_scale_delta(vllm-npu): "
        f"min={scale_delta.min().item()} max={scale_delta.max().item()} "
        f"mean={scale_delta.float().mean().item():.4f} "
        f"unique={torch.unique(scale_delta).cpu().tolist()}"
    )
    _print_diff("vllm_quant_dequant_vs_bf16", x_vllm_dequant, x.float())
    _print_diff("npu_quant_dequant_vs_bf16", x_npu_dequant, x.float())
    _print_diff("npu_dequant_vs_vllm_dequant", x_npu_dequant, x_vllm_dequant)

    # Emulate checkpoint weight + weight_scale_inv storage. Despite the source
    # name, these bytes are direct E8M0 scales and are not numerically inverted.
    weight_q, weight_scale_inv = _vllm_style_quant(weight)
    weight_dequant = _dequant_mxfp8(weight_q, weight_scale_inv)

    # Ascend linear layout:
    #   weight [N,K] -> [K,N]
    #   scale  [N,K/32] -> [K/64,N,2]
    weight_npu = weight_q.transpose(0, 1).contiguous()
    weight_scale_npu = (
        weight_scale_inv.reshape(n, k // 64, 2).transpose(0, 1).contiguous()
    )
    restored_scale = weight_scale_npu.transpose(0, 1).reshape(n, k // BLOCK_SIZE)
    restored_weight = _dequant_mxfp8(
        weight_npu.transpose(0, 1).contiguous(), restored_scale
    )
    _print_diff("weight_layout_roundtrip", restored_weight, weight_dequant)

    expected = x_npu_dequant @ weight_dequant.transpose(0, 1)
    actual = torch_npu.npu_quant_matmul(
        x_npu_q,
        weight_npu,
        weight_scale_npu,
        scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        pertoken_scale=x_npu_scale,
        pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, BLOCK_SIZE],
    )
    _print_diff("npu_quant_matmul_vs_dequant_reference", actual, expected)

    # Cross-check the important compatibility boundary: feed the vLLM-style
    # activation encoding directly to the NPU operator.
    expected_vllm_x = x_vllm_dequant @ weight_dequant.transpose(0, 1)
    actual_vllm_x = torch_npu.npu_quant_matmul(
        x_vllm_q,
        weight_npu,
        weight_scale_npu,
        scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        pertoken_scale=x_vllm_scale,
        pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, BLOCK_SIZE],
    )
    _print_diff(
        "npu_quant_matmul_with_vllm_activation_encoding",
        actual_vllm_x,
        expected_vllm_x,
    )

    assert _relative_l2(x_vllm_dequant, x.float()) < 0.08
    assert _relative_l2(x_npu_dequant, x.float()) < 0.08
    assert torch.equal(restored_scale, weight_scale_inv)
    assert _relative_l2(actual, expected) < 0.08
    assert _relative_l2(actual_vllm_x, expected_vllm_x) < 0.08
