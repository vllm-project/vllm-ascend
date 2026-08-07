from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import (
    AscendDSAImpl,
    _has_weight_scale,
    _is_w8a8_dynamic,
    _wo_a_weight_for_transpose_batchmatmul,
)
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import AscendDeviceType


def test_is_w8a8_dynamic_detects_method_without_weight_scale():
    quant_method = AscendW8A8DynamicLinearMethod.__new__(AscendW8A8DynamicLinearMethod)
    linear = SimpleNamespace(quant_method=SimpleNamespace(quant_method=quant_method))

    assert not _has_weight_scale(linear)
    assert _is_w8a8_dynamic(linear)

    linear.weight_scale = object()

    assert _has_weight_scale(linear)
    assert _is_w8a8_dynamic(linear)


def test_wo_a_weight_for_transpose_batchmatmul_reshapes_2d_to_gdr():
    # 2D ColumnParallelLinear layout: [G * R, D] = [3 * 2, 4]
    weight = torch.arange(24).reshape(6, 4)

    grouped = _wo_a_weight_for_transpose_batchmatmul(weight, n_local_groups=3, o_lora_rank=2)

    assert grouped.shape == (3, 4, 2)
    # First group rows [0:2] become [D, R] after transpose(1, 2).
    torch.testing.assert_close(grouped[0], weight[:2].transpose(0, 1))


def test_wo_a_weight_for_transpose_batchmatmul_keeps_3d_weight():
    weight = torch.empty(3, 4, 2)

    assert _wo_a_weight_for_transpose_batchmatmul(weight, n_local_groups=3, o_lora_rank=2) is weight


def test_wo_a_weight_for_transpose_batchmatmul_rejects_unexpected_rank():
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        _wo_a_weight_for_transpose_batchmatmul(torch.empty(4), n_local_groups=2, o_lora_rank=2)


def test_a5_bf16_o_proj_uses_npu_transpose_batchmatmul_with_gdr_weight():
    impl = SimpleNamespace(
        n_local_groups=3,
        o_lora_rank=2,
        wo_a=SimpleNamespace(weight=torch.arange(24, dtype=torch.float32).reshape(6, 4)),
        wo_b=lambda x: x,
    )
    o_proj_input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    output = torch.empty(2, 6)
    expected = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    with (
        mock.patch(
            "vllm_ascend.attention.dsa_v1.get_ascend_device_type",
            return_value=AscendDeviceType.A5,
        ),
        mock.patch("vllm_ascend.attention.dsa_v1.oproj_tp_enable", return_value=False),
        mock.patch("vllm_ascend.attention.dsa_v1.olora_tp_enable", return_value=False),
        mock.patch("vllm_ascend.attention.dsa_v1.torch_npu.npu_transpose_quant_batchmatmul") as quant_batch_matmul,
        mock.patch(
            "vllm_ascend.attention.dsa_v1.torch_npu.npu_transpose_batchmatmul",
            return_value=expected,
        ) as batch_matmul,
    ):
        result = AscendDSAImpl._forward_o_proj(impl, o_proj_input, output)

    quant_batch_matmul.assert_not_called()
    batch_matmul.assert_called_once()
    passed_weight = batch_matmul.call_args.args[1]
    assert passed_weight.shape == (3, 4, 2)
    torch.testing.assert_close(result, expected)


def test_a3_o_proj_keeps_npu_transpose_batchmatmul():
    impl = SimpleNamespace(
        n_local_groups=3,
        o_lora_rank=2,
        wo_a=SimpleNamespace(weight=torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)),
        wo_b=lambda x: x,
    )
    o_proj_input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    output = torch.empty(2, 6)
    expected = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    with (
        mock.patch(
            "vllm_ascend.attention.dsa_v1.get_ascend_device_type",
            return_value=AscendDeviceType.A3,
        ),
        mock.patch("vllm_ascend.attention.dsa_v1.oproj_tp_enable", return_value=False),
        mock.patch("vllm_ascend.attention.dsa_v1.olora_tp_enable", return_value=False),
        mock.patch(
            "vllm_ascend.attention.dsa_v1.torch_npu.npu_transpose_batchmatmul",
            return_value=expected,
        ) as batch_matmul,
    ):
        result = AscendDSAImpl._forward_o_proj(impl, o_proj_input, output)

    batch_matmul.assert_called_once()
    torch.testing.assert_close(result, expected)
