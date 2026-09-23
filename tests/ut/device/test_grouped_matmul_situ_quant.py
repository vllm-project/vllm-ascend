from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor
from vllm_ascend.quantization.quant_type import QuantType


def test_base_device_rejects_grouped_matmul_situ_quant():
    with pytest.raises(RuntimeError, match="only supported on Ascend A5"):
        BaseDeviceAdaptor.npu_grouped_matmul_situ_quant(
            x=torch.ones(1, 4),
            weight=torch.ones(1, 8, 2),
            weight_scale=torch.ones(1, 8),
            x_scale=torch.ones(1, 1),
            group_list=torch.ones(1, dtype=torch.int64),
            group_list_type=1,
            beta=4.0,
            linear_beta=25.0,
            mxfp_quant_dtype=QuantType.W4A8MXFP,
        )


@pytest.mark.parametrize("use_list", [False, True])
def test_a5_grouped_matmul_situ_quant_calls_registered_op(use_list):
    x = torch.ones(1, 4)
    weight_base = torch.ones(1, 8, 2)
    weight = [weight_base[0]] if use_list else weight_base.transpose(1, 2)
    weight_scale_base = torch.ones(1, 8, 2, 2)
    weight_scale_view = weight_scale_base.transpose(-3, -2)
    weight_scale = [weight_scale_view[0]] if use_list else weight_scale_view
    x_scale = torch.ones(1, 1)
    group_list = torch.ones(1, dtype=torch.int64)
    output = torch.ones(1, 4)
    output_scale = torch.ones(1, 1)

    op = mock.MagicMock(return_value=(output, output_scale))
    op.list = mock.MagicMock(return_value=(output, output_scale))
    with mock.patch(
        "vllm_ascend.device.device_op.torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz",
        op,
        create=True,
    ):
        result = A5DeviceAdaptor.npu_grouped_matmul_situ_quant(
            x=x,
            weight=weight,
            weight_scale=weight_scale,
            x_scale=x_scale,
            group_list=group_list,
            group_list_type=0,
            beta=4.0,
            linear_beta=25.0,
            mxfp_quant_dtype=QuantType.W4A8MXFP,
        )

    assert result[0] is output
    assert result[1] is output_scale
    assert result[2] is None
    called_op = op.list if use_list else op
    other_op = op if use_list else op.list
    called_op.assert_called_once()
    other_op.assert_not_called()
    args = called_op.call_args.args
    assert args[0] is x
    if use_list:
        assert args[1][0] is weight[0]
        assert args[2][0].data_ptr() == weight_scale_base[0].data_ptr()
    else:
        assert args[1].data_ptr() == weight_base.data_ptr()
        assert args[2].data_ptr() == weight_scale_base.data_ptr()
    assert args[3] is None
    assert args[4] is None
    assert args[5] is x_scale
    assert args[6] is None
    assert args[7] is group_list
    assert args[8:] == (1, 0, 1, 0, None, 4.0, 25.0)


def test_a5_grouped_matmul_situ_quant_rejects_other_quantization():
    with pytest.raises(RuntimeError, match="requires W4A8 MXFP"):
        A5DeviceAdaptor.npu_grouped_matmul_situ_quant(
            x=torch.ones(1, 4),
            weight=torch.ones(1, 8, 2),
            weight_scale=torch.ones(1, 8),
            x_scale=torch.ones(1, 1),
            group_list=torch.ones(1, dtype=torch.int64),
            group_list_type=1,
            beta=4.0,
            linear_beta=25.0,
            mxfp_quant_dtype=QuantType.W8A8MXFP,
        )


class _FakeTensor:
    def __init__(self, dtype):
        self.dtype = dtype
        self.view_calls = []

    def view(self, dtype):
        self.view_calls.append(dtype)
        return _FakeTensor(dtype)


def test_a5_restores_mxfp_semantic_dtype_for_tensor_collections():
    semantic_dtype = object()
    uint8_tensor = _FakeTensor(torch.uint8)
    typed_tensor = _FakeTensor(torch.float16)

    restored_list = A5DeviceAdaptor._restore_mxfp_semantic_dtype([uint8_tensor, typed_tensor], semantic_dtype)
    restored_tuple = A5DeviceAdaptor._restore_mxfp_semantic_dtype((uint8_tensor, typed_tensor), semantic_dtype)

    assert isinstance(restored_list, list)
    assert isinstance(restored_tuple, tuple)
    assert uint8_tensor.view_calls == [semantic_dtype, semantic_dtype]
    assert typed_tensor.view_calls == []
    assert restored_list[1] is typed_tensor
    assert restored_tuple[1] is typed_tensor


def test_a5_realigns_gmm_situ_weight_scale_without_copy():
    base = torch.ones(2, 4, 3, 2)
    transposed = base.transpose(-3, -2)

    aligned = A5DeviceAdaptor._align_gmm_situ_weight_scale(transposed)

    assert aligned.is_contiguous()
    assert aligned.shape == base.shape
    assert aligned.data_ptr() == base.data_ptr()
