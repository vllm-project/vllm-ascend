from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor
from vllm_ascend.quantization.quant_type import QuantType


def test_reshape_and_cache_makes_scatter_inputs_contiguous():
    key = torch.randn(2, 3, 4).transpose(0, 1)
    value = torch.randn(2, 3, 4).transpose(0, 1)
    slot_mapping = torch.arange(8, dtype=torch.int32)[::2]
    key_cache = object()
    value_cache = object()

    assert not key.is_contiguous()
    assert not value.is_contiguous()
    assert not slot_mapping.is_contiguous()

    with mock.patch("vllm_ascend.device.device_op.torch_npu.npu_scatter_pa_kv_cache") as mock_scatter:
        BaseDeviceAdaptor.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    mock_scatter.assert_called_once()
    call_kwargs = mock_scatter.call_args.kwargs
    assert call_kwargs["key"] is not key
    assert call_kwargs["value"] is not value
    assert call_kwargs["slot_mapping"] is not slot_mapping
    assert call_kwargs["key"].is_contiguous()
    assert call_kwargs["value"].is_contiguous()
    assert call_kwargs["slot_mapping"].is_contiguous()
    torch.testing.assert_close(call_kwargs["key"], key)
    torch.testing.assert_close(call_kwargs["value"], value)
    torch.testing.assert_close(call_kwargs["slot_mapping"], slot_mapping)
    assert call_kwargs["key_cache"] is key_cache
    assert call_kwargs["value_cache"] is value_cache
    assert call_kwargs["cache_mode"] == "Norm"


def test_kv_cache_load_makes_seq_lens_contiguous():
    cache_kv_c = object()
    cache_k_pe = object()
    block_table = object()
    context_seq_len_npu = torch.arange(8, dtype=torch.int32)[::2]
    seq_starts = object()
    key = object()
    value = object()

    assert not context_seq_len_npu.is_contiguous()

    with mock.patch("vllm_ascend.device.device_op.torch_npu.npu_gather_pa_kv_cache") as mock_gather:
        BaseDeviceAdaptor.kv_cache_load(
            cache_kv_c,
            cache_k_pe,
            block_table,
            context_seq_len_npu,
            seq_starts,
            key,
            value,
        )

    mock_gather.assert_called_once()
    call_args = mock_gather.call_args.args
    assert call_args[0] is cache_kv_c
    assert call_args[1] is cache_k_pe
    assert call_args[2] is block_table
    assert call_args[3] is not context_seq_len_npu
    assert call_args[3].is_contiguous()
    torch.testing.assert_close(call_args[3], context_seq_len_npu)
    assert mock_gather.call_args.kwargs["seq_offset"] is seq_starts
    assert mock_gather.call_args.kwargs["key"] is key
    assert mock_gather.call_args.kwargs["value"] is value


def test_npu_flash_attention_uses_fusion_attention_for_fp32():
    query = torch.randn(5, 4, 64, dtype=torch.float32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
            return_value=(expected,),
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    mock_flash_attention.assert_not_called()
    mock_fusion_attention.assert_called_once()
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
    assert call_kwargs["head_num"] == 4
    assert call_kwargs["scale"] == 0.125
    assert call_kwargs["input_layout"] == "TND"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_flash_attention_uses_unpad_attention_for_low_precision(dtype):
    query = torch.randn(5, 4, 64, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)

    def fake_flash_attention(*, query, key, value, seq_len, scale_value, num_heads, num_kv_heads, out):
        out.copy_(query + 1)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            side_effect=fake_flash_attention,
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    mock_fusion_attention.assert_not_called()
    mock_flash_attention.assert_called_once()
    call_kwargs = mock_flash_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["seq_len"] is seq_lens_cpu
    assert call_kwargs["num_heads"] == 4
    assert call_kwargs["num_kv_heads"] == 4
    assert call_kwargs["scale_value"] == 0.125
    torch.testing.assert_close(output, query + 1)


def test_a5_npu_flash_attention_uses_python_sequence_lengths():
    query = torch.randn(5, 4, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        return_value=(expected,),
    ) as mock_fusion_attention:
        output = A5DeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]


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

    restored_list = A5DeviceAdaptor._restore_mxfp_semantic_dtype(  # type: ignore[arg-type]
        [uint8_tensor, typed_tensor], semantic_dtype
    )
    restored_tuple = A5DeviceAdaptor._restore_mxfp_semantic_dtype(  # type: ignore[arg-type]
        (uint8_tensor, typed_tensor), semantic_dtype
    )

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
