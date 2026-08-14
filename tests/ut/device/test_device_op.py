from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor


def _packed_w8a8_gmm_swiglu_kwargs(*, swiglu_limit=0.0):
    return {
        "x": torch.zeros(2, 4, dtype=torch.int8),
        "weight": torch.zeros(3, 4, 8, dtype=torch.int8),
        "group_list": torch.tensor([1, 1, 2], dtype=torch.int64),
        "group_list_type": 0,
        "weight_scale": torch.ones(3, 8),
        "x_scale": torch.ones(2),
        "act_quant_type": torch.int8,
        "swiglu_limit": swiglu_limit,
    }


def test_gmm_swiglu_normalizes_single_tensor_inputs_for_custom_v2():
    expected = (torch.zeros(2, 4, dtype=torch.int8), torch.ones(2))
    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2",
            return_value=expected,
            create=True,
        ) as custom_v2,
    ):
        output = BaseDeviceAdaptor.npu_grouped_matmul_swiglu_quant(**_packed_w8a8_gmm_swiglu_kwargs())

    assert output[0] is expected[0]
    assert output[1] is expected[1]
    call_kwargs = custom_v2.call_args.kwargs
    assert len(call_kwargs["weight"]) == 1
    assert len(call_kwargs["weight_scale"]) == 1
    assert call_kwargs["weight_assist_matrix"] is None
    assert call_kwargs["dequant_mode"] == 0
    assert call_kwargs["group_list_type"] == 0


@pytest.mark.parametrize(
    ("single_tensor", "per_group", "expected_group_list_type"),
    [(True, False, 0), (True, True, 0), (False, False, 1), (False, True, 1)],
)
def test_gmm_swiglu_custom_v2_covers_tensor_and_scale_layouts(single_tensor, per_group, expected_group_list_type):
    num_experts, hidden_size, intermediate_size = 3, 4, 8
    if single_tensor:
        weight = torch.zeros(num_experts, hidden_size, intermediate_size, dtype=torch.int8)
        scale_shape = (num_experts, 2, intermediate_size) if per_group else (num_experts, intermediate_size)
        weight_scale = torch.ones(scale_shape)
        bias = torch.zeros(num_experts, intermediate_size)
    else:
        weight = [torch.zeros(hidden_size, intermediate_size, dtype=torch.int8) for _ in range(num_experts)]
        scale_shape = (2, intermediate_size) if per_group else (intermediate_size,)
        weight_scale = [torch.ones(scale_shape) for _ in range(num_experts)]
        bias = [torch.zeros(intermediate_size) for _ in range(num_experts)]

    expected = (torch.zeros(2, 4, dtype=torch.int8), torch.ones(2))
    with mock.patch(
        "vllm_ascend.device.device_op.torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2",
        return_value=expected,
        create=True,
    ) as custom_v2:
        output = BaseDeviceAdaptor.npu_grouped_matmul_swiglu_quant(
            x=torch.zeros(2, hidden_size, dtype=torch.int8),
            weight=weight,
            group_list=torch.tensor([1, 1, 0] if expected_group_list_type == 1 else [1, 2, 2]),
            group_list_type=expected_group_list_type,
            weight_scale=weight_scale,
            x_scale=torch.ones(2),
            bias=bias,
            act_quant_type=torch.int8,
            swiglu_limit=1.0,
        )

    assert output is expected
    call_kwargs = custom_v2.call_args.kwargs
    assert len(call_kwargs["weight"]) == (1 if single_tensor else num_experts)
    assert len(call_kwargs["weight_scale"]) == len(call_kwargs["weight"])
    assert len(call_kwargs["weight_assist_matrix"]) == len(call_kwargs["weight"])
    assert call_kwargs["dequant_mode"] == int(per_group)
    assert call_kwargs["group_list_type"] == 0
    torch.testing.assert_close(call_kwargs["group_list"], torch.tensor([1, 2, 2]))
    assert call_kwargs["swiglu_limit"] == 1.0


@pytest.mark.parametrize("use_mxfp_quant", [False, True], ids=["pertoken", "mx"])
def test_a5_gmm_swiglu_routes_supported_quant_modes_to_custom_v2(use_mxfp_quant):
    num_tokens, num_experts, hidden_size, intermediate_size = 2, 3, 4, 8
    expected = (torch.zeros(num_tokens, intermediate_size // 2), torch.ones(num_tokens))
    with (
        mock.patch("vllm_ascend.device.device_op.enable_gmm_swiglu_quant_v2_custom_op", return_value=True),
        mock.patch.object(BaseDeviceAdaptor, "npu_grouped_matmul_swiglu_quant", return_value=expected) as custom_v2,
        mock.patch.object(A5DeviceAdaptor, "maybe_normalize_mxfp_scale_layout", side_effect=lambda scale: scale),
    ):
        output = A5DeviceAdaptor.npu_grouped_matmul_swiglu_quant(
            x=torch.zeros(num_tokens, hidden_size),
            weight=torch.zeros(num_experts, hidden_size, intermediate_size),
            group_list=torch.tensor([1, 0, 1]),
            group_list_type=1,
            weight_scale=torch.ones(num_experts, 1, intermediate_size, 2)
            if use_mxfp_quant
            else torch.ones(num_experts, intermediate_size),
            x_scale=torch.ones(num_tokens, 1, 2) if use_mxfp_quant else torch.ones(num_tokens),
            use_mxfp_quant=use_mxfp_quant,
            act_quant_type=torch.float8_e4m3fn,
            weight_quant_type=torch.float8_e4m3fn,
        )

    assert output[0] is expected[0]
    assert output[1] is expected[1]
    call_kwargs = custom_v2.call_args.kwargs
    expected_mode = 2 if use_mxfp_quant else 0
    assert call_kwargs["dequant_mode"] == expected_mode
    assert call_kwargs["quant_mode"] == expected_mode
    assert call_kwargs["dequant_dtype"] == torch.float32
    assert call_kwargs["quant_dtype"] == torch.float8_e4m3fn


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
