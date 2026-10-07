import math
from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor


@pytest.mark.parametrize("adaptor", [BaseDeviceAdaptor, A5DeviceAdaptor])
@pytest.mark.parametrize("block_table_mode", ["pa", "none", "omitted"])
@pytest.mark.parametrize("contiguous", [False, True])
def test_fia_only_contiguizes_non_pa_inputs(adaptor, block_table_mode, contiguous):
    use_pa = block_table_mode == "pa"
    omit_block_table = block_table_mode == "omitted"
    key = torch.randn(2, 3, 4).transpose(0, 1)
    value = torch.randn_like(key)
    if contiguous:
        key = key.contiguous()
        value = value.contiguous()
    block_table = torch.tensor([[0]], dtype=torch.int32) if use_pa else None
    kwargs = {} if omit_block_table else {"block_table": block_table}
    expected = (object(), object())

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_fused_infer_attention_score", return_value=expected
    ) as mock_fia:
        result = adaptor.npu_fused_infer_attention_score(
            query=torch.randn(3, 2, 4),
            key=key,
            value=value,
            attn_metadata=None,
            key_cache=None,
            value_cache=None,
            current_key=key,
            current_value=value,
            num_heads=2,
            num_key_value_heads=2,
            head_size=4,
            scale=0.5,
            is_prefill_no_cache=not use_pa,
            **kwargs,
        )

    assert result is expected
    mock_fia.assert_called_once()
    call_kwargs = mock_fia.call_args.kwargs
    for name, original in (("key", key), ("value", value)):
        actual = call_kwargs[name]
        torch.testing.assert_close(actual, original)
        if use_pa or contiguous:
            assert actual is original
        else:
            assert actual.is_contiguous()
            assert actual.data_ptr() != original.data_ptr()
        if use_pa:
            assert actual.stride() == original.stride()
    assert call_kwargs.get("block_table") is block_table
    assert ("block_table" in call_kwargs) is (not omit_block_table)


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


def test_base_reshape_and_cache_uses_custom_scatter_for_bnsd():
    key = torch.randn(2, 8, 64)
    value = torch.randn_like(key)
    key_cache = torch.empty(4, 8, 128, 64)
    value_cache = torch.empty_like(key_cache)
    slot_mapping = torch.arange(2, dtype=torch.int32)

    with (
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_scatter_pa_kv_cache",
            create=True,
        ) as mock_custom_scatter,
        mock.patch("vllm_ascend.device.device_op.torch_npu.npu_scatter_pa_kv_cache") as mock_public_scatter,
    ):
        BaseDeviceAdaptor.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            use_bnsd=True,
        )

    mock_public_scatter.assert_not_called()
    mock_custom_scatter.assert_called_once()
    assert mock_custom_scatter.call_args.args[2] is key_cache
    assert mock_custom_scatter.call_args.args[3] is value_cache
    assert mock_custom_scatter.call_args.kwargs["cache_mode"] == "Norm"
    assert mock_custom_scatter.call_args.kwargs["scatter_mode"] == "NHSD"


def test_a5_reshape_and_cache_uses_bsnd_view_for_bnsd():
    key = torch.randn(2, 8, 64)
    value = torch.randn_like(key)
    key_cache = torch.empty(4, 8, 128, 64)
    value_cache = torch.empty_like(key_cache)
    slot_mapping = torch.arange(2, dtype=torch.int32)

    with (
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_scatter_pa_kv_cache",
            create=True,
        ) as mock_custom_scatter,
        mock.patch("vllm_ascend.device.device_op.torch_npu.npu_scatter_pa_kv_cache") as mock_public_scatter,
    ):
        A5DeviceAdaptor.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            use_bnsd=True,
        )

    mock_custom_scatter.assert_not_called()
    mock_public_scatter.assert_called_once()
    call_kwargs = mock_public_scatter.call_args.kwargs
    assert call_kwargs["key_cache"].shape == (4, 128, 8, 64)
    assert call_kwargs["value_cache"].shape == (4, 128, 8, 64)
    assert not call_kwargs["key_cache"].is_contiguous()
    assert not call_kwargs["value_cache"].is_contiguous()
    assert call_kwargs["key_cache"].data_ptr() == key_cache.data_ptr()
    assert call_kwargs["value_cache"].data_ptr() == value_cache.data_ptr()


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


@pytest.mark.parametrize(
    "shape,dim,indices,value,dtype",
    [
        ((8,), 0, [1, 4, 6], True, torch.bool),
        ((4, 3), 0, [1, 3], -1, torch.int32),
        ((4, 3), 1, [0, 2], 7, torch.int64),
        ((2, 3, 4), -2, [-1, 0], -3.5, torch.float32),
        ((2, 3), 1, [], 9, torch.int64),
        ((5,), 0, [1, 1, 3], 6, torch.int64),
    ],
)
def test_a5_index_fill_matches_torch_index_fill(shape, dim, indices, value, dtype):
    source = torch.arange(math.prod(shape)).reshape(shape)
    source = (source % 2).bool() if dtype is torch.bool else source.to(dtype)
    index = torch.tensor(indices, dtype=torch.int64)
    expected = source.clone().index_fill_(dim, index, value)
    actual_input = source.clone()

    actual = A5DeviceAdaptor.index_fill(actual_input, dim, index, value)

    assert actual is actual_input
    torch.testing.assert_close(actual, expected)


def test_a5_index_fill_uses_scatter():
    tensor = mock.Mock()
    tensor.dim.return_value = 1
    tensor.size.return_value = 8
    tensor.shape = (8,)

    result = A5DeviceAdaptor.index_fill(tensor, 0, torch.tensor([1, 3]), 5)

    assert result is tensor
    tensor.scatter_.assert_called_once()
    tensor.index_fill_.assert_not_called()
