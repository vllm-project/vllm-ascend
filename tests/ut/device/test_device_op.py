from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor


@pytest.mark.parametrize("query_dtype", [torch.int8, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float16])
def test_a5_quant_indexer_weights_dtype(query_dtype, weights_dtype):
    query = torch.zeros(2, 32, 128, dtype=query_dtype)
    query_scale = torch.ones(2, 32, 1, dtype=torch.float16)
    key = torch.zeros(4, 128, 1, 128, dtype=query_dtype)
    key_scale = torch.ones(4, 128, 1, 1, dtype=torch.float16)
    weights = torch.arange(64).reshape(2, 32).to(weights_dtype) / 64
    metadata = SimpleNamespace(block_table=torch.arange(4, dtype=torch.int32).reshape(1, 4))
    query_lengths = torch.tensor([2], dtype=torch.int32)
    key_lengths = torch.tensor([512], dtype=torch.int32)
    expected = torch.zeros(2, 1, 2048, dtype=torch.int32)

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_quant_lightning_indexer",
        return_value=expected,
        create=True,
    ) as indexer:
        result = A5DeviceAdaptor.indexer_select_post_process(
            query,
            query_scale,
            query.shape,
            weights,
            (key, key_scale),
            0,
            1,
            metadata,
            query_lengths,
            key_lengths,
            True,
            True,
        )

    indexer.assert_called_once()
    kwargs = indexer.call_args.kwargs
    expected_dtype = torch.float16 if query_dtype == torch.int8 else weights_dtype
    torch.testing.assert_close(kwargs["weights"], weights.to(expected_dtype))
    if expected_dtype == weights_dtype:
        assert kwargs["weights"] is weights
    assert weights.dtype == weights_dtype
    assert kwargs["query"].dtype == query_dtype
    assert kwargs["query"].data_ptr() == query.data_ptr()
    assert kwargs["key"] is key
    assert kwargs["query_dequant_scale"].shape == (2, 32)
    assert kwargs["query_dequant_scale"].data_ptr() == query_scale.data_ptr()
    assert kwargs["key_dequant_scale"].shape == (4, 128, 1)
    assert kwargs["key_dequant_scale"].data_ptr() == key_scale.data_ptr()
    assert kwargs["block_table"] is metadata.block_table
    assert kwargs["actual_seq_lengths_query"] is query_lengths
    assert kwargs["actual_seq_lengths_key"] is key_lengths
    assert result is expected


@pytest.mark.parametrize("enable_sparse_li_c8", [False, True])
def test_a5_unquantized_indexer_preserves_weights(enable_sparse_li_c8):
    query = torch.zeros(2, 32, 128, dtype=torch.bfloat16)
    key = torch.zeros(4, 128, 1, 128, dtype=torch.bfloat16)
    weights = torch.ones(2, 32, dtype=torch.bfloat16)
    metadata = SimpleNamespace(block_table=torch.arange(4, dtype=torch.int32).reshape(1, 4))
    expected = torch.zeros(2, 1, 2048, dtype=torch.int32)

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_lightning_indexer",
        return_value=(expected, None),
        create=True,
    ) as indexer:
        result = A5DeviceAdaptor.indexer_select_post_process(
            query,
            None,
            query.shape,
            weights,
            (key, None),
            0,
            1,
            metadata,
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([512], dtype=torch.int32),
            enable_sparse_li_c8,
            True,
        )

    indexer.assert_called_once()
    assert indexer.call_args.kwargs["weights"] is weights
    assert result is expected


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
