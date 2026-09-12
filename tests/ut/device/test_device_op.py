from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import (
    A5DeviceAdaptor,
    Ascend310PDeviceAdaptor,
    BaseDeviceAdaptor,
)


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


@pytest.mark.parametrize(
    ("adaptor", "expected_quant_mode", "expected_scale_dtype"),
    [
        (BaseDeviceAdaptor, 2, torch.float16),
        (Ascend310PDeviceAdaptor, 2, torch.float16),
        (A5DeviceAdaptor, 1, torch.float32),
    ],
)
def test_dsa_indexer_quant_mode_matches_the_prepared_dtypes(adaptor, expected_quant_mode, expected_scale_dtype):
    # The indexer op rejects weights and dequant scales whose dtypes disagree
    # with the quant_mode it is called with. The modes come from
    # csrc/attention/quant_lightning_indexer_v2: 1 is fp8, 2 is int8.
    assert adaptor.get_dsa_indexer_quant_mode() == expected_quant_mode
    assert adaptor.prepare_dsa_indexer_weights(torch.ones(2, 2)).dtype is expected_scale_dtype
    assert adaptor.prepare_dsa_indexer_key_scale(torch.ones(1, 1, 1, 1)).dtype is expected_scale_dtype
