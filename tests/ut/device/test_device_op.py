from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor


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


def _make_lightning_indexer_inputs(*, use_torch_npu: bool = False):
    return {
        "q_li": object(),
        "q_li_scale": None,
        "q_li_shape_ori": None,
        "weights": object(),
        "kv_cache": (object(), object()),
        "indexer_k_cache_idx": 0,
        "indexer_scale_cache_idx": 1,
        "attn_metadata": SimpleNamespace(block_table=object()),
        "actual_seq_lengths_query": object(),
        "actual_seq_lengths_key": object(),
        "enable_sparse_li_c8": False,
        "use_torch_npu_lightning_indexer": use_torch_npu,
    }


def test_base_lightning_indexer_default_keeps_index_only_contract():
    inputs = _make_lightning_indexer_inputs()
    expected_indices = object()
    unused_scores = object()

    with mock.patch.object(
        torch.ops._C_ascend,
        "npu_lightning_indexer",
        create=True,
        return_value=(expected_indices, unused_scores),
    ) as mock_indexer:
        result = BaseDeviceAdaptor.indexer_select_post_process(**inputs)

    assert result is expected_indices
    assert "return_value" not in mock_indexer.call_args.kwargs


def test_base_custom_lightning_indexer_exposes_selected_scores_when_requested():
    inputs = _make_lightning_indexer_inputs()
    expected_indices = object()
    expected_scores = object()

    with mock.patch.object(
        torch.ops._C_ascend,
        "npu_lightning_indexer",
        create=True,
        return_value=(expected_indices, expected_scores),
    ) as mock_indexer:
        result = BaseDeviceAdaptor.indexer_select_post_process(
            **inputs,
            return_selected_scores=True,
        )

    assert result == (expected_indices, expected_scores)
    assert mock_indexer.call_args.kwargs["return_value"] is True


def test_base_torch_npu_lightning_indexer_exposes_selected_scores_when_requested():
    inputs = _make_lightning_indexer_inputs(use_torch_npu=True)
    expected_indices = object()
    expected_scores = object()

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_lightning_indexer",
        return_value=(expected_indices, expected_scores),
        create=True,
    ) as mock_indexer:
        result = BaseDeviceAdaptor.indexer_select_post_process(
            **inputs,
            return_selected_scores=True,
        )

    assert result == (expected_indices, expected_scores)
    assert mock_indexer.call_args.kwargs["return_value"] is True


def test_base_quantized_lightning_indexer_rejects_selected_score_request():
    inputs = _make_lightning_indexer_inputs()
    inputs.update(
        q_li_scale=object(),
        q_li_shape_ori=(1, 1, 128),
        enable_sparse_li_c8=True,
    )

    with (
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_lightning_indexer_quant",
            create=True,
        ) as mock_indexer,
        pytest.raises(NotImplementedError, match="quantized lightning indexer"),
    ):
        BaseDeviceAdaptor.indexer_select_post_process(
            **inputs,
            return_selected_scores=True,
        )

    mock_indexer.assert_not_called()


def test_a5_lightning_indexer_exposes_selected_scores_when_requested():
    inputs = _make_lightning_indexer_inputs(use_torch_npu=True)
    expected_indices = object()
    expected_scores = object()

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_lightning_indexer",
        return_value=(expected_indices, expected_scores),
        create=True,
    ) as mock_indexer:
        result = A5DeviceAdaptor.indexer_select_post_process(
            **inputs,
            return_selected_scores=True,
        )

    assert result == (expected_indices, expected_scores)
    assert mock_indexer.call_args.kwargs["return_value"] is True


def test_a5_quantized_lightning_indexer_rejects_selected_score_request():
    inputs = _make_lightning_indexer_inputs(use_torch_npu=True)
    inputs.update(
        q_li_scale=object(),
        q_li_shape_ori=(1, 1, 128),
        enable_sparse_li_c8=True,
    )

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_quant_lightning_indexer",
            create=True,
        ) as mock_indexer,
        pytest.raises(NotImplementedError, match="quantized lightning indexer"),
    ):
        A5DeviceAdaptor.indexer_select_post_process(
            **inputs,
            return_selected_scores=True,
        )

    mock_indexer.assert_not_called()
