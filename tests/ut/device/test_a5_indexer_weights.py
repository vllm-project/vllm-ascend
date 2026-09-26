# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor


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
