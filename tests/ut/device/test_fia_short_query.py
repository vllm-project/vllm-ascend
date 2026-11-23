# SPDX-License-Identifier: Apache-2.0
from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import BaseDeviceAdaptor
from vllm_ascend.device.fia_short_query import pad_short_fia_queries, unpad_short_fia_output


@pytest.mark.parametrize("lengths", [[9], [10], [15], [0, 9, 1, 16], [4, 12]])
def test_padding_preserves_causal_attention(lengths):
    torch.manual_seed(19)
    query = torch.randn(sum(lengths), 2, 8)
    cumulative = torch.tensor(lengths).cumsum(0).tolist()
    padded, padded_cumulative, slices = pad_short_fia_queries(query, cumulative)
    torch.testing.assert_close(unpad_short_fia_output(padded, slices), query)
    expected, actual = [], []
    source_start = padded_start = 0
    for index, length in enumerate(lengths):
        end = padded_cumulative[index]
        padded_length = end - padded_start
        key, value = torch.randn(41, 2, 8), torch.randn(41, 2, 8)
        for q, outputs in ((query[source_start : source_start + length], expected), (padded[padded_start:end], actual)):
            # Bottom-right aligned causal masking: position K-Q+j.
            scores = torch.einsum("thd,shd->hts", q, key) / 8**0.5
            mask = torch.arange(41)[None, :] > (41 - len(q) + torch.arange(len(q)))[:, None]
            scores.masked_fill_(mask, float("-inf"))
            outputs.append(torch.einsum("hts,shd->thd", scores.softmax(-1), value))
        assert padded_length == (32 if length else 0)
        source_start += length
        padded_start = end
    torch.testing.assert_close(unpad_short_fia_output(torch.cat(actual), slices), torch.cat(expected))


@pytest.mark.parametrize("lengths", [None, [], [8], [16], [17], [9, 26], [10, 9], [11]])
def test_unsupported_or_inconsistent_lengths_are_untouched(lengths):
    query = torch.zeros(10, 2, 8)
    assert pad_short_fia_queries(query, lengths) is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("query_length", [8, 9, 10, 15, 16, 17])
@pytest.mark.parametrize("extra", [{}, {"actual_seq_lengths_kv": [9]}, {"query_rope": True}, {"pse_shift": True}])
def test_wrapper_dispatch_and_output_shape(enabled, query_length, extra):
    query = torch.randn(query_length, 16, 256, dtype=torch.bfloat16)
    cache = torch.empty(1, 2, 128, 256, dtype=query.dtype)
    captured = {}

    def native(**kwargs):
        captured.update(kwargs)
        return kwargs["query"], torch.empty(0)

    with (
        mock.patch("vllm_ascend.device.device_op.envs.VLLM_ASCEND_FIA_SHORT_QUERY_WORKAROUND", enabled),
        mock.patch("vllm_ascend.device.device_op.torch_npu.npu_fused_infer_attention_score", side_effect=native),
    ):
        kwargs = dict(actual_seq_lengths_kv=[128], **{k: v for k, v in extra.items() if k != "actual_seq_lengths_kv"})
        kwargs.update(extra)
        output, _ = BaseDeviceAdaptor.npu_fused_infer_attention_score(
            query,
            cache,
            cache,
            None,
            key_cache=cache,
            value_cache=cache,
            current_key=cache,
            current_value=cache,
            num_heads=16,
            num_key_value_heads=2,
            head_size=256,
            scale=0.0625,
            is_prefill_no_cache=False,
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            input_layout="TND",
            sparse_mode=3,
            actual_seq_lengths=[query_length],
            **kwargs,
        )
    expected_length = 32 if enabled and not extra and 9 <= query_length < 16 else query_length
    assert captured["actual_seq_lengths"] == [expected_length]
    assert captured["key"] is cache and captured["value"] is cache
    assert captured["actual_seq_lengths_kv"] == extra.get("actual_seq_lengths_kv", [128])
    torch.testing.assert_close(output, query)
