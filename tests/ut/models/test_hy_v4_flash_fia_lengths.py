# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.hy_v4_flash import attention_fusions, decode_fusions
from vllm_ascend.models.hy_v4_flash.fia_lengths import FIALengthProvider, compact_fia_rows


def metadata(lengths, queries=6):
    return SimpleNamespace(
        seq_lens=tuple(lengths), query_start_loc=tuple(range(0, (len(lengths) + 1) * queries, queries))
    )


@pytest.mark.parametrize("window", [None, 128])
@pytest.mark.parametrize("seq_len", [6, 127, 128, 129, 2044, 2048, 2049, 2050, 2054, 131072])
def test_lengths_match_candidate_sets(seq_len, window):
    provider = FIALengthProvider("attn", 0, 6, 2048, window)
    expected = []
    for visible in range(seq_len - 6, seq_len):
        if window is not None:
            selected = range(max(0, visible - window + 1), visible + 1)
        elif visible // 4 > 511:
            selected = list(range(511 * 4)) + list(range(visible // 4 * 4, visible + 1))
        else:
            selected = range(visible + 1)
        expected.append(len(selected))
    assert provider.resolve(metadata([seq_len]))["actual_seq_kvlen"] == expected


def test_replay_lengths_follow_current_requests_and_padding():
    provider = FIALengthProvider("attn", 0, 18, 2048, None, graph_queries=6)
    captured = provider.resolve({"attn": metadata([2054, 2054, 2054])})
    replay = provider.resolve({"attn": metadata([2055, 6])})
    assert captured != replay
    assert replay["actual_seq_kvlen"] == [2046, 2047, 2048, 2045, 2046, 2047, 1, 2, 3, 4, 5, 6] + [1] * 6


def test_request_slice_and_provider_identity():
    a = FIALengthProvider("attn", 6, 6, 128, 128)
    b = FIALengthProvider("attn", 6, 6, 128, 128)
    assert a != b  # Repeated layer invocations each register one task/provider.
    assert a.resolve(metadata([1000, 6]))["actual_seq_kvlen"] == [1, 2, 3, 4, 5, 6]


def test_flash_graph_backend_updates_host_lengths():
    from vllm_ascend.models.hy_v4_flash.attention import FlashAttentionBackend
    from vllm_ascend.utils import use_updatable_graph

    assert use_updatable_graph(FlashAttentionBackend)


def test_compaction_retains_order_and_zeros_all_invalid_rows():
    selected = torch.tensor([[-1, 8, -1, 2, 3], [-1, -1, -1, -1, -1]])
    kv = torch.arange(20).view(2, 5, 1, 2).float()
    nope, rope = compact_fia_rows(selected, kv, kv + 100)
    torch.testing.assert_close(nope[0, :3], kv[0, [1, 3, 4]])
    torch.testing.assert_close(rope[0, :3], kv[0, [1, 3, 4]] + 100)
    assert torch.count_nonzero(nope[0, 3:]) == 0
    assert torch.count_nonzero(nope[1]) == 0
    assert torch.count_nonzero(rope[1]) == 0


@pytest.mark.parametrize("tuple_kv", [False, True])
def test_fp32_attention_preserves_kv_and_sink(tuple_kv):
    torch.manual_seed(1024)
    query = torch.randn(3, 2, 576, dtype=torch.bfloat16)
    kv = torch.randn(3, 4, 576, dtype=torch.float32)
    selected = torch.tensor([[-1, -1, -1, 0], [-1, 0, 1, -1], [-1, -1, -1, -1]])
    sink = torch.tensor([0.3, -0.5])
    scale = 0.04
    rows = (kv[..., :512, None].transpose(-1, -2), kv[..., 512:, None].transpose(-1, -2), None) if tuple_kv else kv
    result = decode_fusions.native_mla_context(
        query, selected, rows, sink, scale, length_provider=None, metadata=None,
    )
    logits = (query.float() @ kv.transpose(-1, -2) * scale).masked_fill(selected[:, None] < 0, -torch.inf)
    denominator = torch.cat((logits, sink[None, :, None].expand(3, -1, -1)), -1).softmax(-1)
    expected = (denominator[..., :-1] @ kv[..., :512]).to(query.dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.count_nonzero(result[2]) == 0
