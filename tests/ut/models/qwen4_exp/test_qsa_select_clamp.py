# SPDX-License-Identifier: Apache-2.0
"""QSA selection supports cache capacities smaller than the top-k budget."""

import pytest
import torch

from vllm_ascend.models.qwen4_exp.ops import qsa_select_paged_tokens
from vllm_ascend.ops.triton.qwen4_exp import qsa as triton_qsa


@pytest.mark.parametrize("token_topk", [16, 64])
@pytest.mark.parametrize("sequence_length", [20, 21])
def test_select_short_request(token_topk: int, sequence_length: int) -> None:
    # Capacity is eight groups; only five complete groups are visible.
    # Strictly descending scores make the selected groups deterministic.
    cache = torch.arange(8, 0, -1, dtype=torch.bfloat16).view(2, 4, 1, 1).expand(2, 4, 1, 8)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    query = torch.ones(1, 2, 8, dtype=torch.bfloat16)
    token_to_req = torch.zeros(1, dtype=torch.int64)
    positions = torch.tensor([sequence_length - 1], dtype=torch.int64)
    seq_lens = torch.tensor([sequence_length], dtype=torch.int64)

    out = qsa_select_paged_tokens(query, cache, block_table, token_to_req, positions, seq_lens, token_topk, 4)

    expected = list(range(min(token_topk, 20))) + list(range(20, sequence_length))
    assert out.shape == (1, token_topk + 3)
    assert out[0, : len(expected)].tolist() == expected
    assert (out[0, len(expected) :] == -1).all()


@pytest.mark.parametrize("num_pages", [1, 3])
@pytest.mark.parametrize("use_e3", [False, True])
def test_triton_selector_preserves_expansion_buffer(monkeypatch, num_pages: int, use_e3: bool) -> None:
    rows = 129  # Exercise both a full chunk and the final partial chunk.
    capacity = num_pages * 192
    selected = min(512, capacity)
    expanded_rows = []

    def score(query, *args):
        logits = torch.arange(capacity, 0, -1, dtype=torch.float32).expand(query.shape[0], -1)
        return logits, None

    def expand(indices, positions, sequence_lengths, token_to_req, compress_ratio, token_topk, out):
        assert indices.shape == (positions.numel(), 512)
        assert (indices[:, :selected] == torch.arange(selected)).all()
        assert (indices[:, selected:] == -1).all()
        expanded_rows.append(indices.shape[0])
        out.fill_(-1)
        return out

    monkeypatch.setattr(triton_qsa, "qsa_mqa_paged", score)
    expander = "expand_qsa_block_indices_e3" if use_e3 else "expand_qsa_block_indices_npu"
    monkeypatch.setattr(triton_qsa, expander, expand)
    output = torch.empty(rows, 2051, dtype=torch.int32)

    result = triton_qsa.qsa_select_paged_tokens(
        torch.zeros(rows, 2, 128),
        torch.zeros(num_pages, 192, 1, 128),
        torch.arange(num_pages, dtype=torch.int32).unsqueeze(0),
        torch.zeros(rows, dtype=torch.int64),
        torch.arange(rows),
        torch.tensor([rows]),
        token_topk=2048,
        compress_ratio=4,
        out=output,
        use_e3=use_e3,
    )

    assert result is output
    assert expanded_rows == [128, 1]
    assert (output == -1).all()
