# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.attention.context_parallel.sfa_split_kv import (
    merge_split_kv,
    run_qsfa_split_kv,
    split_sparse_indices,
)


@pytest.mark.parametrize("valid", [0, 1, 127, 128, 129, 255, 256, 257, 511, 512])
def test_split_preserves_compacted_candidate_ids(valid):
    selected = torch.arange(valid, dtype=torch.int32) * 3 + 17
    source = torch.full((1, 1, 512), -1, dtype=torch.int32)
    source[0, 0, :valid] = selected
    left, right = split_sparse_indices(source).unbind()
    assert left.shape == right.shape == (1, 256)
    assert torch.equal(torch.cat((left[left >= 0], right[right >= 0])), selected)
    assert int((left >= 0).sum()) == min(valid, ((valid + 255) // 256) * 128)


def test_merge_recovers_output_and_rank_stats_for_large_max_gap_and_empty_rows():
    logits = torch.tensor([[[19.0, 18.0, 0.0, -1.0]], [[4.0, -2.0, 0.0, -1.0]]], dtype=torch.float64)
    values = torch.tensor([0.0, 0.0, 64.0, 1.0], dtype=torch.float64)
    part_logits = logits.reshape(4, 1, 2)
    part_values = values.repeat(2).reshape(4, 1, 2)
    maxima = part_logits.amax(-1)
    exps = (part_logits - maxima.unsqueeze(-1)).exp()
    sums = exps.sum(-1)
    outputs = (exps * part_values).sum(-1) / sums
    # Include an all-empty row and a row where only the second part is valid.
    other_m = torch.tensor([float("-inf"), float("-inf"), float("-inf"), 9.0])[:, None]
    other_s = torch.tensor([0.0, 0.0, 0.0, 3.0])[:, None]
    other_o = torch.tensor([0.0, 0.0, 0.0, 42.0])[:, None]
    output, maximum, total = merge_split_kv(
        torch.cat((outputs, other_o)).float().unsqueeze(-1),
        torch.cat((maxima, other_m)).float().unsqueeze(0),
        torch.cat((sums, other_s)).float().unsqueeze(0),
    )
    expected = (logits.softmax(-1) * values).sum(-1)
    torch.testing.assert_close(output[:2, :, 0], expected.float(), rtol=2e-6, atol=0)
    expected_max = logits.amax(-1)
    expected_sum = (logits - expected_max.unsqueeze(-1)).exp().sum(-1)
    torch.testing.assert_close(maximum[:, :2], expected_max.float().unsqueeze(0))
    torch.testing.assert_close(total[:, :2], expected_sum.float().unsqueeze(0))
    torch.testing.assert_close(maximum[:, :2] + total[:, :2].log(), logits.logsumexp(-1).float().unsqueeze(0))
    assert output[2:].flatten().tolist() == [0.0, 42.0]
    assert total[:, 2:].flatten().tolist() == [0.0, 3.0]
    assert maximum[:, 2:].flatten().tolist() == [torch.finfo(torch.float32).min, 9.0]


def test_interleaved_rows_keep_request_boundaries_and_graph_capacity():
    query = torch.arange(4).view(4, 1, 1).float()
    indices = torch.full((4, 1, 256), -1, dtype=torch.int32)
    indices[0, 0] = torch.arange(256, dtype=torch.int32)
    cu = torch.tensor([1, 1, 3], dtype=torch.int32)  # Empty request and one padding row.
    seen = []

    def attend(q, idx, cu_q):
        seen.append(q.shape[0])
        assert q.flatten().tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
        assert cu_q.tolist() == [2, 2, 6]
        count = (idx >= 0).sum(-1)
        assert count.flatten().tolist() == [128, 128, 0, 0, 0, 0, 0, 0]
        return (q + 1) * (count > 0).unsqueeze(-1), torch.zeros_like(count).float()[None], count.float()[None]

    result, _, total = run_qsfa_split_kv(query, indices, cu, attend)
    assert seen == [8]
    assert result.flatten().tolist() == [1, 0, 0, 0]
    assert total.flatten().tolist() == [256, 0, 0, 0]


def test_unsupported_capacity_uses_original_call():
    query = torch.zeros(3, 1, 1)
    indices = torch.zeros(3, 1, 129, dtype=torch.int32)
    cu = torch.tensor([3], dtype=torch.int32)

    def attend(q, idx, cu_q):
        assert q is query and idx is indices and cu_q is cu
        return q, idx, cu_q

    out = run_qsfa_split_kv(query, indices, cu, attend)
    assert out[0] is query and out[1] is indices and out[2] is cu
