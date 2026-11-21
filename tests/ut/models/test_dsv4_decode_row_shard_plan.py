# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest

from vllm_ascend.models.deepseek_v4.decode_row_shard import make_decode_indexer_row_plan


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("extra", [0, 1, 5, 17])
@pytest.mark.parametrize("rows_per_request", [1, 2, 6, 16])
def test_round_robin_scores_each_request_once_and_restores_every_row(tp_size, extra, rows_per_request):
    num_requests = tp_size + extra
    plans = [make_decode_indexer_row_plan(num_requests, rows_per_request, tp_size, rank) for rank in range(tp_size)]
    scored_rows = [row for plan in plans for row in plan.query_rows]
    expected = list(range(num_requests * rows_per_request))
    assert sorted(scored_rows) == expected
    gathered = []
    for plan in plans:
        assert plan.request_ids == tuple(range(plan.tp_rank, num_requests, tp_size))
        for i, req in enumerate(plan.request_ids):
            assert plan.query_rows[i * rows_per_request : (i + 1) * rows_per_request] == tuple(
                range(req * rows_per_request, (req + 1) * rows_per_request)
            )
        gathered.extend([*plan.query_rows, *([-1] * (plan.max_local_rows - len(plan.query_rows)))])
    for plan in plans:
        assert [gathered[row] for row in plan.restore_rows] == expected


@pytest.mark.parametrize("args", [(1, 1, 2, 0), (8, 0, 8, 0), (8, 1, 8, 8), (8, 1, 1, 0)])
def test_invalid_decode_layout_is_rejected(args):
    with pytest.raises(ValueError):
        make_decode_indexer_row_plan(*args)
