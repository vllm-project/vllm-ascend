# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops import cann_keypool as adapters


def test_empty_work_guard_uses_fused_inputs_and_hides_dummy(monkeypatch):
    query = torch.randn(5, 2, 8)
    weights = torch.randn(5, 2)
    cache = torch.randn(16, 32, 1, 8)
    table = torch.arange(8, dtype=torch.int32).view(2, 4)
    ends = torch.tensor([0, 3], dtype=torch.int64)
    lengths = torch.zeros(2, dtype=torch.int64)
    tails = torch.tensor([0, 3], dtype=torch.int64)
    prepared = (
        torch.empty(6, 2, 8),
        torch.empty(6, 2),
        torch.empty(3, dtype=torch.int64),
        torch.empty(3, dtype=torch.int64),
        torch.empty(3, dtype=torch.int64),
        torch.empty(3, 4, dtype=torch.int32),
    )
    prepare = MagicMock(return_value=prepared)
    monkeypatch.setattr(adapters, "prepare_nonempty_keypool_inputs", prepare)
    indices = torch.arange(6, dtype=torch.int32)[:, None].expand(-1, 7).clone()
    indices[-1] = 99999
    values = torch.empty(0)
    op = MagicMock(return_value=(indices, values))
    actual, actual_values = adapters.pool_key_indexer_nonempty(
        op,
        query,
        cache,
        weights,
        tails,
        actual_seq_q=ends,
        actual_seq_k=lengths,
        block_table=table,
        topk=4,
        pool_size=4,
    )
    prepare.assert_called_once_with(query, weights, ends, lengths, tails, table)
    op.assert_called_once_with(
        prepared[0],
        cache,
        prepared[1],
        prepared[4],
        actual_seq_q=prepared[2],
        actual_seq_k=prepared[3],
        block_table=prepared[5],
        layout_q="TND",
        layout_k="PA_BBND",
        topk=4,
        pool_size=4,
        mask_mode=3,
    )
    torch.testing.assert_close(actual, indices[:-1])
    assert actual_values is values


@pytest.mark.parametrize("unsupported", [{"layout_q": "BSND"}, {"layout_k": "BSND"}, {"mask_mode": 0}])
def test_empty_work_guard_rejects_unsupported_layouts(unsupported):
    op = MagicMock()
    with pytest.raises(ValueError, match="causal TND/PA_BBND"):
        adapters.pool_key_indexer_nonempty(
            op,
            None,
            None,
            None,
            None,
            actual_seq_q=None,
            actual_seq_k=None,
            block_table=None,
            topk=4,
            pool_size=4,
            **unsupported,
        )
    op.assert_not_called()
