# SPDX-License-Identifier: Apache-2.0
"""DCP candidate collective boundaries and shared-index coordinate contracts."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.context_parallel import sharded_indexer as indexer


@pytest.mark.parametrize("rows,expected_tiles", [(0, []), (1, [1]), (256, [256]), (257, [256, 1])])
@pytest.mark.parametrize("return_local", [False, True])
def test_distributed_topk_collective_boundaries(monkeypatch, rows, expected_tiles, return_local):
    ops: Any = ModuleType("vllm_ascend.ops.triton.sharded_indexer")
    ops.MAX_FUSED_TOPK = 2048
    ops.pack_local_candidates = Mock(side_effect=lambda ids, scores, *args, **kwargs: ids)

    def merge(packed, count, world, topk, *, out, **kwargs):
        out.copy_(packed[:count])

    ops.merge_packed_candidates = Mock(side_effect=merge)
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setattr(indexer, "MAX_GATHERED_CANDIDATE_BYTES", 1)
    ids = torch.arange(rows * 7, dtype=torch.int32).reshape(rows, 1, 7)
    gather = Mock(side_effect=lambda packed, dim: torch.cat((packed, packed), dim))
    group = SimpleNamespace(world_size=2, rank_in_group=1, all_gather=gather)
    actual = indexer.distributed_topk(ids, ids.float(), group, 7, 128, return_local=return_local)
    assert torch.equal(actual, ids)
    assert [call.args[0].shape[0] for call in gather.call_args_list] == expected_tiles
    assert ops.merge_packed_candidates.call_count == len(expected_tiles)
    for call in ops.merge_packed_candidates.call_args_list:
        assert call.kwargs["local_rank"] == (1 if return_local else None)


@pytest.mark.parametrize("world,expected", [(8, 2048), (16, 1024)])
def test_candidate_memory_budget(world, expected):
    assert indexer.gathered_candidate_row_tile(4096, world, 2048) == expected


@pytest.mark.parametrize(
    "sharded,pcp,dsa_cp,turboquant,expected",
    [
        (True, 1, False, False, True),
        (False, 1, False, False, False),
        (True, 2, False, False, False),
        (True, 1, True, False, False),
        (True, 1, False, True, False),
    ],
)
def test_shared_index_coordinate_contract(sharded, pcp, dsa_cp, turboquant, expected):
    assert indexer.use_local_topk_output(sharded, pcp, dsa_cp, turboquant=turboquant) == expected
