# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.worker.workspace import WorkspaceManager

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.ops.triton.turboquant_prefill import workspace_shapes


@pytest.mark.parametrize(
    "requests,length,queries,budget",
    [
        (0, 30000, 256, 64),
        (1, 0, 256, 64),
        (1, 30000, 1, 64),
        (1, 30000, 83, 64),
        (2, 30000, 256, 64),
        (1, 30000, 256, 1),
        (1, 4096, 256, 0),
    ],
)
def test_workspace_fallback_for_prefix_budget_or_empty(requests, length, queries, budget):
    assert workspace_shapes(requests, length, queries, budget * 1024**2) is None


def test_workspace_budget_includes_table_and_alignment():
    data_bytes = 128 * (512 + 64) * 2
    assert workspace_shapes(1, 128, 32, data_bytes) is None
    assert workspace_shapes(1, 128, 32, data_bytes + 256) is not None


def test_profile_reserves_one_shared_workspace_for_all_layers():
    manager = WorkspaceManager(torch.device("cpu"))
    layers = [object.__new__(AscendSFAImpl) for _ in range(2)]
    with patch("vllm_ascend.attention.sfa_v1.current_workspace_manager", return_value=manager):
        pointers = []
        for layer in layers:
            layer.enable_sparse_sfa_turboquant = True
            layer.turboquant = MagicMock()
            layer.tq_prefill_workspace_bytes = 2 * 1024**2
            layer.forward("layer", torch.empty(1), (), None, output=torch.empty(1))
            pointers.append(manager.get_simultaneous(((1,), torch.uint8))[0].data_ptr())
        assert pointers[0] == pointers[1]
        manager.lock()
        shapes = workspace_shapes(1, 1025, 32, 2 * 1024**2)
        views = manager.get_simultaneous(*shapes)
        assert views[0].data_ptr() == pointers[0]


@pytest.mark.parametrize("prefills,decodes,budget", [(0, 1, 64), (1, 1, 64), (1, 0, 0)])
def test_decode_mixed_and_disabled_do_not_allocate(prefills, decodes, budget):
    layer = object.__new__(AscendSFAImpl)
    layer.tq_prefill_workspace_bytes = budget * 1024**2
    metadata = SimpleNamespace(num_prefills=prefills, num_decodes=decodes)
    with patch("vllm_ascend.attention.sfa_v1.current_workspace_manager") as manager:
        result = layer._try_tq_prefill_workspace(None, None, None, None, metadata, None, None, None)
    assert result is None
    manager.assert_not_called()


@pytest.mark.parametrize("folded", [False, True])
def test_workspace_uses_existing_bf16_attention_dispatch(folded):
    layer = object.__new__(AscendSFAImpl)
    layer.tq_prefill_workspace_bytes = 1024**2
    layer._tq_rotations_folded = folded
    layer.scale = 576**-0.5
    layer.turboquant = MagicMock()
    layer.turboquant.forward.side_effect = lambda x: x + 1
    layer.turboquant.inverse.side_effect = lambda x: x + 2
    metadata = SimpleNamespace(num_prefills=1, num_decodes=0, max_seq_len=128)
    query = torch.zeros(32, 64, 512, dtype=torch.bfloat16)
    rope = torch.zeros(32, 64, 64, dtype=torch.bfloat16)
    cache = torch.empty(1, 128, 1, 386, dtype=torch.int8)
    indices = torch.zeros(32, 1, 2048, dtype=torch.int32)
    qlens, klens = torch.tensor([32]), torch.tensor([128])
    table = torch.zeros(1, 1, dtype=torch.int32)
    output = torch.ones_like(query)
    manager = WorkspaceManager(torch.device("cpu"))
    with (
        patch("vllm_ascend.attention.sfa_v1.current_workspace_manager", return_value=manager),
        patch("vllm_ascend.attention.sfa_v1.expand_turboquant_pages") as expand,
        patch("torch.ops._C_ascend.npu_sparse_flash_attention", create=True, return_value=(output,)) as attention,
    ):
        result = layer._try_tq_prefill_workspace(query, rope, cache, indices, metadata, qlens, klens, table)
    expand.assert_called_once()
    nope, expanded_rope, expanded_table = expand.call_args.args[-3:]
    attention.assert_called_once()
    args = attention.call_args.kwargs
    assert args["key"] is args["value"] is nope
    assert args["key_rope"] is expanded_rope
    assert args["block_table"] is expanded_table
    assert args["actual_seq_lengths_query"] is qlens
    assert args["actual_seq_lengths_kv"] is klens
    assert args["sparse_mode"] == 3
    assert args["return_softmax_lse"] is False
    torch.testing.assert_close(args["query"], query if folded else query + 1)
    torch.testing.assert_close(result, output if folded else output + 2)
