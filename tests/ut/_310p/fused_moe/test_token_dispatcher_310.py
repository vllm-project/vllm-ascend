# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu

from vllm_ascend._310p.fused_moe import token_dispatcher as token_dispatcher_module
from vllm_ascend._310p.fused_moe.token_dispatcher import TokenDispatcherWithAllGather310
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize(
    ("global_to_local", "topk_ids", "local_ids", "counts", "local_count"),
    [
        (None, [[0, 1], [1, 0]], [[0, 1], [1, 0]], [2, 4], None),
        ([-1, 1, -1, 0], [[0, 1], [3, 0]], [[2, 1], [0, 2]], [1, 2, 4], 2),
        ([-1, -1, 0, 1], [[0, 1], [1, 0]], [[2, 2], [2, 2]], [0, 0, 4], 0),
    ],
)
def test_route_and_combine_local_experts(global_to_local, topk_ids, local_ids, counts, local_count):
    dispatcher = TokenDispatcherWithAllGather310(top_k=2, num_experts=4, num_local_experts=2)
    expert_map = None if global_to_local is None else torch.tensor(global_to_local, dtype=torch.int32)
    activations = torch.zeros(4, 16)
    rows = torch.arange(4, dtype=torch.int32)
    dispatch_input = MoETokenDispatchInput(
        hidden_states=torch.zeros(2, 16),
        topk_weights=torch.full((2, 2), 0.5),
        topk_ids=torch.tensor(topk_ids, dtype=torch.int32),
        routing=SimpleNamespace(expert_map=expert_map, apply_router_weight_on_input=False),
        quant=SimpleNamespace(quant_type=QuantType.NONE),
    )
    routing_outputs = activations, rows, torch.tensor(counts), None
    with patch.object(torch_npu, "npu_moe_init_routing_v2", return_value=routing_outputs) as route:
        dispatch_output = dispatcher.token_dispatch(dispatch_input)

    torch.testing.assert_close(route.call_args.args[1], torch.tensor(local_ids, dtype=torch.int32))
    assert route.call_args.kwargs["active_expert_range"] == [0, len(counts)]
    assert route.call_args.kwargs["expert_num"] == len(counts)
    torch.testing.assert_close(dispatch_output.group_list, torch.tensor(counts[:2]))

    expert_output = torch.ones_like(activations)
    expected = expert_output.clone()
    if local_count is not None:
        expert_output[local_count:] = float("nan")
        expected[local_count:] = 0
    with patch.object(TokenDispatcherWithAllGather, "token_combine") as combine:
        dispatcher.token_combine(expert_output, dispatch_output.combine_metadata)
    torch.testing.assert_close(combine.call_args.args[0], expected)


@pytest.mark.parametrize("local_rows", [0, 2, 4])
def test_large_combine_redirects_only_remote_rows(monkeypatch, local_rows):
    monkeypatch.setattr(token_dispatcher_module, "_SINGLE_ZERO_ROW_MIN_ROUTES", 4)
    dispatcher = TokenDispatcherWithAllGather310(top_k=2, num_experts=4, num_local_experts=2)
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    output = torch.arange(4, dtype=torch.float32).view(-1, 1).expand(-1, 16).contiguous() + 1
    expected = output[permutation].masked_fill((permutation >= local_rows).unsqueeze(1), 0)
    output[local_rows:] = float("nan")
    metadata = token_dispatcher_module.MoEAllGatherCombineMetadata310(
        topk_weights=torch.ones(2, 2),
        expanded_row_idx=permutation,
        restore_shape=torch.Size((2, 16)),
        local_row_count=torch.tensor(local_rows),
    )
    with patch.object(TokenDispatcherWithAllGather, "token_combine") as combine:
        dispatcher.token_combine(output, metadata)
    rows, new_metadata, _ = combine.call_args.args
    torch.testing.assert_close(rows[new_metadata.expanded_row_idx], expected)
