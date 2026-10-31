# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Real NPU dispatch/restore contracts, without an expert MLP or EP collective."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.fused_moe import token_dispatcher
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import build_token_dispatch_input
from vllm_ascend.ops.fused_moe.padding import mask_padding_routes
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV, TokenDispatcherWithAllGather
from vllm_ascend.quantization.quant_type import QuantType


def _inputs(dtype, rows, valid_rows):
    x = torch.arange(rows * 128, dtype=torch.float32, device="npu").reshape(rows, 128).to(dtype)
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(rows)], dtype=torch.int32, device="npu")
    weights = torch.full((rows, 2), 0.5, dtype=torch.float32, device="npu")
    valid = torch.arange(rows, device="npu") < valid_rows
    return x, weights, ids, valid


def _allgather_round_trip(dispatcher, x, weights, ids, valid):
    inputs = build_token_dispatch_input(
        fused_experts_input=build_fused_experts_input(
            hidden_states=x,
            topk_weights=weights,
            topk_ids=ids,
            mc2_mask=valid,
            quant_type=QuantType.NONE,
            dynamic_eplb=False,
        )
    )
    output = dispatcher.token_dispatch(inputs)
    restored = dispatcher.token_combine(output.hidden_states, output.combine_metadata)
    return restored, output.group_list


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,valid_rows", [(4, 4), (4, 3), (4, 0), (1, 0), (1, 1)])
def test_allgather_padding_excludes_expert_work(dtype, rows, valid_rows):
    x, weights, ids, valid = _inputs(dtype, rows, valid_rows)
    dispatcher = TokenDispatcherWithAllGather(top_k=2, num_experts=4, num_local_experts=4)
    restored, counts = _allgather_round_trip(dispatcher, x, weights, ids, valid)
    assert int(counts.sum().cpu()) == valid_rows * 2
    torch.testing.assert_close(restored, x * valid[:, None], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,valid_rows", [(4, 4), (4, 3), (4, 0), (1, 0), (1, 1)])
def test_alltoall_local_permutation_excludes_padding(dtype, rows, valid_rows, monkeypatch):
    x, weights, ids, valid = _inputs(dtype, rows, valid_rows)
    weights, ids = mask_padding_routes(weights, ids, valid, 5)
    dispatcher = TokenDispatcherWithAll2AllV.__new__(TokenDispatcherWithAll2AllV)
    dispatcher.lora_context = None
    dispatcher.num_experts = 4
    dispatcher.num_local_experts = 4
    dispatcher.local_expert_indices = list(range(4))
    dispatcher.expert_ids_per_ep_rank = torch.arange(4, dtype=torch.int32, device="npu")
    # One-rank identity collectives: exercise real NPU histc, split-size
    # calculation, permutation and restoration, not multi-rank communication.
    monkeypatch.setattr(token_dispatcher, "get_ep_group", lambda: SimpleNamespace(world_size=1, device_group=None))
    monkeypatch.setattr(token_dispatcher, "gather_from_sequence_parallel_region", lambda value, group: value)
    prepared = dispatcher._dispatch_preprocess(x, ids, skip_padding=True)
    assert prepared[0].shape[0] == valid_rows * 2
    restored = dispatcher._combine_postprocess(
        prepared[0],
        SimpleNamespace(
            reversed_local_input_permutation_mapping=prepared[1],
            topk_weights=weights,
            hidden_shape_before_permute=x.shape,
            hidden_shape=x.shape,
        ),
    )
    torch.testing.assert_close(restored, x * valid[:, None], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_empty_alltoall_int8_quantization(dtype):
    values, scales = token_dispatcher.DeviceOperator.npu_dynamic_quant(
        torch.empty((0, 128), dtype=dtype, device="npu"), act_quant_type=torch.int8, use_mxfp_quant=False
    )
    assert values.shape == (0, 128)
    assert scales.numel() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_allgather_graph_replay_reads_updated_padding(dtype):
    x, weights, ids, valid = _inputs(dtype, 4, 4)
    dispatcher = TokenDispatcherWithAllGather(top_k=2, num_experts=4, num_local_experts=4)
    for _ in range(3):
        _allgather_round_trip(dispatcher, x, weights, ids, valid)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        restored, counts = _allgather_round_trip(dispatcher, x, weights, ids, valid)
    for size in (3, 0, 4, 1):
        valid.copy_(torch.arange(4, device="npu") < size)
        graph.replay()
        torch.npu.synchronize()
        assert int(counts.sum().cpu()) == size * 2
        torch.testing.assert_close(restored, x * valid[:, None], atol=0, rtol=0)
