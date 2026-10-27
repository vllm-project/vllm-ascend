# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.models.deepseek_v4.decode_row_shard import make_graph_decode_indexer_row_plan
from vllm_ascend.models.deepseek_v4.decode_row_shard_graph import (
    allocate_graph_decode_indexer_state,
    gather_graph_decode_indexer_topk,
    refresh_graph_decode_indexer_inputs,
    select_graph_decode_indexer_inputs,
)


def state_for(rank, padded_requests=8, rows_per_request=8):
    plan = make_graph_decode_indexer_row_plan(padded_requests * rows_per_request, rows_per_request, 4, rank, 8)
    assert plan is not None
    block_table = torch.tensor([[100 + i, 200 - i] for i in range(padded_requests)], dtype=torch.int32)
    state = allocate_graph_decode_indexer_state(
        plan, block_table, num_heads=2, head_dim=4, topk=3, max_model_len=65536, metadata_size=1024
    )
    return state, block_table


@pytest.mark.parametrize("padded_requests", [8, 9, 13])
def test_bucket_plan_is_independent_of_live_requests(padded_requests):
    state, table = state_for(3, padded_requests)
    pointers = {name: value.data_ptr() for name, value in vars(state).items() if isinstance(value, torch.Tensor)}
    for live in [padded_requests, 1, 0, padded_requests - 1, padded_requests]:
        seq_lens = torch.arange(101, 101 + padded_requests, dtype=torch.int32)
        refresh_graph_decode_indexer_inputs(state, seq_lens, table, live)
        expected = seq_lens[list(state.plan.request_ids)].clone()
        expected[torch.tensor(state.plan.request_ids) >= live] = 0
        assert torch.equal(state.seq_lens, expected)
        assert torch.equal(state.qli_seqused_k, expected // 4)
        assert torch.equal(state.qli_cmp_residual_k, expected % 4)
        assert state.invalid_rows.numel() == len(state.plan.query_rows)
        assert pointers == {
            name: value.data_ptr() for name, value in vars(state).items() if isinstance(value, torch.Tensor)
        }


def test_query_selection_updates_fixed_buffers():
    state, _ = state_for(0)
    rows = 64
    pointers = (state.query_buffer.data_ptr(), state.weights_buffer.data_ptr(), state.query_scale_buffer.data_ptr())
    for step in [1, 3, 8]:
        query = torch.full((rows, 2, 4), step, dtype=torch.int8)
        weights = torch.full((rows, 2), step + 1, dtype=torch.float16)
        scales = torch.full((rows, 2), step + 2, dtype=torch.float16)
        actual = select_graph_decode_indexer_inputs(state, query, weights, scales)
        assert tuple(x.data_ptr() for x in actual) == pointers
        assert torch.equal(actual[0], query[list(state.plan.query_rows)])
        assert torch.equal(actual[1], weights[list(state.plan.query_rows)])
        assert torch.equal(actual[2], scales[list(state.plan.query_rows)])


@pytest.mark.parametrize("padded_requests", [8, 9, 13])
@pytest.mark.parametrize("rows_per_request", [1, 8])
@pytest.mark.parametrize("live", [0, 1, 5, 8])
def test_every_rank_joins_fixed_collective_including_empty_local_work(live, padded_requests, rows_per_request):
    s = rows_per_request
    reference = torch.arange(padded_requests * s * 3, dtype=torch.int32).reshape(-1, 1, 3)
    reference[live * s :] = -1
    states = [state_for(rank, padded_requests, s)[0] for rank in range(4)]
    gathered = torch.full_like(states[0].gathered_topk, -1)
    for rank, state in enumerate(states):
        rows = list(state.plan.query_rows)
        gathered[rank * state.plan.max_local_rows : rank * state.plan.max_local_rows + len(rows)] = reference[rows]
    for rank, state in enumerate(states):
        table = torch.zeros((padded_requests, 2), dtype=torch.int32)
        refresh_graph_decode_indexer_inputs(state, torch.full((padded_requests,), 100, dtype=torch.int32), table, live)
        group = SimpleNamespace(world_size=4, rank_in_group=rank, device_group=object())
        local = torch.arange(padded_requests * s * 3, dtype=torch.int32).reshape(-1, 1, 3)[list(state.plan.query_rows)]

        def copy_gather(output, payload, *, group, state=state):
            assert payload.shape == state.padded_topk.shape
            assert torch.equal(payload[: len(state.plan.query_rows)], reference[list(state.plan.query_rows)])
            output.copy_(gathered)

        with patch(
            "vllm_ascend.models.deepseek_v4.decode_row_shard_graph.dist.all_gather_into_tensor", side_effect=copy_gather
        ) as collective:
            actual = gather_graph_decode_indexer_topk(state, local, group)
        collective.assert_called_once()
        assert actual is state.restored_topk
        assert torch.equal(actual, reference)


def make_graph_builder():
    builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
    builder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v4", index_n_heads=2, index_head_dim=4, index_topk=3),
        max_model_len=65536,
        enforce_eager=False,
    )
    builder.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
    )
    builder.speculative_config = SimpleNamespace(
        method="dspark", num_speculative_tokens=7, enable_adaptive_verification=False
    )
    builder.compressor_ratio = 4
    builder.num_prefills = 0
    builder.num_actual_tokens = 64
    builder._device_metadata_enabled = False
    builder.device = torch.device("cpu")
    builder.decode_row_shard_graph_states = {}
    builder.seq_lens = torch.full((8,), 101, dtype=torch.int32)
    builder.block_table = torch.arange(16, dtype=torch.int32).reshape(8, 2)
    common = SimpleNamespace(num_reqs=8, num_input_tokens=64, causal=True, query_start_loc_cpu=torch.arange(9) * 8)
    return builder, common


def test_builder_reuses_state_and_refreshes_metadata_when_live_count_changes():
    builder, common = make_graph_builder()
    with (
        patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size", return_value=4),
        patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_rank", return_value=3),
        patch("vllm_ascend.attention.dsa_v1.get_ascend_config", return_value=SimpleNamespace(enable_dsa_cp=False)),
        patch("vllm_ascend.attention.dsa_v1.DeviceOperator.get_dsa_indexer_quant_mode", return_value=2),
        patch.object(
            torch.ops._C_ascend,
            "npu_quant_lightning_indexer_v2_metadata",
            create=True,
            return_value=torch.ones(1024, dtype=torch.int32),
        ) as metadata_op,
    ):
        first = builder._prepare_fullgraph_decode_row_shard(common, 8)
        assert first is not None
        builder.num_actual_tokens = 8
        second = builder._prepare_fullgraph_decode_row_shard(common, 1)
        assert first is second
        assert second.plan.num_requests == 8
        assert torch.all(second.seq_lens == 0)
        assert torch.all(second.invalid_rows)
        assert metadata_op.call_count == 2
        assert metadata_op.call_args.kwargs["max_seqlen_k"] == 65536 // 4
        assert metadata_op.call_args.kwargs["batch_size"] == 2
        common.query_start_loc_cpu[2] += 1
        with pytest.raises(RuntimeError, match="layout changed"):
            builder._prepare_fullgraph_decode_row_shard(common, 1)


def test_small_graph_bucket_does_not_depend_on_live_threshold():
    assert make_graph_decode_indexer_row_plan(56, 8, 4, 0, 8) is None
    plan = make_graph_decode_indexer_row_plan(64, 8, 4, 0, 8)
    assert plan is not None and plan.num_requests == 8


@pytest.mark.parametrize("reason", ["mrv1", "fp8", "pcp", "dcp", "adaptive", "variable_spec", "mode", "noncausal"])
def test_incompatible_fullgraph_configs_use_original_indexer(reason):
    builder, common = make_graph_builder()
    quant_mode = 2
    if reason == "mrv1":
        builder.vllm_config.use_v2_model_runner = False
    elif reason == "fp8":
        quant_mode = 1
    elif reason == "pcp":
        builder.vllm_config.parallel_config.prefill_context_parallel_size = 2
    elif reason == "dcp":
        builder.vllm_config.parallel_config.decode_context_parallel_size = 2
    elif reason == "adaptive":
        builder.speculative_config.enable_adaptive_verification = True
    elif reason == "variable_spec":
        builder.speculative_config.num_speculative_tokens_per_batch_size = {8: 7}
    elif reason == "mode":
        builder.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    elif reason == "noncausal":
        common.causal = False
    with (
        patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size", return_value=4),
        patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm_ascend.attention.dsa_v1.get_ascend_config", return_value=SimpleNamespace(enable_dsa_cp=False)),
        patch("vllm_ascend.attention.dsa_v1.DeviceOperator.get_dsa_indexer_quant_mode", return_value=quant_mode),
    ):
        assert builder._prepare_fullgraph_decode_row_shard(common, 8) is None
    assert not builder.decode_row_shard_graph_states


@pytest.mark.parametrize("bad_input", ["rows", "heads", "dtype"])
def test_graph_input_abi_change_is_rejected(bad_input):
    state, _ = state_for(0)
    query = torch.zeros((64, 2, 4), dtype=torch.int8)
    weights = torch.zeros((64, 2), dtype=torch.float16)
    scales = torch.ones_like(weights)
    if bad_input == "rows":
        query = query[:-1]
    elif bad_input == "heads":
        weights = weights[:, :1]
    elif bad_input == "dtype":
        scales = scales.float()
    with pytest.raises(RuntimeError, match="shape/dtype changed"):
        select_graph_decode_indexer_inputs(state, query, weights, scales)


def test_graph_geometry_and_live_capacity_are_checked():
    state, table = state_for(0)
    lens = torch.full((8,), 100, dtype=torch.int32)
    with pytest.raises(ValueError, match="Live request count"):
        refresh_graph_decode_indexer_inputs(state, lens, table, 9)
    with pytest.raises(RuntimeError, match="geometry changed"):
        refresh_graph_decode_indexer_inputs(state, lens, table[:, :1], 8)


def test_graph_select_topk_adapts_inputs_and_restores_all_rows():
    from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerOps

    state, table = state_for(3)
    refresh_graph_decode_indexer_inputs(state, torch.full((8,), 100, dtype=torch.int32), table, 1)
    query = torch.ones((64, 2, 4), dtype=torch.int8)
    weights = torch.full((64, 2), 2, dtype=torch.bfloat16)
    scale = torch.full((64, 2), 3, dtype=torch.float16)
    key = torch.zeros((5, 8, 1, 4), dtype=torch.int8)
    key_scale = torch.ones((5, 8, 1, 1), dtype=torch.float16)
    local = torch.zeros_like(state.padded_topk[: len(state.plan.query_rows)])
    ops = AscendIndexerOps.__new__(AscendIndexerOps)
    ops.index_topk = 3
    ops.device_operator = SimpleNamespace(
        prepare_dsa_indexer_weights=lambda value: value.to(torch.float16),
        prepare_dsa_indexer_query_scale=lambda value: value,
        prepare_dsa_indexer_key_scale=lambda value: value,
        get_dsa_indexer_quant_mode=lambda: 2,
    )
    with (
        patch("vllm_ascend.models.deepseek_v4.indexer.wait_for_device_metadata") as wait,
        patch("vllm_ascend.models.deepseek_v4.indexer.get_tp_group", return_value=object()),
        patch.object(
            torch.ops._C_ascend, "npu_quant_lightning_indexer_v2", create=True, return_value=(local, None)
        ) as qli,
        patch(
            "vllm_ascend.models.deepseek_v4.indexer.gather_graph_decode_indexer_topk",
            return_value=state.restored_topk,
        ) as gather,
    ):
        result = ops.select_topk(query, weights, scale, key, key_scale, SimpleNamespace(decode_row_shard=state))
    assert result is state.restored_topk
    assert qli.call_args.kwargs["query"] is state.query_buffer
    assert qli.call_args.kwargs["key"] is key
    assert qli.call_args.kwargs["block_table"] is state.block_table
    assert qli.call_args.kwargs["metadata"] is state.qli_metadata
    assert torch.equal(state.weights_buffer, weights[list(state.plan.query_rows)].half())
    wait.assert_not_called()
    gather.assert_called_once()
