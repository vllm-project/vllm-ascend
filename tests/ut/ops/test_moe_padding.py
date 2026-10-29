# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.models.common.ops import sequence_parallel
from vllm_ascend.ops.fused_moe import moe_comm_method, padding, prepare_finalize, token_dispatcher
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import build_token_dispatch_input
from vllm_ascend.quantization.quant_type import QuantType


@pytest.fixture
def context(monkeypatch):
    ctx = SimpleNamespace(
        in_profile_run=False,
        mc2_mask=torch.tensor([True, True, True, True]),
        padded_num_tokens=4,
        max_tokens_across_dp=4,
        max_tokens_across_pcp=0,
        is_padding=torch.tensor([False, False, True, True]),
    )
    monkeypatch.setattr(padding, "_EXTRA_CTX", ctx)
    monkeypatch.setattr(padding, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(prepare_finalize, "_EXTRA_CTX", ctx)
    monkeypatch.setattr(prepare_finalize, "get_dynamic_mx_quant_scale_alg", lambda: 0)
    return ctx


@pytest.mark.parametrize("rank,expected", [(0, [True, True]), (1, [False, False])])
def test_sp_mask_slices_a_global_v1_table(context, monkeypatch, rank, expected):
    # V1 keeps one reserved full-sequence mask, so each rank takes its segment.
    context.is_padding = None
    context.mc2_mask = torch.tensor([True, True, False, False])
    monkeypatch.setattr(padding, "get_tensor_model_parallel_rank", lambda: rank)
    result = padding.get_valid_token_mask(torch.zeros(2, 8), True)
    assert result.tolist() == expected


@pytest.mark.parametrize("rank,expected", [(0, [True, True]), (1, [True, False])])
def test_sp_v2_mask_arrives_already_sharded(context, monkeypatch, rank, expected):
    # Chain the model entry's shard into the MoE read: 3 real tokens padded to
    # 4 rows over TP 2, so rank 1 holds one real row and one padded row. A
    # second rank slice here would drop that real row's expert compute.
    monkeypatch.setattr(sequence_parallel, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(sequence_parallel, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(padding, "get_tensor_model_parallel_rank", lambda: rank)
    full_table = torch.tensor([False, False, False, True])
    context.is_padding = sequence_parallel._ascend_sp_padding_mask_impl(full_table)
    assert context.is_padding.shape[0] == 2
    assert padding.get_valid_token_mask(torch.zeros(2, 8), True).tolist() == expected


def test_v2_mask_keeps_holes_not_just_a_valid_prefix(context):
    context.is_padding = torch.tensor([False, True, False, True])
    assert padding.get_valid_token_mask(torch.zeros(4, 8), False).tolist() == [True, False, True, False]


def test_v1_mask_fallback_and_missing_mask(context):
    context.is_padding = None
    context.mc2_mask = torch.tensor([True, False])
    assert padding.get_valid_token_mask(torch.zeros(4, 8), False).tolist() == [True, False, False, False]
    context.mc2_mask = None
    assert padding.get_valid_token_mask(torch.zeros(4, 8), False) is None


def test_profile_dummy_rows_still_execute_experts(context):
    context.in_profile_run = True
    context.is_padding.fill_(True)
    assert padding.get_valid_token_mask(torch.zeros(4, 8), False).all()
    assert padding.get_valid_token_mask(torch.zeros(0, 8), False).shape == (0,)


def test_route_mask_does_not_mutate_inputs_or_real_routes():
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
    ids = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int32)
    valid = torch.tensor([True, False, True])
    new_weights, new_ids = padding.mask_padding_routes(weights, ids, valid, 3)
    torch.testing.assert_close(new_weights[valid], weights[valid])
    torch.testing.assert_close(new_ids[valid], ids[valid])
    assert new_weights[1].tolist() == [0, 0]
    assert new_ids[1].tolist() == [3, 3]
    assert ids[1].tolist() == [1, 2]
    assert weights[1].tolist() != [0, 0]
    unchanged = padding.mask_padding_routes(weights, ids, None, 3)
    assert unchanged[0] is weights and unchanged[1] is ids


@pytest.mark.parametrize("all_padding", [False, True])
def test_allgather_sp_mask_follows_the_same_collective(context, monkeypatch, all_padding):
    context.is_padding = torch.tensor([False, True, True, True])
    if all_padding:
        context.is_padding.fill_(True)
    monkeypatch.setattr(padding, "get_tensor_model_parallel_rank", lambda: 0)
    # Model EP gather with a second shard that contains no valid rows.
    gather = MagicMock(side_effect=lambda value: torch.cat((value, torch.zeros_like(value))))
    monkeypatch.setattr(torch.ops.vllm, "maybe_all_gather_and_maybe_unpad", gather)
    config = SimpleNamespace(is_sequence_parallel=True)
    layer = prepare_finalize.PrepareAndFinalizeWithAllGather(config)
    result = layer.prepare(torch.ones(2, 8), torch.ones(2, 3), skip_padding=True)
    assert result.mc2_mask.tolist() == ([False] * 4 if all_padding else [True, False, False, False])
    assert result.hidden_states.shape[0] == result.mc2_mask.shape[0]
    assert gather.call_count == 3  # activations, logits, validity


def test_dp_and_pcp_gather_preserve_mask_order(context, monkeypatch):
    context.max_tokens_across_dp = 3
    context.max_tokens_across_pcp = 7
    gather = lambda value, *args, **kwargs: torch.cat((value, torch.zeros_like(value)))
    group = SimpleNamespace(all_gather=gather)
    monkeypatch.setattr(prepare_finalize, "get_pcp_group", lambda: group)
    config = SimpleNamespace(is_sequence_parallel=False, dp_size=2, pcp_size=2, dp_group=group)
    layer = prepare_finalize.PrepareAndFinalizeWithAllGather(config)
    result = layer.prepare(torch.ones(2, 8), torch.ones(2, 3), skip_padding=True)
    assert result.mc2_mask.tolist() == [True, True] + [False] * 12
    assert result.hidden_states.shape[0] == result.mc2_mask.shape[0] == 14


@pytest.mark.parametrize(
    "cls", [prepare_finalize.PrepareAndFinalizeWithAll2All, prepare_finalize.PrepareAndFinalizeWithMC2]
)
def test_split_mask_is_aligned_with_hidden_states(context, monkeypatch, cls):
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(padding, "get_tensor_model_parallel_rank", lambda: 1)
    layer = cls(SimpleNamespace())
    result = layer.prepare(torch.ones(4, 8), torch.ones(4, 3), skip_padding=True)
    assert result.mc2_mask.tolist() == [False, False]
    assert result.hidden_states.shape[0] == result.mc2_mask.shape[0]


def test_alltoall_odd_batch_pads_every_tp_shard_equally(context, monkeypatch):
    context.is_padding = torch.tensor([False] * 5)
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        prepare_finalize,
        "_pad_tokens_with_cat",
        lambda x, size: torch.nn.functional.pad(x, (0, 0, 0, size - x.shape[0])),
    )
    layer = prepare_finalize.PrepareAndFinalizeWithAll2All(SimpleNamespace())
    result = layer.prepare(torch.ones(5, 8), torch.ones(5, 3), skip_padding=True)
    assert result.hidden_states.shape == (3, 8)
    assert result.mc2_mask.tolist() == [True, True, False]
    assert result.padded_hidden_states_shape == (6, 8)
    assert layer.pad_and_split_input_ids(torch.arange(5)).tolist() == [3, 4, 0]


def test_allgather_dispatch_filters_after_expert_map_indexing(monkeypatch):
    dispatcher = token_dispatcher.TokenDispatcherWithAllGather(top_k=2, num_experts=3, num_local_experts=3)
    hidden = torch.ones(3, 8)
    weights = torch.ones(3, 2) / 2
    ids = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int32)
    init_routing = MagicMock(return_value=(torch.empty(6, 8), torch.empty(6), torch.tensor([1, 1, 0]), None))
    monkeypatch.setattr(token_dispatcher.DeviceOperator, "npu_moe_init_routing", init_routing)
    monkeypatch.setattr(token_dispatcher, "get_ep_group", lambda: SimpleNamespace(rank_in_group=0))
    inputs = build_token_dispatch_input(
        fused_experts_input=build_fused_experts_input(
            hidden_states=hidden,
            topk_weights=weights,
            topk_ids=ids,
            expert_map=torch.arange(3),
            mc2_mask=torch.tensor([True, False, False]),
            quant_type=QuantType.NONE,
            dynamic_eplb=False,
        )
    )
    output = dispatcher.token_dispatch(inputs)
    assert init_routing.call_args.args[1].tolist() == [[0, 1], [3, 3], [3, 3]]
    assert output.combine_metadata.topk_weights.tolist() == [[0.5, 0.5], [0, 0], [0, 0]]
    assert ids.tolist() == [[0, 1], [1, 2], [0, 2]]


@pytest.mark.parametrize("ids,expected", [([[0, 1], [4, 4]], 2), ([[4, 4], [4, 4]], 0)])
def test_alltoall_counts_and_permute_size_exclude_sentinel(monkeypatch, ids, expected):
    dispatcher = token_dispatcher.TokenDispatcherWithAll2AllV.__new__(token_dispatcher.TokenDispatcherWithAll2AllV)
    dispatcher.num_experts = 3
    dispatcher.num_local_experts = 3
    dispatcher.local_expert_indices = [0, 1, 2]
    dispatcher.expert_ids_per_ep_rank = torch.arange(3)
    monkeypatch.setattr(type(dispatcher), "ep_size", property(lambda _: 1))
    monkeypatch.setattr(type(dispatcher), "ep_group", property(lambda _: None))
    monkeypatch.setattr(token_dispatcher, "gather_from_sequence_parallel_region", lambda value, group: value)
    # NPU supports integer histc; CPU only supports floating input. Preserve
    # the real histogram calculation while adapting this device boundary.
    cpu_histc = torch.histc
    monkeypatch.setattr(torch, "histc", lambda value, **kwargs: cpu_histc(value.float(), **kwargs).to(torch.int64))
    result = dispatcher._preprocess(torch.tensor(ids, dtype=torch.int32), skip_padding=True)
    assert result[-1] == expected
    assert int(result[0].sum()) == expected
    assert int(result[1].sum()) == expected


def test_alltoall_zero_rows_do_not_call_permute_or_unpermute(monkeypatch):
    dispatcher = token_dispatcher.TokenDispatcherWithAll2AllV.__new__(token_dispatcher.TokenDispatcherWithAll2AllV)
    dispatcher.lora_context = None
    dispatcher._preprocess = lambda *args, **kwargs: (torch.zeros(3), [0], [0], None, 0)
    permute = MagicMock()
    unpermute = MagicMock()
    monkeypatch.setattr(token_dispatcher.torch_npu, "npu_moe_token_permute", permute, raising=False)
    monkeypatch.setattr(token_dispatcher.torch_npu, "npu_moe_token_unpermute", unpermute, raising=False)
    hidden_states = torch.ones(4, 8)
    prepared = dispatcher._dispatch_preprocess(hidden_states, torch.full((4, 2), 5), skip_padding=True)
    assert prepared[0].shape == (0, 8)
    metadata = SimpleNamespace(hidden_shape=torch.Size([4, 8]))
    restored = dispatcher._combine_postprocess(prepared[0], metadata)
    torch.testing.assert_close(restored, torch.zeros(4, 8))
    permute.assert_not_called()
    unpermute.assert_not_called()
    # token_dispatch releases the permutation storage after the collective.
    # Even an empty buffer must not alias the original activations.
    expected_nbytes = hidden_states.untyped_storage().nbytes()
    prepared[0].untyped_storage().resize_(0)
    assert hidden_states.untyped_storage().nbytes() == expected_nbytes
    torch.testing.assert_close(hidden_states, torch.ones(4, 8))


@pytest.mark.parametrize(
    "global_bs,algorithm,a5,expected",
    [
        (0, "fullmesh", False, True),
        (128, "fullmesh", False, False),
        (0, "hierarchy", False, False),
        (0, "fullmesh", True, False),
    ],
)
def test_mc2_padding_capability_is_explicit(monkeypatch, global_bs, algorithm, a5, expected):
    comm = moe_comm_method.MC2CommImpl.__new__(moe_comm_method.MC2CommImpl)
    comm.token_dispatcher = SimpleNamespace(global_bs=global_bs, mc2_comm_alg=algorithm)
    monkeypatch.setattr(moe_comm_method, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: a5))
    assert comm.supports_skip_padding is expected
    fused = moe_comm_method.FusedMC2CommImpl.__new__(moe_comm_method.FusedMC2CommImpl)
    assert fused.supports_skip_padding is False


@pytest.mark.parametrize("rows", [0, 1, 4])
def test_empty_and_zero_valued_real_rows_keep_their_routes(context, rows):
    context.is_padding = torch.zeros(rows, dtype=torch.bool)
    valid = padding.get_valid_token_mask(torch.zeros(rows, 8), False)
    weights = torch.full((rows, 2), 0.5)
    ids = torch.zeros(rows, 2, dtype=torch.int32)
    new_weights, new_ids = padding.mask_padding_routes(weights, ids, valid, 3)
    torch.testing.assert_close(new_weights, weights)
    torch.testing.assert_close(new_ids, ids)


def test_alltoall_opt_out_keeps_the_original_dispatch_contract(context, monkeypatch):
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_rank", lambda: 1)
    layer = prepare_finalize.PrepareAndFinalizeWithAll2All(SimpleNamespace())
    result = layer.prepare(torch.ones(4, 8), torch.ones(4, 3))
    assert result.mc2_mask is None
    assert result.hidden_states.shape == (2, 8)


def test_mc2_sp_padding_preserves_non_prefix_validity(context, monkeypatch):
    context.is_padding = torch.tensor([True, True, False, True])
    context.padded_num_tokens = 8
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(padding, "get_tensor_model_parallel_rank", lambda: 1)
    layer = prepare_finalize.PrepareAndFinalizeWithMC2(SimpleNamespace())
    result = layer.prepare(torch.ones(2, 8), torch.ones(2, 3), replace_allreduce=True, skip_padding=True)
    assert result.hidden_states.shape == (4, 8)
    assert result.mc2_mask.tolist() == [True, False, False, False]
