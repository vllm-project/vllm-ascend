from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEFusedExpertsInput, MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
    MoEAllToAllCombineMetadata,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
)
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AlltoAllCommImpl,
    MoECommMethod,
    _split_fused_experts_input,
)
from vllm_ascend.ops.fused_moe.shared_experts import FusedMoEEvents
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV


def _make_fused_experts_input(
    num_tokens: int,
    *,
    dynamic_eplb: bool = False,
    lora_context=None,
) -> MoEFusedExpertsInput:
    return MoEFusedExpertsInput(
        hidden_states=torch.arange(num_tokens * 4, dtype=torch.float32).view(num_tokens, 4),
        topk_weights=torch.arange(num_tokens * 2, dtype=torch.float32).view(num_tokens, 2) / 10,
        topk_ids=torch.arange(num_tokens * 2, dtype=torch.int64).view(num_tokens, 2) % 4,
        weights=MoEWeights(w1=torch.ones(1), w2=torch.ones(1)),
        routing=MoeRouterInput(
            expert_map=None,
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
            pertoken_scale=torch.arange(num_tokens, dtype=torch.float32),
        ),
        quant=MoEQuantParams(),
        dynamic_eplb=dynamic_eplb,
        lora_context=lora_context,
    )


def _make_token_dispatch_input(num_tokens: int) -> MoETokenDispatchInput:
    fused_input = _make_fused_experts_input(num_tokens)
    return MoETokenDispatchInput(
        hidden_states=fused_input.hidden_states,
        topk_weights=fused_input.topk_weights,
        topk_ids=fused_input.topk_ids,
        routing=fused_input.routing,
        quant=fused_input.quant,
    )


def _make_combine_metadata(num_tokens: int) -> MoEAllToAllCombineMetadata:
    return MoEAllToAllCombineMetadata(
        input_splits=np.array([num_tokens, 0]),
        output_splits=np.array([num_tokens, 0]),
        topk_weights=torch.ones(num_tokens, 1),
        reversed_local_input_permutation_mapping=torch.arange(num_tokens),
        reversed_global_input_permutation_mapping=None,
        hidden_shape=torch.Size([num_tokens, 4]),
        hidden_shape_before_permute=torch.Size([num_tokens, 4]),
    )


@pytest.mark.parametrize("num_tokens,expected_sizes", [(4, (2, 2)), (5, (3, 2))])
def test_split_fused_experts_input_keeps_token_metadata_aligned(num_tokens, expected_sizes):
    fused_input = _make_fused_experts_input(num_tokens)

    micro_batch0, micro_batch1 = _split_fused_experts_input(fused_input)

    assert (micro_batch0.hidden_states.shape[0], micro_batch1.hidden_states.shape[0]) == expected_sizes
    assert torch.equal(
        torch.cat((micro_batch0.hidden_states, micro_batch1.hidden_states)),
        fused_input.hidden_states,
    )
    assert torch.equal(torch.cat((micro_batch0.topk_ids, micro_batch1.topk_ids)), fused_input.topk_ids)
    assert torch.equal(
        torch.cat((micro_batch0.topk_weights, micro_batch1.topk_weights)),
        fused_input.topk_weights,
    )
    assert torch.equal(
        torch.cat((micro_batch0.routing.pertoken_scale, micro_batch1.routing.pertoken_scale)),
        fused_input.routing.pertoken_scale,
    )


def test_dispatch_start_defers_wait_and_sync_wrapper_finishes():
    dispatcher = object.__new__(TokenDispatcherWithAll2AllV)
    dispatcher.lora_context = None
    dispatcher._comm_stream = MagicMock()
    dispatcher._dispatch_preprocess = MagicMock(
        return_value=(
            torch.ones(4, 4),
            torch.arange(4),
            torch.tensor([2, 2]),
            np.array([2, 2]),
            np.array([2, 2]),
            None,
            torch.Size([2, 4]),
            torch.Size([2, 4]),
        )
    )
    dispatcher._dispatch_postprocess = MagicMock(side_effect=lambda hidden, scale, *_args: (hidden, scale, None))
    work = MagicMock()
    collective_output = torch.full((4, 4), 7.0)
    token_input = _make_token_dispatch_input(2)

    with (
        patch.object(TokenDispatcherWithAll2AllV, "ep_group", new_callable=PropertyMock, return_value=MagicMock()),
        patch("torch.npu.current_stream") as current_stream,
        patch(
            "vllm_ascend.ops.fused_moe.token_dispatcher.async_all_to_all",
            return_value=(torch.ones(4, 4), collective_output, work),
        ),
    ):
        current_stream.return_value.record_event.return_value = MagicMock()
        handle = dispatcher.dispatch_start(token_input)
        work.wait.assert_not_called()
        output = dispatcher.dispatch_finish(handle)
        work.wait.assert_called_once_with()
        assert output.hidden_states is collective_output

        start_result = MagicMock()
        finish_result = MagicMock()
        with (
            patch.object(dispatcher, "dispatch_start", return_value=start_result) as start,
            patch.object(dispatcher, "dispatch_finish", return_value=finish_result) as finish,
        ):
            assert dispatcher.token_dispatch(token_input) is finish_result
            start.assert_called_once_with(token_input)
            finish.assert_called_once_with(start_result)


def test_combine_start_defers_wait_and_sync_wrapper_finishes():
    dispatcher = object.__new__(TokenDispatcherWithAll2AllV)
    dispatcher._comm_stream = MagicMock()
    dispatcher._combine_preprocess = MagicMock(side_effect=lambda hidden, _metadata: hidden)
    dispatcher._combine_postprocess = MagicMock(side_effect=lambda hidden, _metadata: hidden)
    metadata = _make_combine_metadata(2)
    work = MagicMock()
    collective_output = torch.full((2, 4), 9.0)

    with (
        patch.object(TokenDispatcherWithAll2AllV, "ep_group", new_callable=PropertyMock, return_value=MagicMock()),
        patch("torch.npu.current_stream") as current_stream,
        patch(
            "vllm_ascend.ops.fused_moe.token_dispatcher.async_all_to_all",
            return_value=(torch.ones(2, 4), collective_output, work),
        ),
    ):
        current_stream.return_value.record_event.return_value = MagicMock()
        handle = dispatcher.combine_start(torch.ones(2, 4), metadata)
        work.wait.assert_not_called()
        output = dispatcher.combine_finish(handle)
        work.wait.assert_called_once_with()
        assert output is collective_output

        start_result = MagicMock()
        finish_result = MagicMock()
        with (
            patch.object(dispatcher, "combine_start", return_value=start_result) as start,
            patch.object(dispatcher, "combine_finish", return_value=finish_result) as finish,
        ):
            hidden_states = torch.ones(2, 4)
            assert dispatcher.token_combine(hidden_states, metadata) is finish_result
            start.assert_called_once_with(hidden_states, metadata, None)
            finish.assert_called_once_with(start_result)


@pytest.mark.parametrize("defer_final_combine", [False, True])
def test_moe_dbo_pipeline_has_fixed_collective_order_and_restores_tokens(defer_final_combine):
    fused_input = _make_fused_experts_input(5)
    dispatcher = object.__new__(TokenDispatcherWithAll2AllV)
    order = []
    dispatch_count = 0
    finish_count = 0
    combine_start_count = 0
    combine_finish_count = 0

    def dispatch_start(token_input):
        nonlocal dispatch_count
        index = dispatch_count
        dispatch_count += 1
        order.append(f"Dstart{index}")
        return token_input

    def dispatch_finish(token_input):
        nonlocal finish_count
        index = finish_count
        finish_count += 1
        order.append(f"Dfinish{index}")
        num_tokens = token_input.hidden_states.shape[0]
        return MoETokenDispatchOutput(
            hidden_states=token_input.hidden_states,
            group_list=torch.tensor([num_tokens, 0]),
            group_list_type=1,
            combine_metadata=_make_combine_metadata(num_tokens),
        )

    def combine_start(hidden_states, _metadata):
        nonlocal combine_start_count
        index = combine_start_count
        combine_start_count += 1
        order.append(f"Cstart{index}")
        return hidden_states

    def combine_finish(hidden_states):
        nonlocal combine_finish_count
        index = combine_finish_count
        combine_finish_count += 1
        order.append(f"Cfinish{index}")
        return hidden_states

    dispatcher.dispatch_start = MagicMock(side_effect=dispatch_start)
    dispatcher.dispatch_finish = MagicMock(side_effect=dispatch_finish)
    dispatcher.combine_start = MagicMock(side_effect=combine_start)
    dispatcher.combine_finish = MagicMock(side_effect=combine_finish)

    comm_impl = object.__new__(AlltoAllCommImpl)
    comm_impl.token_dispatcher = dispatcher
    comm_impl.moe_config = SimpleNamespace()
    comm_impl._apply_mlp = MagicMock(
        side_effect=lambda mlp_input: (
            order.append(f"MLP{sum(item.startswith('MLP') for item in order)}") or mlp_input.hidden_states,
            MagicMock(),
        )
    )

    ascend_config = SimpleNamespace(enable_dsa_cp_moe_dbo_shared_expert_overlap=defer_final_combine)
    with (
        patch("torch.npu.current_stream") as current_stream,
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config", return_value=ascend_config),
    ):
        current_stream.return_value.record_event.return_value = MagicMock()
        result = comm_impl._fused_experts_dsa_cp_moe_dbo(fused_input)

    expected_order = [
        "Dstart0",
        "Dstart1",
        "Dfinish0",
        "MLP0",
        "Dfinish1",
        "Cstart0",
        "MLP1",
        "Cstart1",
        "Cfinish0",
    ]
    if not defer_final_combine:
        expected_order.append("Cfinish1")
    assert order == expected_order

    if defer_final_combine:
        assert result.routed_out is None
        assert result.finish_routed_out is not None
        order.append("SHARED_FULL_BATCH")
        routed_out = result.finish_routed_out()
        assert result.finish_routed_out() is routed_out
        assert order[-2:] == ["SHARED_FULL_BATCH", "Cfinish1"]
    else:
        assert result.finish_routed_out is None
        routed_out = result.routed_out
    assert torch.equal(routed_out, fused_input.hidden_states)
    assert torch.equal(result.expert_tokens, torch.tensor([5, 0]))


def test_shared_expert_can_cover_deferred_c1_finish():
    order = []
    routed_out = torch.full((2, 4), 3.0)

    def finish_routed_out():
        order.append("C1_FINISH")
        return routed_out

    events = FusedMoEEvents(
        before_routed_experts=MagicMock(),
        finish_routed_out=finish_routed_out,
    )
    order.append("SHARED_FULL_BATCH")
    actual = AscendMoERunner._finish_deferred_routed_out(None, events)

    assert order == ["SHARED_FULL_BATCH", "C1_FINISH"]
    assert actual is routed_out


def test_eligibility_is_rank_coordinated_and_cached_once_per_forward():
    fused_input = _make_fused_experts_input(5)
    comm_impl = object.__new__(AlltoAllCommImpl)
    comm_impl.token_dispatcher = MagicMock()
    forward_context = SimpleNamespace()
    ascend_config = SimpleNamespace(enable_dsa_cp_moe_dbo=True)

    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config", return_value=ascend_config),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_forward_context", return_value=forward_context),
        patch.object(comm_impl, "_local_dsa_cp_moe_dbo_candidate", return_value=(True, "eligible")),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.dist.all_reduce") as all_reduce,
    ):
        assert comm_impl._dsa_cp_moe_dbo_eligible(fused_input)
        assert comm_impl._dsa_cp_moe_dbo_eligible(fused_input)
        all_reduce.assert_called_once()


def test_switch_off_falls_back_to_synchronous_path():
    fused_input = _make_fused_experts_input(5)
    comm_impl = object.__new__(AlltoAllCommImpl)
    sentinel = MagicMock()

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config",
            return_value=SimpleNamespace(enable_dsa_cp_moe_dbo=False),
        ),
        patch.object(MoECommMethod, "fused_experts", return_value=sentinel) as synchronous,
    ):
        assert comm_impl.fused_experts(fused_input) is sentinel
        synchronous.assert_called_once_with(fused_input)


def test_any_rank_can_force_consistent_fallback():
    fused_input = _make_fused_experts_input(5)
    comm_impl = object.__new__(AlltoAllCommImpl)
    comm_impl.token_dispatcher = MagicMock()
    forward_context = SimpleNamespace()

    def reject_on_remote_rank(eligibility, **_kwargs):
        eligibility.zero_()

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config",
            return_value=SimpleNamespace(enable_dsa_cp_moe_dbo=True),
        ),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_forward_context", return_value=forward_context),
        patch.object(comm_impl, "_local_dsa_cp_moe_dbo_candidate", return_value=(True, "eligible")),
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.dist.all_reduce",
            side_effect=reject_on_remote_rank,
        ) as all_reduce,
    ):
        assert not comm_impl._dsa_cp_moe_dbo_eligible(fused_input)
        assert not comm_impl._dsa_cp_moe_dbo_eligible(fused_input)
        all_reduce.assert_called_once()


def test_dsa_cp_moe_dbo_config_defaults_and_threshold_validation():
    config = AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    assert not config.enable_dsa_cp_moe_dbo
    assert not config.enable_dsa_cp_moe_dbo_shared_expert_overlap
    assert config.dsa_cp_moe_dbo_token_threshold == 32

    with pytest.raises(ValueError, match="must be at least 2"):
        AscendConfig(
            sparse_kv_offload_config=SimpleNamespace(enabled=False),
            dsa_cp_moe_dbo_token_threshold=1,
        )
