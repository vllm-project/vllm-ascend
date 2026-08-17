from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoEWithLoRA,
    _recover_moe_lora_routing_all2all,
    _recover_moe_lora_routing_allgather,
    has_lora,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
)
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


def test_ascend_fused_moe_lora_initializes_skipped_upstream_fields() -> None:
    parallel_config = SimpleNamespace(tp_size=8, tp_rank=3, ep_rank=0, use_ep=False)
    shared_experts = torch.nn.Module()
    base_layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            hidden_dim=4096,
            num_local_experts=256,
            num_experts=256,
            intermediate_size_per_partition=256,
            experts_per_token=8,
            moe_parallel_config=parallel_config,
            is_act_and_mul=True,
        ),
        _shared_experts=shared_experts,
    )

    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
        patch(
            "vllm_ascend.lora.fused_moe.get_ascend_config",
            return_value=SimpleNamespace(enable_moe_lora_dual_stream=False),
        ),
    ):
        wrapper = AscendFusedMoEWithLoRA(base_layer)

    assert wrapper._lora_stream is None
    assert wrapper._events is None
    assert wrapper.enable_moe_shared_loras is False
    assert wrapper._shared_experts is shared_experts
    assert wrapper.n_slices == 256 * 3


def test_ascend_fused_moe_lora_initializes_npu_aux_stream() -> None:
    parallel_config = SimpleNamespace(tp_size=8, tp_rank=3, ep_rank=0, use_ep=False)
    base_layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            hidden_dim=4096,
            num_local_experts=256,
            num_experts=256,
            intermediate_size_per_partition=256,
            experts_per_token=8,
            moe_parallel_config=parallel_config,
            is_act_and_mul=True,
        ),
        _shared_experts=None,
    )
    aux_stream = object()
    events = [object() for _ in range(4)]

    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
        patch(
            "vllm_ascend.lora.fused_moe.get_ascend_config",
            return_value=SimpleNamespace(enable_moe_lora_dual_stream=True),
        ),
        patch("vllm_ascend.lora.fused_moe._get_moe_lora_aux_stream", return_value=aux_stream),
        patch("vllm_ascend.lora.fused_moe.torch.npu.Event", side_effect=events),
    ):
        wrapper = AscendFusedMoEWithLoRA(base_layer)

    assert wrapper._lora_stream is aux_stream
    assert wrapper._events == tuple(events)


def test_ascend_fused_moe_lora_initializes_aux_stream_for_graph_mode() -> None:
    parallel_config = SimpleNamespace(tp_size=8, tp_rank=3, ep_rank=0, use_ep=False)
    base_layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            hidden_dim=4096,
            num_local_experts=256,
            num_experts=256,
            intermediate_size_per_partition=256,
            experts_per_token=8,
            moe_parallel_config=parallel_config,
            is_act_and_mul=True,
        ),
        _shared_experts=None,
    )
    ascend_config = SimpleNamespace(
        enable_moe_lora_dual_stream=True,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False)),
    )
    aux_stream = object()
    events = [object() for _ in range(4)]

    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
        patch("vllm_ascend.lora.fused_moe.get_ascend_config", return_value=ascend_config),
        patch("vllm_ascend.lora.fused_moe._get_moe_lora_aux_stream", return_value=aux_stream),
        patch("vllm_ascend.lora.fused_moe.torch.npu.Event", side_effect=events),
    ):
        wrapper = AscendFusedMoEWithLoRA(base_layer)

    assert wrapper._lora_stream is aux_stream
    assert wrapper._events == tuple(events)


def test_moe_lora_apply_uses_adapter_enabled() -> None:
    punica_wrapper = Mock()
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
        adapter_enabled="all_enabled",
        fully_sharded=False,
        tp_rank=0,
    )
    routing = (torch.tensor([0]), torch.tensor([0]))

    moe_lora_apply_w13(
        context,
        gate_up_out="gate_up_out",
        hidden_states="hidden_states",
        lora_routing=routing,
    )
    moe_lora_apply_w2(
        context,
        down_out="down_out",
        silu_out="silu_out",
        lora_routing=routing,
    )

    calls = punica_wrapper.add_lora_fused_moe.call_args_list
    assert calls[0].kwargs["adapter_enabled"] == "all_enabled"
    assert calls[1].kwargs["adapter_enabled"] == "all_enabled"
    assert calls[0].kwargs["fully_sharded"] is False
    assert calls[1].kwargs["fully_sharded"] is False
    assert calls[1].kwargs["offset"] == 0


def test_moe_lora_apply_reuses_allgather_combined_idx() -> None:
    punica_wrapper = Mock()
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
        adapter_enabled="all_enabled",
        fully_sharded=False,
        tp_rank=0,
    )
    combined_idx = torch.tensor([3, -1, 5])

    moe_lora_apply_w13(
        context,
        gate_up_out="gate_up_out",
        hidden_states="hidden_states",
        lora_routing=combined_idx,
    )
    moe_lora_apply_w2(
        context,
        down_out="down_out",
        silu_out="silu_out",
        lora_routing=combined_idx,
    )

    calls = punica_wrapper.add_lora_fused_moe.call_args_list
    assert calls[0].kwargs["combined_idx"] is combined_idx
    assert calls[1].kwargs["combined_idx"] is combined_idx
    assert calls[0].kwargs["expert_ids"] is None
    assert calls[1].kwargs["expert_ids"] is None
    assert calls[0].kwargs["token_lora_mapping"] is None
    assert calls[1].kwargs["token_lora_mapping"] is None


def test_moe_lora_apply_propagates_fully_sharded_metadata() -> None:
    punica_wrapper = Mock()
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked=(torch.empty(1, 1, 16, 8),),
        adapter_enabled="all_enabled",
        fully_sharded=True,
        tp_rank=3,
    )
    routing = (torch.tensor([0]), torch.tensor([0]))

    moe_lora_apply_w13(
        context,
        gate_up_out="gate_up_out",
        hidden_states="hidden_states",
        lora_routing=routing,
    )
    moe_lora_apply_w2(
        context,
        down_out="down_out",
        silu_out="silu_out",
        lora_routing=routing,
    )

    calls = punica_wrapper.add_lora_fused_moe.call_args_list
    assert calls[0].kwargs["fully_sharded"] is True
    assert calls[1].kwargs["fully_sharded"] is True
    assert calls[1].kwargs["offset"] == 48


def test_punica_fully_sharded_moe_gathers_rank_shards() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)

    def shrink(_, __, output, ___, ____):
        output.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    wrapper.bgmv_shrink = Mock(side_effect=shrink)
    wrapper.bgmv_expand_slice = Mock()
    lora_a = (torch.zeros(2, 2, 2, 3),)
    lora_b = (torch.zeros(2, 2, 5, 4),)

    with (
        patch(
            "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather",
            side_effect=lambda value: torch.cat((value, value + 10), dim=-1),
        ) as all_gather,
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce") as all_reduce,
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 5),
            x=torch.zeros(2, 3),
            lora_a_stacked=lora_a,
            lora_b_stacked=lora_b,
            expert_ids=torch.tensor([0, 1]),
            adapter_enabled=torch.tensor([1, 1]),
            fully_sharded=True,
            token_lora_mapping=torch.tensor([0, 1]),
        )

    all_gather.assert_called_once()
    all_reduce.assert_not_called()
    expand_args = wrapper.bgmv_expand_slice.call_args.args
    assert torch.equal(
        expand_args[0],
        torch.tensor([[1.0, 2.0, 11.0, 12.0], [3.0, 4.0, 13.0, 14.0]]),
    )
    assert expand_args[1].shape == (4, 5, 4)
    assert torch.equal(expand_args[3], torch.tensor([0, 3]))


def test_punica_moe_uses_precomputed_combined_idx() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()
    combined_idx = torch.tensor([3, -1])

    wrapper.add_lora_fused_moe(
        y=torch.zeros(2, 5),
        x=torch.zeros(2, 3),
        lora_a_stacked=(torch.zeros(2, 2, 2, 3),),
        lora_b_stacked=(torch.zeros(2, 2, 5, 2),),
        expert_ids=None,
        adapter_enabled=torch.tensor([True, True]),
        combined_idx=combined_idx,
    )

    shrink_idx = wrapper.bgmv_shrink.call_args.args[3]
    expand_idx = wrapper.bgmv_expand_slice.call_args.args[3]
    assert shrink_idx.data_ptr() == combined_idx.data_ptr()
    assert expand_idx.data_ptr() == combined_idx.data_ptr()


def test_punica_fully_sharded_moe_reduces_partial_rank() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)

    def shrink(_, __, output, ___, ____):
        output.copy_(torch.arange(8, dtype=torch.float32).view(2, 4))

    wrapper.bgmv_shrink = Mock(side_effect=shrink)
    wrapper.bgmv_expand_slice = Mock()
    lora_a = (torch.zeros(2, 2, 4, 3),)
    lora_b = (torch.zeros(2, 2, 5, 4),)

    with (
        patch("vllm_ascend.lora.punica_npu.tensor_model_parallel_all_gather") as all_gather,
        patch(
            "vllm_ascend.lora.punica_npu.tensor_model_parallel_all_reduce",
            side_effect=lambda value: value + 10,
        ) as all_reduce,
    ):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(2, 10),
            x=torch.zeros(2, 3),
            lora_a_stacked=lora_a,
            lora_b_stacked=lora_b,
            expert_ids=torch.tensor([0, 1]),
            adapter_enabled=torch.tensor([1, 1]),
            fully_sharded=True,
            offset=5,
            token_lora_mapping=torch.tensor([0, 1]),
        )

    all_gather.assert_not_called()
    all_reduce.assert_called_once()
    expand_args = wrapper.bgmv_expand_slice.call_args.args
    assert torch.equal(
        expand_args[0],
        torch.arange(8, dtype=torch.float32).view(2, 4) + 10,
    )
    assert expand_args[1].shape == (4, 5, 4)
    assert expand_args[4] == 5


def test_allgather_routing_preserves_multi_adapter_and_base_mapping() -> None:
    context = SimpleNamespace(
        top_k=2,
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([0, -1, 1])),
        adapter_enabled=torch.tensor([True, True]),
        w13_lora_a_stacked=(torch.empty(2, 2, 1, 1),),
    )
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 1]])
    # Original flat rows [0..5] land at these expert-sorted positions.
    expanded_row_idx = torch.tensor([2, 0, 1, 3, 4, 5])

    with patch("vllm_ascend.lora.fused_moe.torch.argsort", side_effect=AssertionError("unexpected argsort")):
        combined_idx = _recover_moe_lora_routing_allgather(context, expanded_row_idx, topk_ids)

    # num_experts=2, so active rows use lora_slot * 2 + expert_id.
    assert torch.equal(combined_idx, torch.tensor([0, -1, 1, -1, 3, 3]))


def test_all2all_routing_uses_local_experts_and_exchanged_adapters() -> None:
    context = SimpleNamespace(
        local_num_experts=3,
        exchanged_lora_indices=torch.tensor([1, -1, 0, 2]),
    )

    expert_ids, lora_slots = _recover_moe_lora_routing_all2all(
        context,
        group_list=torch.tensor([2, 0, 2]),
    )

    assert torch.equal(expert_ids, torch.tensor([0, 0, 2, 2]))
    assert torch.equal(lora_slots, torch.tensor([1, -1, 0, 2]))


def test_has_lora_follows_batch_metadata() -> None:
    assert not has_lora(None)
    assert not has_lora(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=True)))
    assert has_lora(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=False)))


@pytest.mark.parametrize(
    ("index_mapping", "expected_no_lora"),
    [((0, 0), True), ((0, 1), False), ((2, 0), False)],
)
def test_decode_metadata_refreshes_no_lora(index_mapping, expected_no_lora) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    mapping = SimpleNamespace(index_mapping=index_mapping)
    with patch.object(PunicaWrapperBase, "update_metadata"):
        wrapper.update_metadata(mapping, [], 2, 100)
    assert wrapper.no_lora is expected_no_lora
