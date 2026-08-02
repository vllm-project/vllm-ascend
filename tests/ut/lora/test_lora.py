from types import MethodType, SimpleNamespace
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
    ):
        wrapper = AscendFusedMoEWithLoRA(base_layer)

    assert wrapper._lora_stream is None
    assert wrapper._events is None
    assert wrapper.enable_moe_shared_loras is False
    assert wrapper._shared_experts is shared_experts
    assert wrapper.n_slices == 256 * 3


def test_moe_lora_apply_uses_adapter_enabled() -> None:
    punica_wrapper = Mock()
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
        adapter_enabled="all_enabled",
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


def test_allgather_routing_preserves_multi_adapter_and_base_mapping() -> None:
    context = SimpleNamespace(
        top_k=2,
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([0, -1, 1])),
    )
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 1]])
    # Original flat rows [0..5] land at these expert-sorted positions.
    expanded_row_idx = torch.tensor([2, 0, 1, 3, 4, 5])

    expert_ids, lora_slots = _recover_moe_lora_routing_allgather(context, expanded_row_idx, topk_ids)

    assert torch.equal(expert_ids, torch.tensor([0, 0, 1, 1, 1, 1]))
    assert torch.equal(lora_slots, torch.tensor([0, -1, 0, -1, 1, 1]))


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
@pytest.mark.parametrize("add_inputs", [True, False])
def test_single_lora_linear_masks_base_rows(add_inputs: bool) -> None:
    token_indices = torch.tensor([0, -1, 0, -1, 0])
    adapter_mask = token_indices.eq(0).unsqueeze(1).to(torch.bfloat16)
    wrapper = SimpleNamespace(
        _single_lora_slot=True,
        _get_single_lora_mask=Mock(return_value=adapter_mask),
    )
    x = torch.randn(5, 6, dtype=torch.bfloat16)
    y = torch.randn(5, 7, dtype=torch.bfloat16)
    original_y = y.clone()
    lora_a = (torch.randn(1, 1, 3, 6, dtype=torch.bfloat16),)
    lora_b = (torch.randn(1, 1, 7, 3, dtype=torch.bfloat16),)
    scale = 0.5

    applied = PunicaWrapperNPU._apply_single_lora_linear(
        wrapper,
        y,
        x,
        lora_a,
        lora_b,
        scale,
        (7,),
        add_inputs=add_inputs,
    )

    delta = torch.matmul(
        torch.matmul(x, lora_a[0][0, 0].transpose(0, 1)),
        lora_b[0][0, 0].transpose(0, 1),
    )
    delta.mul_(token_indices.eq(0).unsqueeze(1))
    expected = original_y.add(delta, alpha=scale) if add_inputs else delta.mul(scale)
    assert applied
    torch.testing.assert_close(y, expected)


def test_single_lora_mask_is_refreshed_with_metadata() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._token_lora_indices = torch.tensor([0, -1, 0, -1])
    wrapper._single_lora_mask = torch.empty(4, 1, dtype=torch.bfloat16)
    wrapper.indices_len = [4, 0, 0, 0]

    with patch.object(PunicaWrapperBase, "_update_base_metadata"):
        PunicaWrapperNPU._update_base_metadata(wrapper, Mock(), [], 1, 100)

    torch.testing.assert_close(
        wrapper._single_lora_mask,
        torch.tensor([[1], [0], [1], [0]], dtype=torch.bfloat16),
    )


def test_single_lora_mask_matches_input_rows() -> None:
    wrapper = SimpleNamespace(
        _single_lora_mask=torch.tensor([[1], [0], [1], [0]], dtype=torch.bfloat16),
    )
    x = torch.empty(3, 5)

    mask = PunicaWrapperNPU._get_single_lora_mask(wrapper, x)

    torch.testing.assert_close(mask, wrapper._single_lora_mask[:3])


@pytest.mark.parametrize("add_inputs", [True, False])
@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_single_lora_linear_packed_slices(add_inputs: bool, scale: float) -> None:
    token_indices = torch.tensor([0, -1, 0, -1])
    adapter_mask = token_indices.eq(0).unsqueeze(1).to(torch.bfloat16)
    wrapper = SimpleNamespace(
        _single_lora_slot=True,
        _get_single_lora_mask=Mock(return_value=adapter_mask),
    )
    x = torch.randn(4, 6, dtype=torch.bfloat16)
    y = torch.randn(4, 7, dtype=torch.bfloat16)
    original_y = y.clone()
    lora_a = (
        torch.randn(1, 1, 3, 6, dtype=torch.bfloat16),
        torch.randn(1, 1, 3, 6, dtype=torch.bfloat16),
    )
    lora_b = (
        torch.randn(1, 1, 4, 3, dtype=torch.bfloat16),
        torch.randn(1, 1, 3, 3, dtype=torch.bfloat16),
    )
    applied = PunicaWrapperNPU._apply_single_lora_linear(
        wrapper,
        y,
        x,
        lora_a,
        lora_b,
        scale,
        (4, 3),
        add_inputs=add_inputs,
    )

    deltas = []
    for a_weight, b_weight in zip(lora_a, lora_b, strict=True):
        shrink = torch.matmul(x, a_weight[0, 0].transpose(0, 1))
        shrink.mul_(adapter_mask)
        deltas.append(torch.matmul(shrink, b_weight[0, 0].transpose(0, 1)))
    delta = torch.cat(deltas, dim=1)
    expected = original_y.add(delta, alpha=scale) if add_inputs else delta.mul(scale)
    assert applied
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("add_inputs", [True, False])
@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_single_lora_linear_uses_prepacked_a(add_inputs: bool, scale: float) -> None:
    adapter_mask = torch.tensor([[1], [0], [1], [0]], dtype=torch.bfloat16)
    wrapper = SimpleNamespace(
        _single_lora_slot=True,
        _get_single_lora_mask=Mock(return_value=adapter_mask),
    )
    x = torch.randn(4, 6, dtype=torch.bfloat16)
    y = torch.randn(4, 7, dtype=torch.bfloat16)
    original_y = y.clone()
    lora_a = (
        torch.randn(1, 1, 3, 6, dtype=torch.bfloat16),
        torch.randn(1, 1, 3, 6, dtype=torch.bfloat16),
    )
    packed_lora_a = torch.cat(lora_a, dim=2)
    lora_b = (
        torch.randn(1, 1, 4, 3, dtype=torch.bfloat16),
        torch.randn(1, 1, 3, 3, dtype=torch.bfloat16),
    )
    applied = PunicaWrapperNPU._apply_single_lora_linear(
        wrapper,
        y,
        x,
        lora_a,
        lora_b,
        scale,
        (4, 3),
        packed_lora_a=packed_lora_a,
        add_inputs=add_inputs,
    )

    shrink = torch.matmul(x, packed_lora_a[0, 0].transpose(0, 1))
    shrink.mul_(adapter_mask)
    delta = torch.cat(
        (
            torch.matmul(shrink[:, :3], lora_b[0][0, 0].transpose(0, 1)),
            torch.matmul(shrink[:, 3:], lora_b[1][0, 0].transpose(0, 1)),
        ),
        dim=1,
    )
    expected = original_y.add(delta, alpha=scale) if add_inputs else delta.mul(scale)
    assert applied
    torch.testing.assert_close(y, expected)


def test_non_homogeneous_prefill_linear_falls_back() -> None:
    wrapper = SimpleNamespace(
        no_lora=False,
        _single_lora_slot=False,
        add_shrink=Mock(),
        add_expand=Mock(),
    )
    wrapper._apply_single_lora_linear = MethodType(PunicaWrapperNPU._apply_single_lora_linear, wrapper)
    x = torch.randn(2, 4)
    y = torch.randn(2, 6)
    lora_a = (torch.randn(1, 1, 2, 4),)
    lora_b = (torch.randn(1, 1, 6, 2),)
    buffer = (torch.empty(2, 2),)

    PunicaWrapperNPU.add_lora_linear(
        wrapper,
        y,
        x,
        lora_a,
        lora_b,
        1.0,
        (6,),
        buffer=buffer,
    )

    wrapper.add_shrink.assert_called_once_with(buffer, x, lora_a, 1.0)
    wrapper.add_expand.assert_called_once_with(y, buffer, lora_b, (6,), add_inputs=True)
