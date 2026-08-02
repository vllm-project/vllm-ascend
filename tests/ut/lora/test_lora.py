from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoEWithLoRA,
    _moe_lora_projection_enabled,
    _recover_moe_lora_routing,
    is_moe_lora_active,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
)
from vllm_ascend.lora.punica_npu import (
    PunicaWrapperNPU,
    _moe_lora_bmm_expand_slice_op,
)


def test_moe_wrapper_preserves_shared_expert_module_path() -> None:
    shared_experts = torch.nn.Linear(2, 2)
    base_layer = SimpleNamespace(
        use_ep=False,
        dynamic_eplb=False,
        _shared_experts=shared_experts,
        multistream_overlap_gate=False,
        local_num_experts=4,
        moe_config=SimpleNamespace(is_act_and_mul=True),
    )

    with (
        patch(
            "vllm_ascend.lora.fused_moe._get_lora_device",
            return_value=torch.device("cpu"),
        ),
        patch(
            "vllm_ascend.lora.fused_moe.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm_ascend.lora.fused_moe.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm_ascend.lora.fused_moe.envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2",
            0,
        ),
    ):
        wrapper = AscendFusedMoEWithLoRA(base_layer)

    assert wrapper._shared_experts is shared_experts
    assert wrapper.base_layer._shared_experts is shared_experts
    assert wrapper.n_slices == 12


@pytest.mark.parametrize("shared_experts", [None, object()])
def test_shared_experts_select_compatible_expand_slice(shared_experts) -> None:
    base_layer = SimpleNamespace(
        _shared_experts=shared_experts,
        set_lora_context=Mock(),
    )
    wrapper = SimpleNamespace(
        base_layer=base_layer,
        _build_lora_context=Mock(return_value="context"),
    )
    punica_wrapper = Mock()

    with patch.object(BaseLayerWithLoRA, "set_mapping"):
        AscendFusedMoEWithLoRA.set_mapping(wrapper, punica_wrapper)

    if shared_experts is None:
        punica_wrapper.enable_compatible_lora_bmm_expand_slice.assert_not_called()
    else:
        punica_wrapper.enable_compatible_lora_bmm_expand_slice.assert_called_once()
    base_layer.set_lora_context.assert_called_once_with("context")


@pytest.mark.parametrize(
    ("force_compatible_path", "rank", "slice_size", "expect_bmm"),
    [
        (False, 4, 8, False),
        (False, 16, 8, True),
        (True, 4, 8, True),
    ],
)
def test_expand_slice_selects_compatible_path(
    force_compatible_path,
    rank,
    slice_size,
    expect_bmm,
) -> None:
    wrapper = SimpleNamespace(
        _force_lora_bmm_expand_slice=force_compatible_path,
    )

    selected = PunicaWrapperNPU._requires_bmm_expand_slice(
        wrapper,
        torch.empty(2, rank),
        slice_size,
    )

    assert selected is expect_bmm


@pytest.mark.parametrize("add_inputs", [True, False])
def test_bmm_expand_slice_masks_base_tokens(add_inputs) -> None:
    x = torch.tensor([[1.0, 2.0], [4.0, 5.0], [2.0, 3.0]])
    weights = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
            [[[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]]],
        ]
    )
    indices = torch.tensor([0, -1, 1])
    y = torch.full((3, 5), 7.0)
    original = y.clone()

    _moe_lora_bmm_expand_slice_op(
        y,
        x,
        weights,
        indices,
        1,
        3,
        add_inputs,
    )

    delta = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 6.0, -1.0]])
    expected_slice = original[:, 1:4] + delta if add_inputs else delta
    torch.testing.assert_close(y[:, 1:4], expected_slice)
    torch.testing.assert_close(y[:, :1], original[:, :1])
    torch.testing.assert_close(y[:, 4:], original[:, 4:])


@pytest.mark.parametrize(
    ("lora_b", "w13_num_slices", "expected"),
    [
        ([torch.zeros(2, 3)] * 3, 2, (False, False)),
        (
            [torch.ones(2, 3), torch.zeros(2, 3), torch.zeros(2, 3)],
            2,
            (True, False),
        ),
        (
            [torch.zeros(2, 3), torch.ones(2, 3), torch.zeros(2, 3)],
            2,
            (False, True),
        ),
        (
            [torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, 3)],
            2,
            (True, False),
        ),
        ([torch.ones(2, 3), torch.zeros(2, 3)], 1, (True, False)),
    ],
)
def test_moe_lora_projection_enabled(
    lora_b,
    w13_num_slices,
    expected,
) -> None:
    assert _moe_lora_projection_enabled(lora_b, w13_num_slices) == expected


def test_moe_lora_apply_uses_projection_masks() -> None:
    punica_wrapper = Mock()
    punica_wrapper.token_lora_indices = torch.tensor([0])
    context = SimpleNamespace(
        top_k=1,
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
        adapter_enabled="all_enabled",
        w13_adapter_enabled="w13_enabled",
        w2_adapter_enabled="w2_enabled",
    )

    routing = moe_lora_apply_w13(
        context,
        gate_up_out="gate_up_out",
        hidden_states="hidden_states",
        expanded_row_idx=torch.tensor([0]),
        topk_ids=torch.tensor([[0]]),
    )
    moe_lora_apply_w2(
        context,
        down_out="down_out",
        silu_out="silu_out",
        lora_routing=routing,
    )

    assert punica_wrapper.add_lora_fused_moe.call_count == 2
    calls = punica_wrapper.add_lora_fused_moe.call_args_list
    assert calls[0].kwargs["adapter_enabled"] == "w13_enabled"
    assert calls[1].kwargs["adapter_enabled"] == "w2_enabled"


def test_allgather_routing_preserves_multi_adapter_and_base_mapping() -> None:
    context = SimpleNamespace(
        top_k=2,
        punica_wrapper=SimpleNamespace(
            token_lora_indices=torch.tensor([0, -1, 1]),
        ),
    )
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 1]])
    # Original flat rows [0..5] land at these expert-sorted positions.
    expanded_row_idx = torch.tensor([2, 0, 1, 3, 4, 5])

    expert_ids, lora_slots = _recover_moe_lora_routing(
        context,
        expanded_row_idx,
        topk_ids,
    )

    assert torch.equal(expert_ids, torch.tensor([0, 0, 1, 1, 1, 1]))
    assert torch.equal(lora_slots, torch.tensor([0, -1, 0, -1, 1, 1]))


def test_moe_lora_active_follows_batch_metadata() -> None:
    assert not is_moe_lora_active(None)
    assert not is_moe_lora_active(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=True)))
    assert is_moe_lora_active(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=False)))


@pytest.mark.parametrize(
    ("index_mapping", "expected_no_lora"),
    [
        ((0, 0), True),
        ((0, 1), False),
        ((2, 0), False),
    ],
)
def test_decode_metadata_refreshes_no_lora(
    index_mapping,
    expected_no_lora,
) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    mapping = SimpleNamespace(index_mapping=index_mapping)

    with patch.object(PunicaWrapperBase, "update_metadata"):
        wrapper.update_metadata(mapping, [], 2, 100)

    assert wrapper.no_lora is expected_no_lora
