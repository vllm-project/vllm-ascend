from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoEWithLoRA,
    _recover_moe_lora_routing_all2all,
    _recover_moe_lora_routing_allgather,
    has_lora,
    _moe_lora_projection_enabled,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
)
from vllm_ascend.lora.lora_ops import bmm_expand_slice
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


@pytest.mark.parametrize(
    ("rank", "slice_size", "expect_fallback"),
    [(4, 8, False), (16, 8, True)],
)
def test_expand_slice_selects_fallback_from_tensor_shape(
    rank: int,
    slice_size: int,
    expect_fallback: bool,
) -> None:
    wrapper: Any = SimpleNamespace(
        no_lora=False,
        _bmm_expand_slice=Mock(),
        sgmv_expand_slice=Mock(),
        prefill_metadata=("batches", "tokens", "indices"),
    )
    wrapper._requires_bmm_expand_slice = MethodType(PunicaWrapperNPU._requires_bmm_expand_slice, wrapper)
    x = SimpleNamespace(shape=(2, rank))

    PunicaWrapperNPU._expand_slice_prefill(
        wrapper,
        "y",
        x,
        "weights",
        4,
        slice_size,
        True,
    )

    if expect_fallback:
        wrapper._bmm_expand_slice.assert_called_once_with("y", x, "weights", 4, slice_size, True)
        wrapper.sgmv_expand_slice.assert_not_called()
    else:
        wrapper._bmm_expand_slice.assert_not_called()
        wrapper.sgmv_expand_slice.assert_called_once_with(
            x,
            "weights",
            "y",
            "batches",
            "tokens",
            "indices",
            4,
            slice_size,
            True,
        )


@pytest.mark.parametrize("add_inputs", [False, True])
def test_lora_bmm_expand_slice_fallback_matches_reference(add_inputs: bool) -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
            [[[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]]],
        ]
    )
    indices = torch.tensor([0, 1, -1], dtype=torch.long)
    y = torch.ones((3, 5))

    bmm_expand_slice(x, weights, y, indices, 1, 3, add_inputs)

    delta = torch.stack(
        [
            x[0] @ weights[0, 0].T,
            x[1] @ weights[1, 0].T,
            torch.zeros(3),
        ]
    )
    expected = torch.ones((3, 5))
    expected[:, 1:4] = expected[:, 1:4] + delta if add_inputs else delta
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize(
    ("x_shape", "weight_shape", "indices_shape", "y_shape", "slice_size", "message"),
    [
        ((3, 4), (2, 1, 5, 2), (3,), (3, 8), 5, "shrink rank"),
        ((3, 2), (2, 1, 5, 2), (2,), (3, 8), 5, "same row count"),
        ((3, 2), (2, 1, 5, 2), (3,), (2, 8), 5, "same row count"),
        ((3, 2), (2, 1, 4, 2), (3,), (3, 8), 5, "destination slice"),
    ],
)
def test_lora_bmm_expand_slice_rejects_incompatible_shapes(
    x_shape,
    weight_shape,
    indices_shape,
    y_shape,
    slice_size,
    message,
) -> None:
    x = torch.zeros(x_shape)
    weights = torch.zeros(weight_shape)
    indices = torch.zeros(indices_shape, dtype=torch.long)
    y = torch.zeros(y_shape)

    with pytest.raises(ValueError, match=message):
        bmm_expand_slice(x, weights, y, indices, 1, slice_size, True)


@pytest.mark.parametrize(
    ("lora_b", "w13_num_slices", "expected"),
    [
        ([torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3)], 2, (False, False)),
        ([torch.ones(2, 3), torch.zeros(2, 3), torch.zeros(2, 3)], 2, (True, False)),
        ([torch.zeros(2, 3), torch.ones(2, 3), torch.zeros(2, 3)], 2, (False, True)),
        ([torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, 3)], 2, (True, False)),
        ([torch.ones(2, 3), torch.zeros(2, 3)], 1, (True, False)),
    ],
)
def test_moe_lora_projection_enabled(lora_b, w13_num_slices, expected) -> None:
    assert _moe_lora_projection_enabled(lora_b, w13_num_slices) == expected


def test_moe_lora_apply_uses_projection_specific_enable_masks() -> None:
    punica_wrapper = Mock()
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
        adapter_enabled="all_enabled",
        w13_adapter_enabled="w13_enabled",
        w2_adapter_enabled="w2_enabled",
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

    assert punica_wrapper.add_lora_fused_moe.call_count == 2
    assert punica_wrapper.add_lora_fused_moe.call_args_list[0].kwargs["adapter_enabled"] == "w13_enabled"
    assert punica_wrapper.add_lora_fused_moe.call_args_list[1].kwargs["adapter_enabled"] == "w2_enabled"


def test_moe_lora_projection_masks_follow_adapter_lifecycle() -> None:
    layer = object.__new__(AscendFusedMoEWithLoRA)
    BaseLayerWithLoRA.__init__(layer)
    layer._w13_slices = 2

    def create_weights(module, max_loras, lora_config, model_config=None):
        module.adapter_enabled = torch.zeros(max_loras + 1, dtype=torch.int)

    context = SimpleNamespace()
    with (
        patch.object(FusedMoEWithLoRA, "create_lora_weights", create_weights),
        patch.object(FusedMoEWithLoRA, "set_lora"),
        patch.object(FusedMoEWithLoRA, "reset_lora"),
        patch.object(FusedMoEWithLoRA, "_build_lora_context", return_value=context),
    ):
        layer.create_lora_weights(1, SimpleNamespace())
        layer.set_lora(
            0,
            [torch.empty(0)] * 3,
            [torch.zeros(2, 3), torch.ones(2, 3), torch.zeros(2, 3)],
        )

        assert layer.w13_adapter_enabled.tolist() == [0, 0]
        assert layer.w2_adapter_enabled.tolist() == [1, 0]
        assert layer._build_lora_context() is context
        assert context.w13_adapter_enabled is layer.w13_adapter_enabled
        assert context.w2_adapter_enabled is layer.w2_adapter_enabled

        layer.set_lora(
            0,
            [torch.empty(0)] * 3,
            [torch.ones(2, 3), torch.zeros(2, 3), torch.zeros(2, 3)],
        )
        assert layer.w13_adapter_enabled.tolist() == [1, 0]
        assert layer.w2_adapter_enabled.tolist() == [0, 0]

        layer.reset_lora(0)
        assert layer.w13_adapter_enabled.tolist() == [0, 0]
        assert layer.w2_adapter_enabled.tolist() == [0, 0]
