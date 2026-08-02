from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
    _moe_lora_projection_enabled,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
)
from vllm_ascend.lora.lora_ops import bmm_expand_slice
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


def test_ascend_fused_moe_3d_initializes_upstream_weight_state() -> None:
    moe_config = SimpleNamespace(
        is_act_and_mul=True,
        num_local_experts=10,
        moe_parallel_config=SimpleNamespace(tp_size=4, tp_rank=1),
    )
    shared_experts = object()
    base_layer = SimpleNamespace(
        moe_config=moe_config,
        _shared_experts=shared_experts,
    )

    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value="cpu"),
        patch.object(BaseLayerWithLoRA, "__init__", return_value=None),
    ):
        wrapper = AscendFusedMoE3DWithLoRA(base_layer)

    assert wrapper.enable_moe_shared_loras is False
    assert wrapper._shared_experts is shared_experts
    assert wrapper._w13_a_num_experts == moe_config.num_local_experts
    assert wrapper._lora_stream is None
    assert wrapper._events is None
    assert wrapper._w13_slices == 1
    assert wrapper.n_slices == moe_config.num_local_experts * 2


@pytest.mark.parametrize("shared_experts", [None, object()])
def test_shared_experts_keep_shape_driven_expand_dispatch(shared_experts) -> None:
    assert not hasattr(PunicaWrapperNPU, "enable_compatible_lora_bmm_expand_slice")

    base_layer = SimpleNamespace(
        _shared_experts=shared_experts,
        set_lora_context=Mock(),
    )
    wrapper: Any = SimpleNamespace(
        base_layer=base_layer,
        _build_lora_context=Mock(return_value="context"),
    )
    punica_wrapper = Mock()

    with patch.object(BaseLayerWithLoRA, "set_mapping"):
        AscendFusedMoEWithLoRA.set_mapping(wrapper, punica_wrapper)

    punica_wrapper.enable_compatible_lora_bmm_expand_slice.assert_not_called()
    base_layer.set_lora_context.assert_called_once_with("context")


@pytest.mark.parametrize(
    ("rank", "slice_size", "expect_compatible_path"),
    [
        (4, 8, False),
        (16, 8, True),
    ],
)
def test_expand_slice_path_follows_model_structure_and_tensor_shape(
    rank: int,
    slice_size: int,
    expect_compatible_path: bool,
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

    if expect_compatible_path:
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
    x_shape, weight_shape, indices_shape, y_shape, slice_size, message
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
