from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoEWithLoRA,
    _moe_lora_projection_enabled,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
)
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.patch.worker.patch_lora_vlm_prefix import (
    _detect_prefix,
    _enable_wrapped_language_model_expand_slice,
)


def test_detects_vlm_wrapper_prefix() -> None:
    lora_keys = ["model.layers.0.mlp.down_proj"]
    model_modules = ["language_model.model.layers.0.mlp.down_proj"]

    assert _detect_prefix(lora_keys, model_modules) == "language_model."


def test_aligned_lora_modules_need_no_prefix() -> None:
    lora_keys = ["model.layers.0.mlp.down_proj"]

    assert _detect_prefix(lora_keys, lora_keys) == ""


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
        punica_wrapper.enable_compatible_lora_bmm_expand_slice.assert_called_once_with()
    base_layer.set_lora_context.assert_called_once_with("context")


@pytest.mark.parametrize("language_prefix", ["", "language_model"])
def test_wrapped_language_model_selects_compatible_expand_slice(language_prefix: str) -> None:
    punica_wrapper = Mock()
    manager = SimpleNamespace(
        supports_mm=True,
        mm_mapping=SimpleNamespace(language_model=[language_prefix]),
        punica_wrapper_mapping={language_prefix: punica_wrapper},
    )

    _enable_wrapped_language_model_expand_slice(manager)

    if language_prefix:
        punica_wrapper.enable_compatible_lora_bmm_expand_slice.assert_called_once_with()
    else:
        punica_wrapper.enable_compatible_lora_bmm_expand_slice.assert_not_called()


@pytest.mark.parametrize(
    ("force_compatible_path", "rank", "slice_size", "expect_compatible_path"),
    [
        (False, 4, 8, False),
        (False, 16, 8, True),
        (True, 4, 8, True),
    ],
)
def test_expand_slice_path_follows_model_structure_and_tensor_shape(
    force_compatible_path: bool,
    rank: int,
    slice_size: int,
    expect_compatible_path: bool,
) -> None:
    wrapper = SimpleNamespace(
        no_lora=False,
        _force_lora_bmm_expand_slice=force_compatible_path,
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


@pytest.mark.parametrize("add_inputs", [True, False])
def test_single_lora_linear_masks_base_rows(add_inputs: bool) -> None:
    token_indices = torch.tensor([0, -1, 0, -1, 0])
    wrapper = SimpleNamespace(
        _single_lora_slot=True,
        _get_token_lora_indices=Mock(return_value=token_indices),
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

        layer.reset_lora(0)
        assert layer.w13_adapter_enabled.tolist() == [0, 0]
        assert layer.w2_adapter_enabled.tolist() == [0, 0]
