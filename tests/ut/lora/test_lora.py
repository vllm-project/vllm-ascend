from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.lora.layers.base import BaseLayerWithLoRA

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
)
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU, _lora_bmm_expand_slice_op


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
def test_compatible_lora_bmm_expand_slice_matches_reference(add_inputs: bool) -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
            [[[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]]],
        ]
    )
    indices = torch.tensor([0, 1, -1], dtype=torch.long)
    y = torch.ones((3, 5))

    _lora_bmm_expand_slice_op(y, x, weights, indices, 1, 3, add_inputs)

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
        _lora_bmm_expand_slice_op(y, x, weights, indices, 1, slice_size, True)
