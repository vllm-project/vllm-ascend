from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


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
