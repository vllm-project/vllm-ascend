from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
import vllm
from torch import nn
from vllm.lora.layers import MergedColumnParallelLinearWithLoRA, MergedQKVParallelLinearWithLoRA
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.lora.utils import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
    AscendMergedColumnParallelLinearWithLoRA,
    AscendMergedQKVParallelLinearWithLoRA,
    _PackedLoRAAWeightsMixin,
    refresh_all_lora_classes,
)
from vllm_ascend.utils import AscendDeviceType


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3])
def test_single_lora_slot_enabled_on_a2_and_a3(device_type: AscendDeviceType) -> None:
    lora_config = SimpleNamespace(
        max_lora_rank=8,
        max_loras=1,
        fully_sharded_loras=False,
        lora_dtype=torch.bfloat16,
    )

    with (
        patch.object(PunicaWrapperBase, "__init__", return_value=None),
        patch("vllm_ascend.lora.punica_npu.refresh_all_lora_classes"),
        patch("vllm_ascend.lora.punica_npu.get_ascend_device_type", return_value=device_type),
    ):
        wrapper = PunicaWrapperNPU(16, 4, "cpu", lora_config=lora_config)

    assert wrapper._single_lora_slot
    assert wrapper._single_lora_mask is not None
    assert wrapper._single_lora_mask.shape == (16, 1)


@pytest.mark.parametrize("add_inputs", [True, False])
def test_single_lora_linear_masks_base_rows(add_inputs: bool) -> None:
    token_indices = torch.tensor([0, -1, 0, -1, 0])
    adapter_mask = token_indices.eq(0).unsqueeze(1).to(torch.bfloat16)
    wrapper: Any = SimpleNamespace(
        _single_lora_slot=True,
        _single_lora_mask=adapter_mask,
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


@pytest.mark.parametrize("add_inputs", [True, False])
@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_single_lora_linear_packed_slices(add_inputs: bool, scale: float) -> None:
    token_indices = torch.tensor([0, -1, 0, -1])
    adapter_mask = token_indices.eq(0).unsqueeze(1).to(torch.bfloat16)
    wrapper: Any = SimpleNamespace(
        _single_lora_slot=True,
        _single_lora_mask=adapter_mask,
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
    wrapper: Any = SimpleNamespace(
        _single_lora_slot=True,
        _single_lora_mask=adapter_mask,
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
    wrapper: Any = SimpleNamespace(
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


def test_packed_lora_wrappers_extend_only_non_sharded_merged_layers() -> None:
    assert AscendMergedColumnParallelLinearWithLoRA.__mro__[:3] == (
        AscendMergedColumnParallelLinearWithLoRA,
        _PackedLoRAAWeightsMixin,
        MergedColumnParallelLinearWithLoRA,
    )
    assert MergedQKVParallelLinearWithLoRA in AscendMergedQKVParallelLinearWithLoRA.__mro__
    assert all("Sharded" not in base.__name__ for base in AscendMergedQKVParallelLinearWithLoRA.__mro__)


@pytest.mark.parametrize(("max_loras", "expected"), [(1, True), (2, False)])
def test_merged_column_packed_wrapper_requires_single_adapter_slot(max_loras: int, expected: bool) -> None:
    lora_config: Any = SimpleNamespace(max_loras=max_loras, fully_sharded_loras=False)

    with patch("vllm_ascend.lora.utils.maybe_get_oot_by_class", return_value=nn.Linear):
        can_replace = AscendMergedColumnParallelLinearWithLoRA.can_replace_layer(
            source_layer=nn.Linear(2, 2),
            lora_config=lora_config,
            packed_modules_list=["gate", "up"],
            model_config=None,
        )

    assert can_replace is expected


@pytest.mark.parametrize(("max_loras", "expected"), [(1, True), (2, False)])
def test_qkv_packed_wrapper_requires_single_adapter_slot(max_loras: int, expected: bool) -> None:
    class FakeAscendQKVParallelLinear(nn.Module):
        pass

    lora_config: Any = SimpleNamespace(max_loras=max_loras, fully_sharded_loras=False)
    source_layer = FakeAscendQKVParallelLinear()

    with patch("vllm_ascend.lora.utils.AscendQKVParallelLinear", FakeAscendQKVParallelLinear):
        can_replace = AscendMergedQKVParallelLinearWithLoRA.can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=["q", "k", "v"],
            model_config=None,
        )

    assert can_replace is expected


def test_refresh_lora_classes_prioritizes_packed_wrappers() -> None:
    original_classes = vllm.lora.utils._all_lora_classes
    ascend_classes = (
        AscendMergedColumnParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithLoRA,
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )
    expected_count = len(ascend_classes) + sum(cls not in ascend_classes for cls in original_classes)
    with patch.object(vllm.lora.utils, "_all_lora_classes", original_classes):
        refresh_all_lora_classes()
        refresh_all_lora_classes()
        assert vllm.lora.utils._all_lora_classes[:2] == (
            AscendMergedColumnParallelLinearWithLoRA,
            AscendMergedQKVParallelLinearWithLoRA,
        )
        assert len(vllm.lora.utils._all_lora_classes) == expected_count


def test_packed_lora_a_weights_follow_set_and_reset_lifecycle() -> None:
    layer: Any = object.__new__(AscendMergedColumnParallelLinearWithLoRA)
    nn.Module.__init__(layer)
    layer.n_slices = 2
    layer.input_size = 4
    layer.device = torch.device("cpu")
    layer.lora_a_stacked = (
        torch.ones(1, 1, 2, 4),
        torch.full((1, 1, 2, 4), 2.0),
    )
    lora_config: Any = SimpleNamespace(lora_dtype=torch.float32)

    with patch.object(MergedColumnParallelLinearWithLoRA, "create_lora_weights"):
        layer.create_lora_weights(1, lora_config)

    assert layer.lora_a_packed.shape == (1, 1, 4, 4)
    with patch.object(MergedColumnParallelLinearWithLoRA, "set_lora"):
        layer.set_lora(0, [], [])
    torch.testing.assert_close(layer.lora_a_packed[0, 0, :2], layer.lora_a_stacked[0][0, 0])
    torch.testing.assert_close(layer.lora_a_packed[0, 0, 2:], layer.lora_a_stacked[1][0, 0])

    with patch.object(MergedColumnParallelLinearWithLoRA, "reset_lora"):
        layer.reset_lora(0)
    assert not torch.count_nonzero(layer.lora_a_packed)
