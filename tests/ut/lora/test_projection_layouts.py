# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration with installed vLLM; CPU UT mocks do not establish NPU support."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import vllm.lora.utils as lora_utils
from torch import nn
from vllm.config import LoRAConfig
from vllm.lora.layers import MergedColumnParallelLinearWithLoRA
from vllm.lora.utils import from_layer
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear

from vllm_ascend.lora.utils import (
    AscendColumnParallelLinearWithLoRA,
    AscendVariableSliceWithLoRA,
    refresh_all_lora_classes,
)


def test_single_projection_keeps_standard_column_wrapper():
    base = object.__new__(maybe_get_oot_by_class(ColumnParallelLinear))
    nn.Module.__init__(base)
    original = lora_utils._all_lora_classes
    try:
        refresh_all_lora_classes()
        config = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.float32)
        with (
            patch.object(AscendColumnParallelLinearWithLoRA, "__init__", return_value=None),
            patch.object(AscendColumnParallelLinearWithLoRA, "create_lora_weights"),
        ):
            wrapper = from_layer(base, 1, config, [])
        assert isinstance(wrapper, AscendColumnParallelLinearWithLoRA)
    finally:
        lora_utils._all_lora_classes = original


@pytest.mark.parametrize("tp_rank", [0, 1, 3])
def test_kda_replicated_b_factors_always_use_the_first_source_shard(tp_rank):
    def initialize(wrapper, base):
        nn.Module.__init__(wrapper)
        wrapper.tp_rank = tp_rank
        wrapper.n_slices = 6
        wrapper.output_slices = (2, 2, 2, 1, 4, 4)

    with patch(
        "vllm.lora.layers.column_parallel_linear.MergedColumnParallelLinearVariableSliceWithLoRA.__init__",
        initialize,
    ):
        wrapper = AscendVariableSliceWithLoRA(SimpleNamespace(replicated_shard_ids=(4, 5)))
    assert wrapper.output_ids == (tp_rank, tp_rank, tp_rank, tp_rank, 0, 0)
    factors = [torch.arange(size * (1 if i >= 4 else 4)).reshape(-1, 1) for i, size in enumerate(wrapper.output_slices)]
    actual = wrapper.slice_lora_b(factors)
    for i, (size, values) in enumerate(zip(wrapper.output_slices, actual)):
        start = 0 if i >= 4 else tp_rank * size
        torch.testing.assert_close(values, factors[i][start : start + size])


def test_multiple_adapter_merged_layer_uses_upstream_wrapper():
    base = object.__new__(maybe_get_oot_by_class(MergedColumnParallelLinear))
    nn.Module.__init__(base)
    base.tp_size = 1
    original = lora_utils._all_lora_classes
    try:
        refresh_all_lora_classes()
        config = LoRAConfig(max_loras=2, max_lora_rank=8, lora_dtype=torch.float32)
        with (
            patch.object(MergedColumnParallelLinearWithLoRA, "__init__", return_value=None),
            patch.object(MergedColumnParallelLinearWithLoRA, "create_lora_weights"),
        ):
            wrapper = from_layer(base, 2, config, ["gate_proj", "up_proj"])
        assert type(wrapper) is MergedColumnParallelLinearWithLoRA
    finally:
        lora_utils._all_lora_classes = original


def test_upstream_merged_weights_preserve_other_slots_and_missing_slices():
    projection = object.__new__(MergedColumnParallelLinearWithLoRA)
    nn.Module.__init__(projection)
    projection.n_slices, projection.tp_size = 2, 1
    projection.lora_a_stacked = (torch.zeros(2, 1, 2, 6), torch.zeros(2, 1, 2, 6))
    projection.lora_b_stacked = (torch.zeros(2, 1, 4, 2), torch.zeros(2, 1, 3, 2))
    projection.set_lora(0, [torch.ones(2, 6), None], [torch.ones(4, 2), None])
    projection.set_lora(1, [None, torch.full((2, 6), 0.5)], [None, torch.full((3, 2), 0.25)])
    inputs = torch.arange(1.0, 7.0).reshape(1, 6)

    def delta(slot):
        return torch.cat(
            [
                inputs @ a[slot, 0].T @ b[slot, 0].T
                for a, b in zip(projection.lora_a_stacked, projection.lora_b_stacked)
            ],
            dim=-1,
        )

    expected_a = torch.tensor([[42.0, 42.0, 42.0, 42.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(delta(0), expected_a)
    torch.testing.assert_close(delta(1), torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.25, 5.25, 5.25]]))
    projection.set_lora(1, [None, None], [None, None])
    torch.testing.assert_close(delta(1), torch.zeros(1, 7))
    torch.testing.assert_close(delta(0), expected_a)
    projection.reset_lora(0)
    torch.testing.assert_close(delta(0), torch.zeros(1, 7))
