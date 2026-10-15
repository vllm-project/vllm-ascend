# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the example's trainer-storage to checkpoint-coordinate conversion."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("fused", [False, True])
def test_example_patches_both_pipeline_ends_and_experts(fused):
    path = Path(__file__).resolve().parents[4] / "examples/rl/rlhf_sparse_hccl.py"
    spec = importlib.util.spec_from_file_location("sparse_hccl_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        model_type="qwen3_moe", num_experts=4, moe_intermediate_size=4, hidden_size=3, num_hidden_layers=4
    )
    weights = {}
    for layer in (0, 3):
        prefix = f"model.layers.{layer}.mlp.experts"
        if fused:
            weights[f"{prefix}.gate_up_proj"] = torch.arange(4 * 8 * 3, dtype=torch.float32).reshape(4, 8, 3)
        else:
            for expert in (0, 2):
                weights[f"{prefix}.{expert}.gate_proj.weight"] = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    def get_parameter(name):
        if name not in weights:
            raise AttributeError(name)
        return weights[name]

    model.get_parameter = get_parameter
    before = {name: value.clone() for name, value in weights.items()}
    patches = module.make_patches(model)
    assert len(patches) == 4
    for patch in patches:
        assert patch.full_shape == (4, 3)
        assert patch.indices.dtype == torch.int32
        assert torch.equal(patch.indices, torch.arange(6, dtype=torch.int32))
        prefix = patch.name.rsplit(".", 3)[0]
        expert = int(patch.name.split(".")[-3])
        if fused:
            reference = before[f"{prefix}.gate_up_proj"][expert, :4].clone()
            actual = weights[f"{prefix}.gate_up_proj"][expert, :4]
            assert torch.equal(
                weights[f"{prefix}.gate_up_proj"][expert, 4:], before[f"{prefix}.gate_up_proj"][expert, 4:]
            )
        else:
            reference = before[patch.name].clone()
            actual = weights[patch.name]
        reference.flatten().index_copy_(0, patch.indices.long(), patch.values)
        assert torch.equal(actual, reference)
        assert torch.equal(
            patch.values,
            before[f"{prefix}.gate_up_proj"][expert, :2].flip(0).flatten()
            if fused
            else before[patch.name][:2].flip(0).flatten(),
        )
