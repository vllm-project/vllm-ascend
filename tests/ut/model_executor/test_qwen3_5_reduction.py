# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ascend_config import ExpertReductionConfig
from vllm_ascend.models.qwen3_5 import AscendQwen3_5MoeForCausalLM, AscendQwen3_5MoeMTP
from vllm_ascend.models.qwen3_5_reduction import (
    Qwen3_5ExpertReductionMixin,
    configure_expert_reduction,
    reduce_expert_weights,
)
from vllm_ascend.models.qwen3_5_vl import AscendQwen3_5MoeForConditionalGeneration


def make_config(**parallel):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text", num_experts=16, num_experts_per_tok=2)
        ),
        parallel_config=SimpleNamespace(
            **dict(
                dict(enable_eplb=False, enable_expert_parallel=False, tensor_parallel_size=1, data_parallel_size=1),
                **parallel,
            )
        ),
    )


@pytest.mark.parametrize("count", [-1, 1.5, True, "10"])
def test_config_rejects_invalid_types_and_counts(count):
    with pytest.raises(ValueError):
        ExpertReductionConfig(num_experts=count)


@pytest.mark.parametrize("count", [0, 2, 16])
def test_count_boundaries_and_repeat_construction(count):
    config = make_config()
    first = configure_expert_reduction(config, count)
    assert configure_expert_reduction(config, count) == first
    assert config.model_config.hf_text_config.num_experts == (count or 16)


@pytest.mark.parametrize(
    ("count", "parallel"),
    [(1, {}), (17, {}), (2, {"enable_eplb": True}), (3, {"enable_expert_parallel": True, "tensor_parallel_size": 2})],
)
def test_unsupported_reduction_fails_before_mutating_config(count, parallel):
    config = make_config(**parallel)
    with pytest.raises(ValueError):
        configure_expert_reduction(config, count)
    assert config.model_config.hf_text_config.num_experts == 16


@pytest.mark.parametrize("prefix", ["model.language_model.layers.0.", "mtp.layers.0."])
def test_filter_keeps_retained_scales_shared_experts_and_router_rows(prefix):
    router = torch.arange(48).reshape(16, 3)
    retained = torch.ones(3)
    shared = torch.ones(2)
    inputs = [
        (prefix + "mlp.gate.weight", router),
        (prefix + "mlp.gate.bias", torch.arange(16)),
        (prefix + "mlp.experts.1.gate_proj.weight_scale", retained),
        (prefix + "mlp.experts.2.down_proj.weight", torch.ones(3)),
        (prefix + "mlp.shared_expert.gate_proj.weight", shared),
    ]
    result = dict(reduce_expert_weights(iter(inputs), (16, 2)))
    assert len(result) == 4
    assert torch.equal(result[inputs[0][0]], router[:2])
    assert torch.equal(result[inputs[1][0]], torch.arange(2))
    assert result[inputs[2][0]] is retained
    assert result[inputs[4][0]] is shared
    assert dict(reduce_expert_weights(inputs, None))[inputs[0][0]] is router


@pytest.mark.parametrize(
    ("name", "weight"),
    [
        ("model.layers.0.mlp.gate.weight", torch.ones(2, 3)),
        ("model.layers.0.mlp.experts.gate_up_proj", torch.ones(16, 2, 3)),
    ],
)
def test_mismatched_router_and_packed_checkpoint_are_rejected(name, weight):
    with pytest.raises(ValueError):
        list(reduce_expert_weights([(name, weight)], (16, 2)))


def test_adapter_reduces_before_allocation_and_before_upstream_loader():
    # The base sees the reduced config and unmodified original checkpoint names.
    class Base:
        def __init__(self, *, vllm_config, prefix):
            self.allocated_experts = vllm_config.model_config.hf_text_config.num_experts
            self.prefix = prefix

        def load_weights(self, weights):
            self.loaded = dict(weights)
            return set(self.loaded)

    class Adapter(Qwen3_5ExpertReductionMixin, Base):
        pass

    ascend = SimpleNamespace(expert_reduction=ExpertReductionConfig(num_experts=2))
    with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=ascend):
        model = Adapter(vllm_config=make_config(), prefix="model")
    assert model.allocated_experts == 2
    name = "model.layers.0.mlp.gate.weight"
    assert model.load_weights([(name, torch.ones(16, 3))]) == {name}
    assert model.loaded[name].shape == (2, 3)


@pytest.mark.parametrize(
    "model_cls", [AscendQwen3_5MoeForCausalLM, AscendQwen3_5MoeMTP, AscendQwen3_5MoeForConditionalGeneration]
)
def test_registered_adapter_uses_upstream_loader_with_original_checkpoint(model_cls):
    # Exercise the real upstream loader with tiny CPU parameters, without allocating a full model.
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model._ascend_expert_reduction = (16, 2)
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.gate = torch.nn.Linear(3, 2, bias=False)
    layer.mlp.experts = torch.nn.ModuleList([torch.nn.Linear(3, 3, bias=False) for _ in range(2)])
    model.model.layers = torch.nn.ModuleList([layer])
    prefix = "mtp" if model_cls is AscendQwen3_5MoeMTP else "model"
    if model_cls is AscendQwen3_5MoeForConditionalGeneration:
        model.language_model = torch.nn.Module()
        model.language_model.model = model.model
        del model.model
        prefix = "model.language_model"
    router = torch.arange(48, dtype=torch.float32).reshape(16, 3)
    expert = torch.full((3, 3), 7.0)
    loaded = model.load_weights(
        [
            (f"{prefix}.layers.0.mlp.gate.weight", router),
            (f"{prefix}.layers.0.mlp.experts.1.weight", expert),
            (f"{prefix}.layers.0.mlp.experts.15.weight", torch.zeros(3, 3)),
        ]
    )
    assert len(loaded) == 2
    torch.testing.assert_close(layer.mlp.gate.weight, router[:2])
    torch.testing.assert_close(layer.mlp.experts[1].weight, expert)
