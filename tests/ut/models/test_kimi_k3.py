# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_ascend.models import kimi_k3
from vllm_ascend.models.kimi_k3 import (
    AscendKimiK3ForCausalLM,
    AscendKimiK3ForConditionalGeneration,
    KimiK3MLP,
    KimiK3MoE,
    KimiK3TextModel,
    KimiK3VisionEncoderLayer,
    _move_module_to_device,
    _resolve_packed_expert_weight_name,
    _routed_latent_quant_config,
)
from vllm_ascend.ops.activation import AscendSituAndMul, SituActivationConfig
from vllm_ascend.transformers_utils.configs.kimi_k3 import (
    KimiK3Config,
    KimiK3VisionConfig,
)


def test_kimi_k3_model_declares_checkpoint_packing_contract():
    assert AscendKimiK3ForCausalLM.packed_modules_mapping["experts"] == [
        "experts.0.w1",
        "experts.0.w3",
        "experts.0.w2",
    ]


@pytest.mark.parametrize(
    ("quant_name", "uses_quantized_latent_projections"),
    [
        ("ascend", True),
        ("compressed-tensors", False),
        ("other", False),
    ],
)
def test_kimi_k3_quantizes_latent_projections_only_for_modelslim(
    quant_name: str,
    uses_quantized_latent_projections: bool,
):
    quant_config = MagicMock()
    quant_config.get_name.return_value = quant_name

    actual = _routed_latent_quant_config(quant_config)

    if uses_quantized_latent_projections:
        assert actual is quant_config
    else:
        assert actual is None


def test_kimi_k3_unquantized_model_keeps_latent_projections_unquantized():
    assert _routed_latent_quant_config(None) is None


@pytest.mark.parametrize(
    ("name", "params", "expected"),
    [
        (
            "layers.1.experts.w13_weight",
            {"layers.1.experts.w13_weight": object()},
            "layers.1.experts.w13_weight",
        ),
        (
            "layers.1.experts.w13_weight",
            {"layers.1.experts.w13_weight_packed": object()},
            "layers.1.experts.w13_weight_packed",
        ),
        (
            "layers.1.experts.w2_weight",
            {"layers.1.experts.w2_weight_packed": object()},
            "layers.1.experts.w2_weight_packed",
        ),
        (
            "layers.1.experts.w13_weight_scale",
            {"layers.1.experts.w13_weight_packed": object()},
            "layers.1.experts.w13_weight_scale",
        ),
    ],
)
def test_kimi_k3_resolves_packed_expert_checkpoint_names(
    name: str,
    params: dict[str, object],
    expected: str,
):
    assert _resolve_packed_expert_weight_name(name, params) == expected


def test_kimi_k3_vit_dp_compat_calls_release_helper_without_num_heads(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[None] = []

    def release_helper():
        calls.append(None)
        return False

    monkeypatch.setattr(kimi_k3, "vllm_version_is", lambda version: version == "0.25.1")
    monkeypatch.setattr(kimi_k3, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(kimi_k3, "is_vit_use_data_parallel", release_helper)

    assert kimi_k3._is_vit_use_data_parallel(8) is False
    assert calls == [None]


def test_kimi_k3_vit_dp_compat_recreates_release_tp_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    def unexpected_release_helper():
        pytest.fail("The release helper must not run after the TP fallback")

    monkeypatch.setattr(kimi_k3, "vllm_version_is", lambda version: version == "0.25.1")
    monkeypatch.setattr(kimi_k3, "get_tensor_model_parallel_world_size", lambda: 16)
    monkeypatch.setattr(kimi_k3, "is_vit_use_data_parallel", unexpected_release_helper)

    assert kimi_k3._is_vit_use_data_parallel(12) is True


def test_kimi_k3_vit_dp_compat_passes_num_heads_to_main_helper(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def main_helper(num_heads):
        calls.append(num_heads)
        return True

    monkeypatch.setattr(kimi_k3, "vllm_version_is", lambda version: False)
    monkeypatch.setattr(kimi_k3, "is_vit_use_data_parallel", main_helper)

    assert kimi_k3._is_vit_use_data_parallel(12) is True
    assert calls == [12]


def test_kimi_k3_skips_explicit_move_for_meta_modules():
    module = nn.Linear(4, 4, device="meta")

    actual = _move_module_to_device(
        module,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert actual is module
    assert all(parameter.is_meta for parameter in module.parameters())


def test_kimi_k3_moves_non_meta_modules():
    module = nn.Linear(4, 4)

    actual = _move_module_to_device(
        module,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert actual is module
    assert all(parameter.device.type == "cpu" for parameter in module.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in module.parameters())


def test_kimi_k3_passes_situ_parameters_through_activation_config(monkeypatch):
    class StubModule(nn.Module):
        pass

    fused_moe_kwargs = {}

    def fake_replicated_linear(*args, **kwargs):
        return StubModule()

    def fake_fused_moe(**kwargs):
        fused_moe_kwargs.update(kwargs)
        return StubModule()

    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", fake_replicated_linear)
    monkeypatch.setattr(kimi_k3, "FusedMoE", fake_fused_moe)
    config = SimpleNamespace(
        hidden_act="situ",
        hidden_size=32,
        routed_expert_hidden_size=16,
        num_shared_experts=0,
        num_experts=8,
        rms_norm_eps=1e-6,
        latent_moe_use_norm=False,
        moe_intermediate_size=12,
        num_experts_per_token=2,
        moe_renormalize=True,
        use_grouped_topk=True,
        num_expert_group=4,
        topk_group=2,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=2.5,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    KimiK3MoE(config, prefix="model.layers.1.block_sparse_moe")

    activation = fused_moe_kwargs["activation"]
    assert isinstance(activation, SituActivationConfig)
    assert activation.beta == 4.0
    assert activation.linear_beta == 25.0


def test_kimi_k3_dense_mlp_uses_callable_situ(monkeypatch):
    class StubLinear(nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    monkeypatch.setattr(kimi_k3, "MergedColumnParallelLinear", lambda *args, **kwargs: StubLinear())
    monkeypatch.setattr(kimi_k3, "RowParallelLinear", lambda *args, **kwargs: StubLinear())
    config = SimpleNamespace(
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    mlp = KimiK3MLP(config, hidden_size=4, intermediate_size=2)
    hidden_states = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    output = mlp(hidden_states)

    assert isinstance(mlp.act_fn, AscendSituAndMul)
    assert output.shape == (1, 2)
