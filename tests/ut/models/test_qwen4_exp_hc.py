import torch

from vllm_ascend import models
from vllm_ascend.models.qwen4_exp.config import Qwen4ExpTextConfig
from vllm_ascend.ops.triton.qwen4_exp.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)


def test_qwen4_exp_architectures_are_registered(monkeypatch) -> None:
    registered: dict[str, str] = {}

    monkeypatch.setattr(
        models.ModelRegistry,
        "register_model",
        lambda architecture, target: registered.__setitem__(architecture, target),
    )

    models.register_model()

    assert registered["Qwen4ExpForCausalLM"] == ("vllm_ascend.models.qwen4_exp:AscendQwen4ExpForCausalLM")
    assert registered["Qwen4ExpForConditionalGeneration"] == (
        "vllm_ascend.models.qwen4_exp:AscendQwen4ExpForConditionalGeneration"
    )
    assert registered["Qwen4ExpMTP"] == ("vllm_ascend.models.qwen4_exp:AscendQwen4ExpMTP")


def test_qwen4_exp_config_is_owned_by_plugin() -> None:
    config = Qwen4ExpTextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
        ple_layer_ids=[1],
        ple_embed_dim=64,
        ngram_size=3,
        heads_per_ngram=4,
        layer_types=["full_attention", "linear_attention"],
    )

    assert config.model_type == "qwen4_exp_text"
    assert config.short_conv_layer_ids == [0]
    assert config.ngram_context_len == 2


def test_hyperconnection_ops_match_torch_reference() -> None:
    torch.manual_seed(0)
    hc_count = 4
    group_dim = 8
    residual = torch.randn(3, hc_count * group_dim, dtype=torch.bfloat16)
    gate = torch.randn_like(residual)
    block_output = torch.randn(3, group_dim, dtype=torch.bfloat16)
    injection = torch.randn(3, hc_count, dtype=torch.bfloat16)
    weight = torch.randn(hc_count * group_dim, dtype=torch.bfloat16)
    eps = 1e-6

    actual_silu = hc_silu(gate, hc_count)
    expected_silu = torch.nn.functional.silu(gate.float() / hc_count).to(gate.dtype)
    torch.testing.assert_close(actual_silu, expected_silu)

    actual_mix = hc_gate_mix(residual, gate, hc_count)
    expected_mix = (
        (torch.sigmoid(gate.float()) * residual.float()).view(3, hc_count, group_dim).mean(1).to(residual.dtype)
    )
    torch.testing.assert_close(actual_mix, expected_mix)

    actual_combined = hc_combine(residual, block_output, injection, hc_count)
    expected_injection = 2 * torch.sigmoid(injection.float() / hc_count)
    expected_combined = residual.float().view(3, hc_count, group_dim)
    expected_combined = expected_combined + (block_output.float().unsqueeze(1) * expected_injection.unsqueeze(-1))
    expected_combined = expected_combined.to(residual.dtype).view_as(residual)
    torch.testing.assert_close(actual_combined, expected_combined)

    actual_norm = grouped_gemma_rmsnorm(actual_combined, weight, eps, hc_count)
    grouped = actual_combined.float().view(3, hc_count, group_dim)
    expected_norm = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    expected_norm = (
        (expected_norm * (1 + weight.float().view(1, hc_count, group_dim))).to(residual.dtype).view_as(residual)
    )
    torch.testing.assert_close(actual_norm, expected_norm)

    combined, normalized = hc_combine_norm(
        residual,
        block_output,
        injection,
        weight,
        eps,
        hc_count,
    )
    torch.testing.assert_close(combined, actual_combined)
    torch.testing.assert_close(normalized, actual_norm)


def test_grouped_norm_accepts_shared_affine() -> None:
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    shared_weight = torch.randn(4, dtype=torch.bfloat16)

    result = grouped_gemma_rmsnorm(x, shared_weight, 1e-6, 4)

    assert result.shape == x.shape
    assert result.dtype == x.dtype
