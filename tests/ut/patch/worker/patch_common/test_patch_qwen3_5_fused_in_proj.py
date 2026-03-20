import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.models.qwen3_5 as qwen3_5_module


def _reload_patch_qwen3_5(monkeypatch: pytest.MonkeyPatch, *, enable_fused: bool):
    monkeypatch.setenv(
        "VLLM_ASCEND_ENABLE_QWEN35_FUSED_IN_PROJ",
        "1" if enable_fused else "0",
    )
    monkeypatch.setenv("VLLM_ASCEND_DEBUG_QWEN35_FUSED_IN_PROJ", "0")
    import vllm_ascend.patch.worker.patch_qwen3_5 as patch_qwen3_5

    return importlib.reload(patch_qwen3_5)


def _make_dummy_fused_layer():
    hidden_size = 5
    key_dim = 4
    num_v_heads = 6
    tp_size = 2
    head_v_dim = 2
    value_dim = num_v_heads * head_v_dim
    output_size = key_dim * 2 // tp_size + value_dim * 2 // tp_size + num_v_heads * 2 // tp_size
    weight = torch.arange(output_size * hidden_size, dtype=torch.float32).reshape(output_size, hidden_size)
    return SimpleNamespace(
        key_dim=key_dim,
        value_dim=value_dim,
        num_v_heads=num_v_heads,
        tp_size=tp_size,
        head_v_dim=head_v_dim,
        prefix="model.layers.0.linear_attn",
        in_proj=SimpleNamespace(weight=torch.nn.Parameter(weight)),
    )


def test_split_fused_in_proj_outputs_matches_legacy_projection(monkeypatch: pytest.MonkeyPatch):
    patch_qwen3_5 = _reload_patch_qwen3_5(monkeypatch, enable_fused=True)
    layer = _make_dummy_fused_layer()
    hidden_states = torch.randn(3, 5, dtype=torch.float32)
    projected_states = F.linear(hidden_states, layer.in_proj.weight)

    mixed_qkv, z, b, a = patch_qwen3_5._split_qwen35_fused_in_proj_outputs(layer, projected_states)
    expected_mixed_qkv, expected_z, expected_b, expected_a = patch_qwen3_5._compute_qwen35_legacy_in_proj_outputs(
        layer, hidden_states
    )

    assert torch.allclose(mixed_qkv, expected_mixed_qkv)
    assert torch.allclose(z, expected_z)
    assert torch.allclose(b, expected_b)
    assert torch.allclose(a, expected_a)


def test_qwen35_load_weights_maps_legacy_gdn_shards_to_fused_in_proj(monkeypatch: pytest.MonkeyPatch):
    _reload_patch_qwen3_5(monkeypatch, enable_fused=True)

    param = torch.nn.Parameter(torch.zeros(16, 5))
    loader_calls: list[tuple[tuple[int, ...] | int, torch.Tensor]] = []

    def weight_loader(param, loaded_weight, shard_id):
        loader_calls.append((shard_id, loaded_weight))

    setattr(param, "weight_loader", weight_loader)

    fake_model = SimpleNamespace(
        config=SimpleNamespace(),
        named_parameters=lambda: [("layers.0.linear_attn.in_proj.weight", param)],
        get_expert_mapping=lambda: [],
        load_fused_expert_weights=lambda *args, **kwargs: False,
    )
    weights = [
        ("layers.0.linear_attn.in_proj_qkv.weight", torch.randn(7, 5)),
        ("layers.0.linear_attn.in_proj_z.weight", torch.randn(3, 5)),
        ("layers.0.linear_attn.in_proj_b.weight", torch.randn(3, 5)),
        ("layers.0.linear_attn.in_proj_a.weight", torch.randn(3, 5)),
    ]

    with patch("vllm_ascend.patch.worker.patch_qwen3_5.is_pp_missing_parameter", return_value=False):
        loaded_params = qwen3_5_module.Qwen3_5Model.load_weights(fake_model, weights)

    assert [call[0] for call in loader_calls] == [(0, 1, 2), 3, 4, 5]
    assert loaded_params == {
        "layers.0.linear_attn.in_proj.weight",
    }


def test_qwen35_packed_modules_mapping_switches_to_fused_in_proj(monkeypatch: pytest.MonkeyPatch):
    _reload_patch_qwen3_5(monkeypatch, enable_fused=True)

    mapping = qwen3_5_module.Qwen3_5ForCausalLMBase.packed_modules_mapping
    assert mapping["in_proj"] == [
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
    ]
    assert "in_proj_qkvz" not in mapping
    assert "in_proj_ba" not in mapping
