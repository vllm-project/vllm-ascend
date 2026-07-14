# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file

import vllm_ascend.models.deepseek_v4_dspark as dspark_module
from vllm_ascend.models.deepseek_v4_dspark import (
    DeepseekV4DSparkModel,
    DeepSeekV4DSparkMTP,
    _apply_dspark_quarot_rotation,
    _draft_quant_config,
    _get_dspark_num_mtp_layers,
    _load_dspark_quarot_rotation,
)


def test_dspark_quarot_load_and_apply(tmp_path):
    rotation_path = tmp_path / "optional" / "quarot.safetensors"
    rotation_path.parent.mkdir()
    rotation = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    save_file({"global_rotation": rotation}, rotation_path)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path)),
        quant_config=SimpleNamespace(
            quant_description={
                "optional": {"quarot": {"rotation_map": {"global_rotation": "optional/quarot.safetensors"}}}
            }
        ),
    )

    loaded = _load_dspark_quarot_rotation(vllm_config)

    torch.testing.assert_close(loaded, rotation)
    hidden_states = torch.tensor([[10.0, 20.0]])
    torch.testing.assert_close(
        _apply_dspark_quarot_rotation(hidden_states, loaded, transpose=False),
        torch.tensor([[70.0, 100.0]]),
    )
    torch.testing.assert_close(
        _apply_dspark_quarot_rotation(hidden_states, loaded, transpose=True),
        torch.tensor([[50.0, 110.0]]),
    )


def test_dspark_quarot_requires_same_device():
    with pytest.raises(RuntimeError, match="same device"):
        _apply_dspark_quarot_rotation(
            torch.ones(1, 2),
            torch.empty(2, 2, device="meta"),
            transpose=False,
        )


def test_dspark_draft_quant_config_and_layer_count():
    quant_config = object()
    draft_config = SimpleNamespace(dspark_mtp_dequantized_to_bf16=False)
    vllm_config = SimpleNamespace(
        quant_config=quant_config,
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=draft_config)),
    )

    assert _draft_quant_config(vllm_config) is quant_config
    draft_config.dspark_mtp_dequantized_to_bf16 = True
    assert _draft_quant_config(vllm_config) is None
    assert _get_dspark_num_mtp_layers(SimpleNamespace(n_mtp_layers=4)) == 4
    assert _get_dspark_num_mtp_layers(SimpleNamespace()) == 3


def test_dspark_context_kv_uses_each_layer_cache(monkeypatch):
    calls = []

    class FakeProjection:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, hidden_states):
            return hidden_states + self.offset

    def make_layer(name, offset):
        return SimpleNamespace(
            self_attn=SimpleNamespace(
                wkv=FakeProjection(offset),
                kv_norm=lambda value: value * 2,
                head_dim=4,
                nope_head_dim=2,
                rotary_emb=SimpleNamespace(layername=name),
                dsa_attn=SimpleNamespace(
                    swa_cache_layer=SimpleNamespace(prefix=f"{name}.swa_cache", kv_cache=object()),
                ),
            )
        )

    layers = {
        "43": make_layer("layer.43", 1),
        "44": make_layer("layer.44", 2),
    }
    model = SimpleNamespace(
        layers=layers,
        combine_hidden_states=lambda hidden_states: hidden_states * 3,
    )
    monkeypatch.setattr(
        dspark_module,
        "get_cos_and_sin_dsa",
        lambda _positions: (
            {name: torch.ones(3, 2) for name in ("layer.43", "layer.44")},
            {name: torch.zeros(3, 2) for name in ("layer.43", "layer.44")},
        ),
    )
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "inplace_partial_rotary_mul",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        dspark_module,
        "_scatter_context_kv",
        lambda cache, kv, slots: calls.append((cache, kv.clone(), slots)),
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)
    slot_mappings = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

    DeepseekV4DSparkModel.precompute_and_store_context_kv(model, hidden_states, torch.tensor([5, 6, 7]), slot_mappings)

    assert len(calls) == 2
    torch.testing.assert_close(calls[0][1].squeeze(1), (hidden_states * 3 + 1) * 2)
    torch.testing.assert_close(calls[1][1].squeeze(1), (hidden_states * 3 + 2) * 2)
    assert calls[0][2] is slot_mappings[0]
    assert calls[1][2] is slot_mappings[1]


def test_dspark_context_kv_rejects_wrong_slot_mapping_count():
    model = SimpleNamespace(layers={"43": object(), "44": object()})

    with pytest.raises(ValueError, match="number of draft layers"):
        DeepseekV4DSparkModel.precompute_and_store_context_kv(
            model,
            torch.ones(1, 4),
            torch.tensor([0]),
            [torch.tensor([0])],
        )


def test_dspark_context_kv_scatter_unwraps_cache(monkeypatch):
    from vllm_ascend.device.device_op import DeviceOperator

    calls = []
    formatted_slots = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    monkeypatch.setattr(
        DeviceOperator,
        "format_dsa_slot_mapping",
        lambda slots, block_size: formatted_slots if block_size == 4 else None,
    )
    monkeypatch.setattr(
        DeviceOperator,
        "dsa_kv_compress_scatter",
        lambda cache, kv, slots: calls.append((cache, kv, slots)),
    )
    cache = torch.empty(2, 4, 1, 3)
    kv = torch.empty(1)
    slots = torch.tensor([1, 6], dtype=torch.int64)

    dspark_module._scatter_context_kv([[cache]], kv, slots)

    assert calls == [(cache, kv, formatted_slots)]


def test_dspark_declares_target_shared_embedding_and_lm_head():
    assert DeepSeekV4DSparkMTP.has_own_embed_tokens is False
    assert DeepSeekV4DSparkMTP.has_own_lm_head is False


def test_dspark_load_weights_rejects_unmatched_mtp(monkeypatch):
    monkeypatch.setattr(dspark_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_module, "get_tensor_model_parallel_rank", lambda: 0)
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=43, num_attention_heads=8, expert_dtype="fp4"),
        model=SimpleNamespace(
            num_dspark_layers=3,
            get_expert_mapping=lambda: [],
            finalize_mega_moe_weights=lambda: None,
        ),
        named_parameters=lambda: iter(()),
    )

    with pytest.raises(ValueError, match="model.layers.43.self_attn.q_norm"):
        DeepSeekV4DSparkMTP.load_weights(cast(Any, model), [("mtp.0.attn.q_norm.weight", torch.ones(1))])


def test_dspark_load_weights_skips_nonlocal_expert(monkeypatch):
    monkeypatch.setattr(dspark_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_module, "get_tensor_model_parallel_rank", lambda: 0)

    class NonlocalExpertParam:
        def weight_loader(self, *args, **kwargs):
            assert kwargs["return_success"] is True
            return False

    expert_name = "model.layers.43.mlp.experts.weight"
    required_names = {
        "model.layers.43.main_proj.weight",
        "model.layers.43.main_norm.weight",
        "model.layers.44.norm.weight",
        "model.layers.45.norm.weight",
        "model.hc_head_fn",
        "model.hc_head_base",
        "model.hc_head_scale",
        "model.layers.45.markov_head.markov_w1.weight",
        "model.layers.45.markov_head.markov_w2.weight",
    }
    params: dict[str, object] = {name: SimpleNamespace() for name in required_names}
    params[expert_name] = NonlocalExpertParam()
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=43, num_attention_heads=8, expert_dtype="fp4"),
        model=SimpleNamespace(
            num_dspark_layers=3,
            get_expert_mapping=lambda: [
                (
                    "model.layers.43.mlp.experts",
                    "model.layers.43.mlp.experts.0.down_proj",
                    0,
                    "down_proj",
                )
            ],
            finalize_mega_moe_weights=lambda: None,
        ),
        named_parameters=lambda: iter(params.items()),
    )

    class RequiredParam:
        def weight_loader(self, *_args, **_kwargs):
            return None

    for name in required_names:
        params[name] = RequiredParam()
    weights = [
        ("mtp.0.main_proj.weight", torch.ones(1)),
        ("mtp.0.main_norm.weight", torch.ones(1)),
        ("mtp.0.ffn.experts.0.w2.weight", torch.ones(1)),
        ("mtp.1.norm.weight", torch.ones(1)),
        ("mtp.2.norm.weight", torch.ones(1)),
        ("mtp.2.hc_head_fn", torch.ones(1)),
        ("mtp.2.hc_head_base", torch.ones(1)),
        ("mtp.2.hc_head_scale", torch.ones(1)),
        ("mtp.2.markov_head.markov_w1.weight", torch.ones(1)),
        ("mtp.2.markov_head.markov_w2.weight", torch.ones(1)),
    ]

    loaded = DeepSeekV4DSparkMTP.load_weights(cast(Any, model), weights)

    assert expert_name not in loaded


def test_dspark_load_weights_stacks_shared_expert(monkeypatch):
    monkeypatch.setattr(dspark_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_module, "get_tensor_model_parallel_rank", lambda: 0)
    calls = []

    class FakeParam:
        def __init__(self, name):
            self.name = name

        def weight_loader(self, _param, weight, *args, **kwargs):
            calls.append((self.name, weight.clone(), args, kwargs))
            return True

    param_names = {
        "model.layers.43.main_proj.weight",
        "model.layers.43.main_norm.weight",
        "model.layers.44.mlp.shared_experts.gate_up_proj.weight",
        "model.layers.45.norm.weight",
        "model.hc_head_fn",
        "model.hc_head_base",
        "model.hc_head_scale",
        "model.layers.45.markov_head.markov_w1.weight",
        "model.layers.45.markov_head.markov_w2.weight",
    }
    params = {name: FakeParam(name) for name in param_names}
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=43, num_attention_heads=8, expert_dtype="fp4"),
        model=SimpleNamespace(
            num_dspark_layers=3,
            get_expert_mapping=lambda: [],
            finalize_mega_moe_weights=lambda: None,
        ),
        named_parameters=lambda: iter(params.items()),
    )
    weights = [
        ("mtp.0.main_proj.weight", torch.ones(1)),
        ("mtp.0.main_norm.weight", torch.ones(1)),
        ("mtp.1.ffn.shared_experts.w1.weight", torch.ones(1)),
        ("mtp.1.ffn.shared_experts.w3.weight", torch.ones(1) * 2),
        ("mtp.2.norm.weight", torch.ones(1)),
        ("mtp.2.hc_head_fn", torch.ones(1)),
        ("mtp.2.hc_head_base", torch.ones(1)),
        ("mtp.2.hc_head_scale", torch.ones(1)),
        ("mtp.2.markov_head.markov_w1.weight", torch.ones(1)),
        ("mtp.2.markov_head.markov_w2.weight", torch.ones(1)),
    ]

    loaded = DeepSeekV4DSparkMTP.load_weights(cast(Any, model), weights)

    shared_name = "model.layers.44.mlp.shared_experts.gate_up_proj.weight"
    assert shared_name in loaded
    shared_calls = [call for call in calls if call[0] == shared_name]
    assert [call[2] for call in shared_calls] == [(0,), (1,)]
