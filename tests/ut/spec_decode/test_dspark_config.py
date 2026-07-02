# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path
from types import SimpleNamespace

import torch

from transformers import PretrainedConfig

import vllm_ascend.models.deepseek_v4_dspark as dspark_model_module
from vllm_ascend.models.deepseek_v4_dspark import (
    DeepSeekV4DSparkMTP,
    DeepseekV4DSparkAttention,
    DeepseekV4DSparkModel,
    _get_dspark_num_mtp_layers,
)


def test_dspark_deepseek_v4_hf_config_override():
    repo_root = Path(__file__).parents[3]
    patch_module = runpy.run_path(str(repo_root / "vllm_ascend/patch/platform/patch_speculative_config.py"))

    hf_config = PretrainedConfig(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=[40, 41, 42],
    )

    patched = patch_module["hf_config_override"](hf_config)

    assert patched.model_type == "deepseek_mtp"
    assert patched.architectures == ["DeepSeekV4DSparkMTPModel"]
    assert patched.n_predict == 5
    assert patched.ptd_token_id == 128799


def test_dspark_num_mtp_layers_prefers_upstream_config_name():
    config = SimpleNamespace(n_mtp_layers=4, dspark_num_mtp_layers=2)

    assert _get_dspark_num_mtp_layers(config) == 4


def test_dspark_num_mtp_layers_keeps_legacy_config_fallback():
    config = SimpleNamespace(dspark_num_mtp_layers=2)

    assert _get_dspark_num_mtp_layers(config) == 2


def test_dspark_remap_skips_unused_confidence_head_weights():
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=61))

    assert (
        DeepSeekV4DSparkMTP._remap_dspark_name(
            model,
            "mtp.2.confidence_head.proj.weight",
        )
        is None
    )


def test_dspark_exposes_draft_kv_cache_layer_names():
    def make_layer(prefix: str) -> SimpleNamespace:
        return SimpleNamespace(
            self_attn=SimpleNamespace(
                dsa_attn=SimpleNamespace(
                    swa_cache_layer=SimpleNamespace(prefix=prefix),
                ),
            ),
        )

    model = SimpleNamespace(
        layers={
            "61": make_layer("model.layers.61.self_attn.swa_cache"),
            "62": make_layer("model.layers.62.self_attn.swa_cache"),
        }
    )
    model.get_draft_kv_cache_layer_names = DeepseekV4DSparkModel.get_draft_kv_cache_layer_names.__get__(model)
    wrapper = SimpleNamespace(model=model)

    expected = [
        "model.layers.61.self_attn.swa_cache",
        "model.layers.62.self_attn.swa_cache",
    ]
    assert DeepseekV4DSparkModel.get_draft_kv_cache_layer_names(model) == expected
    assert DeepSeekV4DSparkMTP.get_draft_kv_cache_layer_names(wrapper) == expected


def test_dspark_precompute_context_kv_passes_layer_slot_mappings(monkeypatch):
    calls = []

    def make_layer(name: str) -> SimpleNamespace:
        def precompute_context_kv(main_x, positions, request_slots=None, context_slot_mapping=None):
            calls.append((name, main_x, positions, request_slots, context_slot_mapping))

        return SimpleNamespace(self_attn=SimpleNamespace(precompute_context_kv=precompute_context_kv))

    monkeypatch.setattr(dspark_model_module, "_linear_output", lambda _proj, hidden_states: hidden_states + 1)
    context_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    positions = torch.tensor([4, 5, 6], dtype=torch.int32)
    request_slots = torch.tensor([1, 1, 1], dtype=torch.int32)
    layer_slot_mappings = [
        torch.tensor([10, 11, 12], dtype=torch.int32),
        torch.tensor([20, 21, 22], dtype=torch.int32),
    ]
    model = SimpleNamespace(
        main_proj=object(),
        main_norm=lambda tensor: tensor * 2,
        layers={
            "61": make_layer("61"),
            "62": make_layer("62"),
        },
    )

    DeepseekV4DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        positions,
        context_slot_mapping=layer_slot_mappings,
        context_request_slots=request_slots,
    )

    assert [call[0] for call in calls] == ["61", "62"]
    for idx, call in enumerate(calls):
        _, main_x, call_positions, call_request_slots, call_slot_mapping = call
        torch.testing.assert_close(main_x, (context_states + 1) * 2)
        assert call_positions is positions
        assert call_request_slots is request_slots
        assert call_slot_mapping is layer_slot_mappings[idx]


def test_dspark_forward_passes_query_slot_mapping_to_layers():
    calls = []

    class FakeLayer:
        def __call__(
            self,
            *,
            positions,
            hidden_states,
            input_ids,
            request_slots=None,
            slot_mapping=None,
        ):
            calls.append((positions, hidden_states, input_ids, request_slots, slot_mapping))
            return hidden_states + 1

    input_ids = torch.tensor([1, 2], dtype=torch.int64)
    positions = torch.tensor([10, 11], dtype=torch.int32)
    inputs_embeds = torch.ones(2, 3)
    request_slots = torch.tensor([4, 4], dtype=torch.int32)
    slot_mapping = torch.tensor([80, 81], dtype=torch.int32)
    model = SimpleNamespace(
        embed_tokens=None,
        hc_mult=2,
        layers={
            "61": FakeLayer(),
            "62": FakeLayer(),
        },
        compute_head_hidden=lambda hidden_states: hidden_states,
    )

    output = DeepseekV4DSparkModel.forward(
        model,
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
        request_slots=request_slots,
        slot_mapping=slot_mapping,
    )

    assert len(calls) == 2
    for call in calls:
        call_positions, _, call_input_ids, call_request_slots, call_slot_mapping = call
        assert call_positions is positions
        assert call_input_ids is input_ids
        assert call_request_slots is request_slots
        assert call_slot_mapping is slot_mapping
    torch.testing.assert_close(output, inputs_embeds.unsqueeze(-2).repeat(1, 2, 1) + 2)


def test_dspark_store_standard_swa_kv_uses_dsa_slot_mapping(monkeypatch):
    from vllm_ascend.device import device_op as device_op_module

    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_STANDARD_DSA", "1")
    calls = []

    def fake_format(slot_mapping, block_size):
        calls.append(("format", slot_mapping.clone(), block_size))
        return torch.stack([slot_mapping // block_size, slot_mapping % block_size], dim=-1)

    def fake_scatter(cache, shared_kv, slot_mapping):
        calls.append(("scatter", cache, shared_kv.clone(), slot_mapping.clone()))

    monkeypatch.setattr(device_op_module.DeviceOperator, "format_dsa_slot_mapping", staticmethod(fake_format))
    monkeypatch.setattr(device_op_module.DeviceOperator, "dsa_kv_compress_scatter", staticmethod(fake_scatter))
    cache = torch.zeros(4, 8, 1, 3)
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(
                kv_cache=cache,
                block_size=8,
            )
        )
    )
    shared_kv = torch.arange(6, dtype=torch.float32).view(2, 1, 3)
    slot_mapping = torch.tensor([9, 18], dtype=torch.int64)

    DeepseekV4DSparkAttention._store_standard_swa_kv(attn, shared_kv, slot_mapping)

    assert calls[0][0] == "format"
    torch.testing.assert_close(calls[0][1], torch.tensor([9, 18], dtype=torch.int32))
    assert calls[0][2] == 8
    assert calls[1][0] == "scatter"
    assert calls[1][1] is cache
    torch.testing.assert_close(calls[1][2], shared_kv)
    torch.testing.assert_close(calls[1][3], torch.tensor([[1, 1], [2, 2]], dtype=torch.int32))
