# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

import vllm_ascend.models.deepseek_v4_dspark as dspark_model_module
from vllm_ascend.models.deepseek_v4 import _is_dspark_target_layer
from vllm_ascend.models.deepseek_v4_dspark import (
    DeepseekV4DSparkAttention,
    DeepseekV4DSparkDecoderLayer,
    DeepseekV4DSparkModel,
    DeepSeekV4DSparkMTP,
    _get_dspark_num_mtp_layers,
    _maybe_fp8_e4m3fn_qdq,
    _maybe_fp8_qdq_nope_dims,
    _should_apply_dspark_fp8_qdq,
)
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


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


def test_dspark_fp8_qdq_is_disabled_for_bf16_dequantized_checkpoints():
    assert _should_apply_dspark_fp8_qdq(SimpleNamespace(dspark_mtp_dequantized_to_bf16=True)) is False
    assert _should_apply_dspark_fp8_qdq(SimpleNamespace(dspark_full_dequantized_to_bf16=True)) is False
    assert _should_apply_dspark_fp8_qdq(SimpleNamespace()) is True


def test_dspark_fp8_qdq_helpers_return_input_when_disabled():
    kv = torch.randn(2, 128)
    out = torch.randn(2, 2, 128)

    assert _maybe_fp8_qdq_nope_dims(kv, nope_head_dim=64, apply_fp8_qdq=False) is kv
    assert _maybe_fp8_e4m3fn_qdq(out, apply_fp8_qdq=False, block_size=128) is out


def test_dspark_markov_head_w2_uses_model_default_dtype(monkeypatch):
    captured = {}

    class FakeEmbedding(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class FakeLMHead(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dspark_model_module, "VocabParallelEmbedding", FakeEmbedding)
    monkeypatch.setattr(dspark_model_module, "ParallelLMHead", FakeLMHead)
    monkeypatch.setattr(dspark_model_module, "LogitsProcessor", lambda vocab_size: object())

    dspark_model_module.DSparkMarkovHead(
        SimpleNamespace(vocab_size=16, dspark_markov_rank=4),
        prefix="model.layers.45.markov_head",
    )

    assert "params_dtype" not in captured["kwargs"]
    assert captured["kwargs"]["org_num_embeddings"] == 16


def test_dspark_attention_uses_upstream_no_compression_ratio(monkeypatch):
    def fake_base_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.dsa_attn = SimpleNamespace(compress_ratio=0)
        self.window_size = 8
        self.n_local_heads = 2
        self.head_dim = 4

    monkeypatch.setattr(dspark_model_module.DeepseekV4Attention, "__init__", fake_base_init)
    monkeypatch.setattr(dspark_model_module, "current_platform", SimpleNamespace(device_type="cpu"))

    attn = DeepseekV4DSparkAttention(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16, max_model_len=16),
            scheduler_config=SimpleNamespace(max_num_seqs=2),
        ),
        config=SimpleNamespace(
            dspark_block_size=5,
            dspark_mtp_dequantized_to_bf16=True,
        ),
    )

    assert attn.compress_ratio == 1
    assert attn.dsa_attn.compress_ratio == 1


def test_dspark_target_layer_ids_follow_upstream_one_based_numbering():
    target_layer_ids = {40, 41, 42}

    assert not _is_dspark_target_layer(38, target_layer_ids)
    assert _is_dspark_target_layer(39, target_layer_ids)
    assert _is_dspark_target_layer(40, target_layer_ids)
    assert _is_dspark_target_layer(41, target_layer_ids)
    assert not _is_dspark_target_layer(42, target_layer_ids)


def test_dspark_remap_skips_unused_confidence_head_weights():
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=61))

    assert (
        DeepSeekV4DSparkMTP._remap_dspark_name(
            model,
            "mtp.2.confidence_head.proj.weight",
        )
        is None
    )


def test_dspark_remap_loads_moe_gate_correction_bias():
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=43))

    assert (
        DeepSeekV4DSparkMTP._remap_dspark_name(
            model,
            "mtp.1.ffn.gate.bias",
        )
        == "model.layers.44.mlp.gate.e_score_correction_bias"
    )


def test_dspark_remap_covers_representative_checkpoint_names():
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=43))

    cases = {
        "mtp.0.main_proj.weight": "model.layers.43.main_proj.weight",
        "mtp.0.main_norm.weight": "model.layers.43.main_norm.weight",
        "mtp.0.attn.attn_sink": "model.layers.43.self_attn.attn_sink",
        "mtp.0.attn.wq_a.weight": "model.layers.43.self_attn.wq_a.weight",
        "mtp.0.attn.wkv.weight": "model.layers.43.self_attn.wkv.weight",
        "mtp.1.ffn.shared_experts.w1.weight": ("model.layers.44.mlp.shared_experts.gate_proj.weight"),
        "mtp.1.ffn.shared_experts.w2.weight": ("model.layers.44.mlp.shared_experts.down_proj.weight"),
        "mtp.1.ffn.shared_experts.w3.weight": ("model.layers.44.mlp.shared_experts.up_proj.weight"),
        "mtp.2.hc_head_fn": "model.layers.45.hc_head_fn",
        "mtp.2.markov_head.markov_w2.weight": ("model.layers.45.markov_head.markov_w2.weight"),
        "mtp.2.norm.weight": "model.layers.45.norm.weight",
    }

    for source_name, expected_name in cases.items():
        assert DeepSeekV4DSparkMTP._remap_dspark_name(model, source_name) == expected_name


def test_dspark_load_weights_rejects_unmatched_mtp_params(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)

    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=43,
            num_attention_heads=8,
            expert_dtype="fp4",
        ),
        model=SimpleNamespace(
            num_dspark_layers=3,
            get_expert_mapping=lambda: [],
            finalize_mega_moe_weights=lambda: None,
        ),
    )
    model.named_parameters = lambda: iter(())
    model._remap_dspark_name = DeepSeekV4DSparkMTP._remap_dspark_name.__get__(model)

    with pytest.raises(ValueError, match="model\\.layers\\.43\\.self_attn\\.q_norm\\.weight"):
        DeepSeekV4DSparkMTP.load_weights(
            model,
            [("mtp.0.attn.q_norm.weight", torch.ones(1))],
        )


def test_dspark_load_weights_stacks_shared_expert_gate_up(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)

    calls = []

    class FakeParam:
        def __init__(self, name):
            self.name = name

        def weight_loader(self, param, loaded_weight, *args, **kwargs):
            calls.append((param.name, loaded_weight.clone(), args, kwargs))
            return True

    param_names = [
        "model.layers.43.main_proj.weight",
        "model.layers.43.main_norm.weight",
        "model.layers.44.mlp.shared_experts.gate_up_proj.weight",
        "model.layers.45.norm.weight",
        "model.hc_head_fn",
        "model.hc_head_base",
        "model.hc_head_scale",
        "model.layers.45.markov_head.markov_w1.weight",
        "model.layers.45.markov_head.markov_w2.weight",
    ]
    params = {name: FakeParam(name) for name in param_names}
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=43,
            num_attention_heads=8,
            expert_dtype="fp4",
        ),
        model=SimpleNamespace(
            num_dspark_layers=3,
            get_expert_mapping=lambda: [],
            finalize_mega_moe_weights=lambda: None,
        ),
    )
    model.named_parameters = lambda: iter(params.items())
    model._remap_dspark_name = DeepSeekV4DSparkMTP._remap_dspark_name.__get__(model)

    loaded = DeepSeekV4DSparkMTP.load_weights(
        model,
        [
            ("mtp.0.main_proj.weight", torch.ones(1)),
            ("mtp.0.main_norm.weight", torch.ones(1) * 2),
            ("mtp.1.ffn.shared_experts.w1.weight", torch.ones(1) * 3),
            ("mtp.1.ffn.shared_experts.w3.weight", torch.ones(1) * 4),
            ("mtp.2.norm.weight", torch.ones(1) * 5),
            ("mtp.2.hc_head_fn", torch.ones(1) * 6),
            ("mtp.2.hc_head_base", torch.ones(1) * 7),
            ("mtp.2.hc_head_scale", torch.ones(1) * 8),
            ("mtp.2.markov_head.markov_w1.weight", torch.ones(1) * 9),
            ("mtp.2.markov_head.markov_w2.weight", torch.ones(1) * 10),
        ],
    )

    assert "model.layers.44.mlp.shared_experts.gate_up_proj.weight" in loaded
    shared_calls = [call for call in calls if call[0] == "model.layers.44.mlp.shared_experts.gate_up_proj.weight"]
    assert [call[2] for call in shared_calls] == [(0,), (1,)]
    torch.testing.assert_close(shared_calls[0][1], torch.ones(1) * 3)
    torch.testing.assert_close(shared_calls[1][1], torch.ones(1) * 4)


def test_dspark_model_declares_target_shared_embedding_and_lm_head():
    assert DeepSeekV4DSparkMTP.has_own_embed_tokens is False
    assert DeepSeekV4DSparkMTP.has_own_lm_head is False


def test_draft_model_without_own_lm_head_shares_target_lm_head():
    target_lm_head = object()
    draft_lm_head = object()
    proposer = SimpleNamespace(
        method="mtp",
        model=SimpleNamespace(
            has_own_lm_head=False,
            lm_head=draft_lm_head,
        ),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=False),
            compilation_config=SimpleNamespace(
                cudagraph_mode=SimpleNamespace(
                    has_full_cudagraphs=lambda: False,
                ),
            ),
        ),
        use_cuda_graph=False,
    )

    AscendSpecDecodeBaseProposer._maybe_share_lm_head(
        proposer,
        SimpleNamespace(lm_head=target_lm_head),
    )

    assert proposer.model.lm_head is target_lm_head


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


def test_dspark_precompute_context_kv_selects_prefix_mapped_slot_mappings(monkeypatch):
    calls = []

    def make_layer(name: str, prefix: str) -> SimpleNamespace:
        def precompute_context_kv(main_x, positions, request_slots=None, context_slot_mapping=None):
            calls.append((name, main_x, positions, request_slots, context_slot_mapping))

        return SimpleNamespace(
            self_attn=SimpleNamespace(
                dsa_attn=SimpleNamespace(
                    swa_cache_layer=SimpleNamespace(prefix=prefix),
                ),
                precompute_context_kv=precompute_context_kv,
            )
        )

    monkeypatch.setattr(dspark_model_module, "_linear_output", lambda _proj, hidden_states: hidden_states + 1)
    context_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    positions = torch.tensor([4, 5, 6], dtype=torch.int32)
    request_slots = torch.tensor([1, 1, 1], dtype=torch.int32)
    slot_mapping_61 = torch.tensor([10, 11, 12], dtype=torch.int32)
    slot_mapping_62 = torch.tensor([20, 21, 22], dtype=torch.int32)
    model = SimpleNamespace(
        main_proj=object(),
        main_norm=lambda tensor: tensor * 2,
        layers={
            "61": make_layer("61", "model.layers.61.self_attn.swa_cache"),
            "62": make_layer("62", "model.layers.62.self_attn.swa_cache"),
        },
    )

    DeepseekV4DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        positions,
        context_slot_mapping={
            "model.layers.61.self_attn.swa_cache": slot_mapping_61,
            "model.layers.62.self_attn.swa_cache": slot_mapping_62,
        },
        context_request_slots=request_slots,
    )

    assert calls[0][4] is slot_mapping_61
    assert calls[1][4] is slot_mapping_62


def test_dspark_forward_passes_query_slot_mapping_to_layers():
    calls = []

    class FakeLayer:
        def __call__(
            self,
            *,
            positions,
            hidden_states,
            residual=None,
            post_mix=None,
            res_mix=None,
            input_ids,
            request_slots=None,
            slot_mapping=None,
            block_table=None,
        ):
            del residual, post_mix, res_mix
            calls.append((positions, hidden_states, input_ids, request_slots, slot_mapping, block_table))
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
        compute_head_hidden=lambda hidden_states, *args: hidden_states,
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
        call_positions, _, call_input_ids, call_request_slots, call_slot_mapping, call_block_table = call
        assert call_positions is positions
        assert call_input_ids is input_ids
        assert call_request_slots is request_slots
        assert call_slot_mapping is slot_mapping
        assert call_block_table is None
    torch.testing.assert_close(output, inputs_embeds.unsqueeze(-2).repeat(1, 2, 1) + 2)


def test_dspark_forward_carries_mhc_state_between_layers_and_head():
    calls = []
    head_call = {}

    class FakeLayer:
        def __init__(self, value: float):
            self.value = value

        def __call__(
            self,
            *,
            positions,
            hidden_states,
            residual=None,
            post_mix=None,
            res_mix=None,
            input_ids,
            request_slots=None,
            slot_mapping=None,
            block_table=None,
        ):
            del positions, input_ids, request_slots, slot_mapping, block_table
            calls.append((residual, post_mix, res_mix))
            state = hidden_states + self.value
            if residual is None:
                return state, state + 10, state + 20, state + 30
            return state, residual + self.value, post_mix + self.value, res_mix + self.value

    def fake_compute_head(hidden_states, residual=None, post_mix=None, res_mix=None):
        head_call["hidden_states"] = hidden_states
        head_call["residual"] = residual
        head_call["post_mix"] = post_mix
        head_call["res_mix"] = res_mix
        return hidden_states

    input_ids = torch.tensor([1], dtype=torch.int64)
    positions = torch.tensor([10], dtype=torch.int32)
    inputs_embeds = torch.ones(1, 3)
    model = SimpleNamespace(
        embed_tokens=None,
        hc_mult=2,
        layers={
            "61": FakeLayer(1),
            "62": FakeLayer(2),
        },
        compute_head_hidden=fake_compute_head,
    )

    output = DeepseekV4DSparkModel.forward(
        model,
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
    )

    expanded = inputs_embeds.unsqueeze(-2).repeat(1, 2, 1)
    first_hidden = expanded + 1
    second_hidden = first_hidden + 2
    assert calls[0] == (None, None, None)
    torch.testing.assert_close(calls[1][0], first_hidden + 10)
    torch.testing.assert_close(calls[1][1], first_hidden + 20)
    torch.testing.assert_close(calls[1][2], first_hidden + 30)
    torch.testing.assert_close(head_call["hidden_states"], second_hidden)
    torch.testing.assert_close(head_call["residual"], first_hidden + 12)
    torch.testing.assert_close(head_call["post_mix"], first_hidden + 22)
    torch.testing.assert_close(head_call["res_mix"], first_hidden + 32)
    torch.testing.assert_close(output, second_hidden)


def test_dspark_decoder_layer_uses_upstream_style_mhc_state_flow():
    layer = object.__new__(DeepseekV4DSparkDecoderLayer)
    layer.hc_attn_fn = torch.tensor(1.0)
    layer.hc_attn_scale = torch.tensor(2.0)
    layer.hc_attn_base = torch.tensor(3.0)
    layer.hc_ffn_fn = torch.tensor(4.0)
    layer.hc_ffn_scale = torch.tensor(5.0)
    layer.hc_ffn_base = torch.tensor(6.0)
    layer.input_layernorm = lambda x: x + 10
    layer.post_attention_layernorm = lambda x: x + 100
    layer.self_attn = lambda positions, hidden_states, _kv_cache, **kwargs: hidden_states + 1000
    layer.mlp = lambda hidden_states, input_ids: hidden_states + 10000
    calls = []

    def fake_pre(hidden_states, hc_fn, hc_scale, hc_base):
        calls.append(("pre", hidden_states.clone(), hc_fn, hc_scale, hc_base))
        return hidden_states + 1, hidden_states + 2, hidden_states + 3

    def fake_fused(hidden_states, residual, post_mix, res_mix, hc_fn, hc_scale, hc_base):
        calls.append(
            (
                "fused",
                hidden_states.clone(),
                residual.clone(),
                post_mix.clone(),
                res_mix.clone(),
                hc_fn,
                hc_scale,
                hc_base,
            )
        )
        return residual + 4, post_mix + 5, res_mix + 6, hidden_states + 7

    layer._mhc_pre = fake_pre
    layer._mhc_fused_post_pre = fake_fused
    hidden_states = torch.zeros(1, 2, 3)
    positions = torch.tensor([0], dtype=torch.int32)
    input_ids = torch.tensor([1], dtype=torch.int64)

    out, residual, post_mix, res_mix = DeepseekV4DSparkDecoderLayer.forward(
        layer,
        positions=positions,
        hidden_states=hidden_states,
        input_ids=input_ids,
    )

    assert [call[0] for call in calls] == ["pre", "fused"]
    torch.testing.assert_close(calls[1][1], hidden_states + 1 + 10 + 1000)
    torch.testing.assert_close(calls[1][2], hidden_states)
    torch.testing.assert_close(calls[1][3], hidden_states + 2)
    torch.testing.assert_close(calls[1][4], hidden_states + 3)
    torch.testing.assert_close(out, hidden_states + 1 + 10 + 1000 + 7 + 100 + 10000)
    torch.testing.assert_close(residual, hidden_states + 4)
    torch.testing.assert_close(post_mix, hidden_states + 7)
    torch.testing.assert_close(res_mix, hidden_states + 9)


def test_dspark_attention_rebuilds_standard_query_slot_mapping(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.block_size = 5
    attn.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(block_size=32),
    )
    positions = torch.tensor([26, 27, 28, 29, 30, 0], dtype=torch.int32)
    stale_slot_mapping = torch.tensor([90, 91, 92, 93, 94, -1], dtype=torch.int32)
    block_table = torch.tensor([[3, 17, 0, 0]], dtype=torch.int32)

    slot_mapping = DeepseekV4DSparkAttention._standard_query_slot_mapping_from_block_table(
        attn,
        positions,
        stale_slot_mapping,
        block_table,
    )

    torch.testing.assert_close(
        slot_mapping,
        torch.tensor([122, 123, 124, 125, 126, -1], dtype=torch.int32),
    )


def test_dspark_attention_rebuilds_standard_query_slot_mapping_with_token_to_req(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.block_size = 2
    attn.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(block_size=4),
    )
    positions = torch.tensor([8, 20, 9, 21], dtype=torch.int32)
    slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    token_to_req_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [3, 4, 5, 6, 7, 8],
            [10, 11, 12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )

    def fail_nonzero(*args, **kwargs):
        raise AssertionError("torch.nonzero is not ACLGraph-capture safe")

    with monkeypatch.context() as m:
        m.setattr(torch, "nonzero", fail_nonzero)
        rebuilt = DeepseekV4DSparkAttention._standard_query_slot_mapping_from_block_table(
            attn,
            positions,
            slot_mapping,
            block_table,
            token_to_req_indices,
        )

    torch.testing.assert_close(
        rebuilt,
        torch.tensor([20, 60, 21, 61], dtype=torch.int32),
    )


def test_dspark_forward_selects_prefix_mapped_slot_mapping_and_block_table():
    calls = []

    class FakeLayer:
        def __init__(self, prefix: str):
            self.self_attn = SimpleNamespace(
                dsa_attn=SimpleNamespace(
                    swa_cache_layer=SimpleNamespace(prefix=prefix),
                )
            )

        def __call__(
            self,
            *,
            positions,
            hidden_states,
            residual=None,
            post_mix=None,
            res_mix=None,
            input_ids,
            request_slots=None,
            slot_mapping=None,
            block_table=None,
        ):
            del residual, post_mix, res_mix
            calls.append((positions, hidden_states, input_ids, request_slots, slot_mapping, block_table))
            return hidden_states + 1

    input_ids = torch.tensor([1, 2], dtype=torch.int64)
    positions = torch.tensor([10, 11], dtype=torch.int32)
    inputs_embeds = torch.ones(2, 3)
    request_slots = torch.tensor([4, 4], dtype=torch.int32)
    slot_mapping_61 = torch.tensor([80, 81], dtype=torch.int32)
    slot_mapping_62 = torch.tensor([180, 181], dtype=torch.int32)
    block_table_61 = torch.tensor([[1, 2]], dtype=torch.int32)
    block_table_62 = torch.tensor([[11, 12]], dtype=torch.int32)
    model = SimpleNamespace(
        embed_tokens=None,
        hc_mult=2,
        layers={
            "61": FakeLayer("model.layers.61.self_attn.swa_cache"),
            "62": FakeLayer("model.layers.62.self_attn.swa_cache"),
        },
        compute_head_hidden=lambda hidden_states, *args: hidden_states,
    )

    output = DeepseekV4DSparkModel.forward(
        model,
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
        request_slots=request_slots,
        slot_mapping={
            "model.layers.61.self_attn.swa_cache": slot_mapping_61,
            "model.layers.62.self_attn.swa_cache": slot_mapping_62,
        },
        block_table={
            "model.layers.61.self_attn.swa_cache": block_table_61,
            "model.layers.62.self_attn.swa_cache": block_table_62,
        },
    )

    assert len(calls) == 2
    assert calls[0][4] is slot_mapping_61
    assert calls[0][5] is block_table_61
    assert calls[1][4] is slot_mapping_62
    assert calls[1][5] is block_table_62
    torch.testing.assert_close(output, inputs_embeds.unsqueeze(-2).repeat(1, 2, 1) + 2)


def test_dspark_store_standard_swa_kv_uses_dsa_slot_mapping(monkeypatch):
    from vllm_ascend.device import device_op as device_op_module

    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
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


def test_dspark_store_standard_swa_kv_capture_slices_padding(monkeypatch):
    from vllm_ascend.device import device_op as device_op_module

    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.setattr(dspark_model_module.torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        dspark_model_module,
        "_maybe_get_forward_context",
        lambda: SimpleNamespace(
            cudagraph_runtime_mode=dspark_model_module.CUDAGraphMode.NONE,
            num_actual_tokens=2,
        ),
    )
    monkeypatch.setattr(
        dspark_model_module,
        "_sync_npu_device_for_standard_pta",
        lambda tensor: (_ for _ in ()).throw(AssertionError("capture path must not synchronize")),
    )
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
    shared_kv = torch.arange(12, dtype=torch.float32).view(4, 1, 3)
    slot_mapping = torch.tensor([9, 18, -1, -1], dtype=torch.int32)

    DeepseekV4DSparkAttention._store_standard_swa_kv(attn, shared_kv, slot_mapping)

    assert calls[0][0] == "format"
    torch.testing.assert_close(calls[0][1], torch.tensor([9, 18], dtype=torch.int32))
    assert calls[1][0] == "scatter"
    torch.testing.assert_close(calls[1][2], shared_kv[:2])
    torch.testing.assert_close(calls[1][3], torch.tensor([[1, 1], [2, 2]], dtype=torch.int32))


def test_dspark_standard_attention_can_fallback_to_pta(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_PTA_REF", "1")

    expected = torch.ones(2, 4, 8)
    calls = []

    def fake_sas(*args, **kwargs):
        raise AssertionError("SAS fast path must not run when disabled")

    def fake_pta(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache_sas", fake_sas)
    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache", fake_pta)

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.zeros(4, 16, 1, 8),
            block_size=16,
        )
    )
    attn.attn_sink = torch.zeros(4)
    attn.n_local_heads = 4
    attn.block_size = 2
    attn.window_size = 6
    attn.scale = 0.5

    out = DeepseekV4DSparkAttention._run_standard_dspark_attention(
        attn,
        q=torch.zeros(2, 4, 8),
        positions=torch.tensor([6, 7], dtype=torch.int32),
        slot_mapping=torch.tensor([6, 7], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        dspark_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        dspark_seq_lens=torch.tensor([8], dtype=torch.int32),
    )

    assert out is expected
    assert len(calls) == 1


def test_dspark_standard_attention_uses_sas_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PTA_REF", raising=False)

    expected = torch.ones(2, 4, 8)
    dspark_swa_indices = torch.full((2, 1, 8), -1, dtype=torch.int32)
    dspark_swa_lens = torch.tensor([2, 2], dtype=torch.int32)
    sas_metadata = torch.tensor([123], dtype=torch.int32)
    calls = []

    def fake_sas(*args, **kwargs):
        calls.append(("sas", args, kwargs))
        return expected

    def fake_pta(*args, **kwargs):
        raise AssertionError("PTA should not run when SAS returns an output")

    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache_sas", fake_sas)
    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache", fake_pta)
    monkeypatch.setattr(
        dspark_model_module,
        "_maybe_get_forward_context",
        lambda: SimpleNamespace(
            cudagraph_runtime_mode=dspark_model_module.CUDAGraphMode.FULL,
            draft_attn_metadatas=[
                {
                    "layers.0.self_attn": SimpleNamespace(
                        decode=SimpleNamespace(
                            dspark_swa_indices=dspark_swa_indices,
                            dspark_swa_lens=dspark_swa_lens,
                            sas_metadata=sas_metadata,
                        )
                    )
                }
            ],
        ),
    )

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.zeros(4, 16, 1, 8),
            block_size=16,
            prefix="layers.0.self_attn",
        )
    )
    attn.attn_sink = torch.zeros(4)
    attn.n_local_heads = 4
    attn.block_size = 2
    attn.window_size = 6
    attn.scale = 0.5

    out = DeepseekV4DSparkAttention._run_standard_dspark_attention(
        attn,
        q=torch.zeros(2, 4, 8),
        positions=torch.tensor([6, 7], dtype=torch.int32),
        slot_mapping=torch.tensor([6, 7], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        dspark_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        dspark_seq_lens=torch.tensor([8], dtype=torch.int32),
    )

    assert out is expected
    assert calls[0][0] == "sas"
    assert calls[0][2]["dspark_swa_indices"] is dspark_swa_indices
    assert calls[0][2]["dspark_swa_lens"] is dspark_swa_lens
    assert calls[0][2]["sas_metadata"] is sas_metadata
    assert calls[0][2]["skip_scheduling_guard"] is True
    assert calls[0][2]["raise_on_error"] is True


def test_dspark_standard_attention_passes_actual_query_tokens_to_sas(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PTA_REF", raising=False)

    expected = torch.ones(8, 4, 8)
    calls = []

    def fake_sas(*args, **kwargs):
        calls.append(("sas", args, kwargs))
        return expected

    def fake_pta(*args, **kwargs):
        raise AssertionError("PTA should not run when SAS returns an output")

    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache_sas", fake_sas)
    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache", fake_pta)
    monkeypatch.setattr(
        dspark_model_module,
        "_maybe_get_forward_context",
        lambda: SimpleNamespace(
            cudagraph_runtime_mode=dspark_model_module.CUDAGraphMode.NONE,
            num_actual_tokens=5,
        ),
    )

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.zeros(4, 16, 1, 8),
            block_size=16,
        )
    )
    attn.attn_sink = torch.zeros(4)
    attn.n_local_heads = 4
    attn.block_size = 5
    attn.window_size = 6
    attn.scale = 0.5

    out = DeepseekV4DSparkAttention._run_standard_dspark_attention(
        attn,
        q=torch.zeros(8, 4, 8),
        positions=torch.arange(8, dtype=torch.int32),
        slot_mapping=torch.arange(8, dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        dspark_query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        dspark_seq_lens=torch.tensor([8], dtype=torch.int32),
    )

    assert out is expected
    assert calls[0][2]["num_query_tokens"] == 5
    assert calls[0][2]["skip_scheduling_guard"] is False


def test_dspark_standard_attention_does_not_fallback_to_pta_during_capture(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PTA_REF", raising=False)

    def fake_sas(*args, **kwargs):
        return None

    def fake_pta(*args, **kwargs):
        raise AssertionError("PTA fallback must not run during graph capture")

    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache_sas", fake_sas)
    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache", fake_pta)
    monkeypatch.setattr(
        dspark_model_module,
        "_maybe_get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=dspark_model_module.CUDAGraphMode.FULL),
    )

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.zeros(4, 16, 1, 8),
            block_size=16,
        )
    )
    attn.attn_sink = torch.zeros(4)
    attn.n_local_heads = 4
    attn.block_size = 2
    attn.window_size = 6
    attn.scale = 0.5

    with pytest.raises(RuntimeError, match="did not produce output"):
        DeepseekV4DSparkAttention._run_standard_dspark_attention(
            attn,
            q=torch.zeros(2, 4, 8),
            positions=torch.tensor([6, 7], dtype=torch.int32),
            slot_mapping=torch.tensor([6, 7], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            dspark_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            dspark_seq_lens=torch.tensor([8], dtype=torch.int32),
        )


def test_dspark_standard_attention_requires_standard_cache_metadata_during_capture(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.setattr(
        dspark_model_module,
        "_maybe_get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=dspark_model_module.CUDAGraphMode.FULL),
    )

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.dsa_attn = SimpleNamespace(swa_cache_layer=SimpleNamespace(kv_cache=None, block_size=16))
    attn.block_size = 2
    attn.window_size = 6

    with pytest.raises(RuntimeError, match="requires block_table"):
        DeepseekV4DSparkAttention._run_standard_dspark_attention(
            attn,
            q=torch.zeros(2, 4, 8),
            positions=torch.tensor([6, 7], dtype=torch.int32),
            slot_mapping=torch.tensor([6, 7], dtype=torch.int32),
            block_table=None,
            dspark_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            dspark_seq_lens=torch.tensor([8], dtype=torch.int32),
        )

    with pytest.raises(RuntimeError, match="requires SWA kv_cache"):
        DeepseekV4DSparkAttention._run_standard_dspark_attention(
            attn,
            q=torch.zeros(2, 4, 8),
            positions=torch.tensor([6, 7], dtype=torch.int32),
            slot_mapping=torch.tensor([6, 7], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            dspark_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            dspark_seq_lens=torch.tensor([8], dtype=torch.int32),
        )
