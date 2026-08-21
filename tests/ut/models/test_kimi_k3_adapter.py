# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.models import kimi_k3
from vllm_ascend.models.kimi_k3 import (
    AscendKimiK3ForConditionalGeneration,
    AscendKimiK3MultiModalProjector,
    AscendKimiLinearForCausalLM,
    AscendKimiLinearModel,
    AscendKimiMLAAttention,
    AscendKimiMoE,
)
from vllm_ascend.models.kimi_k3_dspark import (
    AscendK3DSparkDecoderLayer,
    AscendK3DSparkForCausalLM,
)


def test_ascend_attn_res_matches_canonical_k3_math():
    prefix_sum = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    block_residual = torch.tensor(
        [
            [[0.5, 1.5], [2.5, 3.5], [1000.0, 1000.0]],
            [[1.0, 0.0], [0.0, 1.0], [1000.0, 1000.0]],
        ]
    )
    norm = SimpleNamespace(weight=torch.tensor([1.0, 1.5]), variance_epsilon=1e-5)
    proj = SimpleNamespace(weight=torch.tensor([[0.25, -0.5]]))

    output = kimi_k3._apply_ascend_attn_res(
        prefix_sum,
        block_residual,
        proj,
        norm,
        num_valid_blocks=2,
    )

    values = torch.cat(
        (block_residual[:, :2], prefix_sum.unsqueeze(1)),
        dim=1,
    ).float()
    inverse_rms = torch.rsqrt(values.square().mean(-1, keepdim=True) + norm.variance_epsilon)
    normalized_without_gamma = values * inverse_rms
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    probabilities = (normalized_without_gamma * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    expected = torch.matmul(probabilities, values).squeeze(1).to(prefix_sum.dtype)
    torch.testing.assert_close(output, expected)


def _make_moe_config(**overrides):
    values = {
        "hidden_size": 16,
        "moe_intermediate_size": 32,
        "num_experts": 8,
        "num_experts_per_token": 2,
        "moe_renormalize": True,
        "routed_expert_hidden_size": None,
        "latent_moe_use_norm": False,
        "routed_scaling_factor": 1.0,
        "num_shared_experts": None,
        "hidden_act": "silu",
        "activation_situ_beta": None,
        "activation_situ_linear_beta": None,
        "use_grouped_topk": False,
        "num_expert_group": None,
        "topk_group": None,
        "moe_router_activation_func": "softmax",
        "rms_norm_eps": 1e-6,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ascend_kimi_moe_uses_standard_runner_dispatch(monkeypatch):
    class FakeGate(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

    factory = MagicMock(return_value=nn.Identity())
    monkeypatch.setattr(kimi_k3, "GateLinear", FakeGate)
    monkeypatch.setattr(kimi_k3, "FusedMoEFactory", factory)

    moe = AscendKimiMoE(
        config=_make_moe_config(),
        prefix="model.layers.1.block_sparse_moe",
        layer_idx=1,
    )

    assert moe.layer_idx == 1
    assert factory.call_args.kwargs["intermediate_size"] == 32
    assert "runner_cls" not in factory.call_args.kwargs


def test_dspark_decoder_uses_upstream_mlp_activation_contract(
    monkeypatch,
):
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        intermediate_size=16,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )
    vllm_config = SimpleNamespace(cache_config=None)
    mlp_factory = MagicMock(return_value=nn.Identity())
    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.get_draft_quant_config",
        lambda _: None,
    )
    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.AscendKimiMLAAttention",
        lambda **_: nn.Identity(),
    )
    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.KimiMLP",
        mlp_factory,
    )

    with set_current_vllm_config(VllmConfig()):
        AscendK3DSparkDecoderLayer(
            vllm_config=vllm_config,
            config=config,
            layer_idx=0,
            start_layer_id=4,
            prefix="model",
        )

    assert mlp_factory.call_args.kwargs["hidden_act"] == "silu"
    assert "activation_situ_beta" not in mlp_factory.call_args.kwargs
    assert "activation_situ_linear_beta" not in mlp_factory.call_args.kwargs


def test_ascend_kimi_moe_quantizes_modelslim_latent_projections(monkeypatch):
    class FakeGate(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    class FakeLinear(nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.kwargs = kwargs

    config = _make_moe_config(routed_expert_hidden_size=8)
    quant_config = MagicMock()
    quant_config.get_name.return_value = "ascend"
    factory = MagicMock(return_value=nn.Identity())
    monkeypatch.setattr(kimi_k3, "GateLinear", FakeGate)
    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(kimi_k3, "FusedMoEFactory", factory)

    moe = AscendKimiMoE(
        config=config,
        quant_config=quant_config,
        prefix="model.layers.1.block_sparse_moe",
        layer_idx=1,
    )

    assert moe.routed_expert_down_proj.input_size == 16
    assert moe.routed_expert_down_proj.output_size == 8
    assert moe.routed_expert_down_proj.kwargs == {
        "bias": False,
        "quant_config": quant_config,
        "prefix": "model.layers.1.block_sparse_moe.routed_expert_down_proj",
    }
    assert moe.routed_expert_up_proj.input_size == 8
    assert moe.routed_expert_up_proj.output_size == 16
    assert moe.routed_expert_up_proj.kwargs == {
        "bias": False,
        "quant_config": quant_config,
        "prefix": "model.layers.1.block_sparse_moe.routed_expert_up_proj",
    }
    assert factory.call_args.kwargs["routed_input_transform"] is moe.routed_expert_down_proj
    assert factory.call_args.kwargs["routed_output_transform"] is moe.routed_output_transform
    assert "runner_cls" not in factory.call_args.kwargs


def test_kimi_text_model_retains_upstream_checkpoint_packing():
    assert (
        AscendKimiLinearForCausalLM.packed_modules_mapping
        is kimi_k3.UpstreamPackedKimiLinearModel.packed_modules_mapping
    )


def test_kimi_mixed_kda_gate_weights_use_upstream_packed_loader(monkeypatch):
    model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
    nn.Module.__init__(model)
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.in_proj_gfab = nn.Module()
    packed_weight = nn.Parameter(torch.empty(6, 4))
    layer.self_attn.in_proj_gfab.register_parameter("weight", packed_weight)
    layer.router = nn.Linear(4, 1, bias=False)
    model.layers = nn.ModuleList([layer])

    remaining = []

    def fake_upstream_load_weights(_self, weights):
        remaining.extend(weights)
        return {name for name, *_ in remaining}

    monkeypatch.setattr(
        kimi_k3.UpstreamKimiLinearModel,
        "load_weights",
        fake_upstream_load_weights,
    )
    source_weights = [
        ("layers.0.router.weight", torch.full((1, 4), 0.5)),
        ("layers.0.self_attn.g_proj.weight", torch.full((1,), 1.0)),
        ("layers.0.self_attn.f_a_proj.weight", torch.full((1,), 2.0)),
        ("layers.0.self_attn.b_proj.weight", torch.full((1,), 3.0)),
        ("layers.0.self_attn.o_proj.weight", torch.full((1,), 4.0)),
    ]

    loaded = model.load_weights(iter(source_weights))

    assert remaining[0] == source_weights[0]
    assert remaining[-1] == source_weights[-1]
    assert [name for name, _, _ in remaining[1:4]] == [
        "layers.0.self_attn.in_proj_gfab.weight",
    ] * 3
    assert [loaded_weight.item() for _, loaded_weight, _ in remaining[1:4]] == [1.0, 2.0, 3.0]
    assert [kwargs["loaded_shard_id"] for _, _, kwargs in remaining[1:4]] == [0, 1, 2]
    assert loaded == {
        "layers.0.self_attn.in_proj_gfab.weight",
        "layers.0.router.weight",
        "layers.0.self_attn.o_proj.weight",
    }


def test_kimi_text_model_layer_factory_accepts_prefix_keyword(monkeypatch):
    config = SimpleNamespace(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
        attn_res_block_size=None,
        num_attention_heads=1,
    )
    vllm_config = MagicMock()
    vllm_config.model_config.hf_text_config = config
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    decoder_layer = nn.Identity()
    decoder_layer_factory = MagicMock(return_value=decoder_layer)

    def fake_make_layers(num_hidden_layers, layer_fn, *, prefix):
        assert num_hidden_layers == 1
        assert layer_fn(prefix=f"{prefix}.0") is decoder_layer
        return 0, 1, nn.ModuleList([decoder_layer])

    monkeypatch.setattr(kimi_k3, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(kimi_k3, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(kimi_k3, "AscendKimiDecoderLayer", decoder_layer_factory)
    monkeypatch.setattr(kimi_k3, "make_layers", fake_make_layers)

    model = AscendKimiLinearModel(vllm_config=vllm_config, prefix="model")

    assert model.start_layer == 0
    assert model.end_layer == 1
    decoder_layer_factory.assert_called_once_with(
        config,
        vllm_config,
        "model.layers.0",
    )


def test_kimi_mla_cache_spec_preserves_hybrid_page_padding():
    real_page_size = 128 * 576 * torch.bfloat16.itemsize
    padded_page_size = real_page_size + 128
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=padded_page_size,
    )

    assert spec.real_page_size_bytes == real_page_size
    assert spec.page_size_bytes == padded_page_size
    assert AscendMLAAttentionSpec.merge([spec, spec]).page_size_bytes == padded_page_size


def test_ascend_mla_exposes_layer_and_cache_contract():
    attention = AscendKimiMLAAttention.__new__(AscendKimiMLAAttention)
    layer = MagicMock()
    layer.layer_name = "model.layers.1.self_attn.attn"
    layer.impl = object()
    layer.kv_cache = (object(), object())
    layer.kv_cache_dtype = "auto"
    layer._k_scale = 1.0
    attention.mla_attn = MagicMock()
    attention.mla_attn.mla_attn = layer
    attention.mla_attn.is_vl_first_layer = True

    assert attention.layer_name == layer.layer_name
    assert attention.impl is layer.impl
    assert attention.kv_cache is layer.kv_cache
    assert attention.kv_cache_dtype == layer.kv_cache_dtype
    assert attention._k_scale == layer._k_scale
    assert attention.is_vl_first_layer is True


def test_dspark_configures_upstream_mla_without_rebuilding(monkeypatch):
    impl = SimpleNamespace(
        scale=0.0,
        rotary_emb=None,
        use_mla_rope=False,
    )
    layer = SimpleNamespace(
        scale=0.0,
        non_causal_multi_token_decode=False,
        impl=impl,
    )
    upstream_wrapper = SimpleNamespace(mla_attn=layer)

    def fake_upstream_init(self, **_kwargs):
        nn.Module.__init__(self)
        self.scaling = 0.125
        self.mla_attn = upstream_wrapper

    rotary_emb = object()
    monkeypatch.setattr(
        kimi_k3.UpstreamKimiMLAAttention,
        "__init__",
        fake_upstream_init,
    )
    monkeypatch.setattr(kimi_k3, "get_rope", lambda *_args, **_kwargs: rotary_emb)

    attention = AscendKimiMLAAttention(
        config=SimpleNamespace(
            rope_parameters={"rope_type": "default"},
            max_position_embeddings=4096,
        ),
        hidden_size=16,
        num_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        use_output_gate=False,
        use_rope=True,
        prefix="model.layers.1.self_attn",
        non_causal_multi_token_decode=True,
    )

    assert attention.mla_attn is upstream_wrapper
    assert layer.scale == attention.scaling
    assert layer.non_causal_multi_token_decode is True
    assert impl.scale == attention.scaling
    assert impl.rotary_emb is rotary_emb
    assert impl.use_mla_rope is True


def test_projector_applies_optional_modelslim_rotation():
    class ScaleLinear(nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 2, None

    projector = AscendKimiK3MultiModalProjector.__new__(AscendKimiK3MultiModalProjector)
    nn.Module.__init__(projector)
    image_features = torch.tensor([[1.0, 2.0]])

    with patch.object(
        kimi_k3.KimiK25MultiModalProjector,
        "forward",
        lambda self, hidden_states: hidden_states,
    ):
        projector.rot_proj = ScaleLinear()
        torch.testing.assert_close(
            projector(image_features),
            image_features * 2,
        )
        projector.rot_proj = None
        torch.testing.assert_close(projector(image_features), image_features)


def test_projector_rotation_is_removed_when_checkpoint_omits_it(monkeypatch):
    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.mm_projector = nn.Module()
    wrapper.mm_projector.rot_proj = nn.Linear(1, 1, bias=False)

    upstream_load_weights = MagicMock(return_value={"mm_projector.linear_1.weight"})
    monkeypatch.setattr(
        kimi_k3.UpstreamKimiK3ForConditionalGeneration,
        "load_weights",
        upstream_load_weights,
    )

    loaded = wrapper.load_weights(iter(()))

    assert loaded == {"mm_projector.linear_1.weight"}
    assert wrapper.mm_projector.rot_proj is None
    upstream_load_weights.assert_called_once()


def test_projector_rotation_is_kept_when_checkpoint_provides_it(monkeypatch):
    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.mm_projector = nn.Module()
    wrapper.mm_projector.rot_proj = nn.Linear(1, 1, bias=False)

    upstream_load_weights = MagicMock(return_value={"mm_projector.rot_proj.weight"})
    monkeypatch.setattr(
        kimi_k3.UpstreamKimiK3ForConditionalGeneration,
        "load_weights",
        upstream_load_weights,
    )

    wrapper.load_weights(iter(()))

    assert wrapper.mm_projector.rot_proj is not None
    upstream_load_weights.assert_called_once()


class _DraftTokenEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


def _make_k3_dspark_for_embedding_test() -> AscendK3DSparkForCausalLM:
    model = AscendK3DSparkForCausalLM.__new__(AscendK3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.model = _DraftTokenEmbedder()
    return model


def test_k3_dspark_load_weights_keeps_per_layer_context_kv(monkeypatch):
    model = AscendK3DSparkForCausalLM.__new__(AscendK3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.rotation_path = None
    source_weights = [
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            torch.ones(1, 1),
        )
    ]
    seen_names: list[str] = []

    class CapturingLoader:
        def __init__(self, loaded_model, *, skip_substrs):
            assert loaded_model is model
            assert skip_substrs == list(model.checkpoint_skip_substrs)

        def load_weights(self, weights, *, mapper):
            assert mapper is model.hf_to_vllm_mapper
            seen_names.extend(name for name, _ in weights)
            return {"model.layers.0.self_attn.fused_qkv_a_proj.weight"}

    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.AutoWeightsLoader",
        CapturingLoader,
    )

    loaded = model.load_weights(iter(source_weights))

    assert seen_names == [source_weights[0][0]]
    assert loaded == {"model.layers.0.self_attn.fused_qkv_a_proj.weight"}


def test_k3_dspark_reuses_modelslim_rotation_loader(monkeypatch):
    model = AscendK3DSparkForCausalLM.__new__(AscendK3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.rotation_path = "rotation.safetensors"
    source_weights = [
        ("context_proj.weight", torch.ones(2, 4)),
        ("context_norm.weight", torch.ones(2)),
    ]
    rotated_weight = torch.full((2, 4), 2.0)
    seen_weights: list[tuple[str, torch.Tensor]] = []

    class CapturingLoader:
        def __init__(self, loaded_model, *, skip_substrs):
            assert loaded_model is model
            assert skip_substrs == list(model.checkpoint_skip_substrs)

        def load_weights(self, weights, *, mapper):
            assert mapper is model.hf_to_vllm_mapper
            seen_weights.extend(weights)
            return {name for name, _ in seen_weights}

    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.AutoWeightsLoader",
        CapturingLoader,
    )
    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.get_rotation_matrix",
        lambda path: torch.eye(4) if path == model.rotation_path else None,
    )
    process_weight = MagicMock(return_value=rotated_weight)
    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.process_weight",
        process_weight,
    )

    model.load_weights(iter(source_weights))

    process_weight.assert_called_once()
    torch.testing.assert_close(process_weight.call_args.args[0], source_weights[0][1])
    torch.testing.assert_close(process_weight.call_args.args[1], torch.eye(4))
    assert seen_weights[0][0] == "context_proj.weight"
    assert seen_weights[0][1] is rotated_weight
    assert seen_weights[1][0] == source_weights[1][0]
    assert seen_weights[1][1] is source_weights[1][1]


def test_k3_dspark_embed_input_ids_keeps_text_only_path():
    model = _make_k3_dspark_for_embedding_test()

    output = model.embed_input_ids(torch.tensor([1, 2]))

    torch.testing.assert_close(
        output,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_k3_dspark_embed_input_ids_merges_multimodal_embeddings():
    model = _make_k3_dspark_for_embedding_test()
    input_ids = torch.tensor([1, 999, 2])
    is_multimodal = torch.tensor([False, True, False])
    image_embedding = torch.tensor([[9.0, 10.0]])

    output = model.embed_input_ids(
        input_ids,
        multimodal_embeddings=(image_embedding,),
        is_multimodal=is_multimodal,
    )

    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                [1.0, 2.0],
                [9.0, 10.0],
                [3.0, 4.0],
            ]
        ),
    )


def test_k3_dspark_embed_input_ids_without_multimodal_mask_uses_text_path():
    model = _make_k3_dspark_for_embedding_test()

    output = model.embed_input_ids(
        torch.tensor([1]),
        multimodal_embeddings=(torch.tensor([[9.0, 10.0]]),),
    )

    torch.testing.assert_close(output, torch.tensor([[1.0, 2.0]]))
