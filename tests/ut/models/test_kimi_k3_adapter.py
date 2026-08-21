# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
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
    AscendKimiMLP,
    AscendKimiMoE,
)
from vllm_ascend.models.kimi_k3_dspark import (
    AscendK3DSparkDecoderLayer,
    AscendK3DSparkForCausalLM,
    AscendK3DSparkModel,
)


def test_ascend_attn_res_matches_canonical_k3_math(monkeypatch):
    prefix_sum = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    block_residual = torch.tensor(
        [
            [[0.5, 1.5], [2.5, 3.5], [1000.0, 1000.0]],
            [[1.0, 0.0], [0.0, 1.0], [1000.0, 1000.0]],
        ]
    )
    norm = SimpleNamespace(weight=torch.tensor([1.0, 1.5]), variance_epsilon=1e-5)
    proj = SimpleNamespace(weight=torch.tensor([[0.25, -0.5]]))

    monkeypatch.setattr(
        kimi_k3,
        "_EXTRA_CTX",
        SimpleNamespace(flash_comm_v1_enabled=False),
    )

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


def test_ascend_attn_res_avoids_broadcast_score_product(monkeypatch):
    prefix_sum = torch.ones(2, 4)
    block_residual = torch.ones(2, 3, 4)
    norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5)
    proj = SimpleNamespace(weight=torch.ones(1, 4))
    original_matmul = torch.matmul
    score_matmul_shapes = []

    def record_matmul(left, right, *args, **kwargs):
        if left.shape == (2, 3, 4) and right.shape == (4,):
            score_matmul_shapes.append((left.shape, right.shape))
        return original_matmul(left, right, *args, **kwargs)

    monkeypatch.setattr(
        kimi_k3,
        "_EXTRA_CTX",
        SimpleNamespace(flash_comm_v1_enabled=False),
    )
    monkeypatch.setattr(torch, "matmul", record_matmul)

    kimi_k3._apply_ascend_attn_res(
        prefix_sum,
        block_residual,
        proj,
        norm,
        num_valid_blocks=2,
    )

    assert score_matmul_shapes == [((2, 3, 4), (4,))]


def test_ascend_kimi_moe_delegates_padding_to_routed_experts(monkeypatch):
    config = SimpleNamespace(min_moe_intermediate_per_partition=256)
    delegated = {}

    def fake_init(self, *, config, **kwargs):
        nn.Module.__init__(self)
        self.use_latent_moe = False
        delegated["config"] = config
        delegated["kwargs"] = kwargs

    monkeypatch.setattr(kimi_k3.UpstreamKimiMoE, "__init__", fake_init)

    AscendKimiMoE(
        config=config,
        prefix="model.layers.1.block_sparse_moe",
        layer_idx=1,
    )

    assert delegated["config"] is not config
    assert delegated["config"].min_moe_intermediate_per_partition == 0
    assert config.min_moe_intermediate_per_partition == 256
    assert delegated["kwargs"] == {
        "quant_config": None,
        "prefix": "model.layers.1.block_sparse_moe",
        "layer_idx": 1,
    }


def test_ascend_kimi_mlp_forwards_explicit_situ_parameters(
    monkeypatch,
):
    delegated: dict[str, object] = {}

    def fake_init(
        self,
        *args,
        hidden_act,
        activation_situ_beta,
        activation_situ_linear_beta,
        **kwargs,
    ):
        nn.Module.__init__(self)
        delegated.update(
            hidden_act=hidden_act,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
        )

    monkeypatch.setattr(kimi_k3.KimiMLP, "__init__", fake_init)

    with set_current_vllm_config(VllmConfig()):
        mlp = AscendKimiMLP(
            hidden_act="situ",
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
        )

    assert delegated == {
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
    }
    assert mlp.act_fn.beta == 4.0
    assert mlp.act_fn.linear_beta == 25.0


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
        "vllm_ascend.models.kimi_k3_dspark.AscendKimiMLP",
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
    class FakeLinear(nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.kwargs = kwargs

    class FakeRunner(nn.Module):
        def __init__(self):
            super().__init__()
            self.routed_input_transform = nn.Identity()
            self.routed_output_transform = nn.Identity()

    config = SimpleNamespace(
        hidden_size=16,
        min_moe_intermediate_per_partition=256,
    )
    quant_config = MagicMock()
    quant_config.get_name.return_value = "ascend"
    norm = nn.Identity()

    def fake_init(self, *, config, **kwargs):
        nn.Module.__init__(self)
        self.use_latent_moe = True
        self.moe_hidden_size = 8
        self.routed_expert_norm = norm
        self.routed_expert_down_proj = nn.Identity()
        self.routed_expert_up_proj = nn.Identity()
        self.routed_output_transform = nn.Identity()
        self.experts = FakeRunner()

    monkeypatch.setattr(kimi_k3.UpstreamKimiMoE, "__init__", fake_init)
    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", FakeLinear)

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
    assert moe.experts.routed_input_transform is moe.routed_expert_down_proj
    assert moe.experts.routed_output_transform is moe.routed_output_transform


def test_kimi_text_model_retains_upstream_checkpoint_packing():
    assert AscendKimiLinearForCausalLM.packed_modules_mapping == {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvgfab": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
        ],
        "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }


def test_kimi_mixed_kda_gate_weights_load_into_float_packed_projection(monkeypatch):
    model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
    nn.Module.__init__(model)
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.in_proj_gfab = nn.Module()
    layer.self_attn.in_proj_gfab.load_shard_weight = MagicMock()
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

    weight_loader = layer.self_attn.in_proj_gfab.load_shard_weight
    assert [call.args[2] for call in weight_loader.call_args_list] == [0, 1, 2]
    assert [call.args[1].item() for call in weight_loader.call_args_list] == [1.0, 2.0, 3.0]
    assert remaining == [source_weights[0], source_weights[-1]]
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

    loader = MagicMock()
    loader.load_weights.return_value = {"mm_projector.linear_1.weight"}
    monkeypatch.setattr(kimi_k3, "AutoWeightsLoader", lambda model: loader)

    loaded = wrapper.load_weights(iter(()))

    assert loaded == {"mm_projector.linear_1.weight"}
    assert wrapper.mm_projector.rot_proj is None


def test_projector_rotation_is_kept_when_checkpoint_provides_it(monkeypatch):
    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.mm_projector = nn.Module()
    wrapper.mm_projector.rot_proj = nn.Linear(1, 1, bias=False)

    loader = MagicMock()
    loader.load_weights.return_value = {"mm_projector.rot_proj.weight"}
    monkeypatch.setattr(kimi_k3, "AutoWeightsLoader", lambda model: loader)

    wrapper.load_weights(iter(()))

    assert wrapper.mm_projector.rot_proj is not None


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


def test_k3_dspark_embed_input_ids_requires_multimodal_mask():
    model = _make_k3_dspark_for_embedding_test()

    with pytest.raises(ValueError, match="is_multimodal"):
        model.embed_input_ids(
            torch.tensor([1]),
            multimodal_embeddings=(torch.tensor([[9.0, 10.0]]),),
        )


def test_k3_dspark_rejects_incomplete_context_slot_mappings():
    model = AscendK3DSparkModel.__new__(AscendK3DSparkModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])

    with pytest.raises(ValueError, match="one entry per draft layer"):
        model.precompute_and_store_context_kv(
            torch.ones(1, 1),
            torch.zeros(1, dtype=torch.int64),
            [torch.zeros(1, dtype=torch.int32)],
        )
