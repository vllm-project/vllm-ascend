# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
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
from vllm_ascend.utils import vllm_version_is


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

    AscendKimiMoE(
        config=_make_moe_config(),
        prefix="model.layers.1.block_sparse_moe",
        use_sequence_parallel=True,
    )

    assert factory.call_args.kwargs["intermediate_size"] == 32
    assert factory.call_args.kwargs["is_sequence_parallel"] is True
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
        full_attention_causal=True,
    )
    vllm_config = SimpleNamespace(cache_config=None)
    mlp_factory = MagicMock(return_value=nn.Identity())
    attention_factory = MagicMock(return_value=nn.Identity())
    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.get_draft_quant_config",
        lambda _: None,
    )
    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.AscendKimiMLAAttention",
        attention_factory,
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
    assert attention_factory.call_args.kwargs["non_causal_multi_token_decode"] is False


def test_k3_dspark_reports_draft_attention_causality():
    model = AscendK3DSparkForCausalLM.__new__(AscendK3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.model = SimpleNamespace(layers=[object(), object(), object()])

    model.config = SimpleNamespace(dflash_config={"causal": True})
    assert model.get_draft_attn_causal() == [True, True, True]

    model.config = SimpleNamespace(full_attention_causal=True)
    assert model.get_draft_attn_causal() == [True, True, True]

    model.config = SimpleNamespace()
    assert model.get_draft_attn_causal() == [False, False, False]


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
    vllm_config.parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        tensor_parallel_size=2,
    )
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
        use_sequence_parallel=True,
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

    assert attention.layer_name == layer.layer_name
    assert attention.impl is layer.impl
    assert attention.kv_cache is layer.kv_cache
    assert attention.kv_cache_dtype == layer.kv_cache_dtype
    assert attention._k_scale == layer._k_scale


def test_kimi_attention_residual_stays_sequence_sharded(monkeypatch):
    class IdentityAttention(nn.Module):
        def forward(self, *, hidden_states, positions):
            del positions
            return hidden_states

    layer = kimi_k3.AscendKimiDecoderLayer.__new__(kimi_k3.AscendKimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = True
    layer.prev_valid_blocks = 0
    layer.is_block_write_layer = False
    layer.input_layernorm = nn.Identity()
    layer.post_attention_layernorm = nn.Identity()
    layer.mlp = nn.Identity()
    layer.self_attention_res_proj = object()
    layer.self_attention_res_norm = object()
    layer.mlp_res_proj = object()
    layer.mlp_res_norm = object()
    layer.self_attn = IdentityAttention()

    collective_shapes = []

    def fake_all_gather(hidden_states):
        collective_shapes.append(("gather", hidden_states.shape))
        return torch.cat((hidden_states, hidden_states), dim=0)

    def fake_reduce_scatter(hidden_states):
        collective_shapes.append(("reduce_scatter", hidden_states.shape))
        return hidden_states.chunk(2, dim=0)[0]

    monkeypatch.setattr(kimi_k3, "sp_all_gather", fake_all_gather)
    monkeypatch.setattr(kimi_k3, "sp_reduce_scatter", fake_reduce_scatter)
    monkeypatch.setattr(
        kimi_k3,
        "_apply_ascend_attn_res",
        lambda prefix_sum, *_args, **_kwargs: prefix_sum,
    )

    hidden_states = torch.arange(4, dtype=torch.float32).view(2, 2)
    block_residual = torch.zeros(2, 1, 2)
    output, returned_residual = layer.forward_attn_residual(
        positions=torch.arange(3),
        hidden_states=hidden_states,
        block_residual=block_residual,
    )

    assert collective_shapes == [
        ("gather", torch.Size([2, 2])),
        ("reduce_scatter", torch.Size([3, 2])),
    ]
    assert output.shape == torch.Size([2, 2])
    assert returned_residual.shape == torch.Size([2, 1, 2])


def test_kimi_model_allocates_attention_residual_after_sp_shard(monkeypatch):
    class RecordingLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.residual_shape = None

        def forward(self, *, positions, hidden_states, residual):
            self.residual_shape = residual.shape
            return hidden_states, residual

    model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(attn_res_block_size=12)
    model.start_layer = 0
    model.end_layer = 1
    layer = RecordingLayer()
    model.layers = nn.ModuleList([layer])
    model.use_sequence_parallel = True
    model.aux_hidden_state_layers = set()
    model.output_attn_res_proj = object()
    model.output_attn_res_norm = object()
    model._maybe_add_hidden_state = MethodType(
        lambda self, states, *_args: states,
        model,
    )

    monkeypatch.setattr(
        kimi_k3,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        kimi_k3,
        "sp_shard",
        lambda hidden_states: torch.nn.functional.pad(hidden_states, (0, 0, 0, 1))[:2],
    )
    monkeypatch.setattr(
        kimi_k3,
        "sp_all_gather",
        lambda hidden_states: torch.cat((hidden_states, hidden_states), dim=0),
    )
    monkeypatch.setattr(
        kimi_k3,
        "_apply_ascend_attn_res",
        lambda hidden_states, *_args, **_kwargs: hidden_states,
    )

    output = model(
        input_ids=None,
        positions=torch.arange(3),
        intermediate_tensors=None,
        inputs_embeds=torch.arange(6, dtype=torch.float32).view(3, 2),
    )

    assert layer.residual_shape == torch.Size([2, 1, 2])
    assert output.shape == torch.Size([3, 2])


def test_kimi_model_selects_materialized_or_raw_dspark_aux_stream(monkeypatch):
    class Marker(nn.Module):
        def __init__(self, value: int) -> None:
            super().__init__()
            self.value = value

    class RecordingLayer(nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self.prev_valid_blocks = layer_idx
            self.self_attention_res_proj = Marker(layer_idx)
            self.self_attention_res_norm = nn.Identity()

        def forward(self, *, positions, hidden_states, residual):
            del positions
            materialized = kimi_k3._apply_ascend_attn_res(
                hidden_states,
                residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.prev_valid_blocks,
            )
            return materialized + 10, residual

    residual_calls: list[int] = []

    def fake_attn_res(prefix_sum, _residual, projection, _norm, num_valid_blocks):
        residual_calls.append(projection.value)
        return prefix_sum + 100 * num_valid_blocks

    monkeypatch.setattr(kimi_k3, "_apply_ascend_attn_res", fake_attn_res)
    monkeypatch.setattr(
        kimi_k3,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(attn_res_block_size=1)
    model.start_layer = 0
    model.end_layer = 2
    model.layers = nn.ModuleList([RecordingLayer(0), RecordingLayer(1)])
    model.use_sequence_parallel = False
    model.output_attn_res_proj = Marker(2)
    model.output_attn_res_norm = nn.Identity()
    model._set_aux_hidden_state_layers((1,))

    model.dspark_aux_capture_materialized = True
    _, materialized_aux = model(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.tensor([[1.0]]),
    )
    torch.testing.assert_close(materialized_aux[0], torch.tensor([[111.0]]))

    residual_calls.clear()
    model.dspark_aux_capture_materialized = False
    _, raw_aux = model(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.tensor([[1.0]]),
    )
    torch.testing.assert_close(raw_aux[0], torch.tensor([[11.0]]))


def test_kimi_dspark_aux_capture_mode_is_forwarded():
    causal_model = AscendKimiLinearForCausalLM.__new__(AscendKimiLinearForCausalLM)
    nn.Module.__init__(causal_model)
    causal_model.model = SimpleNamespace(dspark_aux_capture_materialized=False)

    causal_model.set_dspark_aux_capture_materialized(True)

    assert causal_model.model.dspark_aux_capture_materialized is True

    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.language_model = MagicMock()

    wrapper.set_dspark_aux_capture_materialized(True)

    wrapper.language_model.set_dspark_aux_capture_materialized.assert_called_once_with(True)


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


def test_projector_creates_rotation_only_when_enabled(monkeypatch):
    def fake_upstream_init(self, *_args, **_kwargs):
        nn.Module.__init__(self)

    monkeypatch.setattr(
        kimi_k3.KimiK25MultiModalProjector,
        "__init__",
        fake_upstream_init,
    )
    rotation = nn.Linear(1, 1, bias=False)
    rotation_factory = MagicMock(return_value=rotation)
    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", rotation_factory)
    config = SimpleNamespace(text_hidden_size=16)

    plain_projector = AscendKimiK3MultiModalProjector(config, prefix="mm_projector")

    assert plain_projector.rot_proj is None
    rotation_factory.assert_not_called()

    rotated_projector = AscendKimiK3MultiModalProjector(
        config,
        prefix="mm_projector",
        enable_rotation=True,
    )

    assert rotated_projector.rot_proj is rotation
    rotation_factory.assert_called_once_with(
        16,
        16,
        bias=False,
        quant_config=None,
        prefix="mm_projector.rot_proj",
    )


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
        def __init__(self, loaded_model, **kwargs):
            assert loaded_model is model
            if vllm_version_is("0.27.1"):
                assert kwargs["skip_substrs"] == list(model.checkpoint_skip_substrs)
            else:
                assert kwargs == {}

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
    model.target_model_path = "/target"
    model.model = SimpleNamespace(embed_tokens=object())
    model.lm_head = object()
    source_weights = [
        ("context_proj.weight", torch.ones(2, 4)),
        ("context_norm.weight", torch.ones(2)),
    ]
    rotated_weight = torch.full((2, 4), 2.0)
    seen_weights: list[tuple[str, torch.Tensor]] = []

    class CapturingLoader:
        def __init__(self, loaded_model, **kwargs):
            assert loaded_model is model
            if vllm_version_is("0.27.1"):
                assert kwargs["skip_substrs"] == list(model.checkpoint_skip_substrs)
            else:
                assert kwargs == {}

        def load_weights(self, weights, *, mapper):
            assert mapper is model.hf_to_vllm_mapper
            seen_weights.extend(weights)
            return {name for name, _ in seen_weights}

    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.AutoWeightsLoader",
        CapturingLoader,
    )
    rotation = torch.eye(4)
    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.get_rotation_matrix",
        lambda path: rotation if path == model.rotation_path else None,
    )
    process_weight = MagicMock(return_value=rotated_weight)
    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.process_weight",
        process_weight,
    )
    load_target_layer = MagicMock()
    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.load_quarot_target_layer",
        load_target_layer,
    )

    model.load_weights(iter(source_weights))

    process_weight.assert_called_once()
    torch.testing.assert_close(process_weight.call_args.args[0], source_weights[0][1])
    torch.testing.assert_close(process_weight.call_args.args[1], rotation)
    assert load_target_layer.call_count == 2
    assert load_target_layer.call_args_list[0].args[:2] == (
        model.model.embed_tokens,
        model.target_model_path,
    )
    assert load_target_layer.call_args_list[1].args[:2] == (
        model.lm_head,
        model.target_model_path,
    )
    assert model.has_own_embed_tokens
    assert model.has_own_lm_head
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
