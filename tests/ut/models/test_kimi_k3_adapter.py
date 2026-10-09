# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import torch
from safetensors.torch import save_file
from torch import nn

from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.attention.utils import mark_fused_preprocess_weights
from vllm_ascend.models import kimi_k3
from vllm_ascend.models.kimi_k3 import (
    AscendKimiK3MultiModalProjector,
    AscendKimiLinearModel,
)
from vllm_ascend.models.kimi_k3_dspark import (
    AscendK3DSparkForCausalLM,
    AscendK3DSparkModel,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod


def test_kimi_disabling_mlapo_refreshes_projection_nz_management():
    for fa_quant_layer in (False, True):
        impl = AscendMLAImpl.__new__(AscendMLAImpl)
        impl.enable_mlapo = True
        impl.fa_quant_layer = fa_quant_layer
        impl.support_fp8_attention = True
        scheme = AscendW8A8MXFP8DynamicLinearMethod.__new__(AscendW8A8MXFP8DynamicLinearMethod)
        scheme.group_size = 32
        projections = []
        for output_size, input_size in ((128, 256), (192, 64)):
            layer = nn.Module()
            layer.quant_method = SimpleNamespace(quant_method=scheme)
            layer.weight = nn.Parameter(torch.randn(output_size, input_size).to(torch.float8_e4m3fn), False)
            layer.weight_scale = nn.Parameter(
                torch.ones(output_size, input_size // scheme.group_size, dtype=torch.uint8), False
            )
            projections.append(layer)
        impl.fused_qkv_a_proj, impl.q_proj = projections
        mark_fused_preprocess_weights(impl)
        assert impl.fused_qkv_a_proj._fused_preprocess_managed
        with (
            patch.object(kimi_k3.UpstreamKimiMLAAttention, "__init__", lambda self, **kwargs: nn.Module.__init__(self)),
            patch.object(
                kimi_k3.AscendKimiMLAAttention,
                "_attention_layer",
                new_callable=PropertyMock,
                return_value=SimpleNamespace(impl=impl),
            ),
        ):
            kimi_k3.AscendKimiMLAAttention(
                config=SimpleNamespace(),
                hidden_size=256,
                num_heads=2,
                qk_nope_head_dim=64,
                qk_rope_head_dim=32,
                v_head_dim=128,
                q_lora_rank=64,
                kv_lora_rank=32,
                use_output_gate=False,
                use_rope=False,
                disable_mlapo=True,
            )
        assert not impl.enable_mlapo
        assert impl.fused_qkv_a_proj._fused_preprocess_managed == fa_quant_layer
        assert impl.q_proj._fused_preprocess_managed == fa_quant_layer
        with (
            patch("vllm_ascend.utils._should_trans_nz", return_value=True),
            patch("torch_npu.npu_format_cast", side_effect=lambda weight, fmt, **kwargs: weight.clone()) as cast,
        ):
            for layer in projections:
                scheme.process_weights_after_loading(layer)
        assert cast.call_count == (0 if fa_quant_layer else 2)


def test_kimi_moe_leaves_routed_input_transform_to_runner():
    moe = kimi_k3.AscendKimiMoE.__new__(kimi_k3.AscendKimiMoE)
    nn.Module.__init__(moe)
    hidden_states = torch.randn(4, 8)
    router_logits = torch.randn(4, 16)
    output = torch.randn(4, 8)
    moe.gate = MagicMock(return_value=(router_logits, None))
    moe.experts = MagicMock(return_value=output)

    result = moe.forward(hidden_states)

    moe.experts.assert_called_once()
    call_kwargs = moe.experts.call_args.kwargs
    torch.testing.assert_close(call_kwargs["hidden_states"], hidden_states)
    torch.testing.assert_close(call_kwargs["router_logits"], router_logits)
    torch.testing.assert_close(result, output)


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


def test_k3_dspark_context_writer_splits_fused_mla_cache(monkeypatch):
    model = AscendK3DSparkModel.__new__(AscendK3DSparkModel)
    nn.Module.__init__(model)

    fused_cache = torch.arange(36, dtype=torch.float32).reshape(2, 3, 1, 6)
    attn = SimpleNamespace(
        fused_qkv_a_proj=MagicMock(return_value=(torch.randn(3, 8),)),
        q_lora_rank=2,
        kv_cache=fused_cache,
        impl=SimpleNamespace(kv_lora_rank=4, exec_kv_prefill=MagicMock()),
    )
    model.layers = [SimpleNamespace(self_attn=attn)]
    monkeypatch.setattr(
        "vllm_ascend.models.kimi_k3_dspark.get_cos_and_sin_mla",
        lambda positions: (torch.ones(3, 2), torch.zeros(3, 2)),
    )

    model.precompute_and_store_context_kv(
        context_states=torch.randn(3, 8),
        context_positions=torch.tensor([0, 1, 2]),
        context_slot_mapping=torch.tensor([0, 1, 2]),
    )

    cache = attn.impl.exec_kv_prefill.call_args.args[3]
    assert isinstance(cache, tuple)
    assert cache[0].shape == (2, 3, 1, 4)
    assert cache[1].shape == (2, 3, 1, 2)
    assert cache[0].untyped_storage() is fused_cache.untyped_storage()
    assert cache[1].untyped_storage() is fused_cache.untyped_storage()
    torch.testing.assert_close(cache[0], fused_cache[..., :4])
    torch.testing.assert_close(cache[1], fused_cache[..., 4:])


def test_kimi_mixed_kda_gate_weights_use_upstream_packed_loader(monkeypatch):
    model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
    nn.Module.__init__(model)
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.fused_bfg_proj = nn.Module()
    packed_weight = nn.Parameter(torch.empty(6, 4))
    layer.self_attn.fused_bfg_proj.register_parameter("weight", packed_weight)
    layer.self_attn.fused_bfg_proj.register_parameter("f_a_weight", nn.Parameter(torch.empty(1)))
    layer.self_attn.fused_bfg_proj.register_parameter("f_b_weight", nn.Parameter(torch.empty(1)))
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
        ("layers.0.self_attn.f_b_proj.weight", torch.full((1,), 3.0)),
        ("layers.0.self_attn.b_proj.weight", torch.full((1,), 4.0)),
        ("layers.0.self_attn.o_proj.weight", torch.full((1,), 5.0)),
    ]

    loaded = model.load_weights(iter(source_weights))

    assert remaining[0] == source_weights[0]
    assert remaining[-1] == source_weights[-1]
    assert [name for name, _, _ in remaining[1:5]] == [
        "layers.0.self_attn.fused_bfg_proj.weight",
        "layers.0.self_attn.fused_bfg_proj.f_a_weight",
        "layers.0.self_attn.fused_bfg_proj.f_b_weight",
        "layers.0.self_attn.fused_bfg_proj.weight",
    ]
    assert [loaded_weight.item() for _, loaded_weight, _ in remaining[1:5]] == [1.0, 2.0, 3.0, 4.0]
    assert [kwargs["loaded_shard_id"] for _, _, kwargs in remaining[1:5]] == [2, None, None, 0]
    assert loaded == {
        "layers.0.self_attn.fused_bfg_proj.weight",
        "layers.0.self_attn.fused_bfg_proj.f_a_weight",
        "layers.0.self_attn.fused_bfg_proj.f_b_weight",
        "layers.0.router.weight",
        "layers.0.self_attn.o_proj.weight",
    }


def test_kimi_model_declares_fused_bfg_checkpoint_mapping():
    assert AscendKimiLinearModel.packed_modules_mapping["fused_bfg_proj"] == [
        "b_proj",
        "f_a_proj",
        "g_proj",
    ]


def test_kimi_dense_mlp_gathers_and_scatters_sequence_shards(monkeypatch):
    mlp = kimi_k3.AscendKimiMLP.__new__(kimi_k3.AscendKimiMLP)
    nn.Module.__init__(mlp)
    mlp.use_sequence_parallel = True
    calls = []

    def fake_all_gather(hidden_states):
        calls.append(("gather", hidden_states.clone()))
        return torch.cat((hidden_states, hidden_states + 10), dim=0)

    def fake_mlp_forward(_self, hidden_states):
        calls.append(("mlp", hidden_states.clone()))
        return hidden_states + 1

    def fake_reduce_scatter(hidden_states):
        calls.append(("reduce_scatter", hidden_states.clone()))
        return hidden_states.chunk(2, dim=0)[0]

    monkeypatch.setattr(kimi_k3, "sp_all_gather", fake_all_gather)
    monkeypatch.setattr(kimi_k3, "sp_reduce_scatter", fake_reduce_scatter)
    monkeypatch.setattr(kimi_k3.KimiMLP, "forward", fake_mlp_forward)

    output = mlp(torch.tensor([[1.0], [2.0]]))

    assert [name for name, _ in calls] == ["gather", "mlp", "reduce_scatter"]
    torch.testing.assert_close(calls[1][1], torch.tensor([[1.0], [2.0], [11.0], [12.0]]))
    torch.testing.assert_close(output, torch.tensor([[2.0], [3.0]]))


def test_kimi_attention_residual_stays_sequence_sharded(monkeypatch):
    class IdentityAttention(nn.Module):
        def forward(self, *, hidden_states, positions):
            del positions
            return hidden_states

    layer = kimi_k3.AscendKimiDecoderLayer.__new__(kimi_k3.AscendKimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = True
    layer.fuse_o_proj_mm_reduce_scatter = False
    layer.prev_valid_blocks = 0
    layer.is_block_write_layer = False
    layer.input_layernorm = nn.Identity()
    layer.post_attention_layernorm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-5)
    layer.mlp = nn.Identity()
    layer.self_attention_res_proj = object()
    layer.self_attention_res_norm = object()
    layer.mlp_res_proj = SimpleNamespace(weight=torch.ones(1, 2))
    layer.mlp_res_norm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-5)
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
    layer.prepare_attn_residual = MethodType(lambda self, prefix, bank, **kwargs: (prefix, prefix, prefix), layer)

    def fake_fused(prefix, addend, *_args, **_kwargs):
        raw = prefix if addend is None else prefix + addend
        return raw, raw, raw

    monkeypatch.setattr(torch.ops._C_ascend, "attn_res_fwd", fake_fused, raising=False)

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

        def prepare_attn_residual(self, prefix, bank, addend=None, **kwargs):
            raw = prefix if addend is None else prefix + addend
            return raw, raw, raw

        def forward(self, *, positions, hidden_states, residual, **kwargs):
            self.residual_shape = residual.shape
            return hidden_states, residual, None

    model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(attn_res_block_size=12)
    model.start_layer = 0
    model.end_layer = 1
    layer = RecordingLayer()
    model.layers = nn.ModuleList([layer])
    model.use_sequence_parallel = True
    model.aux_hidden_state_layers = set()
    model.output_attn_res_proj = SimpleNamespace(weight=torch.ones(1, 2))
    model.output_attn_res_norm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-5)
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
    monkeypatch.setattr(kimi_k3, "_use_attn_res_prefill_cache", lambda: False)

    def fake_fused(prefix, addend, *_args, **_kwargs):
        raw = prefix if addend is None else prefix + addend
        return raw, raw, raw

    monkeypatch.setattr(torch.ops._C_ascend, "attn_res_fwd", fake_fused, raising=False)

    output = model(
        input_ids=None,
        positions=torch.arange(3),
        intermediate_tensors=None,
        inputs_embeds=torch.arange(6, dtype=torch.float32).view(3, 2),
    )

    assert layer.residual_shape == torch.Size([2, 1, 2])
    assert output.shape == torch.Size([3, 2])


def test_kimi_model_selects_materialized_or_raw_dspark_aux_stream(monkeypatch):
    class RecordingLayer(nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self.prev_valid_blocks = layer_idx
            self.self_attention_res_proj = nn.Identity()
            self.self_attention_res_norm = nn.Identity()

        def prepare_attn_residual(self, prefix, bank, addend=None, **kwargs):
            raw = prefix if addend is None else prefix + addend
            return raw + 100 * self.prev_valid_blocks, raw, raw + 100 * self.prev_valid_blocks

        def forward(self, *, positions, hidden_states, residual, prepared_attn_input, **kwargs):
            del positions, hidden_states
            return prepared_attn_input[1] + 10, residual, None

    monkeypatch.setattr(kimi_k3, "_use_attn_res_prefill_cache", lambda: False)

    def fake_fused(prefix, addend, *_args, **_kwargs):
        raw = prefix if addend is None else prefix + addend
        return raw, raw, raw

    monkeypatch.setattr(torch.ops._C_ascend, "attn_res_fwd", fake_fused, raising=False)
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
    model.output_attn_res_proj = SimpleNamespace(weight=torch.ones(1, 1))
    model.output_attn_res_norm = SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-5)
    model._set_aux_hidden_state_layers((1,))

    model.dspark_aux_capture_materialized = True
    _, materialized_aux = model(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.tensor([[1.0]]),
    )
    torch.testing.assert_close(materialized_aux[0], torch.tensor([[111.0]]))

    model.dspark_aux_capture_materialized = False
    _, raw_aux = model(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.tensor([[1.0]]),
    )
    torch.testing.assert_close(raw_aux[0], torch.tensor([[11.0]]))


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


def test_k3_dspark_post_process_rotates_projection_and_target_boundaries(tmp_path, monkeypatch):
    model = AscendK3DSparkForCausalLM.__new__(AscendK3DSparkForCausalLM)
    nn.Module.__init__(model)
    model._owns_embed_tokens = False
    model.model = nn.Module()
    model.model.context_proj = nn.Linear(4, 2, bias=False)
    model.model.context_norm = nn.LayerNorm(2)
    model.model.embed_tokens = nn.Linear(2, 3, bias=False)
    model.lm_head = nn.Linear(2, 3, bias=False)
    model.rotation_path = tmp_path / "rotation.safetensors"
    model.target_model_path = tmp_path

    # A non-symmetric rotation distinguishes projection R from vocabulary R.T.
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    embed_weight = torch.arange(6, dtype=torch.float32).view(3, 2)
    head_weight = embed_weight + 10
    save_file({"global_rotation": rotation}, model.rotation_path)
    save_file(
        {
            "language_model.model.embed_tokens.weight": embed_weight,
            "language_model.lm_head.weight": head_weight,
        },
        tmp_path / "model.safetensors",
    )
    projection = torch.arange(8, dtype=torch.float32).view(2, 4)
    norm_weight = torch.tensor([2.0, 3.0])

    # Draft loading stays unrotated; post-processing uses the target configuration.
    model.load_weights(
        iter(
            [
                ("context_proj.weight", projection),
                ("context_norm.weight", norm_weight),
            ]
        )
    )

    torch.testing.assert_close(model.model.context_proj.weight, projection)

    def vocab_layer(vocab_size, hidden_size, params_dtype):
        layer = nn.Linear(hidden_size, vocab_size, bias=False, dtype=params_dtype)
        layer.quant_method = SimpleNamespace(process_weights_after_loading=lambda layer: None)
        return layer

    monkeypatch.setattr("vllm_ascend.models.qwen3_dspark.VocabParallelEmbedding", vocab_layer)
    monkeypatch.setattr("vllm_ascend.models.qwen3_dspark.ParallelLMHead", vocab_layer)
    config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path), hf_text_config=SimpleNamespace(vocab_size=3, hidden_size=2)),
        quant_config=SimpleNamespace(
            quant_description={"optional": {"quarot": {"rotation_map": {"global_rotation": "rotation.safetensors"}}}}
        ),
    )
    model.post_process(config)

    torch.testing.assert_close(
        model.model.context_proj.weight,
        projection @ torch.block_diag(rotation, rotation),
    )
    torch.testing.assert_close(model.model.context_norm.weight, norm_weight)
    torch.testing.assert_close(model.model.embed_tokens.weight, embed_weight @ rotation.T)
    torch.testing.assert_close(model.lm_head.weight, head_weight @ rotation.T)
    assert model.has_own_embed_tokens
    assert model.has_own_lm_head


def test_k3_dspark_embed_input_ids_merges_multimodal_embeddings():
    model = AscendK3DSparkForCausalLM.__new__(AscendK3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.model = SimpleNamespace(
        embed_input_ids=nn.Embedding.from_pretrained(torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])),
    )
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


def test_k3_dspark_pp_mapper_keeps_frozen_embed_but_drops_heads(monkeypatch):
    from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM as UpstreamK3DSpark

    from vllm_ascend.models import kimi_k3_dspark

    loader = MagicMock()
    monkeypatch.setattr(kimi_k3_dspark, "AutoWeightsLoader", lambda model: loader)
    model = AscendK3DSparkForCausalLM.__new__(AscendK3DSparkForCausalLM)
    nn.Module.__init__(model)
    model._owns_embed_tokens = True
    model.load_weights([])
    pp_mapper = loader.load_weights.call_args.kwargs["mapper"]
    upstream_mapper = UpstreamK3DSpark.hf_to_vllm_mapper

    # The frozen embedding copy must survive on PP stages that cannot alias
    # the target's stage-0 embedding.
    assert pp_mapper._map_name("embed_tokens.weight") == "model.embed_tokens.weight"
    assert upstream_mapper._map_name("embed_tokens.weight") is None

    # Everything else matches upstream: heads stay dropped, layers keep the
    # model. prefix and the stacked projections map identically.
    for name in ("lm_head.weight", "confidence_head.weight"):
        assert pp_mapper._map_name(name) is None
        assert upstream_mapper._map_name(name) is None
    for name in (
        "layers.0.self_attn.q_a_proj.weight",
        "layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "layers.0.mlp.gate_proj.weight",
        "context_proj.weight",
    ):
        assert pp_mapper._map_name(name) == upstream_mapper._map_name(name)
