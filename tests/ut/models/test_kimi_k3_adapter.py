# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm_ascend.models import kimi_k3
from vllm_ascend.models.kimi_k3 import (
    AscendKimiK3MultiModalProjector,
    AscendKimiLinearModel,
)
from vllm_ascend.models.kimi_k3_dspark import (
    AscendK3DSparkForCausalLM,
)


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


@pytest.mark.parametrize(
    "sequence_parallel,request_fusion,hardware_supported",
    [(True, False, True), (True, True, True), (True, True, False), (False, True, True)],
)
def test_kimi_attention_residual_preserves_layout_and_one_reduction(
    monkeypatch, sequence_parallel, request_fusion, hardware_supported
):
    class IdentityAttention(nn.Module):
        def forward(self, *, hidden_states, positions):
            del positions
            if layer.fuse_o_proj_mm_reduce_scatter:
                collective_shapes.append(("mm_reduce_scatter", hidden_states.shape))
                return hidden_states.chunk(2, dim=0)[0]
            return hidden_states

    layer = kimi_k3.AscendKimiDecoderLayer.__new__(kimi_k3.AscendKimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = sequence_parallel
    layer.use_attn_residuals = True
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
    layer.self_attn.o_proj = nn.Linear(2, 2, bias=False)
    fused_factory = MagicMock(return_value=object())
    fused_factory.unsupported_reason.return_value = None
    monkeypatch.setattr(kimi_k3, "KimiOProjMMReduceScatterOp", fused_factory)
    monkeypatch.setattr(
        kimi_k3, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: hardware_supported)
    )
    monkeypatch.setattr(kimi_k3, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1))
    layer.fuse_o_proj_mm_reduce_scatter = request_fusion and layer._enable_o_proj_mm_reduce_scatter(
        SimpleNamespace(lora_config=None)
    )

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

    num_rows = 2 if sequence_parallel else 3
    hidden_states = torch.arange(num_rows * 2, dtype=torch.float32).view(num_rows, 2)
    block_residual = torch.zeros(num_rows, 1, 2)
    output, returned_residual = layer.forward_attn_residual(
        positions=torch.arange(3),
        hidden_states=hidden_states,
        block_residual=block_residual,
    )

    expected_collectives = (
        [
            ("gather", torch.Size([2, 2])),
            ("mm_reduce_scatter" if layer.fuse_o_proj_mm_reduce_scatter else "reduce_scatter", torch.Size([3, 2])),
        ]
        if sequence_parallel
        else []
    )
    assert collective_shapes == expected_collectives
    assert fused_factory.call_count == int(request_fusion and hardware_supported and sequence_parallel)
    assert output.shape == hidden_states.shape
    torch.testing.assert_close(output, hidden_states * 4)
    assert returned_residual.shape == block_residual.shape


@pytest.mark.parametrize("is_mla", [False, True])
def test_kimi_fused_o_proj_preserves_projection_and_sets_mla_output_shard(monkeypatch, is_mla):
    layer = kimi_k3.AscendKimiDecoderLayer.__new__(kimi_k3.AscendKimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = True
    layer.use_attn_residuals = True
    if is_mla:
        attention = kimi_k3.AscendKimiMLAAttention.__new__(kimi_k3.AscendKimiMLAAttention)
        nn.Module.__init__(attention)
        attention.mla_attn = nn.Module()
        attention.mla_attn.output_token_shard_size = 1
    else:
        attention = nn.Module()
    attention.o_proj = nn.Linear(256, 8, bias=False, dtype=torch.bfloat16)
    attention.o_proj.tp_size = 8
    layer.self_attn = attention
    original_weight = attention.o_proj.weight
    original_keys = list(layer.state_dict())
    fused_op = object()
    fused_factory = MagicMock(return_value=fused_op)
    fused_factory.unsupported_reason.return_value = None
    monkeypatch.setattr(kimi_k3, "KimiOProjMMReduceScatterOp", fused_factory)
    monkeypatch.setattr(kimi_k3, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: True))
    monkeypatch.setattr(kimi_k3, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1))

    assert layer._enable_o_proj_mm_reduce_scatter(SimpleNamespace(lora_config=None)) is True

    assert attention.o_proj.custom_op is fused_op
    assert attention.o_proj.weight is original_weight
    assert list(layer.state_dict()) == original_keys
    if is_mla:
        assert attention.mla_attn.output_token_shard_size == 8


@pytest.mark.parametrize(
    "sequence_parallel,attn_residuals,hardware_supported,nz_mode,lora_config",
    [
        (False, True, True, 1, None),
        (True, False, True, 1, None),
        (True, True, False, 1, None),
        (True, True, True, 2, None),
        (True, True, True, 1, object()),
    ],
)
def test_kimi_fused_o_proj_keeps_original_operator_for_unsupported_configuration(
    monkeypatch, sequence_parallel, attn_residuals, hardware_supported, nz_mode, lora_config
):
    layer = kimi_k3.AscendKimiDecoderLayer.__new__(kimi_k3.AscendKimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = sequence_parallel
    layer.use_attn_residuals = attn_residuals
    attention = kimi_k3.AscendKimiMLAAttention.__new__(kimi_k3.AscendKimiMLAAttention)
    nn.Module.__init__(attention)
    attention.o_proj = nn.Linear(2, 2, bias=False)
    original_op = object()
    attention.o_proj.custom_op = original_op
    attention.o_proj.reduce_results = not sequence_parallel
    attention.mla_attn = SimpleNamespace(output_token_shard_size=1)
    layer.self_attn = attention
    original_keys = list(layer.state_dict())
    fused_factory = MagicMock(side_effect=AssertionError("unsupported configuration must not initialize fusion"))
    monkeypatch.setattr(kimi_k3, "KimiOProjMMReduceScatterOp", fused_factory)
    monkeypatch.setattr(
        kimi_k3, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: hardware_supported)
    )
    monkeypatch.setattr(kimi_k3, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=nz_mode))

    assert layer._enable_o_proj_mm_reduce_scatter(SimpleNamespace(lora_config=lora_config)) is False
    fused_factory.assert_not_called()
    assert attention.o_proj.custom_op is original_op
    assert attention.o_proj.reduce_results is (not sequence_parallel)
    assert attention.mla_attn.output_token_shard_size == 1
    assert list(layer.state_dict()) == original_keys


@pytest.mark.parametrize("incompatible", ["fp16", "fp32", "quantized", "custom_op", "bias", "api", "tp", "local_k"])
def test_kimi_fused_o_proj_keeps_incompatible_projection(monkeypatch, incompatible):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_ascend.ops import linear_op

    layer = kimi_k3.AscendKimiDecoderLayer.__new__(kimi_k3.AscendKimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = True
    layer.use_attn_residuals = True
    attention = kimi_k3.AscendKimiMLAAttention.__new__(kimi_k3.AscendKimiMLAAttention)
    nn.Module.__init__(attention)
    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(incompatible, torch.bfloat16)
    attention.o_proj = nn.Linear(
        255 if incompatible == "local_k" else 256, 32, bias=incompatible == "bias", dtype=dtype
    )
    attention.o_proj.custom_op = object() if incompatible == "custom_op" else None
    attention.o_proj.quant_method = object() if incompatible == "quantized" else UnquantizedLinearMethod()
    attention.o_proj.reduce_results = False
    attention.mla_attn = SimpleNamespace(output_token_shard_size=1)
    layer.self_attn = attention
    original_op = attention.o_proj.custom_op
    original_weight = attention.o_proj.weight
    monkeypatch.setattr(kimi_k3, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: True))
    monkeypatch.setattr(kimi_k3, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1))
    monkeypatch.setattr(linear_op, "get_tp_group", lambda: SimpleNamespace(world_size=3 if incompatible == "tp" else 8))
    monkeypatch.setattr(
        linear_op.torch_npu,
        "npu_quant_mm_reduce_scatter",
        None if incompatible == "api" else MagicMock(),
        raising=False,
    )
    fused_init = MagicMock(side_effect=AssertionError("unsupported projection must not initialize fusion"))
    monkeypatch.setattr(kimi_k3.KimiOProjMMReduceScatterOp, "__init__", fused_init)

    assert layer._enable_o_proj_mm_reduce_scatter(SimpleNamespace(lora_config=None)) is False

    fused_init.assert_not_called()
    assert attention.o_proj.custom_op is original_op
    assert attention.o_proj.weight is original_weight
    assert attention.o_proj.reduce_results is False
    assert attention.mla_attn.output_token_shard_size == 1


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
    class RecordingLayer(nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self.prev_valid_blocks = layer_idx
            self.self_attention_res_proj = nn.Identity()
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

    def fake_attn_res(prefix_sum, _residual, _projection, _norm, num_valid_blocks):
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
    model.output_attn_res_proj = nn.Identity()
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


def test_k3_dspark_load_weights_rotates_projection_and_target_boundaries(tmp_path):
    model = AscendK3DSparkForCausalLM.__new__(AscendK3DSparkForCausalLM)
    nn.Module.__init__(model)
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

    # Load the draft projection plus vocabulary weights from the target checkpoint.
    model.load_weights(
        iter(
            [
                ("context_proj.weight", projection),
                ("context_norm.weight", norm_weight),
            ]
        )
    )

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
