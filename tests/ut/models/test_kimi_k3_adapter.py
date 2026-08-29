# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm_ascend.models import kimi_k3
from vllm_ascend.models.kimi_k3 import (
    AscendKimiLinearForCausalLM,
    AscendKimiK3MultiModalProjector,
    AscendKimiLinearModel,
)
from vllm_ascend.models.kimi_k3_dspark import (
    AscendK3DSparkForCausalLM,
)


@pytest.mark.parametrize(
    ("configured_value", "expected"),
    [("0", 8), ("1", 1), ("8", 8)],
)
def test_kimi_num_loaded_layers(monkeypatch, configured_value, expected):
    monkeypatch.setenv("VLLM_ASCEND_KIMI_K3_MAX_LOADED_LAYERS", configured_value)

    assert kimi_k3._get_kimi_k3_num_loaded_layers(8) == expected


@pytest.mark.parametrize("configured_value", ["-1", "9", "not-an-integer"])
def test_kimi_num_loaded_layers_rejects_invalid_values(monkeypatch, configured_value):
    monkeypatch.setenv("VLLM_ASCEND_KIMI_K3_MAX_LOADED_LAYERS", configured_value)

    with pytest.raises(ValueError, match="VLLM_ASCEND_KIMI_K3_MAX_LOADED_LAYERS"):
        kimi_k3._get_kimi_k3_num_loaded_layers(8)


@pytest.mark.parametrize(
    ("weight_name", "expected"),
    [
        ("layers.0.self_attn.q_proj.weight", 0),
        ("model.layers.12.mlp.down_proj.weight", 12),
        ("language_model.model.layers.3.router.weight", 3),
        ("model.embed_tokens.weight", None),
        ("model.layers.invalid.weight", None),
    ],
)
def test_kimi_decoder_layer_idx_from_weight_name(weight_name, expected):
    assert kimi_k3._get_decoder_layer_idx_from_weight_name(weight_name) == expected


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
    model.num_loaded_layers = 1
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
        ("layers.1.router.weight", torch.full((1, 4), 9.0)),
        ("layers.0.self_attn.g_proj.weight", torch.full((1,), 1.0)),
        ("layers.0.self_attn.f_a_proj.weight", torch.full((1,), 2.0)),
        ("layers.0.self_attn.b_proj.weight", torch.full((1,), 3.0)),
        ("layers.0.self_attn.o_proj.weight", torch.full((1,), 4.0)),
    ]

    loaded = model.load_weights(iter(source_weights))

    assert remaining[0] == source_weights[0]
    assert remaining[-1] == source_weights[-1]
    assert all(name != "layers.1.router.weight" for name, *_ in remaining)
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


def test_reduced_kimi_remaps_materialized_dspark_aux_layers(monkeypatch):
    model = AscendKimiLinearForCausalLM.__new__(AscendKimiLinearForCausalLM)
    captured_layers = []
    model.model = SimpleNamespace(
        num_loaded_layers=2,
        config=SimpleNamespace(num_hidden_layers=4),
        aux_hidden_state_layers=(1, 3),
        _set_aux_hidden_state_layers=lambda layers: captured_layers.append(layers),
    )
    monkeypatch.setattr(kimi_k3.logger, "warning_once", lambda *_args: None)

    model.set_dspark_aux_capture_materialized(True)

    assert model.model.dspark_aux_capture_materialized is True
    assert captured_layers == [(0, 1)]


def test_reduced_kimi_rejects_too_many_materialized_dspark_aux_layers():
    model = AscendKimiLinearForCausalLM.__new__(AscendKimiLinearForCausalLM)
    model.model = SimpleNamespace(
        num_loaded_layers=1,
        config=SimpleNamespace(num_hidden_layers=4),
        aux_hidden_state_layers=(0, 1),
    )

    with pytest.raises(ValueError, match="cannot provide 2 distinct DSpark"):
        model.set_dspark_aux_capture_materialized(True)


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
