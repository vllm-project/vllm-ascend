# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.sequence import IntermediateTensors

from vllm_ascend.patch.worker import patch_qwen3_5


@pytest.mark.skipif(
    not hasattr(patch_qwen3_5, "_gdn_init_with_packed_weight"),
    reason="The 310P branch intentionally does not install packed GDN weights.",
)
def test_standard_gdn_constructor_registers_packed_weight_after_base_init():
    def fake_original_init(layer, *args, **kwargs):
        del args, kwargs
        torch.nn.Module.__init__(layer)
        layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
        layer.conv1d = torch.nn.Module()
        layer.conv1d.weight = torch.nn.Parameter(torch.empty(6, 1, 4))
        layer.conv1d.quant_method = SimpleNamespace(process_weights_after_loading=lambda _layer: None)

    with patch.object(patch_qwen3_5, "_GDN_ORIGINAL_INIT", fake_original_init):
        layer = object.__new__(patch_qwen3_5._GDN_PATCH_TARGET)
        patch_qwen3_5._gdn_init_with_packed_weight(layer)

    packed = layer.conv1d.get_parameter("ascend_conv1d_weight")
    assert isinstance(packed, torch.nn.Parameter)
    assert packed.shape == (4, 6)
    assert packed.dtype == torch.bfloat16
    assert not packed.requires_grad


def test_qwen3_5_text_attention_uses_standard_rope():
    attention = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5_moe_text"),
        rotary_emb=SimpleNamespace(),
    )
    assert not patch_qwen3_5._uses_multimodal_rope(attention)


def test_qwen3_5_multimodal_attention_uses_mrope():
    attention = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5_moe"),
        rotary_emb=SimpleNamespace(mrope_section=[11, 11, 10]),
    )
    assert patch_qwen3_5._uses_multimodal_rope(attention)


@pytest.mark.parametrize("layer_type", ["linear_attention", "full_attention"])
def test_qwen3_5_dense_sp_disables_projection_reductions(layer_type):
    def projection(name):
        return SimpleNamespace(
            prefix=name,
            custom_op=None,
            reduce_results=True,
        )

    layer = SimpleNamespace(
        layer_type=layer_type,
        use_attn_reduce_scatter_for_moe=False,
        mlp=SimpleNamespace(down_proj=projection("mlp.down_proj")),
        linear_attn=SimpleNamespace(out_proj=projection("linear.out_proj")),
        self_attn=SimpleNamespace(o_proj=projection("attn.o_proj")),
    )

    patch_qwen3_5._configure_dense_sequence_parallel(layer)

    assert layer.use_attn_reduce_scatter_for_moe
    assert not layer.mlp.down_proj.reduce_results
    output_projection = layer.linear_attn.out_proj if layer_type == "linear_attention" else layer.self_attn.o_proj
    assert not output_projection.reduce_results


@pytest.mark.parametrize("layer_type", ["linear_attention", "full_attention"])
def test_qwen3_5_dense_sp_keeps_residuals_sequence_sharded(layer_type):
    calls = []
    local_hidden = torch.tensor([[1.0, 2.0]])
    local_residual = torch.tensor([[3.0, 4.0]])
    full_hidden = torch.tensor([[1.0, 2.0], [5.0, 6.0]])

    def gather(value):
        calls.append(("gather", value.shape[0]))
        return full_hidden if len([c for c in calls if c[0] == "gather"]) == 1 else value.repeat(2, 1)

    def reduce_scatter(value):
        calls.append(("reduce_scatter", value.shape[0]))
        return value[:1]

    def attention(**kwargs):
        calls.append(("attention", kwargs["hidden_states"].shape[0]))
        return kwargs["hidden_states"] + 10

    def mlp(value):
        calls.append(("mlp", value.shape[0]))
        return value + 20

    layer = SimpleNamespace(
        _use_dense_sequence_parallel=True,
        layer_type=layer_type,
        layer_scale=False,
        input_layernorm=MagicMock(return_value=(local_hidden + 1, local_residual + 1)),
        post_attention_layernorm=MagicMock(side_effect=lambda hidden, residual: (hidden + 1, residual + 1)),
        linear_attn=MagicMock(side_effect=lambda **kwargs: attention(**kwargs)),
        self_attn=MagicMock(side_effect=lambda **kwargs: attention(**kwargs)),
        mlp=MagicMock(side_effect=mlp),
    )

    with (
        patch.object(patch_qwen3_5, "sp_all_gather", side_effect=gather),
        patch.object(
            patch_qwen3_5,
            "sp_reduce_scatter",
            side_effect=reduce_scatter,
        ),
    ):
        output, residual = patch_qwen3_5.AscendQwen3_5DecoderLayer.forward(
            layer,
            hidden_states=local_hidden,
            residual=local_residual,
            positions=torch.tensor([0, 1]),
        )

    assert output.shape == local_hidden.shape
    assert residual.shape == local_residual.shape
    assert calls == [
        ("gather", 1),
        ("attention", 2),
        ("reduce_scatter", 2),
        ("gather", 1),
        ("mlp", 2),
        ("reduce_scatter", 2),
    ]


def test_qwen3_5_decoder_factories_use_ascend_subclass():
    assert patch_qwen3_5._UpstreamQwen3_5DecoderLayer is not patch_qwen3_5.AscendQwen3_5DecoderLayer
    assert issubclass(patch_qwen3_5.AscendQwen3_5DecoderLayer, patch_qwen3_5._UpstreamQwen3_5DecoderLayer)
    assert patch_qwen3_5._qwen3_5.Qwen3_5DecoderLayer is patch_qwen3_5.AscendQwen3_5DecoderLayer
    if patch_qwen3_5.Qwen3_5MultiTokenPredictor is not None:
        assert patch_qwen3_5._qwen3_5_mtp.Qwen3_5DecoderLayer is patch_qwen3_5.AscendQwen3_5DecoderLayer


@pytest.mark.skipif(
    patch_qwen3_5.Qwen3_5MultiTokenPredictor is None,
    reason="Qwen3.5 MTP model is not available in this vLLM version.",
)
def test_qwen3_5_mtp_forward_uses_local_inputs_on_last_pp_rank():
    predictor = patch_qwen3_5.Qwen3_5MultiTokenPredictor.__new__(patch_qwen3_5.Qwen3_5MultiTokenPredictor)
    predictor.num_mtp_layers = 2
    predictor.embed_input_ids = MagicMock(return_value=torch.ones(2, 4))
    predictor.pre_fc_norm_embedding = MagicMock(side_effect=lambda x: x + 1)
    predictor.pre_fc_norm_hidden = MagicMock(side_effect=lambda x: x + 2)
    predictor.fc = MagicMock(side_effect=lambda x: x[:, :4] + x[:, 4:])
    layer0 = MagicMock(return_value=(torch.full((2, 4), 3.0), torch.full((2, 4), 4.0)))
    layer1 = MagicMock(return_value=(torch.full((2, 4), 5.0), torch.full((2, 4), 6.0)))
    layer1.use_attn_reduce_scatter_for_moe = False
    predictor.layers = [layer0, layer1]
    predictor.norm = MagicMock(return_value=(torch.full((2, 4), 7.0), None))

    with patch(
        "vllm_ascend.patch.worker.patch_qwen3_5.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=True),
    ):
        output = predictor.forward(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            hidden_states=torch.zeros(2, 4),
            intermediate_tensors=IntermediateTensors({"hidden_states": torch.full((2, 4), 99.0)}),
            spec_step_idx=3,
        )

    predictor.embed_input_ids.assert_called_once()
    layer1.assert_called_once()
    predictor.norm.assert_called_once()
    assert torch.equal(output, torch.full((2, 4), 7.0))


@pytest.mark.skipif(
    patch_qwen3_5.Qwen3_5MultiTokenPredictor is None,
    reason="Qwen3.5 MTP model is not available in this vLLM version.",
)
def test_qwen3_5_mtp_forward_preserves_dense_sp_boundaries():
    predictor = patch_qwen3_5.Qwen3_5MultiTokenPredictor.__new__(patch_qwen3_5.Qwen3_5MultiTokenPredictor)
    predictor.num_mtp_layers = 1
    predictor.embed_input_ids = MagicMock(return_value=torch.ones(2, 4))
    predictor.pre_fc_norm_embedding = MagicMock(side_effect=lambda x: x)
    predictor.pre_fc_norm_hidden = MagicMock(side_effect=lambda x: x)
    predictor.fc = MagicMock(side_effect=lambda x: x[:, :4])
    layer = MagicMock(return_value=(torch.full((1, 4), 3.0), torch.full((1, 4), 4.0)))
    layer.use_attn_reduce_scatter_for_moe = True
    predictor.layers = [layer]
    predictor.norm = MagicMock(return_value=(torch.full((1, 4), 5.0), None))

    with (
        patch(
            "vllm_ascend.patch.worker.patch_qwen3_5.get_pp_group",
            return_value=SimpleNamespace(is_last_rank=True),
        ),
        patch(
            "vllm.model_executor.models.utils.sequence_parallel_chunk",
            side_effect=lambda x: x[:1],
        ) as chunk,
        patch.object(
            patch_qwen3_5,
            "tensor_model_parallel_all_gather",
            side_effect=lambda x, dim: x.repeat(2, 1),
        ) as gather,
    ):
        output = predictor.forward(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            hidden_states=torch.zeros(2, 4),
        )

    chunk.assert_called_once()
    layer.assert_called_once()
    assert layer.call_args.kwargs["hidden_states"].shape == (1, 4)
    gather.assert_called_once()
    assert torch.equal(output, torch.full((2, 4), 5.0))


@pytest.mark.skipif(
    patch_qwen3_5.Qwen3_5MultiTokenPredictor is None,
    reason="Qwen3.5 MTP model is not available in this vLLM version.",
)
def test_qwen3_5_mtp_forward_returns_intermediate_tensors_on_non_last_pp_rank():
    predictor = patch_qwen3_5.Qwen3_5MultiTokenPredictor.__new__(patch_qwen3_5.Qwen3_5MultiTokenPredictor)
    predictor.num_mtp_layers = 1
    predictor.embed_input_ids = MagicMock(return_value=torch.ones(1, 4))
    predictor.pre_fc_norm_embedding = MagicMock(side_effect=lambda x: x)
    predictor.pre_fc_norm_hidden = MagicMock(side_effect=lambda x: x)
    predictor.fc = MagicMock(side_effect=lambda x: x[:, :4])
    predictor.layers = [MagicMock(return_value=(torch.full((1, 4), 3.0), torch.full((1, 4), 4.0)))]
    predictor.norm = MagicMock()

    with (
        patch(
            "vllm_ascend.patch.worker.patch_qwen3_5.get_pp_group",
            return_value=SimpleNamespace(is_last_rank=False),
        ),
        patch(
            "vllm.model_executor.models.utils.sequence_parallel_chunk",
            side_effect=lambda x: x,
        ),
        patch(
            "vllm_ascend.patch.worker.patch_qwen3_5.sequence_parallel_chunk",
            side_effect=lambda x: x,
            create=True,
        ),
    ):
        output = predictor.forward(
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
            hidden_states=torch.zeros(1, 4),
        )

    assert isinstance(output, IntermediateTensors)
    assert torch.equal(output["hidden_states"], torch.full((1, 4), 3.0))
    assert torch.equal(output["residual"], torch.full((1, 4), 4.0))
    predictor.norm.assert_not_called()
