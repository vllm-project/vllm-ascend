# SPDX-License-Identifier: Apache-2.0
"""Exercise Qwen4Exp feature capture without constructing accelerator layers."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from vllm.distributed import parallel_state
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3, supports_eagle3

MODEL_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp/model.py"


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.fixture
def model_module(monkeypatch):
    """Keep the real wrapper and Eagle interfaces; replace expensive dependencies."""

    class UpstreamModel(nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.config = vllm_config.model_config.hf_text_config
            self.start_layer = 0
            self.end_layer = self.config.num_hidden_layers
            self._dynamic_arg_dims = {"input_ids": 0, "query_start_loc": 0}
            self.upstream_forward = MagicMock(return_value=object())

        def forward(self, *args):
            return self.upstream_forward(*args)

    upstream = _module(
        "vllm.models.qwen4_exp.amd.model",
        Qwen4ExpModel=UpstreamModel,
        Qwen4ExpForCausalLM=type("UpstreamCausalLM", (nn.Module,), {}),
        Qwen4ExpForConditionalGeneration=type("UpstreamMultimodal", (nn.Module,), {}),
    )
    hc = _module("vllm.models.qwen4_exp.amd.hyperconnection")
    package = "vllm_ascend.models.qwen4_exp"
    replacements = {
        "vllm.models.qwen4_exp.amd": _module("vllm.models.qwen4_exp.amd", model=upstream, hyperconnection=hc),
        "vllm.models.qwen4_exp.amd.model": upstream,
        "vllm.models.qwen4_exp.amd.hyperconnection": hc,
        "vllm_ascend.ops.triton.qwen4_exp.hc": _module(
            "vllm_ascend.ops.triton.qwen4_exp.hc",
            **{
                name: MagicMock()
                for name in ("grouped_gemma_rmsnorm", "hc_combine", "hc_combine_norm", "hc_gate_mix", "hc_silu")
            },
        ),
        package: _module(package, __path__=[str(MODEL_PATH.parent)]),
        f"{package}.ple": _module(f"{package}.ple", patch_upstream_ple_short_conv=MagicMock()),
        f"{package}.qsa": _module(f"{package}.qsa", AscendQwen4ExpQSAAttention=object),
    }
    for name, module in replacements.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: False)
    name = f"{package}._dflash_test_model"
    spec = importlib.util.spec_from_file_location(name, MODEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    return module


class _Layer(nn.Module):
    """Distinct HC streams and pending outputs expose capture-order mistakes."""

    def __init__(self, layer_idx: int):
        super().__init__()
        self.hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8) + 100 * layer_idx
        self.block_output = torch.full((2, 2), 5.0 + layer_idx)
        self.injection = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).repeat(2, 1)
        self.completed = self.hidden + self.block_output.repeat(1, 4) * self.injection.repeat_interleave(2, dim=-1)
        self.mlp_hyper_connection = SimpleNamespace(combine=MagicMock(return_value=self.completed))
        self.inputs = None

    def forward(self, **kwargs):
        self.inputs = kwargs
        return self.hidden, self.block_output, self.injection


def _model(module, *, num_layers=3):
    config = SimpleNamespace(num_hidden_layers=num_layers, hc_count=4, hidden_size=2)
    model = module.AscendQwen4ExpModel(vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_text_config=config)))
    model.layers = nn.ModuleList(_Layer(i) for i in range(num_layers))
    model.embed_input_ids = MagicMock(return_value=torch.tensor([[1.0, 3.0], [2.0, 5.0]]))
    model._mtp_hidden_buffer = None
    multi_hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8) + 700
    sample_hidden = torch.tensor([[11.0, 13.0], [17.0, 19.0]])
    model.hyper_connection_mixer = SimpleNamespace(
        combine_and_mix=MagicMock(return_value=(multi_hidden, sample_hidden, None))
    )
    return model


@pytest.mark.parametrize("wrapper_name", ["AscendQwen4ExpForCausalLM", "AscendQwen4ExpForConditionalGeneration"])
def test_eagle_interface_sets_auxiliary_layers_on_backbone(model_module, wrapper_name):
    backbone = _model(model_module)
    wrapper = getattr(model_module, wrapper_name)()
    if wrapper_name.endswith("ForConditionalGeneration"):
        wrapper.language_model = SimpleNamespace(model=backbone)
    else:
        wrapper.model = backbone

    assert isinstance(backbone, EagleModelMixin)
    assert isinstance(wrapper, SupportsEagle3)
    assert supports_eagle3(wrapper)
    wrapper.set_aux_hidden_state_layers((3, 1, 3))
    assert backbone.aux_hidden_state_layers == (1, 3)
    assert not backbone.supports_aux_hidden_states_over_pp


def test_capture_uses_completed_layers_and_means_hc_streams(model_module):
    model = _model(model_module)
    model._set_aux_hidden_state_layers((3, 1))
    input_ids = torch.tensor([3, 5])
    positions = torch.tensor([0, 1])
    query_start_loc = torch.tensor([0, 2])
    ngram_context = torch.tensor([[7, 8]])

    output, captured = model(
        input_ids,
        positions,
        query_start_loc=query_start_loc,
        ngram_context=ngram_context,
    )

    assert output is model.hyper_connection_mixer.combine_and_mix.return_value[1]
    assert len(captured) == 2
    # Four HC streams have offsets [0, 2, 4, 6] and injection weights [1, 2, 3, 4].
    torch.testing.assert_close(captured[0], torch.tensor([[15.5, 16.5], [23.5, 24.5]]))
    torch.testing.assert_close(captured[1], torch.tensor([[220.5, 221.5], [228.5, 229.5]]))
    torch.testing.assert_close(model.layers[0].inputs["hidden_states"], model.embed_input_ids.return_value.repeat(1, 4))
    model.embed_input_ids.assert_called_once_with(input_ids)
    model.layers[1].mlp_hyper_connection.combine.assert_not_called()
    for layer_idx in (0, 2):
        layer = model.layers[layer_idx]
        layer.mlp_hyper_connection.combine.assert_called_once_with(layer.hidden, layer.block_output, layer.injection)
    for previous, following in zip(model.layers, model.layers[1:]):
        assert following.inputs["hidden_states"] is previous.hidden
        assert following.inputs["prev_block_output"] is previous.block_output
        assert following.inputs["prev_injection"] is previous.injection
    for layer in model.layers:
        assert layer.inputs["positions"] is positions
        assert layer.inputs["query_start_loc"] is query_start_loc
        assert layer.inputs["ngram_context"] is ngram_context
    model.hyper_connection_mixer.combine_and_mix.assert_called_once_with(
        model.layers[-1].hidden, model.layers[-1].block_output, model.layers[-1].injection
    )


def test_flash_next_checkpoint_target_layers_capture_five_features(model_module):
    model = _model(model_module, num_layers=48)
    # Checkpoint target_layer_ids are zero-based completed decoder layers.
    target_layer_ids = (5, 15, 25, 35, 45)
    model._set_aux_hidden_state_layers(tuple(layer_idx + 1 for layer_idx in target_layer_ids))

    _, captured = model(torch.tensor([3, 5]), torch.tensor([0, 1]))

    assert len(captured) == 5
    for layer_idx, feature in zip(target_layer_ids, captured):
        torch.testing.assert_close(feature, torch.tensor([[15.5, 16.5], [23.5, 24.5]]) + 102.5 * layer_idx)
    assert torch.cat(captured, dim=-1).shape == (2, 5 * model.config.hidden_size)


def test_inputs_embeds_and_deepstack_are_captured_after_addition(model_module):
    model = _model(model_module)
    model._set_aux_hidden_state_layers((1,))
    inputs_embeds = torch.tensor([[31.0, 37.0], [41.0, 43.0]])
    deepstack_embed = torch.tensor([[1000.0, 2000.0], [3000.0, 4000.0]])

    _, captured = model(
        None,
        torch.tensor([0, 1]),
        inputs_embeds=inputs_embeds,
        deepstack_input_embeds={"deepstack_input_embeds_0": deepstack_embed},
    )

    model.embed_input_ids.assert_not_called()
    torch.testing.assert_close(model.layers[0].inputs["hidden_states"], inputs_embeds.repeat(1, 4))
    torch.testing.assert_close(captured[0], torch.tensor([[1015.5, 2016.5], [3023.5, 4024.5]]))
    first_layer = model.layers[0]
    first_layer.mlp_hyper_connection.combine.assert_called_once_with(
        first_layer.hidden, first_layer.block_output, first_layer.injection
    )
    torch.testing.assert_close(
        model.layers[1].inputs["hidden_states"], model.layers[0].completed + deepstack_embed.repeat(1, 4)
    )
    assert model.layers[1].inputs["prev_block_output"] is None
    assert model.layers[1].inputs["prev_injection"] is None


@pytest.mark.parametrize("mtp_enabled", [False, True])
def test_empty_auxiliary_layers_preserve_upstream_forward(model_module, mtp_enabled):
    model = _model(model_module)
    if mtp_enabled:
        model._mtp_hidden_buffer = torch.full((4, 8), -1.0)
    model._set_aux_hidden_state_layers((1,))
    model._set_aux_hidden_state_layers(())
    arguments = (
        torch.tensor([3, 5]),
        torch.tensor([0, 1]),
        None,
        torch.randn(2, 2),
        torch.tensor([0, 2]),
        torch.tensor([[7, 8]]),
        {"deepstack_input_embeds_0": torch.ones(2, 2)},
    )

    assert model(*arguments) is model.upstream_forward.return_value
    model.upstream_forward.assert_called_once_with(*arguments)
    model.hyper_connection_mixer.combine_and_mix.assert_not_called()
    if mtp_enabled:
        torch.testing.assert_close(model._mtp_hidden_buffer, torch.full((4, 8), -1.0))


def test_auxiliary_capture_retains_mtp_multistream_buffer(model_module):
    model = _model(model_module)
    model._set_aux_hidden_state_layers((3,))
    model._mtp_hidden_buffer = torch.full((4, 8), -1.0)

    model(torch.tensor([3, 5]), torch.tensor([0, 1]))

    torch.testing.assert_close(
        model._mtp_hidden_buffer[:2], model.hyper_connection_mixer.combine_and_mix.return_value[0]
    )
    torch.testing.assert_close(model._mtp_hidden_buffer[2:], torch.full((2, 8), -1.0))


@pytest.mark.parametrize("layers", [(0,), (-1,), (4,), (1, 4)])
def test_invalid_auxiliary_boundaries_fail_before_forward(model_module, layers):
    model = _model(model_module)
    with pytest.raises(ValueError, match="num_hidden_layers"):
        model._set_aux_hidden_state_layers(layers)


def test_auxiliary_capture_requires_input_ids_or_embeds(model_module):
    model = _model(model_module)
    model._set_aux_hidden_state_layers((1,))
    with pytest.raises(ValueError, match="input_ids or inputs_embeds"):
        model(None, torch.tensor([0, 1]))


def test_preserves_upstream_dynamic_shape_annotations(model_module):
    assert _model(model_module)._dynamic_arg_dims == {"input_ids": 0}
