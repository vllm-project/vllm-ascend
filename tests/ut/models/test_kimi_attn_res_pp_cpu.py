# SPDX-License-Identifier: Apache-2.0
"""Exercise the layer call contract and PP receive buffers without NPU imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.ut.models.test_kimi_attn_res_add_cpu import fused_reference


class IntermediateTensors:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        return self.tensors[key]


class UpstreamLayer(torch.nn.Module):
    def forward(self, positions, hidden_states, residual, **kwargs):
        return hidden_states, residual


class UpstreamModel(torch.nn.Module):
    def _maybe_add_hidden_state(self, values, idx, hidden, residual):
        if idx in self.aux_hidden_state_layers:
            values.append(hidden)
        return values

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        h = self.config.hidden_size
        b = (self.start_layer + self.config.attn_res_block_size - 1) // self.config.attn_res_block_size
        return IntermediateTensors(
            {
                "hidden_states": torch.empty(batch_size, h, dtype=dtype, device=device),
                "residual": torch.empty(batch_size, b, h, dtype=dtype, device=device),
            }
        )


def load_classes(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/kimi_k3.py"
    tree = ast.parse(path.read_text())
    layer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendKimiDecoderLayer")
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendKimiLinearModel")
    layer.bases = [ast.Name(id="UpstreamLayer", ctx=ast.Load())]
    model.bases = [ast.Name(id="UpstreamModel", ctx=ast.Load())]
    layer.body = [
        n
        for n in layer.body
        if isinstance(n, ast.FunctionDef) and n.name in {"forward", "prepare_attn_residual", "forward_attn_residual"}
    ]
    model.body = [
        n
        for n in model.body
        if isinstance(n, ast.FunctionDef) and n.name in {"forward", "make_empty_intermediate_tensors"}
    ]
    state = SimpleNamespace(is_first_rank=True, is_last_rank=True, prefill=False)
    scope = dict(
        torch=torch,
        UpstreamLayer=UpstreamLayer,
        UpstreamModel=UpstreamModel,
        IntermediateTensors=IntermediateTensors,
        cdiv=lambda x, y: (x + y - 1) // y,
        get_pp_group=lambda: state,
        _use_attn_res_prefill_kernel=lambda: state.prefill,
    )
    source = "from __future__ import annotations\n" + ast.unparse(ast.Module(body=[layer, model], type_ignores=[]))
    exec(compile(source, str(path), "exec"), scope)
    native = SimpleNamespace(fused=fused_reference, fused_prefill=fused_reference)
    monkeypatch.setattr(torch.ops._C_ascend, "attn_res_fwd", native, raising=False)
    return scope["AscendKimiDecoderLayer"], scope["AscendKimiLinearModel"], state


def test_actual_layer_call_forwards_residual_contract(monkeypatch):
    layer_cls, _, _ = load_classes(monkeypatch)
    layer = layer_cls()
    layer.use_attn_residuals = True
    layer.forward_attn_residual = lambda *args, **kwargs: kwargs
    prepared = object()
    assert layer(None, None, object(), prepared_attn_input=prepared, defer_mlp_add=True, optimize_prefill=True) == dict(
        prepared_attn_input=prepared, defer_mlp_add=True, optimize_prefill=True
    )
    layer.use_attn_residuals = False
    hidden, residual = object(), object()
    assert layer(None, hidden, residual) == (hidden, residual)


@pytest.mark.parametrize("tokens", [0, 1, 32])
@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("captures", [(), (0, 1, 2, 3, 4, 5), (1, 3, 4)])
def test_pp_transport_matches_pp1_including_boundary_captures(monkeypatch, tokens, materialized, prefill, captures):
    layer_cls, model_cls, state = load_classes(monkeypatch)
    state.prefill = prefill
    torch.manual_seed(9127)
    h, layers, block = 16, 5, 2
    weights = []
    for _ in range(layers):
        weights.append(
            [
                torch.randn(1, h).bfloat16(),
                torch.randn(h).bfloat16(),
                torch.randn(1, h).bfloat16(),
                torch.randn(h).bfloat16(),
            ]
        )
    final_proj, final_norm = torch.randn(1, h).bfloat16(), torch.randn(h).bfloat16()

    def make_model(start, end):
        model = model_cls()
        model.config = SimpleNamespace(attn_res_block_size=block, hidden_size=h, num_hidden_layers=layers)
        model.start_layer, model.end_layer = start, end
        model.use_sequence_parallel = False
        model.dspark_aux_capture_materialized = materialized
        model.aux_hidden_state_layers = captures
        model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(start)])
        for idx in range(start, end):
            layer = layer_cls()
            layer.use_attn_residuals = True
            layer.use_sequence_parallel = False
            layer.is_block_write_layer = idx % block == 0
            layer.block_write_idx = idx // block
            layer.prev_valid_blocks = (idx + block - 1) // block
            for attr, value in zip(
                ("self_attention_res_proj", "self_attention_res_norm", "mlp_res_proj", "mlp_res_norm"), weights[idx]
            ):
                setattr(layer, attr, SimpleNamespace(weight=value, variance_epsilon=1e-5))
            layer.input_layernorm = SimpleNamespace(weight=None, variance_epsilon=1e-5)
            layer.post_attention_layernorm = SimpleNamespace(weight=None, variance_epsilon=1e-5)
            layer.self_attn = lambda hidden_states, positions: hidden_states * 0.125
            layer.mlp = lambda value: value * 0.25
            model.layers.append(layer)
        model.output_attn_res_proj = SimpleNamespace(weight=final_proj)
        model.output_attn_res_norm = SimpleNamespace(weight=final_norm, variance_epsilon=1e-5)
        return model

    hidden = torch.randn(tokens, h).bfloat16()
    positions = torch.arange(tokens)
    state.is_first_rank = state.is_last_rank = True
    expected = make_model(0, layers)(None, positions, None, inputs_embeds=hidden)
    transmitted: Any = None
    for start, end in [(0, 2), (2, 3), (3, 5)]:
        model = make_model(start, end)
        state.is_first_rank, state.is_last_rank = start == 0, end == layers
        received = None
        if transmitted is not None:
            received = model.make_empty_intermediate_tensors(tokens, torch.bfloat16, torch.device("cpu"))
            assert received.tensors.keys() == transmitted.tensors.keys()
            for key, value in transmitted.tensors.items():
                assert received[key].shape == value.shape
                received[key].copy_(value)
        transmitted = model(None, positions, received, inputs_embeds=hidden if start == 0 else None)
    if captures:
        actual_hidden, actual_aux = transmitted
        expected_hidden, expected_aux = expected
        assert len(actual_aux) == len(captures)
        for actual, reference in zip(actual_aux, expected_aux):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
    else:
        torch.testing.assert_close(transmitted, expected, rtol=0, atol=0)
