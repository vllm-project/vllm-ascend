# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the production backbone loop with small CPU layers and real Engram gates."""

import ast
from functools import partial
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tests.ut.models.test_engram_elastic_executor import executor_env as executor_env
from tests.ut.models.test_engram_elastic_ops import native_ops as native_ops
from tests.ut.models.test_engram_elastic_ops import register_cpu_ops


def backbone_forward():
    # Loading just the method avoids importing unrelated attention operators;
    # its complete body is executed, without duplicating a model implementation.
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    model = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Model"
    )
    forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    namespace = {"torch": torch, "is_forward_context_available": lambda: False}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["forward"]


@pytest.mark.parametrize("first_layer", [0, 1])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_backbone_loop_matches_serial_model_for_every_request(native_ops, first_layer, overlap, compiled):
    executor, original, _, _ = native_ops
    early, late = original[1], original[14]
    early.layer_id = first_layer
    layers = {first_layer: early, 14: late}
    executor.layer_ids = tuple(layers)
    executor.engram_layers = layers
    executor.next_layer_ids = {first_layer: 14}
    executor._layer_ready = {first_layer: executor._layer_ready[1], 14: executor._layer_ready[14]}
    for layer in layers.values():
        layer.embed_tokens = lambda ids: ids.bfloat16()

    class Backbone(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.layer_idx, self.engram = index, layers.get(index)

        def forward(self, positions, hidden, pre_mix, residual, input_ids=None):
            return hidden + (self.layer_idx + 1) / 32, pre_mix

        def hc_collapse(self, hidden, pre_mix):
            return hidden.mean(1)

    def begin(ids, positions, metadata):
        history = torch.zeros((1, 2), dtype=torch.int32)
        if not overlap:
            return ids[:, None, None].expand(-1, 2, 2).int().clone(), ids >= 0
        return torch.ops.vllm.engram_elastic_begin(
            ids,
            positions,
            torch.tensor([0, ids.numel()]),
            history,
            torch.zeros_like(ids),
            history,
            torch.tensor([ids.numel()]),
            ids < 0,
            history.bool(),
            early.layer_name,
            2,
            2,
        )

    model = SimpleNamespace(
        has_engram=True,
        config=SimpleNamespace(engram_layer_ids=tuple(layers)),
        _engram_elastic_runtime=SimpleNamespace(executor=executor if overlap else None, begin=begin),
        embed_input_ids=lambda ids: ids[:, None].bfloat16().expand(-1, 4),
        shared_attention_state=SimpleNamespace(reset=lambda: None),
        hc_mult=2,
        aux_hidden_state_layers=(),
        _mtp_hidden_buffer=None,
        layers=[Backbone(index) for index in range(15)],
        norm=torch.nn.Identity(),
    )
    forward = MethodType(backbone_forward(), model)
    library = register_cpu_ops()
    if compiled:
        forward = torch.compile(forward, backend="eager", fullgraph=True)
    for ids in (torch.tensor([2, -1, 4]), torch.tensor([7, 3, -1])):
        actual = forward(ids, torch.arange(3), None, engram_elastic_inputs={"staged": True})
        expected = model.embed_input_ids(ids).unsqueeze(1).repeat(1, 2, 1)
        for index in range(15):
            if index in layers:
                layer = layers[index]
                kv = layer.precompute_kv(ids[:, None].expand(-1, 2).bfloat16())
                expected = layer.apply_precomputed(expected, *kv, ids >= 0)
            expected = expected + (index + 1) / 32
        torch.testing.assert_close(actual, expected.mean(1), rtol=0, atol=0)
        executor.release()
        assert not executor._pending_fetches
    del library


def runner_method(name, namespace):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def test_runner_snapshots_placeholder_barriers_before_sanitizing():
    method = runner_method("_sanitize_placeholder_input_ids_for_forward", {"torch": torch, "PLACEHOLDER_TOKEN_ID": -1})
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(engram_config=SimpleNamespace(enabled=True, backend="elastic")),
        input_ids=SimpleNamespace(gpu=torch.tensor([7, -1, 8, -1])),
    )
    method(runner, SimpleNamespace(scheduled_spec_decode_tokens={"req": [-1]}), 4)
    assert runner.input_ids.gpu.tolist() == [7, 0, 8, 0]
    assert runner._engram_dead_input_mask.tolist() == [False, True, False, True]
    pointer = runner._engram_dead_input_mask.data_ptr()
    runner.input_ids.gpu.fill_(9)
    method(runner, SimpleNamespace(scheduled_spec_decode_tokens={}), 4)
    assert pointer == runner._engram_dead_input_mask.data_ptr() and not runner._engram_dead_input_mask.any()


def test_runner_retires_device_metadata_even_if_engram_drain_fails():
    from vllm.config.compilation import CUDAGraphMode

    class Model:
        def __call__(self, **kwargs):
            return torch.zeros(1)

        def retire_engram_lookups(self, **kwargs):
            raise RuntimeError("drain failed")

    metadata = SimpleNamespace(submission_in_flight=True, release=Mock())
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE, device_metadata_executor=metadata)
    namespace = {
        "torch": torch,
        "Any": Any,
        "IntermediateTensors": object,
        "partial": partial,
        "CUDAGraphMode": CUDAGraphMode,
        "get_forward_context": lambda: context,
    }
    method = runner_method("_model_forward", namespace)
    runner = SimpleNamespace(model=Model(), enable_enpu=False, _update_full_graph_params_if_needed=lambda *args: None)
    with pytest.raises(RuntimeError, match="drain failed"):
        method(runner, 1, input_ids=torch.tensor([1]), positions=torch.tensor([0]))
    metadata.release.assert_called_once()
