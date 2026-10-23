# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Exercise DFlash2's real compilation decorators with a CPU Dynamo backend.

Attention construction and linear layers use CPU stand-ins. The decorators are
real, and fullgraph=True rejects graph breaks in the production selector and
grouped-convolution calculations. This does not test the NPU compiler backend.
"""

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode, set_current_vllm_config
from vllm.config.compilation import DynamicShapesType

MODEL_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/models/qwen3_dflash2.py"


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.fixture
def model_module(monkeypatch):
    # Keep vLLM's real compilation decorators, including the inherited parent
    # wrapper; constructing actual attention layers would require an NPU.
    @support_torch_compile
    class DraftModel(nn.Module):
        def __init__(self, *, vllm_config, start_layer_id=0, prefix=""):
            super().__init__()
            self.config = vllm_config.speculative_config.draft_model_config.hf_config
            self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)

        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, inputs_embeds: torch.Tensor | None = None):
            return self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds

        def embed_input_ids(self, input_ids):
            return self.embed_tokens(input_ids)

    class Linear(nn.Linear):
        def __init__(self, input_size, output_size, *, bias=False, params_dtype=None, **kwargs):
            super().__init__(input_size, output_size, bias=bias, dtype=params_dtype)

    replacements = {
        "vllm.model_executor.models.qwen3_dflash": _module(
            "vllm.model_executor.models.qwen3_dflash",
            DFlashQwen3DecoderLayer=nn.Module,
            DFlashQwen3ForCausalLM=nn.Module,
            DFlashQwen3Model=DraftModel,
        ),
        "vllm.model_executor.layers.linear": _module("vllm.model_executor.layers.linear", ReplicatedLinear=Linear),
        "vllm.model_executor.layers.quantization.base_config": _module(
            "vllm.model_executor.layers.quantization.base_config", QuantizationConfig=object
        ),
        "vllm.model_executor.layers.vocab_parallel_embedding": _module(
            "vllm.model_executor.layers.vocab_parallel_embedding", UnquantizedEmbeddingMethod=object
        ),
        "vllm.model_executor.models.utils": _module(
            "vllm.model_executor.models.utils", maybe_prefix=lambda prefix, name: f"{prefix}.{name}" if prefix else name
        ),
        "vllm_ascend.models.qwen3_dspark": _module(
            "vllm_ascend.models.qwen3_dspark", align_draft_weights=lambda *args: None
        ),
        "vllm_ascend.utils": _module("vllm_ascend.utils", get_rotation_path=lambda config: None),
    }
    for name, replacement in replacements.items():
        monkeypatch.setitem(sys.modules, name, replacement)
    name = "_ascend_dflash2_compile_test"
    spec = importlib.util.spec_from_file_location(name, MODEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _config(dynamic_shapes_type=DynamicShapesType.BACKED):
    draft = SimpleNamespace(
        hidden_size=4,
        vocab_size=8,
        dflash_config={"selector_rank": 3, "selector_top_k": 2},
    )
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            backend="eager",
            dynamic_shapes_config=SimpleNamespace(type=dynamic_shapes_type, assume_32_bit_indexing=False),
            traced_files=set(),
            cudagraph_mode=SimpleNamespace(has_piecewise_cudagraphs=lambda: False),
        ),
        model_config=SimpleNamespace(dtype=torch.float32),
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=draft)),
    )


@pytest.fixture
def cpu_compile_wrapper(monkeypatch):
    initialized = []

    def initialize(model, *, compile_prefix="", is_encoder=False):
        model._is_encoder = is_encoder
        model.test_compile_graphs = []

        def backend(graph, example_inputs):
            model.test_compile_graphs.append(graph)
            return graph.forward

        model.test_compiled_forward = torch.compile(model.forward, backend=backend, fullgraph=True)
        initialized.append(model)

    def call(model, *args, **kwargs):
        return model.test_compiled_forward(*args, **kwargs)

    torch._dynamo.reset()
    monkeypatch.setattr(TorchCompileWithNoGuardsWrapper, "__init__", initialize)
    monkeypatch.setattr(TorchCompileWithNoGuardsWrapper, "__call__", call)
    monkeypatch.setattr(
        TorchCompileWithNoGuardsWrapper,
        "original_code_object",
        lambda model: model.forward.__code__,
        raising=False,
    )
    monkeypatch.setattr("vllm.envs.VLLM_USE_AOT_COMPILE", False)
    monkeypatch.setattr("vllm.compilation.decorators.monitor_torch_compile", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr("vllm.compilation.decorators.monitor_profiling_run", lambda *a, **kw: contextlib.nullcontext())
    # Real ForwardContext construction/lifetime is checked by speculator tests.
    context = SimpleNamespace(skip_compiled=False, is_padding=None)
    monkeypatch.setattr("vllm.compilation.decorators.is_forward_context_available", lambda: True)
    monkeypatch.setattr("vllm.compilation.decorators.get_forward_context", lambda: context)
    yield initialized
    torch._dynamo.reset()


def test_dflash2_inherited_compilation_wrapper_remains_enabled(model_module, cpu_compile_wrapper):
    config = _config()
    with set_current_vllm_config(config):
        model = model_module.DFlash2Qwen3Model(vllm_config=config)

    assert not model.do_not_compile
    assert not model.candidate_selector.do_not_compile
    assert cpu_compile_wrapper == [model, model.candidate_selector]
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    input_ids = torch.tensor([1, 2, 3])
    positions = torch.tensor([0, 1, 2])
    torch.testing.assert_close(model(input_ids, positions), model.embed_tokens(input_ids))
    assert model.compiled
    assert len(model.test_compile_graphs) == 1


@pytest.mark.parametrize("num_steps", [4, 5, 6])
@pytest.mark.parametrize("dynamic_shapes_type", [DynamicShapesType.BACKED, DynamicShapesType.UNBACKED])
def test_selector_compiles_dynamic_batches_and_matches_edge_scores(
    model_module, cpu_compile_wrapper, num_steps, dynamic_shapes_type
):
    torch.manual_seed(1)
    selector = model_module.CandidateSelector(
        hidden_size=4,
        vocab_size=8,
        rank=3,
        top_k=2,
        params_dtype=torch.float32,
        prefix="selector",
        vllm_config=_config(dynamic_shapes_type),
    )
    with torch.no_grad():
        selector.predecessor_codebook.copy_(torch.randn_like(selector.predecessor_codebook))
        selector.successor_codebook.copy_(torch.randn_like(selector.successor_codebook))
    for num_reqs in (3, 1, 2):
        candidates = torch.arange(num_reqs * num_steps * 2).reshape(num_reqs, num_steps, 2) % 8
        unary = torch.randn(num_reqs, num_steps, 2)
        hidden_states = torch.randn(num_reqs, num_steps, 4)
        anchor_ids = torch.arange(num_reqs) % 8
        actual = selector(candidates, unary, hidden_states, anchor_ids)
        hidden = F.linear(hidden_states, selector.hidden_projection.weight)
        expected = torch.empty(num_reqs, num_steps, 2, 2)
        for request in range(num_reqs):
            for step in range(num_steps):
                for predecessor in range(2):
                    previous_id = anchor_ids[request] if step == 0 else candidates[request, step - 1, predecessor]
                    for successor in range(2):
                        current_id = candidates[request, step, successor]
                        interaction = (
                            selector.predecessor_codebook[previous_id]
                            * hidden[request, step]
                            * selector.successor_codebook[current_id]
                        ).sum()
                        expected[request, step, predecessor, successor] = unary[request, step, successor] + interaction
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, selector.forward(candidates, unary, hidden_states, anchor_ids))
    assert selector.compiled
    # Backed shapes may specialize batch=1; batch=2 must reuse the batch=3 graph.
    assert 1 <= len(selector.test_compile_graphs) <= 2


@pytest.mark.parametrize("block_size", [5, 6, 7])
def test_grouped_convolution_keeps_request_boundaries(model_module, block_size):
    torch.manual_seed(2)
    num_groups, group_size, taps = 2, 2, 2
    base = torch.randn(taps, num_groups * group_size)

    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(model_module._grouped_conv, backend=backend, fullgraph=True, dynamic=True)
    for num_reqs in (3, 1, 2):
        hidden = torch.randn(num_reqs * block_size, num_groups * group_size)
        delta = torch.randn(num_reqs * block_size, taps, num_groups)
        actual = compiled(hidden, delta, base, block_size, num_groups, group_size, taps)

        expected = torch.zeros_like(hidden)
        for token in range(hidden.shape[0]):
            for group in range(num_groups):
                for lane in range(group_size):
                    column = group * group_size + lane
                    for tap in range(taps):
                        if token % block_size >= tap:
                            expected[token, column] += (base[tap, column] + delta[token, tap, group]) * hidden[
                                token - tap, column
                            ]
        torch.testing.assert_close(actual, expected)
    assert len(graphs) == 1
