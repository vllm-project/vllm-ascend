# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""DFlash2 bypasses Dynamo while remaining capturable by the outer ACL graph."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode, set_current_vllm_config

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


def _config():
    draft = SimpleNamespace(
        hidden_size=4,
        vocab_size=8,
        dflash_config={"selector_rank": 3, "selector_top_k": 2},
    )
    return SimpleNamespace(
        # This must remain enabled: the bypass is local to DFlash2 modules,
        # and does not force the target or its outer graph manager to eager.
        compilation_config=SimpleNamespace(mode=CompilationMode.VLLM_COMPILE),
        model_config=SimpleNamespace(dtype=torch.float32),
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=draft)),
    )


def test_dflash2_parent_compilation_wrapper_is_bypassed(model_module):
    config = _config()
    with (
        set_current_vllm_config(config),
        patch.object(TorchCompileWithNoGuardsWrapper, "__init__", side_effect=AssertionError("Dynamo initialized")),
    ):
        model = model_module.DFlash2Qwen3Model(vllm_config=config)

    assert model.do_not_compile
    assert model.candidate_selector.do_not_compile
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    input_ids = torch.tensor([1, 2, 3])
    positions = torch.tensor([0, 1, 2])
    torch.testing.assert_close(model(input_ids, positions), model.embed_tokens(input_ids))


@pytest.mark.parametrize("num_reqs", [1, 3])
def test_selector_runs_without_forward_context_and_matches_edge_scores(model_module, num_reqs):
    torch.manual_seed(1)
    selector = model_module.CandidateSelector(
        hidden_size=4,
        vocab_size=8,
        rank=3,
        top_k=2,
        params_dtype=torch.float32,
        prefix="selector",
        vllm_config=_config(),
    )
    with torch.no_grad():
        selector.predecessor_codebook.copy_(torch.randn_like(selector.predecessor_codebook))
        selector.successor_codebook.copy_(torch.randn_like(selector.successor_codebook))
    candidates = torch.arange(num_reqs * 3 * 2).reshape(num_reqs, 3, 2) % 8
    unary = torch.randn(num_reqs, 3, 2)
    hidden_states = torch.randn(num_reqs, 3, 4)
    anchor_ids = torch.arange(num_reqs) % 8
    # Scoring is invoked outside _run_model's ForwardContext, both during
    # profiling and inside the enclosing DFlash full-graph capture.
    with patch("vllm.compilation.decorators.get_forward_context", side_effect=AssertionError("No forward context")):
        actual = selector(candidates, unary, hidden_states, anchor_ids)

    hidden = F.linear(hidden_states, selector.hidden_projection.weight)
    expected = torch.empty(num_reqs, 3, 2, 2)
    for request in range(num_reqs):
        for step in range(3):
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


@pytest.mark.parametrize("block_size", [3, 4])
def test_grouped_convolution_keeps_request_boundaries(model_module, block_size):
    torch.manual_seed(2)
    num_groups, group_size, taps = 2, 2, 2
    hidden = torch.randn(2 * block_size, num_groups * group_size)
    delta = torch.randn(2 * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = model_module._grouped_conv(hidden, delta, base, block_size, num_groups, group_size, taps)

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
