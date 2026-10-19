# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Multimodal embedding regression coverage for Ascend V2 EAGLE drafting."""

from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama4_eagle import EagleLlama4ForCausalLM
from vllm.model_executor.models.mistral_large_3_eagle import EagleMistralLarge3ForCausalLM

from vllm_ascend.patch.worker import patch_eagle_multimodal_embeddings as embedding_patch
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator


@pytest.mark.parametrize("model_cls", [EagleLlama4ForCausalLM, EagleMistralLarge3ForCausalLM])
@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
@pytest.mark.parametrize("input_kind", ["text", "empty", "image"])
def test_run_model_merges_eagle_multimodal_embeddings(monkeypatch, model_cls, graph_mode, input_kind):
    """Run the Ascend wrapper and active draft embedding method on CPU.

    Mock the draft forward and graph execution, preserving embedding and merge
    behavior. Extra buffer rows represent graph padding, not image positions.
    """
    table = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    drafter = object.__new__(model_cls)
    drafter.model = SimpleNamespace(embed_input_ids=lambda ids: table[ids])
    input_ids = torch.tensor([1, 5, 6, 2, 7, 7])
    hidden_states = torch.zeros(6, 3)
    model = MagicMock(return_value=hidden_states)
    model.embed_input_ids.side_effect = partial(model_cls.embed_input_ids, drafter)

    speculator = AscendEagleSpeculator.__new__(AscendEagleSpeculator)
    speculator.model = model
    speculator.vllm_config = object()
    speculator.pcp_manager = None
    speculator.replicated_pcp = False
    speculator.supports_mm_inputs = True
    speculator.input_buffers = SimpleNamespace(input_ids=input_ids, positions=torch.arange(6))
    speculator.hidden_states = hidden_states
    speculator.inputs_embeds = torch.full((6, 3), 99.0)
    graph_forward = MagicMock(side_effect=lambda draft_model, model_inputs: draft_model(**model_inputs))
    speculator.prefill_cudagraph_manager = SimpleNamespace(run_pw_graph=graph_forward)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    expected = table[input_ids].clone()
    mm_inputs = None
    if input_kind == "empty":
        mm_inputs = ([], None)
    elif input_kind == "image":
        mm_embeds = [torch.full((2, 3), -1.0)]
        mask = torch.tensor([False, True, True, False])
        mm_inputs = (mm_embeds, mask)
        expected[1:3] = mm_embeds[0]
        expected[4:] = 99.0

    last_hidden, feedback_hidden = speculator._run_model(
        6, None, None, None, cudagraph_runtime_mode=graph_mode, mm_inputs=mm_inputs
    )

    assert last_hidden is hidden_states
    assert feedback_hidden is hidden_states
    model.embed_input_ids.assert_called_once()
    torch.testing.assert_close(model.call_args.kwargs["inputs_embeds"], expected)
    torch.testing.assert_close(model.call_args.kwargs["input_ids"], input_ids)
    if graph_mode == CUDAGraphMode.PIECEWISE:
        graph_forward.assert_called_once()
    else:
        graph_forward.assert_not_called()


def test_patch_only_replaces_legacy_embedding_alias():
    legacy = SimpleNamespace(embed_input_ids=SupportsMultiModal.embed_input_ids)
    embedding_patch._patch_draft_embedding(legacy)
    assert legacy.embed_input_ids is embedding_patch._embed_input_ids
    embedding_patch._patch_draft_embedding(legacy)
    assert legacy.embed_input_ids is embedding_patch._embed_input_ids

    def fixed_embed_input_ids(input_ids):
        return input_ids

    fixed = SimpleNamespace(embed_input_ids=fixed_embed_input_ids)
    embedding_patch._patch_draft_embedding(fixed)
    assert fixed.embed_input_ids is fixed_embed_input_ids


@pytest.mark.parametrize("model_cls", [EagleLlama4ForCausalLM, EagleMistralLarge3ForCausalLM])
def test_eagle_multimodal_embeddings_require_mask(model_cls):
    drafter = object.__new__(model_cls)
    drafter.model = SimpleNamespace(embed_input_ids=lambda ids: torch.zeros(ids.shape[0], 3))
    with pytest.raises(ValueError, match="is_multimodal"):
        model_cls.embed_input_ids(drafter, torch.tensor([1, 2]), [torch.ones(2, 3)])
