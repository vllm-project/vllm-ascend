# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


def make_spec(supported=True):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.draft_logits = None
    spec._draft_topk = None
    spec.acceptance_estimator = None
    spec.use_confidence_head = False
    spec.num_speculative_steps = 3
    spec.sample_indices = torch.arange(6)
    spec.sample_idx_mapping = torch.arange(6)
    spec.sample_pos = torch.arange(6)
    spec._anchor_idx = torch.tensor([0, 3])
    spec.input_buffers = SimpleNamespace(input_ids=torch.tensor([1, 0, 0, 2, 0, 0]))
    spec.draft_tokens = torch.full((2, 3), -1, dtype=torch.int32)
    spec.draft_token_confidence_probs = torch.empty(2, 3)
    previous = []

    def embed(prev):
        previous.append(prev.clone())
        return prev

    def bias(prev):
        result = torch.zeros(2, 5)
        result[torch.arange(2), ((prev + 1) % 5).long()] = 10
        return result

    spec.model = SimpleNamespace(
        compute_draft_logits=MagicMock(return_value=torch.zeros(6, 5)),
        markov_embed=embed,
        markov_bias=bias,
        map_draft_to_target=lambda x: x + 100,
        compute_confidence=MagicMock(return_value=torch.ones(6)),
    )

    def run(a, b, out):
        out.copy_((a + b).argmax(-1))

    spec._add_argmax_workspace = SimpleNamespace(
        supports=MagicMock(return_value=supported), run=MagicMock(side_effect=run)
    )
    spec._sample_logits = MagicMock(side_effect=lambda logits, *_: logits.argmax(-1) + 100)
    return spec, previous


@pytest.mark.parametrize("supported", [True, False])
@pytest.mark.parametrize("confidence", [True, False])
def test_sequential_mapping_and_step_fallback(supported, confidence):
    spec, previous = make_spec(supported)
    spec.use_confidence_head = confidence
    spec._sample_sequential(2, torch.zeros(6, 2))
    assert spec.draft_tokens.tolist() == [[102, 103, 104], [103, 104, 100]]
    assert previous[1].tolist() == [102, 103]
    assert previous[2].tolist() == [103, 104]
    assert spec._add_argmax_workspace.run.call_count == (3 if supported else 0)
    assert spec._sample_logits.call_count == (0 if supported else 3)
    assert spec.model.compute_confidence.call_count == int(confidence)
    if confidence:
        torch.testing.assert_close(spec.draft_token_confidence_probs, torch.ones(2, 3))


@pytest.mark.parametrize("reason", ["disabled", "probabilistic", "topk", "estimator", "transform", "model_hook"])
def test_preserves_parent_sampling(monkeypatch, reason):
    spec, _ = make_spec()
    if reason == "disabled":
        spec._add_argmax_workspace = None
    elif reason == "probabilistic":
        spec.draft_logits = object()
    elif reason == "topk":
        spec._draft_topk = 10
    elif reason == "estimator":
        spec.acceptance_estimator = object()
    elif reason == "transform":
        spec.model.apply_output_logit_transform = lambda x: x
    elif reason == "model_hook":
        spec.model.sample_sequential = lambda *args: False
    parent = MagicMock()
    monkeypatch.setattr(DSparkSpeculator, "_sample_sequential", parent)
    hidden = torch.zeros(6, 2)
    spec._sample_sequential(2, hidden)
    parent.assert_called_once_with(2, hidden)
    spec.model.compute_draft_logits.assert_not_called()
