# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.e2e.common.rlhf.weight_transfer import hccl, reference_check


@pytest.mark.parametrize("generation_fails", [False, True])
def test_draft_outputs_are_observed_before_target_correction(monkeypatch, generation_fails):
    extension = reference_check.WeightUpdateTestWorkerExtension()
    buffer = torch.zeros((1, 2), dtype=torch.int64)
    payload = 1

    def propose(input_batch):
        return buffer.fill_(payload)

    speculator = SimpleNamespace(propose=propose)
    extension.model_runner = SimpleNamespace(speculator=speculator)

    def post_control(server, endpoint, *, json):
        method = json["method"]
        result = {} if method == "snapshot_draft_context_projections" else getattr(extension, method)()
        return SimpleNamespace(json=lambda: {"results": [result]})

    def generate(*args, **kwargs):
        speculator.propose(SimpleNamespace(num_reqs=1))
        # The runner reuses this buffer, and target verification hides any
        # change in the proposal from the final generation signature.
        buffer.fill_(-1)
        if generation_fails:
            raise RuntimeError("generation failed")
        return [("corrected by target", (-0.5,))]

    monkeypatch.setattr(hccl, "post_control", post_control)
    monkeypatch.setattr(hccl, "generation_signature", generate)
    monkeypatch.setattr(hccl, "snapshot_weight_update_models", lambda server: {"target": {}, "draft": {}})
    metric_values = iter(range(4))

    def draft_metrics(server):
        value = next(metric_values)
        return {"num_drafts": value, "num_draft_tokens": 2 * value}

    monkeypatch.setattr(hccl, "draft_metrics", draft_metrics)
    server = SimpleNamespace(get_client=lambda: object())
    case = SimpleNamespace(id="draft", prompts=("fixed prompt",))

    if generation_fails:
        with pytest.raises(RuntimeError, match="generation failed"):
            hccl.collect_draft_observation(server, case)
    else:
        _, initial = hccl.collect_draft_observation(server, case)
        payload = 2
        _, updated = hccl.collect_draft_observation(server, case)
        assert initial["generation"] == updated["generation"]
        assert initial["draft_outputs"] == [[[[1, 1]]]]
        assert updated["draft_outputs"] == [[[[2, 2]]]]

    assert speculator.propose is propose
