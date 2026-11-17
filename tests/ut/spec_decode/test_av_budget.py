# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_ascend.worker.v2.spec_decode.av_budget import configure_av_budget, select_av_budget


def test_budget_tolerance_requires_more_expected_output():
    output = np.array([1.0, 1.8, 2.0, 2.0, 2.1])
    scores = np.array([0.8, 1.0, 0.994, 0.993, 0.98])
    assert select_av_budget(output, scores, 0) == 1
    assert select_av_budget(output, scores, 0.005) == 1
    assert select_av_budget(output, scores, 0.01) == 3
    assert select_av_budget(np.ones(5), scores, 0.01) == 1


def test_manager_keeps_native_default_and_uses_only_scheduled_candidates():
    manager = SimpleNamespace(
        get_num_tokens=lambda scheduled, drafts: 2,
        num_speculative_steps=2,
        num_bonus_tokens=1,
        _max_total_logits=2,
        _stale_idx=0,
        _stale_confidences=[SimpleNamespace(np=np.array([[0.8, 0.5]]))],
        cost_tables=(np.array([1.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.46])),
        req_states=SimpleNamespace(
            req_id_to_index={"a": 0},
            num_computed_tokens_np=np.array([1]),
            prefill_len=SimpleNamespace(np=np.array([1])),
        ),
    )
    native = manager.get_num_tokens
    assert configure_av_budget(manager, 0).get_num_tokens is native
    with pytest.raises(ValueError, match="AV_SCORE_TOLERANCE"):
        configure_av_budget(manager, float("nan"))

    configure_av_budget(manager, 0.01)
    assert manager.get_num_tokens({"a": 3}, {"a": [1, 2]}) == 2
    assert manager._batch_budget == ({"a": 2}, {"a": 1}, 1)
    manager._max_total_logits = 3
    assert manager.get_num_tokens({"a": 3}, {"a": [1, 2]}) == 3
    assert manager._batch_budget[-1] == 2
