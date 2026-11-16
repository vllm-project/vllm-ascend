# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hardware preemption coverage, split to fit the CI job time limit."""

import pytest

from tests.e2e.pull_request.one_card.thinking_budget_generation_utils import check_preemption_budget_and_answer


@pytest.fixture(autouse=True)
def use_v2_spawn_workers(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("speculative", [None, "ngram_gpu", "draft_model", "draft_model_probabilistic"])
@pytest.mark.parametrize("temperature", [0.0, 0.6])
def test_thinking_budget_preemption_preserves_budget_and_answer(monkeypatch, enforce_eager, speculative, temperature):
    check_preemption_budget_and_answer(monkeypatch, enforce_eager, speculative, temperature, constrain_tokens=True)
