# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.ascend_config import SpecKConfig
from vllm_ascend.spec_decode.spec_k import (
    SpecKHistoryUpdate,
    SpecKPolicy,
    SpecKRequestState,
)


def _vllm_config(
    method: str = "draft_model",
    async_scheduling: bool = False,
    enable_prefix_caching: bool = False,
    quantization: str | None = None,
):
    speculative_config = SimpleNamespace(
        uses_draft_model=lambda: method == "draft_model",
        enforce_eager=True,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(quantization=quantization),
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        use_v2_model_runner=False,
    )


def _policy_config(apply_last_token: bool = False):
    return SimpleNamespace(
        ppl_thresholds=(3.0, 2.0),
        apply_last_token=apply_last_token,
    )


def test_spec_k_config_parses_engine_global_policy():
    config = SpecKConfig(
        {
            "enabled": True,
            "ppl_thresholds": [3.0, 2.0],
            "full_top_k_layer_range": [1, 5, 2],
            "apply_last_token": True,
        },
        _vllm_config(),
    )

    assert config.enabled
    assert config.ppl_thresholds == (3.0, 2.0)
    assert config.full_top_k_layer_range == (1, 5, 2)
    assert config.apply_last_token


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"enabled": True}, "must not be empty"),
        (
            {"enabled": True, "ppl_thresholds": [2.0, 3.0]},
            "non-increasing",
        ),
        (
            {"enabled": True, "ppl_thresholds": [float("nan")]},
            "finite, positive",
        ),
        (
            {"enabled": True, "ppl_thresholds": [0.0]},
            "finite, positive",
        ),
        (
            {"enabled": True, "ppl_thresholds": [True]},
            "list of numbers",
        ),
        (
            {"enabled": True, "ppl_thresholds": ["2.0"]},
            "list of numbers",
        ),
    ],
)
def test_spec_k_config_rejects_invalid_thresholds(config, match):
    with pytest.raises(ValueError, match=match):
        SpecKConfig(config, _vllm_config())


def test_spec_k_config_rejects_other_speculative_methods():
    with pytest.raises(ValueError, match="method='draft_model'"):
        SpecKConfig(
            {"enabled": True, "ppl_thresholds": [2.0]},
            _vllm_config(method="ngram"),
        )


def test_spec_k_config_rejects_non_eager_draft_proposer():
    vllm_config = _vllm_config()
    vllm_config.speculative_config.enforce_eager = False

    with pytest.raises(ValueError, match="enforce_eager=true"):
        SpecKConfig(
            {"enabled": True, "ppl_thresholds": [2.0]},
            vllm_config,
        )


def test_spec_k_config_rejects_quantized_target():
    with pytest.raises(ValueError, match="quantized target"):
        SpecKConfig(
            {"enabled": True, "ppl_thresholds": [2.0]},
            _vllm_config(quantization="awq"),
        )


def test_spec_k_config_rejects_prefix_caching():
    with pytest.raises(ValueError, match="prefix caching"):
        SpecKConfig(
            {"enabled": True, "ppl_thresholds": [2.0]},
            _vllm_config(enable_prefix_caching=True),
        )


def test_policy_requires_at_least_one_active_expert():
    with pytest.raises(ValueError, match="at least one expert"):
        SpecKPolicy(
            _policy_config(),
            base_top_k=2,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("apply_last_token", [False, True])
def test_policy_builds_per_token_top_ks(apply_last_token):
    policy = SpecKPolicy(
        _policy_config(apply_last_token),
        base_top_k=4,
        device=torch.device("cpu"),
    )
    logits = torch.tensor([[[10.0, -10.0, -10.0, -10.0], [0.0, 0.0, 0.0, 0.0]]])

    top_ks = policy.top_ks_from_logits(logits)

    expected_last = 3 if apply_last_token else 4
    assert top_ks.tolist() == [[2, 4, expected_last]]


def test_policy_handles_sparse_vocabulary_logits():
    policy = SpecKPolicy(
        _policy_config(),
        base_top_k=4,
        device=torch.device("cpu"),
    )
    logits = torch.tensor([[[0.0, 0.0, float("-inf"), float("-inf")]]])
    original_logits = logits.clone()

    top_ks = policy.top_ks_from_logits(logits)

    assert top_ks.tolist() == [[3, 4]]
    assert torch.equal(logits, original_logits)


def test_request_state_restores_history_and_pending_drafts_by_position():
    state = SpecKRequestState(base_top_k=8)
    state.finalize_step(
        SpecKHistoryUpdate(
            new_output_length=3,
            accepted_top_ks=torch.tensor([5, 4]),
        ),
        sampled_token_top_k=torch.tensor(3),
    )
    state.set_pending_draft_top_ks(torch.tensor([3, 2, 1]))

    top_ks = state.top_ks_for_positions(
        np.array([3, 4, 5, 6, 7, 8], dtype=np.int64),
        num_prompt_tokens=4,
        pending_draft_start_position=6,
        use_pending_draft=True,
    )

    assert top_ks.tolist() == [8, 5, 4, 3, 2, 1]


def test_request_state_reconciles_recompute_and_missing_history():
    state = SpecKRequestState(base_top_k=8)
    state.output_top_ks = torch.tensor([5, 4, 3], dtype=torch.uint8)

    state.reconcile_output_length(2)
    assert state.output_top_ks.tolist() == [5, 4]

    state.reconcile_output_length(4)
    assert state.output_top_ks.tolist() == [5, 4, 8, 8]
