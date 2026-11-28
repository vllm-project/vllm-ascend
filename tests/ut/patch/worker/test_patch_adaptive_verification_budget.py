# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import torch

from vllm_ascend.patch.worker import patch_adaptive_verification_budget as patch_module


def test_sanitize_invalid_confidence_replaces_non_probabilities():
    probs = torch.tensor(
        [
            [float("nan"), 0.5, 1.0, 0.0],
            [float("inf"), 1.5, -0.2, 0.75],
        ]
    )
    idx_mapping = torch.tensor([1, 0])

    sanitized = patch_module._sanitize_invalid_confidence(probs, idx_mapping)

    torch.testing.assert_close(
        sanitized,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.75],
                [0.0, 0.5, 1.0, 0.0],
            ]
        ),
    )
    # The allocator's buffer is copied, not modified in place.
    torch.testing.assert_close(
        probs,
        torch.tensor(
            [
                [float("nan"), 0.5, 1.0, 0.0],
                [float("inf"), 1.5, -0.2, 0.75],
            ]
        ),
        equal_nan=True,
    )


def test_install_sanitizes_before_upstream_allocator(monkeypatch):
    observed = {}

    def fake_assign(confidence_probs, idx_mapping, capacities, draft_budget, survival):
        observed["probs"] = confidence_probs.clone()

    module = SimpleNamespace(assign_draft_token_budget=fake_assign)
    monkeypatch.setattr(patch_module, "_upstream_sanitizes_invalid_confidence", lambda _: False)

    patch_module._install(module)

    probs = torch.tensor([[float("nan"), 0.4], [0.9, 1.5]])
    module.assign_draft_token_budget(
        probs,
        torch.tensor([0, 1]),
        torch.zeros(2, dtype=torch.int32),
        1,
        torch.zeros(1, dtype=torch.int32),
    )

    torch.testing.assert_close(observed["probs"], torch.tensor([[0.0, 0.4], [0.9, 0.0]]))


def test_install_is_idempotent(monkeypatch):
    module = SimpleNamespace(assign_draft_token_budget=lambda *args, **kwargs: None)
    monkeypatch.setattr(patch_module, "_upstream_sanitizes_invalid_confidence", lambda _: False)

    patch_module._install(module)
    wrapper = module.assign_draft_token_budget
    patch_module._install(module)

    assert module.assign_draft_token_budget is wrapper
    assert getattr(wrapper, patch_module._SANITIZER_MARKER) is True


def test_install_skips_when_upstream_already_sanitizes(monkeypatch):
    def fake_assign(*args, **kwargs):
        return None

    module = SimpleNamespace(assign_draft_token_budget=fake_assign)
    monkeypatch.setattr(patch_module, "_upstream_sanitizes_invalid_confidence", lambda _: True)

    patch_module._install(module)

    assert module.assign_draft_token_budget is fake_assign
