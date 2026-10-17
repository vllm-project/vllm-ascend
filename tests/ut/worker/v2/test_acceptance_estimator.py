# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.acceptance_estimator import AscendOnlineAcceptanceEstimator


@pytest.fixture
def estimator():
    return AscendOnlineAcceptanceEstimator(4, 3, torch.device("cpu"))


@pytest.mark.parametrize("draft_step", [0, 2])
def test_predict_uses_persistent_request_temperature(estimator, draft_step):
    estimator.slope.fill_(1.0)
    estimator.intercepts.zero_()
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, -1.0, 0.0]])
    idx_mapping = torch.tensor([2, 0, 3])
    temperature = torch.tensor([0.0, 1.0, 0.5, 2.0])
    confidence = torch.full((3, 3), -1.0)

    estimator.predict(logits, idx_mapping, torch.tensor(draft_step), confidence, temperature)

    scaled = logits / torch.tensor([0.5, 1.0, 2.0]).unsqueeze(1)
    expected = scaled.softmax(dim=-1).amax(dim=-1)
    torch.testing.assert_close(confidence[:, draft_step], expected)
    torch.testing.assert_close(estimator.predictions[idx_mapping, draft_step], expected)
    torch.testing.assert_close(estimator.features[idx_mapping, draft_step], torch.logit(expected))
    assert torch.all(confidence[:, (draft_step + 1) % 3] == -1.0)


@pytest.mark.parametrize("parallel", [False, True])
def test_predict_isolates_multiple_padded_requests(estimator, parallel):
    estimator.slope.fill_(0.7)
    estimator.intercepts.copy_(torch.tensor([0.3, -0.2, 0.6]))
    estimator.features[:4].fill_(7.0)
    estimator.predictions[:4].fill_(0.25)
    idx_mapping = torch.tensor([2, 0, -1, -1], dtype=torch.int32)
    if parallel:
        idx_mapping = idx_mapping.repeat_interleave(3)
        draft_step = torch.arange(3, dtype=torch.int32).repeat(4)
    else:
        draft_step = torch.tensor(1, dtype=torch.int32)
    logits = torch.tensor([2.0, 0.0, -1.0]).repeat(idx_mapping.numel(), 1)
    logits *= torch.arange(1, idx_mapping.numel() + 1).unsqueeze(1) / 4.0
    logits[idx_mapping < 0] = float("nan")
    confidence = torch.full((4, 3), -1.0)

    estimator.predict(logits, idx_mapping, draft_step, confidence, torch.ones(4))

    steps = slice(None) if parallel else 1
    # Independent float64 reference: remove the argmax before logsumexp,
    # rather than recovering its complement from the total probability.
    real_logits = logits[: 6 if parallel else 2].double()
    top = real_logits.max(dim=-1)
    rest = real_logits.scatter(1, top.indices.unsqueeze(1), float("-inf"))
    features = top.values - rest.logsumexp(dim=-1)
    step_idx = draft_step[:6].long() if parallel else draft_step.long()
    expected = torch.sigmoid(estimator.slope.double() * features + estimator.intercepts[step_idx].double()).float()
    expected = expected.reshape(2, 3) if parallel else expected
    torch.testing.assert_close(confidence[:2, steps], expected)
    torch.testing.assert_close(estimator.predictions[[2, 0], steps], expected)
    assert torch.all(confidence[2:, steps] == 0.0)
    assert torch.all(estimator.features[[1, 3]] == 7.0)
    assert torch.all(estimator.predictions[[1, 3]] == 0.25)
    assert torch.all(estimator.features[4:, steps] == 0.0)
    assert torch.all(estimator.predictions[4:, steps] == 0.0)


def test_predict_is_invariant_to_large_logit_offset(estimator):
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    idx_mapping = torch.tensor([2, 0])
    confidence = torch.zeros(2, 3)
    estimator.predict(logits, idx_mapping, torch.tensor(0), confidence, torch.ones(4))
    features = estimator.features[idx_mapping, 0].clone()
    predictions = confidence[:, 0].clone()

    estimator.predict(logits + 1e6, idx_mapping, torch.tensor(0), confidence, torch.ones(4))

    torch.testing.assert_close(estimator.features[idx_mapping, 0], features)
    torch.testing.assert_close(confidence[:, 0], predictions)


def test_predict_clamps_degenerate_distribution(estimator):
    confidence = torch.zeros(2, 3)
    estimator.predict(
        torch.tensor([[0.0, -100.0], [0.0, float("-inf")]]),
        torch.tensor([2, 0]),
        torch.tensor(0),
        confidence,
        torch.ones(4),
    )

    torch.testing.assert_close(estimator.features[[2, 0], 0], torch.full((2,), estimator.MAX_LOG_ODDS))
    assert torch.isfinite(confidence).all()


@pytest.mark.parametrize(
    "num_sampled,num_rejected,counts,residuals",
    [
        (2, 2, [1.0, 1.0, 0.0], [0.25, -0.25, 0.0]),
        (4, 0, [1.0, 1.0, 1.0], [0.25, 0.75, 0.1]),
        (0, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        (1, 3, [1.0, 0.0, 0.0], [-0.75, 0.0, 0.0]),
    ],
)
def test_step_observes_only_accepted_and_first_rejected(estimator, num_sampled, num_rejected, counts, residuals):
    features = torch.tensor([0.5, -0.5, 2.0])
    predictions = torch.tensor([0.75, 0.25, 0.9])
    estimator.features[2] = features
    estimator.predictions[2] = predictions
    estimator.step(torch.tensor([2, -1]), torch.tensor([num_sampled, 4]), torch.tensor([num_rejected, 0]))

    counts = torch.tensor(counts)
    residuals = torch.tensor(residuals)
    weights = predictions * (1.0 - predictions) * counts
    torch.testing.assert_close(estimator.counts, counts)
    torch.testing.assert_close(
        estimator.info, torch.stack((weights * features.square(), weights * features, weights), 1)
    )
    torch.testing.assert_close(estimator.grad, torch.stack((residuals * features, residuals), 1))
    assert estimator._refits == 0


def test_refit_matches_dense_newton_solve(estimator):
    estimator.REFIT_INTERVAL = 2
    estimator.features[2] = torch.tensor([0.5, -0.5, 2.0])
    estimator.predictions[2] = torch.tensor([0.6, 0.4, 0.9])
    idx_mapping = torch.tensor([2])
    sampled = torch.tensor([2])
    rejected = torch.tensor([2])
    estimator.step(idx_mapping, sampled, rejected)

    info = estimator.info * 2
    grad = estimator.grad * 2
    counts = estimator.counts * 2
    hessian = torch.zeros(4, 4)
    hessian[0, 0] = info[:, 0].sum()
    hessian[0, 1:] = info[:, 1]
    hessian[1:, 0] = info[:, 1]
    hessian[1:, 1:] = torch.diag(info[:, 2])
    hessian += estimator.L2 * torch.eye(4)
    gradient = torch.cat((grad[:, 0].sum().reshape(1), grad[:, 1]))
    delta = torch.linalg.solve(hessian, gradient)
    slope_delta = delta[0] * counts.sum() / (counts.sum() + estimator.DAMPING_OBSERVATIONS)
    # Upstream back-substitutes using the already damped shared-slope step.
    bias_delta = (grad[:, 1] - info[:, 1] * slope_delta) / (info[:, 2] + estimator.L2)
    bias_delta *= counts / (counts + estimator.DAMPING_OBSERVATIONS)
    expected_slope = estimator.slope + slope_delta
    expected_intercepts = estimator.intercepts + bias_delta

    estimator.step(idx_mapping, sampled, rejected)

    torch.testing.assert_close(estimator.slope, expected_slope)
    torch.testing.assert_close(estimator.intercepts, expected_intercepts)
    assert estimator._steps_since_refit == 0
    assert estimator._refits == 1
    for stats in (estimator.info, estimator.grad, estimator.counts):
        assert torch.count_nonzero(stats) == 0


def test_refit_without_observations_keeps_coefficients(estimator):
    estimator.REFIT_INTERVAL = 1
    slope = estimator.slope.clone()
    intercepts = estimator.intercepts.clone()

    estimator.step(torch.tensor([-1]), torch.tensor([4]), torch.tensor([0]))

    torch.testing.assert_close(estimator.slope, slope)
    torch.testing.assert_close(estimator.intercepts, intercepts)
    assert estimator._refits == 1


def test_refit_does_not_make_slope_negative(estimator):
    estimator.REFIT_INTERVAL = 1
    estimator.info[:, 0].fill_(1.0)
    estimator.grad[:, 0].fill_(-100.0)
    estimator.counts.fill_(100.0)
    slope = estimator.slope.clone()

    estimator.step(torch.tensor([], dtype=torch.long), torch.tensor([]), torch.tensor([]))

    torch.testing.assert_close(estimator.slope, slope)
    assert torch.isfinite(estimator.intercepts).all()
