# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    OnlineAcceptanceEstimator,
)


class AscendOnlineAcceptanceEstimator(OnlineAcceptanceEstimator):
    """PyTorch implementation of the MRV2 online acceptance estimator.

    Upstream uses custom Triton kernels for both prediction and online fitting.
    Those kernels target CUDA and cannot run on Ascend. Keep the estimator's
    buffers and fitting algorithm unchanged, but express the two kernel-backed
    operations with device-agnostic PyTorch operations supported by torch-npu.
    """

    MAX_LOG_ODDS = 40.0

    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
    ) -> None:
        super().__init__(max_num_reqs, num_speculative_steps, device)

        # ACL graph padded rows use idx_mapping == -1. Reserve a sink row so
        # graph replay can keep fixed shapes without overwriting a live request.
        sentinel = torch.zeros(
            1,
            num_speculative_steps,
            dtype=torch.float32,
            device=device,
        )
        self.features = torch.cat((self.features, sentinel), dim=0)
        self.predictions = torch.cat((self.predictions, sentinel.clone()), dim=0)
        self._sentinel_idx = max_num_reqs

    def step(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        """Accumulate target verdicts and periodically refit on the NPU."""
        idx_mapping = idx_mapping.to(torch.long)
        valid = idx_mapping >= 0
        safe_idx = torch.where(valid, idx_mapping, self._sentinel_idx)
        features = self.features[safe_idx]
        predictions = self.predictions[safe_idx]

        num_accepted = torch.clamp(num_sampled.to(torch.long) - 1, min=0).unsqueeze(1)
        num_admitted = (num_accepted.squeeze(1) + num_rejected.to(torch.long)).unsqueeze(1)
        positions = torch.arange(
            self.num_speculative_steps,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        observed = valid.unsqueeze(1) & (positions <= num_accepted) & (positions < num_admitted)
        labels = (positions < num_accepted).to(torch.float32)

        weights = torch.where(observed, predictions * (1.0 - predictions), 0.0)
        residuals = torch.where(observed, labels - predictions, 0.0)

        self.info[:, 0].add_((weights * features.square()).sum(dim=0))
        self.info[:, 1].add_((weights * features).sum(dim=0))
        self.info[:, 2].add_(weights.sum(dim=0))
        self.grad[:, 0].add_((residuals * features).sum(dim=0))
        self.grad[:, 1].add_(residuals.sum(dim=0))
        self.counts.add_(observed.to(torch.float32).sum(dim=0))

        self._steps_since_refit += 1
        if self._steps_since_refit < self.REFIT_INTERVAL:
            return
        self._steps_since_refit = 0

        # Solve the same arrowhead Newton-IRLS system as the upstream Triton
        # kernel. The slope uses observations from every draft position, while
        # each intercept is damped by the observations at its own position.
        a_k = self.info[:, 0]
        b_k = self.info[:, 1]
        c_k = self.info[:, 2] + self.L2
        g0_k = self.grad[:, 0]
        g1_k = self.grad[:, 1]
        counts = self.counts

        a = a_k.sum() + self.L2
        g0 = g0_k.sum()
        step_slope = (g0 - (b_k * g1_k / c_k).sum()) / (a - (b_k.square() / c_k).sum())
        total_count = counts.sum()
        step_slope *= total_count / (total_count + self.DAMPING_OBSERVATIONS)
        step_slope = torch.where(
            torch.isfinite(step_slope) & (self.slope[0] + step_slope >= 0.0),
            step_slope,
            0.0,
        )

        step_intercepts = (g1_k - b_k * step_slope) / c_k
        step_intercepts *= counts / (counts + self.DAMPING_OBSERVATIONS)
        step_intercepts = torch.where(torch.isfinite(step_intercepts), step_intercepts, 0.0)

        self.slope.add_(step_slope)
        self.intercepts.add_(step_intercepts)
        self.info.zero_()
        self.grad.zero_()
        self.counts.zero_()
        self._refits += 1

    def predict(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        draft_step: torch.Tensor,
        confidence_probs: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        """Predict per-position acceptance from ``logit(max q)`` on NPU."""
        num_tokens = logits.shape[0]
        idx_mapping = idx_mapping.to(torch.long)
        valid = idx_mapping >= 0
        safe_idx = torch.where(valid, idx_mapping, self._sentinel_idx)
        temperature_idx = torch.where(valid, idx_mapping, 0)

        row_temperature = temperature[temperature_idx].to(torch.float32)
        row_temperature = torch.where(
            row_temperature > 0.0,
            row_temperature,
            torch.ones_like(row_temperature),
        )
        scaled_logits = logits.to(torch.float32) / row_temperature.unsqueeze(1)

        max_logits = scaled_logits.amax(dim=-1)
        log_max_prob = max_logits - torch.logsumexp(scaled_logits, dim=-1)
        log_max_prob = torch.minimum(log_max_prob, torch.zeros_like(log_max_prob))
        complement = (-torch.expm1(log_max_prob)).clamp_min(torch.finfo(torch.float32).tiny)
        features = (log_max_prob - torch.log(complement)).clamp(
            min=-self.MAX_LOG_ODDS,
            max=self.MAX_LOG_ODDS,
        )
        features = torch.where(valid, features, 0.0)

        if draft_step.dim() > 0:
            step_idx = draft_step.to(torch.long)
            batch_idx = torch.arange(num_tokens, device=self.device) // self.num_speculative_steps
        else:
            step_idx = draft_step.to(torch.long).expand(num_tokens)
            batch_idx = torch.arange(num_tokens, device=self.device)

        probabilities = torch.sigmoid(self.slope[0] * features + self.intercepts[step_idx])
        probabilities = torch.where(valid, probabilities, 0.0)

        self.features.index_put_((safe_idx, step_idx), features)
        self.predictions.index_put_((safe_idx, step_idx), probabilities)
        confidence_probs.index_put_((batch_idx, step_idx), probabilities)
