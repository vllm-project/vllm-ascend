# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Stage 310P whitelist/bias state for the upstream sampling helper."""

import torch
from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias


class LogitBiasState310:
    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.device = device
        self._staged: tuple[torch.Tensor, ...] | None = None
        self._zeros = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    def invalidate(self) -> None:
        self._staged = None

    def apply(self, logits, expanded_idx_mapping, allowed_token_ids, logit_bias) -> None:
        if self._staged is None:
            allowed_width = max((len(ids) for ids in allowed_token_ids.values()), default=1)
            bias_width = max((len(bias) for bias in logit_bias.values()), default=1)
            allowed = torch.zeros((self.max_num_reqs, allowed_width), dtype=torch.int32)
            bias_ids = torch.zeros((self.max_num_reqs, bias_width), dtype=torch.int32)
            biases = torch.zeros((self.max_num_reqs, bias_width), dtype=torch.float32)
            num_allowed = torch.zeros(self.max_num_reqs, dtype=torch.int32)
            num_bias = torch.zeros_like(num_allowed)
            for req_idx, ids in allowed_token_ids.items():
                num_allowed[req_idx] = len(ids)
                allowed[req_idx, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
            for req_idx, bias in logit_bias.items():
                num_bias[req_idx] = len(bias)
                bias_ids[req_idx, : len(bias)] = torch.tensor(list(bias), dtype=torch.int32)
                biases[req_idx, : len(bias)] = torch.tensor(list(bias.values()), dtype=torch.float32)
            self._staged = tuple(t.to(self.device) for t in (num_allowed, allowed, num_bias, bias_ids, biases))
        num_allowed, allowed, num_bias, bias_ids, biases = self._staged
        # min_tokens is outside this PR's scope. Supply neutral state to the
        # shared upstream helper rather than changing its kernel contract.
        zeros = self._zeros
        # Positions are unused when all stop-token counts are zero.
        positions = expanded_idx_mapping
        stop_ids = zeros.unsqueeze(-1)
        apply_logit_bias(
            logits,
            expanded_idx_mapping,
            positions,
            num_allowed,
            allowed,
            num_bias,
            bias_ids,
            biases,
            zeros,
            zeros,
            zeros,
            stop_ids,
        )
