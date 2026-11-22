# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def build_mla_logits(scores, candidates, sink, scale):
    """Build FP32 logits with the learned sink in the last column."""
    logits = (scores.float() * scale).masked_fill(candidates[:, None, :] < 0, -torch.inf)
    sink_logits = sink.view(1, -1, 1).expand(scores.shape[0], -1, -1)
    return torch.cat((logits, sink_logits), dim=-1)
