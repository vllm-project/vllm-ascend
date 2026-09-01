# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# CompactDist: top-k compressed distribution + vocab restoration mapping.
#
# CompactDist is the sparse surrogate of the dense [B, V] raw_logprobs
# tensor. It carries:
#   - token_index [B, k] i32: vocab ids of the top-k candidates
#     (descending by logit)
#   - logprobs [B, k] f32: full-vocab-normalized logprob (topv - LSE(z))
#
# Both arrays are sorted descending by logit, so topn() is a zero-cost
# slice and gather()/rank() are O(k) linear scans -- all without
# materializing [B, V].

import torch

__all__ = ["CompactDist"]


class CompactDist:
    """top-k compressed distribution + vocab restoration mapping.

    Attributes:
        token_index: [B, k] int32, vocab ids of top-k candidates
            (descending).
        logprobs: [B, k] float32, full-vocab-normalized logprob
            (descending).
    """

    def __init__(
        self, token_index: torch.Tensor, logprobs: torch.Tensor
    ):
        self.token_index = token_index
        self.logprobs = logprobs

    def gather(self, vocab_ids: torch.Tensor) -> torch.Tensor:
        """Look up logprob by vocab id.

        Hit -> the logprob value at the matched position.
        Miss -> -inf (token not in top-k, treated as zero probability).

        All operations are tensor-level (no ``.item()``) to avoid
        CPU-NPU synchronization.

        Args:
            vocab_ids: [B] int64/int32, the vocab ids to look up.

        Returns:
            [B] float32 tensor of logprob values (or -inf on miss).
        """
        hit = self.token_index == vocab_ids.to(torch.int32).unsqueeze(-1)
        pos = hit.long().argmax(dim=-1)  # [B]
        val = self.logprobs.gather(
            1, pos.unsqueeze(1)
        ).squeeze(1)  # [B]
        found = hit.any(dim=-1)  # [B]
        return torch.where(
            found, val, torch.full_like(val, float("-inf"))
        )

    def topn(
        self, n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the top-n (logprob, token_id) pairs.

        Since token_index/logprobs are already descending by logit,
        this is a zero-cost slice.

        Args:
            n: number of top entries to return.

        Returns:
            (logprobs[:, :n], token_index[:, :n])
        """
        return self.logprobs[:, :n], self.token_index[:, :n]
