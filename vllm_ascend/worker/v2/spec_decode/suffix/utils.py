# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initial suffix candidate selection; speculator integration is pending."""

import torch


def _select_suffix_candidates(
    history_drafts: torch.Tensor,
    corpus_drafts: torch.Tensor,
    history_match_lengths: torch.Tensor,
    corpus_match_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the longer suffix match per request, preferring history on ties.

    Drafts have shape [requests, draft_tokens], and match lengths have shape
    [requests]. Return the selected drafts and the per-request corpus mask.
    """
    use_corpus = corpus_match_lengths > history_match_lengths
    drafts = torch.where(use_corpus[:, None], corpus_drafts, history_drafts)
    return drafts, use_corpus
