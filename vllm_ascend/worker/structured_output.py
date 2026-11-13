# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence

import torch
from vllm.v1.structured_output.utils import strip_speculative_padding


def build_invalid_draft_mask(
    req_ids: Sequence[str],
    grammar_req_ids: Sequence[str],
    scheduled_spec_decode_tokens: Mapping[str, Sequence[int]],
    num_draft_tokens: Sequence[int],
    device: torch.device,
) -> torch.Tensor | None:
    """Return draft positions that were not covered by a grammar bitmask.

    The scheduler fills permissive grammar rows after the first missing
    speculative draft. Those rows must not be verified as real drafts. The
    MRv1 rejection sampler receives flattened draft tokens, so construct the
    equivalent flattened mask here without changing the request's draft IDs.
    """
    if not grammar_req_ids or not any(num_draft_tokens):
        return None

    acceptable_drafts = {
        req_id: len(strip_speculative_padding(scheduled_spec_decode_tokens.get(req_id, ())))
        for req_id in grammar_req_ids
    }
    invalid_drafts = torch.zeros(sum(num_draft_tokens), dtype=torch.bool, device=device)

    has_invalid_drafts = False
    offset = 0
    for req_id, num_drafts in zip(req_ids, num_draft_tokens):
        acceptable = acceptable_drafts.get(req_id)
        if acceptable is not None:
            acceptable = min(max(acceptable, 0), num_drafts)
            if acceptable < num_drafts:
                invalid_drafts[offset + acceptable : offset + num_drafts] = True
                has_invalid_drafts = True
        offset += num_drafts

    return invalid_drafts if has_invalid_drafts else None
