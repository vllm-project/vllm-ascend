# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Ascend RecoverSSM request/commit metadata helpers."""

import torch


def recoverssm_spec_rows(
    is_prefilling_cpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    num_decode_draft_tokens_cpu: torch.Tensor,
    num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify the real batch rows that must take the RecoverSSM spec path."""
    query_lens = query_start_loc_cpu[: num_reqs + 1].diff()
    active_decode = (~is_prefilling_cpu[:num_reqs]) & (query_lens > 0)
    spec_mask = (num_decode_draft_tokens_cpu[:num_reqs] >= 0) | active_decode

    drafts = torch.full_like(num_decode_draft_tokens_cpu, -1)
    drafts[:num_reqs] = torch.where(spec_mask, 1, -1).to(drafts.dtype)
    return drafts, spec_mask.nonzero().flatten()


def recoverssm_request_indices(
    rows: torch.Tensor,
    num_spec_decodes: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Map compacted RecoverSSM spec rows back to their original batch rows."""
    if rows.numel() != num_spec_decodes:
        raise ValueError(
            f"RecoverSSM: {rows.numel()} spec rows classified, but the builder produced {num_spec_decodes}"
        )
    if num_spec_decodes == 0 or int(rows[-1]) == num_spec_decodes - 1:
        return None
    return rows.to(torch.int32).to(device, non_blocking=True)
