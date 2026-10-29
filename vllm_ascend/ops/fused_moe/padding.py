# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
import torch.nn.functional as F
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import _EXTRA_CTX


def get_valid_token_mask(hidden_states: torch.Tensor, sequence_parallel: bool) -> torch.Tensor | None:
    """Align runner validity with the local, pre-communication MoE rows.

    V2 owns an explicit padding tensor (including graph/ubatch padding). V1
    exposes the reserved MC2 mask. Do not infer validity from zero activations:
    real tokens may also have zero-valued hidden states. Profile dummy rows
    must execute experts so that memory profiling measures the real workload.

    Only a full-sequence table is sliced down to this rank. Under sequence
    parallelism the model entry already shards V2's `is_padding` alongside the
    activations (`sp_padding_mask`), so it arrives covering exactly the local
    rows; V1's reserved `mc2_mask` is still global and does need the slice.
    """
    num_tokens = hidden_states.shape[0]
    if _EXTRA_CTX.in_profile_run:
        return torch.ones(num_tokens, dtype=torch.bool, device=hidden_states.device)
    is_padding = get_forward_context().is_padding
    mask = is_padding if is_padding is not None else _EXTRA_CTX.mc2_mask
    if mask is None:
        return None
    # Slicing an already-local mask would index past its end on every rank
    # but 0, leaving an empty tensor that pads into "every row is padding".
    is_global_table = mask.shape[0] > num_tokens
    start = get_tensor_model_parallel_rank() * num_tokens if sequence_parallel and is_global_table else 0
    mask = mask[start : start + num_tokens]
    if is_padding is not None:
        mask = ~mask
    if mask.shape[0] < num_tokens:
        mask = F.pad(mask, (0, num_tokens - mask.shape[0]), value=False)
    return mask


def mask_padding_routes(
    topk_weights: torch.Tensor, topk_ids: torch.Tensor, valid_mask: torch.Tensor | None, invalid_expert: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use a backend-specific, out-of-range sentinel only at dispatch.

    Keep valid expert IDs through EPLB / expert-map indexing. NPU backends do
    not share DeepEP's negative-ID contract; each dispatcher chooses a sentinel
    that its sorting/counting path excludes.
    """
    if valid_mask is None:
        return topk_weights, topk_ids
    padding = ~valid_mask.bool().unsqueeze(-1)
    return topk_weights.masked_fill(padding, 0), topk_ids.masked_fill(padding, invalid_expert)
