# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Eager NPU implementation of the upstream XPress refiner's fused passes."""

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.logits_processor import _topk

if TYPE_CHECKING:
    from vllm.model_executor.models.qwen3_xpress import XPressRefinerHead


def _latent_pass(
    head: "XPressRefinerHead",
    block: torch.Tensor,
    tok_am1_ids: torch.Tensor,
    hidden_latent: torch.Tensor,
) -> torch.Tensor:
    """Keep the fp32 intermediates and casts of upstream xpress_latent_pass.

    The upstream _refine_latent reference rounds after each linear/mixer.
    Its fused serving kernel instead keeps the embedding projection, causal
    mixer and residual in fp32, rounding the MLP input/product and final latent.
    """
    buffers = head._fused_buf
    assert buffers is not None
    previous_ids = torch.cat((tok_am1_ids[:, None], block[:, :-1]), dim=1)
    embedded = head.w1(previous_ids)
    latent = hidden_latent.float() + torch.matmul(embedded.float(), buffers["wlat_t"].float())
    mixed = torch.bmm(head.mix_L.float(), latent.permute(2, 1, 0)).permute(2, 1, 0)

    mlp_input = mixed.to(buffers["wg_t"].dtype).float()
    gate = torch.matmul(mlp_input, buffers["wg_t"].float())
    up = torch.matmul(mlp_input, buffers["wu_t"].float())
    product = (gate * torch.sigmoid(gate) * up).to(buffers["wd_t"].dtype)
    refined = mixed + torch.matmul(product.float(), buffers["wd_t"].float())
    # Slot 0 is not read out, but participates in the causal mixer above.
    return refined[:, 1:, :].to(hidden_latent.dtype).contiguous()


@torch.no_grad()
def jacobi_refine_greedy(
    head: "XPressRefinerHead",
    base_logits_full: torch.Tensor,
    h_full: torch.Tensor,
    anchor_ids: torch.Tensor,
    tok_am1_ids: torch.Tensor,
    num_passes: int,
    return_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Resolve a block using the upstream head's loaded/folded weights.

    Return draft IDs [requests, block_size - 1] and optionally the final
    pass's logits [requests * (block_size - 1), candidates or vocabulary].
    The candidate set is fixed from the base logits for all Jacobi passes.
    """
    if base_logits_full.ndim != 3 or h_full.ndim != 3:
        raise ValueError("XPress logits and hidden states must have [requests, block, features] layout.")
    num_reqs, block_size, vocab_size = base_logits_full.shape
    if block_size != head.block_size or vocab_size != head.vocab_size:
        raise ValueError("XPress requires the checkpoint block size and the full target vocabulary.")
    if h_full.shape != (num_reqs, block_size, head.hidden_size):
        raise ValueError("XPress hidden-state shape does not match the checkpoint and logits.")
    if anchor_ids.shape != (num_reqs,) or tok_am1_ids.shape != (num_reqs,):
        raise ValueError("XPress requires one anchor and predecessor token per request.")
    if num_passes < 1:
        raise ValueError("XPress requires at least one Jacobi pass.")
    if not 0 <= head.topc <= vocab_size:
        raise ValueError("xpress_topc must be between 0 and the full vocabulary size.")
    if num_reqs > head.max_num_reqs:
        raise ValueError("XPress request count exceeds the configured max_num_seqs.")
    buffers = head._fused_buf
    if buffers is None:
        raise RuntimeError("XPress refiner weights must be processed before drafting.")

    if num_reqs == 0:
        draft = anchor_ids.new_empty((0, block_size - 1))
        logits = base_logits_full.new_empty((0, head.topc or vocab_size)) if return_logits else None
        return draft, logits

    rows = num_reqs * (block_size - 1)
    hidden_cache = head.hidden_cache(h_full)
    hidden_latent = torch.mm(hidden_cache.reshape(num_reqs * block_size, -1), buffers["whc_t"])
    hidden_latent = hidden_latent.view(num_reqs, block_size, head.rank)
    block = torch.empty(num_reqs, block_size, dtype=torch.long, device=h_full.device)
    block[:, 0] = anchor_ids
    base = base_logits_full[:, 1:, :].reshape(rows, vocab_size)
    block[:, 1:] = base.argmax(dim=-1).view(num_reqs, block_size - 1)

    candidates = None
    if head.topc:
        num_candidates = min(head.topc, vocab_size)
        base, candidates = _topk(base, num_candidates)
        readout = head.w2.weight[candidates].reshape(rows, num_candidates, head.rank)

    logits = None
    for _ in range(num_passes):
        latent = _latent_pass(head, block, tok_am1_ids, hidden_latent).reshape(rows, head.rank)
        if candidates is None:
            bias = torch.mm(latent, buffers["w2_t"])
        else:
            bias = torch.bmm(readout, latent.unsqueeze(-1)).squeeze(-1)
        # Round the sum to the base dtype BEFORE argmax, as in both upstream
        # epilogues. Keeping fp32 here would change ties and the next pass.
        logits = bias.to(base.dtype).add_(base)
        selected = logits.argmax(dim=-1)
        if candidates is not None:
            selected = candidates.gather(1, selected[:, None]).squeeze(1)
        # All positions consume the previous pass before any IDs are replaced.
        block[:, 1:] = selected.view(num_reqs, block_size - 1)

    return block[:, 1:], logits if return_logits else None
