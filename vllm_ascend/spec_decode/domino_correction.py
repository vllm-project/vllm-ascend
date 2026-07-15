# SPDX-License-Identifier: Apache-2.0
#
# This file is a part of the vllm-ascend project.
#
# Pure-torch Domino correction rollout.
#
# This module intentionally imports ONLY torch (no vllm / torch_npu) so it can
# be unit-tested on a CPU-only machine by loading the file directly (bypassing
# the vllm_ascend package __init__, which hard-imports torch_npu). The proposer
# calls ``domino_correct`` with the draft model (duck-typed - it only needs
# ``lm_head`` / ``logits_processor`` / ``embed_input_ids`` / ``embed_proj`` /
# ``_run_prefix_gru`` / ``pure_draft_prefix_len``).
#
# Algorithm (mirrors domino-hf-code/code/dflash.py spec_generate):
#   * ``base_logits[i] = lm_head(parallel_hiddens[:, i])`` predicts draft token
#     i+1 (the Domino checkpoint is trained with shift_label=true, i.e. next-token
#     prediction, and the proposer samples [bonus, mask_1..mask_{k-1}] so position
#     0 is the bonus/root).
#   * A pure-draft prefix of length ``pure_draft_prefix_len`` is sampled greedily
#     from ``base_logits`` and, together with the bonus token, seeds the GRU.
#   * Remaining positions are refined: ``token_i = argmax(base_logits[i] +
#     embed_proj(cat([z_i, s_i])))``; the GRU state advances with the realized
#     token's embedding.

from __future__ import annotations

import torch


def domino_correct(
    hidden_states: torch.Tensor,
    num_speculative_tokens: int,
    model,
    bonus_token_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Run the Domino recurrent correction over parallel draft hidden states.

    Args:
        hidden_states: ``[B*k, H]`` parallel draft hidden states (after the
            shift_label shift, position 0 per request is the bonus/root hidden).
        num_speculative_tokens: ``k``, the number of draft tokens per request.
        model: duck-typed draft model exposing ``lm_head``, ``logits_processor``,
            ``embed_input_ids``, ``embed_proj``, ``_run_prefix_gru`` and
            ``pure_draft_prefix_len``.
        bonus_token_ids: ``[B]`` last-accepted (root) token per request, used to
            seed the GRU. ``None`` during shape-only dummy runs.

    Returns:
        ``[B*k]`` corrected draft token ids (greedy). The caller reshapes to
        ``[B, k]``.
    """
    k = num_speculative_tokens
    num_indices = hidden_states.shape[0]
    batch_size = num_indices // k
    h = hidden_states.view(batch_size, k, -1)  # [B, k, H]

    # base_logits via the (target-shared) lm_head. [B*k, V] -> [B, k, V]
    base_logits = model.logits_processor(model.lm_head, hidden_states)
    base_logits = base_logits.view(batch_size, k, -1)

    prefix_len = int(getattr(model, "pure_draft_prefix_len", 0))

    out = torch.empty(batch_size, k, dtype=torch.long, device=hidden_states.device)

    # Pure-draft prefix: positions 0..prefix_len-1 (drafts 1..prefix_len).
    if prefix_len > 0:
        prefix_tokens = base_logits[:, :prefix_len, :].argmax(dim=-1)  # [B, prefix_len]
        out[:, :prefix_len] = prefix_tokens
    else:
        prefix_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=hidden_states.device)

    if prefix_len >= k:
        return out.view(-1)

    # Seed the GRU with [bonus_token, draft_1, ..., draft_prefix_len].
    bonus = bonus_token_ids
    if bonus is None or bonus.shape[0] < batch_size:
        bonus = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
    else:
        bonus = bonus[:batch_size]
    seed_tokens = torch.cat([bonus.view(batch_size, 1), prefix_tokens], dim=1)  # [B, 1+p]
    seed_embeds = model.embed_input_ids(seed_tokens)  # [B, 1+p, H]
    _, gru_state = model._run_prefix_gru(seed_embeds)  # [1, B, gru_hidden]

    # Sequential correction: positions prefix_len..k-1 (drafts prefix_len+1..k).
    for i in range(prefix_len, k):
        z_i = h[:, i, :]  # [B, H]
        s = gru_state.squeeze(0).to(z_i.dtype)  # [B, gru_hidden] in model dtype
        bias = model.embed_proj(torch.cat([z_i, s], dim=-1))  # [B, V]
        token = (base_logits[:, i, :] + bias).argmax(dim=-1)  # [B]
        out[:, i] = token
        if i + 1 < k:
            embed = model.embed_input_ids(token).unsqueeze(1)  # [B, 1, H]
            _, gru_state = model._run_prefix_gru(embed, gru_state)  # advance

    return out.view(-1)  # [B*k]


# --- Fused path (triton kernels + precompute) --------------------------------
#
# Replaces the eager embed_proj/nn.GRU with:
#  - a precomputed z-branch (z_part_all = h @ w_z.T for all steps at once);
#  - a precomputed gi lookup table (embed @ W_ih.T), indexed by token id;
#  - fused triton kernels for the per-step (SiLU+GEMV+bias+base+argmax) and the
#    GRU cell update (bf16-safe, bypasses DynamicGRUV2).
# Numerically equivalent to ``domino_correct`` (greedy).


class _DominoCorrectionWeights:
    """Cached, contiguous views of the correction-head weights + gi table."""

    __slots__ = (
        "w_z",
        "w_s",
        "fc1_bias",
        "fc2_weight",
        "fc2_bias",
        "gru_w_hh",
        "gi_table",
        "gru_hidden_dim",
    )

    def __init__(
        self,
        w_z,
        w_s,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        gru_w_hh,
        gi_table,
        gru_hidden_dim,
    ):
        self.w_z = w_z
        self.w_s = w_s
        self.fc1_bias = fc1_bias
        self.fc2_weight = fc2_weight
        self.fc2_bias = fc2_bias
        self.gru_w_hh = gru_w_hh
        self.gi_table = gi_table
        self.gru_hidden_dim = gru_hidden_dim


def _get_domino_correction_weights(model) -> _DominoCorrectionWeights:
    """Extract and cache the fused-correction weights on the draft model.

    The gi lookup table (``embed @ W_ih.T`` for every token) is precomputed once
    from the target-shared ``embed_tokens``; it is large (~vocab x 3*gru_hidden)
    but avoids a per-step embed + GEMV.
    """
    cached = getattr(model, "_domino_corr_weights", None)
    if cached is not None:
        return cached

    import torch.nn.functional as F

    fc1 = model.embed_proj[0]  # Linear(H+gru -> M, bias=False)
    fc2 = model.embed_proj[2]  # Linear(M -> V, bias=False)
    # ``config`` is the hf_config on the real model; the UT stub exposes
    # ``hidden_size`` directly.
    hidden_size = getattr(getattr(model, "config", model), "hidden_size")
    gru_hidden_dim = model.prefix_gru.hidden_size

    # fc1.weight is [M, H+gru]; split into the z-branch [:, :H] and s-branch [:, H:].
    w_z = fc1.weight[:, :hidden_size].detach().contiguous()
    w_s = fc1.weight[:, hidden_size:].detach().contiguous()
    fc1_bias = fc1.bias.detach().contiguous() if fc1.bias is not None else None
    fc2_weight = fc2.weight.detach().contiguous()
    fc2_bias = fc2.bias.detach().contiguous() if fc2.bias is not None else None

    # GRU is bias=False: only weight_ih_l0 / weight_hh_l0. weight_ih is folded
    # into the gi lookup table; only weight_hh is needed per step.
    gru_w_hh = model.prefix_gru.weight_hh_l0.detach().contiguous()
    gru_w_ih = model.prefix_gru.weight_ih_l0.detach()
    # embed_tokens lives on the inner DFlashQwen3Model (model.model.embed_tokens)
    # for the real model; the UT stub exposes it directly on `model`.
    inner = getattr(model, "model", model)
    embed_weight = inner.embed_tokens.weight  # target-shared
    gi_table = F.linear(embed_weight, gru_w_ih).contiguous()

    weights = _DominoCorrectionWeights(
        w_z, w_s, fc1_bias, fc2_weight, fc2_bias, gru_w_hh, gi_table, gru_hidden_dim
    )
    model._domino_corr_weights = weights
    return weights


def domino_correct_fused(
    hidden_states: torch.Tensor,
    num_speculative_tokens: int,
    model,
    bonus_token_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Fused Domino correction (triton kernels + precompute).

    Each correction step is 2 kernel launches (fused mlp+bias+base+argmax +
    fused GRU cell) instead of ~5-6 eager kernels, and the GRU bypasses Ascend's
    fp16-only DynamicGRUV2 (bf16-safe). The z-branch and gi lookup table are
    precomputed.
    """
    # Lazy import so this module stays importable on CPU (UT) without triton.
    from vllm_ascend.ops.triton.spec_decode.domino import (
        fused_gru_cell,
        fused_mlp_bias_argmax,
    )

    import torch.nn.functional as F

    k = num_speculative_tokens
    num_indices = hidden_states.shape[0]
    batch_size = num_indices // k
    h = hidden_states.view(batch_size, k, -1)  # [B, k, H]
    device = hidden_states.device
    dtype = hidden_states.dtype

    weights = _get_domino_correction_weights(model)

    base_logits = model.logits_processor(model.lm_head, hidden_states)
    base_logits = base_logits.view(batch_size, k, -1)  # [B, k, V]

    # Precompute the z-branch for ALL steps at once (z is known up front).
    z_part_all = F.linear(h, weights.w_z, weights.fc1_bias)  # [B, k, M]

    prefix_len = int(getattr(model, "pure_draft_prefix_len", 0))
    out = torch.empty(batch_size, k, dtype=torch.long, device=device)

    if prefix_len > 0:
        prefix_tokens = base_logits[:, :prefix_len, :].argmax(dim=-1)  # [B, p]
        out[:, :prefix_len] = prefix_tokens
    else:
        prefix_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=device)

    if prefix_len >= k:
        return out.view(-1)

    # Seed the GRU with [bonus_token, prefix_tokens], via the fused cell.
    # bonus is a fixed buffer (aclgraph-safe); slice to the batch size.
    bonus = bonus_token_ids[:batch_size]
    seed_tokens = torch.cat([bonus.view(batch_size, 1), prefix_tokens], dim=1)  # [B, 1+p]

    gru_state = torch.zeros(batch_size, weights.gru_hidden_dim, dtype=dtype, device=device)
    for t in range(seed_tokens.shape[1]):
        gi = weights.gi_table[seed_tokens[:, t]]  # [B, 3*gru]
        gh = F.linear(gru_state, weights.gru_w_hh)  # [B, 3*gru]
        gru_state = fused_gru_cell(gi, gh, gru_state)  # [B, gru]

    # Sequential correction (vectorized over the batch).
    for i in range(prefix_len, k):
        s_proj = F.linear(gru_state, weights.w_s)  # [B, M]
        token = fused_mlp_bias_argmax(
            z_part_all[:, i, :],  # [B, M]
            s_proj,
            base_logits[:, i, :],  # [B, V]
            weights.fc2_weight,
            weights.fc2_bias,
        )  # [B]
        out[:, i] = token
        if i + 1 < k:
            gi = weights.gi_table[token]
            gh = F.linear(gru_state, weights.gru_w_hh)
            gru_state = fused_gru_cell(gi, gh, gru_state)

    return out.view(-1)  # [B*k]
