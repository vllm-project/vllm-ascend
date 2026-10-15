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
#     i+1. The proposer already selects [bonus, mask_1..mask_{k-1}] for
#     shift_label=true or [mask_1..mask_k] for false. Correction is identical
#     after this alignment, including the bonus token used to seed the GRU.
#   * A pure-draft prefix of length ``pure_draft_prefix_len`` is sampled greedily
#     from ``base_logits`` and, together with the bonus token, seeds the GRU.
#   * Remaining positions are refined: ``token_i = argmax(base_logits[i] +
#     embed_proj(cat([z_i, s_i])))``; the GRU state advances with the realized
#     token's embedding.

from __future__ import annotations

import torch

_GI_TABLE_CHUNK_SIZE = 4096


def _domino_base_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Return global vocabulary logits on every TP rank, regardless of reduce sampling."""
    if getattr(model.lm_head, "tp_size", 1) == 1:
        return model.logits_processor(model.lm_head, hidden_states)

    # AscendLogitsProcessor intentionally leaves logits sharded when reduce
    # sampling is enabled. Domino needs all scores BEFORE adding correction.
    from vllm.distributed import tensor_model_parallel_all_gather

    processor = model.logits_processor
    logits = model.lm_head.quant_method.apply(model.lm_head, hidden_states)
    logits = tensor_model_parallel_all_gather(logits, dim=-1)
    logits = logits[..., : processor.org_vocab_size].contiguous()
    if processor.soft_cap is not None:
        logits = torch.tanh(logits / processor.soft_cap) * processor.soft_cap
    if processor.scale != 1.0:
        logits *= processor.scale
    return logits


@torch.no_grad()
def _build_domino_gi_table(model, embed_tokens, gru_w_ih, vocab_size):
    if getattr(embed_tokens, "tp_size", 1) == 1:
        return torch.nn.functional.linear(embed_tokens.weight, gru_w_ih).contiguous()

    # Use the parallel embedding API to resolve global IDs and padding. Build
    # in chunks so no full [V,H] embedding replica is ever materialized.
    table = torch.empty((vocab_size, gru_w_ih.shape[0]), device=gru_w_ih.device, dtype=gru_w_ih.dtype)
    for start in range(0, vocab_size, _GI_TABLE_CHUNK_SIZE):
        end = min(start + _GI_TABLE_CHUNK_SIZE, vocab_size)
        ids = torch.arange(start, end, device=gru_w_ih.device)
        embeddings = model.embed_input_ids(ids)
        table[start:end].copy_(torch.nn.functional.linear(embeddings, gru_w_ih))
    return table


def _candidate_greedy_argmax(
    candidate_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Select the lowest token id among maximum-score candidates."""
    max_scores = candidate_scores.max(dim=-1, keepdim=True).values
    sentinel = torch.iinfo(candidate_ids.dtype).max
    tied_ids = torch.where(candidate_scores == max_scores, candidate_ids, sentinel)
    return tied_ids.min(dim=-1).values


def domino_correct(
    hidden_states: torch.Tensor,
    num_speculative_tokens: int,
    model,
    bonus_token_ids: torch.Tensor | None,
    candidate_topk: int = 0,
) -> torch.Tensor:
    """Run the Domino recurrent correction over parallel draft hidden states.

    Args:
        hidden_states: ``[B*k, H]`` parallel draft hidden states, already
            aligned by the proposer so row i predicts draft token i+1.
        num_speculative_tokens: ``k``, the number of draft tokens per request.
        model: duck-typed draft model exposing ``lm_head``, ``logits_processor``,
            ``embed_input_ids``, ``embed_proj``, ``_run_prefix_gru`` and
            ``pure_draft_prefix_len``.
        bonus_token_ids: ``[B]`` last-accepted (root) token per request, used to
            seed the GRU. ``None`` during shape-only dummy runs.
        candidate_topk: 0 selects full-vocabulary correction. A positive value
            restricts each corrected position to the base-logit top-k set. This
            eager fallback retains the full correction projection, but preserves
            the same candidate-only semantics as the fused NPU path.

    Returns:
        ``[B*k]`` corrected draft token ids (greedy). The caller reshapes to
        ``[B, k]``.
    """
    k = num_speculative_tokens
    num_indices = hidden_states.shape[0]
    batch_size = num_indices // k
    h = hidden_states.view(batch_size, k, -1)  # [B, k, H]

    # base_logits via the (target-shared) lm_head. [B*k, V] -> [B, k, V]
    base_logits = _domino_base_logits(model, hidden_states)
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
        corrected_logits = base_logits[:, i, :] + bias
        if candidate_topk:
            base_top_values, candidate_ids = torch.topk(base_logits[:, i, :], k=candidate_topk, dim=-1)
            candidate_bias = bias.gather(dim=-1, index=candidate_ids)
            token = _candidate_greedy_argmax(
                base_top_values + candidate_bias,
                candidate_ids,
            )
        else:
            token = corrected_logits.argmax(dim=-1)  # [B]
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

    fc1 = model.embed_proj[0]  # Linear(H+gru -> M, bias=False)
    fc2 = model.embed_proj[2]  # Linear(M -> V, bias=False)
    # ``config`` is the hf_config on the real model; the UT stub exposes
    # ``hidden_size`` directly.
    config = getattr(model, "config", model)
    hidden_size = config.hidden_size
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
    gi_table = _build_domino_gi_table(model, inner.embed_tokens, gru_w_ih, fc2_weight.shape[0])

    weights = _DominoCorrectionWeights(w_z, w_s, fc1_bias, fc2_weight, fc2_bias, gru_w_hh, gi_table, gru_hidden_dim)
    model._domino_corr_weights = weights
    return weights


def domino_correct_fused(
    hidden_states: torch.Tensor,
    num_speculative_tokens: int,
    model,
    bonus_token_ids: torch.Tensor | None,
    candidate_topk: int = 0,
) -> torch.Tensor:
    """Fused Domino correction (triton kernels + precompute).

    The correction argmax writes directly to the output token column, and the
    GRU reads gi from the lookup table in-kernel, eliminating standalone token
    copy and gi-gather kernels. The GRU bypasses Ascend's fp16-only
    DynamicGRUV2 (bf16-safe). The z-branch and gi lookup table are precomputed.

    The fused path requires the fixed bonus-token buffer used during graph
    capture and replay; unlike the eager reference, it cannot omit this buffer.
    """
    if bonus_token_ids is None:
        raise ValueError("Fused Domino correction requires a bonus-token buffer.")

    import torch.nn.functional as F

    # Lazy import so this module stays importable on CPU (UT) without triton.
    from vllm_ascend.ops.triton.spec_decode.domino import (
        fused_gru_cell_from_tokens,
        fused_mlp_bias_argmax,
    )

    if candidate_topk:
        from vllm_ascend.ops.triton.spec_decode.domino_candidate import (
            fused_candidate_mlp_argmax,
        )

    k = num_speculative_tokens
    num_indices = hidden_states.shape[0]
    batch_size = num_indices // k
    h = hidden_states.view(batch_size, k, -1)  # [B, k, H]
    device = hidden_states.device
    dtype = hidden_states.dtype

    weights = _get_domino_correction_weights(model)

    base_logits = _domino_base_logits(model, hidden_states)
    base_logits = base_logits.view(batch_size, k, -1)  # [B, k, V]
    if candidate_topk:
        if candidate_topk > base_logits.shape[-1]:
            raise ValueError(
                "Domino candidate_topk must not exceed the vocabulary size; "
                f"got {candidate_topk} for vocab_size={base_logits.shape[-1]}."
            )
        if candidate_topk & (candidate_topk - 1):
            raise ValueError(f"Domino candidate_topk must be a positive power of two; got {candidate_topk}.")

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
        gh = F.linear(gru_state, weights.gru_w_hh)  # [B, 3*gru]
        gru_state = fused_gru_cell_from_tokens(weights.gi_table, seed_tokens[:, t], gh, gru_state)  # [B, gru]

    # Sequential correction (vectorized over the batch).
    for i in range(prefix_len, k):
        s_proj = F.linear(gru_state, weights.w_s)  # [B, M]
        if candidate_topk:
            base_top_values, candidate_ids = torch.topk(base_logits[:, i, :], k=candidate_topk, dim=-1)
            token = fused_candidate_mlp_argmax(
                z_part_all[:, i, :],
                s_proj,
                base_top_values,
                candidate_ids,
                weights.fc2_weight,
                weights.fc2_bias,
                out_tokens=out[:, i],
            )
        else:
            token = fused_mlp_bias_argmax(
                z_part_all[:, i, :],  # [B, M]
                s_proj,
                base_logits[:, i, :],  # [B, V]
                weights.fc2_weight,
                weights.fc2_bias,
                out_tokens=out[:, i],
            )  # [B]
        if i + 1 < k:
            gh = F.linear(gru_state, weights.gru_w_hh)
            gru_state = fused_gru_cell_from_tokens(weights.gi_table, token, gh, gru_state)

    return out.view(-1)  # [B*k]
