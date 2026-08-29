# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # type: ignore

__all__ = ["PAD_SLOT_ID", "extract_last_width"]


def extract_last_width(x, start_loc, width):
    end_loc = start_loc[1:]
    offsets = torch.arange(width, device=x.device)
    indices = end_loc.unsqueeze(1) - width + offsets.unsqueeze(0)  # (num_seqs, width)

    return x[:, indices].permute(1, 0, 2)


def causal_conv1d_update_npu_DISABLED(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    **kwargs,
):
    """Fully vectorized causal_conv1d_update for NPU (ACL-graph safe).

    Matches the upstream signature used by GLM-5.3 KDA layers. No host syncs
    (no .item()), no data-dependent control flow: varlen batches are padded
    to max_query_len and masked, so the same kernel path runs for every
    sequence and the op can be captured inside an ACL graph.

    x: [batch, dim] decode / [batch, dim, seqlen] / [total, dim] with
       query_start_loc
    conv_state: [num_cache_lines, dim, state_len]
    weight: [dim, width], bias: [dim]
    """
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    device = x.device
    orig_dtype = x.dtype
    x = x.to(conv_state.dtype)

    # ---- Fast path: plain decode (one token per sequence). This is the
    # hottest call site; a handful of broadcast multiply-adds beats the
    # general pad/unfold/einsum pipeline by an order of magnitude.
    if query_start_loc is None and x.dim() == 2:
        B, D = x.shape
        rows = (
            conv_state_indices.to(device)
            if conv_state_indices is not None
            else torch.arange(B, device=device)
        )
        safe_rows = rows.clamp(min=0)
        if weight.shape[0] != D and weight.shape[1] == D:
            weight = weight.transpose(0, 1)
        _, width = weight.shape
        S = width - 1
        st = conv_state[safe_rows]  # [B, D, state_len]
        state_len = st.size(-1)
        # Upstream semantics: the causal window lives in the FIRST width-1
        # columns of the state and only those columns are rewritten.
        win = st[:, :, :S]  # [B, D, S]
        acc = x * weight[:, width - 1]
        for w in range(S):
            acc = acc + win[:, :, w] * weight[:, w]
        if bias is not None:
            acc = acc + bias.to(acc.dtype)
        if activation in ("silu", "swish"):
            acc = torch.nn.functional.silu(acc)
        # Roll the window left by one inside the first S columns.
        new_win = torch.cat([win[:, :, 1:], x.unsqueeze(-1)], dim=-1)  # [B,D,S]
        if state_len > S:
            new_state = torch.cat([new_win, st[:, :, S:]], dim=-1)
        else:
            new_state = new_win
        valid_rows = (rows != PAD_SLOT_ID).view(B, 1, 1)
        writeback = torch.where(valid_rows, new_state, st)
        conv_state[safe_rows] = writeback.to(conv_state.dtype)
        return acc.to(orig_dtype)

    # Normalize inputs to padded [B, L, D] + per-seq lengths + row indices.
    if query_start_loc is not None:
        assert conv_state_indices is not None
        B = conv_state_indices.size(0)
        D = x.size(1)
        L = int(max_query_len) if max_query_len and max_query_len > 0 else 1
        lengths = (query_start_loc[1:] - query_start_loc[:-1]).to(device)
        rows = conv_state_indices.to(device)
        # Scatter the varlen token stream into a padded [B, L, D] tensor.
        # searchsorted keeps every op shape-static (ACL-graph safe); the
        # repeat_interleave variant has data-dependent output shapes and
        # aborts graph capture through aclnnNonzeroV2.
        token_idx = torch.arange(x.size(0), device=device)
        seq_ids = torch.searchsorted(
            query_start_loc[1:].to(device).contiguous(), token_idx, right=True
        )
        cum = token_idx - query_start_loc[:-1].to(device)[seq_ids]
        xp = x.new_zeros(B, L, D)
        xp[seq_ids, cum] = x
        tokens = xp
    else:
        if x.dim() == 2:
            B, D = x.shape
            tokens = x.view(B, 1, D).clone()
            lengths = torch.ones(B, dtype=torch.long, device=device)
        else:
            B, D, L = x.shape
            tokens = x.transpose(1, 2).contiguous()  # [B, L, D]
            lengths = torch.full((B,), L, dtype=torch.long, device=device)
        rows = (
            conv_state_indices.to(device)
            if conv_state_indices is not None
            else torch.arange(B, device=device)
        )

    if num_accepted_tokens is not None:
        lengths = torch.minimum(lengths, num_accepted_tokens.to(device).to(torch.long))

    # weight: [dim, width] (accept transposed)
    if weight.shape[0] != D and weight.shape[1] == D:
        weight = weight.transpose(0, 1)
    weight = weight.contiguous()
    _, width = weight.shape
    S = width - 1

    state_len = conv_state.size(-1)
    # Gather per-sequence states, clamping pad rows to 0 (they are masked out
    # on writeback).
    safe_rows = rows.clamp(min=0)
    st = conv_state[safe_rows]  # [B, D, state_len]
    st = st.transpose(1, 2)  # [B, state_len, D]

    # Sliding window input: state ++ tokens -> [B, S + L, D]
    L = tokens.size(1)
    seq = torch.cat([st[:, :S, :], tokens], dim=1)
    # Windows: [B, L, D, width]
    win = seq.unfold(dimension=1, size=width, step=1)
    # Conv: out[b, l, d] = sum_w win[b, l, d, w] * weight[d, w]
    out = torch.einsum("bldw,dw->bld", win, weight)
    if bias is not None:
        out = out + bias.to(out.dtype)

    # Mask padded / non-accepted positions (leave zeros, no NaNs).
    pos_ar = torch.arange(L, device=device).view(1, L)
    valid = pos_ar < lengths.view(B, 1)
    out = out * valid.unsqueeze(-1).to(out.dtype)

    if activation in ("silu", "swish"):
        out = torch.nn.functional.silu(out)

    # Roll the window forward: new state = last S inputs of (state ++ tokens),
    # counted from each sequence's accepted length so spec-decode rollback
    # keeps the window correct. Only the FIRST S state columns are rewritten
    # (upstream semantics); deeper state_len columns are preserved.
    total_len = S + L
    take_start = lengths + S  # end of accepted window
    take_idx = (take_start - S).view(B, 1) + torch.arange(S, device=device).view(1, S)
    take_idx = take_idx.clamp(max=total_len - 1)
    new_win = torch.gather(seq, 1, take_idx.unsqueeze(-1).expand(B, S, D))  # [B,S,D]
    state_len = conv_state.size(-1)
    if state_len > S:
        new_state = torch.cat([new_win, st[:, S:, :]], dim=1)
    else:
        new_state = new_win

    # Write states back. Boolean-mask indexing has data-dependent shapes, so
    # pad rows are written back unchanged through a full-shape where() instead.
    valid_rows = (rows != PAD_SLOT_ID).view(B, 1, 1)
    writeback = torch.where(valid_rows, new_state, st)
    conv_state[safe_rows] = writeback.transpose(1, 2).to(conv_state.dtype)

    # Return in the caller's layout.
    if query_start_loc is not None:
        result = out[seq_ids, cum].to(orig_dtype)
        return result
    if x.dim() == 2:
        return out.view(B, D).to(orig_dtype)
    return out.transpose(1, 2).to(orig_dtype)
