# SPDX-License-Identifier: Apache-2.0

import os
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # type: ignore

__all__ = ["PAD_SLOT_ID", "extract_last_width"]


def extract_last_width(x, start_loc, width):
    end_loc = start_loc[1:]
    offsets = torch.arange(width, device=x.device)
    indices = end_loc.unsqueeze(1) - width + offsets.unsqueeze(0)  # (num_seqs, width)

    return x[:, indices].permute(1, 0, 2)


_CONV_CUSTOM_AVAILABLE: bool | None = None
_CONV_SYNC_N = 0


def causal_conv1d_update_ascendc(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    **kwargs,
):
    """causal_conv1d_update backed by the AscendC fused operator.

    Routes the hot plain-decode path (one token per sequence) to
    ``torch.ops._C_ascend.npu_causal_conv1d_custom`` — the same single fused
    AscendC kernel the upstream Qwen3-Next/Kimi GDN path uses (1 launch, no
    host sync, ACL-graph capturable). Every other shape (spec verify, 3D
    inputs) and any operator failure delegates to the vectorized torch
    implementation.

    Operator contract (validated against a unit-weight probe):
      x:            (batch, dim)                    [run_mode=1]
      weight:       (width, dim), width in [2, 4]
      conv_state:   (num_cache_lines, state_len, dim) — vLLM's default "SD"
                    layout, passed through without transposing.
      semantics:    y = sum_j weight[j] * window[j] (elementwise per channel),
                    window then shifts by appending x; state lines selected
                    via cache_indices are updated in place, PAD lines skipped.
    """
    global _CONV_CUSTOM_AVAILABLE

    if num_accepted_tokens is None and query_start_loc is None and x.dim() == 2:
        if _CONV_CUSTOM_AVAILABLE is not False:
            if isinstance(activation, bool):
                activation = "silu" if activation else None
            B, D = x.shape
            orig_dtype = x.dtype
            # The aclnn kernel requires x/state in the same dtype; the mamba
            # pool cache may be fp32 while activations are bf16 (the 310P
            # fallback casts internally as well).
            x_c = x.to(conv_state.dtype).contiguous()
            # The operator requires conv_state (num_cache_lines, state_len, dim)
            # and updates the rows selected by cache_indices in place. The
            # aliasing KV pool hands us a NON-CONTIGUOUS DS-layout view of the
            # whole pool; transposing + copying the full pool costs ~0.2ms and
            # writing the full pool back through the strided view up to
            # ~3.7ms per call (measured). Instead, gather ONLY the selected
            # rows into a small contiguous SD pool, run the operator on it,
            # and scatter the (few) updated rows back with plain slice
            # assignments (0.02ms each).
            small_pool = None
            if conv_state_indices is not None:
                idx_src = conv_state_indices
                rows = idx_src.to(torch.long).clamp(min=0)
                sub_ds = conv_state[rows]                       # [B, D, S] gather
                conv_state_t = sub_ds.transpose(1, 2).contiguous()  # [B, S, D]
                small_pool = conv_state_t
                indices_c = torch.arange(
                    rows.numel(), dtype=idx_src.dtype, device=idx_src.device
                )
            elif conv_state.dim() == 3 and conv_state.shape[-2] == D and conv_state.shape[-1] != D:
                conv_state_t = conv_state.transpose(1, 2).contiguous()
                indices_c = None
            else:
                conv_state_t = conv_state.contiguous()
                indices_c = None
            _writeback_rows = (
                rows if small_pool is not None else None
            )
            if weight.shape[0] == D:
                # vLLM hands us (dim, width); the operator wants (width, dim).
                # GLM-5.3-Flash keeps its conv weight in fp32 — the operator's
                # SupportInfo requires every tensor in one dtype (fp16 or
                # bf16), so cast to the cache dtype.
                weight_t = weight.to(conv_state.dtype).transpose(0, 1).contiguous()
            else:
                weight_t = weight.to(conv_state.dtype).contiguous()
            bias_c = (
                bias.to(conv_state.dtype).contiguous() if bias is not None else None
            )
            if small_pool is None:
                indices_c = (
                    conv_state_indices.contiguous()
                    if conv_state_indices is not None
                    else None
                )
            activation_mode = 1 if activation in ("silu", "swish") else 0
            out = torch.empty_like(x_c)
            # DIAGNOSTIC: snapshot inputs BEFORE the op mutates the pool, plus
            # an on-the-spot 310P reference computed on the same snapshot.
            _dump = None
            if os.environ.get("GLM53_CONV_DUMP") == "1" and globals().get("_CONV_SYNC_N", 0) < 8:
                _pre_state = conv_state_t.clone()
                _ref_out = None
                try:
                    from vllm_ascend._310p.ops.causal_conv1d import (
                        causal_conv1d_update as _p310,
                    )

                    _w310 = (
                        weight.to(torch.float32).transpose(0, 1).contiguous()
                        if weight.shape[0] == D
                        else weight.to(torch.float32).contiguous()
                    )
                    _ref_out = _p310(
                        x.to(_pre_state.dtype),
                        _pre_state.transpose(1, 2).contiguous()
                        if _pre_state.shape[-2] != D
                        else _pre_state.clone(),
                        _w310,
                        bias.to(torch.float32) if bias is not None else None,
                        activation_mode == 1,
                        conv_state_indices,
                        None, None,
                    )
                    torch.npu.synchronize()
                except Exception as _ref_err:
                    _ref_out = repr(_ref_err)[:200]
                _dump = {
                    "x": x_c, "conv_state_sd_pre": _pre_state,
                    "weight": weight_t, "indices": indices_c,
                    "bias": bias_c, "act": activation_mode,
                }
            try:
                result = torch.ops._C_ascend.npu_causal_conv1d_custom(
                    out,
                    x_c,
                    weight_t,
                    conv_state=conv_state_t,
                    bias_opt=bias_c,
                    query_start_loc_opt=None,
                    cache_indices_opt=indices_c,
                    initial_state_mode_opt=None,
                    num_accepted_tokens_opt=None,
                    activation_mode=activation_mode,
                    pad_slot_id=PAD_SLOT_ID,
                    run_mode=1,
                )
                if small_pool is not None:
                    # Scatter the updated rows back into the pool view.
                    # ACL-graph capture forbids host syncs (tolist), so the
                    # capture path uses tensorized scatter; eager uses plain
                    # slice assignments (0.02ms each, ~200x faster than
                    # advanced indexing into the strided view).
                    if torch.npu.is_current_stream_capturing():
                        keep = indices_c != PAD_SLOT_ID
                        safe_idx = torch.where(
                            keep, indices_c.to(torch.long),
                            torch.zeros_like(indices_c, dtype=torch.long),
                        )
                        safe_rows = small_pool * keep.view(-1, 1, 1)
                        conv_state.index_copy_(
                            0, safe_idx, safe_rows.transpose(1, 2)
                        )
                    else:
                        valid_host = [
                            int(i)
                            for i, r in zip(
                                _writeback_rows.tolist(),
                                indices_c.tolist(),
                            )
                            if int(r) != PAD_SLOT_ID
                        ]
                        for b, i in enumerate(valid_host):
                            conv_state[i] = small_pool[b].t()
                elif conv_state_t.data_ptr() != conv_state.data_ptr() or not conv_state.is_contiguous():
                    conv_state.copy_(conv_state_t.transpose(1, 2))
                if not _CONV_CUSTOM_AVAILABLE:
                    import sys as _sys
                    print("[conv-ascendc] active (first call ok)", flush=True, file=_sys.stderr)
                _CONV_CUSTOM_AVAILABLE = True
                if _dump is not None:
                    globals()["_CONV_SYNC_N"] = globals().get("_CONV_SYNC_N", 0) + 1
                    _dump["out"] = result
                    _dump["conv_state_sd_post"] = conv_state_t.clone()
                    torch.save(_dump, f"/tmp/conv_dump_{_CONV_SYNC_N}.pt")
                return result.to(orig_dtype)
            except Exception as _op_err:
                if _CONV_CUSTOM_AVAILABLE is True:
                    raise
                import sys as _sys
                print(
                    "[conv-ascendc] op failed, falling back:",
                    repr(_op_err)[:1200],
                    "| x:", tuple(x.shape), x.dtype, "contig" if x.is_contiguous() else "noncontig",
                    "| st:", tuple(conv_state.shape), conv_state.dtype, "contig" if conv_state.is_contiguous() else "noncontig",
                    "| w:", tuple(weight.shape), weight.dtype,
                    flush=True, file=_sys.stderr,
                )
                # Operator missing/broken on this build: latch and fall back.
                _CONV_CUSTOM_AVAILABLE = False

    return causal_conv1d_update_npu(
        x, conv_state, weight, bias, activation, conv_state_indices,
        num_accepted_tokens, query_start_loc, **kwargs,
    )


def causal_conv1d_update_npu(
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

    No host syncs and no data-dependent control flow: varlen batches are
    padded to max_query_len and masked, so the op can be captured inside an
    ACL graph. State semantics follow the upstream kernel: the causal window
    lives in the FIRST width-1 columns and only those columns are rewritten.
    """
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID as _PAD

    if isinstance(activation, bool):
        activation = "silu" if activation else None

    device = x.device
    orig_dtype = x.dtype
    x = x.to(conv_state.dtype)

    # Fast path: plain decode (one token per sequence) — the hottest call
    # site; a handful of broadcast multiply-adds.
    if query_start_loc is None and x.dim() == 2:
        import os as _os, time as _time

        _timing = _os.environ.get("GLM53_TIME_CONV") == "1"
        _t0 = _time.perf_counter() if _timing else 0.0
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
        st = conv_state[safe_rows]
        state_len = st.size(-1)
        win = st[:, :, :S]
        acc = x * weight[:, width - 1]
        for w in range(S):
            acc = acc + win[:, :, w] * weight[:, w]
        if bias is not None:
            acc = acc + bias.to(acc.dtype)
        if activation in ("silu", "swish"):
            acc = torch.nn.functional.silu(acc)
        new_win = torch.cat([win[:, :, 1:], x.unsqueeze(-1)], dim=-1)
        new_state = (
            torch.cat([new_win, st[:, :, S:]], dim=-1)
            if state_len > S
            else new_win
        )
        valid_rows = (rows != _PAD).view(B, 1, 1)
        conv_state[safe_rows] = torch.where(valid_rows, new_state, st).to(
            conv_state.dtype
        )
        if _timing:
            global _CONV_N, _CONV_T
            try:
                _CONV_N += 1
                _CONV_T += _time.perf_counter() - _t0
            except NameError:
                _CONV_N, _CONV_T = 1, _time.perf_counter() - _t0
            if _CONV_N % 340 == 0:
                import sys as _sys

                print(
                    f"[conv-time] n={_CONV_N} avg={_CONV_T/_CONV_N*1000:.3f}ms "
                    f"B={B} D={D}",
                    file=_sys.stderr,
                    flush=True,
                )
                _CONV_N = _CONV_T = 0
        return acc.to(orig_dtype)

    # General varlen path (spec verify / multi-token decode).
    if query_start_loc is not None:
        assert conv_state_indices is not None
        B = conv_state_indices.size(0)
        D = x.size(1)
        L = int(max_query_len) if max_query_len and max_query_len > 0 else 1
        lengths = (query_start_loc[1:] - query_start_loc[:-1]).to(device)
        rows = conv_state_indices.to(device)
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
            tokens = x.transpose(1, 2).contiguous()
            lengths = torch.full((B,), L, dtype=torch.long, device=device)
        rows = (
            conv_state_indices.to(device)
            if conv_state_indices is not None
            else torch.arange(B, device=device)
        )

    if num_accepted_tokens is not None:
        lengths = torch.minimum(
            lengths, num_accepted_tokens.to(device).to(torch.long)
        )

    if weight.shape[0] != D and weight.shape[1] == D:
        weight = weight.transpose(0, 1)
    weight = weight.contiguous()
    _, width = weight.shape
    S = width - 1

    state_len = conv_state.size(-1)
    safe_rows = rows.clamp(min=0)
    st = conv_state[safe_rows].transpose(1, 2)  # [B, state_len, D]

    L = tokens.size(1)
    seq = torch.cat([st[:, :S, :], tokens], dim=1)
    win = seq.unfold(dimension=1, size=width, step=1)
    out = torch.einsum("bldw,dw->bld", win, weight)
    if bias is not None:
        out = out + bias.to(out.dtype)

    pos_ar = torch.arange(L, device=device).view(1, L)
    valid = pos_ar < lengths.view(B, 1)
    out = out * valid.unsqueeze(-1).to(out.dtype)
    if activation in ("silu", "swish"):
        out = torch.nn.functional.silu(out)

    total_len = S + L
    take_start = lengths + S
    take_idx = (take_start - S).view(B, 1) + torch.arange(S, device=device).view(1, S)
    take_idx = take_idx.clamp(max=total_len - 1)
    new_win = torch.gather(seq, 1, take_idx.unsqueeze(-1).expand(B, S, D))
    new_state = (
        torch.cat([new_win, st[:, S:, :]], dim=1)
        if state_len > S
        else new_win
    )
    valid_rows = (rows != _PAD).view(B, 1, 1)
    writeback = torch.where(valid_rows, new_state, st)
    conv_state[safe_rows] = writeback.transpose(1, 2).to(conv_state.dtype)

    if query_start_loc is not None:
        return out[seq_ids, cum].to(orig_dtype)
    if x.dim() == 2:
        return out.view(B, D).to(orig_dtype)
    return out.transpose(1, 2).to(orig_dtype)
