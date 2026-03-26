import torch


def _maybe_l2norm(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return x
    return x / (torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True)) + 1e-6)


def _expand_to_hv(x: torch.Tensor, hv: int) -> torch.Tensor:
    """Expand [H, ...] to [HV, ...] for grouped-value-attention semantics."""
    h = x.shape[0]
    if h == hv:
        return x
    if hv % h != 0:
        raise ValueError(f"Cannot expand head dim from {h} to {hv}.")
    return x.repeat_interleave(hv // h, dim=0)


def _infer_num_states(
    default_n: int,
    initial_state: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
) -> int:
    if initial_state is not None:
        return initial_state.shape[0]
    if ssm_state_indices is None:
        return default_n
    nonneg = ssm_state_indices[ssm_state_indices >= 0]
    if nonneg.numel() == 0:
        return default_n
    return int(nonneg.max().item()) + 1


def _state_index(
    seq_idx: int,
    tok_idx: int,
    ssm_state_indices: torch.Tensor | None,
) -> int:
    if ssm_state_indices is None:
        return seq_idx
    if ssm_state_indices.ndim == 1:
        return int(ssm_state_indices[seq_idx].item())
    return int(ssm_state_indices[seq_idx, tok_idx].item())


def _run_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor | None,
    states: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    use_initial_state: bool,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference PyTorch recurrence for GDN delta rule.

    Shapes follow fla.ops conventions:
    q,k: [B, T, H, K]
    v:   [B, T, HV, V]
    g,beta: [B, T, HV] (beta may also be [B, T, HV, V])
    states: [N_state, HV, K, V]
    """
    B, T, _, Kdim = k.shape
    HV = v.shape[2]
    Vdim = v.shape[-1]

    if cu_seqlens is not None and B != 1:
        raise ValueError("Variable-length mode expects batch size B=1.")

    out = torch.zeros_like(v)

    if cu_seqlens is None:
        seq_ranges = [(i, 0, T) for i in range(B)]
    else:
        n_seq = len(cu_seqlens) - 1
        seq_ranges = [
            (
                i,
                int(cu_seqlens[i].item()),
                int(cu_seqlens[i + 1].item()),
            )
            for i in range(n_seq)
        ]

    for seq_idx, start, end in seq_ranges:
        seq_len = end - start
        if seq_len <= 0:
            continue

        accepted = None
        if num_accepted_tokens is not None:
            accepted = int(num_accepted_tokens[seq_idx].item())
            seq_len = min(seq_len, accepted)
        if seq_len <= 0:
            continue

        if use_initial_state:
            if ssm_state_indices is None:
                init_state_idx = seq_idx
            else:
                init_tok = (accepted - 1) if accepted is not None else 0
                init_state_idx = _state_index(seq_idx, init_tok, ssm_state_indices)
            if init_state_idx < 0:
                # Match triton behavior for invalid PAD_SLOT_ID in continuous batching.
                continue
            if init_state_idx >= states.shape[0]:
                raise IndexError(f"state_idx {init_state_idx} out of range for states size {states.shape[0]}")
            h_t = states[init_state_idx].transpose(-1, -2).to(torch.float32)
        else:
            h_t = torch.zeros(HV, Vdim, Kdim, dtype=torch.float32, device=q.device)

        for rel_t in range(seq_len):
            tok = start + rel_t

            if cu_seqlens is None:
                q_t = q[seq_idx, tok]
                k_t = k[seq_idx, tok]
                v_t = v[seq_idx, tok]
                g_t = g[seq_idx, tok] if g is not None else None
                beta_t = beta[seq_idx, tok] if beta is not None else None
            else:
                q_t = q[0, tok]
                k_t = k[0, tok]
                v_t = v[0, tok]
                g_t = g[0, tok] if g is not None else None
                beta_t = beta[0, tok] if beta is not None else None

            # Match Triton kernel math: load to fp32 first, then apply l2norm.
            q_t = q_t.to(torch.float32)
            k_t = k_t.to(torch.float32)
            q_t = _maybe_l2norm(q_t, use_qk_l2norm_in_kernel)
            k_t = _maybe_l2norm(k_t, use_qk_l2norm_in_kernel)
            v_t = v_t.to(torch.float32)
            q_t = q_t * scale

            q_hv = _expand_to_hv(q_t, HV)
            k_hv = _expand_to_hv(k_t, HV)

            if g_t is not None:
                g_t = g_t.to(torch.float32)
                if g_t.ndim == 0:
                    g_t = g_t.expand(HV)
                elif g_t.shape[0] != HV:
                    g_t = _expand_to_hv(g_t.unsqueeze(-1), HV).squeeze(-1)
                h_t = h_t * torch.exp(g_t).view(HV, 1, 1)

            v_t = v_t - torch.sum(h_t * k_hv.unsqueeze(-2), dim=-1)

            if beta_t is not None:
                beta_t = beta_t.to(torch.float32)
                if beta_t.ndim == 1:
                    if beta_t.shape[0] != HV:
                        beta_t = _expand_to_hv(beta_t.unsqueeze(-1), HV).squeeze(-1)
                    v_t = v_t * beta_t.view(HV, 1)
                else:
                    if beta_t.shape[0] != HV:
                        beta_t = _expand_to_hv(beta_t, HV)
                    v_t = v_t * beta_t

            h_t = h_t + v_t.unsqueeze(-1) * k_hv.unsqueeze(-2)
            o_t = torch.sum(h_t * q_hv.unsqueeze(-2), dim=-1)

            if cu_seqlens is None:
                out[seq_idx, tok] = o_t.to(out.dtype)
            else:
                out[0, tok] = o_t.to(out.dtype)

            state_idx = _state_index(seq_idx, rel_t, ssm_state_indices)
            if state_idx >= 0:
                if state_idx >= states.shape[0]:
                    raise IndexError(f"state_idx {state_idx} out of range for states size {states.shape[0]}")
                states[state_idx] = h_t.transpose(-1, -2).to(states.dtype)

    return out, states


def fused_recurrent_gated_delta_rule_pytorch(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    inplace_final_state=False,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
):
    """PyTorch fallback for fused_recurrent_gated_delta_rule."""
    B, _, _, Kdim = k.shape
    HV = v.shape[2]
    Vdim = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    n_states = _infer_num_states(N, initial_state, ssm_state_indices)
    if initial_state is not None:
        states = initial_state if inplace_final_state else initial_state.clone()
    else:
        states = torch.zeros(n_states, HV, Kdim, Vdim, dtype=q.dtype, device=q.device)

    scale = Kdim**-0.5
    out, states = _run_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        states=states,
        scale=scale,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_initial_state=initial_state is not None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    return out, states
