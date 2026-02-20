#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

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
    B, T, H, Kdim = k.shape
    HV = v.shape[2]

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

        if num_accepted_tokens is not None:
            accepted = int(num_accepted_tokens[seq_idx].item())
            seq_len = min(seq_len, accepted)

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

            state_idx = _state_index(seq_idx, rel_t, ssm_state_indices)
            persist_state = state_idx >= 0
            if persist_state:
                if state_idx >= states.shape[0]:
                    raise IndexError(
                        f"state_idx {state_idx} out of range for states size {states.shape[0]}"
                    )
                h_t = states[state_idx]
            else:
                h_t = torch.zeros(HV, Kdim, v.shape[-1], dtype=q.dtype, device=q.device)

            q_t = _maybe_l2norm(q_t, use_qk_l2norm_in_kernel)
            k_t = _maybe_l2norm(k_t, use_qk_l2norm_in_kernel)
            q_t = q_t * scale

            q_hv = _expand_to_hv(q_t, HV)
            k_hv = _expand_to_hv(k_t, HV)

            if g_t is not None:
                if g_t.ndim == 0:
                    g_t = g_t.expand(HV)
                elif g_t.shape[0] != HV:
                    g_t = _expand_to_hv(g_t.unsqueeze(-1), HV).squeeze(-1)
                h_t = h_t * torch.exp(g_t).view(HV, 1, 1)

            v_t = v_t - torch.sum(h_t * k_hv.unsqueeze(-1), dim=1)

            if beta_t is not None:
                if beta_t.ndim == 1:
                    if beta_t.shape[0] != HV:
                        beta_t = _expand_to_hv(beta_t.unsqueeze(-1), HV).squeeze(-1)
                    v_t = v_t * beta_t.view(HV, 1)
                else:
                    if beta_t.shape[0] != HV:
                        beta_t = _expand_to_hv(beta_t, HV)
                    v_t = v_t * beta_t

            h_t = h_t + k_hv.unsqueeze(-1) * v_t.unsqueeze(-2)
            o_t = torch.sum(h_t * q_hv.unsqueeze(-1), dim=1)

            if cu_seqlens is None:
                out[seq_idx, tok] = o_t
            else:
                out[0, tok] = o_t

            if persist_state:
                states[state_idx] = h_t

    return out, states


def chunk_gated_delta_rule_pytorch(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    head_first=False,
    use_qk_l2norm_in_kernel=False,
):
    """PyTorch fallback for chunk_gated_delta_rule with fla-compatible shapes."""
    if head_first:
        raise DeprecationWarning("head_first=True is not supported in 310P fallback.")

    B, _, _, Kdim = k.shape
    HV = v.shape[2]
    Vdim = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    if initial_state is not None:
        states = initial_state.clone()
    else:
        states = torch.zeros(N, HV, Kdim, Vdim, dtype=q.dtype, device=q.device)

    scale = Kdim ** -0.5
    out, states = _run_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        states=states,
        scale=scale,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    if output_final_state:
        return out, states
    return out, None


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

    scale = Kdim ** -0.5
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
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    return out, states


def _as_bth(x: torch.Tensor, B: int, T: int, H: int) -> torch.Tensor:
    """Convert tensor to [B, T, H] for fallback math."""
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        if B == 1 and x.shape == (T, H):
            return x.unsqueeze(0)
        if x.shape == (B * T, H):
            return x.view(B, T, H)
    if x.ndim == 1 and x.shape[0] == H:
        return x.view(1, 1, H).expand(B, T, H)
    raise ValueError(f"Unsupported shape for BTH conversion: {tuple(x.shape)}")


def fused_sigmoid_gating_delta_rule_update_pytorch(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    initial_state_source,
    initial_state_indices,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
):
    """
    PyTorch fallback for fused_sigmoid_gating_delta_rule_update.

    Implemented by constructing g/beta then calling recurrent fallback.
    """
    B, T, _, Kdim = k.shape
    HV = v.shape[2]

    if scale is None:
        scale = Kdim ** -0.5

    a_bth = _as_bth(a, B, T, HV)
    b_bth = _as_bth(b, B, T, HV)

    x = a_bth + dt_bias.view(1, 1, HV)
    beta_x = softplus_beta * x
    softplus_x = torch.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * torch.log1p(torch.exp(beta_x)),
        x,
    )
    g = -torch.exp(A_log).view(1, 1, HV) * softplus_x
    beta_tensor = torch.sigmoid(b_bth)

    out, _ = fused_recurrent_gated_delta_rule_pytorch(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta_tensor,
        initial_state=initial_state_source,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=initial_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return out
