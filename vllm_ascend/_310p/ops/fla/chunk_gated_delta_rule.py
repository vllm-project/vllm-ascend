import torch

CHUNK_SIZE = 64


def _maybe_l2norm(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return x
    return x / (torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True)) + 1e-6)


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.where(x <= 0, x, torch.full_like(x, float("-inf"))))


def _iter_seq_ranges(B: int, T: int, cu_seqlens: torch.Tensor | None) -> list[tuple[int, int, int]]:
    if cu_seqlens is None:
        return [(i, 0, T) for i in range(B)]
    return [(i, int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())) for i in range(len(cu_seqlens) - 1)]


def _chunk_local_cumsum_torch(
    g: torch.Tensor,
    seq_ranges: list[tuple[int, int, int]],
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    g_cumsum = torch.empty_like(g, dtype=torch.float32)
    for seq_idx, start, end in seq_ranges:
        if end <= start:
            continue
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end)
            if g.shape[0] == 1 and len(seq_ranges) != 1:
                g_chunk = g[0, chunk_start:chunk_end].to(torch.float32)
                g_cumsum[0, chunk_start:chunk_end] = torch.cumsum(g_chunk, dim=0)
            else:
                g_chunk = g[seq_idx, chunk_start:chunk_end].to(torch.float32)
                g_cumsum[seq_idx, chunk_start:chunk_end] = torch.cumsum(g_chunk, dim=0)
    return g_cumsum


def _solve_unit_lower_inverse_quantized(
    A: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Approximate solve_tril behavior for unit-lower matrices with quantized stages.
    Returns inverse of (I + A), where A is strictly lower-triangular.
    """
    L = A.shape[0]
    Ai = torch.eye(L, dtype=out_dtype, device=A.device).to(torch.float32)
    A_fp32 = A.to(torch.float32)
    for i in range(1, L):
        for j in range(i):
            val = -(A_fp32[i, j:i] @ Ai[j:i, j].to(torch.float32))
            Ai[i, j] = val.to(out_dtype).to(torch.float32)
    return Ai.to(out_dtype).to(torch.float32)


def _solve_unit_lower_inverse_quantized_block64(
    A: torch.Tensor,
    valid_len: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Mimic solve_tril(BT=64) block merge path:
    1) inverse of four 16x16 diagonal blocks
    2) merge to two 32x32 lower blocks
    3) merge to one 64x64 lower block
    """
    BT = CHUNK_SIZE
    A_block = torch.zeros(BT, BT, dtype=torch.float32, device=A.device)
    A_block[:valid_len, :valid_len] = A[:valid_len, :valid_len]

    # Stage 1: diagonal 16x16 inverse blocks (matches solve_tril_16x16_kernel output buffer Ad).
    Ai11 = _solve_unit_lower_inverse_quantized(A_block[0:16, 0:16], out_dtype)
    Ai22 = _solve_unit_lower_inverse_quantized(A_block[16:32, 16:32], out_dtype)
    Ai33 = _solve_unit_lower_inverse_quantized(A_block[32:48, 32:48], out_dtype)
    Ai44 = _solve_unit_lower_inverse_quantized(A_block[48:64, 48:64], out_dtype)

    # Stage 2: merge 16x16 -> 32x32.
    A21 = A_block[16:32, 0:16].to(torch.float32)
    A43 = A_block[48:64, 32:48].to(torch.float32)
    Ai21 = -(Ai22.to(torch.float32) @ A21 @ Ai11.to(torch.float32))
    Ai43 = -(Ai44.to(torch.float32) @ A43 @ Ai33.to(torch.float32))

    Ai11_32 = torch.zeros(32, 32, dtype=torch.float32, device=A.device)
    Ai11_32[0:16, 0:16] = Ai11.to(torch.float32)
    Ai11_32[16:32, 16:32] = Ai22.to(torch.float32)
    Ai11_32[16:32, 0:16] = Ai21

    Ai22_32 = torch.zeros(32, 32, dtype=torch.float32, device=A.device)
    Ai22_32[0:16, 0:16] = Ai33.to(torch.float32)
    Ai22_32[16:32, 16:32] = Ai44.to(torch.float32)
    Ai22_32[16:32, 0:16] = Ai43

    # Stage 3: merge 32x32 -> 64x64.
    A21_32 = A_block[32:64, 0:32].to(torch.float32)
    Ai21_32 = -(Ai22_32 @ A21_32 @ Ai11_32)

    # Triton stores merged matrices to output_dtype, then later consumers load as fp32.
    Ai_block = torch.zeros(BT, BT, dtype=torch.float32, device=A.device)
    Ai_block[0:32, 0:32] = Ai11_32.to(out_dtype).to(torch.float32)
    Ai_block[32:64, 32:64] = Ai22_32.to(out_dtype).to(torch.float32)
    Ai_block[32:64, 0:32] = Ai21_32.to(out_dtype).to(torch.float32)
    return Ai_block[:valid_len, :valid_len]


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
    """PyTorch fallback for chunk_gated_delta_rule with chunk-aligned math path."""
    if head_first:
        raise DeprecationWarning("head_first=True is not supported in 310P fallback.")

    B, T, H_qk, Kdim = k.shape
    H_v = v.shape[2]
    Vdim = v.shape[-1]
    if H_v % H_qk != 0:
        raise ValueError(f"Invalid grouped heads: H_v={H_v}, H_qk={H_qk}.")
    if cu_seqlens is not None and B != 1:
        raise ValueError("Variable-length mode expects batch size B=1.")
    if beta.ndim != 3:
        raise ValueError("chunk_gated_delta_rule expects beta with shape [B, T, H].")

    seq_ranges = _iter_seq_ranges(B, T, cu_seqlens)
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if initial_state is not None:
        states = initial_state.to(torch.float32).clone()  # [N, H_v, K, V]
    else:
        states = torch.zeros(N, H_v, Kdim, Vdim, dtype=torch.float32, device=q.device)

    if use_qk_l2norm_in_kernel:
        q = _maybe_l2norm(q.to(torch.float32), True).to(q.dtype)
        k = _maybe_l2norm(k.to(torch.float32), True).to(k.dtype)

    out = torch.zeros_like(v)
    scale = Kdim**-0.5
    g_cumsum = _chunk_local_cumsum_torch(g, seq_ranges, chunk_size=CHUNK_SIZE) if g is not None else None

    group_size = H_v // H_qk
    for seq_idx, start, end in seq_ranges:
        if end <= start:
            continue
        b_idx = 0 if (cu_seqlens is not None and B == 1) else seq_idx
        for h in range(H_v):
            qk_h = h // group_size
            h_state = states[seq_idx, h].to(torch.float32)  # [K, V]
            local_chunk_idx = 0
            for chunk_start in range(start, end, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, end)
                L = chunk_end - chunk_start
                if L <= 0:
                    continue

                q_chunk = q[b_idx, chunk_start:chunk_end, qk_h].to(torch.float32)  # [L, K]
                k_chunk = k[b_idx, chunk_start:chunk_end, qk_h].to(torch.float32)  # [L, K]
                v_chunk = v[b_idx, chunk_start:chunk_end, h].to(torch.float32)  # [L, V]
                beta_chunk = beta[b_idx, chunk_start:chunk_end, h].to(torch.float32)  # [L]

                g_chunk = g_cumsum[b_idx, chunk_start:chunk_end, h] if g_cumsum is not None else None

                # A := lower(beta * K K^T * exp(g_i - g_j)), then Ai = (I + A)^-1.
                A = k_chunk @ k_chunk.transpose(0, 1)
                if g_chunk is not None:
                    A = A * _safe_exp(g_chunk[:, None] - g_chunk[None, :])
                A = A * beta_chunk[:, None]
                A = torch.tril(A, diagonal=-1)
                # Approximate solve_tril numeric path with quantized forward-substitution.
                Ai = _solve_unit_lower_inverse_quantized_block64(A, L, k.dtype)

                # Recompute W/U path from Triton chunk implementation.
                u_chunk = (Ai @ (v_chunk * beta_chunk[:, None])).to(v.dtype)
                if g_chunk is not None:
                    w_rhs = k_chunk * beta_chunk[:, None] * torch.exp(g_chunk)[:, None]
                else:
                    w_rhs = k_chunk * beta_chunk[:, None]
                w_chunk = (Ai @ w_rhs).to(k.dtype)

                h_chunk = h_state
                # chunk_delta_h kernel computes w @ h in low precision (k.dtype),
                # then accumulates to fp32.
                h_snapshot_low = h_chunk.to(k.dtype)
                w_low = w_chunk.to(k.dtype)
                v_new = u_chunk.to(torch.float32) - (w_low @ h_snapshot_low).to(torch.float32)
                v_new_saved = v_new.to(v.dtype)

                # Output path aligns with chunk_o kernel.
                q_eval_low = q_chunk.to(q.dtype)
                k_eval_low = k_chunk.to(k.dtype)
                state_term = (q_eval_low @ h_snapshot_low).to(torch.float32)
                A_qk = (q_eval_low @ k_eval_low.transpose(0, 1)).to(torch.float32)
                if g_chunk is not None:
                    state_term = state_term * torch.exp(g_chunk)[:, None]
                    A_qk = A_qk * _safe_exp(g_chunk[:, None] - g_chunk[None, :])
                A_qk = torch.tril(A_qk, diagonal=0)
                local_term = (A_qk.to(v.dtype) @ v_new_saved).to(torch.float32)
                out_chunk = (state_term + local_term) * scale

                if cu_seqlens is None:
                    out[seq_idx, chunk_start:chunk_end, h] = out_chunk.to(out.dtype)
                else:
                    out[0, chunk_start:chunk_end, h] = out_chunk.to(out.dtype)

                # Final state update path aligns with chunk_delta_h kernel.
                if g_chunk is not None:
                    g_last = g_chunk[-1]
                    v_new_scaled = v_new * _safe_exp(g_last - g_chunk)[:, None]
                    h_state = h_chunk * torch.exp(g_last) + (k_eval_low.transpose(0, 1) @ v_new_scaled.to(k.dtype)).to(
                        torch.float32
                    )
                else:
                    h_state = h_chunk + (k_eval_low.transpose(0, 1) @ v_new.to(k.dtype)).to(torch.float32)
                h_state = h_state.to(torch.float32)
                local_chunk_idx += 1

            states[seq_idx, h] = h_state

    if output_final_state:
        return out, states
    return out, None
