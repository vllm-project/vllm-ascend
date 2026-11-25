"""Role-based two-byte empirical screen; not an allocator upper bound."""


def working_set(M, q_heads, kv_heads, head_size, rope_dim, has_gate, has_bias=False, is_interleaved=True):
    """Integer bytes using padded budget geometry; actual kernel geometry stays unchanged."""
    for value in (M, q_heads, kv_heads, head_size, rope_dim):
        if type(value) is not int:
            raise TypeError("working-set geometry must use builtin integers")
        if value <= 0:
            raise ValueError("working-set geometry must be positive")
    if any(type(flag) is not bool for flag in (has_gate, has_bias, is_interleaved)):
        raise TypeError("working-set flags must be bool")
    q = min(q_heads, 12)
    if M not in (2, 4) or q_heads % q or rope_dim % 2 or rope_dim > head_size:
        raise ValueError("invalid complete-head working-set geometry")
    # Pad budget carriers only; the public wrapper passes the actual D/R to the kernel.
    D = max(128, 1 << (head_size - 1).bit_length())
    h = max(32, 1 << (rope_dim // 2 - 1).bit_length())
    f, w, g, b = 4, 2, int(has_gate), int(has_bias)
    k, nq, nk = kv_heads, M * q, M * kv_heads
    qt, kt = q_heads, kv_heads
    It = 8 * h + (h + 7) // 8
    Ib, Sinit, Ssel, Sdiv = It + 3 * M * h * w + 4 * M, max(128 * M, h * w), 64, 32
    bulk_red_q, bulk_mul_q = 2 * nq * D, max(256, 32 * nq)
    bulk_red_k, bulk_mul_k = 2 * nk * D, max(256, 32 * nk)
    tail_red_q, tail_mul_q = 2 * qt * D, max(256, 32 * qt)
    tail_red_k, tail_mul_k = 2 * kt * D, max(256, 32 * kt)
    # Raw, compact, full and two-half weight carriers; bias mirrors them, not the shared ones.
    weight_prep = 2 * D * w + 2 * D * f + (nq + nk) * D * f + 2 * (nq + nk) * h * f
    P = (1 + b) * weight_prep + (nq + nk) * f
    mask_extra = int(not is_interleaved) * (6 * h * f + 3 * M * h * w + 6 * ((M * h + 7) // 8))
    tail_mask_extra = int(not is_interleaved) * (6 * h * f + 3 * h * w + 6 * ((h + 7) // 8))
    # Compact cos/sin and reserved Q/K coefficient expansion.
    C = 2 * M * h * f + 2 * (nq + nk) * h * f
    Gq, Xq, Xk, Vq, Vk = (1 + g) * nq * D * f, nq * D * f, nk * D * f, nq * f, nk * f
    Eq = max(bulk_red_q, 2 * nq * h * f + bulk_mul_q, 4 * nq * h * f - Vq)
    Ek = max(bulk_red_k, 2 * nk * h * f + bulk_mul_k, 4 * nk * h * f - Vk)
    Kraw = nk * D * w
    Kbulk = Kraw + 2 * Xk + Vk + Ek
    # Bulk Q and whole-K producers execute sequentially; retained K raw is counted.
    bulk = P + C + Ib + max(Gq + Xq + Vq + Eq, Kbulk)
    Ji = (1 + b) * ((q + k) * D * f + 2 * (q + k) * h * f) + 4 * h * f + 6 * M * h * w
    Iinit = max(24 * h + 4 * M, Ib + M * h * w + 2 * ((M * h + 7) // 8))
    Ri = Iinit + 2 * M * h * f + M * h * w + Sinit
    # Init includes weight/bias preparation, selected coefficients and packed predicates.
    init = P + C + Ji + Ri + mask_extra
    LoK = Kraw + 2 * Xk
    Lo = max(Gq + Xq, LoK, M * k * D * w)
    RoK = Vk + max(Ek, 4 * nk * h * f + bulk_mul_k)
    Ro = max(M * q * D * w, nq * f + 2 * nq * h * f + bulk_mul_q, 4 * nq * h * f, RoK)
    # Output/V keeps full owners through cast/store, with separate landing and auxiliary peaks.
    output = P + C + Ib + Lo + Ro
    # The M1 remainder uses all heads, including Hq beyond the bulk Q-head tile.
    T0 = (
        (1 + b) * (2 * D * f + 4 * h * f)
        + (qt + kt) * f
        + 2 * h * f
        + (1 + g) * qt * D * f
        + kt * D * f
        + kt * D * w
        + It
    )
    Lt = (
        qt * D * f
        + kt * D * f
        + (qt + kt) * f
        + max(tail_red_q, tail_red_k, 2 * qt * h * f + tail_mul_q, 2 * kt * h * f + tail_mul_k)
    )
    Rt = (
        ((1 + g) * qt + kt) * D * w
        + (1 + b) * 2 * D * w
        + 7 * h * w
        + 2 * h * f
        + 2 * (qt + kt) * h * f
        + g * qt * D * w
        + Ssel
        + Sdiv
    )
    tail = T0 + Lt + Rt + tail_mask_extra
    return max(init, bulk, output, tail)
