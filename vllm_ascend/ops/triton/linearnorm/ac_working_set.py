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
    # Budget geometry only: q is the bulk Q tile; D/h pad head-size/half-RoPE width.
    # f/w are FP32/two-byte carriers; nq/nk are M head-rows, qt/kt all tail heads.
    # The public wrapper passes the actual D/R to the kernel.
    D = max(128, 1 << (head_size - 1).bit_length())
    h = max(32, 1 << (rope_dim // 2 - 1).bit_length())
    f, w, g, b = 4, 2, int(has_gate), int(has_bias)
    k, nq, nk = kv_heads, M * q, M * kv_heads
    qt, kt = q_heads, kv_heads
    tail_control_bytes = 8 * h + (h + 7) // 8
    bulk_control_bytes, Sinit, Ssel, Sdiv = tail_control_bytes + 3 * M * h * w + 4 * M, max(128 * M, h * w), 64, 32
    bulk_red_q, bulk_mul_q = 2 * nq * D, max(256, 32 * nq)
    bulk_red_k, bulk_mul_k = 2 * nk * D, max(256, 32 * nk)
    tail_red_q, tail_mul_q = 2 * qt * D, max(256, 32 * qt)
    tail_red_k, tail_mul_k = 2 * kt * D, max(256, 32 * kt)
    # Raw, compact, full and two-half weight carriers; bias mirrors them, not the shared ones.
    weight_prep = 2 * D * w + 2 * D * f + (nq + nk) * D * f + 2 * (nq + nk) * h * f
    prepared_parameter_bytes = (1 + b) * weight_prep + (nq + nk) * f
    mask_extra = int(not is_interleaved) * (6 * h * f + 3 * M * h * w + 6 * ((M * h + 7) // 8))
    tail_mask_extra = int(not is_interleaved) * (6 * h * f + 3 * h * w + 6 * ((h + 7) // 8))
    # Compact cos/sin and reserved Q/K coefficient expansion.
    coefficient_bytes = 2 * M * h * f + 2 * (nq + nk) * h * f
    # Q input remains in q_gate_bytes even when Gate is disabled.
    q_gate_bytes, q_compute_bytes, k_compute_bytes, q_statistics_bytes, k_statistics_bytes = (
        (1 + g) * nq * D * f,
        nq * D * f,
        nk * D * f,
        nq * f,
        nk * f,
    )
    q_aux_bytes = max(bulk_red_q, 2 * nq * h * f + bulk_mul_q, 4 * nq * h * f - q_statistics_bytes)
    k_aux_bytes = max(bulk_red_k, 2 * nk * h * f + bulk_mul_k, 4 * nk * h * f - k_statistics_bytes)
    k_raw_bytes = nk * D * w
    k_bulk_bytes = k_raw_bytes + 2 * k_compute_bytes + k_statistics_bytes + k_aux_bytes
    # Bulk Q and whole-K producers execute sequentially; retained K raw is counted.
    bulk = (
        prepared_parameter_bytes
        + coefficient_bytes
        + bulk_control_bytes
        + max(q_gate_bytes + q_compute_bytes + q_statistics_bytes + q_aux_bytes, k_bulk_bytes)
    )
    init_parameter_aux_bytes = (1 + b) * ((q + k) * D * f + 2 * (q + k) * h * f) + 4 * h * f + 6 * M * h * w
    init_control_peak_bytes = max(24 * h + 4 * M, bulk_control_bytes + M * h * w + 2 * ((M * h + 7) // 8))
    init_aux_bytes = init_control_peak_bytes + 2 * M * h * f + M * h * w + Sinit
    # Init includes weight/bias preparation, selected coefficients and packed predicates.
    init = prepared_parameter_bytes + coefficient_bytes + init_parameter_aux_bytes + init_aux_bytes + mask_extra
    k_output_live_bytes = k_raw_bytes + 2 * k_compute_bytes
    output_live_bytes = max(q_gate_bytes + q_compute_bytes, k_output_live_bytes, M * k * D * w)
    k_output_aux_bytes = k_statistics_bytes + max(k_aux_bytes, 4 * nk * h * f + bulk_mul_k)
    output_aux_bytes = max(M * q * D * w, nq * f + 2 * nq * h * f + bulk_mul_q, 4 * nq * h * f, k_output_aux_bytes)
    # Output/V keeps full owners through cast/store, with separate landing and auxiliary peaks.
    output = prepared_parameter_bytes + coefficient_bytes + bulk_control_bytes + output_live_bytes + output_aux_bytes
    # The M1 remainder uses all heads, including Hq beyond the bulk Q-head tile.
    tail_base_bytes = (
        (1 + b) * (2 * D * f + 4 * h * f)
        + (qt + kt) * f
        + 2 * h * f
        + (1 + g) * qt * D * f
        + kt * D * f
        + kt * D * w
        + tail_control_bytes
    )
    tail_live_bytes = (
        qt * D * f
        + kt * D * f
        + (qt + kt) * f
        + max(tail_red_q, tail_red_k, 2 * qt * h * f + tail_mul_q, 2 * kt * h * f + tail_mul_k)
    )
    tail_aux_bytes = (
        ((1 + g) * qt + kt) * D * w
        + (1 + b) * 2 * D * w
        + 7 * h * w
        + 2 * h * f
        + 2 * (qt + kt) * h * f
        + g * qt * D * w
        + Ssel
        + Sdiv
    )
    tail = tail_base_bytes + tail_live_bytes + tail_aux_bytes + tail_mask_extra
    return max(init, bulk, output, tail)
