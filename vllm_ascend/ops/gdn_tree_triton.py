# SPDX-License-Identifier: Apache-2.0
"""Triton GDN recurrence with one state page per draft-tree node.

For each token, decay the parent's state, compute
``delta = beta * (value - state @ key)``, update the state by
``delta[:, None] * key[None, :]``, and project it onto the scaled query.
The computation uses fp32; state pages support fp32 or bf16 storage.

``init_state_indices`` selects the parent page and ``ssm_state_indices``
selects the output page. Nodes must follow their parents. A continuation can
keep the state on chip.

``actual_seq_lengths`` stores a base followed by per-row token counts, not
cumulative endpoints. Active rows precede zero-length graph padding. Each
row can verify the whole tree in one call; there is no fixed token limit.
Optional token/output index tables map a padded plan to packed operands
without materializing gathers and scatters. Padding entries are not read;
all entries inside active prefixes must contain valid token and page IDs.
"""

import torch
from vllm.triton_utils import tl, triton

# ---- Pre-bound fast dispatch cache ----
_CK_CACHE = {}


def _validate_inputs(
    query,
    key,
    value,
    state,
    beta,
    g,
    actual_seq_lengths,
    ssm_state_indices,
    init_state_indices,
    token_indices=None,
    out_indices=None,
):
    """Full input validation (fast path runs this once per specialization)."""
    assert (
        query.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
        and value.dtype == torch.bfloat16
        and beta.dtype == torch.bfloat16
    )
    assert actual_seq_lengths.dtype == torch.int32
    assert ssm_state_indices.dtype == torch.int32
    assert init_state_indices.dtype == torch.int32
    assert state.dtype in (torch.float32, torch.bfloat16)
    if g is not None:
        assert g.dtype == torch.float32
    # Require dense, non-overlapping state pages.
    assert state.dim() == 4
    assert (
        state.stride(3) == 1
        and state.stride(2) == state.size(3)
        and state.stride(1) == state.size(2) * state.size(3)
        and state.stride(0) >= state.size(1) * state.size(2) * state.size(3)
    ), "state must be dense within each non-overlapping page"
    T, NK, DK = query.shape
    NV, DV = value.shape[1], value.shape[2]
    assert state.shape[1] == NV and state.shape[2] == DV and state.shape[3] == DK
    assert beta.shape[0] >= T and (g is None or g.shape[0] >= T)
    assert NV % NK == 0, "GQA requires NV to be an integer multiple of NK"
    assert ssm_state_indices.numel() >= T and init_state_indices.numel() >= T
    if token_indices is not None:
        assert out_indices is not None, "token indirection needs the output table too"
        assert token_indices.dtype in (torch.int32, torch.int64), "the token table must be integer"
        assert out_indices.dtype in (torch.int32, torch.int64), "the output table must be integer"
        assert token_indices.numel() >= T and out_indices.numel() >= T, (
            "the indirection tables must cover every token position of the call"
        )


_raw_stream_fn = None


def _raw_stream(device):
    global _raw_stream_fn
    if _raw_stream_fn is None:
        try:
            from torch_npu._C import _npu_getCurrentRawStreamNoWait as fn
        except ImportError:
            from torch_npu._C import _npu_getCurrentRawStream as fn
        _raw_stream_fn = fn
    return _raw_stream_fn(device)


@triton.jit(do_not_specialize=["B"])
def recurrent_gated_delta_rule_tree_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    state_ptr,
    out_ptr,
    seq_len_metadata_ptr,
    ssm_ptr,
    init_ptr,
    tok_ptr,
    out_tok_ptr,
    scale,
    NK,
    DK,
    NV,
    DV,
    PAGE_STRIDE,
    GQA,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    HAS_G: tl.constexpr,
    SYNC_RELOAD: tl.constexpr,
    HAS_IDX: tl.constexpr,
    METADATA_ROW_STRIDE: tl.constexpr,
    B,
):
    pid_v = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    head64 = head.to(tl.int64)
    khead = head // GQA

    # cu = [base, len_0, len_1, ...] (fork convention, kernel accumulates row starts)
    if METADATA_ROW_STRIDE > 0:
        seq_start = tl.load(seq_len_metadata_ptr) + batch * METADATA_ROW_STRIDE
    else:
        lens = tl.load(
            seq_len_metadata_ptr + 1 + tl.arange(0, BLOCK_B),
            mask=tl.arange(0, BLOCK_B) < B,
            other=0,
        )
        seq_start = tl.load(seq_len_metadata_ptr) + tl.sum(tl.where(tl.arange(0, BLOCK_B) < batch, lens, 0))
    T = tl.load(seq_len_metadata_ptr + batch + 1)

    # ---- Resident address components (int64 base) ----
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_offs = tl.arange(0, BLOCK_K)
    m_v = v_offs < DV
    m_k = k_offs < DK
    v_offs64 = v_offs.to(tl.int64)
    k_offs64 = k_offs.to(tl.int64)
    page_in = head64 * (DV * DK) + v_offs64[:, None] * DK + k_offs64[None, :]

    # Root token's initial state page
    if T > 0:
        init0 = tl.load(init_ptr + seq_start)
        state = tl.load(
            state_ptr + init0.to(tl.int64) * PAGE_STRIDE + page_in,
            mask=(m_v[:, None] & m_k[None, :]),
            other=0.0,
        ).to(tl.float32)
        m_vk = m_v[:, None] & m_k[None, :]

        ssm_prev = tl.full((), -1, tl.int32)

        for t in range(0, T):
            # Direct scalar reads (independent of the recurrence chain, pipelinable)
            tok_t = seq_start + t
            ssm_t = tl.load(ssm_ptr + tok_t)
            init_t = tl.load(init_ptr + tok_t)
            if HAS_IDX:
                # ``tok_t`` is a cell of the caller's padded grid. The tables
                # carry the packed token this cell reads (q/k/v/g/beta stay
                # ungathered) and the output row it writes (no scatter). Cells
                # outside a row's prefix are never reached: the loop stops at
                # the row's ``cu_seqlens`` length.
                src_t = tl.load(tok_ptr + tok_t)
                dst_t = tl.load(out_tok_ptr + tok_t)
            else:
                src_t = tok_t
                dst_t = tok_t
            src_t64 = src_t.to(tl.int64)
            dst_t64 = dst_t.to(tl.int64)
            beta_t = tl.load(beta_ptr + src_t * NV + head).to(tl.float32)

            # Tree branch: parent page != previous token's write page -> reload from parent
            if (t > 0) & (init_t != ssm_prev):
                if SYNC_RELOAD:
                    tl.debug_barrier()
                state = tl.load(
                    state_ptr + init_t.to(tl.int64) * PAGE_STRIDE + page_in,
                    mask=m_vk,
                    other=0.0,
                ).to(tl.float32)
            ssm_prev = ssm_t

            # Current token's q/k/v vectors
            qk_base = src_t64 * (NK * DK) + khead * DK
            q_t = tl.load(q_ptr + qk_base + k_offs64, mask=m_k, other=0.0).to(tl.float32) * scale
            k_t = tl.load(k_ptr + qk_base + k_offs64, mask=m_k, other=0.0).to(tl.float32)
            v_t = tl.load(
                v_ptr + src_t64 * (NV * DV) + head64 * DV + v_offs64,
                mask=m_v,
                other=0.0,
            ).to(tl.float32)

            # ---- GDN recurrence (all fp32) ----
            if HAS_G:
                g_t = tl.exp(tl.load(g_ptr + src_t * NV + head))
                state = state * g_t
            kS = tl.sum(state * k_t[None, :], axis=1)
            delta = (v_t - kS) * beta_t
            state = state + delta[:, None] * k_t[None, :]
            o_t = tl.sum(state * q_t[None, :], axis=1)

            # Write out
            tl.store(
                out_ptr + dst_t64 * (NV * DV) + head64 * DV + v_offs64,
                o_t.to(out_ptr.dtype.element_ty),
                mask=m_v,
            )
            tl.store(
                state_ptr + ssm_t.to(tl.int64) * PAGE_STRIDE + page_in,
                state.to(state_ptr.dtype.element_ty),
                mask=m_vk,
            )


def npu_recurrent_gated_delta_rule_tree_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    *,
    beta: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    init_state_indices: torch.Tensor,
    g: torch.Tensor | None = None,
    token_indices: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    BLOCK_V=64,
    BLOCK_B=64,
    SYNC_RELOAD=True,
    metadata_row_stride=0,
) -> torch.Tensor:
    """Run per-node GDN recurrence using explicit parent and output state pages.

    ``actual_seq_lengths`` is [base, len_0, len_1, ...]. The kernel accumulates
    these counts for packed inputs. A fixed metadata stride gives it the row
    offsets directly for indexed inputs.
    Updates state in-place; returns [query.rows, NV, DV] bf16.

    ``token_indices``/``out_indices`` (optional, both or neither) let the kernel
    address q/k/v/g/beta and the output through integer tables instead of
    pre-gathered copies: position ``p`` of this call reads ``token_indices[p]``
    of the caller's packed q/k/v/g/beta and writes output row ``out_indices[p]``,
    while ``state``/``ssm_state_indices``/``init_state_indices`` stay in this
    call's own token order. They let a caller run the op directly on a padded
    token grid -- the GDN tree verify plan's ``token_indices``/``out_indices`` --
    with no gather, no scatter and no intermediate buffer. ``out`` must then
    span every row ``out_indices`` names.

    ``metadata_row_stride`` is the padded width of the metadata tables (and
    direct operands when no token table is supplied). Zero uses packed rows
    whose starts are the prefix sum of ``actual_seq_lengths``. The stride is
    a host-side shape, so variable live lengths need no host synchronization.
    """
    # Layout fix-up (every call; is_contiguous is cheap)
    if not query.is_contiguous():
        query = query.contiguous()
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    if not beta.is_contiguous():
        beta = beta.contiguous()
    if g is not None and not g.is_contiguous():
        g = g.contiguous()

    T, NK, DK = query.shape
    NV, DV = value.shape[1], value.shape[2]
    B = actual_seq_lengths.numel() - 1
    BLOCK_B = max(BLOCK_B, triton.next_power_of_2(B))
    if not isinstance(metadata_row_stride, int) or metadata_row_stride < 0:
        raise ValueError("metadata_row_stride must be a nonnegative integer")
    if metadata_row_stride:
        cells = B * metadata_row_stride
        if ssm_state_indices.numel() < cells or init_state_indices.numel() < cells:
            raise ValueError("state index tables must cover every fixed-width metadata row")

    has_idx = token_indices is not None or out_indices is not None
    if has_idx:
        if token_indices is None or out_indices is None:
            raise ValueError("token_indices and out_indices must be given together")
        if out is None:
            raise ValueError("token indirection writes caller-owned output rows, so out is required")
        if metadata_row_stride and (
            token_indices.numel() < B * metadata_row_stride or out_indices.numel() < B * metadata_row_stride
        ):
            raise ValueError("token index tables must cover every fixed-width metadata row")
        # The plan slices its tables off stable buffers; keep them dense for the
        # kernel's flat indexing without copying in the common case.
        if not token_indices.is_contiguous():
            token_indices = token_indices.contiguous()
        if not out_indices.is_contiguous():
            out_indices = out_indices.contiguous()
    elif out is None:
        out = torch.empty((T, NV, DV), dtype=torch.bfloat16, device=value.device)

    if B == 0:
        return out

    # Host shapes only: packed ragged rows are bounded by the whole table,
    # not by total/B. The scan itself masks every lane outside the live row.
    BLOCK_K = triton.next_power_of_2(DK)
    v_tiles = triton.cdiv(DV, BLOCK_V)
    grid = (v_tiles, NV, B)
    has_g = g is not None
    args = (
        query,
        key,
        value,
        g if has_g else beta,  # never dereferenced when HAS_G=0
        beta,
        state,
        out,
        actual_seq_lengths,
        ssm_state_indices,
        init_state_indices,
        token_indices if has_idx else beta,  # never dereferenced when HAS_IDX=0
        out_indices if has_idx else beta,  # ditto
        float(scale),
        NK,
        DK,
        NV,
        DV,
        state.stride(0),
        NV // NK,
    )
    spec_key = (
        BLOCK_V,
        BLOCK_K,
        BLOCK_B,
        has_g,
        SYNC_RELOAD,
        has_idx,
        metadata_row_stride,
        NK,
        DK,
        NV,
        DV,
        state.stride(0),
        NV // NK,
        query.dtype,
        state.dtype,
        beta.dtype,
        actual_seq_lengths.dtype,
        ssm_state_indices.dtype,
        init_state_indices.dtype,
        token_indices.dtype if has_idx else None,
        out_indices.dtype if has_idx else None,
        out.dtype,
    )

    ck = _CK_CACHE.get(spec_key)
    if ck is None:
        _validate_inputs(
            query,
            key,
            value,
            state,
            beta,
            g,
            actual_seq_lengths,
            ssm_state_indices,
            init_state_indices,
            token_indices,
            out_indices,
        )
        ck = recurrent_gated_delta_rule_tree_kernel.warmup(
            *args,
            grid=grid,
            BLOCK_V=BLOCK_V,
            BLOCK_K=BLOCK_K,
            BLOCK_B=BLOCK_B,
            HAS_G=has_g,
            SYNC_RELOAD=SYNC_RELOAD,
            HAS_IDX=has_idx,
            METADATA_ROW_STRIDE=metadata_row_stride,
            B=B,
        )
        _CK_CACHE[spec_key] = ck

    # triton-ascend's launcher stub uses the full signature (constexprs are placeholders)
    device = torch.npu.current_device()
    stream = _raw_stream(device)
    ck.run(
        grid[0],
        grid[1],
        grid[2],
        stream,
        ck.function,
        ck.packed_metadata,
        None,
        None,
        None,
        *args,
        BLOCK_V,
        BLOCK_K,
        BLOCK_B,
        has_g,
        SYNC_RELOAD,
        has_idx,
        metadata_row_stride,
        B,
    )
    return out
