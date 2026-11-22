from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tree_mask_reqs", "query_len", "kv_len", "max_nodes"])
def tree_attention_mask_kernel(
    mask_ptr,  # [R, 1, Q, Kv] int8
    visibility_ptr,  # [R, B, B] int8
    root_kv_positions_ptr,  # [R] int32
    num_tree_mask_reqs,
    query_len,
    kv_len,
    max_nodes,
    stride_mask_r,
    stride_mask_q,
    stride_mask_k,
    stride_vis_r,
    stride_vis_i,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D grid (q_block, k_block, req): each program fills one BLOCK_Q x BLOCK_K
    # tile of the per-request [Q, Kv] mask. This replaces the scalar Q*Kv double
    # loop (one int8 op per element, fully serial inside each program) with a
    # handful of vectorized 2D load/where/store block ops, and exposes
    # parallelism across the K-tile axis as well as the request axis instead of
    # only grid-striding over requests. The produced mask values are bit-ident
    # to the scalar reference (committed KV -> 0; draft k>prev visible iff
    # visibility[req, q-1, k-prev-1] != 0), so acceptance length is unchanged.
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    req = tl.program_id(2)
    if req >= num_tree_mask_reqs:
        return

    prev = tl.load(root_kv_positions_ptr + req)

    q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BQ]
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # [BK]
    q_valid = q_offs < query_len
    k_valid = k_offs < kv_len

    # Default: masked out (1). Committed KV (k <= prev) is always visible (0).
    result = tl.full([BLOCK_Q, BLOCK_K], 1, tl.int8)
    result = tl.where(k_offs[None, :] <= prev, 0, result)

    # Draft columns k > prev  ->  draft_col = k - prev - 1, valid in [0, max_nodes).
    draft_col = k_offs - prev - 1
    col_in_range = (draft_col >= 0) & (draft_col < max_nodes)
    draft_col_safe = tl.where(col_in_range, draft_col, 0)

    # visibility[req, q-1, draft_col] is read only for query rows q >= 1.
    vis_q = q_offs - 1
    q_can_draft = (q_offs >= 1) & q_valid
    vis_q_safe = tl.where(q_can_draft, vis_q, 0)

    in_draft = q_can_draft[:, None] & col_in_range[None, :] & k_valid[None, :]

    vis = tl.load(
        visibility_ptr + req * stride_vis_r + vis_q_safe[:, None] * stride_vis_i + draft_col_safe[None, :],
        mask=in_draft,
        other=0,
    )  # [BQ, BK] int8

    # vis != 0 -> can attend -> 0 ; vis == 0 -> masked -> 1
    result = tl.where(in_draft & (vis != 0), 0, result)
    result = tl.where(in_draft & (vis == 0), 1, result)

    out_offs = req * stride_mask_r + q_offs[:, None] * stride_mask_q + k_offs[None, :] * stride_mask_k
    tl.store(mask_ptr + out_offs, result, mask=q_valid[:, None] & k_valid[None, :])


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def fill_tree_attention_mask_triton(
    attn_mask,
    tree_visibility,
    root_kv_positions,
) -> None:
    import torch

    num_tree_mask_reqs = root_kv_positions.shape[0]
    max_nodes = tree_visibility.shape[-1]
    query_len = attn_mask.shape[2]
    kv_len = attn_mask.shape[3]
    # Bool view must share storage with attn_mask (in-place FIA mask).
    mask_i8 = attn_mask.view(torch.int8)
    vis_i8 = tree_visibility.to(torch.int8).contiguous()
    prev = root_kv_positions.to(torch.int32).contiguous()

    # Tile the (Q, Kv) plane. query_len is ~max_nodes+1 (typically 64-ish) so a
    # single 64-wide Q block covers it; kv_len is align_up(...,128) so 128-wide
    # K blocks divide evenly. Both are powers of two (tl.arange requirement).
    BLOCK_Q = 64
    BLOCK_K = 128
    grid = (
        max(_cdiv(query_len, BLOCK_Q), 1),
        max(_cdiv(kv_len, BLOCK_K), 1),
        num_tree_mask_reqs,
    )
    tree_attention_mask_kernel[grid](
        mask_i8,
        vis_i8,
        prev,
        num_tree_mask_reqs,
        query_len,
        kv_len,
        max_nodes,
        mask_i8.stride(0),
        mask_i8.stride(2),
        mask_i8.stride(3),
        vis_i8.stride(0),
        vis_i8.stride(1),
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
    )
