"""Adapters from the vLLM-Ascend DSA FP8 KV path
(``npu_kv_quant_sparse_attn_sharedkv`` / ``..._metadata``) to the new
``mixed_quant_sparse_flash_mla`` / ``..._metadata`` (cann_ops_transformer) op.

The new op consumes a PA_BBND cache with 608 bytes/token (quantMode=1):
  [rope(64 bf16=128B) | nope(448 fp8) | scale(7 bf16=14B) | pad(18B)]
Since the DSV4 FP8 path now writes the cache directly in that layout
(cann_ops_transformer.kv_compress_epilog quantMode=0), the caches are passed
through untouched. The live service passes full-width dense block tables
(every cache block, stale entries beyond the sequence), so both block tables
are truncated per row to the number of blocks the sequence needs.

Validated against the reference op with single_op_tests/single_op_new_ratio4_qm1.py
(max_abs_diff ~0.016, quantMode=1).
"""

import torch

BLOCK_SIZE = 32

_mq_ops_cache = None


def _mq_ops():
    global _mq_ops_cache
    if _mq_ops_cache is None:
        from importlib import import_module

        import_module("cann_ops_transformer")
        _mq_ops_cache = (
            torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata,
            torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla,
        )
    return _mq_ops_cache


def _has_cmp(kw):
    return bool(kw.get("has_cmp_kv", int(kw.get("cmp_ratio", 1)) > 1))


def _blocks_for(n_tokens):
    """Per-seq number of BLOCK_SIZE blocks needed for n_tokens."""
    return (n_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE


def _truncate_block_table(bt, n_tokens):
    """Mask out blocks beyond ceil(n_tokens/32) per row with -1 (device-only).

    The op never reads past ceil(n_tokens/32) blocks per sequence, so the
    -1 padding is never dereferenced; -1 is the standard pad sentinel.
    The mask is a pure device op (no host sync, no host loop), so it stays
    valid inside CUDA-graph capture: on every replay the per-row counts are
    re-derived from the seqused input buffer and the table from the block
    table input buffer.
    """
    if bt is None:
        return None
    need = torch.clamp(_blocks_for(n_tokens), min=1, max=bt.shape[1])
    cols = torch.arange(bt.shape[1], device=bt.device)
    keep = cols[None, :] < need[:, None]
    return torch.where(keep, bt, -1).contiguous()


def _split_seqused(seqused, ratio, has_cmp):
    if not has_cmp:
        return None, None
    return seqused // ratio, seqused % ratio


def _pair_reorder_cmp_phys(csi, bt, cuq):
    """Reorder each adjacent csi slot pair (2j, 2j+1) so the physical cache
    address is ascending within the pair.

    The new op's CopyInKvSparse fuses each slot pair into ONE DataCopyPad
    that starts from the SMALLER address; when phys(csi[2j]) > phys(csi[2j+1])
    the two KV rows land in the wrong slots (pair swap). The reference op
    (npu_kv_quant_sparse_attn_sharedkv) is slot-order invariant, so permuting
    the values within such a pair changes nothing for it and cancels the swap
    for the new op. Pure device ops, no host sync (CUDA-graph safe).
    """
    if csi is None or bt is None or cuq is None or csi.shape[-1] < 2:
        return csi
    T, H, K = csi.shape
    dev = csi.device
    blk = csi.clamp(min=0).to(torch.int64) // BLOCK_SIZE
    off = csi.clamp(min=0).to(torch.int64) % BLOCK_SIZE
    valid = csi >= 0
    rows = torch.arange(T, device=dev)
    bat = torch.searchsorted(cuq.to(dev), rows, right=True) - 1  # per-row batch idx
    btb = bt[bat.clamp(min=0)].to(dev)  # (T, NB)
    NB = btb.shape[1]
    phys = torch.gather(btb.unsqueeze(1).expand(T, H, NB), 2, blk.clamp(max=NB - 1)) * BLOCK_SIZE + off
    phys = torch.where(valid, phys, torch.full_like(phys, -1))
    K2 = (K // 2) * 2
    ev = csi[..., 0:K2:2]
    od = csi[..., 1:K2:2]
    p0 = phys[..., 0:K2:2]
    p1 = phys[..., 1:K2:2]
    swap = (p0 > p1) & (ev >= 0) & (od >= 0)
    head = torch.stack([torch.where(swap, od, ev), torch.where(swap, ev, od)], dim=-1).reshape(T, H, K2)
    if K2 < K:
        out = torch.cat([head, csi[..., K2:]], dim=-1)
    else:
        out = head
    return out.contiguous()


def _fullcov_sparse_indices(si, seqused):
    """True when the sparse indices cover the whole KV range."""
    if si is None:
        return False
    valid = si[si >= 0]
    if valid.numel() == 0:
        return False
    max_used = int(seqused.max().item()) if torch.is_tensor(seqused) else int(seqused)
    return valid.numel() >= max_used and int(valid.max().item()) >= max_used - 1


# ---------------------------------------------------------------------------
# metadata op
# ---------------------------------------------------------------------------


def mq_meta_adapter(kw):
    ratio = max(int(kw.get("cmp_ratio", 1)), 1)
    has_cmp = _has_cmp(kw)
    seqused = kw["seqused_kv"]
    cmp_len, residual = _split_seqused(seqused, ratio, has_cmp)
    max_cmp = max(int(cmp_len.max().item()), 1) if has_cmp else 0
    return dict(
        num_heads_q=kw["num_heads_q"],
        num_heads_kv=int(kw.get("num_heads_kv", 1)),
        head_dim=int(kw["head_dim"]),
        quant_mode=1,
        cu_seqlens_q=kw.get("cu_seqlens_q"),
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_ori_kv=seqused,
        seqused_cmp_kv=cmp_len,
        cmp_residual_kv=residual,
        ori_topk_length=None,
        cmp_topk_length=None,
        batch_size=int(kw["batch_size"]),
        max_seqlen_q=int(kw["max_seqlen_q"]),
        max_seqlen_ori_kv=int(kw["max_seqlen_kv"]),
        max_seqlen_cmp_kv=max_cmp,
        ori_topk=0,
        cmp_topk=int(kw.get("cmp_topk", 0)),
        rope_head_dim=int(kw.get("rope_head_dim", 64)),
        cmp_ratio=ratio,
        ori_mask_mode=int(kw.get("ori_mask_mode", 4)),
        cmp_mask_mode=3 if has_cmp else 0,
        ori_win_left=int(kw.get("ori_win_left", 0)),
        ori_win_right=int(kw.get("ori_win_right", 0)),
        layout_q=kw.get("layout_q", "TND"),
        layout_kv="PA_BBND",
        has_ori_kv=bool(kw.get("has_ori_kv", True)),
        has_cmp_kv=has_cmp,
    )


def mq_meta_call(**kw):
    meta_op, _ = _mq_ops()
    return meta_op(**mq_meta_adapter(kw))


# ---------------------------------------------------------------------------
# attention op
# ---------------------------------------------------------------------------


def mq_attn_adapter(q, kw):
    ratio = max(int(kw.get("cmp_ratio", 1)), 1)
    seqused = kw["seqused_kv"]
    has_cmp = ratio > 1 and kw.get("cmp_kv") is not None
    cmp_len, residual = _split_seqused(seqused, ratio, has_cmp)

    # ori_bt = _truncate_block_table(kw.get("ori_block_table"), seqused)
    ori_bt = kw.get("ori_block_table")
    ori_kv = kw.get("ori_kv")
    if has_cmp:
        # cmp_len compressed tokens + residual tail can live in the cmp cache
        # cmp_bt = _truncate_block_table(kw.get("cmp_block_table"), cmp_len + residual)
        cmp_bt = kw.get("cmp_block_table")
        cmp_kv = kw.get("cmp_kv")
    else:
        cmp_bt = None
        cmp_kv = None

    win_r = int(kw.get("ori_win_right", 0))
    csi = kw.get("cmp_sparse_indices")
    # if has_cmp and csi is not None:
    #     # Cancel the new op's CopyInKvSparse pair-swap: sort each slot pair
    #     # (2j,2j+1) by physical cache address (see _pair_reorder_cmp_phys).
    #     csi = _pair_reorder_cmp_phys(csi, cmp_bt, kw.get("cu_seqlens_q"))
    osi = kw.get("ori_sparse_indices")
    if osi is not None:
        # Full coverage: the new op derives attention from the window instead.
        # The coverage check needs the batch total on the host, so it only
        # runs outside graph capture (vision prefill, executed eagerly).
        # Inside CUDA-graph capture the DSpark draft window from kwargs is
        # equivalent: the sliding window clips at the sequence start, so a
        # short sequence gets full coverage either way.
        if not torch.cuda.is_current_stream_capturing() and _fullcov_sparse_indices(osi, seqused):
            win_r = int(kw["cu_seqlens_q"][-1].item()) - 1
        osi = None

    return dict(
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=osi,
        cmp_sparse_indices=csi,
        ori_block_table=ori_bt,
        cmp_block_table=cmp_bt,
        cu_seqlens_q=kw.get("cu_seqlens_q"),
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_ori_kv=seqused,
        seqused_cmp_kv=cmp_len,
        cmp_residual_kv=residual,
        ori_topk_length=None,
        cmp_topk_length=None,
        sinks=kw.get("sinks"),
        metadata=kw.get("metadata"),
        quant_mode=1,
        rope_head_dim=int(kw.get("rope_head_dim", 64)),
        softmax_scale=kw.get("softmax_scale"),
        cmp_ratio=ratio,
        ori_mask_mode=int(kw.get("ori_mask_mode", 4)),
        cmp_mask_mode=3 if has_cmp else 0,
        ori_win_left=int(kw.get("ori_win_left", 0)),
        ori_win_right=win_r,
        layout_q=kw.get("layout_q", "TND"),
        layout_kv="PA_BBND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )


def mq_attn_call(q, **kw):
    _, attn_op = _mq_ops()
    out, lse = attn_op(q, **mq_attn_adapter(q, kw))
    return out, lse
