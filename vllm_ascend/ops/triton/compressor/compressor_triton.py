"""Triton implementation of the compressor.

Kernel stages:
  K1: Two projection GEMMs (BF16 weights, swizzled launch order, blocks autotuned by M).
  K2: Group pooling (variable lengths, residual concatenation and two-pass softmax).
      Leading residuals are read from the cache.
  K3: Cache update (only the last cache_size tokens per segment, one writer per slot).
K2 must run before K3: tail writes and leading residual reads may access the same
slot. Concurrent execution could overwrite the residual before it is read.
compressor_ref is the pure PyTorch reference implementation.
"""

import numpy as np
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.models.deepseek_v41.cache_config import STATE_RING_ROWS

DEV = "npu"
MAX_CHUNK_ROWS = 8  # Max rows per pooling chunk; intermediates across three passes use <192 KB of UB.


# ============================================================================
# K1: Projection GEMM (BF16 weights, swizzled launch order, FP32 output)
# ============================================================================

# Autotune candidates keyed by M. BK changes FP32 accumulation grouping,
# introducing ULP-level differences within the numerical tolerance.
_proj_tune_configs = [
    triton.Config({"BM": 128, "BN": 256, "BK": 256, "GROUP": 8}),
    triton.Config({"BM": 128, "BN": 256, "BK": 128, "GROUP": 8}),
    triton.Config({"BM": 64, "BN": 256, "BK": 256, "GROUP": 8}),
    triton.Config({"BM": 64, "BN": 256, "BK": 128, "GROUP": 8}),
    triton.Config({"BM": 128, "BN": 128, "BK": 256, "GROUP": 8}),
    triton.Config({"BM": 64, "BN": 128, "BK": 256, "GROUP": 8}),
    triton.Config({"BM": 256, "BN": 128, "BK": 256, "GROUP": 8}),
    triton.Config({"BM": 64, "BN": 256, "BK": 256, "GROUP": 4}),
]


@triton.autotune(configs=_proj_tune_configs, key=["M"])
@triton.jit
def _proj_kernel(
    out_ptr,
    x_ptr,
    w_ptr,
    M,
    IN_DIM: tl.constexpr,
    OUT_DIM: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Compute x (M, IN_DIM) @ w.T with row-major w (OUT_DIM, IN_DIM) and FP32 output."""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    n_tile_count = OUT_DIM // BN
    m_tile_count = tl.cdiv(M, BM)
    total_tiles = m_tile_count * n_tile_count
    tiles_per_group = GROUP * n_tile_count
    for tile_id in range(pid, total_tiles, num_progs):
        swizzle_group = tile_id // tiles_per_group
        group_m_start = swizzle_group * GROUP
        group_m_tiles = m_tile_count - group_m_start if (m_tile_count - group_m_start) < GROUP else GROUP
        rank_in_group = tile_id - swizzle_group * tiles_per_group
        m_tile = group_m_start + rank_in_group % group_m_tiles
        n_tile = rank_in_group // group_m_tiles
        rows = m_tile * BM + offs_m
        cols = n_tile * BN + offs_n
        row_valid = rows < M
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, IN_DIM, BK):
            offs_k = k0 + tl.arange(0, BK)
            x = tl.load(x_ptr + rows[:, None] * IN_DIM + offs_k[None, :], mask=row_valid[:, None], other=0.0)
            w = tl.load(w_ptr + offs_k[:, None] + cols[None, :] * IN_DIM)
            acc = tl.dot(x, w, acc)
        tl.store(out_ptr + rows[:, None] * OUT_DIM + cols[None, :], acc, mask=row_valid[:, None])


# ============================================================================
# K2: Group pooling (O(1) task_id -> (batch, local_group) decoding; controls loaded in the kernel)
# ============================================================================


@triton.jit
def _pooled_blocked(
    group_idx,
    start_pos,
    seg_row_base,
    cache_row,
    kv_ptr,
    score_ptr,
    cache_ptr,
    prev_rows_ptr,
    prev_row_idx,
    offs_h,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    RATIO_PAD: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    HAS_RES,
    HAS_PREV_ROWS: tl.constexpr,
):
    """Pool a group in chunks using three passes: max, denominator and weighted sum.

    HAS_RES=0 skips cache reads. The group covers global tokens
    [group_idx*RATIO, (group_idx+1)*RATIO). For seg_off >= 0, read projection
    row seg_row_base+seg_off; negative offsets read leading residuals from
    cache[cache_row, token_pos % CACHE_SIZE]. PCP prefers the gathered
    predecessor row, falling back to the cache when its index is -1.
    """
    # Keep masked history lanes inside the projection allocation. Ascend may
    # form a DMA address for a masked lane before applying the load mask.
    # The original masks and ring selection still supply all history values.
    history_ptr = cache_ptr + cache_row * CACHE_SIZE * 2 * HEAD_DIM
    history_slot_stride = 2 * HEAD_DIM
    if HAS_PREV_ROWS:
        if prev_row_idx >= 0:
            # Only residual lanes read this row; broadcast the gathered predecessor.
            history_ptr = prev_rows_ptr + prev_row_idx * 2 * HEAD_DIM
            history_slot_stride = 0
    score_max = tl.full((HEAD_DIM,), -float("inf"), dtype=tl.float32)
    for chunk0 in range(0, RATIO_PAD, CHUNK_ROWS):
        rows = chunk0 + tl.arange(0, CHUNK_ROWS)
        row_valid = rows < RATIO
        token_pos = group_idx * RATIO + rows
        seg_off = token_pos - start_pos
        in_seg = seg_off >= 0
        score_seg = tl.load(
            score_ptr + (seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        if HAS_RES:
            slot = token_pos % CACHE_SIZE
            score_cache = tl.load(
                history_ptr + slot[:, None] * history_slot_stride + HEAD_DIM + offs_h[None, :],
                mask=(row_valid & (seg_off < 0))[:, None],
                other=0.0,
            )
            score = tl.where(in_seg[:, None], score_seg, score_cache)
        else:
            score = score_seg
        score = tl.where(row_valid[:, None], score, -float("inf"))
        score_max = tl.maximum(score_max, tl.max(score, axis=0))
    exp_sum = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for chunk0 in range(0, RATIO_PAD, CHUNK_ROWS):
        rows = chunk0 + tl.arange(0, CHUNK_ROWS)
        row_valid = rows < RATIO
        token_pos = group_idx * RATIO + rows
        seg_off = token_pos - start_pos
        in_seg = seg_off >= 0
        score_seg = tl.load(
            score_ptr + (seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        if HAS_RES:
            slot = token_pos % CACHE_SIZE
            score_cache = tl.load(
                history_ptr + slot[:, None] * history_slot_stride + HEAD_DIM + offs_h[None, :],
                mask=(row_valid & (seg_off < 0))[:, None],
                other=0.0,
            )
            score = tl.where(in_seg[:, None], score_seg, score_cache)
        else:
            score = score_seg
        score = tl.where(row_valid[:, None], score, -float("inf"))
        exp_sum += tl.sum(tl.exp(score - score_max[None, :]), axis=0)
    pooled = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for chunk0 in range(0, RATIO_PAD, CHUNK_ROWS):
        rows = chunk0 + tl.arange(0, CHUNK_ROWS)
        row_valid = rows < RATIO
        token_pos = group_idx * RATIO + rows
        seg_off = token_pos - start_pos
        in_seg = seg_off >= 0
        score_seg = tl.load(
            score_ptr + (seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        if HAS_RES:
            slot = token_pos % CACHE_SIZE
            score_cache = tl.load(
                history_ptr + slot[:, None] * history_slot_stride + HEAD_DIM + offs_h[None, :],
                mask=(row_valid & (seg_off < 0))[:, None],
                other=0.0,
            )
            score = tl.where(in_seg[:, None], score_seg, score_cache)
        else:
            score = score_seg
        score = tl.where(row_valid[:, None], score, -float("inf"))
        prob = tl.exp(score - score_max[None, :]) / exp_sum[None, :]
        kv_seg = tl.load(
            kv_ptr + (seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        if HAS_RES:
            slot = token_pos % CACHE_SIZE
            kv_cache = tl.load(
                history_ptr + slot[:, None] * history_slot_stride + offs_h[None, :],
                mask=(row_valid & (seg_off < 0))[:, None],
                other=0.0,
            )
            kv_vals = tl.where(in_seg[:, None], kv_seg, kv_cache)
        else:
            kv_vals = kv_seg
        pooled += tl.sum(prob * kv_vals, axis=0)
    return pooled


@triton.jit
def _pool_kernel(
    out_ptr,
    kv_ptr,
    score_ptr,
    cache_ptr,
    prev_rows_ptr,
    prev_row_indices_ptr,
    norm_w_ptr,
    meta_ptr,  # Packed [start_pos | used_len | out_row_offset | seg_row_base]
    block_table_ptr,
    total_group_slots,
    GROUP_SLOTS_PER_BATCH,  # Group slots per batch: ceil(max_used_len / RATIO)
    NUM_BATCH,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    RATIO_PAD: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    eps: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    NO_PAD: tl.constexpr,
    HAS_PREV_ROWS: tl.constexpr,
    TOKEN_ALIGNED: tl.constexpr = False,
    NUM_CACHE_BLOCKS: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    offs_h = tl.arange(0, HEAD_DIM)
    if not TOKEN_ALIGNED:
        norm_w = tl.load(norm_w_ptr + offs_h).to(tl.float32)
    for task_id in range(pid, total_group_slots, tl.num_programs(0)):
        batch = task_id // GROUP_SLOTS_PER_BATCH
        local_group = task_id - batch * GROUP_SLOTS_PER_BATCH
        start_pos = tl.load(meta_ptr + batch)  # [0, NUM_BATCH)
        used_len = tl.load(meta_ptr + NUM_BATCH + batch)  # [NUM_BATCH, 2*NUM_BATCH)
        groups_in_batch = (start_pos + used_len) // RATIO - start_pos // RATIO
        # A ring page is 32 * 2 * HEAD_DIM FP32 elements. At page 65536 the
        # element offset exceeds INT32_MAX; widen before pointer arithmetic.
        cache_row = tl.load(block_table_ptr + batch).to(tl.int64)
        valid = local_group < groups_in_batch
        if TOKEN_ALIGNED:
            valid = valid & (cache_row > 0) & (cache_row < NUM_CACHE_BLOCKS)
        if valid:
            group_idx = start_pos // RATIO + local_group
            seg_row_base = tl.load(meta_ptr + 3 * NUM_BATCH + batch)  # [3*NUM_BATCH, 4*NUM_BATCH)
            rows = tl.arange(0, CHUNK_ROWS)
            token_pos = group_idx * RATIO + rows
            seg_off = token_pos - start_pos
            # Leading residual count; nonzero only when local_group == 0.
            residual = start_pos - group_idx * RATIO
            cache_row = tl.load(block_table_ptr + batch).to(tl.int64)
            if residual > 0:
                prev_row_idx = -1
                if HAS_PREV_ROWS:
                    prev_row_idx = tl.load(prev_row_indices_ptr + batch).to(tl.int64)
                pooled = _pooled_blocked(
                    group_idx,
                    start_pos,
                    seg_row_base,
                    cache_row,
                    kv_ptr,
                    score_ptr,
                    cache_ptr,
                    prev_rows_ptr,
                    prev_row_idx,
                    offs_h,
                    CACHE_SIZE,
                    HEAD_DIM,
                    RATIO,
                    RATIO_PAD,
                    CHUNK_ROWS,
                    True,
                    HAS_PREV_ROWS,
                )
            else:
                # Read groups without residuals directly if they fit in one block;
                # otherwise, use chunked pooling.
                if SINGLE_BLOCK:
                    if NO_PAD:
                        score = tl.load(score_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :])
                        score_max = tl.max(score, axis=0)
                        e = tl.exp(score - score_max[None, :])
                        exp_sum = tl.sum(e, axis=0)
                        prob = e / exp_sum[None, :]
                        kv_vals = tl.load(kv_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :])
                    else:
                        row_valid = rows < RATIO
                        score = tl.load(
                            score_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :],
                            mask=row_valid[:, None],
                            other=-float("inf"),
                        )
                        score_max = tl.max(score, axis=0)
                        e = tl.exp(score - score_max[None, :])
                        e = tl.where(row_valid[:, None], e, 0.0)
                        exp_sum = tl.sum(e, axis=0)
                        prob = e / exp_sum[None, :]
                        kv_vals = tl.load(
                            kv_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :],
                            mask=row_valid[:, None],
                            other=0.0,
                        )
                    pooled = tl.sum(prob * kv_vals, axis=0)
                else:
                    pooled = _pooled_blocked(
                        group_idx,
                        start_pos,
                        seg_row_base,
                        cache_row,
                        kv_ptr,
                        score_ptr,
                        cache_ptr,
                        prev_rows_ptr,
                        -1,
                        offs_h,
                        CACHE_SIZE,
                        HEAD_DIM,
                        RATIO,
                        RATIO_PAD,
                        CHUNK_ROWS,
                        False,
                        False,
                    )
            if TOKEN_ALIGNED:
                # Keep Aurora's existing RMSNorm outside this kernel. Place
                # each completed pair at its original completion-token row.
                out_row = seg_row_base + (group_idx + 1) * RATIO - 1 - start_pos
                tl.store(out_ptr + out_row * HEAD_DIM + offs_h, pooled.to(tl.bfloat16))
            else:
                pooled_bf16 = pooled.to(tl.bfloat16).to(tl.float32)
                var = tl.sum(pooled_bf16 * pooled_bf16, axis=0) / HEAD_DIM
                norm_out = pooled_bf16 * (1.0 / tl.sqrt(var + eps)) * norm_w
                out_row = tl.load(meta_ptr + 2 * NUM_BATCH + batch) + local_group
                tl.store(out_ptr + out_row * HEAD_DIM + offs_h, norm_out.to(tl.bfloat16))


# ============================================================================
# K3: Cache update (only the last min(used_len, CACHE_SIZE) tokens per segment)
# ============================================================================


@triton.jit
def _cache_update_kernel(
    kv_ptr,
    score_ptr,
    cache_ptr,
    meta_ptr,
    block_table_ptr,
    total_write_slots,
    WRITE_SLOTS_PER_BATCH,  # Write slots per batch: min(max_used_len, CACHE_SIZE)
    NUM_BATCH,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PROTECT_NULL: tl.constexpr = False,
    NUM_CACHE_BLOCKS: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    offs_h = tl.arange(0, HEAD_DIM)
    for task_id in range(pid, total_write_slots, tl.num_programs(0)):
        batch = task_id // WRITE_SLOTS_PER_BATCH
        write_idx = task_id - batch * WRITE_SLOTS_PER_BATCH
        used_len = tl.load(meta_ptr + NUM_BATCH + batch)
        tail_count = used_len if used_len < CACHE_SIZE else CACHE_SIZE
        # Keep the cache element offset 64-bit for high physical page IDs.
        cache_row = tl.load(block_table_ptr + batch).to(tl.int64)
        valid = write_idx < tail_count
        if PROTECT_NULL:
            valid = valid & (cache_row > 0) & (cache_row < NUM_CACHE_BLOCKS)
        if valid:
            token_in_seg = used_len - tail_count + write_idx  # write_idx-th token in the segment tail
            proj_row = tl.load(meta_ptr + 3 * NUM_BATCH + batch) + token_in_seg
            token_pos = tl.load(meta_ptr + batch) + token_in_seg
            slot = token_pos % CACHE_SIZE
            cache_row = tl.load(block_table_ptr + batch).to(tl.int64)
            kv_vals = tl.load(kv_ptr + proj_row * HEAD_DIM + offs_h)
            score_vals = tl.load(score_ptr + proj_row * HEAD_DIM + offs_h)
            tl.store(cache_ptr + (cache_row * CACHE_SIZE + slot) * 2 * HEAD_DIM + offs_h, kv_vals)
            tl.store(cache_ptr + (cache_row * CACHE_SIZE + slot) * 2 * HEAD_DIM + HEAD_DIM + offs_h, score_vals)


# ============================================================================
# Host: Normalize layouts, pack controls and launch kernels
# ============================================================================


def _ints(v):
    """Read list/tuple/NumPy controls directly; tensors require one D2H copy for compatibility."""
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return list(v)


# Asynchronous upload ring for packed controls, using pinned memory and non_blocking copies.
# from_numpy().to(dev) performs synchronous H2D and waits for all previously queued kernels.
_META_UPLOAD_POOLS = {}  # (device, packed element count) -> _MetaUploadRing
_META_RING_SIZE = 32  # Must exceed the maximum number of calls the host can enqueue ahead of the device.


class _MetaUploadRing:
    """Keep pinned staging memory, a device copy and an upload event in each slot.

    Wait for the event before reusing a slot so the staging buffer is not
    overwritten while its previous DMA transfer is still in progress.
    """

    def __init__(self, n_int, dev):
        self.pinned_staging = [torch.empty(n_int, dtype=torch.int32).pin_memory() for _ in range(_META_RING_SIZE)]
        self.device_slots = [torch.empty(n_int, dtype=torch.int32, device=dev) for _ in range(_META_RING_SIZE)]
        self.copy_done_events = [torch.npu.Event() for _ in range(_META_RING_SIZE)]
        self.uploads_launched = 0

    def upload(self, meta_pack):
        slot = self.uploads_launched % _META_RING_SIZE
        self.uploads_launched += 1
        if self.uploads_launched > _META_RING_SIZE:
            self.copy_done_events[slot].synchronize()
        self.pinned_staging[slot].numpy()[:] = meta_pack
        self.device_slots[slot].copy_(self.pinned_staging[slot], non_blocking=True)
        self.copy_done_events[slot].record()
        return self.device_slots[slot]


def _meta_upload(meta_np, dev):
    pool_key = (str(dev), meta_np.size)
    if pool_key not in _META_UPLOAD_POOLS:
        _META_UPLOAD_POOLS[pool_key] = _MetaUploadRing(meta_np.size, dev)
    return _META_UPLOAD_POOLS[pool_key].upload(meta_np)


_CUBE_CORE_NUM = None  # AI core count, cached after the first query


def _cube_core_num():
    """Return the device AI core count used for grid parallelism.

    multi_processor_count counts vector cores (twice num_aicore), so it
    cannot be used as the upper bound.
    """
    global _CUBE_CORE_NUM
    if _CUBE_CORE_NUM is None:
        driver = triton.runtime.driver

        _CUBE_CORE_NUM = driver.active.utils.get_device_properties(torch.npu.current_device())["num_aicore"]
    return _CUBE_CORE_NUM


def compressor(
    x,
    wkv,
    wgate,
    state_cache,
    cmp_ratio,
    norm_w,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    num_cores=None,
):
    """Return cmp_kv in BF16.

    cu_seqlens / seqused / start_pos are host controls; list[int] or NumPy arrays
    are preferred. Tensor inputs are supported but require D2H synchronization.
    state_cache / state_block_table are device tensors. num_cores controls
    parallelism and defaults to the device AI core count.
    """
    hidden_dim = x.shape[-1]
    head_dim = wkv.shape[0]
    assert hidden_dim == 5120 and head_dim == 512, f"规格约束 hidden=5120/D=512, got {hidden_dim}/{head_dim}"
    ratio = int(cmp_ratio)
    assert 2 <= ratio <= 128
    # Normalize layouts to obtain the batch count and per-segment lengths.
    cu_list = _ints(cu_seqlens)
    if cu_list is not None:
        num_batch = len(cu_list) - 1
        seg_len = [cu_list[i + 1] - cu_list[i] for i in range(num_batch)]
        seg_row_base = [cu_list[i] for i in range(num_batch)]
        max_seg_len = max(seg_len)
        is_packed = True
    else:
        num_batch = x.shape[0]
        seg_len = [x.shape[1]] * num_batch
        seg_row_base = [i * x.shape[1] for i in range(num_batch)]
        max_seg_len = x.shape[1]
        is_packed = False
    total_tokens = x.shape[0] if is_packed else num_batch * max_seg_len
    used_lens = _ints(seqused) if seqused is not None else list(seg_len)
    start_pos_list = _ints(start_pos) if start_pos is not None else [0] * num_batch
    cache_size = state_cache.shape[1]

    # Decode task_id -> (batch, slot) in O(1); kernels skip out-of-range slots.
    # Upload packed [start_pos | used_len | out_row_offset | seg_row_base | batch_idx] in one H2D copy.
    max_used_len = max(used_lens)
    group_slots_per_batch = (max_used_len + ratio - 1) // ratio
    write_slots_per_batch = min(max_used_len, cache_size)
    total_group_slots = num_batch * group_slots_per_batch  # At least the sum of per-batch group counts
    total_write_slots = num_batch * write_slots_per_batch
    dev = x.device
    # Compute valid group counts per segment and the output layout.
    group_counts = [(spb + used_lens[b]) // ratio - spb // ratio for b, spb in enumerate(start_pos_list)]
    out_rows_per_batch = (max_seg_len + ratio - 1) // ratio
    if is_packed:
        out_row_offsets, running = [], 0
        for b in range(num_batch):
            out_row_offsets.append(running)
            running += group_counts[b]
        out = torch.zeros(
            min(total_tokens, total_tokens // ratio + num_batch), head_dim, dtype=torch.bfloat16, device=dev
        )
    else:
        out_row_offsets = [b * out_rows_per_batch for b in range(num_batch)]
        out = torch.zeros(num_batch, out_rows_per_batch, head_dim, dtype=torch.bfloat16, device=dev)
        out = out.view(-1, head_dim)
    meta_pack = np.concatenate(
        [
            np.asarray(start_pos_list, dtype=np.int32),
            np.asarray(used_lens, dtype=np.int32),
            np.asarray(out_row_offsets, dtype=np.int32),
            np.asarray(seg_row_base, dtype=np.int32),
            np.asarray(range(num_batch), dtype=np.int32),
        ]
    )
    meta_dev = _meta_upload(meta_pack, dev)
    block_table_ptr = (
        state_block_table.to(torch.int32).to(dev)
        if state_block_table is not None
        else meta_dev[4 * num_batch : 5 * num_batch]
    )

    # K1: Two projections.
    kv = torch.empty(total_tokens, head_dim, dtype=torch.float32, device=dev)
    score = torch.empty(total_tokens, head_dim, dtype=torch.float32, device=dev)
    x_2d = x if is_packed else x.view(total_tokens, hidden_dim)
    cores = num_cores or _cube_core_num()
    gemm_grid = lambda META: (min(cores, triton.cdiv(total_tokens, META["BM"]) * (head_dim // META["BN"])),)
    _proj_kernel[gemm_grid](kv, x_2d, wkv, total_tokens, IN_DIM=hidden_dim, OUT_DIM=head_dim)
    _proj_kernel[gemm_grid](score, x_2d, wgate, total_tokens, IN_DIM=hidden_dim, OUT_DIM=head_dim)

    # K2 group pooling, then K3 cache update: finish reads before writes.
    padded_ratio = max(2, triton.next_power_of_2(ratio))
    if total_group_slots > 0:
        pool_grid = (min(cores, total_group_slots),)
        _pool_kernel[pool_grid](
            out,
            kv,
            score,
            state_cache,
            None,
            None,
            norm_w,
            meta_dev,
            block_table_ptr,
            total_group_slots,
            group_slots_per_batch,
            num_batch,
            CACHE_SIZE=cache_size,
            HEAD_DIM=head_dim,
            RATIO=ratio,
            RATIO_PAD=padded_ratio,
            CHUNK_ROWS=min(padded_ratio, MAX_CHUNK_ROWS),
            eps=1e-20,
            SINGLE_BLOCK=(padded_ratio <= MAX_CHUNK_ROWS),
            NO_PAD=(padded_ratio == ratio),
            HAS_PREV_ROWS=False,
        )
    if total_write_slots > 0:
        cache_grid = (min(cores, total_write_slots),)
        _cache_update_kernel[cache_grid](
            kv,
            score,
            state_cache,
            meta_dev,
            block_table_ptr,
            total_write_slots,
            write_slots_per_batch,
            num_batch,
            CACHE_SIZE=cache_size,
            HEAD_DIM=head_dim,
        )
    shape = (num_batch, out_rows_per_batch, head_dim) if not is_packed else (out.shape[0], head_dim)
    return out.view(*shape)


# ============================================================================
# Reference implementation (pure PyTorch, per batch, with matching in-place cache updates)
# ============================================================================


def compressor_ref(
    x, wkv, wgate, state_cache, cmp_ratio, norm_w, state_block_table=None, cu_seqlens=None, seqused=None, start_pos=None
):
    head_dim = wkv.shape[0]
    ratio = int(cmp_ratio)
    if cu_seqlens is not None:
        cu_list = cu_seqlens.cpu().tolist()
        num_batch = len(cu_list) - 1
        segs = [x[cu_list[i] : cu_list[i + 1]] for i in range(num_batch)]
    else:
        num_batch = x.shape[0]
        segs = [x[i] for i in range(num_batch)]
    used_lens = seqused.cpu().tolist() if seqused is not None else [s.shape[0] for s in segs]
    start_pos_list = _ints(start_pos) if start_pos is not None else [0] * num_batch
    block_table = state_block_table.cpu().tolist() if state_block_table is not None else list(range(num_batch))
    cache_size = state_cache.shape[1]
    batch_outputs, group_counts_all = [], []
    for b in range(num_batch):
        xs = segs[b][: used_lens[b]]
        kv = torch.nn.functional.linear(xs.float(), wkv.float())
        scores = torch.nn.functional.linear(xs.float(), wgate.float())
        seg_start_pos = start_pos_list[b]
        first_group = seg_start_pos // ratio
        residual = seg_start_pos - first_group * ratio
        residual_rows = []
        for j in range(residual):
            pos = first_group * ratio + j
            slot = pos % cache_size
            residual_rows.append(state_cache[block_table[b], slot])
        seg_rows = (
            torch.stack([torch.cat([k, s]) for k, s in zip(kv, scores)])
            if len(kv)
            else torch.empty(0, 2 * head_dim, dtype=torch.float32, device=x.device)
        )
        tokens = torch.cat(
            [
                torch.stack(residual_rows)
                if residual_rows
                else torch.empty(0, 2 * head_dim, dtype=torch.float32, device=x.device),
                seg_rows,
            ]
        )
        token_count = tokens.shape[0]
        group_count = token_count // ratio
        out_batch = torch.empty(group_count, head_dim, dtype=torch.bfloat16, device=x.device)
        for g in range(group_count):
            group_tokens = tokens[g * ratio : (g + 1) * ratio]
            kv_part = group_tokens[:, :head_dim]
            score_part = group_tokens[:, head_dim:]
            prob = score_part.softmax(dim=0)
            pooled = (kv_part * prob).sum(dim=0)
            pooled_bf16 = pooled.to(torch.bfloat16).float()
            var = pooled_bf16.square().mean()
            out_batch[g] = (pooled_bf16 * torch.rsqrt(var + 1e-20) * norm_w.float()).to(torch.bfloat16)
        batch_outputs.append(out_batch)
        group_counts_all.append(group_count)
        for t in range(used_lens[b]):
            pos = seg_start_pos + t
            state_cache[block_table[b], pos % cache_size, :head_dim] = kv[t]
            state_cache[block_table[b], pos % cache_size, head_dim:] = scores[t]
    if cu_seqlens is None:
        seg_len_max = x.shape[1]
        rows_per_batch = (seg_len_max + ratio - 1) // ratio
        out_full = torch.zeros(num_batch, rows_per_batch, head_dim, dtype=torch.bfloat16, device=x.device)
        for b in range(num_batch):
            out_full[b, : group_counts_all[b]] = batch_outputs[b]
        return out_full
    out_offset = 0
    out_full = torch.zeros(
        min(x.shape[0], x.shape[0] // ratio + num_batch), head_dim, dtype=torch.bfloat16, device=x.device
    )
    for b in range(num_batch):
        out_full[out_offset : out_offset + group_counts_all[b]] = batch_outputs[b]
        out_offset += group_counts_all[b]
    return out_full


@triton.jit(do_not_specialize=["num_writes", "num_requests"])
def _write_pcp_ring_rows_kernel(
    kv_score_rows_ptr,
    ring_state_ptr,
    ring_metadata_ptr,
    source_row_indices_ptr,
    request_indices_ptr,
    token_offsets_ptr,
    num_writes,
    num_requests,
    NUM_BLOCKS: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    RING_ROWS: tl.constexpr,
):
    write_idx = tl.program_id(0)
    columns = tl.arange(0, ROW_WIDTH)
    request_idx = tl.load(request_indices_ptr + write_idx, mask=write_idx < num_writes, other=-1)
    if (write_idx < num_writes) & (request_idx >= 0) & (request_idx < num_requests):
        block_idx = tl.load(ring_metadata_ptr + 4 * num_requests + request_idx).to(tl.int64)
        num_tokens = tl.load(ring_metadata_ptr + num_requests + request_idx)
        if (block_idx > 0) & (block_idx < NUM_BLOCKS) & (num_tokens > 0):
            source_row_idx = tl.load(source_row_indices_ptr + write_idx).to(tl.int64)
            start_pos = tl.load(ring_metadata_ptr + request_idx)
            token_offset = tl.load(token_offsets_ptr + write_idx)
            ring_slot = (start_pos + token_offset) % RING_ROWS
            kv_score_row = tl.load(kv_score_rows_ptr + source_row_idx * ROW_WIDTH + columns)
            tl.store(ring_state_ptr + (block_idx * RING_ROWS + ring_slot) * ROW_WIDTH + columns, kv_score_row)


def write_pcp_ring_rows(
    gathered_kv_scores, ring_state, ring_metadata, source_row_indices, request_indices, token_offsets
):
    """Write request-relative rows using rejection-corrected device ring starts."""
    if source_row_indices.numel():
        _write_pcp_ring_rows_kernel[(source_row_indices.numel(),)](
            gathered_kv_scores,
            ring_state,
            ring_metadata,
            source_row_indices,
            request_indices,
            token_offsets,
            num_writes=source_row_indices.numel(),
            num_requests=ring_metadata.shape[1],
            NUM_BLOCKS=ring_state.shape[0],
            ROW_WIDTH=ring_state.shape[-1],
            RING_ROWS=STATE_RING_ROWS,
        )


def compressor_from_projected(
    kv, scores, state_cache, metadata, out, *, max_query_len, num_cores, prev_rows, prev_row_indices
):
    """Pool C2 into token-aligned BF16 rows; PCP updates the ring after gathering.

    metadata is contiguous device INT32 [5, requests]: start positions, used
    lengths, output bases (reserved), input bases, and actual global block IDs.
    Controls must be bounded by the input rows and request table; padded requests
    have used length and block ID zero. No tensor contents are read by the host.
    With prev_rows, chunk predecessors come from the gather or the old ring,
    and the caller writes the global request tails after all local reads.
    """
    if kv.dtype != torch.float32 or scores.dtype != torch.float32 or state_cache.dtype != torch.float32:
        raise ValueError("Aurora projections and ring state must be FP32")
    if kv.ndim != 2 or scores.shape != kv.shape:
        raise ValueError("Expected matching [tokens, width] projections")
    tokens, width = kv.shape
    if width < 1 or width & (width - 1):
        raise ValueError("Compressor width must be a positive power of two")
    if state_cache.ndim != 3 or state_cache.shape[1:] != (STATE_RING_ROWS, 2 * width):
        raise ValueError(f"Expected [blocks, {STATE_RING_ROWS}, 2*width] ring state")
    if out.dtype != torch.bfloat16 or out.shape != kv.shape:
        raise ValueError("Expected token-aligned BF16 pooled output")
    if metadata.dtype != torch.int32 or metadata.ndim != 2 or metadata.shape[0] != 5:
        raise ValueError("Expected INT32 [5, requests] device metadata")
    tensors = [kv, scores, state_cache, metadata, out]
    if (prev_rows is None) != (prev_row_indices is None):
        raise ValueError("Expected both gathered rows and predecessor indices, or neither")
    if prev_rows is not None:
        if prev_rows.dtype != torch.float32 or prev_rows.ndim != 2 or prev_rows.shape[1] != 2 * width:
            raise ValueError("Expected FP32 [gathered rows, 2*width] projections")
        if prev_row_indices.dtype != torch.int64 or prev_row_indices.shape != (metadata.shape[1],):
            raise ValueError("Expected one INT64 predecessor index per local chunk")
        tensors.extend((prev_rows, prev_row_indices))
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("Compressor requires contiguous projections, ring pages and controls")
    if not all(t.device == kv.device for t in tensors):
        raise ValueError("Compressor tensors must share one device")
    if num_cores <= 0 or not 0 <= max_query_len <= tokens:
        raise ValueError("Invalid compressor launch bounds")
    out.zero_()
    batches = metadata.shape[1]
    if tokens == 0 or batches == 0 or max_query_len == 0:
        return out
    if kv.device.type != "npu":
        raise ValueError("Projected Triton compressor requires an NPU")
    groups = (max_query_len + 1) // 2
    writes = min(max_query_len, STATE_RING_ROWS)
    # Same-stream launch order is required: residual reads precede ring writes.
    _pool_kernel[(min(num_cores, batches * groups),)](
        out,
        kv,
        scores,
        state_cache,
        prev_rows,
        prev_row_indices,
        out,
        metadata,
        metadata[4],
        batches * groups,
        groups,
        batches,
        CACHE_SIZE=STATE_RING_ROWS,
        HEAD_DIM=width,
        RATIO=2,
        RATIO_PAD=2,
        CHUNK_ROWS=2,
        eps=0.0,
        SINGLE_BLOCK=True,
        NO_PAD=True,
        TOKEN_ALIGNED=True,
        NUM_CACHE_BLOCKS=state_cache.shape[0],
        HAS_PREV_ROWS=prev_rows is not None,
    )
    if prev_rows is not None:
        return out
    _cache_update_kernel[(min(num_cores, batches * writes),)](
        kv,
        scores,
        state_cache,
        metadata,
        metadata[4],
        batches * writes,
        writes,
        batches,
        CACHE_SIZE=STATE_RING_ROWS,
        HEAD_DIM=width,
        PROTECT_NULL=True,
        NUM_CACHE_BLOCKS=state_cache.shape[0],
    )
    return out
