"""Compressor 算子 triton 实现。

kernel 结构：
  K1 投影 GEMM ×2（bf16 权重，swizzle，block 由 autotune 按 M 择优）
  K2 组池化（变长/残余拼接，两遍 softmax；段首残余从 cache 读取）
  K3 cache 更新（只写每段最后 cache_size 个 token，每 slot 唯一写者）
K2 必须先于 K3 执行：段尾写入与段首残余读取会命中同一 slot，并发时读取
结果会被本段写入覆盖。
compressor_ref 为纯 torch 参考实现。
"""

import numpy as np
import torch
from vllm.triton_utils import tl, triton

DEV = "npu"
MAX_CHUNK_ROWS = 8  # 池化组内分块行数上限（UB 容量：三遍循环中间量 <192KB）


# ============================================================================
# K1: 投影 GEMM（bf16 权重，swizzle，fp32 输出）
# ============================================================================

# autotune 候选（key=['M']）。BK 影响 fp32 累加分组（ulp 级差异），在精度容差内。
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
    """x (M, IN_DIM) @ w^T，w 为 (OUT_DIM, IN_DIM) 行主序，输出 fp32。"""
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
# K2: 组池化（task_id -> (batch, local_group) O(1) 除法解码；控制量 kernel 内 load）
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
    offs_h,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    RATIO_PAD: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    HAS_RES,
):
    """组内分块三遍循环（max / denom / 加权和）。HAS_RES=0 时不访问 cache。

    组覆盖全局 token [group_idx*RATIO, (group_idx+1)*RATIO)：seg_off >= 0 读
    投影大矩阵 (seg_row_base+seg_off)，< 0 为段前残余，读 cache
    (cache_row, token_pos % CACHE_SIZE)。"""
    # Keep masked history lanes inside the projection allocation. Ascend may
    # form a DMA address for a masked lane before applying the load mask.
    # The original masks and ring selection still supply all history values.
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
                cache_ptr + (cache_row * CACHE_SIZE + slot[:, None]) * 2 * HEAD_DIM + HEAD_DIM + offs_h[None, :],
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
                cache_ptr + (cache_row * CACHE_SIZE + slot[:, None]) * 2 * HEAD_DIM + HEAD_DIM + offs_h[None, :],
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
                cache_ptr + (cache_row * CACHE_SIZE + slot[:, None]) * 2 * HEAD_DIM + HEAD_DIM + offs_h[None, :],
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
                cache_ptr + (cache_row * CACHE_SIZE + slot[:, None]) * 2 * HEAD_DIM + offs_h[None, :],
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
    norm_w_ptr,
    meta_ptr,  # 拼包 [start_pos | used_len | out_row_offset | seg_row_base]
    block_table_ptr,
    total_group_slots,
    GROUP_SLOTS_PER_BATCH,  # 每 batch 组槽位上界 ceil(max_used_len / RATIO)
    NUM_BATCH,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    RATIO_PAD: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    eps: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    NO_PAD: tl.constexpr,
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
        cache_row = tl.load(block_table_ptr + batch)
        valid = local_group < groups_in_batch
        if TOKEN_ALIGNED:
            valid = valid & (cache_row > 0) & (cache_row < NUM_CACHE_BLOCKS)
        if valid:
            group_idx = start_pos // RATIO + local_group
            seg_row_base = tl.load(meta_ptr + 3 * NUM_BATCH + batch)  # [3*NUM_BATCH, 4*NUM_BATCH)
            rows = tl.arange(0, CHUNK_ROWS)
            token_pos = group_idx * RATIO + rows
            seg_off = token_pos - start_pos
            residual = start_pos - group_idx * RATIO  # 段前残余数，仅 local_group==0 时可能 >0
            cache_row = tl.load(block_table_ptr + batch)
            if residual > 0:
                # 残余组（每 batch 至多 1 个）：cache 拼接
                pooled = _pooled_blocked(
                    group_idx,
                    start_pos,
                    seg_row_base,
                    cache_row,
                    kv_ptr,
                    score_ptr,
                    cache_ptr,
                    offs_h,
                    CACHE_SIZE,
                    HEAD_DIM,
                    RATIO,
                    RATIO_PAD,
                    CHUNK_ROWS,
                    True,
                )
            else:
                # 非残余组：单块直读；RATIO 超出单块容量时走分块版
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
                        offs_h,
                        CACHE_SIZE,
                        HEAD_DIM,
                        RATIO,
                        RATIO_PAD,
                        CHUNK_ROWS,
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
# K3: cache 更新（只写每段最后 min(used_len, CACHE_SIZE) 个 token）
# ============================================================================


@triton.jit
def _cache_update_kernel(
    kv_ptr,
    score_ptr,
    cache_ptr,
    meta_ptr,
    block_table_ptr,
    total_write_slots,
    WRITE_SLOTS_PER_BATCH,  # 每 batch 写槽位上界 min(max_used_len, CACHE_SIZE)
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
        cache_row = tl.load(block_table_ptr + batch)
        valid = write_idx < tail_count
        if PROTECT_NULL:
            valid = valid & (cache_row > 0) & (cache_row < NUM_CACHE_BLOCKS)
        if valid:
            token_in_seg = used_len - tail_count + write_idx  # 最后 tail_count 个 token 中的第 write_idx 个
            proj_row = tl.load(meta_ptr + 3 * NUM_BATCH + batch) + token_in_seg
            token_pos = tl.load(meta_ptr + batch) + token_in_seg
            slot = token_pos % CACHE_SIZE
            cache_row = tl.load(block_table_ptr + batch)
            kv_vals = tl.load(kv_ptr + proj_row * HEAD_DIM + offs_h)
            score_vals = tl.load(score_ptr + proj_row * HEAD_DIM + offs_h)
            tl.store(cache_ptr + (cache_row * CACHE_SIZE + slot) * 2 * HEAD_DIM + offs_h, kv_vals)
            tl.store(cache_ptr + (cache_row * CACHE_SIZE + slot) * 2 * HEAD_DIM + HEAD_DIM + offs_h, score_vals)


# ============================================================================
# Host：布局归一、控制量拼包、launch
# ============================================================================


def _ints(v):
    """控制量解析：list/tuple/np.ndarray 直取；torch.Tensor 走一次 D2H（仅兼容）。"""
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return list(v)


# 控制量拼包的异步上传环：pinned + non_blocking 拷贝。
# from_numpy().to(dev) 为同步 H2D，会阻塞至此前入队的全部 kernel 完成。
_META_UPLOAD_POOLS = {}  # (device, 拼包元素数) → _MetaUploadRing
_META_RING_SIZE = 32  # 槽位数，须大于 host 领先 device 的最大调用数


class _MetaUploadRing:
    """每槽位含锁页中转、device 副本与上传完成事件；复用槽位前等待其事件，
    避免写入中转缓冲时上一次 DMA 尚未完成。"""

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


_CUBE_CORE_NUM = None  # AI core 数，首次查询后缓存


def _cube_core_num():
    """设备 AI core 数（grid 并行单位）。

    multi_processor_count 为 vector core 数（num_aicore 的 2 倍），不可用作上限。"""
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
    """返回 cmp_kv（bf16）。

    cu_seqlens / seqused / start_pos 为 host 控制量（推荐 list[int]/np.ndarray，
    tensor 兼容但有 D2H 同步）；state_cache / state_block_table 为 device tensor。
    num_cores 为并行核数，缺省取设备 AI core 数。
    """
    hidden_dim = x.shape[-1]
    head_dim = wkv.shape[0]
    assert hidden_dim == 5120 and head_dim == 512, f"规格约束 hidden=5120/D=512, got {hidden_dim}/{head_dim}"
    ratio = int(cmp_ratio)
    assert 2 <= ratio <= 128
    # batch 数与每段长度（布局归一）
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

    # 槽位网格：task_id -> (batch, 槽位) O(1) 除法解码，越界槽位 kernel 内跳过；
    # 控制量拼包 [start_pos | used_len | out_row_offset | seg_row_base | batch_idx] 一次 H2D
    max_used_len = max(used_lens)
    group_slots_per_batch = (max_used_len + ratio - 1) // ratio
    write_slots_per_batch = min(max_used_len, cache_size)
    total_group_slots = num_batch * group_slots_per_batch  # ≥ Σ组数
    total_write_slots = num_batch * write_slots_per_batch
    dev = x.device
    # 每段有效组数与输出布局
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

    # K1 投影 ×2
    kv = torch.empty(total_tokens, head_dim, dtype=torch.float32, device=dev)
    score = torch.empty(total_tokens, head_dim, dtype=torch.float32, device=dev)
    x_2d = x if is_packed else x.view(total_tokens, hidden_dim)
    cores = num_cores or _cube_core_num()
    gemm_grid = lambda META: (min(cores, triton.cdiv(total_tokens, META["BM"]) * (head_dim // META["BN"])),)
    _proj_kernel[gemm_grid](kv, x_2d, wkv, total_tokens, IN_DIM=hidden_dim, OUT_DIM=head_dim)
    _proj_kernel[gemm_grid](score, x_2d, wgate, total_tokens, IN_DIM=hidden_dim, OUT_DIM=head_dim)

    # K2 组池化 → K3 cache 更新（顺序执行，先读后写）
    padded_ratio = max(2, triton.next_power_of_2(ratio))
    if total_group_slots > 0:
        pool_grid = (min(cores, total_group_slots),)
        _pool_kernel[pool_grid](
            out,
            kv,
            score,
            state_cache,
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
# 参考实现（纯 torch，逐 batch；cache 原地更新与实现一致）
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


def compressor_from_projected(kv, scores, state_cache, metadata, out, *, max_query_len, num_cores):
    """Pool C2 into token-aligned BF16 rows, then update a private FP32 ring.

    metadata is contiguous device INT32 [5, requests]: start positions, used
    lengths, output bases (reserved), input bases, and actual global block IDs.
    Controls must be bounded by the input rows and request table; padded requests
    have used length and block ID zero. No tensor contents are read by the host.
    """
    if kv.dtype != torch.float32 or scores.dtype != torch.float32 or state_cache.dtype != torch.float32:
        raise ValueError("Aurora projections and ring state must be FP32")
    if kv.ndim != 2 or scores.shape != kv.shape:
        raise ValueError("Expected matching [tokens, width] projections")
    tokens, width = kv.shape
    if width < 1 or width & (width - 1):
        raise ValueError("Compressor width must be a positive power of two")
    if state_cache.ndim != 3 or state_cache.shape[1:] != (32, 2 * width):
        raise ValueError("Expected [blocks, 32, 2*width] ring state")
    if out.dtype != torch.bfloat16 or out.shape != kv.shape:
        raise ValueError("Expected token-aligned BF16 pooled output")
    if metadata.dtype != torch.int32 or metadata.ndim != 2 or metadata.shape[0] != 5:
        raise ValueError("Expected INT32 [5, requests] device metadata")
    tensors = (kv, scores, state_cache, metadata, out)
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
    writes = min(max_query_len, 32)
    # Same-stream launch order is required: residual reads precede ring writes.
    _pool_kernel[(min(num_cores, batches * groups),)](
        out,
        kv,
        scores,
        state_cache,
        out,
        metadata,
        metadata[4],
        batches * groups,
        groups,
        batches,
        CACHE_SIZE=32,
        HEAD_DIM=width,
        RATIO=2,
        RATIO_PAD=2,
        CHUNK_ROWS=2,
        eps=0.0,
        SINGLE_BLOCK=True,
        NO_PAD=True,
        TOKEN_ALIGNED=True,
        NUM_CACHE_BLOCKS=state_cache.shape[0],
    )
    _cache_update_kernel[(min(num_cores, batches * writes),)](
        kv,
        scores,
        state_cache,
        metadata,
        metadata[4],
        batches * writes,
        writes,
        batches,
        CACHE_SIZE=32,
        HEAD_DIM=width,
        PROTECT_NULL=True,
        NUM_CACHE_BLOCKS=state_cache.shape[0],
    )
    return out
