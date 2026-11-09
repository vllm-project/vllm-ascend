#!/usr/bin/python
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""QuantLightningIndexerV2 DCP (Decode Context Parallel, 解码上下文并行) 专项测试.

覆盖:
1. 单 rank 输出与 CPU golden 一致 (本地 index 集合 + BF16 score);
2. 多 rank 输出按 issue #7 的协议合并后, 与同一全局序列按 P=1 的结果一致;
3. 边界: C 落在 P*S 边界、末 rank 截断、L_r=0、Q_len>1、mask_mode=0/3。

用法:
    pytest -s test_quant_lightning_indexer_v2_dcp.py
    pytest -s test_quant_lightning_indexer_v2_dcp.py -k "case_p2_s1"
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import quant_lightning_indexer_v2_golden as golden_lib  # noqa: E402

try:
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover - CPU-only dev box
    torch_npu = None

# 可选: 直接加载独立编译的最小扩展 (.so), 用于没有安装 vllm_ascend/cann_ops_transformer 的环境
_QLI_DCP_EXT = os.environ.get("QLI_DCP_EXT", "").strip()
if _QLI_DCP_EXT and torch_npu is not None:
    torch.ops.load_library(_QLI_DCP_EXT)

HEAD_DIM = 128
N1 = 64
N2 = 1
BLOCK_SIZE = 16


def _require_npu():
    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available")
    if _qli_ops() is None:
        pytest.skip("neither vllm_ascend._C nor cann_ops_transformer is installed")


_QLI_BACKEND = None


def _qli_backend():
    """优先使用 vllm_ascend 的 _C_ascend 绑定, 回退到独立的 cann_ops_transformer 扩展."""
    global _QLI_BACKEND
    if _QLI_BACKEND is None:
        if hasattr(torch.ops, "_C_ascend") and hasattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2"):
            _QLI_BACKEND = ("vllm", torch.ops._C_ascend)
        elif hasattr(torch.ops, "cann_ops_transformer") and hasattr(
            torch.ops.cann_ops_transformer, "quant_lightning_indexer"
        ):
            _QLI_BACKEND = ("cann", torch.ops.cann_ops_transformer)
    return _QLI_BACKEND


def _qli_ops():
    backend = _qli_backend()
    return None if backend is None else backend[1]


def _meta_op_name():
    return "npu_quant_lightning_indexer_v2_metadata" if _qli_backend()[0] == "vllm" else "quant_lightning_indexer_metadata"


def _indexer_op_name():
    return "npu_quant_lightning_indexer_v2" if _qli_backend()[0] == "vllm" else "quant_lightning_indexer"


def dcp_local_indices(global_len, cp_world, cp_rank, cp_interleave):
    """全局 token 序号 -> 本 rank 本地逻辑序号 (本地按时间序)."""
    return [t for t in range(global_len) if (t // cp_interleave) % cp_world == cp_rank]


def dcp_global_index(local_idx, cp_world, cp_rank, cp_interleave):
    return (local_idx // cp_interleave * cp_world + cp_rank) * cp_interleave + local_idx % cp_interleave


def dcp_visible_len(global_end, q_len, row_idx, local_len, cp_world, cp_rank, cp_interleave):
    u = max(global_end - q_len + row_idx + 1, 0)
    if u <= 0:
        return 0
    period = cp_world * cp_interleave
    full_rounds = (u // period) * cp_interleave
    rem = u % period - cp_rank * cp_interleave
    tail = min(max(rem, 0), cp_interleave)
    return min(local_len, full_rounds + tail)


def _build_metadata(case, topk, mask_mode=3, enable_dcp=False, cmp_ratio=1, cp_world=1, cp_rank=0,
                    cp_interleave=1):
    ops = _qli_ops()
    return getattr(ops, _meta_op_name())(
        num_heads_q=N1,
        num_heads_k=N2,
        head_dim=HEAD_DIM,
        topk=topk,
        quant_mode=2,
        cu_seqlens_q=case["cu_seqlens_q"].npu(),
        seqused_k=case["seqused_k"].npu(),
        causal_seqlens_kv_global=case["causal_lens"].npu() if enable_dcp else None,
        batch_size=case["batch_size"],
        max_seqlen_q=case["q_len"],
        max_seqlen_k=case["max_l"],
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        # FD/LD 在 A2/A3 上只对 DCP 场景打开, metadata 需要知道当前是否 DCP;
        # A5 的 metadata 还需要 C/P/r/S 才能算出 DCP-aware 的 S2 上界。
        enable_dcp=int(bool(enable_dcp)),
        cp_world=cp_world,
        cp_rank=cp_rank,
        cp_kv_cache_interleave_size=cp_interleave,
    )


def _make_case(cp_world, cp_rank, cp_interleave, batch_size, q_len, causal_lens, topk, seed=0):
    """构造同一全局序列在指定 rank 上的本地输入 (PA_BBND 物理池 + 本地逻辑 K)."""
    rng = np.random.default_rng(seed)
    q_t = batch_size * q_len
    causal_lens = list(causal_lens)
    max_c = max(causal_lens)

    # 全局 K / 全局 qk 分数基准 (同一 seed 保证各 rank 看到相同的全局 token)
    global_k = rng.integers(-128, 128, size=(max_c, HEAD_DIM), dtype=np.int16).astype(np.int8)
    global_k_scale = rng.uniform(0.5, 1.5, size=(max_c,)).astype(np.float16)
    query = rng.integers(-128, 128, size=(q_t, N1, HEAD_DIM), dtype=np.int16).astype(np.int8)
    weights = rng.uniform(0.1, 1.0, size=(q_t, N1)).astype(np.float16)
    q_scale = rng.uniform(0.5, 1.5, size=(q_t, N1)).astype(np.float16)

    local_lens = []
    local_ks = []
    local_k_scales = []
    for b in range(batch_size):
        idx = dcp_local_indices(causal_lens[b], cp_world, cp_rank, cp_interleave)
        local_lens.append(len(idx))
        if idx:
            local_ks.append(global_k[idx])
            local_k_scales.append(global_k_scale[idx])
        else:
            local_ks.append(np.zeros((0, HEAD_DIM), dtype=np.int8))
            local_k_scales.append(np.zeros((0,), dtype=np.float16))
    max_l = max(local_lens)
    blocks_per_batch = max(1, (max_l + BLOCK_SIZE - 1) // BLOCK_SIZE)
    block_num = batch_size * blocks_per_batch

    # 物理池: 本地逻辑块随机置换到物理块
    key_pool = np.zeros((block_num, BLOCK_SIZE, N2, HEAD_DIM), dtype=np.int8)
    k_scale_pool = np.ones((block_num, BLOCK_SIZE, N2), dtype=np.float16)
    block_table = np.zeros((batch_size, blocks_per_batch), dtype=np.int32)
    local_k_padded = np.zeros((batch_size, N2, max_l, HEAD_DIM), dtype=np.int8)
    k_scale_logical = np.ones((batch_size, N2, max_l), dtype=np.float16)
    # 物理池: 每个 batch 分到互不重叠的物理块, 块内按本地逻辑块随机置换
    perm_all = rng.permutation(block_num)
    for b in range(batch_size):
        perm = perm_all[b * blocks_per_batch : (b + 1) * blocks_per_batch]
        block_table[b] = perm
        local_k_padded[b, 0, : local_lens[b]] = local_ks[b]
        k_scale_logical[b, 0, : local_lens[b]] = local_k_scales[b]
        for blk in range(blocks_per_batch):
            start = blk * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, local_lens[b])
            if end > start:
                key_pool[perm[blk], : end - start, 0] = local_ks[b][start:end]
                k_scale_pool[perm[blk], : end - start, 0] = k_scale_logical[b, 0, start:end]

    cu_seqlens_q = np.arange(0, q_t + 1, q_len, dtype=np.int32)
    return {
        "batch_size": batch_size,
        "q_len": q_len,
        "q_t": q_t,
        "topk": topk,
        "query": torch.from_numpy(query),
        "key_pool": torch.from_numpy(key_pool.reshape(block_num, BLOCK_SIZE, N2, HEAD_DIM)),
        "weights": torch.from_numpy(weights),
        "q_scale": torch.from_numpy(q_scale),
        "k_scale_pool": torch.from_numpy(k_scale_pool),
        "block_table": torch.from_numpy(block_table),
        "cu_seqlens_q": torch.from_numpy(cu_seqlens_q),
        "seqused_k": torch.tensor(local_lens, dtype=torch.int32),
        "causal_lens": torch.tensor(causal_lens, dtype=torch.int32),
        "local_k": torch.from_numpy(local_k_padded),
        "k_scale_logical": torch.from_numpy(k_scale_logical),
        "local_lens": local_lens,
        "max_l": max_l,
        "max_c": max_c,
    }


def _pad_case(case, pad_l, seed=0):
    """把本地 KV 补到 pad_l，模拟框架传入 padding 后的 seqused_k。

    可见长度仍由全局 causal 上界 C 决定，因此当 pad_l 远大于真实可见前缀时，
    metadata 切出的多数 S2 段对每一行都完全不可见；开 LD 后这些段只能贡献
    (-inf, -1) 的 partial，最终由归约核合并。
    """
    rng = np.random.default_rng(seed + 1)
    B = case["batch_size"]
    local_k = np.zeros((B, N2, pad_l, HEAD_DIM), dtype=np.int8)
    k_scale_logical = np.ones((B, N2, pad_l), dtype=np.float16)
    for b in range(B):
        l = int(case["local_lens"][b])
        local_k[b, 0, :l] = case["local_k"][b, 0, :l].numpy()
        k_scale_logical[b, 0, :l] = case["k_scale_logical"][b, 0, :l].numpy()
        if pad_l > l:
            local_k[b, 0, l:pad_l] = rng.integers(
                -128, 128, size=(pad_l - l, HEAD_DIM), dtype=np.int16
            ).astype(np.int8)
            k_scale_logical[b, 0, l:pad_l] = rng.uniform(0.5, 1.5, size=(pad_l - l,)).astype(np.float16)

    blocks_per_batch = max(1, (pad_l + BLOCK_SIZE - 1) // BLOCK_SIZE)
    block_num = B * blocks_per_batch
    key_pool = np.zeros((block_num, BLOCK_SIZE, N2, HEAD_DIM), dtype=np.int8)
    k_scale_pool = np.ones((block_num, BLOCK_SIZE, N2), dtype=np.float16)
    block_table = np.zeros((B, blocks_per_batch), dtype=np.int32)
    perm = rng.permutation(block_num)
    for b in range(B):
        perm_b = perm[b * blocks_per_batch:(b + 1) * blocks_per_batch]
        block_table[b] = perm_b
        for blk in range(blocks_per_batch):
            st = blk * BLOCK_SIZE
            ed = min(st + BLOCK_SIZE, pad_l)
            key_pool[perm_b[blk], :ed - st, 0] = local_k[b, 0, st:ed]
            k_scale_pool[perm_b[blk], :ed - st, 0] = k_scale_logical[b, 0, st:ed]

    padded = dict(case)
    padded.update(
        {
            "key_pool": torch.from_numpy(key_pool),
            "k_scale_pool": torch.from_numpy(k_scale_pool),
            "block_table": torch.from_numpy(block_table),
            "local_k": torch.from_numpy(local_k),
            "k_scale_logical": torch.from_numpy(k_scale_logical),
            "local_lens": [pad_l] * B,
            "max_l": pad_l,
            "seqused_k": torch.tensor([pad_l] * B, dtype=torch.int32),
        }
    )
    return padded


def _use_contiguous_block_table(case):
    """把 PA 的 block_table 改成连续物理块, 触发 arch35 KeyNd2NzForPA 相邻块合并."""
    batch_size = case["batch_size"]
    blocks_per_batch = case["block_table"].shape[1]
    block_num = case["key_pool"].shape[0]
    block_table = np.arange(block_num, dtype=np.int32).reshape(batch_size, blocks_per_batch)
    key_pool = torch.zeros_like(case["key_pool"])
    k_scale_pool = torch.ones_like(case["k_scale_pool"])
    for b in range(batch_size):
        for blk in range(blocks_per_batch):
            start = blk * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, case["local_lens"][b])
            if end <= start:
                continue
            phys = int(block_table[b, blk])
            key_pool[phys, : end - start, 0] = case["local_k"][b, 0, start:end]
            k_scale_pool[phys, : end - start, 0] = case["k_scale_logical"][b, 0, start:end]
    case["block_table"] = torch.from_numpy(block_table)
    case["key_pool"] = key_pool
    case["k_scale_pool"] = k_scale_pool


def _run_op(case, cp_world=1, cp_rank=0, cp_interleave=1, mask_mode=3, enable_dcp=True, return_value=1,
            idx_offset=None, cmp_ratio=1):
    ops = _qli_ops()
    metadata = _build_metadata(case, case["topk"], mask_mode, enable_dcp=enable_dcp, cmp_ratio=cmp_ratio,
                               cp_world=cp_world, cp_rank=cp_rank, cp_interleave=cp_interleave).npu()
    return getattr(ops, _indexer_op_name())(
        case["query"].npu(),
        case["key_pool"].npu(),
        case["weights"].npu(),
        case["q_scale"].npu(),
        case["k_scale_pool"].npu(),
        case["topk"],
        2,
        cu_seqlens_q=case["cu_seqlens_q"].npu(),
        seqused_k=case["seqused_k"].npu(),
        block_table=case["block_table"].npu(),
        output_idx_offset=idx_offset.npu() if idx_offset is not None else None,
        metadata=metadata,
        max_seqlen_q=case["q_len"],
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        return_value=return_value,
        causal_seqlens_kv_global=case["causal_lens"].npu() if enable_dcp else None,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
        cp_rank=cp_rank,
        cp_kv_cache_interleave_size=cp_interleave,
    )


def _run_golden(case, cp_world=1, cp_rank=0, cp_interleave=1, sparse_mode=3, dcp_enable=True):
    g = golden_lib.GeneralizedQLIV2(
        batch_size=case["batch_size"],
        q_seq=case["q_len"],
        k_seq=case["max_l"],
        q_t_size=case["q_t"],
        k_t_size=0,
        q_head_num=N1,
        k_head_num=N2,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        block_num=BLOCK_SIZE,
        qk_dtype=torch.int8,
        weight_dtype=torch.float16,
        dequant_dtype=torch.float16,
        actual_seq_dtype=torch.int32,
        cu_seqlens_q=case["cu_seqlens_q"].clone(),
        cu_seqlens_k=torch.zeros(case["batch_size"] + 1, dtype=torch.int32),
        act_seq_q=None,
        act_seq_k=case["seqused_k"].clone(),
        cmp_residual_k=torch.zeros(case["batch_size"], dtype=torch.int32),
        max_seqlen_q=case["q_len"],
        quant_mode=2,
        layout_query="TND",
        layout_key="PA_BBND",
        sparse_count=case["topk"],
        sparse_mode=sparse_mode,
        query_datarange=[-128, 127],
        key_datarange=[-128, 127],
        weights_datarange=[0.1, 1.0],
        q_scale_datarange=[0.5, 1.5],
        k_scale_datarange=[0.5, 1.5],
        cmp_ratio=1,
        return_value=1,
        dcp_enable=dcp_enable,
        cp_world=cp_world,
        cp_rank=cp_rank,
        cp_interleave=cp_interleave,
        causal_seqlens_kv_global=case["causal_lens"].clone(),
    )
    y, _, values = g.forward(
        case["query"],
        case["local_k"],
        case["weights"],
        case["q_scale"],
        case["k_scale_logical"],
        case["cu_seqlens_q"],
        torch.zeros(case["batch_size"] + 1, dtype=torch.int32),
        None,
        case["seqused_k"],
        None,
        None,
    )
    return y.numpy(), values.numpy()


def _to_bf16(value):
    """按 round-to-nearest-even 把 FP32 分数转成 BF16, 用于边界并列判定."""
    arr = np.array([value], dtype=np.float32)
    u = arr.view(np.uint32)
    lsb = (u >> 16) & 1
    u = (u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return u.view(np.float32)[0].item()


def _bf16_ulp(value):
    """BF16 在该数量级下的一个 ULP (7 位尾数)."""
    v = abs(float(value))
    if v == 0.0:
        return 2.0**-133
    return 2.0 ** (math.floor(math.log2(v)) - 7)


def _row_scores(case, cp_world, cp_rank, cp_interleave):
    """按算子数学定义逐行重建 golden 分数 (用于并列/边界仲裁), 返回 [T1, max_l]."""
    q = case["query"].to(torch.int32).numpy()
    k = case["local_k"][:, 0].to(torch.int32).numpy()
    w = case["weights"].to(torch.float32).numpy()
    qs = case["q_scale"].to(torch.float32).numpy()
    ks = case["k_scale_logical"][:, 0].to(torch.float32).numpy()
    cu = case["cu_seqlens_q"].numpy()
    scores = np.full((case["q_t"], case["max_l"]), -np.inf, dtype=np.float32)
    for b in range(case["batch_size"]):
        q_len = case["q_len"]
        local_len = case["local_lens"][b]
        global_end = int(case["causal_lens"][b])
        for i in range(q_len):
            row = int(cu[b]) + i
            qk = q[row] @ k[b].T  # [N1, L]
            qk = np.maximum(qk / 1024.0, 0.0)
            wrow = w[row] * qs[row]
            score = (wrow[:, None] * qk).sum(axis=0) * ks[b]
            vis = dcp_visible_len(global_end, q_len, i, local_len, cp_world, cp_rank, cp_interleave)
            scores[row, :vis] = score[:vis]
    return scores


def _compare_rank(case, cp_world, cp_rank, cp_interleave):
    out, values = _run_op(case, cp_world, cp_rank, cp_interleave)
    out = out.cpu().numpy()[:, 0, :]
    values = values.cpu().float().numpy()[:, 0, :]
    gold, gold_values = _run_golden(case, cp_world, cp_rank, cp_interleave)
    gold = gold[:, 0, :]
    gold_values = gold_values[:, 0, :].astype(np.float32)
    scores = None  # 仅在索引集合不一致 (并列边界) 时按需重建, 避免大 shape 重复计算

    for row in range(out.shape[0]):
        pred = out[row][out[row] >= 0]
        ref = gold[row][gold[row] >= 0]
        assert len(pred) == len(ref), f"row {row}: valid count {len(pred)} != {len(ref)}"
        if len(ref) == 0:
            continue
        pred_set, ref_set = set(pred.tolist()), set(ref.tolist())
        if pred_set != ref_set:
            if scores is None:
                scores = _row_scores(case, cp_world, cp_rank, cp_interleave)
            # 并列边界: 只允许"被替换成员与参考边界同分"
            only_pred = sorted(pred_set - ref_set)
            only_ref = sorted(ref_set - pred_set)
            boundary = min(scores[row, list(ref_set)]) if only_ref else None
            for i in only_pred:
                for j in only_ref:
                    # 算子内部按 BF16 sortable score 排序; 量化后同分、或分差在 1 个 BF16 ULP
                    # 以内, 都视为并列成员, 允许跨 rank/分核顺序不同 (设计文档 R1/R8)。
                    same_bf16 = _to_bf16(scores[row, i]) == _to_bf16(scores[row, j])
                    within_bf16_ulp = abs(scores[row, i] - scores[row, j]) <= _bf16_ulp(scores[row, j])
                    assert (
                        same_bf16
                        or within_bf16_ulp
                        or abs(scores[row, i] - scores[row, j]) <= 1e-3 * max(1.0, abs(scores[row, j]))
                    ), (
                        f"row {row}: index set mismatch pred-only={i} ref-only={j} "
                        f"score_pred={scores[row, i]} score_ref={scores[row, j]} boundary={boundary}"
                    )
        # score 校验: 用 golden 的 (index -> value) 映射, 不重建整行分数
        ref_map = {int(i): float(v) for i, v in zip(ref, gold_values[row][gold[row] >= 0])}
        for pos, idx in enumerate(out[row]):
            if idx < 0:
                continue
            ref_v = ref_map.get(int(idx))
            if ref_v is not None:
                np.testing.assert_allclose(float(values[row][pos]), ref_v, rtol=2e-2, atol=2e-3)
    return out, values


def _merge_global_per_row(rank_outputs, cp_world, cp_interleave, topk):
    """按行合并各 rank 的 (score, local_idx), 返回每行的 {global_idx: score} TopK."""
    num_rows = rank_outputs[0][0].shape[0]
    merged_rows = []
    for row in range(num_rows):
        cand = []
        for rank, (idx, val) in enumerate(rank_outputs):
            for local_idx, score in zip(idx[row], val[row]):
                if local_idx < 0:
                    continue
                cand.append((float(score), int(dcp_global_index(int(local_idx), cp_world, rank, cp_interleave))))
        cand.sort(key=lambda x: (-x[0], x[1]))  # 稳定排序: score 降序, 并列取更小全局序号
        merged_rows.append({g: s for s, g in cand[:topk]})
    return merged_rows


CASES = [
    # (P, S, B, Q_len, global lens per batch, topk)
    (2, 1, 1, 1, [257], 512),
    (2, 128, 1, 4, [257], 512),
    (4, 16, 1, 2, [255], 512),
    (2, 128, 2, 1, [129, 257], 512),
    (4, 128, 1, 1, [127], 512),   # 末 rank L=0
    (2, 64, 1, 4, [513], 2048),
    # 大 shape: P=8/16, 本地 L=2048/4096, topk 真正参与筛选
    (8, 128, 1, 4, [16384], 1024),
    (16, 128, 1, 4, [32768], 1024),
    (8, 256, 1, 2, [32768], 2048),
    (16, 1, 1, 4, [8192], 256),
]


@pytest.mark.ci
@pytest.mark.parametrize("cp_world,cp_interleave,batch_size,q_len,causal_lens,topk", CASES)
def test_qliv2_dcp(cp_world, cp_interleave, batch_size, q_len, causal_lens, topk):
    _require_npu()
    rank_outputs = []
    for rank in range(cp_world):
        case = _make_case(
            cp_world, rank, cp_interleave, batch_size, q_len, causal_lens, topk, seed=0
        )
        idx, val = _compare_rank(case, cp_world, rank, cp_interleave)
        rank_outputs.append((idx, val))

    # 多 rank 合并 == 同一全局序列按 P=1 的结果 (边界 BF16 并列容忍)
    merged_maps = _merge_global_per_row(rank_outputs, cp_world, cp_interleave, topk)
    ref_case = _make_case(1, 0, 1, batch_size, q_len, causal_lens, topk, seed=0)
    ref_out, ref_val = _run_op(ref_case, 1, 0, 1)
    ref_out = ref_out.cpu().numpy()[:, 0, :]
    ref_val = ref_val.cpu().float().numpy()[:, 0, :]
    for row, merged_map in enumerate(merged_maps):
        ref_map = {int(g): float(s) for g, s in zip(ref_out[row], ref_val[row]) if g >= 0}
        assert len(merged_map) == len(ref_map), f"row {row}: topk count {len(merged_map)} != {len(ref_map)}"
        boundary = min(min(merged_map.values(), default=0.0), min(ref_map.values(), default=0.0))
        # 边界以上 (不含并列带) 必须完全一致
        above_merged = {g for g, s in merged_map.items() if s > boundary}
        above_ref = {g for g, s in ref_map.items() if s > boundary}
        assert above_merged == above_ref, f"row {row}: above-boundary topk mismatch, diffs={sorted(above_merged ^ above_ref)}"
        # 对称差只允许是边界并列成员 (BF16 量化后同分)
        tie_step = max(abs(boundary) * 2.0 ** -8, 1e-6)  # BF16 相对精度 ~2^-8
        for g in set(merged_map) ^ set(ref_map):
            score = merged_map.get(g, ref_map.get(g))
            assert abs(score - boundary) <= tie_step, (
                f"row {row}: non-tie diff global={g} score={score} boundary={boundary}"
            )


def _is_a5():
    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        return False
    try:
        return torch_npu.npu.get_soc_version() == 260
    except Exception:  # pragma: no cover - older torch_npu
        return False


@pytest.mark.ci
@pytest.mark.skipif(not _is_a5(), reason="A5 (Ascend950) only: topk 8192 and A5 kernel path")
def test_qliv2_dcp_a5_topk8192():
    """A5 专属: DCP + topk=8192 的单 rank golden 校验."""
    _require_npu()
    case = _make_case(2, 0, 128, 1, 4, [8193], 8192, seed=0)
    idx, _ = _compare_rank(case, 2, 0, 128)
    assert idx.shape[-1] == 8192


LARGE_CASES = [
    # (P, S, B, Q_len, C per batch, topk)
    (2, 256, 1, 2, [32769], 8192),  # 大 C / 大 topk, 本地 L=16385, FD 开启
    (4, 512, 1, 4, [16385], 4096),  # 大 S, 本地 L=4097, FD 开启
    (2, 128, 4, 4, [4097] * 4, 2048),  # 多 batch 大 shape, FD 开启
    (4, 128, 1, 16, [16385], 4096),  # 较长 Q_len, 本地 L=4097, FD 开启
]


@pytest.mark.ci
@pytest.mark.skipif(not _is_a5(), reason="A5 large-shape DCP")
@pytest.mark.parametrize("cp_world,cp_interleave,batch_size,q_len,causal_lens,topk", LARGE_CASES)
def test_qliv2_dcp_large_shape(cp_world, cp_interleave, batch_size, q_len, causal_lens, topk):
    """A5 大 shape 泛化: 大 C/S、多 batch、长 Q_len, 只校验 return_value=1."""
    _require_npu()
    rank_outputs = []
    for rank in range(cp_world):
        case = _make_case(cp_world, rank, cp_interleave, batch_size, q_len, causal_lens, topk, seed=0)
        idx, val = _compare_rank(case, cp_world, rank, cp_interleave)
        rank_outputs.append((idx, val))

    merged_rows = _merge_global_per_row(rank_outputs, cp_world, cp_interleave, topk)
    ref_case = _make_case(1, 0, 1, batch_size, q_len, causal_lens, topk, seed=0)
    ref_out, _ = _run_op(ref_case, 1, 0, 1)
    ref_out = ref_out.cpu().numpy()[:, 0, :]
    for row, merged in enumerate(merged_rows):
        ref = set(ref_out[row][ref_out[row] >= 0].tolist())
        assert set(merged) == ref, f"row {row}: merged global topk != P=1 topk, diffs={sorted(set(merged) ^ ref)}"


HIGH_P_CASES = [
    # (P, S, B, Q_len, C per batch, topk)
    (8, 1, 1, 2, [4097], 512),  # P=8, 逐 token 轮转, 本地 L≈512, FD 关闭
    (8, 128, 1, 4, [8193], 1024),  # P=8, 大 interleave, 本地 L≈1025, FD 开启
    (16, 64, 1, 1, [8193], 512),  # P=16, 本地 L≈513, FD 关闭
    (16, 128, 1, 1, [16385], 1024),  # P=16, 本地 L≈1025, FD 开启
]


@pytest.mark.ci
@pytest.mark.skipif(not _is_a5(), reason="A5 high-P DCP")
@pytest.mark.parametrize("cp_world,cp_interleave,batch_size,q_len,causal_lens,topk", HIGH_P_CASES)
def test_qliv2_dcp_high_p(cp_world, cp_interleave, batch_size, q_len, causal_lens, topk):
    """A5 高 CP 组泛化: P=8/16, 覆盖 S=1/64/128 与 FD 开/关, 只校验 return_value=1."""
    _require_npu()
    rank_outputs = []
    for rank in range(cp_world):
        case = _make_case(cp_world, rank, cp_interleave, batch_size, q_len, causal_lens, topk, seed=0)
        idx, val = _compare_rank(case, cp_world, rank, cp_interleave)
        rank_outputs.append((idx, val))

    merged_rows = _merge_global_per_row(rank_outputs, cp_world, cp_interleave, topk)
    ref_case = _make_case(1, 0, 1, batch_size, q_len, causal_lens, topk, seed=0)
    ref_out, _ = _run_op(ref_case, 1, 0, 1)
    ref_out = ref_out.cpu().numpy()[:, 0, :]
    for row, merged in enumerate(merged_rows):
        ref = set(ref_out[row][ref_out[row] >= 0].tolist())
        assert set(merged) == ref, f"row {row}: merged global topk != P=1 topk, diffs={sorted(set(merged) ^ ref)}"


@pytest.mark.ci
def test_qliv2_contiguous_pa_block_table():
    """连续物理 block_table: 覆盖 arch35 KeyNd2NzForPA 相邻物理块合并路径."""
    _require_npu()
    case = _make_case(2, 0, 128, 1, 2, [4097], 1024, seed=0)
    _use_contiguous_block_table(case)
    _compare_rank(case, 2, 0, 128)


@pytest.mark.ci
def test_qliv2_dcp_mask_mode0():
    """mask_mode=0 (无 causal) 下 DCP 只影响框架合并, 算子行为与本地全可见一致."""
    _require_npu()
    case = _make_case(2, 0, 128, 1, 1, [257], 512, seed=7)
    out, _ = _run_op(case, 2, 0, 128, mask_mode=0)
    out = out.cpu().numpy()[:, 0, :]
    # 全可见: 行内有效 index 数 = min(topk, 本地 L)
    valid = (out >= 0).sum(axis=1)
    assert valid[0] == min(512, case["local_lens"][0])


NON_DCP_CASES = [
    (1, 1, [257]),
    (1, 4, [513]),
    (2, 2, [129, 257]),
]


@pytest.mark.ci
@pytest.mark.parametrize("batch_size,q_len,causal_lens", NON_DCP_CASES)
def test_qliv2_non_dcp_regression(batch_size, q_len, causal_lens):
    """非 DCP 路径回归: P=1 时 index 应与 DCP P=1 完全一致, return_value=0 时 values 为空."""
    _require_npu()
    case = _make_case(1, 0, 1, batch_size, q_len, causal_lens, 512, seed=0)
    ref_idx, _ = _run_op(case, 1, 0, 1)
    idx, val = _run_op(case, 1, 0, 1, enable_dcp=False, return_value=0)
    np.testing.assert_array_equal(idx.cpu().numpy(), ref_idx.cpu().numpy())
    assert val.numel() == 0, "return_value=0 时 sparse_values 必须为空 tensor"


@pytest.mark.ci
def test_qliv2_non_dcp_return_value():
    """arch22 新打开的 return_value=1: 与 CPU golden (非 DCP) 对齐 index 与 BF16 score."""
    _require_npu()
    case = _make_case(1, 0, 1, 1, 4, [513], 512, seed=0)
    idx, val = _run_op(case, 1, 0, 1, enable_dcp=False, return_value=1)
    idx = idx.cpu().numpy()[:, 0, :]
    val = val.cpu().float().numpy()[:, 0, :]
    gold, gold_val = _run_golden(case, 1, 0, 1, dcp_enable=False)
    gold = gold[:, 0, :]
    gold_val = gold_val[:, 0, :].astype(np.float32)
    for row in range(idx.shape[0]):
        pred = idx[row][idx[row] >= 0]
        ref = gold[row][gold[row] >= 0]
        assert set(pred.tolist()) == set(ref.tolist()), f"row {row}: index set mismatch"
        if len(ref) == 0:
            continue
        # 同序比较 score (并列顺序由排序实现决定, 先按 index 对齐再比较)
        ref_map = {int(i): float(v) for i, v in zip(ref, gold_val[row][gold[row] >= 0])}
        for i, v in zip(idx[row][idx[row] >= 0], val[row][idx[row] >= 0]):
            np.testing.assert_allclose(float(v), ref_map[int(i)], rtol=2e-2, atol=2e-3)


@pytest.mark.ci
@pytest.mark.parametrize("local_len,offset", [(1, 7), (17, 7), (17, 100000), (257, 0)])
def test_qliv2_output_idx_offset_padding(local_len, offset):
    """arch22 A16: offset 只作用于有效 index, padding 必须保持字面 -1 (对齐 arch35)."""
    _require_npu()
    case = _make_case(1, 0, 1, 1, 1, [local_len], 512, seed=0)
    off = torch.tensor([[offset]], dtype=torch.int32)
    idx, val = _run_op(case, 1, 0, 1, enable_dcp=False, return_value=1, idx_offset=off)
    idx = idx.cpu().numpy()[:, 0, :]
    val = val.cpu().float().numpy()[:, 0, :]
    n = min(512, local_len)
    row = idx[0]
    valid, tail = row[:n], row[n:]
    assert (tail == -1).all(), f"padding 必须是字面 -1, 实际前几个: {tail[:8]}"
    assert np.isneginf(val[0][n:]).all(), "padding 的 value 必须是 -inf"
    assert len(set(valid.tolist())) == n, "有效 index 不应重复"
    if offset == 0:
        assert (valid >= 0).all() and (valid < local_len).all()
    else:
        assert (valid >= offset).all() and (valid < offset + local_len).all()
    # 有效 index 与真值分数对齐 (offset 不影响选值, 只是平移 index)
    scores = _row_scores(case, 1, 0, 1)[0]
    for pos, gi in enumerate(valid):
        li = int(gi) - offset
        np.testing.assert_allclose(val[0][pos], scores[li], rtol=2e-2, atol=2e-3)



@pytest.mark.ci
@pytest.mark.parametrize(
    "cp_rank,causal_len,expected_valid,visible,expect_fd",
    [
        (0, 1152, 256, 256, False),    # 可见 256 < FD 阈值：padding 到 16384 也不开 FD
        (1, 1152, 128, 128, False),    # 可见 128
        (7, 1152, 128, 128, False),    # 可见 128
        (1, 128, 0, 0, False),         # 本 rank 可见长度为 0：输出全 -1/-inf，FD 关闭
        (0, 98304, 512, 12288, True),  # 可见 12288 > 阈值：FD 开，但切分只到可见上界
    ],
)
def test_qliv2_dcp_ld_invisible_segments(cp_rank, causal_len, expected_valid, visible, expect_fd):
    """DCP-aware metadata：padding L 大于可见前缀时的 FD 阈值与切分归属。

    metadata 在 A2/A3 与 A5 上都按 DCP 可见长度算 s2End 和 FD 阈值：
    - 可见 256/128/0 时阈值不满足，FD 关闭，kernel 走自算上界，padding 段不白算；
    - 可见 12288 时 FD 打开，但切分范围只到可见上界，不会把不可见段分给核。
    """
    _require_npu()
    base = _make_case(8, cp_rank, 128, 1, 1, [causal_len], 512, seed=0)
    case = _pad_case(base, 16384)

    meta = _build_metadata(case, case["topk"], 3, enable_dcp=True, cp_world=8, cp_rank=cp_rank,
                           cp_interleave=128).cpu().numpy()
    qld = meta[288:288 + 72 * 8].reshape(72, 8)[:, 0]
    if expect_fd:
        assert (qld == 1).any(), "FD/LD 未打开，用例没有覆盖 LD 归约路径"
    else:
        assert (qld == 0).all(), "可见长度小于阈值时不应打开 FD"

    out, values = _compare_rank(case, 8, cp_rank, 128)
    valid_counts = (out >= 0).sum(axis=1)
    assert (valid_counts == expected_valid).all(), f"valid count {valid_counts} != {expected_valid}"
    if expected_valid == 0:
        assert (out == -1).all()
        assert np.isneginf(values).all()
    else:
        assert (out[:, :expected_valid] >= 0).all()
        assert (out[:, expected_valid:] == -1).all()
        assert (out[:, :expected_valid] < visible).all(), "选到了可见前缀之外的 token"


@pytest.mark.ci
def test_qliv2_ld_gate_non_dcp_vs_dcp():
    """门禁回归：A2/A3 非 DCP 长 S2 不开 LD；A5 维持原有行为；DCP 长 S2 开 LD。"""
    _require_npu()
    non_dcp = _make_case(1, 0, 1, 1, 1, [16384], 512, seed=0)
    m0 = _build_metadata(non_dcp, 512, 3, enable_dcp=False).cpu().numpy()
    m0_qld = m0[288:288 + 72 * 8].reshape(72, 8)[:, 0]
    if _is_a5():
        # A5 (arch35) 的 FD/LD 不区分 DCP 场景, 长 S2 维持原有行为。
        assert (m0_qld == 1).any(), "A5 非 DCP 长 S2 应维持原有 FD/LD 行为"
    else:
        assert (m0_qld == 0).all(), "A2/A3 非 DCP 不应打开 LD"

    dcp = _make_case(8, 0, 128, 1, 1, [131072], 512, seed=0)  # 本地 L=16384
    m1 = _build_metadata(dcp, 512, 3, enable_dcp=True, cp_world=8, cp_rank=0,
                         cp_interleave=128).cpu().numpy()
    assert (m1[288:288 + 72 * 8].reshape(72, 8)[:, 0] == 1).any(), "DCP 长 S2 应打开 LD"


@pytest.mark.ci
def test_qliv2_dcp_rejects_output_idx_offset():
    """DCP 与 output_idx_offset 互斥，必须在 tiling 层拒绝。"""
    _require_npu()
    case = _make_case(2, 0, 128, 1, 1, [513], 512, seed=0)
    off = torch.zeros(case["q_t"], N2, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="output_idx_offset"):
        _run_op(case, 2, 0, 128, idx_offset=off)


@pytest.mark.ci
def test_qliv2_dcp_rejects_cmp_ratio_gt1():
    """DCP 与压缩路径互斥（cmp_ratio 必须为 1）。"""
    _require_npu()
    case = _make_case(2, 0, 128, 1, 1, [513], 512, seed=0)
    with pytest.raises(RuntimeError, match="cmp_ratio|cmp_residual_k|enable_dcp"):
        _run_op(case, 2, 0, 128, mask_mode=0, cmp_ratio=2)
