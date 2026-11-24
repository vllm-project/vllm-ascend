#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mode2(mask) vs mode4(skip) 等价性验证 (QLI candidate skip).

同一输入、同一候选列表 (由 mode1 source pass 产出) 下, 分别以 candidate_mode=2 与
candidate_mode=4 调用 npu_quant_lightning_indexer_v3, 逐 (T, N2) 行比较多重集:
sorted(indices) 完全一致。覆盖 总块数<=2048 (全候选+泄漏填充) 与 >2048 (真实过滤)、
尾块不对齐、cmp_residual、varlen batch 等边界。运行: 容器内 python3 本脚本。
"""

import sys

import numpy as np
import torch
import torch_npu

PYTEST_DIR = "/vllm-workspace/vllm-ascend/csrc/attention/quant_lightning_indexer_v2/tests/pytest"
sys.path.insert(0, PYTEST_DIR)

import quant_lightning_indexer_v2_golden as golden  # noqa: E402
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401  # 触发 torch.ops._C_ascend 注册

CAND_TOPK_BLOCKS = 2048
CAND_BLOCK_SIZE = 8
SPARSE_COUNT = 512


def build_params(batch_size, q_seq, seqused_k_list, cmp_residual_list, block_num):
    """33 元组参数 (normalize_qliv2_params 顺序), TND query + PA_BBND key + INT8."""
    cu_q = [0]
    for _ in range(batch_size):
        cu_q.append(cu_q[-1] + q_seq)
    return (
        batch_size,  # batch_size
        q_seq,  # q_seq (每 batch query 行数)
        max(seqused_k_list),  # k_seq
        batch_size * q_seq,  # q_t_size (TND T)
        0,  # k_t_size (PA_BBND 不用)
        64,  # q_head_num (metadata 校验要求 64)
        1,  # k_head_num (MLA 单 kv 头)
        128,  # head_dim
        128,  # block_size (K 物理块)
        block_num,  # block_num
        "INT8",  # qk_dtype
        "FP16",  # weight_dtype
        "FLOAT16",  # dequant_dtype
        "INT32",  # actual_seq_dtype
        cu_q,  # cu_seqlens_q (TND 必传)
        None,  # cu_seqlens_k (PA_BBND 不传)
        [q_seq] * batch_size,  # seqused_q
        seqused_k_list,  # seqused_k (压缩域 S2 长度)
        cmp_residual_list,  # cmp_residual_k
        -1,  # max_seqlen_q
        2,  # quant_mode (INT8, 2201 仅支持 2)
        "TND",  # layout_query
        "PA_BBND",  # layout_key
        SPARSE_COUNT,  # sparse_count
        3,  # sparse_mode
        (-2, 2),  # query_datarange
        (-2, 2),  # key_datarange
        (-2, 2),  # weights_datarange
        (0.5, 1.5),  # q_scale_datarange
        (0.5, 1.5),  # k_scale_datarange
        2,  # cmp_ratio
        0,  # return_value
        None,  # output_idx_offset
    )


def call_v3(data, metadata, mode, candidates):
    return torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
        data["query"].npu(),
        data["key"].npu(),
        data["weights"].npu(),
        data["query_dequant_scale"].npu(),
        data["key_dequant_scale"].npu(),
        data["sparse_count"],
        data["quant_mode"],
        candidate_topk_index=candidates,
        cu_seqlens_q=data["cu_seqlens_query"].npu() if data["cu_seqlens_query"] is not None else None,
        cu_seqlens_k=None,
        seqused_q=data["seqused_q"].npu() if data["seqused_q"] is not None else None,
        seqused_k=data["seqused_k"].npu() if data["seqused_k"] is not None else None,
        cmp_residual_k=data["cmp_residual_k_for_npu"].npu() if data["cmp_residual_k_for_npu"] is not None else None,
        block_table=data["block_table"].npu(),
        metadata=metadata,
        max_seqlen_q=data["max_seqlen_q_meta"],
        layout_q=data["layout_query"],
        layout_k=data["layout_key"],
        mask_mode=data["sparse_mode"],
        cmp_ratio=data["cmp_ratio"],
        candidate_mode=mode,
        candidate_topk_blocks=CAND_TOPK_BLOCKS,
        candidate_block_size=CAND_BLOCK_SIZE,
    )


def run_case(name, params, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = golden.generate_qliv2_test_data(params, generate_golden=False)
    md = data["metadata_input"]
    metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
        cu_seqlens_q=md["cu_seqlens_q"].npu() if md["cu_seqlens_q"] is not None else None,
        cu_seqlens_k=md["cu_seqlens_k"].npu() if md["cu_seqlens_k"] is not None else None,
        seqused_q=md["seqused_q"].npu() if md["seqused_q"] is not None else None,
        seqused_k=md["seqused_k"].npu() if md["seqused_k"] is not None else None,
        cmp_residual_k=md["cmp_residual_k"].npu() if md["cmp_residual_k"] is not None else None,
        batch_size=md["batch_size"],
        max_seqlen_q=md["max_seqlen_q"],
        max_seqlen_k=md["max_seqlen_k"],
        num_heads_q=md["num_heads_q"],
        num_heads_k=md["num_heads_k"],
        head_dim=md["head_dim"],
        topk=md["topk"],
        quant_mode=md["quant_mode"],
        mask_mode=md["mask_mode"],
        layout_q=md["layout_q"],
        layout_k=md["layout_k"],
        cmp_ratio=md["cmp_ratio"],
    ).npu()

    # mode1 (source): 产出候选块列表
    _, _, cand_out = call_v3(data, metadata, 1, None)
    cand_out = cand_out.npu()
    torch.npu.synchronize()
    assert cand_out.dim() == 3 and cand_out.shape[-1] == CAND_TOPK_BLOCKS, (
        f"{name}: candidate_out shape 异常: {tuple(cand_out.shape)}"
    )
    cand_cpu = cand_out.cpu()

    # mode2 (mask) 两次 — 先验证确定性基线
    sel2a = call_v3(data, metadata, 2, cand_out)[0].cpu()
    sel2b = call_v3(data, metadata, 2, cand_out)[0].cpu()
    torch.npu.synchronize()
    if not torch.equal(sel2a, sel2b):
        print(f"[{name}] FAIL: mode2 两次调用不确定 (基线噪声)")
        return False
    # mode4 (skip)
    sel4 = call_v3(data, metadata, 4, cand_out)[0].cpu()
    torch.npu.synchronize()

    # 逐行多重集比较 (tie 顺序可能不同, 集合须一致)
    bad_rows = 0
    for t in range(sel2a.shape[0]):
        for n in range(sel2a.shape[1]):
            a = np.sort(sel2a[t, n].numpy())
            b = np.sort(sel4[t, n].numpy())
            if not np.array_equal(a, b):
                bad_rows += 1
                if bad_rows <= 2:
                    diff_idx = np.where(a != b)[0][:5]
                    print(f"[{name}] row=({t},{n}) multiset mismatch @{diff_idx}: m2={a[diff_idx]} m4={b[diff_idx]}")
    try:
        seqused_dbg = data["actual_seq_lengths_key"].tolist()
    except Exception:
        seqused_dbg = list(data["actual_seq_lengths_key"])
    print(
        f"[{name}] {'PASS' if bad_rows == 0 else 'FAIL'} "
        f"(rows={sel2a.shape[0]}x{sel2a.shape[1]}, bad={bad_rows}, "
        f"seqused_k={seqused_dbg}, cand_non_neg1_first_row="
        f"{int((cand_cpu[0, 0] >= 0).sum().item())})"
    )
    return bad_rows == 0


def main():
    torch_npu.npu.set_device(0)
    cases = [
        # (name, batch, q_seq, seqused_k 压缩长度列表, residual, block_num)
        ("tiny_leak", 1, 6, [100], [0], 4),  # 13 块 ≤ 2048: 全候选, -1 填充泄漏区
        ("small_allcand", 1, 6, [8192], [0], 80),  # 1024 块 ≤ 2048: 全候选无 pad
        ("large_filter", 1, 6, [25600], [0], 220),  # 3200 块 > 2048: 真实过滤
        ("large_tailodd", 1, 6, [25605], [1], 220),  # 尾块不对齐 + residual<cmp_ratio: 修正链路径
        ("varlen_b2", 2, 6, [30000, 12345], [0, 1], 400),  # 变长 batch: 每行候选不同 (block_num 覆盖 235+97)
        ("boundary_2048", 1, 6, [16384], [0], 140),  # 恰 2048 块: 边界
    ]
    ok = True
    for i, (name, b, q, klist, res, bn) in enumerate(cases):
        params = build_params(b, q, klist, res, bn)
        ok &= run_case(name, params, seed=1024 + i)
    print("EQUIV-RESULT:", "ALL-PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
